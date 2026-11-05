"""SQLite token ledger: atomic admission, eviction, reconciliation, persistence."""

from __future__ import annotations

import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from model_router.storage import SCHEMA_VERSION, TokenLedgerStore

WINDOW = 60_000


def test_schema_is_created_on_disk(store: TokenLedgerStore, db_path: Path) -> None:
    assert db_path.exists()
    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert {"token_ledger", "request_log"} <= tables
    assert mode == "wal"


def test_admission_respects_the_limit_and_computes_retry_after(store: TokenLedgerStore) -> None:
    first = store.try_reserve("t", "r1", 60, now_ms=1_000, window_ms=WINDOW, limit=100)
    assert first.admitted
    assert first.used_before == 0
    second = store.try_reserve("t", "r2", 30, now_ms=2_000, window_ms=WINDOW, limit=100)
    assert second.admitted
    assert second.used_before == 60

    rejected = store.try_reserve("t", "r3", 20, now_ms=3_000, window_ms=WINDOW, limit=100)
    assert rejected.admitted is False
    assert rejected.used_before == 90
    # Freeing the first row (60 tokens, admitted at t=1000) makes 20 fit: 1000 + 60000 - 3000.
    assert rejected.retry_after_ms == 58_000

    bigger = store.try_reserve("t", "r4", 90, now_ms=3_000, window_ms=WINDOW, limit=100)
    # Needs both rows gone: the second row expires at 2000 + 60000.
    assert bigger.retry_after_ms == 59_000

    impossible = store.try_reserve("t", "r5", 101, now_ms=3_000, window_ms=WINDOW, limit=100)
    assert impossible.admitted is False
    assert impossible.retry_after_ms is None


def test_expired_rows_are_evicted_during_admission(store: TokenLedgerStore) -> None:
    assert store.try_reserve("t", "r1", 100, now_ms=0, window_ms=WINDOW, limit=100).admitted
    # A row admitted at t counts until t + window (exclusive): still present one ms before.
    assert not store.try_reserve(
        "t", "r2", 1, now_ms=WINDOW - 1, window_ms=WINDOW, limit=100
    ).admitted
    admitted = store.try_reserve("t", "r3", 100, now_ms=WINDOW, window_ms=WINDOW, limit=100)
    assert admitted.admitted
    assert [row[0] for row in store.ledger_rows("t")] == ["r3"]


def test_settle_and_release_apply_only_to_reserved_rows(store: TokenLedgerStore) -> None:
    store.try_reserve("t", "r1", 500, now_ms=0, window_ms=WINDOW, limit=1000)
    assert store.settle("r1", 120) is True
    assert store.settle("r1", 999) is False  # already settled
    assert store.release("r1") is False
    assert store.window_usage("t", now_ms=1, window_ms=WINDOW) == 120

    store.try_reserve("t", "r2", 500, now_ms=0, window_ms=WINDOW, limit=1000)
    assert store.release("r2") is True
    assert store.window_usage("t", now_ms=1, window_ms=WINDOW) == 120
    assert store.ledger_rows("t")[1][2:] == (0, "released")
    assert store.settle("missing", 1) is False


def test_tenants_are_isolated(store: TokenLedgerStore) -> None:
    store.try_reserve("a", "r1", 100, now_ms=0, window_ms=WINDOW, limit=100)
    assert store.try_reserve("b", "r2", 100, now_ms=0, window_ms=WINDOW, limit=100).admitted
    assert store.window_usage("a", now_ms=1, window_ms=WINDOW) == 100
    assert store.window_usage("b", now_ms=1, window_ms=WINDOW) == 100


def test_global_eviction_and_request_log(store: TokenLedgerStore, db_path: Path) -> None:
    store.try_reserve("a", "old", 10, now_ms=0, window_ms=WINDOW, limit=100)
    store.try_reserve("b", "new", 10, now_ms=50_000, window_ms=WINDOW, limit=100)
    store.record_request(
        request_id="old",
        tenant_id="a",
        created_at_ms=0,
        outcome="served",
        provider="primary",
        http_status=200,
        latency_ms=1.0,
        attempts_json="[]",
    )
    removed = store.evict_expired(now_ms=WINDOW + 1, window_ms=WINDOW, keep_log_ms=10)
    assert removed == 1
    assert store.ledger_rows("a") == []
    assert len(store.ledger_rows("b")) == 1
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM request_log").fetchone()[0] == 0


def test_state_survives_reopening_the_database(db_path: Path) -> None:
    first = TokenLedgerStore(db_path)
    first.try_reserve("t", "r1", 40, now_ms=0, window_ms=WINDOW, limit=100)
    first.close()
    second = TokenLedgerStore(db_path)
    try:
        assert second.window_usage("t", now_ms=1, window_ms=WINDOW) == 40
        assert not second.try_reserve("t", "r2", 61, now_ms=1, window_ms=WINDOW, limit=100).admitted
    finally:
        second.close()


def test_concurrent_admission_across_connections_never_exceeds_the_limit(db_path: Path) -> None:
    """Two independent connections (as two processes would have) racing on one ledger."""
    stores = [TokenLedgerStore(db_path) for _ in range(2)]
    try:

        def worker(index: int) -> bool:
            ledger = stores[index % 2]
            result = ledger.try_reserve(
                "t", f"r{index}", 10, now_ms=1_000, window_ms=WINDOW, limit=100
            )
            return result.admitted

        with ThreadPoolExecutor(max_workers=16) as pool:
            outcomes = list(pool.map(worker, range(200)))
        assert sum(outcomes) == 10
        assert stores[0].window_usage("t", now_ms=1_001, window_ms=WINDOW) == 100
    finally:
        for ledger in stores:
            ledger.close()


def test_newer_schema_versions_are_refused(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    with pytest.raises(sqlite3.OperationalError, match="newer than this build"):
        TokenLedgerStore(db_path)


def test_version_one_calibration_table_is_rebuilt(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE calibration (provider TEXT PRIMARY KEY, factor REAL NOT NULL,"
            " samples INTEGER NOT NULL, updated_at_ms INTEGER NOT NULL)"
        )
        conn.execute("INSERT INTO calibration VALUES ('p', 2.0, 5, 0)")
    store = TokenLedgerStore(db_path)
    try:
        assert store.load_calibration() == {}
        store.save_calibration("t", "p", 1.5, 1)
        assert store.load_calibration() == {("t", "p"): (1.5, 1)}
        with sqlite3.connect(db_path) as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        store.close()


def test_request_log_rows_are_counted_per_tenant(store: TokenLedgerStore) -> None:
    for request_id in ("a1", "a2"):
        store.record_request(
            request_id=request_id,
            tenant_id="a",
            created_at_ms=0,
            outcome="served",
            provider="primary",
            http_status=200,
            latency_ms=1.0,
            attempts_json="[]",
        )
    assert store.request_log_count("a") == 2
    assert store.request_log_count("b") == 0


@pytest.mark.skipif(
    os.name != "posix" or getattr(os, "geteuid", lambda: -1)() == 0,
    reason="POSIX file modes, and root ignores them",
)
def test_unwritable_ledger_file_fails_the_probe_not_the_open(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    TokenLedgerStore(db_path).close()
    db_path.chmod(0o400)
    try:
        with pytest.raises(sqlite3.OperationalError):
            TokenLedgerStore(db_path)
    finally:
        db_path.chmod(0o600)
