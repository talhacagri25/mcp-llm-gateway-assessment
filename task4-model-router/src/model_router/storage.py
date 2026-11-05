"""SQLite-backed token ledger (on disk).

One row per admitted request. A row starts ``reserved`` with the estimated
token cost, becomes ``settled`` with the provider's reported usage, or is
``released`` (zero tokens) when no provider served the request. The sliding
window usage of a tenant is the sum of ``tokens`` over rows admitted within
the window; rows older than the window are evicted.

Admission is a single ``BEGIN IMMEDIATE`` transaction (read the window sum,
compare, insert), which makes check-and-reserve atomic across coroutines,
threads and processes sharing the database file. WAL mode keeps readers from
blocking the writer.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import stat
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS token_ledger (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id      TEXT    NOT NULL,
    request_id     TEXT    NOT NULL UNIQUE,
    admitted_at_ms INTEGER NOT NULL,
    tokens         INTEGER NOT NULL CHECK (tokens >= 0),
    state          TEXT    NOT NULL CHECK (state IN ('reserved', 'settled', 'released'))
);
CREATE INDEX IF NOT EXISTS ix_token_ledger_tenant_time
    ON token_ledger (tenant_id, admitted_at_ms);

CREATE TABLE IF NOT EXISTS request_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id  TEXT    NOT NULL,
    tenant_id   TEXT    NOT NULL,
    created_at_ms INTEGER NOT NULL,
    outcome     TEXT    NOT NULL,
    provider    TEXT,
    http_status INTEGER,
    latency_ms  REAL,
    attempts    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_request_log_tenant_time ON request_log (tenant_id, created_at_ms);

CREATE TABLE IF NOT EXISTS write_probe (
    id         INTEGER PRIMARY KEY CHECK (id = 1),
    touched_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS calibration (
    tenant_id     TEXT    NOT NULL,
    provider      TEXT    NOT NULL,
    factor        REAL    NOT NULL,
    samples       INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    PRIMARY KEY (tenant_id, provider)
);
"""
SCHEMA_VERSION = 2
"""Stored in ``PRAGMA user_version``: a file written by a newer build is refused, an
older one is migrated in place."""


def _create_private(path: Path) -> None:
    """Create the database file readable by its owner only; SQLite gives WAL and SHM
    files the mode of the main file, and an existing world-readable file is tightened.
    A file this process may not write is left alone: the write probe reports it."""
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except PermissionError:
        return
    try:
        # POSIX only: Windows has no mode bits (it reports 666 for any writable file)
        # and protects files through ACLs instead.
        if hasattr(os, "fchmod") and stat.S_IMODE(os.fstat(fd).st_mode) & 0o077:
            os.fchmod(fd, 0o600)
    finally:
        os.close(fd)


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    admitted: bool
    used_before: int
    """Tokens already charged to the window before this request."""
    limit: int
    retry_after_ms: int | None = None
    """When not admitted: milliseconds until enough tokens leave the window, or None if the
    request can never fit (its cost exceeds the whole limit)."""


class TokenLedgerStore:
    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        _create_private(self._path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            self._path, check_same_thread=False, isolation_level=None, timeout=5.0
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._migrate()
        # Fail at start-up, not on the first request, when the file is read-only.
        self.probe()

    def _migrate(self) -> None:
        version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        if version > SCHEMA_VERSION:
            raise sqlite3.OperationalError(
                f"ledger schema version {version} is newer than this build ({SCHEMA_VERSION})"
            )
        if version < 2:
            # Version 1 kept calibration per provider only; it is a cache and is rebuilt.
            self._conn.execute("DROP TABLE IF EXISTS calibration")
        self._conn.executescript(SCHEMA)
        if version != SCHEMA_VERSION:
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @property
    def path(self) -> Path:
        return self._path

    def probe(self) -> None:
        """Raise ``sqlite3.Error`` unless the ledger can be written right now.

        A real page write inside a transaction that is rolled back: in WAL
        mode merely taking the write lock succeeds on a read-only file, the
        failure only surfaces when a page is written. Nothing is left behind,
        and the probe detects read-only files, locked databases and
        unreachable storage alike.
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO write_probe (id, touched_ms) VALUES (1, ?)",
                    (int(time.time() * 1000),),
                )
            finally:
                self._rollback()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Serialise access in-process and take the SQLite write lock immediately."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._rollback()
                raise
            try:
                self._conn.execute("COMMIT")
            except sqlite3.Error:
                # A failed COMMIT (disk full, I/O error) leaves the transaction open;
                # roll it back so the next BEGIN IMMEDIATE can succeed.
                self._rollback()
                raise

    def _rollback(self) -> None:
        try:
            self._conn.execute("ROLLBACK")
        except sqlite3.Error:
            logger.exception("rollback failed", extra={"event": "ledger.rollback_failed"})

    # -- admission -------------------------------------------------------------

    def try_reserve(
        self,
        tenant_id: str,
        request_id: str,
        tokens: int,
        *,
        now_ms: int,
        window_ms: int,
        limit: int,
    ) -> AdmissionResult:
        cutoff = now_ms - window_ms
        with self._transaction() as conn:
            conn.execute(
                "DELETE FROM token_ledger WHERE tenant_id = ? AND admitted_at_ms <= ?",
                (tenant_id, cutoff),
            )
            used = self._window_sum(conn, tenant_id, cutoff)
            if used + tokens <= limit:
                conn.execute(
                    "INSERT INTO token_ledger"
                    " (tenant_id, request_id, admitted_at_ms, tokens, state)"
                    " VALUES (?, ?, ?, ?, 'reserved')",
                    (tenant_id, request_id, now_ms, tokens),
                )
                return AdmissionResult(admitted=True, used_before=used, limit=limit)
            if tokens > limit:
                return AdmissionResult(admitted=False, used_before=used, limit=limit)
            rows = conn.execute(
                "SELECT admitted_at_ms, tokens FROM token_ledger"
                " WHERE tenant_id = ? AND admitted_at_ms > ? ORDER BY admitted_at_ms",
                (tenant_id, cutoff),
            ).fetchall()
        freed = 0
        for admitted_at_ms, row_tokens in rows:
            freed += row_tokens
            if used - freed + tokens <= limit:
                return AdmissionResult(
                    admitted=False,
                    used_before=used,
                    limit=limit,
                    retry_after_ms=max(1, admitted_at_ms + window_ms - now_ms),
                )
        return AdmissionResult(
            admitted=False, used_before=used, limit=limit, retry_after_ms=window_ms
        )

    @staticmethod
    def _window_sum(conn: sqlite3.Connection, tenant_id: str, cutoff: int) -> int:
        row = conn.execute(
            "SELECT COALESCE(SUM(tokens), 0) FROM token_ledger"
            " WHERE tenant_id = ? AND admitted_at_ms > ?",
            (tenant_id, cutoff),
        ).fetchone()
        return int(row[0])

    # -- reconciliation --------------------------------------------------------

    def settle(self, request_id: str, actual_tokens: int) -> bool:
        with self._transaction() as conn:
            cursor = conn.execute(
                "UPDATE token_ledger SET tokens = ?, state = 'settled'"
                " WHERE request_id = ? AND state = 'reserved'",
                (max(0, actual_tokens), request_id),
            )
            return cursor.rowcount == 1

    def release(self, request_id: str) -> bool:
        with self._transaction() as conn:
            cursor = conn.execute(
                "UPDATE token_ledger SET tokens = 0, state = 'released'"
                " WHERE request_id = ? AND state = 'reserved'",
                (request_id,),
            )
            return cursor.rowcount == 1

    # -- observability ---------------------------------------------------------

    def window_usage(self, tenant_id: str, *, now_ms: int, window_ms: int) -> int:
        with self._lock:
            return self._window_sum(self._conn, tenant_id, now_ms - window_ms)

    def evict_expired(self, *, now_ms: int, window_ms: int, keep_log_ms: int | None = None) -> int:
        with self._transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM token_ledger WHERE admitted_at_ms <= ?", (now_ms - window_ms,)
            )
            removed = cursor.rowcount
            if keep_log_ms is not None:
                conn.execute(
                    "DELETE FROM request_log WHERE created_at_ms <= ?", (now_ms - keep_log_ms,)
                )
            return int(removed)

    def ledger_rows(self, tenant_id: str) -> list[tuple[str, int, int, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT request_id, admitted_at_ms, tokens, state FROM token_ledger"
                " WHERE tenant_id = ? ORDER BY admitted_at_ms, id",
                (tenant_id,),
            ).fetchall()
        return [(str(r[0]), int(r[1]), int(r[2]), str(r[3])) for r in rows]

    def load_calibration(self) -> dict[tuple[str, str], tuple[float, int]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT tenant_id, provider, factor, samples FROM calibration"
            ).fetchall()
        return {(str(r[0]), str(r[1])): (float(r[2]), int(r[3])) for r in rows}

    def save_calibration(self, tenant_id: str, provider: str, factor: float, samples: int) -> None:
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO calibration (tenant_id, provider, factor, samples, updated_at_ms)"
                " VALUES (?, ?, ?, ?, ?) ON CONFLICT(tenant_id, provider) DO UPDATE SET"
                " factor = excluded.factor, samples = excluded.samples,"
                " updated_at_ms = excluded.updated_at_ms",
                (tenant_id, provider, factor, samples, int(time.time() * 1000)),
            )

    def request_log_count(self, tenant_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM request_log WHERE tenant_id = ?", (tenant_id,)
            ).fetchone()
        return int(row[0])

    def record_request(
        self,
        *,
        request_id: str,
        tenant_id: str,
        created_at_ms: int,
        outcome: str,
        provider: str | None,
        http_status: int | None,
        latency_ms: float | None,
        attempts_json: str,
    ) -> None:
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO request_log (request_id, tenant_id, created_at_ms, outcome, provider,"
                " http_status, latency_ms, attempts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    request_id,
                    tenant_id,
                    created_at_ms,
                    outcome,
                    provider,
                    http_status,
                    latency_ms,
                    attempts_json,
                ),
            )
