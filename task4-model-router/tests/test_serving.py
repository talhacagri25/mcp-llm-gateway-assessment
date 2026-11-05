"""Embedded server options: conservative defaults and flag validation."""

from __future__ import annotations

import argparse

import pytest

from model_router.serving import add_server_arguments, uvicorn_options


def _parse(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_server_arguments(parser, default_port=9400)
    return parser.parse_args(argv)


def test_defaults_are_conservative() -> None:
    options = uvicorn_options(_parse())
    assert options["server_header"] is False
    assert options["access_log"] is False
    assert options["proxy_headers"] is False  # X-Forwarded-* ignored unless proxies are named
    assert options["limit_concurrency"] is None
    assert options["timeout_keep_alive"] == 5
    assert options["timeout_graceful_shutdown"] == 30
    assert (options["host"], options["port"]) == ("127.0.0.1", 9400)


def test_proxy_trust_and_limits_are_explicit() -> None:
    options = uvicorn_options(
        _parse("--forwarded-allow-ips", "10.0.0.5", "--limit-concurrency", "200", "--port", "80")
    )
    assert options["proxy_headers"] is True
    assert options["forwarded_allow_ips"] == "10.0.0.5"
    assert options["limit_concurrency"] == 200
    assert options["port"] == 80


@pytest.mark.parametrize(
    "argv",
    [
        ("--limit-concurrency", "-1"),
        ("--timeout-keep-alive", "0"),
        ("--timeout-graceful-shutdown", "0"),
        ("--ssl-keyfile", "key.pem"),
        ("--ssl-certfile", "cert.pem"),
    ],
)
def test_inconsistent_flags_are_rejected(argv: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="must"):
        uvicorn_options(_parse(*argv))
