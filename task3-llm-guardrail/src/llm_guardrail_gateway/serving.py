"""Command line options for the embedded uvicorn server.

The defaults are the conservative ones for a service that faces callers it
does not control: no ``Server`` header, keep-alive connections dropped after a
short idle period, a bounded graceful shutdown, and ``X-Forwarded-*`` headers
ignored unless the operator names the proxies allowed to set them.
"""

from __future__ import annotations

import argparse
from typing import Any


def add_server_arguments(parser: argparse.ArgumentParser, *, default_port: int) -> None:
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument(
        "--limit-concurrency",
        type=int,
        default=0,
        help="Connections served at once before new ones are answered 503 (0 = no limit).",
    )
    parser.add_argument(
        "--timeout-keep-alive",
        type=int,
        default=5,
        help="Seconds an idle keep-alive connection stays open (default: 5).",
    )
    parser.add_argument(
        "--timeout-graceful-shutdown",
        type=int,
        default=30,
        help="Seconds to wait for in-flight requests on shutdown (default: 30).",
    )
    parser.add_argument(
        "--forwarded-allow-ips",
        default=None,
        help="Proxies whose X-Forwarded-* headers are trusted (comma separated or '*'). "
        "Without it those headers are ignored.",
    )
    parser.add_argument("--ssl-keyfile", default=None, help="TLS private key (PEM).")
    parser.add_argument("--ssl-certfile", default=None, help="TLS certificate chain (PEM).")


def uvicorn_options(args: argparse.Namespace) -> dict[str, Any]:
    """Keyword arguments for ``uvicorn.run``; raises ``ValueError`` on inconsistent flags."""
    if args.limit_concurrency < 0:
        raise ValueError("--limit-concurrency must be 0 or more")
    if args.timeout_keep_alive <= 0 or args.timeout_graceful_shutdown <= 0:
        raise ValueError("timeouts must be positive")
    if bool(args.ssl_keyfile) != bool(args.ssl_certfile):
        raise ValueError("--ssl-keyfile and --ssl-certfile must be given together")
    return {
        "host": args.host,
        "port": args.port,
        "log_config": None,
        "access_log": False,
        "server_header": False,
        "limit_concurrency": args.limit_concurrency or None,
        "timeout_keep_alive": args.timeout_keep_alive,
        "timeout_graceful_shutdown": args.timeout_graceful_shutdown,
        "proxy_headers": args.forwarded_allow_ips is not None,
        "forwarded_allow_ips": args.forwarded_allow_ips,
        "ssl_keyfile": args.ssl_keyfile,
        "ssl_certfile": args.ssl_certfile,
    }
