"""Command line entry point: ``python -m mock_model_provider``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import uvicorn

from mock_model_provider.app import create_app


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mock-model-provider")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9401)
    parser.add_argument("--name", default=None, help="Provider name (default: MOCK_PROVIDER_NAME)")
    parser.add_argument(
        "--behavior",
        default=None,
        help="ok | rate_limited | slow | error | bad_request | unauthorized | garbage | flaky",
    )
    parser.add_argument("--delay-ms", type=int, default=None)
    args = parser.parse_args(argv)
    uvicorn.run(
        create_app(args.name, args.behavior, args.delay_ms),
        host=args.host,
        port=args.port,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
