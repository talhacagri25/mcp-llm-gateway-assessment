"""Command line entry point: ``python -m mock_mcp_server``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import uvicorn

from mock_mcp_server.app import create_app


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mock-mcp-server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    args = parser.parse_args(argv)
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
