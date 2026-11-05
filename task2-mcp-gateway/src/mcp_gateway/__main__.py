"""Command line entry point: ``python -m mcp_gateway``.

``uvicorn --factory mcp_gateway.app:app_from_env`` serves the same
application under uvicorn's own process manager when that is preferred.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from typing import Any

import uvicorn

from mcp_gateway import __version__
from mcp_gateway.app import create_app
from mcp_gateway.config import ConfigurationError, GatewaySettings
from mcp_gateway.logging_config import configure_logging, redact_url
from mcp_gateway.serving import add_server_arguments, uvicorn_options

logger = logging.getLogger(__name__)


def _parse_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, dict[str, Any]]:
    parser = argparse.ArgumentParser(
        prog="mcp-gateway",
        description="MCP security gateway: bearer authentication and tool-level authorisation.",
    )
    add_server_arguments(parser, default_port=9000)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)
    try:
        options = uvicorn_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args, options


def main(argv: Sequence[str] | None = None) -> int:
    args, options = _parse_args(argv)
    try:
        settings = GatewaySettings.from_env()
    except ConfigurationError as exc:
        configure_logging("ERROR")
        logger.error("invalid configuration: %s", exc, extra={"event": "gateway.config_error"})
        return 2
    configure_logging(settings.log_level)
    logger.info(
        "starting gateway",
        extra={
            "event": "gateway.boot",
            "host": args.host,
            "port": args.port,
            "downstream": redact_url(settings.downstream_url),
        },
    )
    try:
        app = create_app(settings)
    except (OSError, ValueError) as exc:
        # The token file is missing, unreadable or malformed.
        logger.error("cannot load token source: %s", exc, extra={"event": "gateway.config_error"})
        return 2
    uvicorn.run(app, **options)
    return 0


if __name__ == "__main__":
    sys.exit(main())
