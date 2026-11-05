"""Command line entry point: ``python -m model_router``.

``uvicorn --factory model_router.app:app_from_env`` serves the same
application under uvicorn's own process manager when that is preferred.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from collections.abc import Sequence

import uvicorn

from model_router import __version__
from model_router.app import create_app
from model_router.config import ConfigurationError, RouterSettings
from model_router.logging_config import configure_logging
from model_router.serving import add_server_arguments, uvicorn_options

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="model-router",
        description="Rate limiting and model failover router for LLM gateways.",
    )
    add_server_arguments(parser, default_port=9400)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)
    try:
        options = uvicorn_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        settings = RouterSettings.from_env()
    except ConfigurationError as exc:
        configure_logging("ERROR")
        logger.error("invalid configuration: %s", exc, extra={"event": "router.config_error"})
        return 2
    configure_logging(settings.log_level)
    try:
        app = create_app(settings)
    except (sqlite3.Error, OSError) as exc:
        logger.error(
            "cannot open the ledger database at %s: %s",
            settings.database_path,
            exc,
            extra={"event": "router.storage_error"},
        )
        return 2
    uvicorn.run(app, **options)
    return 0


if __name__ == "__main__":
    sys.exit(main())
