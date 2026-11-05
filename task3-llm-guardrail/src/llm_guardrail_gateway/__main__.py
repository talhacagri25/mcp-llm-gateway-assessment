"""Command line entry point: ``python -m llm_guardrail_gateway``.

``uvicorn --factory llm_guardrail_gateway.app:app_from_env`` serves the same
application under uvicorn's own process manager when that is preferred.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

import uvicorn

from llm_guardrail_gateway import __version__
from llm_guardrail_gateway.app import create_app
from llm_guardrail_gateway.config import ConfigurationError, GuardrailSettings
from llm_guardrail_gateway.logging_config import configure_logging
from llm_guardrail_gateway.serving import add_server_arguments, uvicorn_options

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llm-guardrail-gateway",
        description="LLM gateway that redacts PII from streamed completions.",
    )
    add_server_arguments(parser, default_port=9300)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)
    try:
        options = uvicorn_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        settings = GuardrailSettings.from_env()
    except ConfigurationError as exc:
        configure_logging("ERROR")
        logger.error("invalid configuration: %s", exc, extra={"event": "gateway.config_error"})
        return 2
    configure_logging(settings.log_level)
    uvicorn.run(create_app(settings), **options)
    return 0


if __name__ == "__main__":
    sys.exit(main())
