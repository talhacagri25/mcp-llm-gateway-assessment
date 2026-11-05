"""Command line entry point: ``python -m customer_mcp_server``.

Transport discipline
--------------------
* The MCP stdio transport owns stdout. Nothing else in the process writes to it.
* All logging is configured to stderr before the transport starts.
* ``stdio_server`` re-points file descriptor 1 at stderr for the lifetime of
  the session and serves the wire through a private duplicate. Because it
  restores the descriptor afterwards, text that a library buffered in
  ``sys.stdout`` during the session would be flushed onto the wire at
  interpreter exit; ``sys.stdout`` is therefore redirected to stderr for the
  rest of the process once the session has started.
* Frames that cannot be decoded are answered with ``-32700`` / ``-32600``
  errors instead of being dropped, frames above ``--max-frame-bytes`` are
  discarded and answered with ``-32600`` without being parsed, and an answer
  that is being written survives the client closing stdin (see
  ``transport.py``).

Process lifecycle
-----------------
* ``SIGINT`` and ``SIGTERM`` terminate the process immediately with exit
  status 130 and 143 after flushing the logs. There is no state to persist,
  and an orderly shutdown would otherwise wait on the SDK's stdin reader
  thread, which only returns once the client closes the pipe.
* End of input (the client closed stdin) is the normal shutdown: exit status 0.
* Any other failure is logged as one JSON record and ends with exit status 1.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from collections.abc import Sequence
from contextlib import suppress
from types import FrameType

import anyio
from mcp.server.stdio import stdio_server

from customer_mcp_server import __version__
from customer_mcp_server.logging_config import configure_logging
from customer_mcp_server.server import build_server
from customer_mcp_server.service import CustomerService, InMemoryCustomerRepository
from customer_mcp_server.transport import (
    DEFAULT_MAX_FRAME_BYTES,
    BoundedLineReader,
    ProtocolErrorReportingReadStream,
    ShieldedWriteStream,
)

logger = logging.getLogger(__name__)

_TERMINATION_SIGNALS = (signal.SIGINT, signal.SIGTERM)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="customer-mcp-server",
        description="MCP server exposing customer tools over stdio.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity of the JSON logs written to stderr (default: INFO).",
    )
    parser.add_argument(
        "--max-frame-bytes",
        type=int,
        default=DEFAULT_MAX_FRAME_BYTES,
        help="Largest JSON-RPC frame accepted on stdin (default: 4 MiB).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)
    if args.max_frame_bytes < 1024:
        parser.error("--max-frame-bytes must be at least 1024")
    return args


def _flush_std_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        with suppress(Exception):
            stream.flush()


def _divert_python_stdout() -> None:
    """Send everything written through ``sys.stdout`` to stderr from now on.

    Called once the transport has claimed file descriptor 1, so the flush of
    anything already buffered lands on the diverted descriptor (stderr), never
    on the wire.
    """
    with suppress(Exception):
        sys.stdout.flush()
    sys.stdout = sys.stderr


def _exit_on_signal(signum: int, frame: FrameType | None) -> None:
    name = signal.Signals(signum).name
    logger.info("terminating on signal", extra={"event": "server.signal", "signal": name})
    _flush_std_streams()
    os._exit(128 + signum)


def install_signal_handlers() -> None:
    for sig in _TERMINATION_SIGNALS:
        signal.signal(sig, _exit_on_signal)


async def serve(max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES) -> None:
    service = CustomerService(InMemoryCustomerRepository.with_sample_data())
    server = build_server(service)
    logger.info("server starting", extra={"event": "server.start", "transport": "stdio"})
    reader = BoundedLineReader(sys.stdin.buffer, max_frame_bytes)
    # stdout stays with the SDK (it claims fd 1); stdin is read through the bounded reader.
    async with stdio_server(stdin=reader) as (read_stream, write_stream):  # type: ignore[arg-type]
        _divert_python_stdout()
        guarded_write_stream = ShieldedWriteStream(write_stream)
        reader.set_reporter(guarded_write_stream.send_protocol_error)
        guarded_read_stream = ProtocolErrorReportingReadStream(read_stream, guarded_write_stream)
        await server.run(
            guarded_read_stream, guarded_write_stream, server.create_initialization_options()
        )
    logger.info("server stopped", extra={"event": "server.stop"})


def _is_broken_pipe(exc: BaseException) -> bool:
    if isinstance(exc, BrokenPipeError | anyio.BrokenResourceError):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return all(_is_broken_pipe(member) for member in exc.exceptions)
    return False


def main(argv: Sequence[str] | None = None) -> int:
    """Run the server and return the process exit status."""
    args = _parse_args(argv)
    configure_logging(args.log_level)
    install_signal_handlers()
    try:
        anyio.run(serve, args.max_frame_bytes)
    except KeyboardInterrupt:  # pragma: no cover - signals normally exit in the handler
        logger.info("interrupted", extra={"event": "server.interrupted"})
        return 130
    except Exception as exc:
        if _is_broken_pipe(exc):
            # The client went away without closing stdin first: one line, no traceback.
            logger.warning("client closed the connection", extra={"event": "server.broken_pipe"})
            return 1
        logger.exception(
            "server terminated by an unexpected error", extra={"event": "server.crashed"}
        )
        return 1
    return 0


def run() -> None:
    """Console entry point.

    ``os._exit`` rather than ``sys.exit``: the SDK's stdin reader is a
    non-daemon worker thread blocked in ``readline`` until the client closes
    the pipe, and a regular interpreter shutdown would wait for it.
    """
    code = main()
    _flush_std_streams()
    os._exit(code)


if __name__ == "__main__":
    run()
