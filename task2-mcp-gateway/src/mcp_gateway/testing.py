"""Test support: run an ASGI application on a real local port in a thread."""

from __future__ import annotations

import threading
import time
from typing import Any

import uvicorn


class UvicornTestServer:
    """Serve ``app`` on an ephemeral port from a background thread."""

    def __init__(self, app: Any, host: str = "127.0.0.1") -> None:
        config = uvicorn.Config(
            app, host=host, port=0, log_level="warning", log_config=None, access_log=False
        )
        self._host = host
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self.base_url = ""

    def start(self, timeout: float = 10.0) -> str:
        self._thread.start()
        deadline = time.monotonic() + timeout
        while not self._server.started:
            if time.monotonic() > deadline or not self._thread.is_alive():
                raise RuntimeError("uvicorn test server failed to start")
            time.sleep(0.01)
        port = self._server.servers[0].sockets[0].getsockname()[1]
        self.base_url = f"http://{self._host}:{port}"
        return self.base_url

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)

    def __enter__(self) -> UvicornTestServer:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
