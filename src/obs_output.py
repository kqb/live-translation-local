"""OBS output module with text file and WebSocket/HTTP server for browser source."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from aiohttp import web
import websockets
from websockets.server import serve as ws_serve


@dataclass
class OutputConfig:
    """Configuration for OBS output."""

    text_file: str = "./subtitles.txt"
    websocket_enabled: bool = True
    websocket_port: int = 8765
    http_port: int = 8766
    max_lines: int = 2
    clear_after: float = 5.0  # seconds


# HTML template for browser source overlay
HTML_OVERLAY = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Translation Overlay</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            background: transparent;
            font-family: 'Segoe UI', 'Arial', sans-serif;
            overflow: hidden;
            width: 100vw;
            height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
            align-items: center;
            padding-bottom: 50px;
        }

        #subtitle-container {
            max-width: 90%;
            text-align: center;
            opacity: 0;
            transition: opacity 0.3s ease-in-out;
        }

        #subtitle-container.visible {
            opacity: 1;
        }

        .original-text {
            color: white;
            font-size: 32px;
            font-weight: 500;
            text-shadow:
                2px 2px 4px rgba(0, 0, 0, 0.8),
                -1px -1px 2px rgba(0, 0, 0, 0.6),
                1px -1px 2px rgba(0, 0, 0, 0.6),
                -1px 1px 2px rgba(0, 0, 0, 0.6);
            margin-bottom: 10px;
            line-height: 1.4;
        }

        .translated-text {
            color: #87CEEB;
            font-size: 28px;
            font-weight: 400;
            font-style: italic;
            text-shadow:
                2px 2px 4px rgba(0, 0, 0, 0.8),
                -1px -1px 2px rgba(0, 0, 0, 0.6),
                1px -1px 2px rgba(0, 0, 0, 0.6),
                -1px 1px 2px rgba(0, 0, 0, 0.6);
            line-height: 1.4;
        }

        .fade-out {
            animation: fadeOut 0.5s ease-out forwards;
        }

        @keyframes fadeOut {
            from { opacity: 1; }
            to { opacity: 0; }
        }

        .fade-in {
            animation: fadeIn 0.3s ease-in forwards;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
    </style>
</head>
<body>
    <div id="subtitle-container">
        <div class="original-text" id="original"></div>
        <div class="translated-text" id="translated"></div>
    </div>

    <script>
        const container = document.getElementById('subtitle-container');
        const originalEl = document.getElementById('original');
        const translatedEl = document.getElementById('translated');

        let ws = null;
        let reconnectTimer = null;
        let clearTimer = null;
        const RECONNECT_INTERVAL = 2000;
        const CLEAR_TIMEOUT = 5000;

        function connect() {
            const wsPort = window.location.port ?
                parseInt(window.location.port) - 1 : 8765;
            const wsUrl = `ws://${window.location.hostname}:${wsPort}`;

            ws = new WebSocket(wsUrl);

            ws.onopen = function() {
                console.log('WebSocket connected');
                if (reconnectTimer) {
                    clearInterval(reconnectTimer);
                    reconnectTimer = null;
                }
            };

            ws.onmessage = function(event) {
                try {
                    const data = JSON.parse(event.data);
                    updateSubtitles(data);
                } catch (e) {
                    console.error('Error parsing message:', e);
                }
            };

            ws.onclose = function() {
                console.log('WebSocket closed, reconnecting...');
                scheduleReconnect();
            };

            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
                ws.close();
            };
        }

        function scheduleReconnect() {
            if (!reconnectTimer) {
                reconnectTimer = setInterval(function() {
                    console.log('Attempting to reconnect...');
                    connect();
                }, RECONNECT_INTERVAL);
            }
        }

        function updateSubtitles(data) {
            // Clear any pending clear timer
            if (clearTimer) {
                clearTimeout(clearTimer);
                clearTimer = null;
            }

            if (data.type === 'subtitle') {
                originalEl.textContent = data.original || '';
                translatedEl.textContent = data.translated || '';

                if (data.original || data.translated) {
                    container.classList.remove('fade-out');
                    container.classList.add('visible', 'fade-in');

                    // Schedule auto-clear
                    const clearAfter = (data.clearAfter || 5) * 1000;
                    clearTimer = setTimeout(function() {
                        clearSubtitles();
                    }, clearAfter);
                }
            } else if (data.type === 'clear') {
                clearSubtitles();
            }
        }

        function clearSubtitles() {
            container.classList.remove('fade-in');
            container.classList.add('fade-out');

            setTimeout(function() {
                originalEl.textContent = '';
                translatedEl.textContent = '';
                container.classList.remove('visible', 'fade-out');
            }, 500);
        }

        // Initial connection
        connect();
    </script>
</body>
</html>
"""


class OBSOutput:
    """Output handler for OBS Studio integration."""

    def __init__(self, config: OutputConfig):
        """Initialize OBS output.

        Args:
            config: Output configuration.
        """
        self.config = config

        # Text file path
        self._text_file = Path(config.text_file)

        # WebSocket clients
        self._ws_clients: set = set()
        self._ws_lock = threading.Lock()

        # Server state
        self._ws_server = None
        self._http_runner = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server_thread: Optional[threading.Thread] = None
        self._running = False

        # Subtitle history (for max_lines)
        self._subtitle_history: list[dict] = []

        # Last update time for auto-clear
        self._last_update_time = 0.0

    def _write_text_file(self, original: str, translated: str) -> None:
        """Write subtitles to text file.

        Args:
            original: Original transcribed text.
            translated: Translated text.
        """
        try:
            content = f"{original}\n{translated}" if translated else original
            self._text_file.write_text(content, encoding="utf-8")
        except Exception:
            pass  # Don't crash on file write errors

    async def _broadcast_message(self, message: dict) -> None:
        """Broadcast a message to all WebSocket clients.

        Args:
            message: Message to broadcast.
        """
        if not self._ws_clients:
            return

        message_json = json.dumps(message)
        disconnected = set()

        with self._ws_lock:
            clients = self._ws_clients.copy()

        for client in clients:
            try:
                await client.send(message_json)
            except Exception:
                disconnected.add(client)

        # Remove disconnected clients
        if disconnected:
            with self._ws_lock:
                self._ws_clients -= disconnected

    async def _ws_handler(self, websocket) -> None:
        """Handle a WebSocket connection.

        Args:
            websocket: WebSocket connection.
        """
        with self._ws_lock:
            self._ws_clients.add(websocket)

        try:
            async for _ in websocket:
                pass  # We don't expect incoming messages
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            with self._ws_lock:
                self._ws_clients.discard(websocket)

    async def _http_handler(self, request: web.Request) -> web.Response:
        """Handle HTTP requests for the overlay.

        Args:
            request: HTTP request.

        Returns:
            HTTP response with HTML overlay.
        """
        return web.Response(text=HTML_OVERLAY, content_type="text/html")

    async def _run_servers(self) -> None:
        """Run WebSocket and HTTP servers."""
        if not self.config.websocket_enabled:
            return

        # Start WebSocket server
        self._ws_server = await ws_serve(
            self._ws_handler,
            "0.0.0.0",
            self.config.websocket_port,
        )

        # Start HTTP server
        app = web.Application()
        app.router.add_get("/", self._http_handler)
        app.router.add_get("/overlay", self._http_handler)

        runner = web.AppRunner(app)
        await runner.setup()
        self._http_runner = runner

        site = web.TCPSite(runner, "0.0.0.0", self.config.http_port)
        await site.start()

        # Keep running until stopped
        while self._running:
            await asyncio.sleep(0.1)

        # Cleanup
        self._ws_server.close()
        await self._ws_server.wait_closed()
        await runner.cleanup()

    def _server_thread_target(self) -> None:
        """Target function for server thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._run_servers())
        except Exception:
            pass
        finally:
            self._loop.close()

    def start(self) -> None:
        """Start the output servers."""
        if self._running:
            return

        self._running = True

        # Initialize text file
        self._write_text_file("", "")

        if self.config.websocket_enabled:
            # Start server thread
            self._server_thread = threading.Thread(
                target=self._server_thread_target,
                daemon=True,
            )
            self._server_thread.start()

            # Give servers time to start
            time.sleep(0.5)

    def stop(self) -> None:
        """Stop the output servers."""
        self._running = False

        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)
            self._server_thread = None

        # Clear text file
        self._write_text_file("", "")

    def update(self, original: str, translated: str) -> None:
        """Update the subtitles.

        Args:
            original: Original transcribed text.
            translated: Translated text.
        """
        self._last_update_time = time.time()

        # Write to text file
        self._write_text_file(original, translated)

        # Broadcast to WebSocket clients
        if self.config.websocket_enabled and self._loop is not None:
            message = {
                "type": "subtitle",
                "original": original,
                "translated": translated,
                "clearAfter": self.config.clear_after,
            }

            # Schedule broadcast on the event loop
            try:
                asyncio.run_coroutine_threadsafe(
                    self._broadcast_message(message),
                    self._loop,
                )
            except Exception:
                pass

    def clear(self) -> None:
        """Clear the subtitles."""
        self._write_text_file("", "")

        if self.config.websocket_enabled and self._loop is not None:
            message = {"type": "clear"}

            try:
                asyncio.run_coroutine_threadsafe(
                    self._broadcast_message(message),
                    self._loop,
                )
            except Exception:
                pass

    def should_clear(self) -> bool:
        """Check if subtitles should be cleared due to silence.

        Returns:
            True if subtitles should be cleared.
        """
        if self._last_update_time == 0:
            return False

        elapsed = time.time() - self._last_update_time
        return elapsed >= self.config.clear_after

    @property
    def is_running(self) -> bool:
        """Check if servers are running."""
        return self._running

    @property
    def websocket_url(self) -> str:
        """Get the WebSocket URL."""
        return f"ws://localhost:{self.config.websocket_port}"

    @property
    def overlay_url(self) -> str:
        """Get the HTTP overlay URL."""
        return f"http://localhost:{self.config.http_port}/overlay"

    def __enter__(self) -> "OBSOutput":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
