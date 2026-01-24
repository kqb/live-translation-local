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
    scrolling_mode: bool = True  # YouTube-style scrolling subtitles
    history_lines: int = 10  # Number of historical lines to keep visible


# HTML template for browser source overlay with webcam
HTML_OVERLAY = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Translation with Webcam</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            background: #000;
            font-family: 'Hiragino Sans', 'Hiragino Kaku Gothic ProN', 'Yu Gothic', 'Noto Sans JP', 'Meiryo', 'MS Gothic', sans-serif;
            overflow: hidden;
            width: 100vw;
            height: 100vh;
            position: relative;
        }

        #video-container {
            width: 100%;
            height: 100%;
            position: relative;
            background: #000;
        }

        #webcam {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        #error-message {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            text-align: center;
            padding: 20px;
            background: rgba(0, 0, 0, 0.8);
            border-radius: 8px;
            display: none;
        }

        #error-message.visible {
            display: block;
        }

        #subtitle-container {
            position: absolute;
            bottom: 50px;
            left: 50%;
            transform: translateX(-50%);
            max-width: 90%;
            text-align: center;
            opacity: 0;
            transition: opacity 0.3s ease-in-out;
            z-index: 10;
        }

        #subtitle-container.visible {
            opacity: 1;
        }

        .original-text {
            color: white;
            font-size: 28px;
            font-weight: 400;
            text-shadow:
                2px 2px 4px rgba(0, 0, 0, 0.8),
                -1px -1px 2px rgba(0, 0, 0, 0.6),
                1px -1px 2px rgba(0, 0, 0, 0.6),
                -1px 1px 2px rgba(0, 0, 0, 0.6);
            margin-bottom: 8px;
            line-height: 1.4;
        }

        .translated-text {
            color: #87CEEB;
            font-size: 32px;
            font-weight: 500;
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

        /* Scrolling subtitle styles */
        .subtitle-block {
            margin-bottom: 16px;
            opacity: 0;
            transform: translateY(20px);
        }

        .subtitle-block.slide-up {
            animation: slideUp 0.4s ease-out forwards;
        }

        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
    </style>
</head>
<body>
    <div id="video-container">
        <video id="webcam" autoplay playsinline></video>
        <div id="error-message">
            <h2>Camera Access Required</h2>
            <p>Please allow camera access to continue.</p>
        </div>
    </div>

    <div id="subtitle-container">
        <div class="original-text" id="original"></div>
        <div class="translated-text" id="translated"></div>
    </div>

    <script>
        const videoEl = document.getElementById('webcam');
        const errorEl = document.getElementById('error-message');
        const container = document.getElementById('subtitle-container');
        const originalEl = document.getElementById('original');
        const translatedEl = document.getElementById('translated');

        let ws = null;
        let reconnectTimer = null;
        let clearTimer = null;
        const RECONNECT_INTERVAL = 2000;

        // Initialize webcam
        async function initWebcam() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 1920 },
                        height: { ideal: 1080 },
                        facingMode: 'user'
                    },
                    audio: false
                });
                videoEl.srcObject = stream;
                errorEl.classList.remove('visible');
            } catch (err) {
                console.error('Error accessing webcam:', err);
                errorEl.classList.add('visible');
            }
        }

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

            if (data.type === 'scrolling_subtitle') {
                // YouTube-style scrolling subtitles
                updateScrollingSubtitles(data.history || []);
            } else if (data.type === 'subtitle') {
                // Original fade in/out behavior
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

        function updateScrollingSubtitles(history) {
            // Clear existing subtitles safely
            while (container.firstChild) {
                container.removeChild(container.firstChild);
            }

            // Add each history entry
            history.forEach(function(entry, index) {
                const subtitleBlock = document.createElement('div');
                subtitleBlock.className = 'subtitle-block slide-up';
                subtitleBlock.style.animationDelay = (index * 0.05) + 's';

                if (entry.original) {
                    const origDiv = document.createElement('div');
                    origDiv.className = 'original-text';
                    origDiv.textContent = entry.original;
                    subtitleBlock.appendChild(origDiv);
                }

                if (entry.translated) {
                    const transDiv = document.createElement('div');
                    transDiv.className = 'translated-text';
                    transDiv.textContent = entry.translated;
                    subtitleBlock.appendChild(transDiv);
                }

                container.appendChild(subtitleBlock);
            });

            container.classList.add('visible');
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

        // Initialize on load
        initWebcam();
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
            # Write both original and translated for hybrid input
            if translated:
                content = f"{original}\n{translated}"
            else:
                content = original
            self._text_file.write_text(content, encoding="utf-8")
        except Exception:
            pass  # Don't crash on file write errors

    def _write_scrolling_text_file(self) -> None:
        """Write scrolling subtitle history to text file.

        For scrolling mode, only write the most recent entry to keep
        the text file clean for OBS Text (GDI+) sources.
        The browser overlay will show the full scrolling history.
        """
        try:
            if not self._subtitle_history:
                self._text_file.write_text("", encoding="utf-8")
                return

            # Get most recent entry
            latest = self._subtitle_history[-1]

            if latest["translated"]:
                content = f"{latest['original']}\n{latest['translated']}"
            else:
                content = latest["original"]

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

        if self.config.scrolling_mode:
            # Add to history
            self._subtitle_history.append({
                "original": original,
                "translated": translated,
                "timestamp": time.time(),
            })

            # Keep only recent history
            max_history = self.config.history_lines
            if len(self._subtitle_history) > max_history:
                self._subtitle_history = self._subtitle_history[-max_history:]

            # Write all history to text file
            self._write_scrolling_text_file()

            # Broadcast history to WebSocket clients
            if self.config.websocket_enabled and self._loop and not self._loop.is_closed():
                message = {
                    "type": "scrolling_subtitle",
                    "history": self._subtitle_history,
                    "scrolling": True,
                }

                try:
                    asyncio.run_coroutine_threadsafe(
                        self._broadcast_message(message),
                        self._loop,
                    )
                except Exception:
                    pass
        else:
            # Original behavior: replace subtitles
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
                if self._loop and not self._loop.is_closed():
                    try:
                        asyncio.run_coroutine_threadsafe(
                            self._broadcast_message(message),
                            self._loop,
                        )
                    except Exception:
                        pass

    def clear(self) -> None:
        """Clear the subtitles."""
        self._subtitle_history.clear()
        self._write_text_file("", "")

        if self.config.websocket_enabled and self._loop and not self._loop.is_closed():
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
