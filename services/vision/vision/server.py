"""Diffusion des détections sur un WebSocket local (ws://127.0.0.1:8765).

Destiné au futur `vision-bridge` (PLAN.md §5.1). Messages JSON, un par ligne :

    {"type": "hello", "version": "...", "state": "waiting"|"running"}
    {"type": "game", "gameId": ..., "patch": ..., "gameMode": ..., "champions": [...], "calibration": {...}}
    {"type": "detections", "gameId": ..., "frameId": ..., "t": ..., "latencyMs": ..., "detections": [...]}
    {"type": "status", "state": "waiting"|"running"|"ended", "gameId": ...}

Le serveur tourne dans son propre thread (boucle asyncio) ; `broadcast()` est
appelable depuis n'importe quel thread. Il n'écoute que sur 127.0.0.1 et
ignore tout message entrant : le service ne reçoit aucune commande.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading

import websockets
from websockets.asyncio.server import serve

from vision import __version__

log = logging.getLogger(__name__)


class DetectionServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host, self.port = host, port
        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self.last_game_message: dict | None = None
        self.state = "waiting"

    # ---- cycle de vie -----------------------------------------------------

    def start(self) -> "DetectionServer":
        self._thread = threading.Thread(target=self._run, name="ws-server", daemon=True)
        self._thread.start()
        if not self._ready.wait(5.0):
            log.warning("Le serveur WebSocket n'a pas démarré en 5 s.")
        return self

    def stop(self, timeout: float = 2.0) -> None:
        """Ferme proprement les connexions (après les derniers messages) et arrête la boucle."""
        if self._loop and self._stop:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread:
            self._thread.join(timeout)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except OSError as e:
            log.error("WebSocket indisponible sur %s:%d : %s", self.host, self.port, e)
            self._ready.set()
        finally:
            self._loop.close()

    async def _serve(self) -> None:
        self._stop = asyncio.Event()
        async with serve(self._handler, self.host, self.port):
            log.info("WebSocket : ws://%s:%d", self.host, self.port)
            self._ready.set()
            await self._stop.wait()
            await asyncio.sleep(0.1)  # laisse partir les derniers messages diffusés
            await asyncio.gather(*(ws.close() for ws in list(self._clients)), return_exceptions=True)

    async def _handler(self, ws) -> None:
        self._clients.add(ws)
        try:
            await ws.send(json.dumps({"type": "hello", "version": __version__, "state": self.state}))
            if self.last_game_message:
                await ws.send(json.dumps(self.last_game_message, ensure_ascii=False))
            async for _ in ws:  # les messages entrants sont ignorés (lecture seule)
                pass
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(ws)

    # ---- diffusion --------------------------------------------------------

    @property
    def clients(self) -> int:
        return len(self._clients)

    def broadcast(self, message: dict) -> None:
        if message.get("type") == "game":
            self.last_game_message = message
        if message.get("type") == "status":
            self.state = message.get("state", self.state)
        if not self._loop or not self._clients:
            return
        data = json.dumps(message, ensure_ascii=False)
        self._loop.call_soon_threadsafe(self._broadcast_now, data)

    def _broadcast_now(self, data: str) -> None:
        for ws in list(self._clients):
            asyncio.ensure_future(self._send(ws, data))

    async def _send(self, ws, data: str) -> None:
        try:
            await ws.send(data)
        except websockets.ConnectionClosed:
            self._clients.discard(ws)
