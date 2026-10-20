"""Capture de l'écran ou de la fenêtre du jeu avec `mss`.

- `ScreenSource` : un moniteur entier, une région explicite, ou la fenêtre du
  jeu retrouvée via user32 sur Windows ou Quartz sur macOS (dépendance
  optionnelle `pyobjc`).
- `ImageSource` : un fichier PNG ou un dossier de PNG, pour tester toute la
  chaîne sans partie en cours.

Sur Windows, `mss` capture par GDI sans autorisation particulière ; le jeu
doit être en « fenêtré sans bordure » ou fenêtré (le plein écran exclusif
peut donner une image noire).

Sur macOS, `mss` a besoin de l'autorisation « Enregistrement de l'écran » pour
l'application qui lance Python (Terminal, iTerm…). Sans elle, la capture
renvoie le fond d'écran sans aucune fenêtre, sans erreur. Les coordonnées
passées à `mss` sont en points ; les images renvoyées sont en pixels
physiques (×2 sur un écran Retina).

Chaque frame est horodatée avec le `gameTime` courant fourni par un appelable
(`clock`), typiquement `GameClock.now`.
"""

from __future__ import annotations

import logging
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

log = logging.getLogger(__name__)

GAME_WINDOW_OWNERS = ("League of Legends", "LeagueofLegends", "League of Legends (TM) Client")
GAME_WINDOW_TITLES_WINDOWS = ("League of Legends (TM) Client",)


@dataclass
class Frame:
    index: int
    image: np.ndarray  # BGR
    t: float  # gameTime estimé (s)
    wall: float  # time.time()

    @property
    def id(self) -> str:
        return f"{self.index:06d}"

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return w, h


def find_game_window_macos() -> dict | None:
    """Rectangle (points) de la fenêtre du jeu via Quartz, ou None.

    Lecture seule : on ne fait qu'interroger la liste des fenêtres à l'écran.
    """
    try:
        import Quartz  # type: ignore
    except ImportError:
        return None
    windows = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
        Quartz.kCGNullWindowID,
    )
    best = None
    for w in windows or []:
        owner = str(w.get("kCGWindowOwnerName", ""))
        if not any(o.lower() in owner.lower() for o in GAME_WINDOW_OWNERS):
            continue
        b = w.get("kCGWindowBounds", {})
        rect = {"left": int(b.get("X", 0)), "top": int(b.get("Y", 0)),
                "width": int(b.get("Width", 0)), "height": int(b.get("Height", 0))}
        # On garde la plus grande fenêtre du jeu (les autres sont des surfaces auxiliaires).
        if rect["width"] * rect["height"] > (best["width"] * best["height"] if best else 0):
            best = rect
    return best


def find_game_window_windows() -> dict | None:
    """Rectangle (pixels) de la fenêtre du jeu via user32 (FindWindowW + GetClientRect), ou None.

    Lecture seule : aucune interaction avec le processus, on ne fait que lire
    la position de sa fenêtre. En plein écran exclusif, la capture GDI peut
    renvoyer du noir : préférer le mode « fenêtré sans bordure ».
    """
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    for title in GAME_WINDOW_TITLES_WINDOWS:
        hwnd = user32.FindWindowW(None, title)
        if not hwnd:
            continue
        rect = wintypes.RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
            continue
        origin = wintypes.POINT(0, 0)
        user32.ClientToScreen(hwnd, ctypes.byref(origin))
        w, h = rect.right - rect.left, rect.bottom - rect.top
        if w > 0 and h > 0:
            return {"left": int(origin.x), "top": int(origin.y), "width": int(w), "height": int(h)}
    return None


def find_game_window() -> dict | None:
    system = platform.system()
    if system == "Darwin":
        return find_game_window_macos()
    if system == "Windows":
        return find_game_window_windows()
    return None


class ScreenSource:
    """Capture périodique d'une zone de l'écran."""

    def __init__(self, fps: float = 2.0, monitor: int = 1, region: dict | None = None,
                 window: bool = False, clock: Callable[[], float] | None = None):
        import mss

        self.fps = fps
        self.clock = clock or (lambda: 0.0)
        self._sct = mss.mss()
        if region is not None:
            self.region = region
        elif window:
            found = find_game_window()
            if found is None:
                log.warning("Fenêtre du jeu introuvable (jeu non lancé, ou pyobjc absent sur macOS) : "
                            "capture du moniteur %d.", monitor)
                self.region = self._sct.monitors[monitor]
            else:
                log.info("Fenêtre du jeu : %s", found)
                self.region = found
        else:
            if monitor >= len(self._sct.monitors):
                raise ValueError(f"Moniteur {monitor} inexistant ({len(self._sct.monitors) - 1} disponibles).")
            self.region = self._sct.monitors[monitor]
        self._index = 0

    def grab(self) -> Frame:
        shot = self._sct.grab(self.region)
        img = np.asarray(shot, dtype=np.uint8)[:, :, :3]  # BGRA → BGR (vue, puis copie contiguë)
        frame = Frame(index=self._index, image=np.ascontiguousarray(img), t=self.clock(), wall=time.time())
        self._index += 1
        return frame

    def frames(self, stop: Callable[[], bool] | None = None) -> Iterator[Frame]:
        """Itère à `fps` images/s en absorbant le temps de traitement de l'appelant."""
        period = 1.0 / self.fps
        next_at = time.monotonic()
        while stop is None or not stop():
            yield self.grab()
            next_at += period
            delay = next_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_at = time.monotonic()  # en retard : on ne rattrape pas, on repart d'ici

    def close(self) -> None:
        self._sct.close()


class ImageSource:
    """Source de frames depuis un PNG (répété) ou un dossier de PNG (ordre alphabétique)."""

    def __init__(self, path: Path | str, fps: float = 2.0, loop: bool = False,
                 clock: Callable[[], float] | None = None, realtime: bool = False):
        self.path = Path(path)
        self.fps = fps
        self.loop = loop
        self.realtime = realtime
        self.clock = clock
        if self.path.is_dir():
            self.files = sorted(p for p in self.path.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
        else:
            self.files = [self.path]
        if not self.files:
            raise FileNotFoundError(f"Aucune image dans {self.path}")

    def frames(self, stop: Callable[[], bool] | None = None) -> Iterator[Frame]:
        index = 0
        period = 1.0 / self.fps
        while True:
            for f in self.files:
                if stop is not None and stop():
                    return
                img = cv2.imread(str(f), cv2.IMREAD_COLOR)
                if img is None:
                    log.warning("Image illisible : %s", f)
                    continue
                t = self.clock() if self.clock else index * period
                yield Frame(index=index, image=img, t=t, wall=time.time())
                index += 1
                if self.realtime:
                    time.sleep(period)
            if not self.loop:
                return

    def close(self) -> None:
        pass
