"""Calibration de la minimap : rectangle dans la capture et mapping pixel → carte.

Trois sources, par priorité :
1. `calibration.json` (ajustement manuel sauvegardé par résolution de capture) ;
2. estimation depuis la résolution et `MinimapScale` lu dans `game.cfg` ;
3. estimation avec `MinimapScale = 1.0` si `game.cfg` est introuvable.

Convention des coordonnées carte normalisées : x de 0 (ouest) à 1 (est), y de
0 (sud) à 1 (nord). C'est l'orientation des positions de l'API Riot (timeline),
donc la base ORDER (bleue) est près de (0, 0) et la base CHAOS (rouge) près de
(1, 1). Sur la minimap, le nord est en haut : y est donc inversé par rapport
aux pixels.
"""

from __future__ import annotations

import json
import logging
import os
import platform
from dataclasses import dataclass
from pathlib import Path

from vision.models import Calibration, MinimapRect, Pos

log = logging.getLogger(__name__)

CALIBRATION_FILE = Path(__file__).resolve().parent.parent / "calibration.json"

# Côté de la minimap rapporté à la hauteur de la capture, pour MinimapScale = 0
# et 1. Valeurs issues de mesures communautaires (≈ 283 px pour 1080p à
# l'échelle 1.0) : ce sont des estimations à confirmer avec `calibrate`.
SIDE_RATIO_MIN = 0.17
SIDE_RATIO_MAX = 0.262
# Diamètre d'une icône de champion rapporté au côté de la minimap (≈ 24 px pour 283 px).
ICON_RATIO = 0.085
# Marge entre le bord de l'écran et la minimap (cadre du HUD), en fraction de la hauteur.
EDGE_MARGIN_RATIO = 0.0

# Candidats pour game.cfg. Les chemins macOS n'ont pas pu être vérifiés sur la
# machine de développement (jeu non installé) : `--game-cfg` permet d'en donner un.
GAME_CFG_CANDIDATES = {
    "Darwin": [
        "/Applications/League of Legends.app/Contents/LoL/Config/game.cfg",
        "~/Applications/League of Legends.app/Contents/LoL/Config/game.cfg",
    ],
    "Windows": [
        r"C:\Riot Games\League of Legends\Config\game.cfg",
        r"D:\Riot Games\League of Legends\Config\game.cfg",
    ],
    "Linux": [
        "~/.local/share/lutris/runners/winesteam/drive_c/Riot Games/League of Legends/Config/game.cfg",
    ],
}

# Carte 11 (Faille de l'invocateur), unités Riot : bornes publiées par Riot pour
# la timeline. Repères utilisés pour vérifier visuellement la calibration.
MAP_MIN = (-120.0, -120.0)
MAP_MAX = (14870.0, 14980.0)
LANDMARKS_UNITS: dict[str, tuple[float, float]] = {
    # Nexus (approximatifs) et tours extérieures (positions des événements BUILDING_KILL).
    "nexus ORDER": (1550.0, 1650.0),
    "nexus CHAOS": (13270.0, 13240.0),
    "T1 top ORDER": (981.0, 10441.0),
    "T1 mid ORDER": (5846.0, 6396.0),
    "T1 bot ORDER": (10504.0, 1029.0),
    "T1 top CHAOS": (4318.0, 13875.0),
    "T1 mid CHAOS": (8955.0, 8510.0),
    "T1 bot CHAOS": (13866.0, 4505.0),
}


def units_to_norm(x: float, y: float) -> Pos:
    """Unités Riot → coordonnées normalisées (0..1, y vers le nord)."""
    return Pos(x=(x - MAP_MIN[0]) / (MAP_MAX[0] - MAP_MIN[0]),
               y=(y - MAP_MIN[1]) / (MAP_MAX[1] - MAP_MIN[1]))


LANDMARKS: dict[str, Pos] = {k: units_to_norm(*v) for k, v in LANDMARKS_UNITS.items()}
# Point d'apparition (plateforme de la fontaine) de chaque équipe : les champions y réapparaissent
# et y rappellent, empilés, l'anneau mangé par l'arc bleu de la plateforme et l'arc vert de soin.
SPAWN_UNITS: dict[str, tuple[float, float]] = {"ORDER": (400.0, 465.0), "CHAOS": (14350.0, 14395.0)}
SPAWN: dict[str, Pos] = {k: units_to_norm(*v) for k, v in SPAWN_UNITS.items()}

# Toutes les structures de la Faille (tours, inhibiteurs, nexus), en unités Riot : leurs icônes
# sur la minimap (boucliers bleus et rouges) ne doivent pas être prises pour des anneaux.
STRUCTURES_UNITS: list[tuple[float, float]] = [
    # ORDER
    (981, 10441), (1512, 6699), (1169, 4287), (1171, 3571),
    (5846, 6396), (5048, 4812), (3651, 3696), (3203, 3208),
    (10504, 1029), (6919, 1483), (4281, 1253), (3452, 1236),
    (2177, 1807), (1748, 2270), (1551, 1659),
    # CHAOS
    (4318, 13875), (7943, 13411), (10481, 13650), (11261, 13676),
    (8955, 8510), (9767, 10113), (11134, 11207), (11598, 11667),
    (13866, 4505), (13327, 8226), (13624, 10572), (13604, 11316),
    (12611, 13084), (13052, 12612), (13269, 13241),
]
STRUCTURES: list[Pos] = [units_to_norm(x, y) for x, y in STRUCTURES_UNITS]
# Repères à cliquer pour l'ajustement automatique du rectangle (loin les uns des autres).
FIT_LANDMARKS = ["T1 top ORDER", "T1 top CHAOS", "T1 bot ORDER", "T1 bot CHAOS"]


def fit_rect_from_points(points: list[tuple[str, float, float]]) -> tuple[MinimapRect, float]:
    """Rectangle carré de la minimap à partir de repères cliqués (nom, x, y en pixels de la capture).

    Modèle : px = x0 + s·nx, py = y0 + s·(1 − ny), inconnues (x0, y0, s) résolues
    par moindres carrés. Renvoie le rectangle et le résidu moyen en pixels.
    """
    import numpy as np

    rows, rhs = [], []
    for name, px, py in points:
        n = LANDMARKS[name]
        rows.append([1.0, 0.0, n.x]); rhs.append(px)
        rows.append([0.0, 1.0, 1.0 - n.y]); rhs.append(py)
    if len(points) < 2:
        raise ValueError("Au moins deux repères sont nécessaires.")
    sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    x0, y0, side = (float(v) for v in sol)
    residual = float(np.sqrt(np.mean((np.array(rows) @ sol - np.array(rhs)) ** 2)))
    side_i = max(32, int(round(side)))
    return MinimapRect(x=int(round(x0)), y=int(round(y0)), w=side_i, h=side_i), residual


# ---- game.cfg -------------------------------------------------------------

@dataclass
class GameConfig:
    width: int | None = None
    height: int | None = None
    minimap_scale: float = 1.0
    global_scale: float = 0.0
    flip_minimap: bool = False
    windowed: bool | None = None
    path: Path | None = None


def parse_game_cfg(text: str) -> GameConfig:
    """Parse le `game.cfg` du jeu (format INI, sections [General] et [HUD])."""
    cfg = GameConfig()
    section = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith((";", "#")):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        try:
            if section == "general":
                if key == "Width":
                    cfg.width = int(float(value))
                elif key == "Height":
                    cfg.height = int(float(value))
                elif key == "Windowed":
                    cfg.windowed = value not in ("0", "false", "False")
            elif section == "hud":
                if key == "MinimapScale":
                    cfg.minimap_scale = max(0.0, min(1.0, float(value)))
                elif key == "GlobalScale":
                    cfg.global_scale = max(0.0, min(1.0, float(value)))
                elif key == "FlipMiniMap":
                    cfg.flip_minimap = value not in ("0", "false", "False")
        except ValueError:
            log.warning("Valeur illisible dans game.cfg : %s=%s", key, value)
    return cfg


def find_game_cfg(explicit: str | os.PathLike | None = None) -> Path | None:
    """Trouve `game.cfg` : chemin explicite, variable LOL_GAME_CFG, puis candidats par OS."""
    candidates: list[str] = []
    if explicit:
        candidates.append(str(explicit))
    if os.environ.get("LOL_GAME_CFG"):
        candidates.append(os.environ["LOL_GAME_CFG"])
    candidates += GAME_CFG_CANDIDATES.get(platform.system(), [])
    for c in candidates:
        p = Path(os.path.expanduser(c))
        if p.is_file():
            return p
    return None


def load_game_cfg(explicit: str | os.PathLike | None = None) -> GameConfig:
    p = find_game_cfg(explicit)
    if p is None:
        log.warning("game.cfg introuvable : MinimapScale=1.0 supposé. Utilise `calibrate` pour ajuster.")
        return GameConfig()
    cfg = parse_game_cfg(p.read_text(encoding="utf-8", errors="replace"))
    cfg.path = p
    log.info("game.cfg : %s (MinimapScale=%.2f, FlipMiniMap=%s, %sx%s)",
             p, cfg.minimap_scale, cfg.flip_minimap, cfg.width, cfg.height)
    return cfg


# ---- estimation -----------------------------------------------------------

def estimate_calibration(width: int, height: int, minimap_scale: float = 1.0, flipped: bool = False) -> Calibration:
    """Estime le rectangle de la minimap pour une capture `width`×`height`.

    La minimap est carrée, collée dans le coin inférieur droit (inférieur
    gauche si `FlipMiniMap`), et son côté croît linéairement avec
    `MinimapScale`.
    """
    scale = max(0.0, min(1.0, minimap_scale))
    ratio = SIDE_RATIO_MIN + (SIDE_RATIO_MAX - SIDE_RATIO_MIN) * scale
    side = int(round(height * ratio))
    side = max(32, min(side, width, height))
    margin = int(round(height * EDGE_MARGIN_RATIO))
    y = height - side - margin
    x = margin if flipped else width - side - margin
    return Calibration(
        width=width, height=height,
        minimap=MinimapRect(x=x, y=y, w=side, h=side),
        iconDiameter=max(8, int(round(side * ICON_RATIO))),
        minimapScale=scale, flipped=flipped, source="estimate",
    )


# ---- fichier calibration.json --------------------------------------------

def load_calibrations(path: Path = CALIBRATION_FILE) -> dict[str, Calibration]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: Calibration.model_validate(v) for k, v in data.items()}


def save_calibration(calib: Calibration, path: Path = CALIBRATION_FILE) -> None:
    all_ = load_calibrations(path)
    all_[calib.key] = calib
    path.write_text(json.dumps({k: v.model_dump() for k, v in all_.items()}, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    log.info("Calibration %s sauvegardée dans %s", calib.key, path)


def get_calibration(width: int, height: int, game_cfg: GameConfig | None = None,
                    path: Path = CALIBRATION_FILE) -> Calibration:
    """Calibration pour une résolution de capture : manuelle si elle existe, sinon estimée."""
    saved = load_calibrations(path).get(f"{width}x{height}")
    if saved is not None:
        return saved
    cfg = game_cfg or GameConfig()
    return estimate_calibration(width, height, cfg.minimap_scale, cfg.flip_minimap)


# ---- mapping pixel ↔ carte --------------------------------------------------

class MinimapMapper:
    """Conversion entre pixels (capture ou crop) et coordonnées carte normalisées."""

    def __init__(self, calib: Calibration):
        self.calib = calib
        self.rect = calib.minimap
        # Marge effective autour du rectangle (bornée par les bords de la capture) : les icônes de la
        # fontaine dépassent du carré de la minimap, et un template ne peut pas se poser à cheval sur
        # le bord du crop. Les coordonnées `px` sont celles du crop élargi.
        m = max(0, calib.cropMargin)
        self.margin_left = min(m, self.rect.x)
        self.margin_top = min(m, self.rect.y)
        self.margin_right = min(m, max(0, calib.width - self.rect.x2))
        self.margin_bottom = min(m, max(0, calib.height - self.rect.y2))

    def crop_to_map(self, px: float, py: float) -> Pos:
        """Pixel du crop minimap (origine en haut à gauche du crop élargi) → carte (0..1, y vers le nord)."""
        x = (px - self.margin_left) / self.rect.w
        y = 1.0 - (py - self.margin_top) / self.rect.h
        return Pos(x=min(1.0, max(0.0, x)), y=min(1.0, max(0.0, y)))

    def map_to_crop(self, pos: Pos) -> tuple[float, float]:
        return pos.x * self.rect.w + self.margin_left, (1.0 - pos.y) * self.rect.h + self.margin_top

    def frame_to_map(self, px: float, py: float) -> Pos:
        return self.crop_to_map(px - self.rect.x + self.margin_left, py - self.rect.y + self.margin_top)

    def map_to_frame(self, pos: Pos) -> tuple[float, float]:
        cx, cy = self.map_to_crop(pos)
        return cx + self.rect.x - self.margin_left, cy + self.rect.y - self.margin_top

    def crop(self, frame):
        """Extrait le crop minimap élargi de la marge d'une frame (numpy BGR), sans copie."""
        r = self.rect
        return frame[r.y - self.margin_top:r.y2 + self.margin_bottom, r.x - self.margin_left:r.x2 + self.margin_right]

    def landmarks_crop(self) -> dict[str, tuple[float, float]]:
        """Repères (nexus, tours) en pixels du crop, pour la vérification visuelle."""
        return {k: self.map_to_crop(v) for k, v in LANDMARKS.items()}

    def structures_crop(self) -> list[tuple[float, float]]:
        """Positions des structures (tours, inhibiteurs, nexus) en pixels du crop."""
        return [self.map_to_crop(p) for p in STRUCTURES]

    def spawn_crop(self, team: str) -> tuple[float, float]:
        """Point d'apparition d'une équipe (ORDER ou CHAOS) en pixels du crop."""
        return self.map_to_crop(SPAWN[team])
