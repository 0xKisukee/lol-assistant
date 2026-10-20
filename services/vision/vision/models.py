"""Modèles de données partagés (pydantic).

Le schéma `Detection` reprend l'interface TypeScript de PLAN.md §5.3 : les
noms de champs sont en camelCase pour que le futur `vision-bridge` puisse
consommer le JSON tel quel.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Team = Literal["ORDER", "CHAOS"]
DetectionKind = Literal[
    "champion", "ping", "ward", "camp", "minionBlob", "hudCooldown", "hudTrinket", "hpBar"
]


class Pos(BaseModel):
    """Position. Normalisée (0..1) pour `pos`, en pixels pour `px`."""

    x: float
    y: float


class Detection(BaseModel):
    """Une détection produite par un détecteur, horodatée en temps de jeu."""

    t: float = Field(description="gameTime (s) estimé à l'instant de la capture")
    frameId: str
    kind: DetectionKind = "champion"
    label: str = Field(description="Nom du champion (id Data Dragon), type de ping, slot HUD…")
    pos: Pos | None = Field(default=None, description="Coordonnées carte normalisées (0..1), y vers le haut")
    value: float | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    # Extensions propres au service (ignorées par le domaine si inutiles).
    team: Team | None = None
    px: Pos | None = Field(default=None, description="Centre en pixels dans le crop minimap (pour la visionneuse)")
    hidden: bool = Field(default=False, description="Déduit, non vu : allié vivant placé sous l'icône la plus proche de sa dernière position")


class Champion(BaseModel):
    """Un des dix champions de la partie, tel que lu dans la Live Client API."""

    id: str = Field(description="Identifiant Data Dragon, ex. 'MonkeyKing'")
    name: str = Field(description="Nom affiché, ex. 'Wukong'")
    team: Team
    summonerName: str = ""
    isMe: bool = False
    skinId: int = Field(default=0, description="skinID de la Live Client API ; l'icône de minimap suit le skin")


class MinimapRect(BaseModel):
    """Rectangle de la minimap dans la capture, en pixels."""

    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h


class Calibration(BaseModel):
    """Calibration d'une résolution de capture : rectangle de la minimap et taille des icônes."""

    width: int = Field(description="Largeur de la capture (pixels)")
    height: int = Field(description="Hauteur de la capture (pixels)")
    minimap: MinimapRect
    iconDiameter: int = Field(description="Diamètre d'un portrait de champion sur la minimap (pixels)")
    iconZoom: float = Field(default=1.0, description="Zoom appliqué au portrait avant découpe circulaire")
    minimapScale: float | None = Field(default=None, description="MinimapScale lu dans game.cfg, informatif")
    flipped: bool = Field(default=False, description="FlipMiniMap : minimap en bas à gauche")
    source: Literal["estimate", "manual"] = Field(default="estimate", description="Origine du rectangle")
    cropMargin: int = Field(default=0, description="Marge ajoutée autour du rectangle dans le crop traité (icônes au bord, fontaine)")

    @property
    def key(self) -> str:
        return f"{self.width}x{self.height}"


class FrameRecord(BaseModel):
    """Une ligne de `detections.jsonl` : une frame et ses détections."""

    frameId: str
    t: float
    wall: float = Field(description="Horodatage mur (time.time()) de la capture")
    file: str = Field(description="Chemin relatif du crop WebP dans le dossier de la partie")
    latencyMs: float = 0.0
    dead: list[str] = Field(default_factory=list, description="Champions morts (absents de la minimap) à cet instant")
    detections: list[Detection] = Field(default_factory=list)


class Manifest(BaseModel):
    """`manifest.json` d'un dossier de partie."""

    gameId: str
    startedAt: str
    patch: str | None = None
    gameMode: str | None = None
    mapName: str | None = None
    resolution: tuple[int, int]
    calibration: Calibration
    champions: list[Champion]
    fps: float
    source: str = "screen"
    detector: dict = Field(default_factory=dict)
    frames: int = 0
    endedAt: str | None = None
