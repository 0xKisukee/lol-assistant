"""Portraits Data Dragon : téléchargement, cache local et préparation en templates.

- version : première entrée de https://ddragon.leagueoflegends.com/api/versions.json ;
- index : cdn/<v>/data/en_US/champion.json (pour passer du nom affiché à l'id) ;
- portraits : cdn/<v>/img/champion/<Id>.png (carrés de 128×128).

Le cache vit dans `data/cache/ddragon/<version>/`. Les templates sont des
disques découpés dans le portrait, redimensionnés au diamètre des icônes de
la minimap (connu par la calibration), avec leur masque circulaire.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import requests

log = logging.getLogger(__name__)

DDRAGON = "https://ddragon.leagueoflegends.com"
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data" / "cache" / "ddragon"


@dataclass
class Template:
    """Un portrait prêt pour `cv2.matchTemplate` : image BGR, masque, diamètre."""

    label: str
    image: np.ndarray  # BGR, carré de côté `diameter`
    mask: np.ndarray  # uint8, 255 dans le disque
    diameter: int


class Assets:
    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE, version: str | None = None, offline: bool = False):
        self.cache_dir = Path(cache_dir)
        self.offline = offline
        self._version = version
        self._index: dict[str, dict] | None = None

    # ---- version et index -------------------------------------------------

    @property
    def version(self) -> str:
        if self._version is None:
            self._version = self._latest_version()
        return self._version

    def _latest_version(self) -> str:
        marker = self.cache_dir / "latest.txt"
        if not self.offline:
            try:
                r = requests.get(f"{DDRAGON}/api/versions.json", timeout=10)
                r.raise_for_status()
                v = r.json()[0]
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                marker.write_text(v)
                return v
            except (requests.RequestException, ValueError, IndexError) as e:
                log.warning("Version Data Dragon injoignable (%s), repli sur le cache.", e)
        if marker.exists():
            return marker.read_text().strip()
        # Dernier recours : un dossier de version déjà présent dans le cache.
        versions = sorted(p.name for p in self.cache_dir.glob("*") if p.is_dir())
        if versions:
            return versions[-1]
        raise RuntimeError("Aucune version Data Dragon disponible (hors ligne et cache vide).")

    @property
    def version_dir(self) -> Path:
        d = self.cache_dir / self.version
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _fetch(self, url: str, dest: Path) -> Path:
        if dest.exists():
            return dest
        if self.offline:
            raise FileNotFoundError(f"{dest} absent du cache et mode hors ligne.")
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return dest

    def champion_index(self) -> dict[str, dict]:
        """Index Data Dragon : id → {id, name, key}."""
        if self._index is None:
            p = self._fetch(f"{DDRAGON}/cdn/{self.version}/data/en_US/champion.json",
                            self.version_dir / "champion.json")
            data = json.loads(p.read_text(encoding="utf-8"))["data"]
            self._index = {k: {"id": v["id"], "name": v["name"], "key": v["key"]} for k, v in data.items()}
        return self._index

    def resolve_id(self, champion_id: str, display_name: str | None = None) -> str:
        """Retrouve l'id Data Dragon d'un champion vu dans la Live Client API.

        `rawChampionName` donne l'identifiant interne (ex. MonkeyKing) qui est
        presque toujours l'id Data Dragon ; on vérifie, sinon on passe par le
        nom affiché, sinon par une comparaison insensible à la casse.
        """
        index = self.champion_index()
        if champion_id in index:
            return champion_id
        if display_name:
            for cid, meta in index.items():
                if meta["name"] == display_name:
                    return cid
        low = champion_id.lower()
        for cid in index:
            if cid.lower() == low:
                return cid
        raise KeyError(f"Champion inconnu de Data Dragon {self.version} : {champion_id!r} / {display_name!r}")

    # ---- portraits --------------------------------------------------------

    def portrait_path(self, champion_id: str) -> Path:
        return self._fetch(f"{DDRAGON}/cdn/{self.version}/img/champion/{champion_id}.png",
                           self.version_dir / "champion" / f"{champion_id}.png")

    def portrait(self, champion_id: str) -> np.ndarray:
        img = cv2.imread(str(self.portrait_path(champion_id)), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Portrait illisible : {champion_id}")
        return img

    def portraits(self, champion_ids: list[str]) -> dict[str, np.ndarray]:
        return {cid: self.portrait(cid) for cid in champion_ids}


def make_template(label: str, portrait: np.ndarray, diameter: int, zoom: float = 1.0) -> Template:
    """Découpe le disque central du portrait et le ramène au diamètre voulu.

    `zoom` > 1 resserre sur le centre du portrait (l'icône de la minimap
    montre surtout le visage) ; 1.0 prend le disque inscrit dans le carré.
    """
    h, w = portrait.shape[:2]
    side = int(round(min(h, w) / max(zoom, 1e-3)))
    side = max(8, min(side, min(h, w)))
    cx, cy = w // 2, h // 2
    x0, y0 = cx - side // 2, cy - side // 2
    square = portrait[y0:y0 + side, x0:x0 + side]
    diameter = max(6, int(diameter))
    resized = cv2.resize(square, (diameter, diameter), interpolation=cv2.INTER_AREA)
    mask = np.zeros((diameter, diameter), dtype=np.uint8)
    r = diameter / 2.0
    # Disque légèrement rétréci pour ne pas inclure l'anneau d'équipe dans le template.
    cv2.circle(mask, (int(r), int(r)), max(1, int(round(r - 1))), 255, -1)
    return Template(label=label, image=resized, mask=mask, diameter=diameter)


def make_templates(portraits: dict[str, np.ndarray], diameter: int, zoom: float = 1.0) -> list[Template]:
    return [make_template(label, img, diameter, zoom) for label, img in portraits.items()]
