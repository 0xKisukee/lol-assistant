"""Détection des dix champions sur le crop de la minimap.

Méthode : template matching (corrélation normalisée masquée, voir
`vision.matching`) des portraits Data Dragon ramenés au diamètre des icônes
de la minimap, à quelques échelles autour de la valeur calibrée. Puis :

1. extraction de plusieurs pics par champion (un portrait peut ressembler à
   un autre, on laisse l'affectation globale trancher) ;
2. confirmation d'équipe par la couleur de l'anneau autour de l'icône (bleu
   pour les alliés, rouge pour les ennemis) : bonus si cohérent, malus sinon ;
   en dessous de `ring_required_below`, une détection sans anneau visible est
   rejetée (le fond de carte n'a pas d'anneau, les icônes en ont toujours un) ;
3. affectation gloutonne par score décroissant : un champion au plus une
   fois, deux détections jamais à moins de `nms_factor` diamètres ;
4. suivi temporel simple : dernière position et âge par champion ; une
   détection proche de la dernière position connue est acceptée avec un
   seuil plus bas (hystérésis), ce qui lisse les clignotements.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from typing import Literal

import cv2
import numpy as np

from vision.assets import Template
from vision.calib import MinimapMapper
from vision.detectors.base import Detector
from vision.matching import FrameSpectrum, NCCBank
from vision.models import Calibration, Detection, Pos, Team

log = logging.getLogger(__name__)

RingColor = Literal["blue", "red"] | None


@dataclass
class DetectorConfig:
    threshold: float = 0.55  # score minimal pour une détection « neuve »
    low_threshold: float = 0.42  # score accepté près de la dernière position connue
    scales: tuple[float, ...] = (0.9, 1.0, 1.1)  # échelles autour du diamètre calibré
    peaks_per_champion: int = 3
    nms_factor: float = 0.8  # distance minimale entre deux détections, en diamètres
    ring_bonus: float = 0.08
    ring_penalty: float = 0.15
    ring_required_below: float = 0.85  # sous ce score, une détection doit avoir un anneau visible
    ring_mode: Literal["ally", "side"] = "ally"  # bleu = allié (défaut) ou bleu = ORDER
    track_radius_factor: float = 1.5  # rayon de continuité, en diamètres
    track_max_age: int = 6  # frames sans détection avant d'oublier une piste
    use_ring: bool = True
    engine: Literal["fft", "opencv"] = "fft"  # "opencv" = chemin masqué natif, lent, pour vérification


@dataclass
class Track:
    label: str
    px: float
    py: float
    t: float
    age: int = 0
    hits: int = 0


@dataclass
class Candidate:
    label: str
    score: float  # score brut (NCC)
    adjusted: float  # score après anneau
    cx: float
    cy: float
    diameter: int
    ring: RingColor = None
    continuity: bool = False


def ring_color(crop: np.ndarray, cx: float, cy: float, diameter: int, sectors: int = 8,
               min_sectors: int = 6) -> RingColor:
    """Couleur dominante de l'anneau autour d'une icône : 'blue', 'red' ou None.

    On échantillonne une couronne juste à l'extérieur du disque du portrait,
    découpée en secteurs angulaires : la couleur doit être présente dans au
    moins `min_sectors` secteurs. Une ligne de la carte (rivière, lane) qui
    traverse la couronne n'en colore que deux ou trois, un vrai anneau les
    colore tous. Teintes OpenCV (0..179) : bleu ≈ 95..130, rouge ≈ 0..12 et 165..179.
    """
    h, w = crop.shape[:2]
    r_in = diameter / 2.0 - 0.5
    r_out = diameter / 2.0 + max(2.0, 0.18 * diameter)
    x0, x1 = int(max(0, cx - r_out - 1)), int(min(w, cx + r_out + 2))
    y0, y1 = int(max(0, cy - r_out - 1)), int(min(h, cy + r_out + 2))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    patch = crop[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx, dy = xx - cx, yy - cy
    dist = np.sqrt(dx ** 2 + dy ** 2)
    ring = (dist >= r_in) & (dist <= r_out)
    if ring.sum() < 8:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    vivid = (sat > 70) & (val > 60)
    is_blue = vivid & (hue >= 95) & (hue <= 130)
    is_red = vivid & ((hue <= 12) | (hue >= 165))
    sector = ((np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * sectors).astype(int) % sectors
    blue_sectors = red_sectors = 0
    for k in range(sectors):
        sel = ring & (sector == k)
        n = int(sel.sum())
        if n == 0:
            continue  # secteur hors image (icône au bord) : ni pour ni contre
        if is_blue[sel].mean() > 0.2:
            blue_sectors += 1
        if is_red[sel].mean() > 0.2:
            red_sectors += 1
    # Icône au bord du crop : on abaisse le quorum au nombre de secteurs visibles.
    visible = sum(1 for k in range(sectors) if (ring & (sector == k)).any())
    quorum = min(min_sectors, max(3, visible - 1))
    if blue_sectors >= quorum and blue_sectors > red_sectors:
        return "blue"
    if red_sectors >= quorum and red_sectors > blue_sectors:
        return "red"
    return None


def _rescale(tpl: Template, scale: float) -> Template:
    d = max(6, int(round(tpl.diameter * scale)))
    if d == tpl.diameter:
        return tpl
    img = cv2.resize(tpl.image, (d, d), interpolation=cv2.INTER_AREA)
    mask = np.zeros((d, d), dtype=np.uint8)
    cv2.circle(mask, (d // 2, d // 2), max(1, int(round(d / 2.0 - 1))), 255, -1)
    return Template(label=tpl.label, image=img, mask=mask, diameter=d)


class ChampionDetector(Detector):
    name = "champions-template"

    def __init__(self, templates: list[Template], teams: dict[str, Team], my_team: Team | None = None,
                 config: DetectorConfig | None = None, me_label: str | None = None):
        self.config = config or DetectorConfig()
        self.teams = teams
        self.my_team = my_team
        self.me_label = me_label
        self.base_diameter = templates[0].diameter if templates else 0
        # Un banc de templates par échelle (diamètres distincts seulement).
        self.banks: list[NCCBank] = []
        seen: set[int] = set()
        for s in self.config.scales:
            scaled = [_rescale(t, s) for t in templates]
            if not scaled or scaled[0].diameter in seen:
                continue
            seen.add(scaled[0].diameter)
            self.banks.append(NCCBank(scaled))
        self.tracks: dict[str, Track] = {}
        self.last_latency_ms = 0.0
        self._mapper: MinimapMapper | None = None
        self._warned_small = False

    # ---- utilitaires ------------------------------------------------------

    def expected_ring(self, label: str) -> RingColor:
        team = self.teams.get(label)
        if team is None:
            return None
        if self.config.ring_mode == "ally" and self.my_team is not None:
            return "blue" if team == self.my_team else "red"
        return "blue" if team == "ORDER" else "red"

    def describe(self) -> dict:
        d = asdict(self.config)
        d.update(name=self.name, baseDiameter=self.base_diameter, myTeam=self.my_team,
                 diameters=[b.diameter for b in self.banks])
        return d

    def reset(self) -> None:
        self.tracks.clear()

    # ---- cœur --------------------------------------------------------------

    def _responses(self, crop: np.ndarray) -> list[tuple[int, dict[str, np.ndarray]]]:
        """Cartes de corrélation par diamètre : [(diamètre, {label: carte})]."""
        if self.config.engine == "opencv":
            out = []
            for bank in self.banks:
                maps = {}
                for tpl in bank.templates:
                    if crop.shape[0] < tpl.diameter or crop.shape[1] < tpl.diameter:
                        continue
                    res = cv2.matchTemplate(crop, tpl.image, cv2.TM_CCOEFF_NORMED, mask=tpl.mask)
                    maps[tpl.label] = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
                out.append((bank.diameter, maps))
            return out
        spec = FrameSpectrum.from_image(crop)
        return [(bank.diameter, bank.match(spec)) for bank in self.banks]

    def _peaks(self, res: np.ndarray, diameter: int) -> list[tuple[float, float, float]]:
        """Pics (score, cx, cy) d'une carte de corrélation, avec suppression locale."""
        res = res.copy()
        peaks: list[tuple[float, float, float]] = []
        half = diameter / 2.0
        suppress = max(2, int(round(diameter * 0.6)))
        for _ in range(self.config.peaks_per_champion):
            _, mx, _, loc = cv2.minMaxLoc(res)
            if mx < self.config.low_threshold:
                break
            x, y = loc
            peaks.append((float(mx), x + half, y + half))
            y0, y1 = max(0, y - suppress), min(res.shape[0], y + suppress + 1)
            x0, x1 = max(0, x - suppress), min(res.shape[1], x + suppress + 1)
            res[y0:y1, x0:x1] = -1.0
        return peaks

    def candidates(self, crop: np.ndarray) -> list[Candidate]:
        cfg = self.config
        out: list[Candidate] = []
        for diameter, maps in self._responses(crop):
            for label, res in maps.items():
                for score, cx, cy in self._peaks(res, diameter):
                    cand = Candidate(label=label, score=score, adjusted=score, cx=cx, cy=cy, diameter=diameter)
                    if cfg.use_ring:
                        cand.ring = ring_color(crop, cx, cy, diameter)
                        expected = self.expected_ring(label)
                        if cand.ring is not None and expected is not None:
                            cand.adjusted += cfg.ring_bonus if cand.ring == expected else -cfg.ring_penalty
                    track = self.tracks.get(label)
                    if track is not None and track.age <= cfg.track_max_age:
                        radius = cfg.track_radius_factor * self.base_diameter
                        if (cx - track.px) ** 2 + (cy - track.py) ** 2 <= radius * radius:
                            cand.continuity = True
                    out.append(cand)
        return out

    def assign(self, cands: list[Candidate]) -> list[Candidate]:
        """Affectation gloutonne : un champion une fois, pas deux détections au même endroit."""
        cfg = self.config
        accepted: list[Candidate] = []
        taken: set[str] = set()
        min_dist = cfg.nms_factor * self.base_diameter
        for c in sorted(cands, key=lambda c: c.adjusted, reverse=True):
            if c.label in taken:
                continue
            thr = cfg.low_threshold if c.continuity else cfg.threshold
            if c.adjusted < thr:
                continue
            # Sans anneau visible, seul un score très élevé est accepté (le fond de carte n'a
            # pas d'anneau ; une icône en a toujours un, sauf peut-être celle du joueur local).
            if cfg.use_ring and c.ring is None and c.adjusted < cfg.ring_required_below and c.label != self.me_label:
                continue
            if any((c.cx - a.cx) ** 2 + (c.cy - a.cy) ** 2 < min_dist * min_dist for a in accepted):
                continue
            accepted.append(c)
            taken.add(c.label)
        return accepted

    def _update_tracks(self, accepted: list[Candidate], t: float) -> None:
        seen = {c.label for c in accepted}
        for c in accepted:
            tr = self.tracks.get(c.label)
            if tr is None:
                self.tracks[c.label] = Track(c.label, c.cx, c.cy, t, age=0, hits=1)
            else:
                tr.px, tr.py, tr.t, tr.age, tr.hits = c.cx, c.cy, t, 0, tr.hits + 1
        for label, tr in list(self.tracks.items()):
            if label not in seen:
                tr.age += 1
                if tr.age > self.config.track_max_age:
                    del self.tracks[label]

    def detect(self, crop: np.ndarray, calib: Calibration, t: float, frame_id: str) -> list[Detection]:
        start = time.perf_counter()
        if self._mapper is None or self._mapper.calib is not calib:
            self._mapper = MinimapMapper(calib)
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        if (crop.shape[0] < self.base_diameter or crop.shape[1] < self.base_diameter) and not self._warned_small:
            log.warning("Crop minimap (%dx%d) plus petit que les templates (%d px) : calibration à revoir.",
                        crop.shape[1], crop.shape[0], self.base_diameter)
            self._warned_small = True
        accepted = self.assign(self.candidates(crop))
        self._update_tracks(accepted, t)
        dets: list[Detection] = []
        for c in accepted:
            dets.append(Detection(
                t=t, frameId=frame_id, kind="champion", label=c.label,
                pos=self._mapper.crop_to_map(c.cx, c.cy),
                confidence=float(min(1.0, max(0.0, c.adjusted))),
                team=self.teams.get(c.label),
                px=Pos(x=round(c.cx, 1), y=round(c.cy, 1)),
            ))
        self.last_latency_ms = (time.perf_counter() - start) * 1000.0
        return dets
