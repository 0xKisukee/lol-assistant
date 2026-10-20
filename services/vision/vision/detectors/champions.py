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
   fois, deux détections jamais à moins de `nms_factor` diamètres (petit,
   pour laisser passer les icônes qui se chevauchent : seule une icône
   revendiquée deux fois au même endroit est éliminée) ;
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
from vision.matching import SIGMA_MIN, FrameSpectrum, NCCBank
from vision.models import Calibration, Detection, Pos, Team

log = logging.getLogger(__name__)

RingColor = Literal["blue", "red"] | None
STATIC_RECHECK = 20  # une frame sur N, les pics fixes confirmés ne sont pas effacés (ré-observation)
STATIC_FORGET = 60  # frames sans ré-observation avant d'oublier un pic fixe confirmé


@dataclass
class DetectorConfig:
    threshold: float = 0.55  # score minimal pour une détection « neuve »
    low_threshold: float = 0.42  # score accepté près de la dernière position connue
    scales: tuple[float, ...] = (0.9, 1.0, 1.1)  # échelles autour du diamètre calibré
    peaks_per_champion: int = 6  # les structures fixes de la carte peuvent occuper plusieurs pics
    nms_factor: float = 0.35  # distance minimale entre deux détections, en diamètres (icônes superposées autorisées)
    ring_bonus: float = 0.08
    ring_penalty: float = 0.15
    ring_required_below: float = 0.85  # sous ce score, une détection doit avoir un anneau visible
    ring_mode: Literal["ally", "side"] = "ally"  # bleu = allié (défaut) ou bleu = ORDER
    track_radius_factor: float = 1.5  # rayon de continuité, en diamètres
    track_max_age: int = 6  # frames sans détection avant d'oublier une piste
    max_ringless_frames: int = 3  # frames consécutives où un champion suivi peut être accepté sans anneau
    use_ring: bool = True
    engine: Literal["fft", "opencv"] = "fft"  # "opencv" = chemin masqué natif, lent, pour vérification
    scale_probe_frames: int = 30  # frames pendant lesquelles toutes les échelles sont essayées, puis verrouillage
    recover_overlaps: bool = True  # seconde passe : champions manquants cachés sous une icône acceptée
    overlap_threshold: float = 0.5  # NCC minimale sur le croissant visible (l'anneau attendu est exigé en plus)
    overlap_min_visible: float = 0.3  # fraction minimale du disque visible
    overlap_budget_ms: float = 120.0  # temps maximal de la seconde passe par frame
    recover_weak_below: float = 0.66  # une détection acceptée sous ce score peut être remplacée par la seconde passe
    ring_only_confidence: float = 0.65  # confiance d'une icône déduite de son seul anneau (un seul allié manquant)
    erase_white_lines: bool = True  # effacer les traits blancs fins (ordre de déplacement, rectangle caméra)
    static_frames: int = 8  # un pic sans anneau présent au même endroit depuis autant de frames est un élément fixe de la carte (0 = désactivé)


@dataclass
class Track:
    label: str
    px: float
    py: float
    t: float
    age: int = 0
    hits: int = 0
    ringless: int = 0  # acceptations consécutives sans anneau visible (borne les fantômes)


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
    recovered: bool = False  # trouvé par la seconde passe (croissant visible)
    static: bool = False  # pic sans anneau vu au même endroit depuis plusieurs frames (structure fixe)


def ring_masks(patch: np.ndarray, thin: int = 8, raw: bool = False):
    """Masques des pixels bleus et rouges **fins** d'un patch BGR (anneaux, pas surfaces).

    Un anneau de champion fait 2 à 3 px d'épaisseur ; les icônes de tours, d'inhibiteurs
    et le sol de la base sont des surfaces pleines de la même couleur. Une ouverture
    morphologique de `thin` px efface les traits fins : ce qui disparaît est fin.
    """
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    vivid = (sat > 70) & (val > 60)
    blue = (vivid & (hue >= 95) & (hue <= 130)).astype(np.uint8)
    red = (vivid & ((hue <= 12) | (hue >= 165))).astype(np.uint8)
    kernel = np.ones((thin, thin), np.uint8)
    blue_thin = cv2.subtract(blue, cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel))
    red_thin = cv2.subtract(red, cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel))
    if raw:
        return blue_thin.astype(bool), red_thin.astype(bool), blue.astype(bool), red.astype(bool)
    return blue_thin.astype(bool), red_thin.astype(bool)


def ring_color(crop: np.ndarray, cx: float, cy: float, diameter: int, sectors: int = 12,
               min_sectors: int = 8, occluders: list[tuple[float, float, float]] | None = None) -> RingColor:
    """Couleur dominante de l'anneau autour d'une icône : 'blue', 'red' ou None.

    On échantillonne une couronne centrée sur le bord du disque (±14 % du
    diamètre), découpée en secteurs angulaires : la couleur doit être présente dans au
    moins `min_sectors` secteurs. Une ligne de la carte (rivière, lane) qui
    traverse la couronne n'en colore que deux ou trois, un vrai anneau les
    colore tous ; 8 sur 12 tolère une icône voisine qui en recouvre un tiers. Teintes OpenCV (0..179) : bleu ≈ 95..130, rouge ≈ 0..12 et 165..179.
    """
    h, w = crop.shape[:2]
    # Couronne à cheval sur le bord du disque : l'anneau du jeu est peint sur le bord
    # extérieur de l'icône, que `diameter` l'inclue (calibration à l'œil) ou non.
    half = max(2.0, 0.14 * diameter)
    r_in = diameter / 2.0 - half
    r_out = diameter / 2.0 + half
    x0, x1 = int(max(0, cx - r_out - 1)), int(min(w, cx + r_out + 2))
    y0, y1 = int(max(0, cy - r_out - 1)), int(min(h, cy + r_out + 2))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    patch = crop[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx, dy = xx - cx, yy - cy
    dist = np.sqrt(dx ** 2 + dy ** 2)
    ring_full = (dist >= r_in) & (dist <= r_out)
    ring = ring_full.copy()
    for ox, oy, orad in occluders or ():  # pixels sous une icône voisine : ni pour ni contre
        ring &= ((xx - ox) ** 2 + (yy - oy) ** 2) > orad * orad
    if ring.sum() < 8:
        return None
    is_blue, is_red = ring_masks(patch)
    sector = ((np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * sectors).astype(int) % sectors
    blue_sectors = red_sectors = visible = 0
    for k in range(sectors):
        sel = ring & (sector == k)
        n = int(sel.sum())
        # Secteur hors image (icône au bord) ou en majorité sous un occulteur : ni pour ni contre.
        if n == 0 or n < 0.5 * int((ring_full & (sector == k)).sum()):
            continue
        visible += 1
        if is_blue[sel].mean() > 0.2:
            blue_sectors += 1
        if is_red[sel].mean() > 0.2:
            red_sectors += 1
    # Icône au bord ou partiellement cachée : on abaisse le quorum au nombre de secteurs visibles.
    quorum = min(min_sectors, max(3, visible - 1))
    if blue_sectors >= quorum and blue_sectors > red_sectors:
        return "blue"
    if red_sectors >= quorum and red_sectors > blue_sectors:
        return "red"
    return None


def erase_white_lines(crop: np.ndarray, thickness: int = 5) -> np.ndarray:
    """Efface les traits blancs fins (ligne d'ordre de déplacement, rectangle de la caméra).

    Ces traits de 1 à 3 px passent sur les icônes, surtout celle du joueur local,
    et cassent la corrélation. On repère les pixels quasi blancs qui appartiennent
    à des structures fines (disparues après une ouverture morphologique) et on
    les reconstruit depuis leurs voisins (inpainting). Les surfaces blanches
    plus larges (chiffres des timers, icônes) sont laissées telles quelles.
    """
    white = cv2.inRange(crop, (225, 225, 225), (255, 255, 255))
    if not white.any():
        return crop
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))
    thick = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)
    lines = cv2.subtract(white, thick)
    if cv2.countNonZero(lines) < 4:
        return crop
    lines = cv2.dilate(lines, np.ones((2, 2), np.uint8))  # bords anti-aliasés des traits
    return cv2.inpaint(crop, lines, 2, cv2.INPAINT_TELEA)


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
        self._frames = 0
        self._scale_scores: dict[int, list[float]] = {b.diameter: [] for b in self.banks}
        self.locked_diameter: int | None = None
        self.last_candidates: list[Candidate] = []  # première passe de la dernière frame (diagnostic)
        self.last_accepted: list[Candidate] = []  # acceptés de la dernière frame, seconde passe comprise
        # Pics persistants par label : [x, y, frames consécutives, frames sans anneau].
        self._static: dict[str, list[list[float]]] = {}

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
                 diameters=[b.diameter for b in self.banks], lockedDiameter=self.locked_diameter)
        return d

    def reset(self) -> None:
        self.tracks.clear()
        self._static.clear()

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

    def _peaks(self, res: np.ndarray, diameter: int, label: str | None = None) -> list[tuple[float, float, float]]:
        """Pics (score, cx, cy) d'une carte de corrélation, avec suppression locale.

        Les éléments fixes de la carte confirmés pour ce label sont effacés d'abord (sauf
        une frame sur `STATIC_RECHECK`, pour les ré-observer), sinon ils occupent les
        quelques pics gardés et masquent une vraie icône partiellement recouverte.
        """
        res = res.copy()
        peaks: list[tuple[float, float, float]] = []
        half = diameter / 2.0
        suppress = max(2, int(round(diameter * 0.6)))
        if label is not None and self.config.static_frames and self._frames % STATIC_RECHECK != 0:
            # Un pic fixe est localisé à 3 px près : on n'efface qu'un petit voisinage, pour ne
            # pas emporter une vraie icône qui passerait à côté (champion qui farme sous sa tour).
            r = max(2, int(round(diameter * 0.2)))
            for e in self._static.get(label, []):
                if e[4]:  # confirmé
                    x, y = int(round(e[0] - half)), int(round(e[1] - half))
                    y0, y1 = max(0, y - r), min(res.shape[0], y + r + 1)
                    x0, x1 = max(0, x - r), min(res.shape[1], x + r + 1)
                    res[y0:y1, x0:x1] = -1.0
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

    def candidates(self, crop: np.ndarray, exclude: set[str] | None = None) -> list[Candidate]:
        cfg = self.config
        out: list[Candidate] = []
        for diameter, maps in self._responses(crop):
            for label, res in maps.items():
                if exclude and label in exclude:
                    continue
                for score, cx, cy in self._peaks(res, diameter, label):
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
        self._update_static(out)
        return out

    def _update_static(self, cands: list[Candidate]) -> None:
        """Suit, par label, les pics qui restent au même endroit (à 3 px près) frame après frame.

        Désactivé si `static_frames` vaut 0.

        Une tour ou une structure de la carte produit le même pic frame après frame, le
        plus souvent sans anneau (parfois « bleu » à cause d'une tour alliée voisine) ; un
        champion a un anneau, et bouge. Au-delà de `static_frames` frames consécutives dont
        au moins la moitié sans anneau, le pic est **confirmé** fixe : ignoré à l'acceptation
        tant qu'il n'a pas l'anneau attendu, et effacé de la carte de corrélation avant
        l'extraction des pics. Un pic confirmé qu'on ne revoit plus est oublié après
        `STATIC_FORGET` frames (tour détruite, par exemple).
        Entrée : [x, y, frames vues, frames sans anneau, confirmé, dernière frame vue].
        """
        if not self.config.static_frames:
            return
        tol2 = 3.0 ** 2
        frame = self._frames
        kept: dict[str, list[list]] = {}
        updated: set[int] = set()  # une entrée n'est mise à jour qu'une fois par frame (plusieurs échelles)
        for c in sorted(cands, key=lambda c: c.score, reverse=True):
            entries = self._static.get(c.label, [])
            hit = None
            for e in entries:
                if (c.cx - e[0]) ** 2 + (c.cy - e[1]) ** 2 <= tol2:
                    hit = e
                    break
            if hit is None:
                hit = [c.cx, c.cy, 0.0, 0.0, False, frame]
                entries.append(hit)
                self._static[c.label] = entries
            if id(hit) in updated:
                c.static = bool(hit[4])
                continue
            updated.add(id(hit))
            if hit[4] and c.ring is not None and c.ring == self.expected_ring(c.label):
                # Un « pic fixe » qui porte l'anneau attendu est un champion immobile : on le libère.
                hit[2], hit[3], hit[4] = 0.0, 0.0, False
            hit[0] = 0.7 * hit[0] + 0.3 * c.cx
            hit[1] = 0.7 * hit[1] + 0.3 * c.cy
            hit[2] += 1
            hit[3] += 1 if c.ring is None else 0
            hit[5] = frame
            if hit[2] >= self.config.static_frames and hit[3] >= 0.5 * hit[2]:
                hit[4] = True
            c.static = bool(hit[4])
        for label, entries in self._static.items():
            alive = [e for e in entries if e[5] == frame or (e[4] and frame - e[5] <= STATIC_FORGET)]
            if alive:
                kept[label] = alive
        self._static = kept

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
            # Élément fixe de la carte (tour, structure) qui ressemble au portrait : ignoré tant
            # qu'il ne porte pas l'anneau attendu.
            if cfg.use_ring and c.static and (c.ring is None or c.ring != self.expected_ring(c.label)):
                continue
            # Sans anneau visible, seul un score très élevé est accepté (le fond de carte n'a
            # pas d'anneau ; une icône en a toujours un). Un champion déjà suivi (continuité)
            # peut avoir son anneau caché par une icône voisine : on lui demande alors le seuil
            # « neuf » au lieu de l'anneau, mais pas plus de `max_ringless_frames` frames de
            # suite, sinon une texture du fond entretiendrait un fantôme indéfiniment.
            if cfg.use_ring and c.ring is None:
                track = self.tracks.get(c.label)
                if c.continuity and track is not None and track.ringless < cfg.max_ringless_frames:
                    if c.adjusted < cfg.threshold:
                        continue
                elif c.adjusted < cfg.ring_required_below:
                    continue
            if any((c.cx - a.cx) ** 2 + (c.cy - a.cy) ** 2 < min_dist * min_dist for a in accepted):
                continue
            accepted.append(c)
            taken.add(c.label)
        return accepted

    def _probe_scales(self, accepted: list[Candidate]) -> None:
        """Pendant les premières frames, note les scores bruts par diamètre, puis garde le meilleur banc."""
        if self.locked_diameter is not None or len(self.banks) <= 1:
            return
        for c in accepted:
            self._scale_scores[c.diameter].append(c.score)
        if self._frames < self.config.scale_probe_frames:
            return
        best = max(self.banks, key=lambda b: (sum(self._scale_scores[b.diameter]), len(self._scale_scores[b.diameter])))
        if not self._scale_scores[best.diameter]:
            self.config.scale_probe_frames = self._frames + 10  # rien détecté encore : on continue à sonder
            return
        self.locked_diameter = best.diameter
        self.banks = [best]
        self.base_diameter = best.diameter  # anneau, NMS, continuité et seconde passe suivent l'échelle réelle
        log.info("Échelle verrouillée : icônes de %d px (scores cumulés %s)", best.diameter,
                 {d: round(sum(v), 1) for d, v in self._scale_scores.items()})


    # ---- icônes recouvertes --------------------------------------------------

    def _neighbor_ring_signal(self, crop: np.ndarray, c: Candidate, sectors: int = 12) -> dict[str, list[float]]:
        """Couleurs d'anneau (bleu/rouge) qui dépassent autour de l'icône, avec leurs directions.

        Un croissant d'anneau d'une icône cachée dessous dépasse forcément du côté où se
        trouve son centre ; sa couleur dit quelle équipe chercher, sa direction où chercher.
        Un point de sbire isolé ne colore qu'un secteur : il faut au moins deux secteurs.
        Renvoie {couleur: [angles (rad) des secteurs colorés]}.
        """
        d = c.diameter
        h, w = crop.shape[:2]
        r_in, r_out = 0.58 * d, 1.1 * d  # dès le bord extérieur du propre anneau de l'icône
        x0, x1 = int(max(0, c.cx - r_out)), int(min(w, c.cx + r_out + 1))
        y0, y1 = int(max(0, c.cy - r_out)), int(min(h, c.cy + r_out + 1))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return {}
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx, dy = xx - c.cx, yy - c.cy
        dist = np.sqrt(dx ** 2 + dy ** 2)
        band = (dist >= r_in) & (dist <= r_out)
        blue_thin, red_thin = ring_masks(crop[y0:y1, x0:x1])
        colors = {"blue": blue_thin, "red": red_thin}
        sector = ((np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * sectors).astype(int) % sectors
        out: dict[str, list[float]] = {}
        for name, mask in colors.items():
            angles = []
            for k in range(sectors):
                sel = band & (sector == k)
                if sel.sum() >= 4 and float(mask[sel].mean()) > 0.08:
                    angles.append((k + 0.5) / sectors * 2 * np.pi - np.pi)
            if len(angles) >= 2:
                out[name] = angles
        return out

    def _fit_ring_center(self, crop: np.ndarray, a: Candidate, color: str, d: int) -> tuple[float, float] | None:
        """Centre d'un anneau de couleur `color` qui dépasse autour de l'icône `a`, par ajustement de cercle.

        Pixels vivement colorés à moins de 1,4 diamètre du centre de `a`, hors de son disque ;
        cercle de rayon connu (d/2) ajusté par Gauss-Newton avec élimination des points
        aberrants (anneau d'une autre icône, points de sbires). None si l'ajustement est mauvais.
        """
        h, w = crop.shape[:2]
        R = 1.4 * d
        x0, x1 = int(max(0, a.cx - R)), int(min(w, a.cx + R + 1))
        y0, y1 = int(max(0, a.cy - R)), int(min(h, a.cy + R + 1))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        blue_thin, red_thin, blue_raw, red_raw = ring_masks(crop[y0:y1, x0:x1], raw=True)
        colored = blue_thin if color == "blue" else red_thin
        surface = blue_raw if color == "blue" else red_raw
        yy, xx = np.mgrid[y0:y1, x0:x1]
        outside = ((xx - a.cx) ** 2 + (yy - a.cy) ** 2) > (a.diameter / 2 + 1) ** 2
        near = ((xx - a.cx) ** 2 + (yy - a.cy) ** 2) <= R * R
        pts = np.argwhere(colored & outside & near).astype(np.float64)  # (y, x)
        if len(pts) < 10:
            return None
        pts = pts[:, ::-1] + [x0, y0]  # (x, y) dans le crop
        r = d / 2.0
        direction = pts.mean(axis=0) - np.array([a.cx, a.cy])
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return None
        c = np.array([a.cx, a.cy]) + direction / norm * 0.6 * d
        for it in range(10):
            v = pts - c
            dist = np.linalg.norm(v, axis=1)
            dist[dist < 1e-6] = 1e-6
            res = dist - r
            if it >= 3:  # élimination des aberrants une fois l'estimation dégrossie
                keep = np.abs(res) < 0.2 * d
                if keep.sum() < 10:
                    return None
                pts, v, dist, res = pts[keep], v[keep], dist[keep], res[keep]
            J = -v / dist[:, None]
            delta, *_ = np.linalg.lstsq(J, -res, rcond=None)
            c = c + delta
            if np.linalg.norm(delta) < 0.05:
                break
        v = pts - c
        res = np.linalg.norm(v, axis=1) - r
        inliers = np.abs(res) < 0.15 * d
        if inliers.sum() < 14:
            return None
        ang = np.sort(np.arctan2(v[inliers, 1], v[inliers, 0]))
        span = float(np.ptp(ang))
        if span < np.deg2rad(60):
            return None  # trop peu d'arc pour un cercle fiable
        cx, cy = float(c[0]), float(c[1])
        if not (r <= cx <= w - r and r <= cy <= h - r):
            return None
        # Géométrie d'un recouvrement : le centre caché est à 0,15-1,05 diamètre de l'icône du dessus.
        dist_a = ((cx - a.cx) ** 2 + (cy - a.cy) ** 2) ** 0.5
        if not 0.15 * d <= dist_a <= 1.05 * d:
            return None
        # Un anneau est creux : l'intérieur du cercle (hors disque de l'icône du dessus) ne doit pas
        # être rempli de la même couleur (surface : rivière, sol de base, icône de tour).
        inner = (((xx - cx) ** 2 + (yy - cy) ** 2) <= (0.7 * r) ** 2) & outside
        if inner.sum() >= 8 and float(surface[inner].mean()) > 0.3:
            return None
        # … et son bord est bien un trait : juste à l'extérieur du cercle, pas de surface colorée non plus.
        outer = (((xx - cx) ** 2 + (yy - cy) ** 2) >= (1.15 * r) ** 2) & (((xx - cx) ** 2 + (yy - cy) ** 2) <= (1.5 * r) ** 2) & outside
        if outer.sum() >= 8 and float(surface[outer].mean()) > 0.3:
            return None
        return cx, cy

    @staticmethod
    def _too_close(cx: float, cy: float, others: list[Candidate], dist: float, skip: Candidate | None = None) -> bool:
        return any(o is not skip and (o.cx - cx) ** 2 + (o.cy - cy) ** 2 < dist * dist for o in others)

    def _templates_at_base(self) -> dict[str, Template]:
        for b in self.banks:
            if b.diameter == self.base_diameter or len(self.banks) == 1:
                return {t.label: t for t in b.templates}
        return {t.label: t for t in self.banks[0].templates}

    def recover_overlaps(self, crop: np.ndarray, accepted: list[Candidate],
                         exclude: set[str] | None = None) -> list[Candidate]:
        """Cherche les champions non détectés dans le croissant visible autour des icônes acceptées.

        Pour chaque icône acceptée autour de laquelle dépassent des pixels d'anneau d'une
        icône cachée, et pour chaque champion manquant de l'équipe de cette couleur, on
        calcule la NCC masquée (masque du template moins le disque de l'icône du dessus)
        sur une grille de positions à 0,3-1,05 diamètre du centre. L'anneau de la couleur
        attendue doit être visible sur le croissant.
        """
        cfg = self.config
        weak = {a.label: a for a in accepted if a.adjusted < cfg.recover_weak_below}
        taken = {a.label for a in accepted if a.label not in weak}
        missing = [l for l in self.teams if l not in taken and not (exclude and l in exclude)]
        if not missing or not accepted:
            return []
        # Les alliés d'abord : vivants, ils sont forcément sur la minimap, donc cachés s'ils manquent.
        if self.my_team is not None:
            missing.sort(key=lambda l: self.teams.get(l) != self.my_team)
        tpls = self._templates_at_base()
        d = self.base_diameter
        h, w = crop.shape[:2]
        half = d // 2
        found: list[Candidate] = []
        gy, gx = np.mgrid[0:d, 0:d]
        start = time.perf_counter()
        for a in sorted(accepted, key=lambda c: c.adjusted, reverse=True):
            if a.label in weak:
                continue  # une détection faible ne sert pas d'ancre
            if (time.perf_counter() - start) * 1000 > cfg.overlap_budget_ms:
                break
            signal = self._neighbor_ring_signal(crop, a)
            if not signal:
                continue
            labels = [l for l in missing if l not in taken and (not cfg.use_ring or self.expected_ring(l) in signal)]
            if not labels:
                continue
            found_here: set[str] = set()
            # Directions où chercher : celles des secteurs colorés (±50°), toutes couleurs confondues.
            hot = [ang for angs in signal.values() for ang in angs]
            step = max(2, d // 14)  # 2 px jusqu'à 41 px d'icône : un décalage de 2 px coûte déjà cher en NCC
            offs = []
            for dy in range(-int(1.05 * d), int(1.05 * d) + 1, step):
                for dx in range(-int(1.05 * d), int(1.05 * d) + 1, step):
                    r2 = dx * dx + dy * dy
                    if not (0.3 * d) ** 2 <= r2 <= (1.05 * d) ** 2:
                        continue
                    ang = np.arctan2(dy, dx)
                    if min(abs((ang - hz + np.pi) % (2 * np.pi) - np.pi) for hz in hot) > np.deg2rad(50):
                        continue
                    cx, cy = int(round(a.cx + dx)), int(round(a.cy + dy))
                    if half <= cx < w - half and half <= cy < h - half:
                        offs.append((cx, cy))
            if not offs:
                continue
            wins = np.stack([crop[cy - half:cy - half + d, cx - half:cx - half + d] for cx, cy in offs]).astype(np.float32)
            occl = np.stack([((gx + cx - half - a.cx) ** 2 + (gy + cy - half - a.cy) ** 2) <= (a.diameter / 2 + 1) ** 2
                             for cx, cy in offs])
            for label in labels:
                if (time.perf_counter() - start) * 1000 > cfg.overlap_budget_ms:
                    break
                tpl = tpls.get(label)
                if tpl is None or tpl.diameter != d:
                    continue
                base_mask = tpl.mask > 0
                vis = base_mask[None] & ~occl
                n_vis = vis.sum(axis=(1, 2)).astype(np.float32)
                ok = n_vis >= cfg.overlap_min_visible * base_mask.sum()
                if not ok.any():
                    continue
                M = vis[..., None].astype(np.float32)
                T = np.broadcast_to(tpl.image.astype(np.float32)[None], wins.shape)
                n = n_vis * 3
                sw = np.einsum("nijc,nijc->n", wins, M)
                st = np.einsum("nijc,nijc->n", T, M)
                sww = np.einsum("nijc,nijc,nijc->n", wins, wins, M)
                stt = np.einsum("nijc,nijc,nijc->n", T, T, M)
                swt = np.einsum("nijc,nijc,nijc->n", wins, T, M)
                num = swt - sw * st / n
                den = np.sqrt(np.maximum(sww - sw * sw / n, n * SIGMA_MIN ** 2) * np.maximum(stt - st * st / n, 1e-6))
                ncc = np.where(ok, num / den, -1.0)
                i = int(ncc.argmax())
                if ncc[i] < cfg.overlap_threshold:
                    continue
                cx, cy = offs[i]
                if self._too_close(cx, cy, accepted + found, 0.3 * d, skip=a):
                    continue  # déjà une détection à cet endroit (autre champion) : même icône
                expected = self.expected_ring(label)
                ring = ring_color(crop, cx, cy, d, occluders=[(a.cx, a.cy, a.diameter / 2 + 1)]) if cfg.use_ring else None
                if cfg.use_ring and ring != expected:
                    continue  # pas d'anneau de la bonne couleur sur le croissant : ce n'est pas lui
                cand = Candidate(label=label, score=float(ncc[i]), adjusted=float(ncc[i]) + (cfg.ring_bonus if ring else 0.0),
                                 cx=float(cx), cy=float(cy), diameter=d, ring=ring)
                found.append(cand)
                taken.add(label)
                found_here.add(label)
            # Déduction par l'anneau seul : un croissant d'anneau allié dépasse, aucun portrait n'y
            # correspond (icône sombre, trop recouverte), mais un seul allié vivant manque : c'est lui.
            if cfg.use_ring and self.my_team is not None:
                ally_color = "blue" if cfg.ring_mode == "ally" else ("blue" if self.my_team == "ORDER" else "red")
                if ally_color in signal:
                    remaining = [l for l in labels if l not in taken and self.teams.get(l) == self.my_team]
                    if len(remaining) == 1 and not found_here:
                        center = self._fit_ring_center(crop, a, ally_color, d)
                        if center is not None and not self._too_close(center[0], center[1], accepted + found, 0.5 * d, skip=a):
                            label = remaining[0]
                            cand = Candidate(label=label, score=0.0, adjusted=cfg.ring_only_confidence,
                                             cx=center[0], cy=center[1], diameter=d, ring=ally_color)
                            found.append(cand)
                            taken.add(label)
        # Remplacement des détections faibles par ce que la seconde passe a trouvé de mieux.
        out: list[Candidate] = []
        for f in found:
            w_ = weak.get(f.label)
            if w_ is not None:
                if f.adjusted > w_.adjusted:
                    accepted.remove(w_)
                    out.append(f)
            else:
                out.append(f)
        return out

    def _update_tracks(self, accepted: list[Candidate], t: float) -> None:
        seen = {c.label for c in accepted}
        for c in accepted:
            tr = self.tracks.get(c.label)
            if tr is None:
                self.tracks[c.label] = Track(c.label, c.cx, c.cy, t, age=0, hits=1)
                tr = self.tracks[c.label]
            else:
                tr.px, tr.py, tr.t, tr.age, tr.hits = c.cx, c.cy, t, 0, tr.hits + 1
            tr.ringless = tr.ringless + 1 if (self.config.use_ring and c.ring is None) else 0
        for label, tr in list(self.tracks.items()):
            if label not in seen:
                tr.age += 1
                if tr.age > self.config.track_max_age:
                    del self.tracks[label]

    def detect(self, crop: np.ndarray, calib: Calibration, t: float, frame_id: str,
               exclude: set[str] | None = None) -> list[Detection]:
        """`exclude` : champions à ne pas chercher (morts : le jeu ne les dessine pas sur la minimap)."""
        start = time.perf_counter()
        for label in exclude or ():
            self.tracks.pop(label, None)  # un mort réapparaît à la base, pas là où il était
        if self._mapper is None or self._mapper.calib is not calib:
            self._mapper = MinimapMapper(calib)
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        if self.config.erase_white_lines:
            crop = erase_white_lines(crop)
        if (crop.shape[0] < self.base_diameter or crop.shape[1] < self.base_diameter) and not self._warned_small:
            log.warning("Crop minimap (%dx%d) plus petit que les templates (%d px) : calibration à revoir.",
                        crop.shape[1], crop.shape[0], self.base_diameter)
            self._warned_small = True
        self._frames += 1
        self.last_candidates = self.candidates(crop, exclude)
        accepted = self.assign(self.last_candidates)
        if self.config.recover_overlaps:
            recovered = self.recover_overlaps(crop, accepted, exclude)
            for c in recovered:
                c.recovered = True
            accepted += recovered
        self.last_accepted = accepted
        self._update_tracks(accepted, t)
        self._probe_scales(accepted)
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
