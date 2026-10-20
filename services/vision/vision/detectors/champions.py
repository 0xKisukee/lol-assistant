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
UNKNOWN_LABEL = "?"  # icône dont on voit l'anneau dépasser sans reconnaître le portrait
STATIC_RECHECK = 20  # une frame sur N, les pics fixes confirmés ne sont pas effacés (ré-observation)
STATIC_FORGET = 60  # frames sans ré-observation avant d'oublier un pic fixe confirmé
# Pixels bleus (teinte OpenCV 95-130, un anneau allié est autour de 104). La rivière (teinte 96-98,
# saturation ≥ 170) et les sols bleu pâle de la base (teinte 95-100, saturation ≤ 95) sont des
# surfaces collées à l'anneau d'une icône posée dessus : l'ouverture morphologique les fusionne avec
# l'anneau, qui disparaît alors des pixels « fins » (Poppy sur la rivière : score 0,85, « pas
# d'anneau »). On écarte ces surfaces par teinte et saturation : 4 à 8 % des pixels d'anneau.
BLUE_HUE_MIN, BLUE_HUE_MAX = 95, 130
SURFACE_EXCLUDE = True
RING_BAND = "tight"  # couronne de lecture de l'anneau : "tight" (serrée sur l'anneau) ou "wide" (±14 % du diamètre)


def surface_blocks(hue: np.ndarray, sat: np.ndarray) -> np.ndarray:
    """Surfaces de couleur rivière (teinte ≤ 98, saturation ≥ 170) ou sol bleu pâle (saturation ≤ 95) :
    blocs pleins de 5 px au moins (bool). Un pixel d'anneau qui a cette couleur par mélange avec
    le fond reste isolé et n'en fait pas partie."""
    surface = ((hue <= 98) & ((sat >= 170) | (sat <= 95))).astype(np.uint8)
    if not SURFACE_EXCLUDE or surface.ndim != 2 or surface.shape[0] < 5 or surface.shape[1] < 5:
        return np.zeros(surface.shape, bool)
    return cv2.morphologyEx(surface, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0


def blue_mask(hue: np.ndarray, sat: np.ndarray, val: np.ndarray) -> np.ndarray:
    """Pixels bleus vifs comptés pour les anneaux, sans les surfaces rivière et sol pâle (bool)."""
    blue = (sat > 70) & (val > 60) & (hue >= BLUE_HUE_MIN) & (hue <= BLUE_HUE_MAX)
    return blue & ~surface_blocks(hue, sat)


@dataclass
class DetectorConfig:
    threshold: float = 0.55  # score minimal pour une détection « neuve »
    low_threshold: float = 0.42  # score accepté près de la dernière position connue
    scales: tuple[float, ...] = (0.9, 1.0, 1.1)  # échelles autour du diamètre calibré
    peaks_per_champion: int = 10  # les structures et le terrain sombre occupent plusieurs pics avant la vraie icône
    nms_factor: float = 0.35  # distance minimale entre deux détections, en diamètres (icônes superposées autorisées)
    ring_bonus: float = 0.08
    ring_penalty: float = 0.15
    ring_required_below: float = 0.85  # sous ce score, une détection doit avoir un anneau visible
    ring_mode: Literal["ally", "side"] = "ally"  # bleu = allié (défaut) ou bleu = ORDER
    track_radius_factor: float = 1.5  # rayon de continuité, en diamètres
    track_max_age: int = 6  # frames sans détection avant d'oublier une piste
    max_ringless_frames: int = 0  # frames consécutives où un champion suivi peut être accepté sans anneau (0 : jamais, les alliés cachés sont déduits sous leur hôte)
    use_ring: bool = True
    engine: Literal["fft", "opencv"] = "fft"  # "opencv" = chemin masqué natif, lent, pour vérification
    scale_probe_frames: int = 30  # frames pendant lesquelles toutes les échelles sont essayées, puis verrouillage
    recover_overlaps: bool = True  # seconde passe : champions manquants cachés sous une icône acceptée
    overlap_threshold: float = 0.5  # NCC minimale sur le croissant visible (l'anneau attendu est exigé en plus)
    overlap_min_visible: float = 0.3  # fraction minimale du disque visible
    overlap_budget_ms: float = 120.0  # temps maximal de la seconde passe par frame
    recover_weak_below: float = 0.66  # une détection acceptée sous ce score peut être remplacée par la seconde passe
    ring_only_recovery: bool = False  # déduction par l'anneau seul (un seul allié manquant) : net négatif sur la partie de référence
    ring_only_confidence: float = 0.65  # confiance d'une icône déduite de son seul anneau
    spawn_init_seconds: float = 30.0  # avant cet instant de jeu, un allié jamais vu est supposé à sa fontaine
    # Fontaine de mon équipe : l'anneau y est illisible (arc bleu de la plateforme, arc vert de soin),
    # et les champions y restent immobiles au même pixel. Dans ce rayon autour du point d'apparition
    # (en diamètres), un portrait allié reconnu à `spawn_threshold` ou plus est accepté sans anneau et
    # n'est jamais un pic fixe. La fontaine vide ne dépasse pas 0,46 avec les templates alliés
    # (0,59 tous templates) à 60 px du point d'apparition, mesuré sur 750 frames de la partie 3.
    spawn_radius_factor: float = 1.5
    spawn_threshold: float = 0.68  # icônes réelles à la fontaine : 0,70 à 0,84 ; fond synthétique des tests jusqu'à 0,65
    infer_hidden_allies: bool = True  # allié vivant non vu : placé sous l'icône la plus proche de sa dernière position
    hidden_radius_factor: float = 2.0  # distance maximale (en diamètres) entre la dernière position et l'icône hôte
    hidden_radius_growth: float = 0.25  # élargissement par frame non vue (en diamètres), jusqu'à hidden_radius_max
    hidden_radius_max: float = 6.0
    hidden_max_frames: int = 10 ** 9  # un allié vivant est toujours sur la carte : suivi caché sans limite de durée
    hidden_confidence: float = 0.4
    infer_hidden_enemies: bool = True  # ennemi vu récemment, hôte tout près, croissant d'anneau ennemi qui dépasse
    unknown_marker: bool = True  # anneau d'une équipe ajusté sous une icône sans portrait reconnu : « ? » de cette équipe
    unknown_confidence: float = 0.3
    overlap_relative_min: float = 0.3  # croissant confirmé par un cercle : le portrait le plus ressemblant est pris dès ce score…
    overlap_relative_margin: float = 0.1  # … s'il dépasse le deuxième d'au moins cette marge
    overlap_relative_confidence: float = 0.45
    hidden_enemy_max_age: int = 6  # frames depuis la dernière vue (3 s à 2 i/s)
    hidden_enemy_radius_factor: float = 1.5
    erase_white_lines: bool = True  # effacer les traits blancs fins (ordre de déplacement, rectangle caméra)
    static_ring_guard: bool = True  # un pic fixe confirmé n'est pas effacé si l'anneau attendu y est lu (champion immobile)
    static_frames: int = 8  # un pic sans anneau présent au même endroit depuis autant de frames est un élément fixe de la carte (0 = désactivé)
    static_exempt_score: float = 0.85  # au-dessus, jamais pris pour une structure (bot immobile à l'anneau illisible)


@dataclass
class Track:
    label: str
    px: float
    py: float
    t: float
    age: int = 0
    hits: int = 0
    ringless: int = 0  # acceptations consécutives sans anneau visible (borne les fantômes)
    hidden_frames: int = 0  # frames consécutives passées « caché sous une icône »


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
    hidden: bool = False  # déduit : allié vivant non vu, placé sous l'icône hôte
    at_spawn: bool = False  # au point d'apparition de mon équipe (anneau non exigé, jamais fixe)


class FrameMasks:
    """HSV et masques de couleur du crop entier, calculés une fois par frame."""

    def __init__(self, crop: np.ndarray, thin: int = 8):
        self.hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue, sat, val = self.hsv[..., 0], self.hsv[..., 1], self.hsv[..., 2]
        vivid = (sat > 70) & (val > 60)
        blue = blue_mask(hue, sat, val).astype(np.uint8)
        red = (vivid & ((hue <= 12) | (hue >= 165))).astype(np.uint8)
        kernel = np.ones((thin, thin), np.uint8)
        # Points de sbires et pings : disques de 5-7 px, souvent en chaîne qui se touchent. Après une
        # ouverture de 5 px, un point reste une petite tache ronde isolée (boîte ≤ 8 px) alors qu'un
        # anneau, même épais, reste un arc long : on retranche ces taches (sans dilatation, pour ne
        # pas abîmer un anneau voisin).
        dots_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        blue_dots = self._small_blobs(cv2.morphologyEx(blue, cv2.MORPH_OPEN, dots_kernel))
        red_dots = self._small_blobs(cv2.morphologyEx(red, cv2.MORPH_OPEN, dots_kernel))
        self.blue_thin = self._drop_dots(cv2.subtract(cv2.subtract(blue, cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel)), blue_dots))
        self.red_thin = self._drop_dots(cv2.subtract(cv2.subtract(red, cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel)), red_dots))
        # Pour chercher un anneau qui dépasse (croissants, marqueurs) : traits de 4 px au plus. Une
        # chaîne de points de sbires fait 6-7 px d'épaisseur et disparaît ici ; les anneaux épais
        # (halos) ne servent qu'à confirmer une icône déjà trouvée, via ring_color.
        k5 = np.ones((5, 5), np.uint8)
        self.blue_thin5 = self._drop_dots(cv2.subtract(blue, cv2.morphologyEx(blue, cv2.MORPH_OPEN, k5)))
        self.red_thin5 = self._drop_dots(cv2.subtract(red, cv2.morphologyEx(red, cv2.MORPH_OPEN, k5)))
        self.blue_raw, self.red_raw = blue.astype(bool), red.astype(bool)
        self.blue_area = self._drop_dots(blue)  # surfaces colorées (portraits), sans les points de sbires
        self.red_area = self._drop_dots(red)
        # Halo de rappel (cyan vif) : sans les surfaces rivière / sol, sinon une icône posée sur la
        # rivière (V ≈ 150) passerait pour un halo complet.
        self.cyan_blue = (sat > 70) & (val > 120) & (hue >= 85) & (hue <= 130) & ~surface_blocks(hue, sat)

    @staticmethod
    def _small_blobs(mask: np.ndarray, max_box: int = 8) -> np.ndarray:
        """Masque uint8 (0/1) des composantes dont la boîte englobante tient dans max_box (taches rondes)."""
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return np.zeros_like(mask)
        small = (stats[:, cv2.CC_STAT_WIDTH] <= max_box) & (stats[:, cv2.CC_STAT_HEIGHT] <= max_box)
        small[0] = False
        return small[labels].astype(np.uint8)

    @staticmethod
    def _drop_dots(mask: np.ndarray, max_box: int = 7) -> np.ndarray:
        """Retire les petites taches compactes (points de sbires, pings) : un anneau est un trait
        long, un sbire un disque de 5 px. Composantes dont la boîte englobante tient dans max_box."""
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return mask.astype(bool)
        small = (stats[:, cv2.CC_STAT_WIDTH] <= max_box) & (stats[:, cv2.CC_STAT_HEIGHT] <= max_box)
        small[0] = False
        return mask.astype(bool) & ~small[labels]

    def window(self, x0: int, y0: int, x1: int, y1: int):
        return (self.blue_thin[y0:y1, x0:x1], self.red_thin[y0:y1, x0:x1],
                self.blue_raw[y0:y1, x0:x1], self.red_raw[y0:y1, x0:x1])


def ring_masks(patch: np.ndarray, thin: int = 8, raw: bool = False):
    """Masques des pixels bleus et rouges **fins** d'un patch BGR (anneaux, pas surfaces).

    Un anneau de champion fait 2 à 3 px d'épaisseur ; les icônes de tours, d'inhibiteurs
    et le sol de la base sont des surfaces pleines de la même couleur. Une ouverture
    morphologique de `thin` px efface les traits fins : ce qui disparaît est fin.
    """
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    vivid = (sat > 70) & (val > 60)
    blue = blue_mask(hue, sat, val).astype(np.uint8)
    red = (vivid & ((hue <= 12) | (hue >= 165))).astype(np.uint8)
    kernel = np.ones((thin, thin), np.uint8)
    blue_thin = cv2.subtract(blue, cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel))
    red_thin = cv2.subtract(red, cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel))
    if raw:
        return blue_thin.astype(bool), red_thin.astype(bool), blue.astype(bool), red.astype(bool)
    return blue_thin.astype(bool), red_thin.astype(bool)


_RING_GRIDS: dict[tuple[int, int], tuple] = {}


def _ring_grid(diameter: int, sectors: int):
    """Grilles (distance, secteur, masques couronne/halo) centrées, mises en cache par diamètre."""
    key = (diameter, sectors, RING_BAND)
    if key not in _RING_GRIDS:
        if RING_BAND == "tight":
            # Anneau mesuré sur les enregistrements à r = d/2 − 2 … d/2 + 3 px (icônes de 33 et 38 px) :
            # couronne serrée dessus, l'anneau en occupe ~40 % par secteur. Avec ±14 % du diamètre,
            # il n'en occupait que 22 % à 38 px, à la limite du seuil de 0,2 : quelques pixels perdus
            # (rivière, sbire, icône voisine) suffisaient à perdre l'anneau.
            r_in = diameter / 2.0 - max(2.0, 0.07 * diameter)
            r_out = diameter / 2.0 + max(2.5, 0.09 * diameter)
        else:
            half = max(2.0, 0.14 * diameter)
            r_in, r_out = diameter / 2.0 - half, diameter / 2.0 + half
        R = int(np.ceil(r_out)) + 1
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        dist = np.sqrt(xx ** 2 + yy ** 2)
        sector = ((np.arctan2(yy, xx) + np.pi) / (2 * np.pi) * sectors).astype(int) % sectors
        ring = (dist >= r_in) & (dist <= r_out)
        glow = (dist >= diameter / 2.0 - 0.05 * diameter) & (dist <= diameter / 2.0 + 0.3 * diameter)
        _RING_GRIDS[key] = (R, xx, yy, sector, ring, glow)
    return _RING_GRIDS[key]


def ring_color(crop: np.ndarray, cx: float, cy: float, diameter: int, sectors: int = 12,
               min_sectors: int = 8, occluders: list[tuple[float, float, float]] | None = None,
               masks: "FrameMasks | None" = None) -> RingColor:
    """Couleur dominante de l'anneau autour d'une icône : 'blue', 'red' ou None.

    On échantillonne une couronne centrée sur le bord du disque (±14 % du
    diamètre), découpée en secteurs angulaires : la couleur doit être présente
    (en pixels **fins**) dans au moins `min_sectors` secteurs. Une ligne de la
    carte qui traverse la couronne n'en colore que deux ou trois, un vrai anneau
    les colore tous ; 8 sur 12 tolère une icône voisine qui en recouvre un tiers.
    Les secteurs hors image ou sous un occulteur ne comptent ni pour ni contre.
    À défaut, un halo de rappel (anneau épais, vif, quasi complet) est reconnu.
    """
    h, w = crop.shape[:2]
    R, gxx, gyy, gsector, gring, gglow = _ring_grid(diameter, sectors)
    icx, icy = int(round(cx)), int(round(cy))
    x0, x1 = icx - R, icx + R + 1
    y0, y1 = icy - R, icy + R + 1
    # Découpe de la grille aux bords de l'image.
    gx0, gy0 = max(0, -x0), max(0, -y0)
    gx1, gy1 = (2 * R + 1) - max(0, x1 - w), (2 * R + 1) - max(0, y1 - h)
    if gx1 - gx0 < 3 or gy1 - gy0 < 3:
        return None
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    sector = gsector[gy0:gy1, gx0:gx1]
    ring = gring[gy0:gy1, gx0:gx1].copy()
    ring_full = ring.copy()
    if occluders:
        xx, yy = gxx[gy0:gy1, gx0:gx1] + icx, gyy[gy0:gy1, gx0:gx1] + icy
        for ox, oy, orad in occluders:  # pixels sous une icône voisine : ni pour ni contre
            ring &= ((xx - ox) ** 2 + (yy - oy) ** 2) > orad * orad
    if ring.sum() < 8:
        return None
    if masks is not None:
        is_blue, is_red, raw_blue, raw_red = masks.window(x0, y0, x1, y1)
    else:
        is_blue, is_red, raw_blue, raw_red = ring_masks(crop[y0:y1, x0:x1], raw=True)
    # Statistiques par secteur, vectorisées.
    sec = sector[ring]
    n_vis = np.bincount(sec, minlength=sectors)
    n_full = np.bincount(sector[ring_full], minlength=sectors)
    counted = (n_vis > 0) & (n_vis >= 0.5 * n_full)  # secteur visible et pas majoritairement occulté
    blue_frac = np.bincount(sec, weights=is_blue[ring], minlength=sectors) / np.maximum(n_vis, 1)
    red_frac = np.bincount(sec, weights=is_red[ring], minlength=sectors) / np.maximum(n_vis, 1)
    visible = int(counted.sum())
    blue_sectors = int((counted & (blue_frac > 0.2)).sum())
    red_sectors = int((counted & (red_frac > 0.2)).sum())
    quorum = min(min_sectors, max(3, visible - 1))
    if blue_sectors >= quorum and blue_sectors > red_sectors:
        return "blue"
    if red_sectors >= quorum and red_sectors > blue_sectors:
        return "red"
    # Halo de rappel (back) ou de sort : un anneau épais, vivement coloré et presque complet remplace
    # l'anneau fin. Une icône de tour ou le sol ne produisent jamais une couronne pleine sur 10 secteurs.
    glow = gglow[gy0:gy1, gx0:gx1].copy()
    if occluders:
        for ox, oy, orad in occluders:
            glow &= ((xx - ox) ** 2 + (yy - oy) ** 2) > orad * orad
    if glow.sum() < 8:
        return None
    if masks is not None:
        cyan_blue = masks.cyan_blue[y0:y1, x0:x1]
    else:
        hsv = cv2.cvtColor(crop[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        cyan_blue = ((hsv[..., 1] > 70) & (hsv[..., 2] > 120) & (hsv[..., 0] >= 85) & (hsv[..., 0] <= 130)
                     & ~surface_blocks(hsv[..., 0], hsv[..., 1]))
    gsec = sector[glow]
    n_g = np.bincount(gsec, minlength=sectors)
    cb = np.bincount(gsec, weights=cyan_blue[glow], minlength=sectors) / np.maximum(n_g, 1)
    rr = np.bincount(gsec, weights=raw_red[glow], minlength=sectors) / np.maximum(n_g, 1)
    full_blue = int(((n_g >= 4) & (cb > 0.5)).sum())
    full_red = int(((n_g >= 4) & (rr > 0.5)).sum())
    if full_blue >= 10 and full_blue > full_red:
        return "blue"
    if full_red >= 10 and full_red > full_blue:
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
        self._spawned: set[str] = set()  # alliés déjà initialisés à la fontaine (une seule fois)
        self._unknown_team: dict[tuple[int, int], str] = {}  # équipe des marqueurs « ? » de la frame courante
        self._masks: FrameMasks | None = None  # HSV et masques du crop courant
        self._crop: np.ndarray | None = None  # crop de la frame courante (lignes blanches effacées)
        self._last_exclude: set[str] = set()  # morts à la frame précédente (réapparition à la fontaine)
        self._halo_cache: dict[tuple[int, int], float] = {}  # rayon d'exclusion par icône, par frame
        self._last_fit_span = 0.0  # arc (degrés) couvert par le dernier cercle ajusté
        self._last_fit_inliers = 0
        self._structures: list[tuple[float, float]] = []  # icônes de structures (pixels du crop)
        self._spawn: tuple[float, float] | None = None  # point d'apparition de mon équipe (pixels du crop)
        # Couleurs dominantes des portraits (fraction de pixels vivement rouges / bleus) : un
        # portrait rouge (Brand) qui dépasse d'une icône ressemble à un anneau rouge.
        self._portrait_color: dict[str, dict[str, float]] = {}
        for tpl in templates:
            hsv = cv2.cvtColor(tpl.image, cv2.COLOR_BGR2HSV)
            m = tpl.mask > 0
            hue, sat, val = hsv[..., 0][m], hsv[..., 1][m], hsv[..., 2][m]
            vivid = (sat > 70) & (val > 60)
            self._portrait_color[tpl.label] = {
                "red": float((vivid & ((hue <= 12) | (hue >= 165))).mean()) if m.any() else 0.0,
                "blue": float(blue_mask(hue, sat, val).mean()) if m.any() else 0.0,
            }
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
        self._spawned.clear()
        self._last_exclude = set()

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
            expected = self.expected_ring(label)
            for e in self._static.get(label, []):
                if e[4]:  # confirmé
                    # Un champion immobile à cet endroit (anneau attendu lu) n'est pas effacé : sinon il
                    # disparaîtrait jusqu'à la prochaine ré-observation (Yi sous sa tour, 8 frames).
                    if self._crop is not None and self.config.use_ring and self.config.static_ring_guard and expected is not None and \
                            ring_color(self._crop, e[0], e[1], diameter, masks=self._masks) == expected:
                        e[2], e[3], e[4] = 0.0, 0.0, False
                        continue
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
        self._crop = crop
        for diameter, maps in self._responses(crop):
            for label, res in maps.items():
                if exclude and label in exclude:
                    continue
                for score, cx, cy in self._peaks(res, diameter, label):
                    cand = Candidate(label=label, score=score, adjusted=score, cx=cx, cy=cy, diameter=diameter)
                    if self._spawn is not None and self.teams.get(label) == self.my_team:
                        r_s = cfg.spawn_radius_factor * self.base_diameter
                        cand.at_spawn = (cx - self._spawn[0]) ** 2 + (cy - self._spawn[1]) ** 2 <= r_s * r_s
                    if cfg.use_ring:
                        cand.ring = ring_color(crop, cx, cy, diameter, masks=self._masks)
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
        cfg_static_exempt = self.config.static_exempt_score
        tol2 = 3.0 ** 2
        frame = self._frames
        kept: dict[str, list[list]] = {}
        updated: set[int] = set()  # une entrée n'est mise à jour qu'une fois par frame (plusieurs échelles)
        for c in sorted(cands, key=lambda c: c.score, reverse=True):
            if c.score >= cfg_static_exempt or c.at_spawn:
                continue  # une structure ne ressemble jamais autant à un portrait : c'est une icône immobile
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
            if c.at_spawn and c.score >= cfg.spawn_threshold:
                # Au point d'apparition, l'anneau est mangé par la plateforme : le portrait suffit.
                if not any((c.cx - a.cx) ** 2 + (c.cy - a.cy) ** 2 < min_dist * min_dist for a in accepted):
                    accepted.append(c)
                    taken.add(c.label)
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

    def _neighbor_ring_signal(self, crop: np.ndarray, c: Candidate, sectors: int = 12,
                              others: list[Candidate] | None = None) -> dict[str, list[float]]:
        """Couleurs d'anneau (bleu/rouge) qui dépassent autour de l'icône, avec leurs directions.

        Un croissant d'anneau d'une icône cachée dessous dépasse forcément du côté où se
        trouve son centre ; sa couleur dit quelle équipe chercher, sa direction où chercher.
        Un point de sbire isolé ne colore qu'un secteur : il faut au moins deux secteurs.
        Renvoie {couleur: [angles (rad) des secteurs colorés]}.
        """
        d = c.diameter
        h, w = crop.shape[:2]
        r_in, r_out = max(0.64 * d, self._exclusion_radius(c, d)), 1.1 * d  # au-delà de l'anneau, ou du halo
        x0, x1 = int(max(0, c.cx - r_out)), int(min(w, c.cx + r_out + 1))
        y0, y1 = int(max(0, c.cy - r_out)), int(min(h, c.cy + r_out + 1))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return {}
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx, dy = xx - c.cx, yy - c.cy
        dist = np.sqrt(dx ** 2 + dy ** 2)
        band = (dist >= r_in) & (dist <= r_out)
        # Les disques des autres icônes détectées ne comptent pas : leur portrait peut contenir
        # du rouge ou du bleu (la lueur de Sion), et une icône cachée n'est sous aucune d'elles.
        for o in others or ():
            if o is not c and not o.hidden:
                band &= ((xx - o.cx) ** 2 + (yy - o.cy) ** 2) > self._exclusion_radius(o, d) ** 2
        for sx, sy in self._structures:  # icônes de tours et de nexus : jamais des anneaux
            if abs(sx - c.cx) < 1.6 * d and abs(sy - c.cy) < 1.6 * d:
                band &= ((xx - sx) ** 2 + (yy - sy) ** 2) > (0.5 * d) ** 2
        if self._masks is not None:
            blue_thin = self._masks.blue_thin5[y0:y1, x0:x1]
            red_thin = self._masks.red_thin5[y0:y1, x0:x1]
        else:
            blue_thin, red_thin = ring_masks(crop[y0:y1, x0:x1], thin=5)
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

    def _fit_ring_center(self, crop: np.ndarray, a: Candidate, color: str, d: int,
                         min_span_deg: float = 60.0, others: list[Candidate] | None = None) -> tuple[float, float] | None:
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
        if self._masks is not None:
            _, _, blue_raw, red_raw = self._masks.window(x0, y0, x1, y1)
            blue_thin = self._masks.blue_thin5[y0:y1, x0:x1]
            red_thin = self._masks.red_thin5[y0:y1, x0:x1]
        else:
            blue_thin, red_thin, blue_raw, red_raw = ring_masks(crop[y0:y1, x0:x1], thin=5, raw=True)
        colored = blue_thin if color == "blue" else red_thin
        surface = blue_raw if color == "blue" else red_raw
        yy, xx = np.mgrid[y0:y1, x0:x1]
        # Hors de l'icône du dessus, de son anneau anti-aliasé et d'un éventuel halo (jusqu'à 0,8 d),
        # et de même pour les autres icônes vues.
        outside = ((xx - a.cx) ** 2 + (yy - a.cy) ** 2) > self._exclusion_radius(a, d) ** 2
        for o in others or ():
            if o is not a and not o.hidden:
                outside &= ((xx - o.cx) ** 2 + (yy - o.cy) ** 2) > self._exclusion_radius(o, d) ** 2
        for sx, sy in self._structures:
            if abs(sx - a.cx) < 2 * d and abs(sy - a.cy) < 2 * d:
                outside &= ((xx - sx) ** 2 + (yy - sy) ** 2) > (0.5 * d) ** 2
        near = ((xx - a.cx) ** 2 + (yy - a.cy) ** 2) <= R * R
        pts = np.argwhere(colored & outside & near).astype(np.float64)  # (y, x)
        if len(pts) < 10:
            return None
        pts = pts[:, ::-1] + [x0, y0]  # (x, y) dans le crop
        r = d / 2.0
        # Ajustement par Gauss-Newton, à rayon connu, avec élimination des aberrants. Si le cercle
        # obtenu est le propre anneau de l'icône du dessus (centre trop proche), on retire ses
        # points et on recommence : l'anneau caché est ce qui reste.
        for attempt in range(3):
            if len(pts) < 10:
                return None
            direction = pts.mean(axis=0) - np.array([a.cx, a.cy])
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                return None
            c = np.array([a.cx, a.cy]) + direction / norm * 0.6 * d
            work = pts
            for it in range(10):
                v = work - c
                dist = np.linalg.norm(v, axis=1)
                dist[dist < 1e-6] = 1e-6
                res = dist - r
                if it >= 3:  # élimination des aberrants une fois l'estimation dégrossie
                    keep = np.abs(res) < 0.2 * d
                    if keep.sum() < 10:
                        return None
                    work, v, dist, res = work[keep], v[keep], dist[keep], res[keep]
                J = -v / dist[:, None]
                delta, *_ = np.linalg.lstsq(J, -res, rcond=None)
                c = c + delta
                if np.linalg.norm(delta) < 0.05:
                    break
            v_all = pts - c
            res_all = np.linalg.norm(v_all, axis=1) - r
            inliers = np.abs(res_all) < 0.15 * d
            if inliers.sum() < 14:
                return None
            if np.hypot(c[0] - a.cx, c[1] - a.cy) < 0.3 * d:
                pts = pts[~inliers]  # c'était l'anneau de l'icône du dessus : on l'écarte et on réessaie
                continue
            break
        else:
            return None
        v = pts - c
        res = np.linalg.norm(v, axis=1) - r
        inliers = np.abs(res) < 0.15 * d
        if inliers.sum() < 14:
            return None
        ang = np.sort(np.arctan2(v[inliers, 1], v[inliers, 0]))
        span = float(np.ptp(ang))
        if span < np.deg2rad(min_span_deg):
            return None  # trop peu d'arc pour un cercle fiable
        cx, cy = float(c[0]), float(c[1])
        self._last_fit_span = float(np.degrees(span))
        self._last_fit_inliers = int(inliers.sum())
        if not (r <= cx <= w - r and r <= cy <= h - r):
            return None
        # Géométrie d'un recouvrement : le centre caché est à 0,3-1,05 diamètre de l'icône du dessus.
        dist_a = ((cx - a.cx) ** 2 + (cy - a.cy) ** 2) ** 0.5
        if not 0.3 * d <= dist_a <= 1.05 * d:
            return None
        # Un anneau est creux : juste sous l'anneau (bande 0,55-0,85 r, hors disque de l'icône du
        # dessus) il y a du portrait, pas la même couleur (sinon c'est une surface : rivière, sol de
        # base, icône de tour, ou le croissant plein d'un portrait rouge comme Brand). Si cette bande
        # n'est pas assez visible pour trancher, pas de cercle.
        rr = (xx - cx) ** 2 + (yy - cy) ** 2
        inner = (rr >= (0.4 * r) ** 2) & (rr <= (0.75 * r) ** 2) & outside  # loin de l'anneau (centre imprécis de 2-3 px)
        if inner.sum() < 8 or float(surface[inner].mean()) > 0.3:
            return None
        # … et son bord est bien un trait : juste à l'extérieur du cercle, pas de surface colorée non plus.
        outer = (((xx - cx) ** 2 + (yy - cy) ** 2) >= (1.15 * r) ** 2) & (((xx - cx) ** 2 + (yy - cy) ** 2) <= (1.5 * r) ** 2) & outside
        if outer.sum() >= 8 and float(surface[outer].mean()) > 0.3:
            return None
        return cx, cy

    @staticmethod
    def _too_close(cx: float, cy: float, others: list[Candidate], dist: float, skip: Candidate | None = None) -> bool:
        return any(o is not skip and (o.cx - cx) ** 2 + (o.cy - cy) ** 2 < dist * dist for o in others)

    def _exclusion_radius(self, c: Candidate, d: int) -> float:
        """Rayon à exclure autour d'une icône vue : jusqu'au bord de son halo s'il y en a un
        (couronne 0,55-0,8 d majoritairement colorée), sinon juste au-delà de son anneau."""
        if self._masks is None:
            return 0.62 * d
        key = (round(c.cx), round(c.cy))
        if key in self._halo_cache:
            return self._halo_cache[key]
        h, w = self._masks.red_raw.shape
        R = int(0.8 * d) + 1
        x0, x1 = max(0, int(c.cx) - R), min(w, int(c.cx) + R + 1)
        y0, y1 = max(0, int(c.cy) - R), min(h, int(c.cy) + R + 1)
        radius = 0.62 * d
        if x1 - x0 > 3 and y1 - y0 > 3:
            yy, xx = np.mgrid[y0:y1, x0:x1]
            dist2 = (xx - c.cx) ** 2 + (yy - c.cy) ** 2
            ring = (dist2 >= (0.55 * d) ** 2) & (dist2 <= (0.8 * d) ** 2)
            if ring.sum() >= 8:
                # Un halo fait le tour complet de l'icône ; l'anneau d'une voisine ne colore que
                # quelques secteurs. Il faut 10 secteurs sur 12 majoritairement colorés.
                colored = (self._masks.red_raw | self._masks.blue_raw | self._masks.cyan_blue)[y0:y1, x0:x1]
                sector = ((np.arctan2(yy - c.cy, xx - c.cx) + np.pi) / (2 * np.pi) * 12).astype(int) % 12
                sec = sector[ring]
                n_s = np.bincount(sec, minlength=12)
                frac = np.bincount(sec, weights=colored[ring], minlength=12) / np.maximum(n_s, 1)
                if int(((n_s >= 4) & (frac > 0.5)).sum()) >= 10:
                    radius = 0.9 * d
        self._halo_cache[key] = radius
        return radius

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
            signal = self._neighbor_ring_signal(crop, a, others=accepted + found)
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
            # Le masque de visibilité est le même pour tous les champions (même disque) : tout ce qui
            # ne dépend pas du template se calcule une fois par ancre, le reste tient en trois
            # produits matrice-vecteur par champion.
            base_mask = next(iter(tpls.values())).mask > 0
            vis = base_mask[None] & ~occl  # (N, d, d)
            n_vis = vis.sum(axis=(1, 2)).astype(np.float32)
            ok_all = n_vis >= cfg.overlap_min_visible * base_mask.sum()
            if not ok_all.any():
                continue
            N = len(offs)
            Mf = np.repeat(vis[..., None], 3, axis=3).reshape(N, -1).astype(np.float32)  # (N, P)
            Wf = wins.reshape(N, -1)
            WM = Wf * Mf
            n = n_vis * 3
            sw = WM.sum(axis=1)
            sww = (WM * Wf).sum(axis=1)
            var_w = np.maximum(sww - sw * sw / n, n * SIGMA_MIN ** 2)
            best_by_label: dict[str, tuple[float, int]] = {}  # label → (meilleure NCC, index d'offset)
            for label in labels:
                if (time.perf_counter() - start) * 1000 > cfg.overlap_budget_ms:
                    break
                tpl = tpls.get(label)
                if tpl is None or tpl.diameter != d:
                    continue
                T = tpl.image.astype(np.float32).reshape(-1)
                st = Mf @ T
                stt = Mf @ (T * T)
                swt = WM @ T
                num = swt - sw * st / n
                den = np.sqrt(var_w * np.maximum(stt - st * st / n, 1e-6))
                ncc = np.where(ok_all, num / den, -1.0)
                i = int(ncc.argmax())
                best_by_label[label] = (float(ncc[i]), i)
                if ncc[i] < cfg.overlap_threshold:
                    continue
                cx, cy = offs[i]
                if self._too_close(cx, cy, accepted + found, 0.3 * d, skip=a):
                    continue  # déjà une détection à cet endroit (autre champion) : même icône
                expected = self.expected_ring(label)
                ring = (ring_color(crop, cx, cy, d, occluders=[(a.cx, a.cy, a.diameter / 2 + 1)], masks=self._masks)
                        if cfg.use_ring else None)
                if cfg.use_ring and ring != expected:
                    continue  # pas d'anneau de la bonne couleur sur le croissant : ce n'est pas lui
                cand = Candidate(label=label, score=float(ncc[i]), adjusted=float(ncc[i]) + (cfg.ring_bonus if ring else 0.0),
                                 cx=float(cx), cy=float(cy), diameter=d, ring=ring)
                found.append(cand)
                taken.add(label)
                found_here.add(label)
            # Rien de sûr par le portrait : un cercle s'ajuste-t-il sur un anneau qui dépasse ? Si oui, une
            # icône est bien là. Parmi les champions manquants de cette équipe, on prend le portrait le plus
            # ressemblant s'il se détache nettement (on connaît la composition des deux équipes) ; sinon « ? ».
            if cfg.use_ring and self.my_team is not None and not found_here:
                for color, angles in signal.items():
                    team = (self.my_team if color == ("blue" if cfg.ring_mode == "ally" or self.my_team == "ORDER" else "red")
                            else ("CHAOS" if self.my_team == "ORDER" else "ORDER"))
                    if team == self.my_team:
                        continue  # les alliés cachés sont déduits sous leur hôte, mieux que par ressemblance
                    # Un allié non vu dont le portrait est de cette couleur (Brand, rouge) et dont la
                    # piste est dans son rayon de recherche peut expliquer le signal : pas de marqueur.
                    seen_now = {o.label for o in accepted + found if not o.hidden}
                    if any(l not in seen_now and self.teams.get(l) == self.my_team and
                           self._portrait_color.get(l, {}).get(color, 0.0) > 0.12 and
                           (tr.px - a.cx) ** 2 + (tr.py - a.cy) ** 2 <=
                           (min(cfg.hidden_radius_max, cfg.hidden_radius_factor + cfg.hidden_radius_growth * tr.age) * d) ** 2
                           for l, tr in self.tracks.items()):
                        continue
                    if any(o is not a and self.teams.get(o.label) == team and
                           (o.cx - a.cx) ** 2 + (o.cy - a.cy) ** 2 < (1.3 * d) ** 2 for o in accepted + found):
                        continue  # une icône de cette équipe est déjà détectée tout près : c'est son anneau
                    center = self._fit_ring_center(crop, a, color, d, min_span_deg=90.0, others=accepted + found)
                    if center is None:
                        continue
                    if self._too_close(center[0], center[1], accepted + found, 0.3 * d, skip=a):
                        continue
                    team_labels = [l for l in labels if self.teams.get(l) == team and l in best_by_label
                                   and self._hue_consistent(crop, label=l, offset=offs[best_by_label[l][1]], occl=occl[best_by_label[l][1]])]
                    ranked = sorted(((best_by_label[l][0], l) for l in team_labels), reverse=True)
                    if ranked and ranked[0][0] >= cfg.overlap_relative_min and \
                            (len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= cfg.overlap_relative_margin):
                        score, label = ranked[0]
                        cx, cy = offs[best_by_label[label][1]]
                        # Attribution par ressemblance : une déduction, pas une reconnaissance (hidden=True).
                        cand = Candidate(label=label, score=score, adjusted=cfg.overlap_relative_confidence,
                                         cx=float(cx), cy=float(cy), diameter=d, ring=color, hidden=True)
                        found.append(cand)
                        taken.add(label)
                        found_here.add(label)
                    elif cfg.unknown_marker and self._last_fit_span >= 150.0 and self._last_fit_inliers >= 25 \
                            and not self._too_close(center[0], center[1], accepted + found, 0.6 * d):
                        # Sans portrait reconnu, il faut un arc d'anneau franc (≥ 150°) : une chaîne de
                        # points de sbires ou un bord de rivière n'en font pas autant.
                        found.append(Candidate(label=UNKNOWN_LABEL, score=0.0, adjusted=cfg.unknown_confidence,
                                               cx=center[0], cy=center[1], diameter=d, ring=color, hidden=True))
                        self._unknown_team[(round(center[0]), round(center[1]))] = team
            # Déduction par l'anneau seul : un croissant d'anneau allié dépasse, aucun portrait n'y
            # correspond (icône sombre, trop recouverte), mais un seul allié vivant manque : c'est lui.
            if cfg.use_ring and cfg.ring_only_recovery and self.my_team is not None:
                ally_color = "blue" if cfg.ring_mode == "ally" else ("blue" if self.my_team == "ORDER" else "red")
                if ally_color in signal:
                    remaining = [l for l in labels if l not in taken and self.teams.get(l) == self.my_team]
                    if len(remaining) == 1 and not found_here:
                        center = self._fit_ring_center(crop, a, ally_color, d, others=accepted + found)
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

    def _infer_hidden_allies(self, accepted: list[Candidate], exclude: set[str] | None,
                             crop: np.ndarray | None = None) -> list[Candidate]:
        """Alliés vivants non détectés : placés sous l'icône détectée la plus proche de leur dernière position.

        Un allié vivant est toujours dessiné sur la minimap ; s'il n'est vu nulle part, il est
        sous une autre icône. On choisit l'hôte le plus proche de sa dernière position connue
        (à `hidden_radius_factor` diamètres au plus) et on le suit avec lui, au plus
        `hidden_max_frames` frames, avec une confiance réduite et le marqueur `hidden`.
        """
        if self.my_team is None or not accepted:
            return []
        cfg = self.config
        seen = {c.label for c in accepted}
        out: list[Candidate] = []
        enemy_color = "red" if (cfg.ring_mode == "ally" or self.my_team == "ORDER") else "blue"
        signal_cache: dict[int, dict] = {}
        for label, team in self.teams.items():
            if label in seen or (exclude and label in exclude):
                continue
            if team != self.my_team:
                # Ennemi : seulement s'il vient d'être vu, qu'une icône est tout près de sa dernière
                # position et qu'un croissant d'anneau ennemi dépasse de cette icône (preuve qu'une
                # icône est dessous). Sinon il est dans le brouillard, et on ne l'invente pas.
                if not cfg.infer_hidden_enemies:
                    continue
                tr = self.tracks.get(label)
                if tr is None or tr.age > cfg.hidden_enemy_max_age or tr.hidden_frames >= cfg.hidden_enemy_max_age:
                    continue
                if tr.last_score < 0.7:
                    continue  # jamais vraiment vu : pas de déduction
                hosts = [c for c in accepted if not c.hidden and (c.ring is not None or c.at_spawn or c.score >= cfg.ring_required_below)]
                if not hosts:
                    continue
                host = min(hosts, key=lambda c: (c.cx - tr.px) ** 2 + (c.cy - tr.py) ** 2)
                r_e = cfg.hidden_enemy_radius_factor * self.base_diameter
                if (host.cx - tr.px) ** 2 + (host.cy - tr.py) ** 2 > r_e * r_e:
                    continue
                if id(host) not in signal_cache:
                    sig = self._neighbor_ring_signal(crop, host, others=accepted) if crop is not None else {}
                    fit = (self._fit_ring_center(crop, host, enemy_color, self.base_diameter, others=accepted)
                           if crop is not None and enemy_color in sig else None)
                    signal_cache[id(host)] = {"fit": fit}
                fit = signal_cache[id(host)]["fit"]
                if fit is None:
                    continue  # pas d'anneau ennemi creux qui dépasse : rien ne prouve une icône dessous
                out.append(Candidate(label=label, score=0.0, adjusted=cfg.hidden_confidence, cx=fit[0], cy=fit[1],
                                     diameter=self.base_diameter, ring=None, hidden=True))
                continue
            tr = self.tracks.get(label)
            if tr is None or tr.hidden_frames >= cfg.hidden_max_frames:
                continue
            # Plus la piste est ancienne, plus l'allié a pu s'éloigner : rayon de recherche croissant.
            radius = min(cfg.hidden_radius_max, cfg.hidden_radius_factor + cfg.hidden_radius_growth * tr.age) * self.base_diameter
            # Hôtes possibles : des icônes vues pour de bon (anneau lu ou score élevé), jamais une
            # détection déduite ni un pic douteux, sinon des fantômes s'hébergeraient entre eux.
            hosts = [c for c in accepted if not c.hidden and (c.ring is not None or c.at_spawn or c.score >= cfg.ring_required_below)]
            if not hosts:
                continue
            near_hosts = [c for c in hosts if (c.cx - tr.px) ** 2 + (c.cy - tr.py) ** 2 <= radius * radius]
            if not near_hosts:
                continue
            host = min(near_hosts, key=lambda c: (c.cx - tr.px) ** 2 + (c.cy - tr.py) ** 2)
            # Portrait franchement coloré (Brand, rouge à 60 %) : le croissant de cette couleur qui
            # dépasse d'un hôte désigne l'hôte mieux que la distance.
            if crop is not None and self._masks is not None:
                pc = self._portrait_color.get(label, {})
                color = "red" if pc.get("red", 0) >= 0.3 else ("blue" if pc.get("blue", 0) >= 0.3 else None)
                if color is not None:
                    best = max(near_hosts, key=lambda c: self._crescent_color_fraction(crop, c, color, accepted))
                    if self._crescent_color_fraction(crop, best, color, accepted) >= 0.03:
                        host = best
            out.append(Candidate(label=label, score=0.0, adjusted=cfg.hidden_confidence, cx=host.cx, cy=host.cy,
                                 diameter=self.base_diameter, ring=None, hidden=True))
        return out

    def _hue_consistent(self, crop: np.ndarray, label: str, offset: tuple[int, int], occl: np.ndarray) -> bool:
        """Le croissant visible a-t-il les couleurs du portrait proposé ? (histogrammes de teinte des
        pixels vifs, corrélation ≥ 0,4 ; un croissant orange ne peut pas être Nasus.)"""
        d = self.base_diameter
        half = d // 2
        cx, cy = offset
        win = crop[cy - half:cy - half + d, cx - half:cx - half + d]
        tpl = self._templates_at_base().get(label)
        if tpl is None or win.shape[:2] != (d, d):
            return True
        vis = (tpl.mask > 0) & ~occl
        if vis.sum() < 20:
            return True
        hw = cv2.cvtColor(np.ascontiguousarray(win), cv2.COLOR_BGR2HSV)
        ht = cv2.cvtColor(tpl.image, cv2.COLOR_BGR2HSV)
        def hist(h, m):
            sel = m & (h[..., 1] > 60) & (h[..., 2] > 50)
            if sel.sum() < 10:
                return None
            return np.bincount((h[..., 0][sel] // 12).astype(int), minlength=15).astype(np.float32)
        a, b = hist(hw, vis), hist(ht, tpl.mask > 0)
        if a is None or b is None:
            return True  # pas assez de couleur pour trancher
        a, b = a / a.sum(), b / b.sum()
        corr = float(np.minimum(a, b).sum())  # intersection d'histogrammes (0..1)
        return corr >= 0.4

    def _crescent_color_fraction(self, crop: np.ndarray, host: Candidate, color: str,
                                 accepted: list[Candidate]) -> float:
        """Fraction de pixels de surface `color` (portraits, sans points de sbires) dans la couronne
        0,64-1,1 diamètre autour de l'hôte, hors disques des autres icônes vues et des structures."""
        d = self.base_diameter
        h, w = crop.shape[:2]
        r_out = 1.1 * d
        x0, x1 = int(max(0, host.cx - r_out)), int(min(w, host.cx + r_out + 1))
        y0, y1 = int(max(0, host.cy - r_out)), int(min(h, host.cy + r_out + 1))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return 0.0
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist2 = (xx - host.cx) ** 2 + (yy - host.cy) ** 2
        band = (dist2 >= (0.64 * d) ** 2) & (dist2 <= r_out * r_out)
        for o in accepted:
            if o is not host and not o.hidden:
                band &= ((xx - o.cx) ** 2 + (yy - o.cy) ** 2) > (o.diameter / 2 + 3) ** 2
        for sx, sy in self._structures:
            if abs(sx - host.cx) < 2 * d and abs(sy - host.cy) < 2 * d:
                band &= ((xx - sx) ** 2 + (yy - sy) ** 2) > (0.5 * d) ** 2
        if band.sum() < 8:
            return 0.0
        area = self._masks.red_area if color == "red" else self._masks.blue_area
        return float(area[y0:y1, x0:x1][band].mean())

    def _update_tracks(self, accepted: list[Candidate], t: float) -> None:
        seen = {c.label for c in accepted}
        for c in accepted:
            if c.label == UNKNOWN_LABEL:
                continue
            tr = self.tracks.get(c.label)
            if c.hidden:
                # Suivi « caché » : la piste suit l'hôte, mais vieillit comme une piste non vue
                # (pas de continuité à seuil abaissé, pas de remise à zéro du compteur sans anneau).
                if tr is not None:
                    tr.px, tr.py, tr.t = c.cx, c.cy, t
                    tr.age += 1
                    tr.hidden_frames += 1
                continue
            if tr is None:
                self.tracks[c.label] = Track(c.label, c.cx, c.cy, t, age=0, hits=1)
                tr = self.tracks[c.label]
            else:
                tr.px, tr.py, tr.t, tr.age, tr.hits = c.cx, c.cy, t, 0, tr.hits + 1
            tr.last_score = c.score
            tr.hidden_frames = 0
            tr.ringless = tr.ringless + 1 if (self.config.use_ring and c.ring is None) else 0
        for label, tr in list(self.tracks.items()):
            if label not in seen:
                tr.age += 1
                tr.hidden_frames = 0
                ally = self.my_team is not None and self.teams.get(label) == self.my_team
                if tr.age > self.config.track_max_age and not (ally and self.config.infer_hidden_allies):
                    del self.tracks[label]  # un allié vivant est forcément sur la minimap : sa piste reste
            elif tr.hidden_frames >= self.config.hidden_max_frames:
                del self.tracks[label]

    def _fountain(self, crop: np.ndarray) -> tuple[float, float]:
        """Point d'apparition de mon équipe dans le crop (calibration), sinon le coin approximatif."""
        if self._spawn is not None:
            return self._spawn
        h, w = crop.shape[:2]
        return (0.055 * w, 0.955 * h) if self.my_team == "ORDER" else (0.945 * w, 0.045 * h)

    def _respawn_tracks(self, crop: np.ndarray, t: float, exclude: set[str] | None) -> None:
        """Un allié mort à la frame précédente et vivant maintenant réapparaît à la fontaine : sa
        piste repart de là (déduit sous un allié vu à la fontaine, puis suivi quand il en sort)."""
        excl = set(exclude or ())
        if self.my_team is not None:
            fx, fy = self._fountain(crop)
            for label in self._last_exclude - excl:
                if self.teams.get(label) == self.my_team and label not in self.tracks:
                    self.tracks[label] = Track(label, fx, fy, t, age=1, hits=0)
        self._last_exclude = excl

    def _init_spawn_tracks(self, crop: np.ndarray, t: float, exclude: set[str] | None) -> None:
        """Début de partie : un allié jamais vu est à sa fontaine (empilé avec les autres)."""
        if self.my_team is None or t > self.config.spawn_init_seconds:
            return
        fx, fy = self._fountain(crop)
        for label, team in self.teams.items():
            if team == self.my_team and label not in self.tracks and label not in self._spawned \
                    and not (exclude and label in exclude):
                self.tracks[label] = Track(label, fx, fy, t, age=1, hits=0)
                self._spawned.add(label)

    def detect(self, crop: np.ndarray, calib: Calibration, t: float, frame_id: str,
               exclude: set[str] | None = None) -> list[Detection]:
        """`exclude` : champions à ne pas chercher (morts : le jeu ne les dessine pas sur la minimap)."""
        start = time.perf_counter()
        self._unknown_team = {}
        self._halo_cache = {}
        if self._mapper is None or self._mapper.calib is not calib:
            self._mapper = MinimapMapper(calib)
            self._structures = self._mapper.structures_crop()
            self._spawn = self._mapper.spawn_crop(self.my_team) if self.my_team in ("ORDER", "CHAOS") else None
        self._init_spawn_tracks(crop, t, exclude)
        for label in exclude or ():
            self.tracks.pop(label, None)  # un mort réapparaît à la base, pas là où il était
        self._respawn_tracks(crop, t, exclude)
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        if self.config.erase_white_lines:
            crop = erase_white_lines(crop)
        self._masks = FrameMasks(crop) if self.config.use_ring else None
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
        if self.config.infer_hidden_allies:
            accepted += self._infer_hidden_allies(accepted, exclude, crop)
        self.last_accepted = accepted
        self._update_tracks(accepted, t)
        self._probe_scales(accepted)
        dets: list[Detection] = []
        for c in accepted:
            dets.append(Detection(
                t=t, frameId=frame_id, kind="champion", label=c.label,
                pos=self._mapper.crop_to_map(c.cx, c.cy),
                confidence=float(min(1.0, max(0.0, c.adjusted))),
                team=self.teams.get(c.label) or self._unknown_team.get((round(c.cx), round(c.cy))),
                px=Pos(x=round(c.cx, 1), y=round(c.cy, 1)),
                hidden=c.hidden,
            ))
        self.last_latency_ms = (time.perf_counter() - start) * 1000.0
        return dets
