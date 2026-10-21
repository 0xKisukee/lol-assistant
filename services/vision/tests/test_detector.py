"""Détecteur de champions sur la minimap synthétique."""

import time

import numpy as np
import pytest

from vision.assets import make_templates
from vision.detectors.champions import ChampionDetector, DetectorConfig, ring_color
from vision.models import Calibration, MinimapRect


def _calib(side: int, icon: int) -> Calibration:
    return Calibration(width=side, height=side, minimap=MinimapRect(x=0, y=0, w=side, h=side), iconDiameter=icon)


def _detector(portraits, truth, diameter=None, **cfg):
    d = diameter or truth["iconDiameter"]
    teams = {c["label"]: c["team"] for c in truth["champions"]}
    return ChampionDetector(make_templates(portraits, d), teams, my_team=truth["myTeam"], config=DetectorConfig(**cfg))


def _score(dets, truth, tol=3.0):
    gt = {c["label"]: (c["px"], c["py"]) for c in truth["champions"]}
    correct = sum(1 for d in dets if d.label in gt and np.hypot(d.px.x - gt[d.label][0], d.px.y - gt[d.label][1]) <= tol)
    return correct, len(dets) - correct


def test_detects_all_ten(minimap, portraits, truth):
    det = _detector(portraits, truth)
    dets = det.detect(minimap, _calib(300, truth["iconDiameter"]), 12.5, "000001")
    correct, wrong = _score(dets, truth)
    assert correct == 10 and wrong == 0
    for d in dets:
        assert d.kind == "champion" and d.frameId == "000001" and d.t == 12.5
        assert 0 <= d.confidence <= 1
        assert d.team == next(c["team"] for c in truth["champions"] if c["label"] == d.label)
        assert 0 <= d.pos.x <= 1 and 0 <= d.pos.y <= 1
    # Coordonnées carte : Aatrox est en bas à gauche du crop → sud-ouest.
    aatrox = next(d for d in dets if d.label == "Aatrox")
    assert aatrox.pos.x < 0.2 and aatrox.pos.y < 0.2


@pytest.mark.parametrize("diameter", [22, 26])
def test_robust_to_wrong_diameter(minimap, portraits, truth, diameter):
    det = _detector(portraits, truth, diameter=diameter)
    correct, wrong = _score(det.detect(minimap, _calib(300, diameter), 0, "f"), truth)
    assert correct >= 9 and wrong == 0


def test_no_false_positive_on_empty_background(minimap_empty, portraits, truth):
    det = _detector(portraits, truth)
    assert det.detect(minimap_empty, _calib(300, truth["iconDiameter"]), 0, "e") == []


def test_ring_color(minimap, minimap_empty, truth):
    d = truth["iconDiameter"]
    for c in truth["champions"]:
        expected = "blue" if c["team"] == "ORDER" else "red"
        assert ring_color(minimap, c["px"], c["py"], d) == expected, c["label"]
    # Sans icône (fond, rivière comprise), pas d'anneau.
    for c in truth["champions"]:
        assert ring_color(minimap_empty, c["px"], c["py"], d) is None


def test_ring_contradiction_lowers_confidence(minimap, portraits, truth):
    """Si on annonce les équipes à l'envers, l'anneau contredit et la confiance baisse."""
    teams_ok = {c["label"]: c["team"] for c in truth["champions"]}
    teams_bad = {k: ("CHAOS" if v == "ORDER" else "ORDER") for k, v in teams_ok.items()}
    tpls = make_templates(portraits, truth["iconDiameter"])
    ok = ChampionDetector(tpls, teams_ok, my_team="ORDER").detect(minimap, _calib(300, 24), 0, "a")
    bad = ChampionDetector(tpls, teams_bad, my_team="ORDER").detect(minimap, _calib(300, 24), 0, "b")
    conf_ok = {d.label: d.confidence for d in ok}
    for d in bad:
        assert d.confidence < conf_ok[d.label]


def test_tracking_hysteresis_and_age(minimap, minimap_empty, portraits, truth):
    det = _detector(portraits, truth, spawn_init_seconds=0.0)  # pas de fontaine : scénario synthétique
    calib = _calib(300, truth["iconDiameter"])
    det.detect(minimap, calib, 0.0, "0")
    assert set(det.tracks) == {c["label"] for c in truth["champions"]}
    assert all(t.age == 0 and t.hits == 1 for t in det.tracks.values())
    # Frame légèrement décalée : continuité, toujours 10.
    shifted = np.roll(minimap, 2, axis=1)
    dets = det.detect(shifted, calib, 0.5, "1")
    assert len(dets) == 10 and all(t.hits == 2 for t in det.tracks.values())
    # Fond vide : un champion suivi peut survivre sans anneau quelques frames (icône cachée
    # par une voisine), jamais plus de max_ringless_frames acceptations au total ; puis les
    # pistes vieillissent et disparaissent, et rien ne se réinstalle.
    cfg = det.config
    from collections import Counter
    ghosts = Counter()
    frames = []
    for i in range(15):
        dets = det.detect(minimap_empty, calib, 1.0 + i, str(i + 2))
        frames.append(dets)
        for x in dets:
            if not x.hidden:
                ghosts[x.label] += 1
    assert all(n <= cfg.max_ringless_frames for n in ghosts.values()), ghosts
    assert all(f == [] for f in frames[-5:])
    # Les pistes des ennemis disparaissent ; celles des alliés vivants restent (ils sont forcément
    # quelque part sur la minimap), sans produire de détection tant qu'aucun hôte n'est vu.
    allies = {c["label"] for c in truth["champions"] if c["team"] == truth["myTeam"]}
    assert set(det.tracks) <= allies


def test_latency_budget(minimap, portraits, truth):
    det = _detector(portraits, truth)
    calib = _calib(300, truth["iconDiameter"])
    det.detect(minimap, calib, 0, "warm")
    t0 = time.perf_counter()
    for i in range(5):
        det.detect(minimap, calib, i, str(i))
    per_frame = (time.perf_counter() - t0) / 5 * 1000
    assert per_frame < 300, f"{per_frame:.0f} ms par frame"


def test_opencv_engine_agrees(minimap, portraits, truth):
    det = _detector(portraits, truth, engine="opencv", scales=(1.0,))
    correct, wrong = _score(det.detect(minimap, _calib(300, 24), 0, "o"), truth)
    assert correct >= 9 and wrong == 0


def test_describe_has_config(portraits, truth):
    d = _detector(portraits, truth).describe()
    assert d["name"] == "champions-template" and d["baseDiameter"] == truth["iconDiameter"]


def test_overlapping_icons(minimap_empty, portraits, truth):
    """Deux icônes qui se chevauchent (centres à 0,6 diamètre, ~35 % de recouvrement) : les deux sont détectées.

    En dessous (icône recouverte à moitié ou plus), le template matching ne voit plus le
    portrait du dessous : limite connue, le détecteur appris du jalon 2 doit y répondre.
    """
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent / "fixtures"))
    from make_fixtures import paste_icon, BLUE, RED

    d = truth["iconDiameter"]
    crop = minimap_empty.copy()
    paste_icon(crop, portraits["Aatrox"], 150, 150, d, BLUE)  # dessous
    paste_icon(crop, portraits["Garen"], 150 + int(0.6 * d), 150, d, RED)  # dessus, recouvre ~35 % d'Aatrox
    det = _detector(portraits, truth)
    dets = det.detect(crop, _calib(300, d), 0, "o")
    labels = {x.label: x for x in dets}
    assert "Garen" in labels and abs(labels["Garen"].px.x - (150 + int(0.6 * d))) <= 3
    assert "Aatrox" in labels and abs(labels["Aatrox"].px.x - 150) <= 3
    assert len(dets) == 2


def test_exclude_dead(minimap, portraits, truth):
    det = _detector(portraits, truth)
    calib = _calib(300, truth["iconDiameter"])
    det.detect(minimap, calib, 0, "a")
    dets = det.detect(minimap, calib, 1, "b", exclude={"Aatrox", "Jinx"})
    assert {d.label for d in dets} == {c["label"] for c in truth["champions"]} - {"Aatrox", "Jinx"}
    assert "Aatrox" not in det.tracks and "Jinx" not in det.tracks


def test_overlap_recovery(minimap_empty, portraits, truth):
    """Icône recouverte à plus de la moitié : récupérée par la seconde passe (croissant visible), sans fantôme."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent / "fixtures"))
    from make_fixtures import paste_icon, BLUE, RED

    d = truth["iconDiameter"]
    for frac in (0.5, 0.4):
        crop = minimap_empty.copy()
        paste_icon(crop, portraits["Aatrox"], 150, 150, d, BLUE)
        paste_icon(crop, portraits["Garen"], 150 + int(frac * d), 150, d, RED)
        det = _detector(portraits, truth)
        dets = {x.label: x for x in det.detect(crop, _calib(300, d), 0, "o")}
        assert set(dets) == {"Aatrox", "Garen"}, (frac, set(dets))
        assert abs(dets["Aatrox"].px.x - 150) <= 3 and abs(dets["Aatrox"].px.y - 150) <= 3
    # Sans voisin caché : rien d'inventé.
    crop = minimap_empty.copy()
    paste_icon(crop, portraits["Garen"], 150, 150, d, RED)
    det = _detector(portraits, truth)
    assert [x.label for x in det.detect(crop, _calib(300, d), 0, "s")] == ["Garen"]


def test_white_lines_erased(minimap, portraits, truth):
    """Ligne d'ordre et cadre caméra blancs sur une icône : toujours détectée au bon endroit."""
    import cv2
    from vision.detectors.champions import erase_white_lines

    d = truth["iconDiameter"]
    gt = {c["label"]: (c["px"], c["py"]) for c in truth["champions"]}
    crop = minimap.copy()
    x, y = gt["Corki"]
    lines = np.zeros(crop.shape[:2], np.uint8)
    for img, col in ((crop, (255, 255, 255)), (lines, 255)):
        cv2.line(img, (x - 60, y + 40), (x + 70, y - 50), col, 2)  # ordre de déplacement
        cv2.rectangle(img, (x - 35, y - 20), (x + 45, y + 25), col, 2)  # cadre caméra
    cleaned = erase_white_lines(crop)
    still_white = cv2.inRange(cleaned, (225, 225, 225), (255, 255, 255))[lines > 0]
    assert (still_white > 0).mean() < 0.2  # les traits ont été reconstruits depuis leurs voisins
    det = _detector(portraits, truth)
    dets = {x_.label: x_ for x_ in det.detect(crop, _calib(300, d), 0, "l")}
    assert "Corki" in dets and abs(dets["Corki"].px.x - x) <= 3 and abs(dets["Corki"].px.y - y) <= 3
    correct, wrong = _score(list(dets.values()), truth)
    assert correct >= 9 and wrong == 0


def test_static_ringless_peak_suppressed(minimap_empty, portraits, truth):
    """Un faux pic sans anneau qui reste au même endroit (structure de la carte) finit ignoré ;
    une vraie icône (avec anneau) au même endroit reste détectée."""
    import cv2
    from vision.detectors.champions import DetectorConfig

    d = truth["iconDiameter"]
    teams = {c["label"]: c["team"] for c in truth["champions"]}
    tpls = make_templates(portraits, d)
    # « Structure » : le portrait d'Aatrox collé sans anneau, comme une icône fixe de la carte.
    crop = minimap_empty.copy()
    icon = cv2.resize(portraits["Aatrox"], (d, d))
    mask = np.zeros((d, d), np.uint8); cv2.circle(mask, (d // 2, d // 2), d // 2 - 1, 255, -1)
    y0, x0 = 100 - d // 2, 200 - d // 2
    region = crop[y0:y0 + d, x0:x0 + d]
    region[mask > 0] = icon[mask > 0]
    det = ChampionDetector(tpls, teams, my_team="ORDER",
                           config=DetectorConfig(static_frames=4, ring_required_below=0.5, static_exempt_score=1.01))
    seen = [sum(1 for x in det.detect(crop, _calib(300, d), i, str(i)) if x.label == "Aatrox") for i in range(8)]
    assert seen[0] == 1 and seen[-1] == 0  # accepté d'abord (score élevé sans anneau), puis reconnu comme fixe
    # Avec un anneau, ce n'est plus une structure : détecté en continu.
    cv2.circle(crop, (200, 100), d // 2, (235, 140, 40), 2)
    det = ChampionDetector(tpls, teams, my_team="ORDER", config=DetectorConfig(static_frames=4))
    assert all(sum(1 for x in det.detect(crop, _calib(300, d), i, str(i)) if x.label == "Aatrox") == 1 for i in range(8))


def test_still_champion_not_treated_as_static(minimap, portraits, truth):
    """Dix icônes immobiles pendant 40 frames, trois échelles : toutes détectées à chaque frame."""
    det = _detector(portraits, truth)
    calib = _calib(300, truth["iconDiameter"])
    counts = [len(det.detect(minimap, calib, i * 0.5, str(i))) for i in range(40)]
    assert min(counts) == 10, counts


def test_ring_only_recovery_single_missing_ally(minimap_empty, portraits, truth):
    """Icône recouverte aux deux tiers : le portrait ne se reconnaît plus, mais l'anneau allié qui
    dépasse suffit quand un seul allié vivant manque (les autres sont marqués morts)."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent / "fixtures"))
    from make_fixtures import paste_icon, BLUE, RED

    d = truth["iconDiameter"]
    crop = minimap_empty.copy()
    paste_icon(crop, portraits["Aatrox"], 150, 150, d, BLUE)
    paste_icon(crop, portraits["Garen"], 150 + int(0.3 * d), 150 + 2, d, RED)  # recouvre ~70 % d'Aatrox
    det = _detector(portraits, truth, ring_only_recovery=True)
    others = {c["label"] for c in truth["champions"] if c["team"] == "ORDER"} - {"Aatrox"}
    dets = {x.label: x for x in det.detect(crop, _calib(300, d), 0, "r", exclude=others)}
    assert "Garen" in dets
    assert "Aatrox" in dets, dets.keys()
    assert abs(dets["Aatrox"].px.x - 150) <= 4 and abs(dets["Aatrox"].px.y - 150) <= 4
    assert dets["Aatrox"].confidence < dets["Garen"].confidence


def test_hidden_ally_follows_host(minimap_empty, portraits, truth):
    """Allié vu, puis entièrement recouvert par un autre : déduit sous l'icône hôte, marqué caché."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent / "fixtures"))
    from make_fixtures import paste_icon, BLUE, RED

    d = truth["iconDiameter"]
    det = _detector(portraits, truth)
    calib = _calib(300, d)
    others = {c["label"] for c in truth["champions"] if c["team"] == "ORDER"} - {"Aatrox"}
    crop = minimap_empty.copy()
    paste_icon(crop, portraits["Aatrox"], 150, 150, d, BLUE)
    paste_icon(crop, portraits["Garen"], 190, 150, d, RED)
    first = {x.label: x for x in det.detect(crop, calib, 0, "a", exclude=others)}
    assert "Aatrox" in first and not first["Aatrox"].hidden
    # Garen vient se placer exactement sur Aatrox : Aatrox disparaît, mais reste déduit sous Garen.
    crop = minimap_empty.copy()
    paste_icon(crop, portraits["Aatrox"], 160, 150, d, BLUE)
    paste_icon(crop, portraits["Garen"], 160, 150, d, RED)
    second = {x.label: x for x in det.detect(crop, calib, 0.5, "b", exclude=others)}
    assert "Aatrox" in second and second["Aatrox"].hidden
    assert abs(second["Aatrox"].px.x - 160) <= 2 and second["Aatrox"].confidence < 0.5
    # Un ennemi non vu n'est jamais déduit.
    assert all(not x.hidden for x in second.values() if x.label != "Aatrox")


def test_ring_read_on_river(minimap_empty, portraits, truth):
    """Icône alliée posée sur la rivière (surface bleue saturée) : l'anneau bleu, collé à la rivière,
    doit quand même être lu (la rivière est écartée par sa teinte et sa saturation)."""
    import cv2
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent / "fixtures"))
    from make_fixtures import paste_icon, BLUE

    d = truth["iconDiameter"]
    crop = minimap_empty.copy()
    river = cv2.cvtColor(np.uint8([[[97, 228, 153]]]), cv2.COLOR_HSV2BGR)[0, 0]  # teinte 97, très saturée
    crop[100:200, 60:240] = river
    paste_icon(crop, portraits["Aatrox"], 150, 150, d, BLUE)
    assert ring_color(crop, 150, 150, d) == "blue"
    # Et sur un sol bleu pâle (plateforme de la fontaine).
    crop = minimap_empty.copy()
    crop[100:200, 60:240] = cv2.cvtColor(np.uint8([[[96, 80, 150]]]), cv2.COLOR_HSV2BGR)[0, 0]
    paste_icon(crop, portraits["Aatrox"], 150, 150, d, BLUE)
    assert ring_color(crop, 150, 150, d) == "blue"
    # Sans icône, la rivière seule n'est pas un anneau.
    crop = minimap_empty.copy()
    crop[100:200, 60:240] = river
    assert ring_color(crop, 150, 150, d) is None


def test_static_peak_released_when_champion_stands_there(minimap_empty, portraits, truth):
    """Pic fixe confirmé (structure), puis un vrai champion avec anneau s'arrête exactement dessus :
    il doit être détecté tout de suite, pas seulement à la ré-observation (une frame sur 20)."""
    import cv2
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent / "fixtures"))
    from make_fixtures import paste_icon, BLUE

    d = truth["iconDiameter"]
    teams = {c["label"]: c["team"] for c in truth["champions"]}
    tpls = make_templates(portraits, d)
    crop = minimap_empty.copy()
    icon = cv2.resize(portraits["Aatrox"], (d, d))
    mask = np.zeros((d, d), np.uint8); cv2.circle(mask, (d // 2, d // 2), d // 2 - 1, 255, -1)
    y0, x0 = 100 - d // 2, 200 - d // 2
    region = crop[y0:y0 + d, x0:x0 + d]
    region[mask > 0] = icon[mask > 0]
    with_ring = crop.copy()
    paste_icon(with_ring, portraits["Aatrox"], 200, 100, d, BLUE)
    for guard in (False, True):
        det = ChampionDetector(tpls, teams, my_team="ORDER",
                               config=DetectorConfig(static_frames=4, ring_required_below=0.5, static_exempt_score=1.01,
                                                     static_ring_guard=guard))
        for i in range(8):
            det.detect(crop, _calib(300, d), i, str(i))
        # Pic fixe confirmé : plus vu (sauf déduit « caché » sous un faux pic sans anneau, artefact
        # de ring_required_below=0.5 sur le fond synthétique : on ne compte que les vraies détections).
        assert not any(x.label == "Aatrox" and not x.hidden for x in det.detect(crop, _calib(300, d), 8, "8"))
        seen = [sum(1 for x in det.detect(with_ring, _calib(300, d), 9 + i, str(9 + i)) if x.label == "Aatrox" and not x.hidden)
                for i in range(3)]
        if guard:
            assert seen[1] == 1 and seen[2] == 1, seen
        else:
            assert seen[2] == 0, seen  # sans la garde, effacé jusqu'à la ré-observation


def test_respawn_track_restarts_at_fountain(minimap_empty, portraits, truth):
    """Allié mort puis vivant, non visible (empilé à la fontaine sous un autre allié) : déduit sous
    l'allié vu à la fontaine, grâce à la piste remise à la fontaine à la réapparition."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent / "fixtures"))
    from make_fixtures import paste_icon, BLUE

    d = truth["iconDiameter"]
    det = _detector(portraits, truth)
    calib = _calib(300, d)
    others = {c["label"] for c in truth["champions"] if c["team"] == "ORDER"} - {"Aatrox", "Bard"}
    crop = minimap_empty.copy()
    paste_icon(crop, portraits["Bard"], 24, 280, d, BLUE)  # Bard à la fontaine (coin bas gauche)
    t = 100.0  # après la phase de départ
    for i in range(3):  # Aatrox mort
        dets = det.detect(crop, calib, t + i, str(i), exclude=others | {"Aatrox"})
        assert not any(x.label == "Aatrox" for x in dets)
    dets = {x.label: x for x in det.detect(crop, calib, t + 3, "3", exclude=others)}  # ressuscité, invisible
    assert "Bard" in dets and not dets["Bard"].hidden
    assert "Aatrox" in dets and dets["Aatrox"].hidden
    assert abs(dets["Aatrox"].px.x - 24) <= 2 and abs(dets["Aatrox"].px.y - 280) <= 2


def test_icon_at_spawn_accepted_without_ring(minimap_empty, portraits, truth):
    """Au point d'apparition, l'anneau est illisible (plateforme, arc de soin) : un portrait reconnu
    à 0,62 ou plus y est accepté sans anneau, n'est jamais pris pour un pic fixe, et sert d'hôte à
    un allié réapparu non vu."""
    import cv2
    from vision.calib import MinimapMapper

    d = truth["iconDiameter"]
    calib = _calib(300, d)
    sx, sy = MinimapMapper(calib).spawn_crop("ORDER")
    cx, cy = int(round(sx)) + 2, int(round(sy)) - 2
    crop = minimap_empty.copy()
    cv2.circle(crop, (int(round(sx)), int(round(sy))), 2 * d, (150, 150, 140), -1)  # plateforme unie, comme la vraie
    icon = cv2.resize(portraits["Aatrox"], (d, d))
    mask = np.zeros((d, d), np.uint8); cv2.circle(mask, (d // 2, d // 2), d // 2 - 1, 255, -1)
    region = crop[cy - d // 2:cy - d // 2 + d, cx - d // 2:cx - d // 2 + d]
    region[mask > 0] = icon[mask > 0]  # portrait sans anneau, exactement au spawn
    det = _detector(portraits, truth, static_frames=4)
    others = {c["label"] for c in truth["champions"] if c["team"] == "ORDER"} - {"Aatrox", "Bard"}
    for i in range(12):  # immobile longtemps : jamais effacé comme pic fixe
        dets = {x.label: x for x in det.detect(crop, calib, 100 + i, str(i), exclude=others | {"Bard"})}
        assert "Aatrox" in dets and not dets["Aatrox"].hidden, (i, dets.keys())
        assert abs(dets["Aatrox"].px.x - cx) <= 1 and abs(dets["Aatrox"].px.y - cy) <= 1
    # Bard ressuscite, invisible (empilé sous Aatrox au spawn) : déduit sous Aatrox.
    dets = {x.label: x for x in det.detect(crop, calib, 112, "12", exclude=others)}
    assert "Bard" in dets and dets["Bard"].hidden and abs(dets["Bard"].px.x - cx) <= 1
    # Le même portrait sans anneau ailleurs sur la carte reste refusé (score < 0,85 sans anneau).
    crop2 = minimap_empty.copy()
    region = crop2[150 - d // 2:150 - d // 2 + d, 150 - d // 2:150 - d // 2 + d]
    region[mask > 0] = icon[mask > 0]
    det2 = _detector(portraits, truth, ring_required_below=1.01)
    assert not any(x.label == "Aatrox" and not x.hidden for x in det2.detect(crop2, calib, 100, "a", exclude=others | {"Bard"}))


def test_concentric_hidden_enemy_marked(minimap_empty, portraits, truth):
    """Ennemi presque exactement sous un autre ennemi (décalage 0,25 diamètre) : son anneau dépasse en
    bande épaisse d'un côté ; on signale quelqu'un là (« ? » de son équipe, ou le portrait attribué)."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent / "fixtures"))
    from make_fixtures import paste_icon, RED

    d = truth["iconDiameter"]
    crop = minimap_empty.copy()
    paste_icon(crop, portraits["Hecarim"], 150, 150, d, RED)  # dessous
    paste_icon(crop, portraits["Garen"], 150 + int(round(0.25 * d)), 150, d, RED)  # dessus
    det = _detector(portraits, truth)
    dead = {c["label"] for c in truth["champions"]} - {"Garen", "Hecarim"}
    dets = det.detect(crop, _calib(300, d), 100.0, "c", exclude=dead)
    labels = {x.label for x in dets}
    assert "Garen" in labels
    under = [x for x in dets if x.label in ("?", "Hecarim") and x.hidden and abs(x.px.x - 150) <= 6 and abs(x.px.y - 150) <= 6]
    assert under, [(x.label, x.px, x.hidden) for x in dets]
    assert all(x.team == "CHAOS" for x in under)
    # Un seul ennemi, rien dessous : rien d'inventé.
    crop = minimap_empty.copy()
    paste_icon(crop, portraits["Garen"], 150, 150, d, RED)
    det = _detector(portraits, truth)
    assert [x.label for x in det.detect(crop, _calib(300, d), 100.0, "s", exclude=dead)] == ["Garen"]


def test_arc_span_wraparound():
    """L'étendue d'un arc centré sur ±180° n'est pas un cercle complet."""
    from vision.detectors.champions import _arc_span_deg
    a = np.deg2rad(np.array([150.0, 160.0, 170.0, -170.0, -160.0, -150.0]))
    assert abs(_arc_span_deg(a) - 60.0) < 1e-6
    assert abs(_arc_span_deg(np.deg2rad(np.array([0.0, 90.0, 180.0, -90.0]))) - 270.0) < 1e-6
    assert _arc_span_deg(np.array([0.3])) == 0.0
