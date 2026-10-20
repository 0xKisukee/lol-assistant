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
    det = _detector(portraits, truth)
    calib = _calib(300, truth["iconDiameter"])
    det.detect(minimap, calib, 0.0, "0")
    assert set(det.tracks) == {c["label"] for c in truth["champions"]}
    assert all(t.age == 0 and t.hits == 1 for t in det.tracks.values())
    # Frame légèrement décalée : continuité, toujours 10.
    shifted = np.roll(minimap, 2, axis=1)
    dets = det.detect(shifted, calib, 0.5, "1")
    assert len(dets) == 10 and all(t.hits == 2 for t in det.tracks.values())
    # Fond vide : rien, les pistes vieillissent puis disparaissent.
    for i in range(det.config.track_max_age + 1):
        assert det.detect(minimap_empty, calib, 1.0 + i, str(i + 2)) == []
    assert det.tracks == {}


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
