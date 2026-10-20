"""Estimation du rectangle de la minimap et mapping pixel ↔ carte."""

import json
from pathlib import Path

import pytest

from vision.calib import (LANDMARKS, MinimapMapper, estimate_calibration, get_calibration, load_calibrations,
                          save_calibration, units_to_norm, GameConfig)
from vision.models import Calibration, MinimapRect, Pos


@pytest.mark.parametrize("w,h", [(1920, 1080), (2560, 1440), (3456, 2234)])
def test_estimate_is_square_in_bottom_right(w, h):
    c = estimate_calibration(w, h, 1.0)
    r = c.minimap
    assert r.w == r.h
    assert r.x2 == w and r.y2 == h
    assert 0.2 * h < r.w < 0.3 * h
    assert c.iconDiameter >= 8
    assert c.source == "estimate"


def test_estimate_scale_monotonic_and_flip():
    small = estimate_calibration(1920, 1080, 0.0).minimap
    big = estimate_calibration(1920, 1080, 1.0).minimap
    assert small.w < big.w
    flipped = estimate_calibration(1920, 1080, 1.0, flipped=True).minimap
    assert flipped.x == 0 and flipped.y2 == 1080


def test_mapping_corners_and_roundtrip():
    c = Calibration(width=1920, height=1080, minimap=MinimapRect(x=1620, y=780, w=300, h=300), iconDiameter=24)
    m = MinimapMapper(c)
    # Coin bas-gauche de la minimap = sud-ouest de la carte = (0, 0) ; haut-droit = (1, 1).
    assert m.crop_to_map(0, 300) == Pos(x=0.0, y=0.0)
    assert m.crop_to_map(300, 0) == Pos(x=1.0, y=1.0)
    assert m.frame_to_map(1620, 1080) == Pos(x=0.0, y=0.0)
    for px, py in [(10, 20), (150, 150), (299, 1)]:
        pos = m.crop_to_map(px, py)
        bx, by = m.map_to_crop(pos)
        assert abs(bx - px) < 1e-6 and abs(by - py) < 1e-6
        fx, fy = m.map_to_frame(m.frame_to_map(1620 + px, 780 + py))
        assert abs(fx - (1620 + px)) < 1e-6 and abs(fy - (780 + py)) < 1e-6


def test_mapping_clamps_outside():
    m = MinimapMapper(Calibration(width=100, height=100, minimap=MinimapRect(x=0, y=0, w=100, h=100), iconDiameter=10))
    assert m.crop_to_map(-5, 120) == Pos(x=0.0, y=0.0)


def test_landmarks_bases():
    """La base ORDER est au sud-ouest (près de (0,0)), la base CHAOS au nord-est."""
    order, chaos = LANDMARKS["nexus ORDER"], LANDMARKS["nexus CHAOS"]
    assert order.x < 0.2 and order.y < 0.2
    assert chaos.x > 0.8 and chaos.y > 0.8
    m = MinimapMapper(Calibration(width=100, height=100, minimap=MinimapRect(x=0, y=0, w=100, h=100), iconDiameter=10))
    ox, oy = m.map_to_crop(order)
    cx, cy = m.map_to_crop(chaos)
    assert ox < 20 and oy > 80  # bas-gauche du crop
    assert cx > 80 and cy < 20  # haut-droit du crop
    assert units_to_norm(-120, -120) == Pos(x=0.0, y=0.0)


def test_crop_extracts_rect(frame_1080p):
    c = Calibration(width=1920, height=1080, minimap=MinimapRect(x=1620, y=780, w=300, h=300), iconDiameter=24)
    crop = MinimapMapper(c).crop(frame_1080p)
    assert crop.shape[:2] == (300, 300)


def test_calibration_file_roundtrip(tmp_path: Path):
    path = tmp_path / "calibration.json"
    c = Calibration(width=1920, height=1080, minimap=MinimapRect(x=1600, y=770, w=310, h=310), iconDiameter=26,
                    source="manual")
    save_calibration(c, path)
    loaded = load_calibrations(path)
    assert loaded["1920x1080"] == c
    # get_calibration préfère la version manuelle, et estime sinon.
    assert get_calibration(1920, 1080, GameConfig(), path).source == "manual"
    assert get_calibration(2560, 1440, GameConfig(), path).source == "estimate"
    data = json.loads(path.read_text())
    assert "1920x1080" in data
