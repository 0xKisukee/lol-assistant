"""Le moteur FFT calcule bien la NCC masquée (comparaison avec un calcul direct)."""

import numpy as np

from vision.assets import make_templates
from vision.matching import FrameSpectrum, NCCBank, next_fast_len


def test_next_fast_len():
    assert next_fast_len(300) == 300
    assert next_fast_len(301) == 320
    assert next_fast_len(7) == 8


def _direct_ncc(window, tpl, mask):
    M = mask > 0
    w = window.astype(np.float64)[M]
    t = tpl.astype(np.float64)[M]
    w, t = w - w.mean(), t - t.mean()
    return float((w * t).sum() / np.sqrt((w * w).sum() * (t * t).sum()))


def test_fft_matches_direct(minimap, portraits, truth):
    tpls = make_templates(portraits, truth["iconDiameter"])
    maps = NCCBank(tpls).match(FrameSpectrum.from_image(minimap))
    d = truth["iconDiameter"]
    assert all(m.shape == (minimap.shape[0] - d + 1, minimap.shape[1] - d + 1) for m in maps.values())
    rng = np.random.default_rng(1)
    for tpl in tpls:
        for _ in range(5):
            x, y = int(rng.integers(0, 300 - d)), int(rng.integers(0, 300 - d))
            direct = _direct_ncc(minimap[y:y + d, x:x + d], tpl.image, tpl.mask)
            assert abs(maps[tpl.label][y, x] - direct) < 2e-3


def test_peak_at_true_position(minimap, portraits, truth):
    d = truth["iconDiameter"]
    maps = NCCBank(make_templates(portraits, d)).match(FrameSpectrum.from_image(minimap))
    for c in truth["champions"]:
        res = maps[c["label"]]
        y, x = np.unravel_index(int(res.argmax()), res.shape)
        assert abs(x + d / 2 - c["px"]) <= 2 and abs(y + d / 2 - c["py"]) <= 2, c["label"]
        assert res.max() > 0.6


def test_small_crop_returns_empty(portraits):
    tpls = make_templates(portraits, 24)
    assert NCCBank(tpls).match(FrameSpectrum.from_image(np.zeros((10, 10, 3), np.uint8))) == {}


def test_template_from_alpha_icon():
    """Une icône BGRA (CommunityDragon) donne un masque limité aux pixels opaques et resserré."""
    import cv2
    from vision.assets import make_template
    icon = np.zeros((128, 128, 4), np.uint8)
    icon[..., :3] = 90
    cv2.circle(icon, (64, 64), 60, (10, 200, 30, 255), -1)
    tpl = make_template("X", icon, 30)
    assert tpl.image.shape == (30, 30, 3) and tpl.mask.shape == (30, 30)
    assert tpl.mask[15, 15] == 255 and tpl.mask[0, 0] == 0 and tpl.mask[15, 29] == 0  # bord exclu
    area = (tpl.mask > 0).sum()
    assert 0.5 * np.pi * 15 ** 2 < area < 0.75 * np.pi * 15 ** 2  # ≈ (0,82)² du disque
