"""Parsing de game.cfg et recherche du fichier."""

from pathlib import Path

from vision.calib import find_game_cfg, parse_game_cfg

SAMPLE = """\
[General]
Width=1920
Height=1080
Windowed=0
WindowMode=2
; commentaire
[HUD]
MinimapScale=0.6500
GlobalScale=0.2500
FlipMiniMap=1
ChatScale=100
[Performance]
ShadowQuality=2
"""


def test_parse_sample():
    cfg = parse_game_cfg(SAMPLE)
    assert (cfg.width, cfg.height) == (1920, 1080)
    assert cfg.windowed is False
    assert abs(cfg.minimap_scale - 0.65) < 1e-6
    assert abs(cfg.global_scale - 0.25) < 1e-6
    assert cfg.flip_minimap is True


def test_parse_defaults_and_clamping():
    cfg = parse_game_cfg("[HUD]\nMinimapScale=7\nFlipMiniMap=0\n")
    assert cfg.minimap_scale == 1.0  # borné à 1
    assert cfg.flip_minimap is False
    assert cfg.width is None
    assert parse_game_cfg("").minimap_scale == 1.0


def test_parse_ignores_garbage():
    cfg = parse_game_cfg("[HUD]\nMinimapScale=abc\nlignesansegal\n")
    assert cfg.minimap_scale == 1.0


def test_find_game_cfg_explicit(tmp_path: Path, monkeypatch):
    p = tmp_path / "game.cfg"
    p.write_text(SAMPLE)
    assert find_game_cfg(p) == p
    monkeypatch.setenv("LOL_GAME_CFG", str(p))
    assert find_game_cfg(None) == p
    monkeypatch.delenv("LOL_GAME_CFG")
    assert find_game_cfg(tmp_path / "absent.cfg") in (None, *[])  # pas de plantage
