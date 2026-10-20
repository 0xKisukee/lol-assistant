"""Fixtures partagées : portraits et minimap synthétiques de tests/fixtures/."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def truth() -> dict:
    return json.loads((FIXTURES / "minimap_synthetic.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def portraits(truth) -> dict:
    out = {}
    for c in truth["champions"]:
        img = cv2.imread(str(FIXTURES / "portraits" / f"{c['label']}.png"), cv2.IMREAD_COLOR)
        assert img is not None, c["label"]
        out[c["label"]] = img
    return out


@pytest.fixture(scope="session")
def minimap() -> "cv2.Mat":
    img = cv2.imread(str(FIXTURES / "minimap_synthetic.png"), cv2.IMREAD_COLOR)
    assert img is not None
    return img


@pytest.fixture(scope="session")
def minimap_empty():
    img = cv2.imread(str(FIXTURES / "minimap_empty.png"), cv2.IMREAD_COLOR)
    assert img is not None
    return img


@pytest.fixture(scope="session")
def frame_1080p():
    img = cv2.imread(str(FIXTURES / "frame_1080p.png"), cv2.IMREAD_COLOR)
    assert img is not None
    return img
