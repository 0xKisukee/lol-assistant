"""Détecteurs. Chacun expose `detect(crop, calib, t, frame_id) -> list[Detection]`."""

from vision.detectors.base import Detector
from vision.detectors.champions import ChampionDetector, DetectorConfig

__all__ = ["Detector", "ChampionDetector", "DetectorConfig"]
