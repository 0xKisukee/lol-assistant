"""Interface commune des détecteurs (PLAN.md §5.5)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from vision.models import Calibration, Detection


class Detector(ABC):
    """Un détecteur prend le crop de la minimap et renvoie des détections horodatées."""

    name: str = "detector"

    @abstractmethod
    def detect(self, crop: np.ndarray, calib: Calibration, t: float, frame_id: str) -> list[Detection]:
        ...

    def describe(self) -> dict:
        """Paramètres à consigner dans le manifest de l'enregistrement."""
        return {"name": self.name}
