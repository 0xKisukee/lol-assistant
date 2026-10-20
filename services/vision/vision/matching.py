"""Corrélation normalisée masquée (NCC) rapide par FFT.

`cv2.matchTemplate` avec masque emprunte un chemin lent (≈ 45 ms par template
sur un crop de 300 px). On calcule ici exactement la même quantité
(TM_CCOEFF_NORMED masqué, à 1e-3 près) avec des corrélations circulaires par
FFT numpy, en sommant les canaux dans le domaine fréquentiel : une FFT inverse
par template, et les termes qui ne dépendent que du masque sont partagés
entre tous les templates de même diamètre.

Pour une fenêtre W, un template T et un masque binaire M (n = |M|·canaux) :

    NCC = Σ M·(W − W̄)·(T − T̄) / sqrt(Σ M·(W − W̄)² · Σ M·(T − T̄)²)
        = S_wt / sqrt((S_ww − S_w²/n) · S_tt)

où S_wt = corr(W, M·(T − T̄)), S_w = corr(W, M), S_ww = corr(W², M).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vision.assets import Template

# Écart-type minimal (niveaux de gris) d'une fenêtre pour qu'un score soit significatif.
SIGMA_MIN = 3.0


def next_fast_len(n: int) -> int:
    """Plus petite taille ≥ n dont les facteurs premiers sont 2, 3 ou 5 (FFT rapide)."""
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


@dataclass
class FrameSpectrum:
    """Spectres du crop courant, partagés par tous les bancs de templates."""

    height: int
    width: int
    fft_shape: tuple[int, int]
    Fw: np.ndarray  # rfft2 de W, forme (fh, fw//2+1, C)
    Fww: np.ndarray  # rfft2 de W²

    @classmethod
    def from_image(cls, crop: np.ndarray) -> "FrameSpectrum":
        if crop.ndim == 2:
            crop = crop[..., None]
        h, w = crop.shape[:2]
        fft_shape = (next_fast_len(h), next_fast_len(w))
        # float32 : numpy ≥ 2 calcule la FFT nativement en simple précision (2× plus vite).
        # S_ww − S_w²/n est une différence de grands nombres (≈ 1e8, erreur absolue ≈ 10) ;
        # le plancher de variance (n·SIGMA_MIN², ≈ 1e4 et plus) la rend sûre.
        W = crop.astype(np.float32)
        Fw = np.fft.rfft2(W, s=fft_shape, axes=(0, 1))
        Fww = np.fft.rfft2(W * W, s=fft_shape, axes=(0, 1))
        return cls(h, w, fft_shape, Fw, Fww)


class NCCBank:
    """Templates de même diamètre, prêts pour la corrélation sur un crop."""

    def __init__(self, templates: list[Template]):
        if not templates:
            raise ValueError("Banc de templates vide.")
        self.templates = templates
        self.diameter = templates[0].diameter
        if any(t.diameter != self.diameter for t in templates):
            raise ValueError("Tous les templates d'un banc doivent avoir le même diamètre.")
        self.channels = templates[0].image.shape[2] if templates[0].image.ndim == 3 else 1
        self._cache_shape: tuple[int, int] | None = None
        self._spec_t: list[np.ndarray] = []
        self._S_tt: list[float] = []
        self._spec_m: np.ndarray | None = None
        self._n = 0.0

    def _prepare(self, fft_shape: tuple[int, int]) -> None:
        if self._cache_shape == fft_shape:
            return
        d, C = self.diameter, self.channels
        M = (self.templates[0].mask > 0).astype(np.float32)
        self._n = float(M.sum() * C)
        self._spec_m = np.conj(np.fft.rfft2(M, s=fft_shape))
        self._spec_t, self._S_tt = [], []
        for tpl in self.templates:
            T = tpl.image.astype(np.float32)
            if T.ndim == 2:
                T = T[..., None]
            Mi = (tpl.mask > 0).astype(np.float32)[..., None]
            mean_t = float((T * Mi).sum() / self._n)
            Tp = (T - mean_t) * Mi
            self._S_tt.append(float((Tp * Tp).sum()))
            self._spec_t.append(np.conj(np.fft.rfft2(Tp, s=fft_shape, axes=(0, 1))))
        self._cache_shape = fft_shape

    def match(self, spec: FrameSpectrum) -> dict[str, np.ndarray]:
        """Cartes de corrélation (forme (H−d+1, W−d+1)) par label. Vide si le crop est trop petit."""
        d = self.diameter
        oh, ow = spec.height - d + 1, spec.width - d + 1
        if oh <= 0 or ow <= 0:
            return {}
        self._prepare(spec.fft_shape)
        fh, fw = spec.fft_shape
        sm = self._spec_m[..., None]
        S_w = np.fft.irfft2((spec.Fw * sm).sum(axis=2), s=(fh, fw))[:oh, :ow]
        S_ww = np.fft.irfft2((spec.Fww * sm).sum(axis=2), s=(fh, fw))[:oh, :ow]
        # Plancher de variance : une fenêtre plate (écart-type < SIGMA_MIN niveaux)
        # ne peut pas produire un score élevé par simple bruit numérique.
        var_w = np.maximum(S_ww - S_w * S_w / self._n, self._n * SIGMA_MIN ** 2)
        out: dict[str, np.ndarray] = {}
        for tpl, st, stt in zip(self.templates, self._spec_t, self._S_tt):
            S_wt = np.fft.irfft2((spec.Fw * st).sum(axis=2), s=(fh, fw))[:oh, :ow]
            res = S_wt / np.sqrt(var_w * stt)
            out[tpl.label] = np.clip(np.nan_to_num(res, nan=-1.0), -1.0, 1.0).astype(np.float32)
        return out
