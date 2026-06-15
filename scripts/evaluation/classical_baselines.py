"""
Classical (non-learning) spectral-domain speech-enhancement baselines:
spectral subtraction and Wiener filtering.

Both operate on the *same* STFT framing as the neural pipeline
(`config.compute_stft` / `config.compute_istft`: n_fft=512, hop=128, Hann) and
reconstruct the enhanced waveform reusing the **noisy phase** -- exactly the
phase-reuse strategy used by the Park R-CED magnitude model, so the comparison
is apples-to-apples in everything except the magnitude estimator.

The stationary-noise magnitude spectrum is estimated per frequency from the
low-energy frames of the clip (a minimum-statistics-style estimate), which fits
the curated stationary noise set (AirConditioner, CopyMachine, ...).

Reference: P. C. Loizou, "Speech Enhancement: Theory and Practice", 2013.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import compute_stft, compute_istft  # noqa: E402


def estimate_noise_mag(noisy_mag: np.ndarray, percentile: float = 15.0) -> np.ndarray:
    """Per-frequency stationary-noise magnitude estimate.

    For stationary noise the quietest frames in each frequency bin are dominated
    by noise, so a low percentile over time is a robust noise-floor estimate that
    needs no explicit noise-only segment.

    Args:
        noisy_mag: |STFT| of the noisy signal, shape (n_freq, n_frames).
        percentile: time-axis percentile used as the noise estimate.

    Returns:
        Noise magnitude estimate of shape (n_freq, 1).
    """
    return np.percentile(noisy_mag, percentile, axis=1, keepdims=True)


def spectral_subtraction(noisy: np.ndarray,
                         alpha: float = 2.0,
                         beta: float = 0.01,
                         noise_percentile: float = 15.0) -> np.ndarray:
    """Magnitude spectral subtraction with over-subtraction and a spectral floor.

    |S^| = max(|Y| - alpha*|N^|, beta*|Y|), phase reused from Y.
    """
    stft = compute_stft(noisy)
    mag = np.abs(stft).astype(np.float64)
    phase = np.angle(stft)

    noise_mag = estimate_noise_mag(mag, noise_percentile)
    est_mag = np.maximum(mag - alpha * noise_mag, beta * mag)

    est_stft = est_mag * np.exp(1j * phase)
    return compute_istft(est_stft).astype(np.float32)


def wiener_filter(noisy: np.ndarray,
                  noise_percentile: float = 15.0,
                  alpha_dd: float = 0.98,
                  xi_min_db: float = -25.0) -> np.ndarray:
    """Wiener filter with a decision-directed a priori SNR (Ephraim-Malah).

    Gain G = xi / (1 + xi), where the a priori SNR xi is tracked across frames by
    the decision-directed update; phase reused from the noisy signal.
    """
    stft = compute_stft(noisy)
    mag = np.abs(stft).astype(np.float64)
    phase = np.angle(stft)

    noise_power = estimate_noise_mag(mag, noise_percentile)[:, 0] ** 2 + 1e-12  # (F,)
    power = mag ** 2
    gamma = power / noise_power[:, None]                                        # a posteriori SNR
    xi_min = 10.0 ** (xi_min_db / 10.0)

    n_freq, n_frames = mag.shape
    gain = np.empty((n_freq, n_frames), dtype=np.float64)
    prev_s_power = np.zeros(n_freq)
    for t in range(n_frames):
        gamma_ml = np.maximum(gamma[:, t] - 1.0, 0.0)
        if t == 0:
            xi = np.maximum(gamma_ml, xi_min)
        else:
            xi = alpha_dd * (prev_s_power / noise_power) + (1.0 - alpha_dd) * gamma_ml
            xi = np.maximum(xi, xi_min)
        g = xi / (1.0 + xi)
        prev_s_power = (g * mag[:, t]) ** 2
        gain[:, t] = g

    est_stft = (gain * mag) * np.exp(1j * phase)
    return compute_istft(est_stft).astype(np.float32)


# Registry so callers can iterate over the classical methods uniformly.
CLASSICAL_METHODS = {
    "specsub": spectral_subtraction,
    "wiener": wiener_filter,
}
