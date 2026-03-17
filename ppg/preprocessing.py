"""
PPG Signal Preprocessing
=========================
Bandpass filter → Hanning smoothing → Tukey tapering.
All parameters come from PPGConfig so nothing is hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, windows

from .config import PPGConfig
from .loader import SignalData


@dataclass
class PreprocessedSignal:
    """Output of the preprocessing pipeline for one channel."""

    channel_name: str
    raw: np.ndarray  # resampled but unfiltered (= SignalData.signal)
    filtered: np.ndarray  # after bandpass
    smoothed: np.ndarray  # after Hanning smoothing  ← used for analysis
    tapered: np.ndarray  # after Tukey taper (informational)
    time: np.ndarray  # uniform time axis in seconds
    sampling_rate: int
    lowcut: float
    highcut: float

    @property
    def processed(self) -> np.ndarray:
        """Alias: the signal used by all downstream analysis."""
        return self.smoothed


def preprocess(sd: SignalData, config: PPGConfig) -> PreprocessedSignal:
    """
    Apply the three-stage preprocessing pipeline to a single channel.

    Stages
    ------
    1. 4th-order zero-phase Butterworth bandpass  [lowcut – highcut Hz]
    2. Hanning-window convolution smoothing
    3. Tukey tapering (edge-effect suppression)

    Parameters
    ----------
    sd     : SignalData from loader.load_ppg_data()
    config : PPGConfig with all tunable parameters

    Returns
    -------
    PreprocessedSignal – access `.processed` for downstream use
    """
    signal = sd.signal
    sr = sd.sampling_rate
    N = len(signal)

    # ── 1. Bandpass filter ────────────────────────────────────────────────
    nyq = 0.5 * sr
    lo = config.bandpass_lowcut / nyq
    hi = config.bandpass_highcut / nyq
    b, a = butter(config.bandpass_order, [lo, hi], btype="band")
    filtered = filtfilt(b, a, signal)

    # ── 2. Hanning smoothing ──────────────────────────────────────────────
    win = np.hanning(config.smooth_window_len)
    smoothed = np.convolve(filtered, win / win.sum(), mode="same")

    # ── 3. Tukey tapering ─────────────────────────────────────────────────
    tapered = smoothed * windows.tukey(N, alpha=config.taper_alpha)

    return PreprocessedSignal(
        channel_name=sd.channel_name,
        raw=signal,
        filtered=filtered,
        smoothed=smoothed,
        tapered=tapered,
        time=sd.time,
        sampling_rate=sr,
        lowcut=config.bandpass_lowcut,
        highcut=config.bandpass_highcut,
    )


def print_preprocess_summary(pre: PreprocessedSignal) -> None:
    """Print a concise preprocessing report to stdout."""
    snr_check_denom = np.std(pre.processed - pre.smoothed)
    snr_check = (
        np.std(pre.processed) / snr_check_denom if snr_check_denom > 0 else float("inf")
    )
    print(f"\n{'─' * 70}")
    print(f"Preprocessing  [{pre.channel_name}]")
    print(f"{'─' * 70}")
    print(
        f"  Bandpass      : {pre.lowcut}–{pre.highcut} Hz, "
        f"filtered range [{pre.filtered.min():.2f}, {pre.filtered.max():.2f}]"
    )
    print(f"  Processed std : {np.std(pre.processed):.2f}")
    print(f"  Smoothing SNR : {snr_check:.2f}")
