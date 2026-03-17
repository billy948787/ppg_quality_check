"""
PPG Peak / Valley Detection
============================
Uses scipy find_peaks on the processed signal.
Valleys (inverted peaks) are used as fiducial points for RR/NN intervals
because they are more stable across PPG morphologies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

from .config import PPGConfig
from .preprocessing import PreprocessedSignal


@dataclass
class PeakDetectionResult:
    """Detected peaks and derived interval series for one channel."""

    channel_name: str

    # Indices into the processed signal array
    valleys: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    peaks: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    # RR / NN intervals in milliseconds
    rr_intervals: np.ndarray = field(default_factory=lambda: np.array([]))
    nn_intervals: np.ndarray = field(default_factory=lambda: np.array([]))

    sampling_rate: int = 100

    @property
    def has_enough_peaks(self) -> bool:
        return len(self.valleys) > 2

    @property
    def has_enough_for_freq(self) -> bool:
        return len(self.valleys) > 10


def detect_peaks(pre: PreprocessedSignal, config: PPGConfig) -> PeakDetectionResult:
    """
    Detect valleys and peaks in the processed PPG signal.

    Valley detection (inverted signal) is the primary fiducial because PPG
    valleys are sharper and more consistent than systolic peaks.

    Returns
    -------
    PeakDetectionResult with .valleys, .peaks, .rr_intervals, .nn_intervals
    """
    signal = pre.processed
    sr = pre.sampling_rate
    min_dist = int(sr * config.min_peak_distance_s)
    prominence = np.std(signal) * config.peak_prominence_factor

    valleys, _ = find_peaks(-signal, distance=min_dist, prominence=prominence)
    peaks, _ = find_peaks(signal, distance=min_dist, prominence=prominence)

    rr_intervals: np.ndarray = np.array([])
    if len(valleys) > 1:
        rr_intervals = np.diff(valleys) * (1000.0 / sr)

    return PeakDetectionResult(
        channel_name=pre.channel_name,
        valleys=valleys,
        peaks=peaks,
        rr_intervals=rr_intervals,
        nn_intervals=rr_intervals.copy(),
        sampling_rate=sr,
    )


def print_peak_summary(result: PeakDetectionResult) -> None:
    """Print peak detection statistics."""
    print(f"\n{'─' * 70}")
    print(f"Peak Detection  [{result.channel_name}]")
    print(f"{'─' * 70}")
    print(f"  Valleys (fiducials): {len(result.valleys)}")
    print(f"  Peaks (systolic)   : {len(result.peaks)}")
    if len(result.rr_intervals) > 0:
        print(
            f"  RR intervals : n={len(result.rr_intervals)}  "
            f"mean={np.mean(result.rr_intervals):.1f} ms  "
            f"std={np.std(result.rr_intervals):.1f} ms  "
            f"range=[{np.min(result.rr_intervals):.1f}, "
            f"{np.max(result.rr_intervals):.1f}] ms"
        )
    else:
        print("  ⚠ Fewer than 2 valleys detected – RR intervals unavailable")
