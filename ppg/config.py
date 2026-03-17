"""
PPG Quality Analysis – Configuration
=====================================
All tunable parameters live here.  Pass a PPGConfig instance to every
pipeline function so that nothing is hard-coded in the analysis modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class BandDef:
    """A named frequency band used for waveform-energy analysis."""

    name: str
    low: Optional[float]  # Hz – None means DC / 0
    high: Optional[float]  # Hz – None means Nyquist

    def as_tuple(self) -> Tuple[Optional[float], Optional[float]]:
        return (self.low, self.high)


# ── Default waveform energy bands ────────────────────────────────────────────
DEFAULT_WAVEFORM_BANDS: List[BandDef] = [
    BandDef("Total", None, None),
    BandDef("LF (0–0.5 Hz)", 0.0, 0.5),
    BandDef("PPG (0.5–4 Hz)", 0.5, 4.0),
    BandDef("HF (4–8 Hz)", 4.0, 8.0),
]

# ── Known column name patterns (lower-cased substring match) ──────────────────
# Signal columns are matched in priority order; the first hit wins per column.
SIGNAL_COLUMN_PATTERNS: List[str] = [
    "pleth",  # pleth_1, pleth_2, pleth_3 …
    "ppg",  # ppg_value, ppg_1 …
    "ir",  # ir channel
    "red",  # red channel
    "green",  # green channel
    "nir",  # near-IR
]

TIMESTAMP_COLUMN_NAMES: List[str] = [
    "time",
    "timestamp",
    "sensor_timestamp",
    "ts",
    "datetime",
    "t",
]


@dataclass
class PPGConfig:
    """
    Central configuration for the PPG quality analysis pipeline.

    Usage example
    -------------
    >>> cfg = PPGConfig(file_path="s17_run.csv")
    >>> cfg = PPGConfig(
    ...     file_path="raw.csv",
    ...     signal_columns=["pleth_1", "pleth_2", "pleth_3"],
    ...     timestamp_column="sensor_timestamp",
    ...     target_sampling_rate=100,
    ...     segment_duration_s=30,
    ... )
    """

    # ── Input ─────────────────────────────────────────────────────────────────
    file_path: str = "s17_run.csv"

    # Signal / timestamp column override (None → auto-detect)
    signal_columns: Optional[List[str]] = None
    timestamp_column: Optional[str] = None

    # ── Resampling ────────────────────────────────────────────────────────────
    target_sampling_rate: int = 100  # Hz – uniform grid after interpolation

    # ── Preprocessing ─────────────────────────────────────────────────────────
    bandpass_lowcut: float = 0.5  # Hz
    bandpass_highcut: float = 5.0  # Hz
    bandpass_order: int = 4
    smooth_window_len: int = 5  # samples (Hanning convolution)
    taper_alpha: float = 0.1  # Tukey window parameter

    # ── Peak detection ────────────────────────────────────────────────────────
    min_peak_distance_s: float = 0.5  # seconds
    peak_prominence_factor: float = 0.75  # × std(signal)

    # ── Features to compute ───────────────────────────────────────────────────
    compute_standard: bool = True
    compute_hrv_time: bool = True
    compute_hrv_freq: bool = True
    compute_poincare: bool = True
    compute_dtw: bool = True
    compute_rpeaks: bool = True
    compute_waveform: bool = True
    compute_hrv_full: bool = True

    # Waveform energy bands – override to customise
    waveform_bands: List[BandDef] = field(
        default_factory=lambda: list(DEFAULT_WAVEFORM_BANDS)
    )

    # ── Segment analysis ──────────────────────────────────────────────────────
    segment_duration_s: int = 30  # seconds per segment

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: str = "."
    save_plots: bool = True
    show_plots: bool = True
