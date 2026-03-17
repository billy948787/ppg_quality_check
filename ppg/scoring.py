"""
Composite SQI Scoring
======================
Weighted piecewise-linear combination of individual metric sub-scores,
plus spectral SNR / beat regularity / clipping rate computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import welch


# ─────────────────────────────────────────────────────────────────────────────
# Helper scoring functions
# ─────────────────────────────────────────────────────────────────────────────


def _pw_score(value, x_pts, y_pts) -> float:
    """Piecewise-linear mapping of a metric value to 0–100."""
    if value is None:
        return float("nan")
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if np.isnan(v):
        return float("nan")
    return float(np.clip(np.interp(v, x_pts, y_pts), 0.0, 100.0))


def _range_score(
    value, opt_lo: float, opt_hi: float, zero_lo: float, zero_hi: float
) -> float:
    """
    Score = 100 within [opt_lo, opt_hi]; linearly decays to 0 at the outer boundaries.
    """
    if value is None:
        return float("nan")
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if np.isnan(v):
        return float("nan")
    if opt_lo <= v <= opt_hi:
        return 100.0
    elif v < opt_lo:
        span = opt_lo - zero_lo
        return float(np.clip((v - zero_lo) / (span + 1e-9) * 100, 0.0, 100.0))
    else:
        span = zero_hi - opt_hi
        return float(np.clip((zero_hi - v) / (span + 1e-9) * 100, 0.0, 100.0))


def _letter_grade(score: float) -> Tuple[str, str]:
    """Map a 0–100 score to (letter_grade, label)."""
    if score >= 85:
        return "A", "Excellent"
    elif score >= 70:
        return "B", "Good"
    elif score >= 55:
        return "C", "Fair"
    elif score >= 40:
        return "D", "Marginal"
    return "F", "Poor"


# ─────────────────────────────────────────────────────────────────────────────
# Signal-level quality metrics
# ─────────────────────────────────────────────────────────────────────────────


def compute_spectral_snr(
    processed: np.ndarray,
    sampling_rate: int,
    ppg_band: Tuple[float, float] = (0.5, 4.0),
) -> Tuple[float, float]:
    """
    Compute spectral SNR and baseline wander index from Welch PSD.

    Returns
    -------
    (spectral_snr, baseline_wander_index)
      spectral_snr          – fraction of total PSD power in the PPG cardiac band
      baseline_wander_index – fraction of power below ppg_band[0] (< 0.5 Hz)
    """
    N = len(processed)
    freqs, psd = welch(processed, fs=sampling_rate, nperseg=min(512, N // 2))
    total_pwr = float(np.trapz(psd, freqs))
    if total_pwr == 0:
        return 0.0, 1.0

    ppg_mask = (freqs >= ppg_band[0]) & (freqs <= ppg_band[1])
    ppg_pwr = (
        float(np.trapz(psd[ppg_mask], freqs[ppg_mask])) if np.any(ppg_mask) else 0.0
    )
    spectral_snr = ppg_pwr / total_pwr

    bw_mask = freqs < ppg_band[0]
    bw_pwr = float(np.trapz(psd[bw_mask], freqs[bw_mask])) if np.any(bw_mask) else 0.0
    baseline_wander = bw_pwr / total_pwr

    return spectral_snr, baseline_wander


def compute_beat_regularity(rr_intervals: np.ndarray) -> float:
    """
    Beat regularity = 1 – CV_RR, clipped to [0, 1].

    1.0 = perfectly regular; 0.0 = highly irregular.
    Returns nan if fewer than 3 intervals.
    """
    if len(rr_intervals) < 3:
        return float("nan")
    cv_rr = np.std(rr_intervals) / (np.mean(rr_intervals) + 1e-8)
    return float(np.clip(1.0 - cv_rr, 0.0, 1.0))


def compute_clipping_rate(raw: np.ndarray, threshold_pct: float = 0.005) -> float:
    """
    Fraction of raw samples within `threshold_pct` of the ADC saturation boundary.

    Returns 1.0 if the signal is constant (fully saturated).
    """
    sig_range = raw.max() - raw.min()
    if sig_range == 0:
        return 1.0
    clip_thr = sig_range * threshold_pct
    clipped = (raw >= raw.max() - clip_thr) | (raw <= raw.min() + clip_thr)
    return float(np.mean(clipped))


# ─────────────────────────────────────────────────────────────────────────────
# Composite score
# ─────────────────────────────────────────────────────────────────────────────

SubScore = Tuple[str, float, float]  # (name, weight, score_0_100)


@dataclass
class CompositeScore:
    """Weighted composite SQI score for one channel."""

    score: float  # 0 – 100
    grade: str  # A / B / C / D / F
    label: str  # Excellent / Good / Fair / Marginal / Poor
    sub_scores: List[SubScore] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.score:.1f}/100  [Grade {self.grade} – {self.label}]"


def compute_composite_score(
    standard: Dict,
    rpeak: Dict,
    spectral_snr: float,
    beat_template_corr: float,
    beat_regularity: float,
) -> CompositeScore:
    """
    Build the composite SQI score from individual metric sub-scores.

    Weights sum to 1.0.  Metrics with nan sub-scores are excluded from the
    weighted average (weight is redistributed proportionally).
    """
    sub_scores: List[SubScore] = [
        (
            "SNR",
            0.18,
            _pw_score(standard.get("SNR"), [0, 1, 2, 5, 10], [0, 20, 50, 80, 100]),
        ),
        (
            "MSQ",
            0.16,
            _pw_score(
                rpeak.get("MSQ (Multi-Detector)"),
                [0, 0.27, 0.50, 0.80, 1.0],
                [0, 25, 55, 85, 100],
            ),
        ),
        (
            "Spectral SNR",
            0.14,
            _pw_score(
                spectral_snr,
                [0, 0.3, 0.5, 0.7, 0.9, 1.0],
                [0, 20, 40, 65, 90, 100],
            ),
        ),
        (
            "Ectopic Ratio",
            0.12,
            _pw_score(
                rpeak.get("Ectopic Ratio (Malik)"),
                [0, 0.05, 0.10, 0.30, 1.0],
                [100, 80, 50, 5, 0],
            ),
        ),
        (
            "Perfusion",
            0.10,
            _pw_score(
                standard.get("Perfusion Index (%)"),
                [0, 0.05, 0.5, 2.0, 5.0],
                [0, 15, 50, 80, 100],
            ),
        ),
        (
            "Template Corr",
            0.10,
            _pw_score(
                beat_template_corr,
                [0, 0.5, 0.75, 0.90, 1.0],
                [0, 30, 60, 85, 100],
            ),
        ),
        (
            "Beat Regularity",
            0.08,
            _pw_score(
                beat_regularity,
                [0, 0.5, 0.75, 0.90, 1.0],
                [0, 30, 60, 85, 100],
            ),
        ),
        (
            "Kurtosis",
            0.06,
            _range_score(standard.get("Kurtosis"), -1.25, 1.17, -4.0, 4.0),
        ),
        (
            "Skewness",
            0.06,
            _range_score(standard.get("Skewness"), -0.26, 0.87, -3.0, 3.0),
        ),
    ]

    wgt_sum = sum(w * s for _, w, s in sub_scores if not np.isnan(s))
    wgt_tot = sum(w for _, w, s in sub_scores if not np.isnan(s))
    score = wgt_sum / wgt_tot if wgt_tot > 0 else 0.0
    grade, label = _letter_grade(score)

    return CompositeScore(score=score, grade=grade, label=label, sub_scores=sub_scores)
