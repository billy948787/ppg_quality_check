"""
Segment-Level SQI Analysis
============================
Splits the signal into fixed-duration windows and computes per-window
SQI metrics + composite score, returning structured SegmentResult objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.signal import find_peaks

from vital_sqi.sqi.hrv_sqi import hr_mean_sqi, rmssd_sqi, sdnn_sqi
from vital_sqi.sqi.standard_sqi import (
    entropy_sqi,
    kurtosis_sqi,
    mean_crossing_rate_sqi,
    perfusion_sqi,
    signal_to_noise_sqi,
    skewness_sqi,
    zero_crossings_rate_sqi,
)

from .config import PPGConfig
from .rules import build_rulesets, evaluate_quality, make_sqi_dict
from .scoring import _pw_score, _range_score


@dataclass
class SegmentResult:
    """SQI metrics and quality label for one time segment."""

    segment_idx: int  # 1-based
    start_s: float
    end_s: float

    # Standard SQI
    kurtosis: float
    skewness: float
    entropy: float
    snr: float
    zcr: float
    mcr: float
    perfusion: float

    # Peak-derived (may be nan if insufficient data)
    n_valleys: int
    hr_mean: float
    rmssd: float
    sdnn: float

    # Quality assessment
    composite_score: float
    quality_label: str  # GOOD / FAIR / POOR


def analyze_segments(
    processed: np.ndarray,
    raw: np.ndarray,
    sampling_rate: int,
    config: PPGConfig,
) -> List[SegmentResult]:
    """
    Split the signal into `config.segment_duration_s`-second windows and
    compute per-segment SQI metrics.

    Parameters
    ----------
    processed     : smoothed PPG signal (used for most SQIs)
    raw           : resampled but unfiltered signal (for SNR / perfusion)
    sampling_rate : in Hz
    config        : PPGConfig with segment_duration_s and peak detection params

    Returns
    -------
    list[SegmentResult]  – one per window (last window may be shorter)
    """
    N = len(processed)
    seg_size = config.segment_duration_s * sampling_rate

    if seg_size >= N:
        seg_size = N

    n_segments = int(np.ceil(N / seg_size))
    min_dist = int(sampling_rate * config.min_peak_distance_s)

    results: List[SegmentResult] = []

    for i in range(n_segments):
        start = i * seg_size
        end = min(start + seg_size, N)
        seg = processed[start:end]
        raw_seg = raw[start:end]

        # ── Standard SQI ──────────────────────────────────────────────────
        kurt = float(kurtosis_sqi(seg))
        skew = float(skewness_sqi(seg))
        ent = float(entropy_sqi(seg))
        snr = float(signal_to_noise_sqi(raw_seg))
        zcr = float(zero_crossings_rate_sqi(seg))
        mcr = float(mean_crossing_rate_sqi(seg))
        perf = float(perfusion_sqi(raw_seg, seg))

        # ── Peak-based HRV ────────────────────────────────────────────────
        n_valleys = 0
        hr_mean = float("nan")
        rmssd_val = float("nan")
        sdnn_val = float("nan")
        try:
            prom = np.std(seg) * 0.5
            seg_valleys, _ = find_peaks(-seg, distance=min_dist, prominence=prom)
            n_valleys = len(seg_valleys)
            if n_valleys > 2:
                rr = np.diff(seg_valleys) * (1000.0 / sampling_rate)
                hr_mean = float(hr_mean_sqi(rr))
                rmssd_val = float(rmssd_sqi(rr))
                sdnn_val = float(sdnn_sqi(rr))
        except Exception:
            pass

        # ── Per-segment composite score ────────────────────────────────────
        sub = [
            ("SNR", 0.30, _pw_score(snr, [0, 1, 2, 5, 10], [0, 20, 50, 80, 100])),
            (
                "Perfusion",
                0.20,
                _pw_score(perf, [0, 0.05, 0.5, 2, 5], [0, 15, 50, 80, 100]),
            ),
            ("Kurtosis", 0.15, _range_score(kurt, -1.25, 1.17, -4, 4)),
            ("Skewness", 0.15, _range_score(skew, -0.26, 0.87, -3, 3)),
            ("ZCR", 0.10, _range_score(zcr, 0.03, 0.07, 0, 0.15)),
            ("MCR", 0.10, _range_score(mcr, 0.02, 0.07, 0, 0.15)),
        ]
        w_sum = sum(w * s for _, w, s in sub if not np.isnan(s))
        w_tot = sum(w for _, w, s in sub if not np.isnan(s))
        composite = w_sum / w_tot if w_tot > 0 else 0.0

        # ── Rule-based label ──────────────────────────────────────────────
        sqi_dict = {
            "Skewness": skew,
            "Kurtosis": kurt,
            "SNR": snr,
            "Perfusion": perf,
            "Zero-Crossing-Rate": zcr,
            "Mean-Crossing-Rate": mcr,
        }
        # Segment-level rules use only the six base metrics
        from vital_sqi.rule import Rule, RuleSet

        def _rng(name, lo, hi):
            r = Rule(name)
            r.update_def(
                ["<", ">", ">=", "<"],
                [lo, lo, hi, hi],
                ["reject", "accept", "reject", "accept"],
            )
            return r

        def _gt(name, thr):
            r = Rule(name)
            r.update_def(["<", ">"], [thr, thr], ["reject", "accept"])
            return r

        seg_strict = RuleSet(
            {
                1: _rng("Skewness", -0.26, 0.87),
                2: _rng("Kurtosis", -1.25, 1.17),
                3: _gt("SNR", 2.0),
                4: _gt("Perfusion", 0.5),
                5: _rng("Zero-Crossing-Rate", 0.03, 0.07),
                6: _rng("Mean-Crossing-Rate", 0.02, 0.07),
            }
        )
        seg_loose = RuleSet(
            {
                1: _rng("Skewness", -1.0, 2.0),
                2: _rng("Kurtosis", -2.0, 3.0),
                3: _gt("SNR", 0.0),
                4: _gt("Perfusion", 0.05),
                5: _rng("Zero-Crossing-Rate", 0.01, 0.15),
                6: _rng("Mean-Crossing-Rate", 0.01, 0.15),
            }
        )
        import pandas as pd

        seg_df = pd.DataFrame([sqi_dict])
        quality = "POOR"
        try:
            quality = "GOOD" if seg_strict.execute(seg_df) == "accept" else quality
        except Exception:
            pass
        if quality == "POOR":
            try:
                quality = "FAIR" if seg_loose.execute(seg_df) == "accept" else quality
            except Exception:
                pass

        results.append(
            SegmentResult(
                segment_idx=i + 1,
                start_s=start / sampling_rate,
                end_s=end / sampling_rate,
                kurtosis=kurt,
                skewness=skew,
                entropy=ent,
                snr=snr,
                zcr=zcr,
                mcr=mcr,
                perfusion=perf,
                n_valleys=n_valleys,
                hr_mean=hr_mean,
                rmssd=rmssd_val,
                sdnn=sdnn_val,
                composite_score=composite,
                quality_label=quality,
            )
        )

    return results
