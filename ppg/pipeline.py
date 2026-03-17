"""
Single-Channel Analysis Pipeline
==================================
Orchestrates the full analysis for one PPG channel/band and collects
all results into a ChannelResult dataclass.

Entry point: run_channel_analysis(signal_data, config) → ChannelResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import PPGConfig
from .loader import SignalData
from .peaks import PeakDetectionResult, detect_peaks, print_peak_summary
from .preprocessing import PreprocessedSignal, preprocess, print_preprocess_summary
from .scoring import (
    CompositeScore,
    compute_beat_regularity,
    compute_clipping_rate,
    compute_composite_score,
    compute_spectral_snr,
)
from .segments import SegmentResult, analyze_segments
from .rules import build_rulesets, evaluate_quality, make_sqi_dict
from .features import (
    compute_standard_sqi,
    compute_hrv_time,
    compute_hr_stats,
    compute_hrv_freq,
    compute_poincare,
    compute_hrv_full,
    compute_dtw,
    compute_waveform_energy,
    compute_beat_template_corr,
    compute_rpeak_sqi,
)


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ChannelResult:
    """All analysis outputs for one PPG channel / wavelength band."""

    channel_name: str
    signal_data: SignalData
    preprocessed: PreprocessedSignal
    peaks: PeakDetectionResult

    # ── Feature dicts ─────────────────────────────────────────────────────
    standard_sqi: Dict[str, float] = field(default_factory=dict)
    hrv_time: Dict[str, float] = field(default_factory=dict)
    hr_stats: Dict[str, float] = field(default_factory=dict)
    hrv_freq: Dict[str, float] = field(default_factory=dict)
    poincare: Dict[str, float] = field(default_factory=dict)
    dtw: Dict[str, float] = field(default_factory=dict)
    rpeak_sqi: Dict[str, Any] = field(default_factory=dict)
    waveform_energy: Dict[str, float] = field(default_factory=dict)
    hrv_full: Tuple[Dict, Dict, Dict, Dict] = field(
        default_factory=lambda: ({}, {}, {}, {})
    )

    # ── Signal-level quality metrics ──────────────────────────────────────
    spectral_snr: float = float("nan")
    baseline_wander_idx: float = float("nan")
    beat_template_corr: float = float("nan")
    beat_template_n_beats: int = 0
    beat_regularity: float = float("nan")
    clipping_rate: float = float("nan")

    # ── Composite score ───────────────────────────────────────────────────
    composite: Optional[CompositeScore] = None

    # ── Rule-based label ──────────────────────────────────────────────────
    quality_label: str = "POOR"

    # ── Segment analysis ──────────────────────────────────────────────────
    segments: List[SegmentResult] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline function
# ─────────────────────────────────────────────────────────────────────────────


def run_channel_analysis(sd: SignalData, config: PPGConfig) -> ChannelResult:
    """
    Run the full PPG quality analysis pipeline for a single channel.

    Steps
    -----
    1. Preprocess (bandpass → smooth → taper)
    2. Detect peaks / valleys
    3. Compute all enabled feature groups
    4. Compute signal-level quality metrics
    5. Compute composite SQI score
    6. Rule-based quality label
    7. Segment-level analysis

    Parameters
    ----------
    sd     : SignalData from loader.load_ppg_data()
    config : PPGConfig

    Returns
    -------
    ChannelResult with all results populated
    """
    # ── 1. Preprocess ────────────────────────────────────────────────────
    pre = preprocess(sd, config)

    # ── 2. Peak detection ────────────────────────────────────────────────
    peak_result = detect_peaks(pre, config)
    nn = peak_result.nn_intervals

    result = ChannelResult(
        channel_name=sd.channel_name,
        signal_data=sd,
        preprocessed=pre,
        peaks=peak_result,
    )

    # ── 3. Features ──────────────────────────────────────────────────────
    if config.compute_standard:
        result.standard_sqi = compute_standard_sqi(pre.processed, pre.raw)

    if config.compute_hrv_time:
        result.hrv_time = compute_hrv_time(nn)

    result.hr_stats = compute_hr_stats(nn)  # always compute (used by rules)

    if config.compute_hrv_freq:
        result.hrv_freq = compute_hrv_freq(nn)

    if config.compute_poincare:
        result.poincare = compute_poincare(nn)

    if config.compute_dtw:
        result.dtw = compute_dtw(pre.processed, peak_result.valleys, peak_result.peaks)

    if config.compute_rpeaks:
        result.rpeak_sqi = compute_rpeak_sqi(pre.processed, pre.sampling_rate)

    if config.compute_waveform:
        result.waveform_energy = compute_waveform_energy(
            pre.processed, pre.sampling_rate, config.waveform_bands
        )

    if config.compute_hrv_full:
        result.hrv_full = compute_hrv_full(pre.processed, pre.sampling_rate)

    # ── 4. Signal-level quality metrics ──────────────────────────────────
    result.spectral_snr, result.baseline_wander_idx = compute_spectral_snr(
        pre.processed, pre.sampling_rate
    )
    result.beat_template_corr, result.beat_template_n_beats = (
        compute_beat_template_corr(pre.processed, peak_result.valleys)
    )
    result.beat_regularity = compute_beat_regularity(peak_result.rr_intervals)
    result.clipping_rate = compute_clipping_rate(pre.raw)

    # ── 5. Composite SQI ─────────────────────────────────────────────────
    result.composite = compute_composite_score(
        standard=result.standard_sqi,
        rpeak=result.rpeak_sqi,
        spectral_snr=result.spectral_snr,
        beat_template_corr=result.beat_template_corr,
        beat_regularity=result.beat_regularity,
    )

    # ── 6. Rule-based label ───────────────────────────────────────────────
    strict_rs, loose_rs = build_rulesets(result.rpeak_sqi, result.hr_stats)
    sqi_dict = make_sqi_dict(result.standard_sqi, result.rpeak_sqi, result.hr_stats)
    result.quality_label = evaluate_quality(sqi_dict, strict_rs, loose_rs)

    # ── 7. Segment analysis ───────────────────────────────────────────────
    result.segments = analyze_segments(
        pre.processed, pre.raw, pre.sampling_rate, config
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Console reporting
# ─────────────────────────────────────────────────────────────────────────────


def print_channel_report(r: ChannelResult) -> None:
    """Print a comprehensive text report to stdout."""
    ch = r.channel_name
    bar = "─" * 70

    # ── Preprocessing ─────────────────────────────────────────────────────
    print_preprocess_summary(r.preprocessed)

    # ── Peaks ─────────────────────────────────────────────────────────────
    print_peak_summary(r.peaks)

    # ── Standard SQI ─────────────────────────────────────────────────────
    if r.standard_sqi:
        print(f"\n{bar}\nStandard SQI Metrics  [{ch}]\n{bar}")
        for k, v in r.standard_sqi.items():
            print(f"  {k:30s}: {v:.6f}")

    # ── HRV Time ─────────────────────────────────────────────────────────
    if r.hrv_time:
        print(f"\n{bar}\nHRV Time Domain  [{ch}]\n{bar}")
        for k, v in r.hrv_time.items():
            print(f"  {k:30s}: {v:.4f}")

    # ── HR Stats ─────────────────────────────────────────────────────────
    if r.hr_stats:
        print(f"\n{bar}\nHeart Rate Statistics  [{ch}]\n{bar}")
        for k, v in r.hr_stats.items():
            print(f"  {k:30s}: {v:.2f}")

    # ── HRV Frequency ────────────────────────────────────────────────────
    if r.hrv_freq:
        print(f"\n{bar}\nHRV Frequency Domain  [{ch}]\n{bar}")
        for k, v in r.hrv_freq.items():
            if k.startswith("_"):
                continue
            print(f"  {k:30s}: {v:.6f}")

    # ── Poincaré ─────────────────────────────────────────────────────────
    if r.poincare:
        print(f"\n{bar}\nPoincaré Features  [{ch}]\n{bar}")
        for k, v in r.poincare.items():
            print(f"  {k:40s}: {v:.6f}")

    # ── DTW ──────────────────────────────────────────────────────────────
    if r.dtw:
        print(f"\n{bar}\nDTW Template Matching  [{ch}]\n{bar}")
        for k, v in r.dtw.items():
            print(f"  {k:40s}: {v:.6f}")

    # ── R-peak SQI ────────────────────────────────────────────────────────
    print(f"\n{bar}\nR-Peak SQI  [{ch}]\n{bar}")
    ectopic = r.rpeak_sqi.get("Ectopic Ratio (Malik)", float("nan"))
    msq = r.rpeak_sqi.get("MSQ (Multi-Detector)", float("nan"))
    corr = r.rpeak_sqi.get("Correlogram", [])
    print(f"  {'Ectopic Ratio (Malik)':30s}: {ectopic:.6f}")
    print(f"  {'MSQ (Multi-Detector)':30s}: {msq:.6f}")
    if len(corr) >= 6:
        print(f"  Correlogram idx : {corr[:3]}")
        print(f"  Correlogram val : {corr[3:]}")

    # ── Waveform Energy ───────────────────────────────────────────────────
    if r.waveform_energy:
        print(f"\n{bar}\nWaveform Energy  [{ch}]\n{bar}")
        for k, v in r.waveform_energy.items():
            print(f"  {k:40s}: {v:.6f}")

    # ── Full HRV ─────────────────────────────────────────────────────────
    t_feats, f_feats, g_feats, csi_feats = r.hrv_full
    if any([t_feats, f_feats, g_feats, csi_feats]):
        print(f"\n{bar}\nFull HRV Analysis (hrvanalysis)  [{ch}]\n{bar}")
        for label, feats in [
            ("Time Domain", t_feats),
            ("Frequency Domain", f_feats),
            ("Geometrical", g_feats),
            ("CSI/CVI", csi_feats),
        ]:
            if feats:
                print(f"\n  [{label}]")
                for k, v in feats.items():
                    print(f"    {k:35s}: {v}")

    # ── Additional quality ────────────────────────────────────────────────
    print(f"\n{bar}\nAdditional Quality Metrics  [{ch}]\n{bar}")
    print(f"  {'Spectral SNR (PPG / total)':42s}: {r.spectral_snr:.4f}")
    print(f"  {'Baseline Wander Index':42s}: {r.baseline_wander_idx:.4f}")
    tc = (
        f"{r.beat_template_corr:.4f} ({r.beat_template_n_beats} beats)"
        if not np.isnan(r.beat_template_corr)
        else "N/A"
    )
    print(f"  {'Beat Template Correlation':42s}: {tc}")
    br = f"{r.beat_regularity:.4f}" if not np.isnan(r.beat_regularity) else "N/A"
    print(f"  {'Beat Regularity (1 – CV_RR)':42s}: {br}")
    print(f"  {'Clipping / Saturation Rate':42s}: {r.clipping_rate:.4f}")

    # ── Composite score ───────────────────────────────────────────────────
    if r.composite:
        c = r.composite
        print(f"\n{bar}\nComposite SQI Score  [{ch}]\n{bar}")
        print(f"\n  {'Metric':<22} {'Wt':>5}  {'Score/100':>9}  Status")
        print("  " + "─" * 52)
        for name, wt, score in c.sub_scores:
            if np.isnan(score):
                print(f"  {name:<22} {wt:>4.0%}  {'N/A':>9}  —")
            else:
                status = "PASS" if score >= 70 else "WARN" if score >= 40 else "FAIL"
                print(f"  {name:<22} {wt:>4.0%}  {score:>8.1f}  {status}")
        print("  " + "═" * 52)
        print(
            f"  {'COMPOSITE SQI':<22} {'100%':>5}  {c.score:>8.1f}  "
            f"Grade {c.grade} ({c.label})"
        )

    # ── Segment summary ───────────────────────────────────────────────────
    if r.segments:
        print(f"\n{bar}\nSegment Analysis  [{ch}]\n{bar}")
        for seg in r.segments:
            hr_str = f"{seg.hr_mean:.0f} bpm" if not np.isnan(seg.hr_mean) else "N/A"
            rmssd_str = f"{seg.rmssd:.1f} ms" if not np.isnan(seg.rmssd) else "N/A"
            print(
                f"  Seg {seg.segment_idx:2d} [{seg.start_s:.0f}–{seg.end_s:.0f}s]  "
                f"Score={seg.composite_score:5.1f}  {seg.quality_label}  "
                f"HR={hr_str}  RMSSD={rmssd_str}"
            )
        good = sum(1 for s in r.segments if s.quality_label == "GOOD")
        fair = sum(1 for s in r.segments if s.quality_label == "FAIR")
        poor = sum(1 for s in r.segments if s.quality_label == "POOR")
        total = len(r.segments)
        print(
            f"\n  {good} GOOD / {fair} FAIR / {poor} POOR  "
            f"– usable: {good + fair}/{total} "
            f"({100 * (good + fair) / max(total, 1):.0f}%)"
        )

    # ── Final summary box ─────────────────────────────────────────────────
    c = r.composite
    tc_str = (
        f"{r.beat_template_corr:.3f}" if not np.isnan(r.beat_template_corr) else " N/A"
    )
    br_str = f"{r.beat_regularity:.3f}" if not np.isnan(r.beat_regularity) else " N/A"
    bw = 68
    print("\n" + "╔" + "═" * bw + "╗")
    print("║" + f"  FINAL QUALITY ASSESSMENT – {ch}  ".center(bw) + "║")
    print("╠" + "═" * bw + "╣")
    lines = [
        f"Composite SQI        : {c.score:6.1f} / 100",
        f"Grade                : {c.grade} – {c.label}",
        f"Rule-Based Label     : {r.quality_label}",
        f"Signal Duration      : {r.signal_data.duration_s:6.1f} s  ({len(r.segments)} segments)",
        f"Spectral SNR         : {r.spectral_snr:.3f}   Baseline Wander : {r.baseline_wander_idx:.3f}",
        f"Template Corr        : {tc_str}   Beat Regularity : {br_str}",
        f"Clipping Rate        : {r.clipping_rate:.4f}",
    ]
    for line in lines:
        print("║  " + line.ljust(bw - 2) + "║")
    print("╚" + "═" * bw + "╝")
