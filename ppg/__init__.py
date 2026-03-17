"""
ppg – PPG Signal Quality Analysis Package
==========================================
A modular pipeline for photoplethysmography (PPG) signal quality assessment
supporting multiple wavelength channels (multi-band analysis).

Quick start
-----------
>>> from ppg import PPGConfig, load_ppg_data, run_channel_analysis, generate_all_plots
>>> config = PPGConfig(file_path="s17_run.csv")
>>> channels = load_ppg_data(config)
>>> results = {name: run_channel_analysis(sd, config) for name, sd in channels.items()}
>>> generate_all_plots(results, config)
"""

from .config import BandDef, PPGConfig
from .loader import SignalData, load_ppg_data, print_load_summary
from .preprocessing import PreprocessedSignal, preprocess
from .peaks import PeakDetectionResult, detect_peaks
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
from .scoring import CompositeScore, compute_composite_score
from .rules import build_rulesets, evaluate_quality, make_sqi_dict
from .segments import SegmentResult, analyze_segments
from .pipeline import ChannelResult, run_channel_analysis, print_channel_report
from .visualization import (
    plot_preprocessing,
    plot_analysis_report,
    plot_multiband_comparison,
    generate_all_plots,
)

__all__ = [
    # config
    "BandDef",
    "PPGConfig",
    # loader
    "SignalData",
    "load_ppg_data",
    "print_load_summary",
    # preprocessing
    "PreprocessedSignal",
    "preprocess",
    # peaks
    "PeakDetectionResult",
    "detect_peaks",
    # features
    "compute_standard_sqi",
    "compute_hrv_time",
    "compute_hr_stats",
    "compute_hrv_freq",
    "compute_poincare",
    "compute_hrv_full",
    "compute_dtw",
    "compute_waveform_energy",
    "compute_beat_template_corr",
    "compute_rpeak_sqi",
    # scoring
    "CompositeScore",
    "compute_composite_score",
    # rules
    "build_rulesets",
    "evaluate_quality",
    "make_sqi_dict",
    # segments
    "SegmentResult",
    "analyze_segments",
    # pipeline
    "ChannelResult",
    "run_channel_analysis",
    "print_channel_report",
    # visualization
    "plot_preprocessing",
    "plot_analysis_report",
    "plot_multiband_comparison",
    "generate_all_plots",
]
