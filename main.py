"""
PPG Signal Quality Analysis – Entry Point
==========================================
Slim entry point: configure, load, analyse, report, plot.

Usage
-----
    python main.py [file.csv] [--no-show] [--no-save] [--out-dir DIR]
                   [--fs HZ] [--seg-s SECONDS]

Defaults to 's17_run.csv' when no file is given.
"""

from __future__ import annotations

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PPG Signal Quality Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("file", help="CSV file to analyse")
    p.add_argument(
        "--no-show", action="store_true", help="Do not display plots interactively"
    )
    p.add_argument("--no-save", action="store_true", help="Do not save plot files")
    p.add_argument(
        "--out-dir", default=".", metavar="DIR", help="Directory for saved plots"
    )
    p.add_argument(
        "--fs", type=int, default=100, metavar="HZ", help="Target sampling rate (Hz)"
    )
    p.add_argument(
        "--seg-s", type=int, default=30, metavar="SECONDS", help="Segment duration (s)"
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Imports (deferred so --help is fast) ────────────────────────────────
    from ppg import PPGConfig, load_ppg_data, print_load_summary
    from ppg import run_channel_analysis, print_channel_report
    from ppg import generate_all_plots

    # ── Configuration ────────────────────────────────────────────────────────
    config = PPGConfig(
        file_path=args.file,
        target_sampling_rate=args.fs,
        segment_duration_s=args.seg_s,
        output_dir=args.out_dir,
        save_plots=not args.no_save,
        show_plots=not args.no_show,
    )

    print(f"\n{'=' * 70}")
    print(f"  PPG Quality Analysis  –  {config.file_path}")
    print(f"{'=' * 70}")

    # ── Load ─────────────────────────────────────────────────────────────────
    channels = load_ppg_data(config)
    print_load_summary(channels)

    # ── Analyse each channel ─────────────────────────────────────────────────
    results = {}
    for ch_name, signal_data in channels.items():
        print(f"\n{'─' * 70}")
        print(f"  Analysing channel: {ch_name}")
        print(f"{'─' * 70}")
        result = run_channel_analysis(signal_data, config)
        print_channel_report(result)
        results[ch_name] = result

    # ── Multi-band summary (when > 1 channel) ────────────────────────────────
    if len(results) > 1:
        print(f"\n{'=' * 70}")
        print("  MULTI-BAND SUMMARY")
        print(f"{'=' * 70}")
        print(f"  {'Channel':<20} {'Score':>7}  {'Grade':<12}  {'Label'}")
        print(f"  {'─' * 20}  {'─' * 7}  {'─' * 12}  {'─' * 10}")
        for ch, r in results.items():
            c = r.composite
            score_str = f"{c.score:7.1f}" if c else "    N/A"
            grade_str = f"{c.grade} – {c.label}" if c else "N/A"
            print(f"  {ch:<20} {score_str}  {grade_str:<12}  {r.quality_label}")

    # ── Visualise ────────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  Generating plots...")
    generate_all_plots(results, config)
    print("\nDone.")


if __name__ == "__main__":
    main()
