"""Run manuscript analyses from the public-only data bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from static.analysis import correlation_analysis, descriptive_statistics, hierarchical_regression, paper_tables
from static.preprocessing.constants import get_public_file
from static.preprocessing.public_validate import get_common_public_ids, validate_public_bundle


def _features_ready() -> bool:
    return get_public_file("features").exists()


def _qc_sample_size() -> int:
    return int(len(get_common_public_ids(validate=True)))


def _safe_run(step: str, func, *args, **kwargs) -> None:
    try:
        func(*args, **kwargs)
    except Exception as exc:
        print(f"[WARN] {step} failed: {exc}")


def _remove_nonpaper_outputs() -> None:
    """
    Remove helper/diagnostic artifacts not reported in manuscript text/tables.
    """
    paths: list[Path] = [
        Path("outputs/stats/core/correlation_ci.csv"),
        Path("outputs/stats/core/correlation_heatmap.png"),
        Path("outputs/stats/core/hierarchical_summary.txt"),
        Path("outputs/stats/core/model_comparison.csv"),
        Path("outputs/stats/core/ucla_correlations_detailed.csv"),
        Path("outputs/stats/core/table1_categorical.csv"),
        Path("outputs/stats/core/table1_descriptives.csv"),
        Path("outputs/stats/core/table1_descriptives_by_gender.csv"),
        Path("outputs/stats/core/table1_gender_comparison.csv"),
        Path("outputs/stats/core/hierarchical_results.csv"),
        Path("outputs/stats/core/correlation_matrix.csv"),
        Path("outputs/stats/core/correlation_pvalues.csv"),
        Path("outputs/stats/supplementary/stroop_interference_quartile_loneliness_extremes25_summary.csv"),
        Path("outputs/stats/supplementary/stroop_interference_quartile_loneliness_extremes25_trend_slopes.csv"),
        Path("outputs/stats/supplementary/stroop_rt_timeseg4_loneliness_extremes25_table.csv"),
        Path("outputs/stats/supplementary/wcst_phase_rt_loneliness_extremes25_alltrials_summary.csv"),
        Path("outputs/stats/supplementary/wcst_phase_rt_loneliness_extremes25_alltrials_slopes.csv"),
        Path("outputs/stats/supplementary/wcst_n212_main_results_extract.csv"),
        Path("outputs/stats/supplementary/wcst_phase_category_weighted_ols.csv"),
        Path("outputs/stats/supplementary/wcst_phase_rt_ols.csv"),
        Path("outputs/stats/supplementary/stroop_threshold_sensitivity.csv"),
        Path("outputs/stats/supplementary/stroop_lmm/stroop_lmm_predictors.csv"),
    ]
    for path in paths:
        if path.exists():
            path.unlink()


def _run_supplementary_extras() -> None:
    from static.figures_tables import plot_stroop_interference_quartile_loneliness_extremes25_trend
    from static.figures_tables import plot_wcst_phase_loneliness_extremes25_trend
    from static.stroop_lmm import run_stroop_trial_lmm
    from static.stroop_supplementary import run_stroop_interference_reliability
    from static.stroop_supplementary import run_stroop_random_slope_variance
    from static.wcst_phase import run_wcst_phase_rt_ols
    from static.wcst_phase import run_wcst_phase_split_half_reliability

    _safe_run("wcst_phase_rt_ols_alltrials", run_wcst_phase_rt_ols.main, 3, True, False, False)
    _safe_run("wcst_phase_pre_exploit_rt_ols_alltrials", run_wcst_phase_rt_ols.main, 3, True, False, True)
    _safe_run("wcst_phase_pre_exploit_rt_ols_alltrials_m2", run_wcst_phase_rt_ols.main, 2, True, False, True)
    _safe_run("wcst_phase_pre_exploit_rt_ols_alltrials_m4", run_wcst_phase_rt_ols.main, 4, True, False, True)

    _safe_run("wcst_phase_split_half_reliability", run_wcst_phase_split_half_reliability.main, 3, False)
    _safe_run("wcst_phase_split_half_reliability_m2", run_wcst_phase_split_half_reliability.main, 2, False)
    _safe_run("wcst_phase_split_half_reliability_m4", run_wcst_phase_split_half_reliability.main, 4, False)

    _safe_run("stroop_trial_lmm", run_stroop_trial_lmm.main)
    _safe_run("stroop_random_slope_variance", run_stroop_random_slope_variance.main)
    _safe_run("stroop_interference_reliability", run_stroop_interference_reliability.main)

    _safe_run(
        "plot_stroop_interference_quartile_loneliness_extremes25_trend",
        plot_stroop_interference_quartile_loneliness_extremes25_trend.main,
    )
    _safe_run(
        "plot_wcst_phase_loneliness_extremes25_trend",
        plot_wcst_phase_loneliness_extremes25_trend.main,
        3,
        True,
    )


def main(
    run_analysis: bool,
    expected_n: int = 212,
    allow_n_mismatch: bool = False,
    skip_preprocess: bool = False,
) -> None:
    if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if skip_preprocess:
        print("[WARN] --skip-preprocess is deprecated in public-only mode and has no effect.")

    validate_public_bundle(raise_on_error=True)

    if not _features_ready():
        raise RuntimeError(
            "Missing required file: data/public/features_public.csv. "
            "Public-only pipeline requires all five public CSV files."
        )

    if run_analysis:
        n_qc = _qc_sample_size()
        if expected_n > 0 and n_qc != expected_n and not allow_n_mismatch:
            raise RuntimeError(
                f"QC sample size mismatch: expected N={expected_n}, found N={n_qc}. "
                "This pipeline is locked to manuscript sample size by default. "
                "Use --allow-n-mismatch to override intentionally."
            )
        descriptive_statistics.run(task="overall", verbose=True)
        correlation_analysis.run(task="overall", verbose=True)
        hierarchical_regression.run(task="overall", cov_type="nonrobust", verbose=True)
        _run_supplementary_extras()
        _safe_run("paper_tables", paper_tables.run, task="overall", verbose=True)
        _remove_nonpaper_outputs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run manuscript analyses from data/public only.")
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Deprecated in public-only mode (ignored).",
    )
    parser.add_argument("--skip-analysis", action="store_true", help="Skip running core analyses.")
    parser.add_argument(
        "--expected-n",
        type=int,
        default=212,
        help="Expected common-sample N for manuscript reproduction (default: 212).",
    )
    parser.add_argument(
        "--allow-n-mismatch",
        action="store_true",
        help="Allow analysis to run even when common public_id count differs from --expected-n.",
    )
    args = parser.parse_args()

    main(
        run_analysis=not args.skip_analysis,
        expected_n=args.expected_n,
        allow_n_mismatch=args.allow_n_mismatch,
        skip_preprocess=args.skip_preprocess,
    )
