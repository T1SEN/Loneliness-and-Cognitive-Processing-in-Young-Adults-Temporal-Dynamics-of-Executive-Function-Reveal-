"""Rebuild overall dataset and run core analyses (overall-only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from static.preprocessing.constants import get_results_dir
from static.preprocessing.cli import run_preprocess_pipeline
from static.analysis import descriptive_statistics, correlation_analysis, hierarchical_regression
from static.analysis import supplementary_analyses


def _features_ready() -> bool:
    features_path = get_results_dir("overall") / "5_overall_features.csv"
    return features_path.exists()


def main(run_preprocess: bool, run_analysis: bool) -> None:
    if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if run_preprocess:
        run_preprocess_pipeline(build=True, features=True, save=True, verbose=True)

    if not _features_ready():
        if run_preprocess:
            run_preprocess_pipeline(build=False, features=True, save=True, verbose=True)

    if not _features_ready():
        print("[WARN] 5_overall_features.csv not found in complete_overall.")
        print("[WARN] Analyses may be incomplete without overall feature columns.")
        if run_analysis:
            print("[WARN] Skipping analysis steps.")
        return

    if run_analysis:
        descriptive_statistics.run(task="overall", verbose=True)
        correlation_analysis.run(task="overall", verbose=True)
        hierarchical_regression.run(task="overall", cov_type="nonrobust", verbose=True)
        supplementary_analyses.run(task="overall")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild overall dataset and run core analyses.")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip rebuilding complete_overall.")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip running core analyses.")
    args = parser.parse_args()

    main(run_preprocess=not args.skip_preprocess, run_analysis=not args.skip_analysis)
