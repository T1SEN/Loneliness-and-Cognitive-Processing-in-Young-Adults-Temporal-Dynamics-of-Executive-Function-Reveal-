"""
Batch runner for legacy advanced analysis scripts.

Usage:
    python scripts/run_legacy_advanced.py

This script executes every module under
``analysis.archive.legacy_advanced`` via ``python -m`` so that imports
resolve correctly.  Stdout/stderr from each module are captured to
``results/analysis_outputs/legacy_advanced_logs/<module>.log`` and a
summary JSON file is written for quick inspection.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

MODULES = [
    "analysis.archive.legacy_advanced.adaptive_recovery_dynamics",
    "analysis.archive.legacy_advanced.attentional_lapse_mixture",
    "analysis.archive.legacy_advanced.causal_dag_extended",
    "analysis.archive.legacy_advanced.composite_vulnerability_indices",
    "analysis.archive.legacy_advanced.dass_ef_hier_bayes",
    "analysis.archive.legacy_advanced.dass_ef_specificity",
    "analysis.archive.legacy_advanced.ef_vulnerability_clustering",
    "analysis.archive.legacy_advanced.error_burst_clustering",
    "analysis.archive.legacy_advanced.exgaussian_rt_analysis",
    "analysis.archive.legacy_advanced.framework1_regression_mixtures",
    "analysis.archive.legacy_advanced.framework2_normative_modeling",
    "analysis.archive.legacy_advanced.framework3_latent_factors_sem",
    "analysis.archive.legacy_advanced.framework4_causal_dag_simulation",
    "analysis.archive.legacy_advanced.gendered_temporal_vulnerability",
    "analysis.archive.legacy_advanced.iiv_decomposition_analysis",
    "analysis.archive.legacy_advanced.latent_metacontrol_sem",
    "analysis.archive.legacy_advanced.multivariate_ef_analysis",
    "analysis.archive.legacy_advanced.network_psychometrics",
    "analysis.archive.legacy_advanced.network_psychometrics_extended",
    "analysis.archive.legacy_advanced.perseveration_momentum_analysis",
    "analysis.archive.legacy_advanced.post_error_slowing_gender_moderation",
    "analysis.archive.legacy_advanced.post_error_slowing_integrated",
    "analysis.archive.legacy_advanced.proactive_reactive_control",
    "analysis.archive.legacy_advanced.prp_exgaussian_dass_controlled",
    "analysis.archive.legacy_advanced.prp_exgaussian_decomposition",
    "analysis.archive.legacy_advanced.prp_post_error_adjustments",
    "analysis.archive.legacy_advanced.prp_response_order_analysis",
    "analysis.archive.legacy_advanced.prp_rt_variability_extended",
    "analysis.archive.legacy_advanced.rt_percentile_group_comparison",
    "analysis.archive.legacy_advanced.sequential_dynamics_analysis",
    "analysis.archive.legacy_advanced.stroop_exgaussian_decomposition",
    "analysis.archive.legacy_advanced.stroop_post_error_adjustments",
    "analysis.archive.legacy_advanced.stroop_rt_variability_extended",
    "analysis.archive.legacy_advanced.trial_level_bayesian",
    "analysis.archive.legacy_advanced.trial_level_cascade_glmm",
    "analysis.archive.legacy_advanced.trial_level_mixed_effects",
    "analysis.archive.legacy_advanced.trial_level_mvpa_vulnerability",
    "analysis.archive.legacy_advanced.ucla_dass_moderation_commonality",
    "analysis.archive.legacy_advanced.wcst_error_type_decomposition",
    "analysis.archive.legacy_advanced.wcst_mechanism_comprehensive",
    "analysis.archive.legacy_advanced.wcst_post_error_adaptation_quick",
    "analysis.archive.legacy_advanced.wcst_switching_dynamics_quick",
]


def run_module(module: str, log_dir: Path) -> dict:
    """Run a module via ``python -m`` and capture its output."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{module.split('.')[-1]}.log"
    cmd = [sys.executable, "-m", module]
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"# Command: {' '.join(cmd)}\n")
        log_file.write(f"# Working dir: {Path.cwd()}\n")
        log_file.write(f"# Start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        proc = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
        duration = time.time() - start
        log_file.write(f"\n# End: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"# Duration: {duration:.1f}s\n")
        log_file.write(f"# Exit code: {proc.returncode}\n")
    return {
        "module": module,
        "log": str(log_path),
        "exit_code": proc.returncode,
        "duration_sec": round(duration, 1),
        "status": "ok" if proc.returncode == 0 else "error",
    }


def main() -> None:
    log_dir = Path("results/analysis_outputs/legacy_advanced_logs")
    summary = []
    for module in MODULES:
        print(f"[RUN] {module}")
        summary.append(run_module(module, log_dir))
    summary_path = log_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary written to {summary_path}")
    failures = [item for item in summary if item["status"] != "ok"]
    if failures:
        print("\nFailures:")
        for item in failures:
            print(f"  - {item['module']} (exit {item['exit_code']}) -> {item['log']}")


if __name__ == "__main__":
    main()
