from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import patsy
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from publication.analysis.utils import get_figures_dir, get_output_dir
from publication.lmm.run_stroop_trial_lmm import prepare_interference_trials, load_base_data, add_zscores


def _fit_mixedlm_with_warnings(
    formula: str,
    df: pd.DataFrame,
    re_formula: str,
    method: str,
) -> tuple[object, list[str]]:
    warning_msgs: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = smf.mixedlm(
            formula,
            data=df,
            groups=df["participant_id"],
            re_formula=re_formula,
        )
        result = model.fit(reml=False, method=method, maxiter=200)
    for warn in caught:
        if issubclass(warn.category, ConvergenceWarning):
            warning_msgs.append(str(warn.message))
    return result, warning_msgs


def fit_with_fallback(df: pd.DataFrame, formula: str) -> tuple[object, dict[str, str], list[str]]:
    attempts = [
        {"re_formula": "1", "method": "lbfgs"},
        {"re_formula": "1", "method": "powell"},
    ]
    fallback_result = None
    fallback_attempt = attempts[-1]
    fallback_warnings: list[str] = []

    for attempt in attempts:
        try:
            result, warning_msgs = _fit_mixedlm_with_warnings(
                formula,
                df,
                attempt["re_formula"],
                attempt["method"],
            )
        except Exception:
            continue

        if getattr(result, "converged", False):
            boundary_warning = any("boundary" in msg.lower() for msg in warning_msgs)
            hessian_warning = any("hessian" in msg.lower() and "not positive" in msg.lower() for msg in warning_msgs)
            if not boundary_warning and not hessian_warning:
                return result, attempt, warning_msgs
            if fallback_result is None:
                fallback_result = result
                fallback_attempt = attempt
                fallback_warnings = warning_msgs

    if fallback_result is not None:
        return fallback_result, fallback_attempt, fallback_warnings

    raise RuntimeError("Stroop interference LMM failed to converge.")


def _build_prediction_grid(
    df: pd.DataFrame,
    trial_points: int = 60,
) -> pd.DataFrame:
    trial_scaled = np.linspace(0, 1, trial_points)
    loneliness_levels = [-1.0, 0.0, 1.0]
    cond_map = {
        "congruent": -0.5,
        "incongruent": 0.5,
    }

    gender_mean = float(df["gender_male"].mean()) if "gender_male" in df.columns else 0.0

    rows = []
    for z_ucla in loneliness_levels:
        for cond_label, cond_code in cond_map.items():
            for t in trial_scaled:
                rows.append({
                    "trial_scaled": float(t),
                    "cond_code": float(cond_code),
                    "z_ucla_score": float(z_ucla),
                    "z_dass_depression": 0.0,
                    "z_dass_anxiety": 0.0,
                    "z_dass_stress": 0.0,
                    "z_age": 0.0,
                    "gender_male": gender_mean,
                    "loneliness_level": z_ucla,
                    "condition": cond_label,
                })

    return pd.DataFrame(rows)


def _predict_fixed_effects(result: object, grid: pd.DataFrame) -> pd.Series:
    design = patsy.build_design_matrices([result.model.data.design_info], grid, return_type="dataframe")[0]
    fe_params = result.fe_params
    design = design.loc[:, fe_params.index]
    return design @ fe_params


def _plot_panel(
    grid: pd.DataFrame,
    y_col: str,
    fig_path: Path,
    title: str,
    y_label: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    panel_map = {
        -1.0: "Loneliness -1 SD",
        0.0: "Loneliness 0 SD",
        1.0: "Loneliness +1 SD",
    }
    colors = {"congruent": "#1f77b4", "incongruent": "#d62728"}

    for ax, level in zip(axes, [-1.0, 0.0, 1.0]):
        panel = grid[grid["loneliness_level"] == level]
        for cond_label in ("congruent", "incongruent"):
            sub = panel[panel["condition"] == cond_label]
            ax.plot(
                sub["trial_scaled"],
                sub[y_col],
                color=colors[cond_label],
                linewidth=2,
                label=cond_label,
            )
        ax.set_title(panel_map[level])
        ax.set_xlabel("Trial position (0-1)")
        ax.grid(True, axis="y", alpha=0.2)

    axes[0].set_ylabel(y_label)
    axes[0].legend(loc="upper right")

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def main() -> None:
    if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    base = load_base_data()
    base = add_zscores(
        base,
        ["ucla_score", "dass_depression", "dass_anxiety", "dass_stress", "age"],
    )

    trials = prepare_interference_trials()
    if trials.empty:
        raise RuntimeError("No Stroop interference trials available.")

    predictors = base[
        [
            "participant_id",
            "z_ucla_score",
            "z_dass_depression",
            "z_dass_anxiety",
            "z_dass_stress",
            "z_age",
            "gender_male",
        ]
    ].dropna()

    df = trials.merge(predictors, on="participant_id", how="inner")
    if df.empty:
        raise RuntimeError("No trials available after merging predictors.")

    formula = (
        "log_rt ~ trial_scaled * cond_code * z_ucla_score + "
        "z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + gender_male"
    )

    result, attempt, warnings_list = fit_with_fallback(df, formula)

    grid = _build_prediction_grid(df)
    grid["pred_log_rt"] = _predict_fixed_effects(result, grid)
    grid["pred_rt_ms"] = np.exp(grid["pred_log_rt"])

    output_dir = get_output_dir("stroop")
    figures_dir = get_figures_dir()

    pred_path = output_dir / "stroop_interference_triallevel_interaction_predictions.csv"
    grid.to_csv(pred_path, index=False, encoding="utf-8-sig")

    title = "Stroop Interference: Trial Position x Condition x Loneliness"
    fig_path_log = figures_dir / "stroop_interference_triallevel_interaction_logrt.pdf"
    _plot_panel(
        grid,
        y_col="pred_log_rt",
        fig_path=fig_path_log,
        title=title,
        y_label="Predicted log RT",
    )

    fig_path_ms = figures_dir / "stroop_interference_triallevel_interaction_rtms.pdf"
    _plot_panel(
        grid,
        y_col="pred_rt_ms",
        fig_path=fig_path_ms,
        title=title,
        y_label="Predicted RT (ms)",
    )

    print(f"Saved: {pred_path}")
    print(f"Saved: {fig_path_log}")
    print(f"Saved: {fig_path_ms}")


if __name__ == "__main__":
    main()
