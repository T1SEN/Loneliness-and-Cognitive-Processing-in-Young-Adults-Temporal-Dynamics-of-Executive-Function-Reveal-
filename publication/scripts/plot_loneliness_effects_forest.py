from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from publication.analysis.utils import get_analysis_data, get_figures_dir


def add_zscores(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        std = df[col].std()
        df[f"z_{col}"] = (df[col] - df[col].mean()) / std if std and np.isfinite(std) else np.nan
    return df


def _fit_ols(df: pd.DataFrame, dv: str) -> smf.ols:
    formula = (
        f"{dv} ~ z_ucla_score + z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + gender_male"
    )
    return smf.ols(formula, data=df).fit()


def _extract_ucla_ci(model: smf.ols) -> dict[str, float]:
    beta = float(model.params.get("z_ucla_score", np.nan))
    se = float(model.bse.get("z_ucla_score", np.nan))
    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se
    return {
        "beta": beta,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def main() -> None:
    if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    stroop = get_analysis_data("stroop")
    wcst = get_analysis_data("wcst")

    stroop = add_zscores(stroop, ["ucla_score", "dass_depression", "dass_anxiety", "dass_stress", "age"])
    wcst = add_zscores(wcst, ["ucla_score", "dass_depression", "dass_anxiety", "dass_stress", "age"])

    outcomes = [
        ("Stroop Interference RT", "stroop_interference", stroop),
        ("Stroop Interference Slope", "stroop_interference_slope", stroop),
        ("WCST PE rate", "pe_rate", wcst),
        ("WCST Post-error RT", "wcst_post_error_rt", wcst),
    ]

    rows = []
    for label, col, df in outcomes:
        use = df[[
            col,
            "z_ucla_score",
            "z_dass_depression",
            "z_dass_anxiety",
            "z_dass_stress",
            "z_age",
            "gender_male",
        ]].dropna()
        if use.empty:
            continue
        model = _fit_ols(use, col)
        stats = _extract_ucla_ci(model)
        rows.append({
            "outcome": label,
            "beta": stats["beta"],
            "ci_low": stats["ci_low"],
            "ci_high": stats["ci_high"],
            "n": int(len(use)),
        })

    summary = pd.DataFrame(rows)
    output_dir = Path(ROOT) / "publication" / "data" / "outputs" / "analysis" / "overall"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "loneliness_effect_forest_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    order = ["Stroop Interference RT", "Stroop Interference Slope", "WCST PE rate", "WCST Post-error RT"]
    summary["outcome"] = pd.Categorical(summary["outcome"], categories=order, ordered=True)
    summary = summary.sort_values("outcome")

    y = np.arange(len(summary))
    ax.errorbar(
        summary["beta"],
        y,
        xerr=[summary["beta"] - summary["ci_low"], summary["ci_high"] - summary["beta"]],
        fmt="o",
        color="#1f77b4",
        ecolor="#1f77b4",
        capsize=4,
        linewidth=1.5,
    )
    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(summary["outcome"].tolist())
    ax.set_xlabel("Loneliness effect (standardized beta)")
    ax.set_title("Loneliness Effects: Traditional vs Process Measures")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()

    figures_dir = get_figures_dir()
    fig_path_png = figures_dir / "loneliness_effects_forest.png"
    fig_path_pdf = figures_dir / "loneliness_effects_forest.pdf"
    fig.savefig(fig_path_png, dpi=160)
    fig.savefig(fig_path_pdf)
    plt.close(fig)

    print(f"Saved: {summary_path}")
    print(f"Saved: {fig_path_png}")
    print(f"Saved: {fig_path_pdf}")


if __name__ == "__main__":
    main()
