from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import patsy
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


def _assign_loneliness_tertiles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["ucla_score"]).copy()
    q33 = df["ucla_score"].quantile(1 / 3)
    q67 = df["ucla_score"].quantile(2 / 3)
    df["loneliness_group"] = "Medium"
    df.loc[df["ucla_score"] <= q33, "loneliness_group"] = "Low"
    df.loc[df["ucla_score"] >= q67, "loneliness_group"] = "High"
    df["loneliness_group"] = pd.Categorical(df["loneliness_group"], categories=["Low", "Medium", "High"], ordered=True)
    return df


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Missing columns: {candidates}")


def _fit_model(df: pd.DataFrame, dv: str, control_col: str | None) -> smf.ols:
    formula = (
        f"{dv} ~ C(loneliness_group) + z_dass_depression + z_dass_anxiety + "
        "z_dass_stress + z_age + gender_male"
    )
    if control_col:
        formula = formula + f" + z_{control_col}"
    return smf.ols(formula, data=df).fit()


def _predict_group_means(model: smf.ols, df: pd.DataFrame, control_col: str | None) -> pd.DataFrame:
    group_levels = ["Low", "Medium", "High"]
    base = {
        "z_dass_depression": float(df["z_dass_depression"].mean()),
        "z_dass_anxiety": float(df["z_dass_anxiety"].mean()),
        "z_dass_stress": float(df["z_dass_stress"].mean()),
        "z_age": float(df["z_age"].mean()),
        "gender_male": float(df["gender_male"].mean()),
    }
    if control_col:
        base[f"z_{control_col}"] = float(df[f"z_{control_col}"].mean())

    rows = []
    for group in group_levels:
        row = {**base, "loneliness_group": group}
        rows.append(row)
    grid = pd.DataFrame(rows)

    design = patsy.build_design_matrices([model.model.data.design_info], grid, return_type="dataframe")[0]
    params = model.params
    cov = model.cov_params()
    pred = design @ params
    se = np.sqrt(np.diag(design.values @ cov.values @ design.values.T))

    out = grid.copy()
    out["pred_mean"] = pred
    out["se"] = se
    out["ci_low"] = pred - 1.96 * se
    out["ci_high"] = pred + 1.96 * se
    return out[["loneliness_group", "pred_mean", "ci_low", "ci_high", "se"]]


def _plot_panel(ax, summary: pd.DataFrame, title: str) -> None:
    x = np.arange(len(summary), dtype=float)
    y = summary["pred_mean"].to_numpy()
    yerr = np.vstack([y - summary["ci_low"].to_numpy(), summary["ci_high"].to_numpy() - y])

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        color="#1f77b4",
        ecolor="#1f77b4",
        capsize=4,
        markersize=7,
        linewidth=1.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(summary["loneliness_group"].tolist())
    ax.set_title(title)
    ax.set_xlabel("Loneliness group")
    ax.grid(True, axis="y", alpha=0.2)


def main() -> None:
    if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    df = get_analysis_data("overall")

    dv_col = _pick_column(df, ["wcst_post_error_rt", "wcst_post_error_rt_mean", "wcst_post_error_rt_ms"])
    overall_col = _pick_column(df, ["wcst_mean_rt_all", "wcst_mean_rt", "wcst_mean_rt_all_correct"])

    needed = [
        dv_col,
        overall_col,
        "ucla_score",
        "dass_depression",
        "dass_anxiety",
        "dass_stress",
        "age",
        "gender_male",
    ]
    df = df[needed].dropna().copy()

    df = add_zscores(df, ["dass_depression", "dass_anxiety", "dass_stress", "age", overall_col])
    df = _assign_loneliness_tertiles(df)

    base_df = df.dropna(subset=[dv_col, "loneliness_group"]).copy()

    model_simple = _fit_model(base_df, dv=dv_col, control_col=None)
    summary_simple = _predict_group_means(model_simple, base_df, control_col=None)
    summary_simple["panel"] = "simple"

    model_adj = _fit_model(base_df, dv=dv_col, control_col=overall_col)
    summary_adj = _predict_group_means(model_adj, base_df, control_col=overall_col)
    summary_adj["panel"] = "adjusted_overall_rt"

    summary = pd.concat([summary_simple, summary_adj], ignore_index=True)

    output_dir = Path(ROOT) / "publication" / "data" / "outputs" / "analysis" / "overall"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "wcst_post_error_rt_loneliness_panels_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    figures_dir = get_figures_dir()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)

    _plot_panel(axes[0], summary_simple, "Post-error RT (unadjusted)")
    _plot_panel(axes[1], summary_adj, "Post-error RT (adjusted for overall RT)")

    axes[0].set_ylabel("Predicted post-error RT (ms)")
    fig.suptitle("WCST Post-error RT by Loneliness Group", y=1.02)
    fig.tight_layout()

    fig_path_png = figures_dir / "wcst_post_error_rt_loneliness_panels.png"
    fig_path_pdf = figures_dir / "wcst_post_error_rt_loneliness_panels.pdf"
    fig.savefig(fig_path_png, dpi=160)
    fig.savefig(fig_path_pdf)
    plt.close(fig)

    print(f"Saved: {summary_path}")
    print(f"Saved: {fig_path_png}")
    print(f"Saved: {fig_path_pdf}")


if __name__ == "__main__":
    main()
