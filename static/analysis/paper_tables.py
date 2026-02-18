"""
Generate manuscript-ready table values (Tables 1-3).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from static.analysis.utils import get_analysis_data
from static.preprocessing.constants import (
    DATA_DIR,
    OUTPUT_STATS_CORE_DIR,
    OUTPUT_STATS_SUPP_DIR,
    OUTPUT_TABLES_CORE_DIR,
    get_results_dir,
)
from static.preprocessing.surveys import load_survey_items


DASS_SUBSCALES = {
    "DASS-21 Depression": [3, 5, 10, 13, 16, 17, 21],
    "DASS-21 Anxiety": [2, 4, 7, 9, 15, 19, 20],
    "DASS-21 Stress": [1, 6, 8, 11, 12, 14, 18],
}

UCLA_ITEMS = list(range(1, 21))
UCLA_REVERSE = {1, 5, 6, 9, 10, 15, 16, 19, 20}


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _cronbach_alpha(df: pd.DataFrame) -> float:
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    k = df.shape[1]
    if k < 2 or len(df) < 2:
        return float("nan")
    item_vars = df.var(ddof=1)
    total_var = df.sum(axis=1).var(ddof=1)
    if not np.isfinite(total_var) or total_var == 0:
        return float("nan")
    return float((k / (k - 1)) * (1 - item_vars.sum() / total_var))


def _load_survey_items(task: str) -> pd.DataFrame:
    data_dir = get_results_dir(task)
    try:
        items = load_survey_items(data_dir)
        if not items.empty:
            return items
    except Exception:
        pass

    public_path = DATA_DIR / "public" / "surveys_public.csv"
    if not public_path.exists():
        return pd.DataFrame()

    surveys = pd.read_csv(public_path, encoding="utf-8-sig")
    surveys["surveyName"] = surveys["surveyName"].astype(str).str.lower()

    def _extract(prefix: str, n_items: int) -> pd.DataFrame:
        subset = surveys[surveys["surveyName"] == prefix].copy()
        if subset.empty:
            return pd.DataFrame()
        cols = [f"q{i}" for i in range(1, n_items + 1) if f"q{i}" in subset.columns]
        if not cols:
            return pd.DataFrame()
        tmp = subset[cols].copy()
        tmp.columns = [f"{prefix}_{c.lstrip('q')}" for c in cols]
        for c in tmp.columns:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        return tmp

    ucla_items = _extract("ucla", 20)
    dass_items = _extract("dass", 21)
    if ucla_items.empty and dass_items.empty:
        return pd.DataFrame()
    return pd.concat([ucla_items, dass_items], axis=1)


def _compute_scale_alphas(task: str) -> dict[str, float]:
    items = _load_survey_items(task)
    if items.empty:
        return {}

    alphas: dict[str, float] = {}

    # UCLA alpha (reverse-coded)
    ucla_cols = [f"ucla_{i}" for i in UCLA_ITEMS if f"ucla_{i}" in items.columns]
    if len(ucla_cols) == 20:
        ucla_df = items[ucla_cols].copy()
        for i in UCLA_REVERSE:
            col = f"ucla_{i}"
            if col in ucla_df.columns:
                ucla_df[col] = ucla_df[col].apply(lambda x: 5 - x if pd.notna(x) else x)
        alphas["UCLA Loneliness"] = _cronbach_alpha(ucla_df)

    # DASS alphas
    for label, idxs in DASS_SUBSCALES.items():
        cols = [f"dass_{i}" for i in idxs if f"dass_{i}" in items.columns]
        if len(cols) != len(idxs):
            continue
        alphas[label] = _cronbach_alpha(items[cols])

    return alphas


def _format_m_sd(mean: float, sd: float, decimals: int = 2) -> str:
    if not np.isfinite(mean) or not np.isfinite(sd):
        return "NA"
    return f"{mean:.{decimals}f} ({sd:.{decimals}f})"


def _load_sd_lookup(task: str) -> dict[str, float]:
    _ = task
    sd_lookup: dict[str, float] = {}
    try:
        df = get_analysis_data(task)
        for col in df.columns:
            if col.startswith("wcst_") or col.startswith("stroop_"):
                sd = pd.to_numeric(df[col], errors="coerce").std(ddof=1)
                if np.isfinite(sd):
                    sd_lookup[col] = float(sd)
    except Exception:
        df = None

    desc_path = OUTPUT_STATS_CORE_DIR / "table1_descriptives.csv"
    desc = _safe_read_csv(desc_path)
    if not desc.empty and "Column" in desc.columns:
        for _, row in desc.iterrows():
            col = row.get("Column")
            sd = row.get("SD")
            if isinstance(col, str) and np.isfinite(sd):
                sd_lookup[col] = float(sd)

    features_path = DATA_DIR / "public" / "features_public.csv"
    if features_path.exists():
        feats = pd.read_csv(features_path, encoding="utf-8-sig")
        for col in feats.columns:
            if col not in sd_lookup:
                sd = pd.to_numeric(feats[col], errors="coerce").std(ddof=1)
                if np.isfinite(sd):
                    sd_lookup[col] = float(sd)

    return sd_lookup


def _format_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "< .001"
    return f"{p:.3f}"


def compute_table1(task: str, output_dir: Path) -> dict[str, pd.DataFrame]:
    _ = task
    stats_dir = OUTPUT_STATS_CORE_DIR
    desc_by_gender = _safe_read_csv(stats_dir / "table1_descriptives_by_gender.csv")
    desc_total = _safe_read_csv(stats_dir / "table1_descriptives.csv")
    alphas = _compute_scale_alphas(task)

    panel_a_vars = [
        ("age", "Age (years)"),
        ("ucla_score", "UCLA Loneliness"),
        ("dass_depression", "DASS-21 Depression"),
        ("dass_anxiety", "DASS-21 Anxiety"),
        ("dass_stress", "DASS-21 Stress"),
    ]

    panel_a_rows = []
    for col, label in panel_a_vars:
        row = {"Measure": label, "Column": col}
        for group in ["Total", "Male", "Female"]:
            sub = desc_by_gender[(desc_by_gender["Group"] == group) & (desc_by_gender["Column"] == col)]
            if sub.empty:
                row[f"{group}_M"] = np.nan
                row[f"{group}_SD"] = np.nan
                row[f"{group}_MSD"] = "NA"
            else:
                mean = float(sub["Mean"].iloc[0])
                sd = float(sub["SD"].iloc[0])
                row[f"{group}_M"] = mean
                row[f"{group}_SD"] = sd
                row[f"{group}_MSD"] = _format_m_sd(mean, sd)
        row["Alpha"] = alphas.get(label, np.nan)
        panel_a_rows.append(row)

    panel_a = pd.DataFrame(panel_a_rows)

    # Panel B (task performance)
    reliability_notes = {
        "Interference RT slope": (
            "Split-half reliability not estimated; each half contains too few trials per condition per segment "
            "(~9) for stable per-person slope estimates."
        ),
        "WCST PE rate (%)": (
            "Split-half reliability not estimated; PE rate is a single proportion tied to rule shifts, "
            "yielding non-equivalent halves and unstable split-half estimates."
        ),
    }

    rel_dir = OUTPUT_STATS_SUPP_DIR
    stroop_rel = _safe_read_csv(rel_dir / "stroop_interference_reliability.csv")
    wcst_rel = _safe_read_csv(rel_dir / "wcst_phase_split_half_reliability.csv")

    def _get_stroop_rsb() -> float:
        if stroop_rel.empty:
            return np.nan
        row = stroop_rel[stroop_rel["method"] == "odd_even"]
        if row.empty:
            row = stroop_rel.iloc[:1]
        return float(row["r_sb"].iloc[0]) if "r_sb" in row.columns else np.nan

    def _get_wcst_rsb(phase: str) -> float:
        if wcst_rel.empty:
            return np.nan
        row = wcst_rel[wcst_rel["phase"] == phase]
        if row.empty:
            return np.nan
        return float(row["spearman_brown"].iloc[0])

    panel_b_vars = [
        ("stroop_interference", "Stroop Interference RT (ms)", _get_stroop_rsb()),
        ("stroop_interference_slope", "Interference RT slope", np.nan),
        ("wcst_perseverative_error_rate", "WCST PE rate (%)", np.nan),
        ("wcst_exploration_rt_all", "Exploration RT (ms)", _get_wcst_rsb("exploration")),
        ("wcst_confirmation_rt_all", "Confirmation RT (ms)", _get_wcst_rsb("confirmation")),
        ("wcst_exploitation_rt_all", "Exploitation RT (ms)", _get_wcst_rsb("exploitation")),
    ]

    panel_b_rows = []
    for col, label, rsb in panel_b_vars:
        sub = desc_total[desc_total["Column"] == col]
        if sub.empty:
            mean = np.nan
            sd = np.nan
            n_val = np.nan
        else:
            mean = float(sub["Mean"].iloc[0])
            sd = float(sub["SD"].iloc[0])
            n_val = float(sub["N"].iloc[0])
        panel_b_rows.append(
            {
                "Measure": label,
                "Column": col,
                "N": n_val,
                "Mean": mean,
                "SD": sd,
                "M_SD": _format_m_sd(mean, sd),
                "r_sb": rsb,
                "Reliability_note": reliability_notes.get(label, ""),
            }
        )

    panel_b = pd.DataFrame(panel_b_rows)

    panel_a.to_csv(output_dir / "table1_panel_a.csv", index=False, encoding="utf-8-sig")
    panel_b.to_csv(output_dir / "table1_panel_b.csv", index=False, encoding="utf-8-sig")

    return {"panel_a": panel_a, "panel_b": panel_b}


def compute_table2(task: str, output_dir: Path) -> pd.DataFrame:
    _ = task
    r_path = OUTPUT_STATS_CORE_DIR / "correlation_matrix.csv"
    p_path = OUTPUT_STATS_CORE_DIR / "correlation_pvalues.csv"
    r = _safe_read_csv(r_path)
    p = _safe_read_csv(p_path)
    if r.empty or p.empty:
        return pd.DataFrame()

    r = r.set_index(r.columns[0])
    p = p.set_index(p.columns[0])

    order = [
        "UCLA",
        "DASS-Dep",
        "DASS-Anx",
        "DASS-Str",
        "Stroop Interference RT",
        "Stroop Interference RT Slope",
        "WCST PE Rate",
        "WCST Confirmation RT (all trials)",
        "WCST Exploitation RT (all trials)",
        "WCST Confirm - Exploit RT (all trials)",
    ]

    # ensure all labels exist
    order = [label for label in order if label in r.index]
    r = r.loc[order, order]
    p = p.loc[order, order]

    out = pd.DataFrame("", index=order, columns=order)
    for i, row_label in enumerate(order):
        for j, col_label in enumerate(order):
            if i <= j:
                continue
            r_val = r.loc[row_label, col_label]
            p_val = p.loc[row_label, col_label]
            if not np.isfinite(r_val) or not np.isfinite(p_val):
                continue
            if p_val < 0.001:
                star = "***"
            elif p_val < 0.01:
                star = "**"
            elif p_val < 0.05:
                star = "*"
            else:
                star = ""
            out.loc[row_label, col_label] = f"{r_val:.3f}{star}"

    out.to_csv(output_dir / "table2_correlations.csv", encoding="utf-8-sig")
    return out


def compute_table3(task: str, output_dir: Path) -> pd.DataFrame:
    hr_path = OUTPUT_STATS_CORE_DIR / "hierarchical_results.csv"
    hr = _safe_read_csv(hr_path)
    if hr.empty:
        return pd.DataFrame()

    r_path = OUTPUT_STATS_CORE_DIR / "correlation_matrix.csv"
    r = _safe_read_csv(r_path)
    if r.empty:
        return pd.DataFrame()
    r = r.set_index(r.columns[0])

    sd_lookup = _load_sd_lookup(task)

    outcomes = [
        ("stroop_interference", "Stroop Interference RT", "Stroop Interference RT"),
        ("stroop_interference_slope", "Interference RT slope", "Stroop Interference RT Slope"),
        ("wcst_perseverative_error_rate", "WCST PE rate", "WCST PE Rate"),
        ("wcst_confirmation_rt_all", "Confirmation RT", "WCST Confirmation RT (all trials)"),
        ("wcst_exploitation_rt_all", "Exploitation RT", "WCST Exploitation RT (all trials)"),
        ("wcst_confirmation_minus_exploitation_rt_all", "Confirm-Exploit", "WCST Confirm - Exploit RT (all trials)"),
    ]

    rows = []
    for col, label, corr_label in outcomes:
        match = hr[hr["outcome_column"] == col]
        if match.empty:
            continue
        row = match.iloc[0]
        r_val = r.loc[corr_label, "UCLA"] if corr_label in r.index else np.nan
        sd_y = sd_lookup.get(col, np.nan)
        beta = row["ucla_beta"] / sd_y if np.isfinite(sd_y) and sd_y != 0 else np.nan

        rows.append(
            {
                "Outcome": label,
                "r": r_val,
                "R2_Step1": row.get("model0_r2", np.nan),
                "R2_Step2": row.get("model1_r2", np.nan),
                "b": row.get("ucla_beta", np.nan),
                "SE": row.get("ucla_se", np.nan),
                "beta": beta,
                "p": row.get("ucla_p", np.nan),
                "Delta_R2": row.get("delta_r2_ucla", np.nan),
                "F_change": row.get("F_ucla", np.nan),
            }
        )

    out = pd.DataFrame(rows)

    # Add formatted columns for direct table use
    out["r_fmt"] = out["r"].map(lambda x: f"{x:.3f}" if np.isfinite(x) else "NA")
    out["R2_Step1_fmt"] = out["R2_Step1"].map(lambda x: f"{x:.3f}" if np.isfinite(x) else "NA")
    out["R2_Step2_fmt"] = out["R2_Step2"].map(lambda x: f"{x:.3f}" if np.isfinite(x) else "NA")
    out["b_fmt"] = out["b"].map(lambda x: f"{x:.2f}" if np.isfinite(x) else "NA")
    out["SE_fmt"] = out["SE"].map(lambda x: f"{x:.2f}" if np.isfinite(x) else "NA")
    out["beta_fmt"] = out["beta"].map(lambda x: f"{x:.2f}" if np.isfinite(x) else "NA")
    out["p_fmt"] = out["p"].map(_format_p)
    out["Delta_R2_fmt"] = out["Delta_R2"].map(lambda x: f"{x:.3f}" if np.isfinite(x) else "NA")
    out["F_change_fmt"] = out["F_change"].map(lambda x: f"{x:.2f}" if np.isfinite(x) else "NA")

    out.to_csv(output_dir / "table3_hierarchical.csv", index=False, encoding="utf-8-sig")
    return out


def run(task: str = "overall", verbose: bool = True) -> dict[str, pd.DataFrame]:
    output_dir = OUTPUT_TABLES_CORE_DIR / task
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    results.update(compute_table1(task, output_dir))
    results["table2"] = compute_table2(task, output_dir)
    results["table3"] = compute_table3(task, output_dir)

    if verbose:
        print(f"Saved table outputs to: {output_dir}")
        for key, df in results.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                print(f"  - {key}: {len(df)} rows")
            else:
                print(f"  - {key}: empty")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate manuscript table values.")
    parser.add_argument(
        "--task",
        default="overall",
        help="Task to analyze (default: overall).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(task=args.task, verbose=not args.quiet)
