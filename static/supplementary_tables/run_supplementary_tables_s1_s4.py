from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "static").exists():
    ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from static.preprocessing.constants import (  # noqa: E402
    OUTPUT_TABLES_SUPP_DIR,
    STROOP_RT_MAX,
    STROOP_RT_MIN,
    WCST_RT_MAX,
    WCST_RT_MIN,
    get_results_dir,
    get_stroop_trials_path,
)
from static.preprocessing.datasets import load_master_dataset  # noqa: E402
from static.preprocessing.public_validate import get_common_public_ids  # noqa: E402
from static.preprocessing.stroop.qc import clean_stroop_trials  # noqa: E402
from static.preprocessing.wcst.phase import label_wcst_phases  # noqa: E402
from static.preprocessing.wcst.qc import prepare_wcst_trials as prepare_wcst_trials_qc  # noqa: E402
from static.stroop_lmm import run_stroop_trial_lmm as stroop_lmm  # noqa: E402
from static.wcst_phase import run_wcst_phase_rt_ols as wcst_phase_supp  # noqa: E402


REQUIRED_PREDICTORS = [
    "z_ucla_score",
    "z_dass_depression",
    "z_dass_anxiety",
    "z_dass_stress",
    "z_age",
    "gender_male",
]

S2_FIXED_EFFECTS = [
    ("Intercept", "Intercept"),
    ("Trial position", "trial_scaled"),
    ("Condition", "cond_code"),
    ("Loneliness", "z_ucla_score"),
    ("Trial x Condition", "trial_scaled:cond_code"),
    ("Trial x Loneliness", "trial_scaled:z_ucla_score"),
    ("Condition x Loneliness", "cond_code:z_ucla_score"),
    ("Trial x Condition x Loneliness", "trial_scaled:cond_code:z_ucla_score"),
]


def _format_p_for_md(p_val: float) -> str:
    if not np.isfinite(p_val):
        return "NA"
    if p_val < 0.001:
        return "< .001"
    p_str = f"{p_val:.3f}"
    return p_str[1:] if p_str.startswith("0") else p_str


def _format_num_for_md(value: float, decimals: int) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{decimals}f}"


def _markdown_from_dataframe(df: pd.DataFrame) -> str:
    if df.empty:
        return "| NA |\n| --- |\n| NA |"
    columns = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]) for col in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _save_outputs(raw_df: pd.DataFrame, md_df: pd.DataFrame, basename: str, title: str) -> tuple[Path, Path]:
    OUTPUT_TABLES_SUPP_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_TABLES_SUPP_DIR / f"{basename}.csv"
    md_path = OUTPUT_TABLES_SUPP_DIR / f"{basename}.md"

    raw_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    md_lines = [
        f"# {title}",
        "",
        _markdown_from_dataframe(md_df),
        "",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return csv_path, md_path


def _assert_output_file(path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"Expected output file was not created: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"Output file is empty: {path}")


def _run_step2_vs_step3(df: pd.DataFrame, outcome: str, min_n: int = 30) -> dict[str, float]:
    cols = [outcome] + REQUIRED_PREDICTORS
    sub = df[cols].dropna().copy()

    if len(sub) < min_n:
        return {
            "n": int(len(sub)),
            "B": np.nan,
            "SE_B": np.nan,
            "beta": np.nan,
            "F": np.nan,
            "p": np.nan,
            "delta_r2": np.nan,
        }

    reduced_formula = (
        f"{outcome} ~ z_dass_depression + z_dass_anxiety + z_dass_stress + "
        "z_age + C(gender_male)"
    )
    full_formula = (
        f"{outcome} ~ z_ucla_score + z_dass_depression + z_dass_anxiety + "
        "z_dass_stress + z_age + C(gender_male)"
    )

    model_reduced = smf.ols(reduced_formula, data=sub).fit()
    model_full = smf.ols(full_formula, data=sub).fit()
    comparison = anova_lm(model_reduced, model_full)

    b_val = float(model_full.params.get("z_ucla_score", np.nan))
    se_val = float(model_full.bse.get("z_ucla_score", np.nan))
    p_val = float(model_full.pvalues.get("z_ucla_score", np.nan))
    f_val = float(comparison.loc[1, "F"]) if "F" in comparison.columns else np.nan
    delta_r2 = float(model_full.rsquared - model_reduced.rsquared)
    sd_outcome = float(sub[outcome].std(ddof=1))
    beta_val = b_val / sd_outcome if np.isfinite(sd_outcome) and sd_outcome != 0 else np.nan

    return {
        "n": int(len(sub)),
        "B": b_val,
        "SE_B": se_val,
        "beta": beta_val,
        "F": f_val,
        "p": p_val,
        "delta_r2": delta_r2,
    }


def _load_master_qc() -> tuple[pd.DataFrame, set[str]]:
    qc_ids = get_common_public_ids(validate=True)
    master = load_master_dataset(task="overall")
    master = master[master["participant_id"].isin(qc_ids)].copy()
    return master, qc_ids


def _load_stroop_trials_for_s1(qc_ids: set[str]) -> pd.DataFrame:
    trials_path = get_stroop_trials_path("overall")
    if not trials_path.exists():
        return pd.DataFrame()
    trials = pd.read_csv(trials_path, encoding="utf-8-sig")
    if trials.empty:
        return trials
    trials = clean_stroop_trials(trials)
    if qc_ids:
        trials = trials[trials["participant_id"].isin(qc_ids)].copy()
    return trials


def _compute_slopes_for_k(stroop: pd.DataFrame, k: int) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    if stroop.empty:
        return pd.DataFrame(columns=["participant_id", f"stroop_slope_k{k}"])

    stroop = stroop.dropna(subset=["trial_order"]).copy()
    stroop["trial_order"] = pd.to_numeric(stroop["trial_order"], errors="coerce")
    stroop = stroop.dropna(subset=["trial_order"]).copy()

    for pid, grp in stroop.groupby("participant_id"):
        grp = grp.sort_values("trial_order").copy()
        n_trials = len(grp)
        if n_trials == 0:
            continue

        positions = np.arange(1, n_trials + 1)
        edges = np.linspace(0, n_trials, k + 1)
        seg = pd.cut(
            positions,
            bins=edges,
            labels=list(range(1, k + 1)),
            include_lowest=True,
        )
        grp["segment_k"] = seg.astype(int)

        valid = (
            grp["cond"].isin({"congruent", "incongruent"})
            & grp["correct"]
            & (~grp["timeout"])
            & grp["rt_ms"].between(STROOP_RT_MIN, STROOP_RT_MAX)
        )
        gvalid = grp[valid].copy()
        if gvalid.empty:
            slope = np.nan
        else:
            seg_means = gvalid.groupby(["segment_k", "cond"])["rt_ms"].mean().unstack()
            if "incongruent" not in seg_means.columns or "congruent" not in seg_means.columns:
                slope = np.nan
            else:
                seg_means["interference"] = seg_means["incongruent"] - seg_means["congruent"]
                seg_means = seg_means.reset_index()[["segment_k", "interference"]].dropna()
                if len(seg_means) < 2:
                    slope = np.nan
                else:
                    x = seg_means["segment_k"].astype(float).to_numpy()
                    y = seg_means["interference"].to_numpy()
                    slope = float(np.polyfit(x, y, 1)[0])

        rows.append({"participant_id": pid, f"stroop_slope_k{k}": slope})

    return pd.DataFrame(rows)


def _compute_s1(master: pd.DataFrame, qc_ids: set[str]) -> pd.DataFrame:
    stroop_trials = _load_stroop_trials_for_s1(qc_ids)
    if stroop_trials.empty:
        raise RuntimeError("No Stroop trials available for S1.")

    slopes = None
    for k in [2, 3, 4, 6]:
        slope_k = _compute_slopes_for_k(stroop_trials, k)
        slopes = slope_k if slopes is None else slopes.merge(slope_k, on="participant_id", how="outer")

    if slopes is None or slopes.empty:
        raise RuntimeError("No Stroop slope data computed for S1.")

    merged = master.merge(slopes, on="participant_id", how="inner")
    rows = []
    for k in [2, 3, 4, 6]:
        stats = _run_step2_vs_step3(merged, f"stroop_slope_k{k}")
        rows.append(
            {
                "bin_split_k": int(k),
                "n": stats["n"],
                "B": stats["B"],
                "SE_B": stats["SE_B"],
                "beta": stats["beta"],
                "F": stats["F"],
                "p": stats["p"],
                "delta_r2": stats["delta_r2"],
            }
        )
    return pd.DataFrame(rows)


def _compute_s2() -> pd.DataFrame:
    base = stroop_lmm.load_base_data()
    base = stroop_lmm.add_zscores(
        base,
        ["ucla_score", "dass_depression", "dass_anxiety", "dass_stress", "age"],
    )
    predictors = base[["participant_id"] + REQUIRED_PREDICTORS].dropna()

    trials = stroop_lmm.prepare_interference_trials()
    model_data = trials.merge(predictors, on="participant_id", how="inner")

    formula = (
        "log_rt ~ trial_scaled * cond_code * z_ucla_score + "
        "z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + gender_male"
    )
    attempts = [
        {"re_formula": "1 + trial_scaled", "method": "lbfgs"},
        {"re_formula": "1 + trial_scaled", "method": "powell"},
        {"re_formula": "1", "method": "lbfgs"},
        {"re_formula": "1", "method": "powell"},
    ]

    result = None
    last_note: str | None = None
    for attempt in attempts:
        try:
            model = smf.mixedlm(
                formula,
                data=model_data,
                groups=model_data["participant_id"],
                re_formula=attempt["re_formula"],
            )
            fitted = model.fit(reml=False, method=attempt["method"], maxiter=200)
            if bool(getattr(fitted, "converged", False)):
                result = fitted
                break
            last_note = (
                "not_converged "
                f"re_formula={attempt['re_formula']} method={attempt['method']}"
            )
            result = fitted
        except Exception as exc:  # pragma: no cover - defensive runtime fallback
            last_note = (
                f"fit_error re_formula={attempt['re_formula']} method={attempt['method']} "
                f"error={exc}"
            )

    if result is None or not bool(getattr(result, "converged", False)):
        raise RuntimeError(f"S2 MixedLM did not converge. Last note: {last_note}")
    confint = result.conf_int()

    rows = []
    for label, term in S2_FIXED_EFFECTS:
        ci_low = float(confint.loc[term, 0]) if term in confint.index else np.nan
        ci_high = float(confint.loc[term, 1]) if term in confint.index else np.nan
        rows.append(
            {
                "fixed_effect": label,
                "b": float(result.params.get(term, np.nan)),
                "se": float(result.bse.get(term, np.nan)),
                "z": float(result.tvalues.get(term, np.nan)),
                "p": float(result.pvalues.get(term, np.nan)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_trials": int(len(model_data)),
                "n_participants": int(model_data["participant_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _compute_s3(master: pd.DataFrame, qc_ids: set[str]) -> pd.DataFrame:
    wcst = prepare_wcst_trials_qc(get_results_dir("overall"))
    wcst = wcst[wcst["participant_id"].isin(qc_ids)].copy()

    rows = []
    for threshold in [2, 3, 4]:
        wcst_t = wcst.dropna(subset=["trial_order"]).copy()
        wcst_t = label_wcst_phases(
            wcst_t,
            rule_col="rule",
            trial_col="trial_order",
            confirm_len=threshold,
        )
        valid = (
            wcst_t["rt_ms"].between(WCST_RT_MIN, WCST_RT_MAX)
            & (~wcst_t["timeout"])
            & (wcst_t["phase"].notna())
        )
        phase_rt = wcst_t[valid].copy()
        phase_means = phase_rt.groupby(["participant_id", "phase"], observed=False)["rt_ms"].mean().unstack()

        for phase_name in ["exploration", "confirmation", "exploitation"]:
            if phase_name not in phase_means.columns:
                phase_means[phase_name] = np.nan

        phase_means = phase_means.rename(
            columns={
                "exploration": "rt_ms_exploration",
                "confirmation": "rt_ms_confirmation",
                "exploitation": "rt_ms_exploitation",
            }
        )
        phase_means["confirmation_minus_exploitation"] = (
            phase_means["rt_ms_confirmation"] - phase_means["rt_ms_exploitation"]
        )
        phase_means = phase_means.reset_index()

        reg_data = master.merge(phase_means, on="participant_id", how="left")
        mapping = [
            ("Exploration RT", "rt_ms_exploration"),
            ("Confirmation RT", "rt_ms_confirmation"),
            ("Exploitation RT", "rt_ms_exploitation"),
            ("Confirm-Exploit", "confirmation_minus_exploitation"),
        ]
        for stage_label, outcome_col in mapping:
            stats = _run_step2_vs_step3(reg_data, outcome_col)
            rows.append(
                {
                    "threshold": int(threshold),
                    "stage": stage_label,
                    "n": stats["n"],
                    "B": stats["B"],
                    "SE_B": stats["SE_B"],
                    "beta": stats["beta"],
                    "F": stats["F"],
                    "p": stats["p"],
                    "delta_r2": stats["delta_r2"],
                }
            )
    return pd.DataFrame(rows)


def _compute_s4(qc_ids: set[str]) -> pd.DataFrame:
    base = wcst_phase_supp.load_base_data()
    base = wcst_phase_supp.add_zscores(
        base,
        ["ucla_score", "dass_depression", "dass_anxiety", "dass_stress", "age"],
    )

    prepared = wcst_phase_supp.prepare_wcst_trials()
    wcst = prepared["wcst"]
    rt_col = prepared["rt_col"]
    trial_col = prepared["trial_col"]
    rule_col = prepared["rule_col"]
    if not isinstance(wcst, pd.DataFrame) or wcst.empty or rt_col is None or trial_col is None or rule_col is None:
        raise RuntimeError("WCST trials missing required columns for S4.")

    wcst = wcst[wcst["participant_id"].isin(qc_ids)].copy()
    wcst["rt_ms"] = pd.to_numeric(wcst[rt_col], errors="coerce")
    wcst = wcst[wcst["rt_ms"].notna()]
    if "is_rt_valid" in wcst.columns:
        wcst = wcst[wcst["is_rt_valid"] == True]
    if "timeout" in wcst.columns:
        wcst = wcst[~wcst["timeout"]]

    wcst["rule"] = wcst[rule_col].astype(str).str.lower().replace({"color": "colour"})
    wcst = label_wcst_phases(wcst, rule_col="rule", trial_col=trial_col, confirm_len=3)
    wcst = wcst.dropna(subset=["rt_ms", "phase"])

    phase_means = wcst_phase_supp._prepare_pre_exploit_means(wcst, value_col="rt_ms", prefix="rt_ms")
    predictors = base[["participant_id"] + REQUIRED_PREDICTORS].dropna()
    reg_data = phase_means.merge(predictors, on="participant_id", how="inner")

    mapping = [
        ("Pre-exploitation RT", "rt_ms_pre_exploitation"),
        ("Exploitation RT", "rt_ms_exploitation"),
        ("Pre-exploitation - Exploitation", "pre_exploitation_minus_exploitation"),
    ]

    rows = []
    for outcome_label, outcome_col in mapping:
        stats = _run_step2_vs_step3(reg_data, outcome_col)
        rows.append(
            {
                "outcome": outcome_label,
                "n": stats["n"],
                "B": stats["B"],
                "SE_B": stats["SE_B"],
                "beta": stats["beta"],
                "F": stats["F"],
                "p": stats["p"],
                "delta_r2": stats["delta_r2"],
            }
        )
    return pd.DataFrame(rows)


def _to_md_s1(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["bin_split_k"] = df["bin_split_k"].astype(int)
    out["n"] = df["n"].astype(int)
    out["B"] = df["B"].map(lambda x: _format_num_for_md(x, 2))
    out["SE_B"] = df["SE_B"].map(lambda x: _format_num_for_md(x, 2))
    out["beta"] = df["beta"].map(lambda x: _format_num_for_md(x, 2))
    out["F"] = df["F"].map(lambda x: _format_num_for_md(x, 2))
    out["p"] = df["p"].map(_format_p_for_md)
    out["delta_r2"] = df["delta_r2"].map(lambda x: _format_num_for_md(x, 3))
    return out


def _to_md_s2(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["fixed_effect"] = df["fixed_effect"]
    out["b"] = df["b"].map(lambda x: _format_num_for_md(x, 3))
    out["se"] = df["se"].map(lambda x: _format_num_for_md(x, 3))
    out["z"] = df["z"].map(lambda x: _format_num_for_md(x, 2))
    out["p"] = df["p"].map(_format_p_for_md)
    out["95% CI"] = df.apply(
        lambda r: f"[{_format_num_for_md(r['ci_low'], 3)}, {_format_num_for_md(r['ci_high'], 3)}]",
        axis=1,
    )
    out["n_trials"] = df["n_trials"].astype(int)
    out["n_participants"] = df["n_participants"].astype(int)
    return out


def _to_md_s3(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["threshold"] = df["threshold"].astype(int)
    out["stage"] = df["stage"]
    out["n"] = df["n"].astype(int)
    out["B"] = df["B"].map(lambda x: _format_num_for_md(x, 2))
    out["SE_B"] = df["SE_B"].map(lambda x: _format_num_for_md(x, 2))
    out["beta"] = df["beta"].map(lambda x: _format_num_for_md(x, 2))
    out["F"] = df["F"].map(lambda x: _format_num_for_md(x, 2))
    out["p"] = df["p"].map(_format_p_for_md)
    out["delta_r2"] = df["delta_r2"].map(lambda x: _format_num_for_md(x, 3))
    return out


def _to_md_s4(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["outcome"] = df["outcome"]
    out["n"] = df["n"].astype(int)
    out["B"] = df["B"].map(lambda x: _format_num_for_md(x, 2))
    out["SE_B"] = df["SE_B"].map(lambda x: _format_num_for_md(x, 2))
    out["beta"] = df["beta"].map(lambda x: _format_num_for_md(x, 2))
    out["F"] = df["F"].map(lambda x: _format_num_for_md(x, 2))
    out["p"] = df["p"].map(_format_p_for_md)
    out["delta_r2"] = df["delta_r2"].map(lambda x: _format_num_for_md(x, 3))
    return out


def run(verbose: bool = True) -> dict[str, pd.DataFrame]:
    master, qc_ids = _load_master_qc()

    s1 = _compute_s1(master, qc_ids)
    s2 = _compute_s2()
    s3 = _compute_s3(master, qc_ids)
    s4 = _compute_s4(qc_ids)

    s1 = s1[["bin_split_k", "n", "B", "SE_B", "beta", "F", "p", "delta_r2"]]
    s2 = s2[["fixed_effect", "b", "se", "z", "p", "ci_low", "ci_high", "n_trials", "n_participants"]]
    s3 = s3[["threshold", "stage", "n", "B", "SE_B", "beta", "F", "p", "delta_r2"]]
    s4 = s4[["outcome", "n", "B", "SE_B", "beta", "F", "p", "delta_r2"]]

    saved_paths: list[tuple[Path, Path]] = []
    saved_paths.append(_save_outputs(s1, _to_md_s1(s1), "supp_table_s1", "Supplementary Table S1"))
    saved_paths.append(_save_outputs(s2, _to_md_s2(s2), "supp_table_s2", "Supplementary Table S2"))
    saved_paths.append(_save_outputs(s3, _to_md_s3(s3), "supp_table_s3", "Supplementary Table S3"))
    saved_paths.append(_save_outputs(s4, _to_md_s4(s4), "supp_table_s4", "Supplementary Table S4"))
    for csv_path, md_path in saved_paths:
        _assert_output_file(csv_path)
        _assert_output_file(md_path)

    if verbose:
        print("Saved supplementary S1-S4 tables:")
        for csv_path, md_path in saved_paths:
            print(f"  - {csv_path}")
            print(f"  - {md_path}")

    return {
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "s4": s4,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate supplementary tables S1-S4.")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(verbose=not args.quiet)


if __name__ == "__main__":
    main()
