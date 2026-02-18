from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from static.analysis.utils import get_output_dir, run_ucla_regression
from static.preprocessing.constants import WCST_RT_MIN, WCST_RT_MAX, get_results_dir
from static.preprocessing.core import ensure_participant_id
from static.preprocessing.datasets import load_master_dataset
from static.preprocessing.surveys import load_dass_scores, load_participants, load_ucla_scores
from static.preprocessing.wcst.phase import label_wcst_phases
from static.preprocessing.wcst.qc import clean_wcst_trials
from static.preprocessing.wcst.utils import prepare_wcst_trials


PHASE_ORDER = ["exploration", "confirmation", "exploitation"]


def add_zscores(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        std = df[col].std()
        df[f"z_{col}"] = (df[col] - df[col].mean()) / std if std and np.isfinite(std) else np.nan
    return df


def load_base_data() -> pd.DataFrame:
    data_dir = get_results_dir("overall")
    participants = load_participants(data_dir)
    ucla = load_ucla_scores(data_dir).rename(columns={"ucla_total": "ucla_score"})
    dass = load_dass_scores(data_dir)

    base = participants.merge(ucla, on="participant_id", how="inner")
    base = base.merge(dass, on="participant_id", how="left")
    base["gender_male"] = base["gender"].map({"male": 1, "female": 0})
    return base


def load_qc_ids(task: str) -> set[str]:
    task_dir = get_results_dir(task)
    qc_ids_path = task_dir / "filtered_participant_ids.csv"
    if not qc_ids_path.exists():
        return set()
    qc_ids = pd.read_csv(qc_ids_path, encoding="utf-8-sig")
    qc_ids = ensure_participant_id(qc_ids)
    if "participant_id" not in qc_ids.columns:
        return set()
    return set(qc_ids["participant_id"].dropna().astype(str))


def _prepare_phase_means(df: pd.DataFrame, value_col: str, prefix: str) -> pd.DataFrame:
    phase_means = (
        df.groupby(["participant_id", "phase"], observed=False)[value_col]
        .mean()
        .unstack()
        .reset_index()
    )
    for phase in PHASE_ORDER:
        if phase not in phase_means.columns:
            phase_means[phase] = np.nan
    phase_means = phase_means.rename(
        columns={
            "exploration": f"{prefix}_exploration",
            "confirmation": f"{prefix}_confirmation",
            "exploitation": f"{prefix}_exploitation",
        }
    )
    phase_means["exploration_minus_exploitation"] = (
        phase_means[f"{prefix}_exploration"] - phase_means[f"{prefix}_exploitation"]
    )
    phase_means["confirmation_minus_exploitation"] = (
        phase_means[f"{prefix}_confirmation"] - phase_means[f"{prefix}_exploitation"]
    )
    phase_means["exploration_minus_confirmation"] = (
        phase_means[f"{prefix}_exploration"] - phase_means[f"{prefix}_confirmation"]
    )
    return phase_means


def _prepare_pre_exploit_means(df: pd.DataFrame, value_col: str, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df["merged_phase"] = df["phase"].map(
        {
            "exploration": "pre_exploitation",
            "confirmation": "pre_exploitation",
            "exploitation": "exploitation",
        }
    )
    phase_means = (
        df.groupby(["participant_id", "merged_phase"], observed=False)[value_col]
        .mean()
        .unstack()
        .reset_index()
    )
    if "pre_exploitation" not in phase_means.columns:
        phase_means["pre_exploitation"] = np.nan
    if "exploitation" not in phase_means.columns:
        phase_means["exploitation"] = np.nan
    phase_means = phase_means.rename(
        columns={
            "pre_exploitation": f"{prefix}_pre_exploitation",
            "exploitation": f"{prefix}_exploitation",
        }
    )
    phase_means["pre_exploitation_minus_exploitation"] = (
        phase_means[f"{prefix}_pre_exploitation"] - phase_means[f"{prefix}_exploitation"]
    )
    return phase_means


def _read_wcst_trials_for_phase() -> pd.DataFrame:
    data_dir = get_results_dir("overall")
    trials_path = data_dir / "4b_wcst_trials.csv"
    if not trials_path.exists():
        return pd.DataFrame()
    wcst_raw = pd.read_csv(trials_path, encoding="utf-8-sig")
    if wcst_raw.empty:
        return wcst_raw
    wcst = clean_wcst_trials(wcst_raw)
    qc_ids = load_qc_ids("overall")
    if qc_ids:
        wcst = wcst[wcst["participant_id"].isin(qc_ids)].copy()
    return wcst


def _phase_means_alltrials(wcst: pd.DataFrame) -> pd.DataFrame:
    rt_valid = (
        wcst["rt_ms"].between(WCST_RT_MIN, WCST_RT_MAX)
        & (~wcst["timeout"])
        & (wcst["phase"].notna())
    )
    rt_df = wcst[rt_valid].copy()
    means = (
        rt_df.groupby(["participant_id", "phase"], observed=False)["rt_ms"]
        .mean()
        .unstack()
        .reset_index()
    )
    means = means.rename(
        columns={
            "exploration": "wcst_exploration_rt_all",
            "confirmation": "wcst_confirmation_rt_all",
            "exploitation": "wcst_exploitation_rt_all",
        }
    )
    means["wcst_confirmation_minus_exploitation_rt_all"] = (
        means["wcst_confirmation_rt_all"] - means["wcst_exploitation_rt_all"]
    )
    pre_vals = rt_df[rt_df["phase"].isin(["exploration", "confirmation"])]
    pre_mean = pre_vals.groupby("participant_id")["rt_ms"].mean()
    means["wcst_pre_exploitation_rt_all"] = means["participant_id"].map(pre_mean)
    means["wcst_pre_exploitation_minus_exploitation_rt_all"] = (
        means["wcst_pre_exploitation_rt_all"] - means["wcst_exploitation_rt_all"]
    )
    return means


def _run_ucla_regression(df: pd.DataFrame, outcome: str) -> dict[str, float] | None:
    required = [
        outcome,
        "z_ucla_score",
        "z_dass_depression",
        "z_dass_anxiety",
        "z_dass_stress",
        "z_age",
        "gender_male",
    ]
    cols = [c for c in required if c in df.columns]
    sub = df[cols].dropna()
    if len(sub) < 30:
        return None
    formula = (
        f"{outcome} ~ z_ucla_score + z_dass_depression + "
        "z_dass_anxiety + z_dass_stress + z_age + C(gender_male)"
    )
    model = smf.ols(formula, data=sub).fit()
    reduced = smf.ols(
        f"{outcome} ~ z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + C(gender_male)",
        data=sub,
    ).fit()
    delta_r2 = float(model.rsquared - reduced.rsquared)
    return {
        "n": int(len(sub)),
        "ucla_beta": float(model.params.get("z_ucla_score", np.nan)),
        "ucla_se": float(model.bse.get("z_ucla_score", np.nan)),
        "ucla_t": float(model.tvalues.get("z_ucla_score", np.nan)),
        "ucla_p": float(model.pvalues.get("z_ucla_score", np.nan)),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "delta_r2_ucla": delta_r2,
    }


def _run_phase_regressions(
    phase_df: pd.DataFrame,
    outcomes: list[tuple[str, str]],
) -> pd.DataFrame:
    master = load_master_dataset(task="overall", merge_trial_features=False)
    qc_ids = load_qc_ids("overall")
    if qc_ids:
        master = master[master["participant_id"].isin(qc_ids)].copy()
    merged = master.merge(phase_df, on="participant_id", how="inner")
    rows: list[dict[str, float]] = []
    for col, label in outcomes:
        res = _run_ucla_regression(merged, col)
        if res is None:
            continue
        rows.append({"outcome": label, **res})
    return pd.DataFrame(rows)


def _compute_phase_complete(confirm_len: int = 3) -> pd.DataFrame:
    wcst = _read_wcst_trials_for_phase()
    if wcst.empty:
        return pd.DataFrame()
    wcst = wcst.dropna(subset=["trial_order"]).copy()
    wcst = label_wcst_phases(wcst, rule_col="rule", trial_col="trial_order", confirm_len=confirm_len)
    phase_means = _phase_means_alltrials(wcst)
    return phase_means


def _compute_phase_threshold_sensitivity(confirm_lens: list[int]) -> pd.DataFrame:
    rows = []
    for confirm_len in confirm_lens:
        phase_means = _compute_phase_complete(confirm_len=confirm_len)
        if phase_means.empty:
            continue
        outcomes = [
            ("wcst_exploration_rt_all", "exploration"),
            ("wcst_confirmation_rt_all", "confirmation"),
            ("wcst_exploitation_rt_all", "exploitation"),
            ("wcst_confirmation_minus_exploitation_rt_all", "confirmation_minus_exploitation"),
        ]
        results = _run_phase_regressions(phase_means, outcomes)
        if results.empty:
            continue
        results["confirm_len"] = confirm_len
        rows.append(results)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _get_complete6_ids(wcst: pd.DataFrame) -> set[str]:
    if wcst.empty:
        return set()
    wcst = wcst.sort_values(["participant_id", "trial_order"]).copy()
    cat_counts = (
        wcst.groupby("participant_id")["rule"]
        .apply(lambda s: s.ne(s.shift()).sum())
    )
    return set(cat_counts[cat_counts >= 6].index.astype(str))


def _compute_phase_complete_complete6(confirm_len: int = 3) -> pd.DataFrame:
    wcst = _read_wcst_trials_for_phase()
    if wcst.empty:
        return pd.DataFrame()
    complete6_ids = _get_complete6_ids(wcst)
    if not complete6_ids:
        return pd.DataFrame()
    wcst = wcst[wcst["participant_id"].isin(complete6_ids)].copy()
    wcst = wcst.dropna(subset=["trial_order"]).copy()
    wcst = label_wcst_phases(wcst, rule_col="rule", trial_col="trial_order", confirm_len=confirm_len)
    phase_means = _phase_means_alltrials(wcst)
    return phase_means


def run_phase_complete_outputs(confirm_len: int = 3) -> pd.DataFrame:
    output_dir = get_output_dir("overall", bucket="supplementary")
    phase_complete = _compute_phase_complete(confirm_len=confirm_len)
    if phase_complete.empty:
        return pd.DataFrame()
    # Keep this stage-complete output focused on exploration-only effects.
    # Primary confirmation / confirmation-minus-exploitation results are reported
    # from core hierarchical models (N=212), avoiding duplicate rows with a stricter
    # stage-complete subset (N=208).
    outcomes = [
        ("wcst_exploration_rt_all", "exploration"),
    ]
    phase_complete_results = _run_phase_regressions(phase_complete, outcomes)
    if not phase_complete_results.empty:
        phase_complete_results.to_csv(
            output_dir / "wcst_phase_3phase_complete_ols_alltrials.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return phase_complete_results


def run_phase_threshold_sensitivity_outputs(confirm_lens: list[int] | None = None) -> pd.DataFrame:
    if confirm_lens is None:
        confirm_lens = [2, 4]
    output_dir = get_output_dir("overall", bucket="supplementary")
    threshold_results = _compute_phase_threshold_sensitivity(confirm_lens)
    if not threshold_results.empty:
        threshold_results.to_csv(
            output_dir / "wcst_phase_3phase_threshold_sensitivity_complete_ols_alltrials.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return threshold_results


def run_complete6_outputs(confirm_len: int = 3) -> pd.DataFrame:
    output_dir = get_output_dir("overall", bucket="supplementary")
    phase_complete = _compute_phase_complete_complete6(confirm_len=confirm_len)
    if phase_complete.empty:
        return pd.DataFrame()
    outcomes = [
        ("wcst_exploration_rt_all", "exploration"),
        ("wcst_confirmation_rt_all", "confirmation"),
        ("wcst_exploitation_rt_all", "exploitation"),
        ("wcst_confirmation_minus_exploitation_rt_all", "confirmation_minus_exploitation"),
        ("wcst_pre_exploitation_rt_all", "pre_exploitation"),
        ("wcst_pre_exploitation_minus_exploitation_rt_all", "pre_exploitation_minus_exploitation"),
    ]
    results = _run_phase_regressions(phase_complete, outcomes)
    if not results.empty:
        results.to_csv(
            output_dir / "wcst_phase_3_2phase_6categories_ols_alltrials.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return results


def main(confirm_len: int, include_errors: bool, use_log: bool, merge_pre: bool) -> None:
    base = load_base_data()
    base = add_zscores(
        base,
        ["ucla_score", "dass_depression", "dass_anxiety", "dass_stress", "age"],
    )

    prepared = prepare_wcst_trials()
    wcst = prepared["wcst"]
    rt_col = prepared["rt_col"]
    trial_col = prepared["trial_col"]
    rule_col = prepared["rule_col"]
    if not isinstance(wcst, pd.DataFrame) or wcst.empty or rt_col is None or trial_col is None or rule_col is None:
        raise RuntimeError("WCST trials missing required columns.")

    qc_ids = load_qc_ids("overall")
    if qc_ids:
        wcst = wcst[wcst["participant_id"].isin(qc_ids)]

    wcst["rt_ms"] = pd.to_numeric(wcst[rt_col], errors="coerce")
    wcst = wcst[wcst["rt_ms"].notna()]
    if "is_rt_valid" in wcst.columns:
        wcst = wcst[wcst["is_rt_valid"] == True]
    if "timeout" in wcst.columns:
        wcst = wcst[~wcst["timeout"]]

    wcst["rule"] = wcst[rule_col].astype(str).str.lower().replace({"color": "colour"})
    wcst = label_wcst_phases(wcst, rule_col="rule", trial_col=trial_col, confirm_len=confirm_len)
    if not include_errors:
        wcst = wcst[wcst["correct"].astype(bool)]

    if use_log:
        wcst["log_rt"] = np.where(wcst["rt_ms"] > 0, np.log(wcst["rt_ms"]), np.nan)
        value_col = "log_rt"
        prefix = "log_rt"
    else:
        value_col = "rt_ms"
        prefix = "rt_ms"
    wcst = wcst.dropna(subset=[value_col, "phase"])

    if merge_pre:
        phase_means = _prepare_pre_exploit_means(wcst, value_col=value_col, prefix=prefix)
    else:
        phase_means = _prepare_phase_means(wcst, value_col=value_col, prefix=prefix)
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

    data = phase_means.merge(predictors, on="participant_id", how="inner")

    if merge_pre:
        outcomes = [
            ("pre_exploitation", f"{prefix}_pre_exploitation", "phase_mean"),
            ("exploitation", f"{prefix}_exploitation", "phase_mean"),
            ("pre_exploitation_minus_exploitation", "pre_exploitation_minus_exploitation", "phase_contrast"),
        ]
    else:
        outcomes = [
            ("exploration", f"{prefix}_exploration", "phase_mean"),
            ("confirmation", f"{prefix}_confirmation", "phase_mean"),
            ("exploitation", f"{prefix}_exploitation", "phase_mean"),
            ("exploration_minus_exploitation", "exploration_minus_exploitation", "phase_contrast"),
            ("confirmation_minus_exploitation", "confirmation_minus_exploitation", "phase_contrast"),
            ("exploration_minus_confirmation", "exploration_minus_confirmation", "phase_contrast"),
        ]

    results: list[dict[str, object]] = []
    for label, col, out_type in outcomes:
        res = run_ucla_regression(data, col, cov_type="nonrobust")
        if res is None:
            continue
        res["outcome"] = label
        res["outcome_type"] = out_type
        res["confirm_len"] = confirm_len
        res["include_errors"] = include_errors
        res["use_log"] = use_log
        res["merge_pre"] = merge_pre
        results.append(res)

    results_df = pd.DataFrame(results)
    output_dir = get_output_dir("overall", bucket="supplementary")
    if use_log and not include_errors and not merge_pre:
        if confirm_len == 3:
            filename = "wcst_phase_rt_log_ols.csv"
        else:
            filename = f"wcst_phase_rt_log_ols_m{confirm_len}.csv"
    else:
        if merge_pre:
            base_name = "wcst_phase_pre_exploit_rt_log_ols" if use_log else "wcst_phase_pre_exploit_rt_ols"
        else:
            base_name = "wcst_phase_rt_log_ols" if use_log else "wcst_phase_rt_ols"
        if confirm_len == 3:
            filename = base_name
        else:
            filename = f"{base_name}_m{confirm_len}"
        if include_errors:
            filename = f"{filename}_alltrials"
        filename = f"{filename}.csv"
    out_path = output_dir / filename
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path}")
    if not results_df.empty:
        print(results_df[["outcome", "n", "ucla_beta", "ucla_p", "cov_type"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-len", type=int, default=3)
    parser.add_argument("--include-errors", action="store_true")
    parser.add_argument("--use-log", action="store_true")
    parser.add_argument("--merge-pre", action="store_true")
    args = parser.parse_args()
    main(args.confirm_len, args.include_errors, args.use_log, args.merge_pre)


