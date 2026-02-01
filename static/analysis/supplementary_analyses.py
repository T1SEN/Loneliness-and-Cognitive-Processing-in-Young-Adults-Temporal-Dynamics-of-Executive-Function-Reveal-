"""
Supplementary analyses for manuscript alignment.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm

from static.analysis.utils import get_output_dir, get_primary_outcomes, get_analysis_data
from static.preprocessing.constants import (
    STROOP_RT_MIN,
    STROOP_RT_MAX,
    WCST_RT_MIN,
    WCST_RT_MAX,
    get_results_dir,
)
from static.preprocessing.core import ensure_participant_id
from static.preprocessing.datasets import load_master_dataset
from static.preprocessing.wcst.qc import clean_wcst_trials
from static.preprocessing.wcst.phase import label_wcst_phases


def _read_csv(path: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except FileNotFoundError:
        return pd.DataFrame()


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "t", "yes"})


def _load_qc_ids(task: str) -> set[str]:
    task_dir = get_results_dir(task)
    ids_path = task_dir / "filtered_participant_ids.csv"
    if not ids_path.exists():
        return set()
    ids_df = pd.read_csv(ids_path, encoding="utf-8-sig")
    ids_df = ensure_participant_id(ids_df)
    if "participant_id" not in ids_df.columns:
        return set()
    return set(ids_df["participant_id"].dropna().astype(str))


def _prepare_stroop_trials() -> pd.DataFrame:
    data_dir = get_results_dir("overall")
    trials = _read_csv(str(data_dir / "4a_stroop_trials.csv"))
    if trials.empty:
        return trials
    trials = ensure_participant_id(trials)
    qc_ids = _load_qc_ids("overall")
    if qc_ids:
        trials = trials[trials["participant_id"].isin(qc_ids)]
    trials["is_rt_valid"] = _coerce_bool(trials["is_rt_valid"])
    trials["timeout"] = _coerce_bool(trials["timeout"])
    trials["correct"] = _coerce_bool(trials["correct"])
    trials["cond"] = trials["cond"].astype(str).str.lower()
    trials["rt_ms"] = pd.to_numeric(trials["rt_ms"], errors="coerce")
    if "trial_index" in trials.columns:
        trials["trial_order"] = pd.to_numeric(trials["trial_index"], errors="coerce")
    else:
        trials["trial_order"] = pd.to_numeric(trials["trial_order"], errors="coerce")
    valid = (
        trials["cond"].isin({"congruent", "incongruent"})
        & trials["correct"]
        & (~trials["timeout"])
        & trials["is_rt_valid"]
        & trials["rt_ms"].between(STROOP_RT_MIN, STROOP_RT_MAX)
    )
    return trials[valid].dropna(subset=["participant_id", "trial_order", "rt_ms"])


def _prepare_wcst_trials() -> pd.DataFrame:
    data_dir = get_results_dir("overall")
    wcst_raw = _read_csv(str(data_dir / "4b_wcst_trials.csv"))
    if wcst_raw.empty:
        return wcst_raw
    wcst = clean_wcst_trials(wcst_raw)
    qc_ids = _load_qc_ids("overall")
    if qc_ids:
        wcst = wcst[wcst["participant_id"].isin(qc_ids)].copy()
    return wcst


def _compute_stroop_interference_ttest() -> pd.DataFrame:
    trials = _prepare_stroop_trials()
    if trials.empty:
        return pd.DataFrame()
    means = trials.groupby(["participant_id", "cond"])["rt_ms"].mean().unstack()
    if "incongruent" not in means.columns or "congruent" not in means.columns:
        return pd.DataFrame()
    interference = means["incongruent"] - means["congruent"]
    interference = interference.dropna()
    n = int(len(interference))
    t_stat, p_val = stats.ttest_1samp(interference, 0.0)
    return pd.DataFrame(
        [
            {
                "n": n,
                "mean": float(interference.mean()),
                "sd": float(interference.std(ddof=1)),
                "t": float(t_stat),
                "p": float(p_val),
            }
        ]
    )


def _compute_wcst_normative_stats() -> pd.DataFrame:
    wcst = _prepare_wcst_trials()
    if wcst.empty:
        return pd.DataFrame()
    wcst = wcst.sort_values(["participant_id", "trial_order"])
    cat_counts = (
        wcst.groupby("participant_id")["rule"]
        .apply(lambda s: s.ne(s.shift()).sum())
        .rename("n_categories")
    )
    cat_counts = cat_counts.clip(upper=6)
    n = int(cat_counts.shape[0])
    complete6 = int((cat_counts >= 6).sum())
    pct_complete6 = float(complete6 / n * 100) if n else np.nan
    return pd.DataFrame(
        [
            {
                "n": n,
                "mean_categories": float(cat_counts.mean()),
                "sd_categories": float(cat_counts.std(ddof=1)),
                "n_complete6": complete6,
                "pct_complete6": pct_complete6,
            }
        ]
    )


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


def _run_phase_regressions(phase_df: pd.DataFrame, outcomes: list[tuple[str, str]]) -> pd.DataFrame:
    master = load_master_dataset(task="overall", merge_trial_features=False)
    qc_ids = _load_qc_ids("overall")
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
    return {
        "n": int(len(sub)),
        "ucla_beta": float(model.params.get("z_ucla_score", np.nan)),
        "ucla_se": float(model.bse.get("z_ucla_score", np.nan)),
        "ucla_t": float(model.tvalues.get("z_ucla_score", np.nan)),
        "ucla_p": float(model.pvalues.get("z_ucla_score", np.nan)),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
    }


def _compute_phase_complete(confirm_len: int = 3) -> pd.DataFrame:
    wcst = _prepare_wcst_trials()
    if wcst.empty:
        return pd.DataFrame()
    wcst = wcst.dropna(subset=["trial_order"]).copy()
    wcst = label_wcst_phases(wcst, rule_col="rule", trial_col="trial_order", confirm_len=confirm_len)
    phase_means = _phase_means_alltrials(wcst)
    required = [
        "wcst_exploration_rt_all",
        "wcst_confirmation_rt_all",
        "wcst_exploitation_rt_all",
    ]
    phase_means = phase_means.dropna(subset=required)
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


def _compute_dass_total_models() -> pd.DataFrame:
    df = get_analysis_data("overall")
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["dass_total"] = df[["dass_depression", "dass_anxiety", "dass_stress"]].sum(axis=1)
    std = df["dass_total"].std()
    df["z_dass_total"] = (df["dass_total"] - df["dass_total"].mean()) / std if std else np.nan
    outcomes = get_primary_outcomes("overall")
    rows = []
    for outcome, label in outcomes:
        required = ["z_ucla_score", "z_dass_total", "z_age", "gender_male", outcome]
        sub = df[required].dropna()
        if len(sub) < 30:
            continue
        formula = f"{outcome} ~ z_ucla_score + z_dass_total + z_age + C(gender_male)"
        model = smf.ols(formula, data=sub).fit()
        rows.append(
            {
                "outcome": label,
                "outcome_column": outcome,
                "n": int(len(sub)),
                "ucla_beta": float(model.params.get("z_ucla_score", np.nan)),
                "ucla_p": float(model.pvalues.get("z_ucla_score", np.nan)),
                "dass_total_beta": float(model.params.get("z_dass_total", np.nan)),
                "dass_total_p": float(model.pvalues.get("z_dass_total", np.nan)),
                "r2": float(model.rsquared),
                "adj_r2": float(model.rsquared_adj),
            }
        )
    return pd.DataFrame(rows)


def _compute_ucla_first_models() -> pd.DataFrame:
    df = get_analysis_data("overall")
    if df.empty:
        return pd.DataFrame()
    outcomes = get_primary_outcomes("overall")
    rows = []
    for outcome, label in outcomes:
        required = [
            outcome,
            "z_ucla_score",
            "z_dass_depression",
            "z_dass_anxiety",
            "z_dass_stress",
            "z_age",
            "gender_male",
        ]
        sub = df[required].dropna()
        if len(sub) < 30:
            continue
        m0 = smf.ols(f"{outcome} ~ z_age + C(gender_male)", data=sub).fit()
        m1 = smf.ols(f"{outcome} ~ z_age + C(gender_male) + z_ucla_score", data=sub).fit()
        m2 = smf.ols(
            f"{outcome} ~ z_age + C(gender_male) + z_ucla_score + "
            "z_dass_depression + z_dass_anxiety + z_dass_stress",
            data=sub,
        ).fit()
        try:
            anova_1v0 = anova_lm(m0, m1)
            p_ucla = float(anova_1v0["Pr(>F)"][1])
        except Exception:
            p_ucla = float(m1.pvalues.get("z_ucla_score", np.nan))
        try:
            anova_2v1 = anova_lm(m1, m2)
            p_dass = float(anova_2v1["Pr(>F)"][1])
        except Exception:
            p_dass = float(m2.f_pvalue)
        rows.append(
            {
                "outcome": label,
                "outcome_column": outcome,
                "n": int(len(sub)),
                "model0_r2": float(m0.rsquared),
                "model1_r2": float(m1.rsquared),
                "model2_r2": float(m2.rsquared),
                "delta_r2_ucla_first": float(m1.rsquared - m0.rsquared),
                "delta_r2_dass_after_ucla": float(m2.rsquared - m1.rsquared),
                "p_ucla_first": p_ucla,
                "p_dass_after_ucla": p_dass,
            }
        )
    return pd.DataFrame(rows)


def _bh_fdr(pvals: list[float]) -> list[float]:
    pvals = np.array(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.arange(1, n + 1)
    p_sorted = pvals[order]
    q_sorted = p_sorted * n / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)
    q = np.empty_like(pvals)
    q[order] = q_sorted
    return q.tolist()


def _compute_fdr_table() -> pd.DataFrame:
    output_dir = get_output_dir("overall")
    results_path = output_dir / "hierarchical_results.csv"
    if not results_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(results_path, encoding="utf-8-sig")
    if df.empty or "ucla_p" not in df.columns:
        return pd.DataFrame()
    pvals = df["ucla_p"].astype(float).tolist()
    qvals = _bh_fdr(pvals)
    df_out = df[["outcome", "outcome_column", "ucla_p"]].copy()
    df_out["fdr_q"] = qvals
    return df_out


def run(task: str = "overall") -> dict[str, pd.DataFrame]:
    output_dir = get_output_dir(task)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stroop t-test
    stroop_ttest = _compute_stroop_interference_ttest()
    if not stroop_ttest.empty:
        stroop_ttest.to_csv(output_dir / "stroop_interference_ttest.csv", index=False, encoding="utf-8-sig")

    # WCST normative stats
    wcst_norms = _compute_wcst_normative_stats()
    if not wcst_norms.empty:
        wcst_norms.to_csv(output_dir / "wcst_normative_stats.csv", index=False, encoding="utf-8-sig")

    # Phase-complete (confirm_len=3)
    phase_complete = _compute_phase_complete(confirm_len=3)
    if not phase_complete.empty:
        outcomes = [
            ("wcst_exploration_rt_all", "exploration"),
            ("wcst_confirmation_rt_all", "confirmation"),
            ("wcst_exploitation_rt_all", "exploitation"),
            ("wcst_confirmation_minus_exploitation_rt_all", "confirmation_minus_exploitation"),
            ("wcst_pre_exploitation_rt_all", "pre_exploitation"),
            ("wcst_pre_exploitation_minus_exploitation_rt_all", "pre_exploitation_minus_exploitation"),
        ]
        phase_complete_results = _run_phase_regressions(phase_complete, outcomes)
        phase_complete_results.to_csv(
            output_dir / "wcst_phase_3phase_complete_ols_alltrials.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # Threshold sensitivity (confirm_len=2,4) on phase-complete sample
    threshold_results = _compute_phase_threshold_sensitivity([2, 4])
    if not threshold_results.empty:
        threshold_results.to_csv(
            output_dir / "wcst_phase_3phase_threshold_sensitivity_complete_ols_alltrials.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # DASS total model
    dass_total = _compute_dass_total_models()
    if not dass_total.empty:
        dass_total.to_csv(output_dir / "dass_total_models.csv", index=False, encoding="utf-8-sig")

    # UCLA-first model
    ucla_first = _compute_ucla_first_models()
    if not ucla_first.empty:
        ucla_first.to_csv(output_dir / "ucla_first_model_comparison.csv", index=False, encoding="utf-8-sig")

    # FDR table
    fdr_table = _compute_fdr_table()
    if not fdr_table.empty:
        fdr_table.to_csv(output_dir / "ucla_fdr_qvalues.csv", index=False, encoding="utf-8-sig")

    return {
        "stroop_ttest": stroop_ttest,
        "wcst_norms": wcst_norms,
        "phase_complete": phase_complete,
        "threshold_sensitivity": threshold_results,
        "dass_total": dass_total,
        "ucla_first": ucla_first,
        "fdr": fdr_table,
    }


if __name__ == "__main__":
    run("overall")
