from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from publication.preprocessing.constants import get_results_dir
from publication.preprocessing.core import ensure_participant_id
from publication.preprocessing.surveys import load_dass_scores, load_participants, load_ucla_scores
from publication.preprocessing.wcst._shared import prepare_wcst_trials


OUTPUT_DIR = ROOT / "publication" / "data" / "outputs" / "analysis" / "reviewer_response"


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


def _scale_within(x: pd.Series) -> pd.Series:
    denom = x.max() - x.min()
    if denom == 0 or not np.isfinite(denom):
        return pd.Series([np.nan] * len(x), index=x.index)
    return (x - x.min()) / denom


def add_trial_scaled(df: pd.DataFrame, trial_col: str) -> pd.DataFrame:
    df = df.sort_values(["participant_id", trial_col]).copy()
    df["trial_scaled"] = df.groupby("participant_id")[trial_col].transform(_scale_within)
    return df


def add_switch_flag(df: pd.DataFrame, trial_col: str) -> pd.DataFrame:
    df = df.sort_values(["participant_id", trial_col]).copy()
    prev_rule = df.groupby("participant_id")["rule"].shift(1)
    switch = (df["rule"] != prev_rule).astype("boolean")
    switch = switch.mask(prev_rule.isna(), pd.NA)
    df["switch"] = switch
    df["switch_code"] = switch.astype("Int64")
    return df


def add_post_error_flags(df: pd.DataFrame, trial_col: str) -> pd.DataFrame:
    df = df.sort_values(["participant_id", trial_col]).copy()
    if "correct" not in df.columns:
        df["post_error"] = pd.NA
        df["post_error_code"] = pd.NA
        return df
    df["prev_correct"] = df.groupby("participant_id")["correct"].shift(1)
    post_error = df["prev_correct"] == False
    post_error = post_error.astype("boolean").fillna(False)
    df["post_error"] = post_error
    df["post_error_code"] = post_error.astype("Int64")
    return df


def _fit_mixedlm(
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


def fit_with_fallback(
    formula: str,
    df: pd.DataFrame,
    re_formulas: list[str],
) -> tuple[object, str, str, list[str]]:
    last_result = None
    last_re = re_formulas[-1]
    last_method = "lbfgs"
    last_warnings: list[str] = []
    for re_formula in re_formulas:
        for method in ("lbfgs", "powell"):
            try:
                result, warning_msgs = _fit_mixedlm(formula, df, re_formula, method)
                last_result = result
                last_re = re_formula
                last_method = method
                last_warnings = warning_msgs
                if getattr(result, "converged", False) and not warning_msgs:
                    return result, re_formula, method, warning_msgs
            except Exception:
                continue
    if last_result is None:
        raise RuntimeError("All MixedLM attempts failed.")
    return last_result, last_re, last_method, last_warnings


def _get_term(result: object, term: str) -> dict[str, float]:
    params = getattr(result, "params", {})
    bse = getattr(result, "bse", {})
    tvalues = getattr(result, "tvalues", {})
    pvalues = getattr(result, "pvalues", {})
    return {
        "beta": float(params.get(term, np.nan)),
        "se": float(bse.get(term, np.nan)),
        "z": float(tvalues.get(term, np.nan)),
        "p": float(pvalues.get(term, np.nan)),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    qc_ids = load_qc_ids("wcst")
    if qc_ids:
        wcst = wcst[wcst["participant_id"].isin(qc_ids)]

    wcst["rt_ms"] = pd.to_numeric(wcst[rt_col], errors="coerce")
    wcst = wcst[wcst["rt_ms"].notna()]
    if "is_rt_valid" in wcst.columns:
        wcst = wcst[wcst["is_rt_valid"] == True]

    wcst["rule"] = wcst[rule_col].astype(str).str.lower().replace({"color": "colour"})
    wcst[trial_col] = pd.to_numeric(wcst[trial_col], errors="coerce")
    wcst = wcst[wcst[trial_col].notna()]
    wcst["log_rt"] = np.where(wcst["rt_ms"] > 0, np.log(wcst["rt_ms"]), np.nan)

    wcst = add_trial_scaled(wcst, trial_col=trial_col)
    wcst = add_switch_flag(wcst, trial_col=trial_col)
    wcst = add_post_error_flags(wcst, trial_col=trial_col)
    wcst = wcst.dropna(subset=["log_rt"])

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

    df = wcst.merge(predictors, on="participant_id", how="inner")
    if df.empty:
        raise RuntimeError("No WCST trials available after merging predictors.")

    df["switch_code"] = pd.to_numeric(df["switch_code"], errors="coerce")
    df["post_error_code"] = pd.to_numeric(df["post_error_code"], errors="coerce")
    covariates = "z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + gender_male"
    rows: list[dict[str, object]] = []

    df_trial = df.dropna(subset=["trial_scaled"])
    formula_trial = f"log_rt ~ trial_scaled * z_ucla_score + {covariates}"
    result_trial, re_trial, method_trial, warn_trial = fit_with_fallback(
        formula_trial,
        df_trial,
        ["1 + trial_scaled", "1"],
    )
    ucla_trial = _get_term(result_trial, "z_ucla_score")
    trial_main = _get_term(result_trial, "trial_scaled")
    trial_x_ucla = _get_term(result_trial, "trial_scaled:z_ucla_score")
    rows.append({
        "model": "trial_slope_lmm",
        "n_trials": int(len(df_trial)),
        "n_participants": int(df_trial["participant_id"].nunique()),
        "re_formula": re_trial,
        "method": method_trial,
        "warning_count": len(warn_trial),
        "ucla_beta": ucla_trial["beta"],
        "ucla_se": ucla_trial["se"],
        "ucla_z": ucla_trial["z"],
        "ucla_p": ucla_trial["p"],
        "trial_scaled_beta": trial_main["beta"],
        "trial_scaled_se": trial_main["se"],
        "trial_scaled_z": trial_main["z"],
        "trial_scaled_p": trial_main["p"],
        "trial_scaled_x_ucla_beta": trial_x_ucla["beta"],
        "trial_scaled_x_ucla_se": trial_x_ucla["se"],
        "trial_scaled_x_ucla_z": trial_x_ucla["z"],
        "trial_scaled_x_ucla_p": trial_x_ucla["p"],
    })

    df_switch = df.dropna(subset=["switch_code"])
    formula_switch = f"log_rt ~ switch_code * z_ucla_score + {covariates}"
    result_switch, re_switch, method_switch, warn_switch = fit_with_fallback(
        formula_switch,
        df_switch,
        ["1"],
    )
    ucla_switch = _get_term(result_switch, "z_ucla_score")
    switch_main = _get_term(result_switch, "switch_code")
    switch_x_ucla = _get_term(result_switch, "switch_code:z_ucla_score")
    rows.append({
        "model": "switch_lmm",
        "n_trials": int(len(df_switch)),
        "n_participants": int(df_switch["participant_id"].nunique()),
        "re_formula": re_switch,
        "method": method_switch,
        "warning_count": len(warn_switch),
        "ucla_beta": ucla_switch["beta"],
        "ucla_se": ucla_switch["se"],
        "ucla_z": ucla_switch["z"],
        "ucla_p": ucla_switch["p"],
        "switch_beta": switch_main["beta"],
        "switch_se": switch_main["se"],
        "switch_z": switch_main["z"],
        "switch_p": switch_main["p"],
        "switch_x_ucla_beta": switch_x_ucla["beta"],
        "switch_x_ucla_se": switch_x_ucla["se"],
        "switch_x_ucla_z": switch_x_ucla["z"],
        "switch_x_ucla_p": switch_x_ucla["p"],
    })

    if "correct" in df.columns:
        df_correct = df[df["correct"] == True].copy()
    else:
        df_correct = pd.DataFrame()

    if not df_correct.empty:
        df_correct = df_correct.dropna(subset=["trial_scaled", "switch_code"])
        formula_switch_slope = (
            f"log_rt ~ trial_scaled * switch_code * z_ucla_score + {covariates}"
        )
        result_switch_slope, re_switch_slope, method_switch_slope, warn_switch_slope = fit_with_fallback(
            formula_switch_slope,
            df_correct,
            ["1 + trial_scaled + switch_code", "1 + trial_scaled", "1"],
        )
        ucla_switch_slope = _get_term(result_switch_slope, "z_ucla_score")
        trial_main = _get_term(result_switch_slope, "trial_scaled")
        switch_main = _get_term(result_switch_slope, "switch_code")
        trial_x_switch = _get_term(result_switch_slope, "trial_scaled:switch_code")
        trial_x_ucla = _get_term(result_switch_slope, "trial_scaled:z_ucla_score")
        switch_x_ucla = _get_term(result_switch_slope, "switch_code:z_ucla_score")
        trial_switch_ucla = _get_term(
            result_switch_slope,
            "trial_scaled:switch_code:z_ucla_score",
        )
        rows.append({
            "model": "switch_slope_lmm_correct",
            "n_trials": int(len(df_correct)),
            "n_participants": int(df_correct["participant_id"].nunique()),
            "re_formula": re_switch_slope,
            "method": method_switch_slope,
            "warning_count": len(warn_switch_slope),
            "ucla_beta": ucla_switch_slope["beta"],
            "ucla_se": ucla_switch_slope["se"],
            "ucla_z": ucla_switch_slope["z"],
            "ucla_p": ucla_switch_slope["p"],
            "trial_scaled_beta": trial_main["beta"],
            "trial_scaled_se": trial_main["se"],
            "trial_scaled_z": trial_main["z"],
            "trial_scaled_p": trial_main["p"],
            "switch_beta": switch_main["beta"],
            "switch_se": switch_main["se"],
            "switch_z": switch_main["z"],
            "switch_p": switch_main["p"],
            "trial_scaled_x_switch_beta": trial_x_switch["beta"],
            "trial_scaled_x_switch_se": trial_x_switch["se"],
            "trial_scaled_x_switch_z": trial_x_switch["z"],
            "trial_scaled_x_switch_p": trial_x_switch["p"],
            "trial_scaled_x_ucla_beta": trial_x_ucla["beta"],
            "trial_scaled_x_ucla_se": trial_x_ucla["se"],
            "trial_scaled_x_ucla_z": trial_x_ucla["z"],
            "trial_scaled_x_ucla_p": trial_x_ucla["p"],
            "switch_x_ucla_beta": switch_x_ucla["beta"],
            "switch_x_ucla_se": switch_x_ucla["se"],
            "switch_x_ucla_z": switch_x_ucla["z"],
            "switch_x_ucla_p": switch_x_ucla["p"],
            "trial_scaled_x_switch_x_ucla_beta": trial_switch_ucla["beta"],
            "trial_scaled_x_switch_x_ucla_se": trial_switch_ucla["se"],
            "trial_scaled_x_switch_x_ucla_z": trial_switch_ucla["z"],
            "trial_scaled_x_switch_x_ucla_p": trial_switch_ucla["p"],
        })

    df_post = df.dropna(subset=["trial_scaled", "post_error_code"])
    if not df_post.empty:
        formula_post = (
            f"log_rt ~ trial_scaled * post_error_code * z_ucla_score + {covariates}"
        )
        result_post, re_post, method_post, warn_post = fit_with_fallback(
            formula_post,
            df_post,
            ["1 + trial_scaled + post_error_code", "1 + trial_scaled", "1"],
        )
        ucla_post = _get_term(result_post, "z_ucla_score")
        trial_main = _get_term(result_post, "trial_scaled")
        post_main = _get_term(result_post, "post_error_code")
        trial_x_post = _get_term(result_post, "trial_scaled:post_error_code")
        trial_x_ucla = _get_term(result_post, "trial_scaled:z_ucla_score")
        post_x_ucla = _get_term(result_post, "post_error_code:z_ucla_score")
        trial_post_ucla = _get_term(
            result_post,
            "trial_scaled:post_error_code:z_ucla_score",
        )
        rows.append({
            "model": "post_error_slope_lmm",
            "n_trials": int(len(df_post)),
            "n_participants": int(df_post["participant_id"].nunique()),
            "re_formula": re_post,
            "method": method_post,
            "warning_count": len(warn_post),
            "ucla_beta": ucla_post["beta"],
            "ucla_se": ucla_post["se"],
            "ucla_z": ucla_post["z"],
            "ucla_p": ucla_post["p"],
            "trial_scaled_beta": trial_main["beta"],
            "trial_scaled_se": trial_main["se"],
            "trial_scaled_z": trial_main["z"],
            "trial_scaled_p": trial_main["p"],
            "post_error_beta": post_main["beta"],
            "post_error_se": post_main["se"],
            "post_error_z": post_main["z"],
            "post_error_p": post_main["p"],
            "trial_scaled_x_post_error_beta": trial_x_post["beta"],
            "trial_scaled_x_post_error_se": trial_x_post["se"],
            "trial_scaled_x_post_error_z": trial_x_post["z"],
            "trial_scaled_x_post_error_p": trial_x_post["p"],
            "trial_scaled_x_ucla_beta": trial_x_ucla["beta"],
            "trial_scaled_x_ucla_se": trial_x_ucla["se"],
            "trial_scaled_x_ucla_z": trial_x_ucla["z"],
            "trial_scaled_x_ucla_p": trial_x_ucla["p"],
            "post_error_x_ucla_beta": post_x_ucla["beta"],
            "post_error_x_ucla_se": post_x_ucla["se"],
            "post_error_x_ucla_z": post_x_ucla["z"],
            "post_error_x_ucla_p": post_x_ucla["p"],
            "trial_scaled_x_post_error_x_ucla_beta": trial_post_ucla["beta"],
            "trial_scaled_x_post_error_x_ucla_se": trial_post_ucla["se"],
            "trial_scaled_x_post_error_x_ucla_z": trial_post_ucla["z"],
            "trial_scaled_x_post_error_x_ucla_p": trial_post_ucla["p"],
        })

    out_df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "wcst_trial_slope_switch_lmm.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(out_df.to_string(index=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
