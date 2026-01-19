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
N_CATEGORIES = 6


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


def assign_category_numbers(
    wcst: pd.DataFrame,
    rule_col: str,
    trial_col: str,
) -> pd.DataFrame:
    df = wcst.sort_values(["participant_id", trial_col]).copy()
    df["category_num"] = np.nan

    for pid, grp in df.groupby("participant_id"):
        grp_sorted = grp.sort_values(trial_col).copy()
        idxs = grp_sorted.index.to_list()
        rules = (
            grp_sorted[rule_col].astype(str).str.lower().replace({"color": "colour"}).to_numpy()
        )

        change_indices = [i for i in range(1, len(rules)) if rules[i] != rules[i - 1]]
        segment_starts = [0] + change_indices
        segment_ends = change_indices + [len(rules)]

        for cat_idx, (start, end) in enumerate(zip(segment_starts, segment_ends), start=1):
            if cat_idx > N_CATEGORIES:
                break
            if start >= end:
                continue
            for i in range(start, end):
                df.at[idxs[i], "category_num"] = float(cat_idx)

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
    wcst = assign_category_numbers(wcst, rule_col="rule", trial_col=trial_col)
    wcst["log_rt"] = np.where(wcst["rt_ms"] > 0, np.log(wcst["rt_ms"]), np.nan)
    wcst = wcst.dropna(subset=["log_rt", "category_num"])

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

    formula_overall = (
        "log_rt ~ z_ucla_score + z_dass_depression + "
        "z_dass_anxiety + z_dass_stress + z_age + gender_male"
    )
    result_overall, re_overall, method_overall, warn_overall = fit_with_fallback(
        formula_overall,
        df,
        ["1"],
    )

    formula_category = (
        "log_rt ~ category_num * z_ucla_score + z_dass_depression + "
        "z_dass_anxiety + z_dass_stress + z_age + gender_male"
    )
    result_category, re_category, method_category, warn_category = fit_with_fallback(
        formula_category,
        df,
        ["1 + category_num", "1"],
    )

    rows: list[dict[str, object]] = []

    ucla_overall = _get_term(result_overall, "z_ucla_score")
    rows.append({
        "model": "overall_mean_lmm",
        "n_trials": int(len(df)),
        "n_participants": int(df["participant_id"].nunique()),
        "re_formula": re_overall,
        "method": method_overall,
        "warning_count": len(warn_overall),
        "ucla_beta": ucla_overall["beta"],
        "ucla_se": ucla_overall["se"],
        "ucla_z": ucla_overall["z"],
        "ucla_p": ucla_overall["p"],
    })

    ucla_main = _get_term(result_category, "z_ucla_score")
    ucla_slope = _get_term(result_category, "category_num:z_ucla_score")
    rows.append({
        "model": "category_slope_lmm",
        "n_trials": int(len(df)),
        "n_participants": int(df["participant_id"].nunique()),
        "re_formula": re_category,
        "method": method_category,
        "warning_count": len(warn_category),
        "ucla_beta": ucla_main["beta"],
        "ucla_se": ucla_main["se"],
        "ucla_z": ucla_main["z"],
        "ucla_p": ucla_main["p"],
        "category_x_ucla_beta": ucla_slope["beta"],
        "category_x_ucla_se": ucla_slope["se"],
        "category_x_ucla_z": ucla_slope["z"],
        "category_x_ucla_p": ucla_slope["p"],
    })

    out_df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "wcst_overall_category_rt_lmm.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(out_df.to_string(index=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
