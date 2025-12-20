"""WCST WSLS mechanism feature derivation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..constants import get_results_dir
from .loaders import load_wcst_trials

MECHANISM_FILENAME = "5_wcst_wsls_mechanism_features.csv"

REFERENCE_CARDS = [
    {"name": "one_yellow_circle", "count": 1, "color": "yellow", "shape": "circle"},
    {"name": "two_black_rectangle", "count": 2, "color": "black", "shape": "rectangle"},
    {"name": "three_blue_star", "count": 3, "color": "blue", "shape": "star"},
    {"name": "four_red_triangle", "count": 4, "color": "red", "shape": "triangle"},
]

CARD_ATTRS = {card["name"]: card for card in REFERENCE_CARDS}
CARD_COLORS = {name: attrs["color"] for name, attrs in CARD_ATTRS.items()}
CARD_SHAPES = {name: attrs["shape"] for name, attrs in CARD_ATTRS.items()}
CARD_NUMBERS = {name: attrs["count"] for name, attrs in CARD_ATTRS.items()}


def _pick_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for cand in candidates:
        if cand in df.columns:
            return cand
    return None


def _prepare_wsls_trials(df: pd.DataFrame) -> pd.DataFrame:
    chosen_col = _pick_column(df, ["chosenCard", "chosen_card"])
    color_col = _pick_column(df, ["cardColor", "card_color", "color"])
    shape_col = _pick_column(df, ["cardShape", "card_shape", "shape"])
    number_col = _pick_column(df, ["cardNumber", "card_number", "count"])
    trial_col = _pick_column(df, ["trialIndex", "trial_index", "trial"])

    required = [chosen_col, color_col, shape_col, number_col, "correct", "participant_id"]
    if any(col is None for col in required):
        return pd.DataFrame()

    out = df.copy()
    out["chosen_card"] = out[chosen_col].astype(str).str.strip().str.lower()
    out["stim_color_raw"] = out[color_col]
    out["stim_shape_raw"] = out[shape_col]
    out["stim_number_raw"] = out[number_col]

    out["stim_color"] = out["stim_color_raw"].astype(str).str.strip().str.lower()
    out["stim_shape"] = out["stim_shape_raw"].astype(str).str.strip().str.lower()
    out["stim_number"] = pd.to_numeric(out["stim_number_raw"], errors="coerce")

    out["chosen_color"] = out["chosen_card"].map(CARD_COLORS)
    out["chosen_shape"] = out["chosen_card"].map(CARD_SHAPES)
    out["chosen_number"] = out["chosen_card"].map(CARD_NUMBERS)

    valid_mask = (
        out["chosen_card"].isin(CARD_ATTRS)
        & out["stim_color_raw"].notna()
        & out["stim_shape_raw"].notna()
        & out["stim_number_raw"].notna()
        & out["correct"].notna()
    )

    match_color = out["chosen_color"] == out["stim_color"]
    match_shape = out["chosen_shape"] == out["stim_shape"]
    match_number = out["chosen_number"] == out["stim_number"]
    match_count = match_color.astype(int) + match_shape.astype(int) + match_number.astype(int)

    out["rule_choice"] = np.nan
    out.loc[valid_mask & (match_count == 1) & match_color, "rule_choice"] = "colour"
    out.loc[valid_mask & (match_count == 1) & match_shape, "rule_choice"] = "shape"
    out.loc[valid_mask & (match_count == 1) & match_number, "rule_choice"] = "number"

    if trial_col:
        out["trial_order"] = pd.to_numeric(out[trial_col], errors="coerce")
    else:
        out["trial_order"] = np.arange(len(out))

    return out


def compute_wcst_wsls_features(
    data_dir: Path | None = None,
) -> pd.DataFrame:
    trials, _ = load_wcst_trials(data_dir=data_dir, apply_trial_filters=True)
    prepared = _prepare_wsls_trials(trials)
    if prepared.empty:
        return pd.DataFrame()

    records: List[Dict[str, float]] = []
    for pid, grp in prepared.groupby("participant_id"):
        grp = grp.sort_values("trial_order").reset_index(drop=True)
        rule_choice = grp["rule_choice"]
        correct = grp["correct"]

        prev_rule = rule_choice.shift(1)
        prev_correct = correct.shift(1)
        valid_pairs = rule_choice.notna() & prev_rule.notna() & prev_correct.notna()

        stay = rule_choice == prev_rule
        win_mask = prev_correct == True
        lose_mask = prev_correct == False

        win_total = int((valid_pairs & win_mask).sum())
        lose_total = int((valid_pairs & lose_mask).sum())
        win_stay = int((valid_pairs & win_mask & stay).sum())
        lose_shift = int((valid_pairs & lose_mask & ~stay).sum())

        p_stay_win = win_stay / win_total if win_total > 0 else np.nan
        p_shift_lose = lose_shift / lose_total if lose_total > 0 else np.nan
        excluded_n = int(rule_choice.isna().sum())

        records.append({
            "participant_id": pid,
            "wcst_wsls_p_stay_win": p_stay_win,
            "wcst_wsls_p_shift_lose": p_shift_lose,
            "wcst_wsls_win_n": win_total,
            "wcst_wsls_lose_n": lose_total,
            "wcst_wsls_pair_n": int(valid_pairs.sum()),
            "wcst_wsls_excluded_n": excluded_n,
        })

    return pd.DataFrame(records)


def load_or_compute_wcst_wsls_mechanism_features(
    data_dir: Path | None = None,
    overwrite: bool = False,
    save: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    if data_dir is None:
        data_dir = get_results_dir("wcst")

    output_path = data_dir / MECHANISM_FILENAME
    if output_path.exists() and not overwrite:
        return pd.read_csv(output_path, encoding="utf-8-sig")

    features = compute_wcst_wsls_features(data_dir=data_dir)
    if save and not features.empty:
        features.to_csv(output_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"[OK] WCST WSLS mechanism features saved: {output_path}")
    return features
