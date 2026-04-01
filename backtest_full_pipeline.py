"""
フルパイプライン バックテスト
LightGBM + value_ai.py を 2021〜2024 年に適用し、年別・券種別 ROI を検証する。
"""
from __future__ import annotations

import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── 定数 ──────────────────────────────────────────────────────────────
TRAINING_CSV = "keiba_training_data.csv"
TARGET_YEARS = [2021, 2022, 2023, 2024]
BANKROLL = 10_000        # 1レース当たり仮想軍資金（円）
STAKE_UNIT = 100         # 最小掛け金単位

ML_FEATURE_COLUMNS = [
    "feat_gate", "feat_age", "feat_popularity", "feat_win_odds_log",
    "feat_last3f", "feat_jockey_weight", "feat_n_runners",
    "feat_running_style_enc", "feat_track_condition_enc",
    "feat_signal_total_adjust",
    "feat_cond_diff_age", "feat_cond_diff_gate", "feat_cond_diff_style",
    "feat_cond_diff_popularity", "feat_cond_diff_last3f",
    "feat_cond_diff_weight", "feat_cond_diff_jockey", "feat_cond_diff_track",
]

# running_style_enc の逆引き（pace_balance 用）
_ENC_TO_STYLE = {0: "front", 1: "stalker", 2: "closer", 3: "unknown"}

# PLACE_ODDS_FACTORS（value_ai.py と同値）
_PLACE_FACTORS = {1: 1.55, 2: 1.90, 3: 1.90, 4: 2.50, 5: 2.50, 6: 2.50,
                  7: 3.10, 8: 3.10, 9: 3.10}
_PLACE_FACTOR_DEFAULT = 4.00


def load_and_group_csv(
    csv_path: str = TRAINING_CSV,
    years: Optional[List[int]] = None,
) -> Dict[int, Dict[str, List[Dict[str, Any]]]]:
    """
    CSVを読み込み {year: {race_id: [row_dict, ...]}} 形式で返す。

    Args:
        csv_path: CSVファイルパス
        years: 対象年リスト。None の場合は TARGET_YEARS を使用

    Returns:
        {year(int): {race_id(str): [row(dict), ...]}}

    Raises:
        FileNotFoundError: csv_path が存在しない場合（pandas から伝播）
    """
    if years is None:
        years = TARGET_YEARS

    df = pd.read_csv(csv_path, low_memory=False)
    df["year"] = pd.to_datetime(df["race_date"], errors="coerce").dt.year
    na_count = df["year"].isna().sum()
    if na_count > 0:
        print(f"[警告] race_date パース失敗: {na_count}/{len(df)} 行を除外します")
    df = df[df["year"].isin(years)].copy()
    if len(df) == 0:
        print(f"[警告] 指定年 {years} に該当するデータが0件です。CSVパスとyears引数を確認してください。")

    # ML特徴量の欠損を 0 で埋める
    for col in ML_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    result: Dict[int, Dict[str, List[Dict[str, Any]]]] = defaultdict(dict)
    for _, row in df.iterrows():
        year = int(row["year"])
        race_id = str(row["race_id"])
        if race_id not in result[year]:
            result[year][race_id] = []
        result[year][race_id].append(row.to_dict())

    return dict(result)


def build_feature_dict(row: Dict[str, Any], win_prob: float) -> Dict[str, Any]:
    """
    CSV の 1行 dict を value_ai.py が期待するキー形式に変換する。

    - win_odds: exp(feat_win_odds_log) で復元
    - running_style: enc 値を文字列に変換
    - jockey_delta / place_odds: 未収録のためデフォルト値

    Raises:
        なし（不正値はデフォルト値にフォールバック）
    """
    val_odds_log = row.get("feat_win_odds_log")
    raw_odds_log = float(val_odds_log) if val_odds_log is not None else 0.0
    win_odds = math.exp(raw_odds_log) if raw_odds_log != 0.0 else None

    val_enc = row.get("feat_running_style_enc")
    enc = int(float(val_enc)) if val_enc is not None else 3
    running_style = _ENC_TO_STYLE.get(enc, "unknown")

    val_pop = row.get("feat_popularity")
    pop = int(float(val_pop)) if val_pop is not None else 99
    place_factor = _PLACE_FACTORS.get(pop, _PLACE_FACTOR_DEFAULT)
    place_odds = round(win_odds * place_factor / 100, 2) if win_odds else None

    val_last3f = row.get("feat_last3f")
    last3f = float(val_last3f) if val_last3f is not None else 0.0

    val_gate = row.get("feat_gate")
    gate = int(float(val_gate)) if val_gate is not None else 0

    val_age = row.get("feat_age")
    age = int(float(val_age)) if val_age is not None else 0

    val_runners = row.get("feat_n_runners")
    n_runners = int(float(val_runners)) if val_runners is not None else 0

    val_target_win = row.get("target_win")
    target_win = int(float(val_target_win)) if val_target_win is not None else 0

    val_target_top3 = row.get("target_top3")
    target_top3 = int(float(val_target_top3)) if val_target_top3 is not None else 0

    return {
        "horse_name":       str(row.get("horse_name") or ""),
        "win_prob":         win_prob,
        "model_score":      win_prob,
        "win_odds":         win_odds,
        "place_odds":       place_odds,
        "popularity_rank":  pop,
        "running_style":    running_style,
        "last3f":           last3f,
        "gate":             gate,
        "age":              age,
        "n_runners":        n_runners,
        "jockey_delta":     0.0,   # CSV未収録
        "jockey_reason_codes": [],
        # 正解ラベル（払戻計算用、value_ai には渡さない）
        "_target_win":      target_win,
        "_target_top3":     target_top3,
        "_win_odds_log":    raw_odds_log,
    }


def build_pace_balance(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    レース内の脚質カウントを返す（classify_race_structure の pace_balance 用）。

    Returns:
        {"逃げ": n, "先行": n, "差し": n, "追込": n}
    """
    pb: Dict[str, int] = {"逃げ": 0, "先行": 0, "差し": 0, "追込": 0}
    style_map = {0: "逃げ", 1: "先行", 2: "差し", 3: "追込"}
    for row in rows:
        val = row.get("feat_running_style_enc")
        enc = int(float(val)) if val is not None else 3
        key = style_map.get(enc)
        if key:
            pb[key] += 1
    return pb
