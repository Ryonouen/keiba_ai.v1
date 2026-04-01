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
