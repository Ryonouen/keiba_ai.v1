"""
feature_expander.py
keiba_training_data.csv に騎手統計・距離・馬場特徴量を追加する。

新規列（EXPANDED_COLS）:
  feat_jockey_overall_win_rate  — 全体勝率
  feat_jockey_overall_top3_rate — 全体複勝率
  feat_jockey_win_rate_by_dist  — 距離帯別勝率
  feat_jockey_win_rate_by_surface — 馬場種別勝率
  feat_jockey_win_rate_by_venue — 会場別勝率
  feat_distance_bucket_enc      — 距離帯エンコード 0-3
  feat_surface_turf_flag        — 芝=1 / ダート=0 / 不明=0

使い方:
  from feature_expander import expand_features, build_jockey_lookup
  python3 feature_expander.py  # CLI: keiba_training_data.csv を上書き
"""
from __future__ import annotations

import argparse
import logging
import math
import os
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_CSV   = os.path.join(_HERE, "keiba_training_data.csv")
JOCKEY_STATS   = os.path.join(_HERE, "jockey_stats.csv")

EXPANDED_COLS = [
    "feat_jockey_overall_win_rate",
    "feat_jockey_overall_top3_rate",
    "feat_jockey_win_rate_by_dist",
    "feat_jockey_win_rate_by_surface",
    "feat_jockey_win_rate_by_venue",
    "feat_distance_bucket_enc",
    "feat_surface_turf_flag",
]

# 距離帯境界 (m)
_DIST_BUCKETS = [
    (0,    1399, 0),   # 短距離
    (1400, 1799, 1),   # マイル
    (1800, 2199, 2),   # 中距離
    (2200, 9999, 3),   # 長距離
]

# jockey_stats.csv の by_distance の condition_value
_DIST_LABEL_MAP = {
    0: "短距離",
    1: "マイル",
    2: "中距離",
    3: "長距離",
}

# 会場コード → 会場名
_VENUE_CODE_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}


# =========================================================
# ヘルパー関数
# =========================================================

def _distance_to_bucket(distance: float) -> int:
    """距離（m）を 0-3 の帯エンコードに変換する。NaN → 0。"""
    try:
        d = float(distance)
        if math.isnan(d):
            return 0
    except (TypeError, ValueError):
        return 0
    for lo, hi, enc in _DIST_BUCKETS:
        if lo <= d <= hi:
            return enc
    return 0


def _surface_to_flag(surface: Optional[str]) -> int:
    """馬場種別を 芝=1 / その他=0 に変換する。"""
    if surface is None:
        return 0
    return 1 if str(surface).strip() == "芝" else 0


# =========================================================
# jockey_stats 読み込み・ルックアップ構築
# =========================================================

def build_jockey_lookup(
    jockey_stats_df: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    """
    jockey_stats.csv の DataFrame から jockey_name → 各種勝率 の dict を構築する。
    """
    lookup: Dict[str, Dict[str, Any]] = {}

    for _, row in jockey_stats_df.iterrows():
        name  = str(row.get("jockey_name") or "").strip()
        ctype = str(row.get("condition_type") or "")
        cval  = str(row.get("condition_value") or "").strip()
        rides = int(row.get("rides") or 0)
        wins  = int(row.get("wins") or 0)
        top3  = int(row.get("top3") or 0)

        if not name or rides == 0:
            continue

        if name not in lookup:
            lookup[name] = {
                "overall_win_rate":  0.0,
                "overall_top3_rate": 0.0,
                "by_distance":       {},
                "by_surface":        {},
                "by_venue":          {},
            }

        win_rate  = wins / rides
        top3_rate = top3 / rides

        if ctype == "overall":
            lookup[name]["overall_win_rate"]  = round(win_rate, 4)
            lookup[name]["overall_top3_rate"] = round(top3_rate, 4)
        elif ctype == "by_distance":
            lookup[name]["by_distance"][cval] = round(win_rate, 4)
        elif ctype == "by_surface":
            lookup[name]["by_surface"][cval]  = round(win_rate, 4)
        elif ctype == "by_track":
            lookup[name]["by_venue"][cval]    = round(win_rate, 4)

    return lookup


# =========================================================
# 特徴量展開
# =========================================================

def expand_features(
    df: pd.DataFrame,
    jockey_stats_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    CSV の DataFrame に EXPANDED_COLS を追加して返す（in-place 変更なし）。
    """
    df = df.copy()
    lookup = build_jockey_lookup(jockey_stats_df)

    # 初期値ゼロ
    for col in EXPANDED_COLS:
        if col not in df.columns:
            df[col] = 0.0

    for idx, row in df.iterrows():
        jname   = str(row.get("jockey_name") or "").strip()
        surface = row.get("surface")
        dist    = row.get("distance")

        # 距離・馬場フラグ（jockey_stats 不要）
        dist_bucket = _distance_to_bucket(dist)
        df.at[idx, "feat_distance_bucket_enc"] = float(dist_bucket)
        df.at[idx, "feat_surface_turf_flag"]   = float(_surface_to_flag(surface))

        if not jname or jname not in lookup:
            continue

        jdata = lookup[jname]
        df.at[idx, "feat_jockey_overall_win_rate"]  = jdata["overall_win_rate"]
        df.at[idx, "feat_jockey_overall_top3_rate"] = jdata["overall_top3_rate"]

        # 距離帯別
        dist_label = _DIST_LABEL_MAP.get(dist_bucket, "")
        df.at[idx, "feat_jockey_win_rate_by_dist"] = jdata["by_distance"].get(dist_label, 0.0)

        # 馬場別
        if surface and str(surface).strip() in jdata["by_surface"]:
            df.at[idx, "feat_jockey_win_rate_by_surface"] = jdata["by_surface"][str(surface).strip()]

        # 会場別（race_id[4:6] = 会場コード）
        race_id_str = str(row.get("race_id") or "")
        if len(race_id_str) >= 6:
            venue_code = race_id_str[4:6]
            venue_name = _VENUE_CODE_MAP.get(venue_code, "")
            df.at[idx, "feat_jockey_win_rate_by_venue"] = jdata["by_venue"].get(venue_name, 0.0)

    logger.info("expand_features: %d 行に %d 列を追加", len(df), len(EXPANDED_COLS))
    return df


# =========================================================
# CLI
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="特徴量拡充: keiba_training_data.csv に騎手統計列を追加")
    parser.add_argument("--csv",          default=TRAINING_CSV, help="入力CSV（上書き）")
    parser.add_argument("--jockey-stats", default=JOCKEY_STATS, help="jockey_stats.csvパス")
    parser.add_argument("--output",       default=None,         help="出力先（省略時は入力CSVを上書き）")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"エラー: {args.csv} が存在しません。")
        return
    if not os.path.exists(args.jockey_stats):
        print(f"エラー: {args.jockey_stats} が存在しません。先に generate_jockey_stats.py を実行してください。")
        return

    print(f"CSV 読み込み中: {args.csv}")
    df = pd.read_csv(args.csv, low_memory=False)
    print(f"  {len(df):,} 行")

    print(f"jockey_stats 読み込み中: {args.jockey_stats}")
    jstats = pd.read_csv(args.jockey_stats)
    print(f"  {len(jstats):,} 行")

    print("特徴量展開中...")
    df_expanded = expand_features(df, jstats)

    out = args.output or args.csv
    df_expanded.to_csv(out, index=False)
    print(f"保存完了: {out}  ({len(df_expanded):,} 行, {len(df_expanded.columns)} 列)")
    for col in EXPANDED_COLS:
        nonzero = (df_expanded[col] != 0).sum()
        print(f"  {col}: {nonzero:,} 行に値あり")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
