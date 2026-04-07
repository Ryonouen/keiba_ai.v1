"""tests/test_feature_expander.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
import feature_expander as fe


# ── テスト用データ ──────────────────────────────────────────────────

def _make_csv_df():
    """CSV の最小サンプル（jockey_name / surface / distance 有り／無しを混在）"""
    return pd.DataFrame([
        {
            "race_id": "202505010101", "horse_name": "ホースA",
            "jockey_name": "武豊", "surface": "芝", "distance": 1600.0,
            "race_class_index": 2.0,
            "feat_popularity": 1, "target_win": 1, "target_top3": 1,
        },
        {
            "race_id": "202505010101", "horse_name": "ホースB",
            "jockey_name": "福永祐一", "surface": "芝", "distance": 1600.0,
            "race_class_index": 2.0,
            "feat_popularity": 2, "target_win": 0, "target_top3": 0,
        },
        {
            "race_id": "202505010102", "horse_name": "ホースC",
            "jockey_name": None, "surface": None, "distance": None,
            "race_class_index": None,
            "feat_popularity": 3, "target_win": 0, "target_top3": 1,
        },
    ])

def _make_jockey_stats_df():
    return pd.DataFrame([
        # overall
        {"jockey_name": "武豊",     "condition_type": "overall",     "condition_value": "全体",   "rides": 1000, "wins": 200, "top2": 300, "top3": 400},
        {"jockey_name": "福永祐一", "condition_type": "overall",     "condition_value": "全体",   "rides": 500,  "wins": 50,  "top2": 100, "top3": 150},
        # by_distance
        {"jockey_name": "武豊",     "condition_type": "by_distance", "condition_value": "マイル", "rides": 300,  "wins": 70,  "top2": 100, "top3": 130},
        {"jockey_name": "福永祐一", "condition_type": "by_distance", "condition_value": "マイル", "rides": 150,  "wins": 20,  "top2": 40,  "top3": 60},
        # by_surface
        {"jockey_name": "武豊",     "condition_type": "by_surface",  "condition_value": "芝",    "rides": 700,  "wins": 150, "top2": 220, "top3": 290},
        {"jockey_name": "福永祐一", "condition_type": "by_surface",  "condition_value": "芝",    "rides": 350,  "wins": 40,  "top2": 80,  "top3": 120},
        # by_track（会場）
        {"jockey_name": "武豊",     "condition_type": "by_track",    "condition_value": "東京",  "rides": 200,  "wins": 50,  "top2": 80,  "top3": 100},
        {"jockey_name": "福永祐一", "condition_type": "by_track",    "condition_value": "東京",  "rides": 100,  "wins": 10,  "top2": 20,  "top3": 30},
    ])


# ── _distance_to_bucket ────────────────────────────────────────────

def test_distance_bucket_short():
    assert fe._distance_to_bucket(1200.0) == 0  # 短距離

def test_distance_bucket_mile():
    assert fe._distance_to_bucket(1600.0) == 1  # マイル

def test_distance_bucket_medium():
    assert fe._distance_to_bucket(2000.0) == 2  # 中距離

def test_distance_bucket_long():
    assert fe._distance_to_bucket(2400.0) == 3  # 長距離

def test_distance_bucket_nan():
    assert fe._distance_to_bucket(float("nan")) == 0


# ── _surface_to_flag ──────────────────────────────────────────────

def test_surface_turf_flag():
    assert fe._surface_to_flag("芝") == 1

def test_surface_dirt_flag():
    assert fe._surface_to_flag("ダート") == 0

def test_surface_none_flag():
    assert fe._surface_to_flag(None) == 0


# ── build_jockey_lookup ───────────────────────────────────────────

def test_lookup_has_overall_win_rate():
    stats = _make_jockey_stats_df()
    lookup = fe.build_jockey_lookup(stats)
    assert "武豊" in lookup
    assert abs(lookup["武豊"]["overall_win_rate"] - 0.20) < 0.001

def test_lookup_has_distance_win_rate():
    stats = _make_jockey_stats_df()
    lookup = fe.build_jockey_lookup(stats)
    assert abs(lookup["武豊"]["by_distance"]["マイル"] - 70/300) < 0.001

def test_lookup_unknown_jockey_returns_empty():
    stats = _make_jockey_stats_df()
    lookup = fe.build_jockey_lookup(stats)
    assert "不明騎手" not in lookup


# ── expand_features ───────────────────────────────────────────────

def test_expand_adds_7_columns():
    df = _make_csv_df()
    stats = _make_jockey_stats_df()
    result = fe.expand_features(df, stats)
    for col in fe.EXPANDED_COLS:
        assert col in result.columns, f"Missing column: {col}"

def test_expand_known_jockey_has_nonzero_win_rate():
    df = _make_csv_df()
    stats = _make_jockey_stats_df()
    result = fe.expand_features(df, stats)
    takyu_row = result[result["horse_name"] == "ホースA"].iloc[0]
    assert takyu_row["feat_jockey_overall_win_rate"] > 0

def test_expand_unknown_jockey_falls_back_to_zero():
    df = _make_csv_df()
    stats = _make_jockey_stats_df()
    result = fe.expand_features(df, stats)
    no_jockey_row = result[result["horse_name"] == "ホースC"].iloc[0]
    assert no_jockey_row["feat_jockey_overall_win_rate"] == 0.0

def test_expand_distance_bucket_mile():
    df = _make_csv_df()
    stats = _make_jockey_stats_df()
    result = fe.expand_features(df, stats)
    row = result[result["horse_name"] == "ホースA"].iloc[0]
    assert row["feat_distance_bucket_enc"] == 1  # マイル

def test_expand_surface_turf():
    df = _make_csv_df()
    stats = _make_jockey_stats_df()
    result = fe.expand_features(df, stats)
    row = result[result["horse_name"] == "ホースA"].iloc[0]
    assert row["feat_surface_turf_flag"] == 1

# ── ML_FEATURE_COLUMNS_EXPANDED ───────────────────────────────────

def test_expanded_cols_superset_of_base():
    from race_ai_engine import ML_FEATURE_COLUMNS
    from feature_expander import EXPANDED_COLS
    from race_ai_engine import ML_FEATURE_COLUMNS_EXPANDED
    assert set(ML_FEATURE_COLUMNS).issubset(set(ML_FEATURE_COLUMNS_EXPANDED))
    for col in EXPANDED_COLS:
        assert col in ML_FEATURE_COLUMNS_EXPANDED, f"Missing: {col}"

def test_expanded_cols_no_duplicates():
    from race_ai_engine import ML_FEATURE_COLUMNS_EXPANDED
    assert len(ML_FEATURE_COLUMNS_EXPANDED) == len(set(ML_FEATURE_COLUMNS_EXPANDED))
