import math
import pandas as pd
import pytest
from typing import List, Dict, Any
from backtest_full_pipeline import load_and_group_csv, build_feature_dict, build_pace_balance


def test_load_and_group_csv_returns_dict(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text(
        "race_id,race_date,horse_name,feat_win_odds_log,feat_popularity,"
        "feat_running_style_enc,feat_last3f,feat_gate,feat_age,feat_n_runners,"
        "feat_track_condition_enc,feat_signal_total_adjust,"
        "feat_cond_diff_age,feat_cond_diff_gate,feat_cond_diff_style,"
        "feat_cond_diff_popularity,feat_cond_diff_last3f,feat_cond_diff_weight,"
        "feat_cond_diff_jockey,feat_cond_diff_track,feat_recent_form,"
        "feat_trend_index,feat_consistency_index,target_win,target_top3\n"
        "202101010101,2021-01-05,HorseA,1.0,1,0,35.0,3,4,10,0,0.1,"
        "0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5,0.0,1.0,1,1\n"
        "202101010101,2021-01-05,HorseB,2.0,2,1,36.0,5,5,10,0,0.0,"
        "0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.3,0.0,0.8,0,0\n"
    )
    result = load_and_group_csv(str(csv), years=[2021])
    assert 2021 in result
    assert "202101010101" in result[2021]
    assert len(result[2021]["202101010101"]) == 2


def test_load_and_group_csv_filters_year(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text(
        "race_id,race_date,horse_name,feat_win_odds_log,feat_popularity,"
        "feat_running_style_enc,feat_last3f,feat_gate,feat_age,feat_n_runners,"
        "feat_track_condition_enc,feat_signal_total_adjust,"
        "feat_cond_diff_age,feat_cond_diff_gate,feat_cond_diff_style,"
        "feat_cond_diff_popularity,feat_cond_diff_last3f,feat_cond_diff_weight,"
        "feat_cond_diff_jockey,feat_cond_diff_track,feat_recent_form,"
        "feat_trend_index,feat_consistency_index,target_win,target_top3\n"
        "202101010101,2021-01-05,HorseA,1.0,1,0,35.0,3,4,10,0,0.1,"
        "0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5,0.0,1.0,1,1\n"
        "202201010101,2022-01-05,HorseB,2.0,2,1,36.0,5,5,10,0,0.0,"
        "0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.3,0.0,0.8,0,0\n"
    )
    result = load_and_group_csv(str(csv), years=[2021])
    assert 2021 in result
    assert 2022 not in result


def test_build_feature_dict_win_odds():
    row = {
        "horse_name": "HorseA",
        "feat_win_odds_log": math.log(5.0),
        "feat_popularity": 2,
        "feat_running_style_enc": 1,
        "feat_last3f": 35.5,
        "feat_gate": 3,
        "feat_age": 4,
        "feat_n_runners": 12,
        "feat_track_condition_enc": 0,
        "feat_signal_total_adjust": 0.1,
        "feat_cond_diff_age": 0.0,
        "feat_cond_diff_gate": 0.0,
        "feat_cond_diff_style": 0.0,
        "feat_cond_diff_popularity": 0.0,
        "feat_cond_diff_last3f": 0.0,
        "feat_cond_diff_weight": 0.0,
        "feat_cond_diff_jockey": 0.0,
        "feat_cond_diff_track": 0.0,
        "feat_recent_form": 0.4,
        "target_win": 0,
        "target_top3": 1,
    }
    f = build_feature_dict(row, win_prob=0.15)
    assert abs(f["win_odds"] - 5.0) < 0.01
    assert f["win_prob"] == 0.15
    assert f["horse_name"] == "HorseA"
    assert f["popularity_rank"] == 2
    assert f["running_style"] == "stalker"


def test_build_pace_balance():
    rows = [
        {"feat_running_style_enc": 0},  # front (逃げ)
        {"feat_running_style_enc": 0},  # front
        {"feat_running_style_enc": 1},  # stalker (先行)
        {"feat_running_style_enc": 2},  # closer (差し)
    ]
    pb = build_pace_balance(rows)
    assert pb["逃げ"] == 2
    assert pb["先行"] == 1
