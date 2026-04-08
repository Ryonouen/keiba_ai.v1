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


# ── Task 3: train_lgbm_for_year ────────────────────────────────────────────

import math as _math
import pytest
from typing import List as _List
from backtest_full_pipeline import train_lgbm_for_year, ML_FEATURE_COLUMNS


def _make_df(years: _List[int]) -> pd.DataFrame:
    rows = []
    for y in years:
        for i in range(20):
            rows.append({
                "race_id": f"{y}01010101",
                "race_date": f"{y}-01-05",
                "horse_name": f"H{i}",
                "feat_gate": i % 8 + 1,
                "feat_age": 3,
                "feat_popularity": i + 1,
                "feat_win_odds_log": _math.log(i + 1.5),
                "feat_last3f": 35.0 + i * 0.1,
                "feat_jockey_weight": 55.0,
                "feat_n_runners": 20,
                "feat_running_style_enc": i % 3,
                "feat_track_condition_enc": 0,
                "feat_signal_total_adjust": 0.0,
                "feat_cond_diff_age": 0.0,
                "feat_cond_diff_gate": 0.0,
                "feat_cond_diff_style": 0.0,
                "feat_cond_diff_popularity": 0.0,
                "feat_cond_diff_last3f": 0.0,
                "feat_cond_diff_weight": 0.0,
                "feat_cond_diff_jockey": 0.0,
                "feat_cond_diff_track": 0.0,
                "target_win": 1 if i == 0 else 0,
                "target_top3": 1 if i < 3 else 0,
                "year": y,
            })
    return pd.DataFrame(rows)


def test_train_lgbm_for_year_returns_model():
    df = _make_df([2019, 2020, 2021])
    model = train_lgbm_for_year(df, test_year=2021)
    assert model is not None
    X = df[df["year"] < 2021][ML_FEATURE_COLUMNS].fillna(0)
    preds = model.predict(X.values)
    assert len(preds) == len(X)
    assert all(0 <= p <= 1 for p in preds)


def test_train_lgbm_raises_when_no_data():
    df = _make_df([2021])
    with pytest.raises(ValueError, match="訓練データが0件"):
        train_lgbm_for_year(df, test_year=2021)


# ── Task 4: run_value_ai_pipeline ─────────────────────────────────────────

from backtest_full_pipeline import run_value_ai_pipeline


def _make_features(n: int = 8):
    feats = []
    for i in range(n):
        odds = 2.0 + i * 3.0
        feats.append({
            "horse_name": f"H{i}",
            "win_prob": max(0.01, 0.4 - i * 0.04),
            "model_score": max(0.01, 0.4 - i * 0.04),
            "win_odds": odds,
            "place_odds": round(odds * 1.9 / 100, 2),
            "popularity_rank": i + 1,
            "running_style": ["front", "stalker", "closer"][i % 3],
            "last3f": 35.0,
            "gate": i + 1,
            "age": 4,
            "n_runners": n,
            "jockey_delta": 0.0,
            "jockey_reason_codes": [],
            "_target_win": 1 if i == 0 else 0,
            "_target_top3": 1 if i < 3 else 0,
            "_win_odds_log": _math.log(odds),
        })
    return feats


def test_run_value_ai_pipeline_returns_plan():
    features = _make_features(8)
    pace_balance = {"逃げ": 2, "先行": 3, "差し": 2, "追込": 1}
    plan = run_value_ai_pipeline(features, pace_balance, bankroll=10000)
    assert "skip" in plan
    assert "bet_type" in plan


def test_run_value_ai_pipeline_skip_when_empty():
    plan = run_value_ai_pipeline([], {}, bankroll=10000)
    assert plan["skip"] is True


# ── Task 5: simulate_payout ────────────────────────────────────────────────

from backtest_full_pipeline import simulate_payout


def test_simulate_payout_tansho_hit():
    features = _make_features(8)  # H0 が勝ち馬（_target_win=1）
    plan = {
        "bet_type": "単勝",
        "horses": ["H0"],
        "total_stake": 100,
        "skip": False,
    }
    result = simulate_payout(plan, features)
    expected_odds = 2.0  # H0 の win_odds
    assert result["hit"] is True
    assert abs(result["payout"] - expected_odds * 100) < 1.0


def test_simulate_payout_tansho_miss():
    features = _make_features(8)
    plan = {
        "bet_type": "単勝",
        "horses": ["H1"],
        "total_stake": 100,
        "skip": False,
    }
    result = simulate_payout(plan, features)
    assert result["hit"] is False
    assert result["payout"] == 0


def test_simulate_payout_skip():
    features = _make_features(8)
    plan = {"skip": True, "total_stake": 0}
    result = simulate_payout(plan, features)
    assert result is None


def test_simulate_payout_real_dividend_tansho():
    """dividends 辞書を渡すと real_dividend=True で払戻が計算される。"""
    features = _make_features(8)  # H0 が勝ち馬
    plan = {"bet_type": "単勝", "horses": ["H0"], "total_stake": 100, "skip": False}
    dividends = {"単勝": 1270}  # 12.7倍
    result = simulate_payout(plan, features, dividends=dividends)
    assert result["hit"] is True
    assert result["real_dividend"] is True
    assert abs(result["payout"] - 1270 / 100 * 100) < 0.01


def test_simulate_payout_real_dividend_miss_ignores_dividends():
    """外れ時は dividends があっても payout=0 で real_dividend=False。"""
    features = _make_features(8)
    plan = {"bet_type": "単勝", "horses": ["H1"], "total_stake": 100, "skip": False}
    dividends = {"単勝": 1270}
    result = simulate_payout(plan, features, dividends=dividends)
    assert result["hit"] is False
    assert result["payout"] == 0
    assert result["real_dividend"] is False


def test_simulate_payout_no_real_dividend_flag_without_dividends():
    """dividends=None のときは real_dividend=False（理論近似値）。"""
    features = _make_features(8)
    plan = {"bet_type": "単勝", "horses": ["H0"], "total_stake": 100, "skip": False}
    result = simulate_payout(plan, features, dividends=None)
    assert result["hit"] is True
    assert result["real_dividend"] is False
