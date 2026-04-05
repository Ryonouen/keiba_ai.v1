# tests/test_dashboard_loader.py
import json
import os
import sys
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dashboard_loader as dl

# ── フィクスチャ ────────────────────────────────────────────
PRED_V2 = {
    "202606030401": {
        "race_id": "202606030401",
        "race_name": "３歳未勝利 出馬表 | 2026年4月5日 中山1R レース情報(JRA) - netkeiba",
        "analysis_date": "20260405",
        "start_time": "10:00",
        "start_datetime": "2026-04-05T10:00:00",
        "horses": [
            {
                "horse_name": "ウマA",
                "ai_win_prob": 0.15,
                "feature_dict": {
                    "win_odds": 5.6,
                    "feat_popularity": 1,
                    "running_style": "front",
                },
            },
            {
                "horse_name": "ウマB",
                "ai_win_prob": 0.10,
                "feature_dict": None,
            },
        ],
    }
}

PRED_V1 = {
    "202606030402": {
        "race_id": "202606030402",
        "race_name": "３歳未勝利 出馬表 | 2026年4月5日 中山2R レース情報(JRA) - netkeiba",
        "analysis_date": "20260405",
        "horses": [
            {
                "horse_name": "ウマC",
                "ai_win_prob": 0.12,
                "win_odds": None,
                "popularity": None,
            }
        ],
    }
}

BETS = {
    "202606030401": [
        {
            "bet_type": "tansho",
            "bet_type_label": "単勝",
            "bet_combination": ["ウマA"],
            "stake_amount": 100,
        }
    ]
}

OUTCOMES_HIT = {
    "202606030401": [
        {
            "bet_type": "tansho",
            "bet_type_label": "単勝",
            "bet_combination": ["ウマA"],
            "stake": 100,
            "hit": True,
            "payout": 560,
            "roi": 5.6,
        }
    ]
}

OUTCOMES_MISS = {
    "202606030401": [
        {
            "bet_type": "tansho",
            "bet_type_label": "単勝",
            "bet_combination": ["ウマA"],
            "stake": 100,
            "hit": False,
            "payout": 0,
            "roi": 0.0,
        }
    ]
}


def _mock_load(preds, bets, outcomes):
    """_load_json を差し替えるヘルパー。"""
    def side_effect(path):
        if "predictions" in path:
            return preds
        if "suggestions" in path:
            return bets
        if "outcomes" in path:
            return outcomes
        return {}
    return side_effect


# ── テスト ────────────────────────────────────────────────────
def test_load_today_races_filters_by_date():
    """analysis_date が一致するレースだけ返す。"""
    preds = {**PRED_V2, **PRED_V1}
    with patch.object(dl, "_load_json", side_effect=_mock_load(preds, {}, {})):
        races = dl.load_races_for_date("20260405")
    assert len(races) == 2
    ids = {r["race_id"] for r in races}
    assert "202606030401" in ids
    assert "202606030402" in ids


def test_kpi_calculation():
    """的中率・ROI・投資額・回収額が正確に計算される。"""
    with patch.object(dl, "_load_json", side_effect=_mock_load(PRED_V2, BETS, OUTCOMES_HIT)):
        races = dl.load_races_for_date("20260405")
    kpi = dl.calc_kpi(races)
    assert kpi["total_stake"] == 100
    assert kpi["total_payout"] == 560
    assert kpi["roi"] == 560.0
    assert kpi["hit_count"] == 1
    assert kpi["total_bets"] == 1


def test_bet_type_breakdown():
    """券種別集計が正確に計算される。"""
    with patch.object(dl, "_load_json", side_effect=_mock_load(PRED_V2, BETS, OUTCOMES_HIT)):
        races = dl.load_races_for_date("20260405")
    rows = dl.calc_kpi_by_bet_type(races)
    tansho = next(r for r in rows if r["bet_type"] == "tansho")
    assert tansho["count"] == 1
    assert tansho["hit"] == 1
    assert tansho["hit_rate"] == 100.0
    assert tansho["roi"] == 560.0


def test_horse_odds_fallback_v1():
    """v1スキーマ（feature_dict なし）で win_odds と running_style が None になる。"""
    with patch.object(dl, "_load_json", side_effect=_mock_load(PRED_V1, {}, {})):
        races = dl.load_races_for_date("20260405")
    horse = races[0]["horses"][0]
    assert horse["win_odds"] is None
    assert horse["running_style"] is None


def test_horse_running_style_mapped_v2():
    """v2スキーマで running_style が日本語にマッピングされる。"""
    with patch.object(dl, "_load_json", side_effect=_mock_load(PRED_V2, {}, {})):
        races = dl.load_races_for_date("20260405")
    horse_a = next(h for h in races[0]["horses"] if h["horse_name"] == "ウマA")
    assert horse_a["running_style"] == "逃げ"


def test_race_status_prerace():
    """outcomes なし・start_datetime が未来 → status == 'prerace'。"""
    future_pred = {
        "202606030401": {
            **PRED_V2["202606030401"],
            "start_datetime": (datetime.now() + timedelta(hours=2)).isoformat(),
        }
    }
    with patch.object(dl, "_load_json", side_effect=_mock_load(future_pred, {}, {})):
        races = dl.load_races_for_date("20260405")
    assert races[0]["status"] == "prerace"


def test_race_status_result():
    """outcomes あり → status == 'result'。"""
    with patch.object(dl, "_load_json", side_effect=_mock_load(PRED_V2, BETS, OUTCOMES_HIT)):
        races = dl.load_races_for_date("20260405")
    assert races[0]["status"] == "result"


def test_date_list_descending():
    """get_available_dates() が降順で返る。"""
    preds = {
        "A": {"analysis_date": "20260403"},
        "B": {"analysis_date": "20260405"},
        "C": {"analysis_date": "20260404"},
    }
    with patch.object(dl, "_load_json", return_value=preds):
        dates = dl.get_available_dates()
    assert dates == ["20260405", "20260404", "20260403"]
