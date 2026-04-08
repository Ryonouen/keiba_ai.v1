# tests/test_dashboard_loader.py
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


def test_kpi_calculation_miss_does_not_increment_hit_count():
    """外れ馬券は hit_count に加算されない。"""
    with patch.object(dl, "_load_json", side_effect=_mock_load(PRED_V2, BETS, OUTCOMES_MISS)):
        races = dl.load_races_for_date("20260405")
    kpi = dl.calc_kpi(races)
    assert kpi["hit_count"] == 0
    assert kpi["total_bets"] == 1
    assert kpi["total_payout"] == 0


def test_race_status_awaiting():
    """outcomes なし・start_datetime が 30分以上前 → status == 'awaiting'。"""
    past_pred = {
        "202606030401": {
            **PRED_V2["202606030401"],
            "start_datetime": (datetime.now() - timedelta(hours=1)).isoformat(),
        }
    }
    with patch.object(dl, "_load_json", side_effect=_mock_load(past_pred, {}, {})):
        races = dl.load_races_for_date("20260405")
    assert races[0]["status"] == "awaiting"


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


# ── Task 1: calc_upset_score ──────────────────────────────────

def test_upset_score_concentrated():
    """1頭が勝率90%を占める → エントロピー低 → 低スコア（堅い or やや堅い）"""
    horses = [
        {"horse_name": "A", "ai_win_prob": 0.9,  "win_odds": 1.2},
        {"horse_name": "B", "ai_win_prob": 0.05, "win_odds": 20.0},
        {"horse_name": "C", "ai_win_prob": 0.05, "win_odds": 20.0},
    ]
    result = dl.calc_upset_score(horses)
    assert result["score"] < 40
    assert result["label"] in ("堅い", "やや堅い")
    assert "color" in result


def test_upset_score_uniform():
    """10頭均等分布 → エントロピー最大 → 高スコア（やや荒れ or 荒れ）"""
    horses = [
        {"horse_name": str(i), "ai_win_prob": 0.1, "win_odds": 10.0}
        for i in range(10)
    ]
    result = dl.calc_upset_score(horses)
    assert result["score"] >= 55
    assert result["label"] in ("やや荒れ", "荒れ", "中間")


def test_upset_score_no_odds():
    """オッズが全馬 None → エントロピーのみで計算、戻り値のキーが揃っている"""
    horses = [
        {"horse_name": "A", "ai_win_prob": 0.5, "win_odds": None},
        {"horse_name": "B", "ai_win_prob": 0.3, "win_odds": None},
        {"horse_name": "C", "ai_win_prob": 0.2, "win_odds": None},
    ]
    result = dl.calc_upset_score(horses)
    assert 0 <= result["score"] <= 100
    assert result["label"] in ("堅い", "やや堅い", "中間", "やや荒れ", "荒れ")
    assert result["color"].startswith("#")


# ── Task 2: calc_hot_bets ─────────────────────────────────────

def test_hot_bets_threshold():
    """confidence 0.74 → 激熱なし / 0.75 → 激熱あり（境界値）"""
    bet_below = [{"bet_type": "tansho", "confidence": 0.74, "expected_value": None}]
    bet_at    = [{"bet_type": "tansho", "confidence": 0.75, "expected_value": None}]
    assert dl.calc_hot_bets(bet_below) == []
    assert len(dl.calc_hot_bets(bet_at)) == 1


def test_hot_bets_ev_filter():
    """confidence >= 0.75 でも expected_value <= 1.1 なら除外"""
    bets = [{"bet_type": "tansho", "confidence": 0.80, "expected_value": 1.05}]
    assert dl.calc_hot_bets(bets) == []


def test_hot_bets_ev_none_passes():
    """expected_value が None なら EV 条件をスキップして通過"""
    bets = [{"bet_type": "tansho", "confidence": 0.80, "expected_value": None}]
    assert len(dl.calc_hot_bets(bets)) == 1


# ── Task 3: load_races_for_date 出力拡張 ──────────────────────

def test_load_races_includes_upset_and_hot():
    """load_races_for_date の出力に upset_score / upset_label / upset_color / hot_bets が含まれる"""
    preds = {
        "202606030401": {
            "race_id": "202606030401",
            "race_name": "テスト | 2026年4月5日 中山1R レース情報",
            "analysis_date": "20260405",
            "horses": [
                {"horse_name": "A", "ai_win_prob": 0.4,
                 "feature_dict": {"win_odds": 3.0, "feat_popularity": 1, "running_style": "front"}},
                {"horse_name": "B", "ai_win_prob": 0.3,
                 "feature_dict": {"win_odds": 5.0, "feat_popularity": 2, "running_style": "stalker"}},
            ],
        }
    }
    bets = {
        "202606030401": [
            {"bet_type": "tansho", "confidence": 0.80, "expected_value": None,
             "bet_combination": ["A"], "stake_amount": 100}
        ]
    }
    with patch.object(dl, "_load_json", side_effect=_mock_load(preds, bets, {})):
        races = dl.load_races_for_date("20260405")

    assert len(races) == 1
    race = races[0]
    assert "upset_score" in race
    assert "upset_label" in race
    assert "upset_color" in race
    assert "hot_bets" in race
    assert isinstance(race["upset_score"], int)
    assert 0 <= race["upset_score"] <= 100
    assert race["upset_label"] in ("堅い", "やや堅い", "中間", "やや荒れ", "荒れ")
    assert race["upset_color"].startswith("#")


# ── Task 1: get_races_by_venue / get_weekend_date_strs ────────

from datetime import date as _date

def test_get_races_by_venue_groups_correctly():
    """レースが競馬場ごとにグループ化される。"""
    preds = {
        "R1": {
            "race_id": "R1",
            "race_name": "テスト | 2026年4月5日 中山1R レース情報",
            "analysis_date": "20260405",
            "horses": [],
        },
        "R2": {
            "race_id": "R2",
            "race_name": "テスト | 2026年4月5日 阪神1R レース情報",
            "analysis_date": "20260405",
            "horses": [],
        },
        "R3": {
            "race_id": "R3",
            "race_name": "テスト | 2026年4月5日 中山2R レース情報",
            "analysis_date": "20260405",
            "horses": [],
        },
    }
    with patch.object(dl, "_load_json", side_effect=_mock_load(preds, {}, {})):
        result = dl.get_races_by_venue("20260405")
    assert set(result.keys()) == {"中山", "阪神"}
    assert len(result["中山"]) == 2
    assert len(result["阪神"]) == 1


def test_get_races_by_venue_empty_for_no_data():
    """データなしの日は空 dict を返す。"""
    with patch.object(dl, "_load_json", return_value={}):
        result = dl.get_races_by_venue("20260101")
    assert result == {}


def test_get_weekend_date_strs_saturday():
    """土曜日: 今日=土, 土=今日, 日=翌日"""
    sat = _date(2026, 4, 11)  # Saturday
    today_s, sat_s, sun_s = dl.get_weekend_date_strs(sat)
    assert today_s == "20260411"
    assert sat_s == "20260411"
    assert sun_s == "20260412"


def test_get_weekend_date_strs_sunday():
    """日曜日: 今日=日, 土=前日, 日=今日"""
    sun = _date(2026, 4, 12)  # Sunday
    today_s, sat_s, sun_s = dl.get_weekend_date_strs(sun)
    assert today_s == "20260412"
    assert sat_s == "20260411"
    assert sun_s == "20260412"


def test_get_weekend_date_strs_weekday():
    """平日: 今日=水, 土=次の土曜, 日=次の日曜"""
    wed = _date(2026, 4, 8)  # Wednesday
    today_s, sat_s, sun_s = dl.get_weekend_date_strs(wed)
    assert today_s == "20260408"
    assert sat_s == "20260411"  # 2026-04-11 is Saturday
    assert sun_s == "20260412"  # 2026-04-12 is Sunday


# ── Task 2: monthly / yearly aggregations ─────────────────────

def test_get_available_months():
    """利用可能な月リストが YYYYMM 形式で降順に返る。"""
    preds = {
        "A": {"analysis_date": "20260403"},
        "B": {"analysis_date": "20260405"},
        "C": {"analysis_date": "20250312"},
    }
    with patch.object(dl, "_load_json", return_value=preds):
        months = dl.get_available_months()
    assert months == ["202604", "202503"]


def test_get_available_years():
    """利用可能な年リストが YYYY 形式で降順に返る。"""
    preds = {
        "A": {"analysis_date": "20260403"},
        "B": {"analysis_date": "20250405"},
    }
    with patch.object(dl, "_load_json", return_value=preds):
        years = dl.get_available_years()
    assert years == ["2026", "2025"]


def test_get_daily_kpi_for_month():
    """指定月の日別KPIリストが返る。的中データあり → roi 計算済み。"""
    preds_month = {
        "R1": {
            "race_id": "R1",
            "race_name": "テスト | 2026年4月5日 中山1R レース情報",
            "analysis_date": "20260405",
            "horses": [],
        },
    }
    outcomes_month = {
        "R1": [{"bet_type": "tansho", "stake": 100, "hit": True, "payout": 200}],
    }
    with patch.object(dl, "_load_json", side_effect=_mock_load(preds_month, {}, outcomes_month)):
        rows = dl.get_daily_kpi_for_month(2026, 4)
    assert len(rows) == 1
    assert rows[0]["date"] == "20260405"
    assert rows[0]["total_stake"] == 100
    assert rows[0]["total_payout"] == 200
    assert rows[0]["roi"] == 200.0
    assert rows[0]["hit_count"] == 1
    assert rows[0]["total_bets"] == 1


def test_get_daily_kpi_for_month_skips_zero_stake():
    """results なし（total_stake == 0）の日はリストに含まれない。"""
    preds_no_outcome = {
        "R1": {
            "race_id": "R1",
            "race_name": "テスト | 2026年4月5日 中山1R レース情報",
            "analysis_date": "20260405",
            "horses": [],
        },
    }
    # outcomes なし → total_stake == 0 → スキップされる
    with patch.object(dl, "_load_json", side_effect=_mock_load(preds_no_outcome, {}, {})):
        rows = dl.get_daily_kpi_for_month(2026, 4)
    assert rows == []


def test_get_monthly_kpi_for_year():
    """指定年の月別KPIリストが返る。"""
    preds_month = {
        "R1": {
            "race_id": "R1",
            "race_name": "テスト | 2026年4月5日 中山1R レース情報",
            "analysis_date": "20260405",
            "horses": [],
        },
    }
    outcomes_month = {
        "R1": [{"bet_type": "tansho", "stake": 100, "hit": True, "payout": 150}],
    }
    with patch.object(dl, "_load_json", side_effect=_mock_load(preds_month, {}, outcomes_month)):
        rows = dl.get_monthly_kpi_for_year(2026)
    assert len(rows) == 1
    assert rows[0]["month"] == "202604"
    assert rows[0]["total_stake"] == 100
    assert rows[0]["roi"] == 150.0
    assert rows[0]["hit_count"] == 1
    assert rows[0]["total_bets"] == 1
