"""tests/test_daily_pipeline.py — パイプラインCRUD + 評価ロジックのユニットテスト"""
import sys, os, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pipeline_store

def _tmp_store(monkeypatch):
    """テスト用に一時ディレクトリへストアパスをリダイレクト"""
    d = tempfile.mkdtemp()
    monkeypatch.setattr(pipeline_store, "PREDICTIONS_FILE",    os.path.join(d, "pred.json"))
    monkeypatch.setattr(pipeline_store, "BET_SUGGESTIONS_FILE", os.path.join(d, "bets.json"))
    monkeypatch.setattr(pipeline_store, "RACE_RESULTS_FILE",   os.path.join(d, "results.json"))
    monkeypatch.setattr(pipeline_store, "BET_OUTCOMES_FILE",   os.path.join(d, "outcomes.json"))
    return d


def test_save_and_load_prediction(tmp_path, monkeypatch):
    _tmp_store(monkeypatch)
    pipeline_store.save_prediction(
        race_id="202501050811",
        race_meta={"race_title": "テストレース", "race_date": "2025-01-05"},
        features=[{"horse_name": "馬A", "win_prob": 0.3}],
    )
    pred = pipeline_store.load_prediction("202501050811")
    assert pred["race_name"] == "テストレース"
    assert pred["horses"][0]["horse_name"] == "馬A"
    assert pred["horses"][0]["ai_win_prob"] == 0.3


def test_save_and_load_bet_suggestions(tmp_path, monkeypatch):
    _tmp_store(monkeypatch)
    bets = [
        {"bet_type": "単勝", "bet_combination": ["馬A"], "stake_amount": 100,
         "selection_reason": "テスト", "confidence": 0.8,
         "expected_value": None, "implied_probability": None},
    ]
    pipeline_store.save_bet_suggestions("202501050811", bets)
    loaded = pipeline_store.load_bet_suggestions("202501050811")
    assert len(loaded) == 1
    assert loaded[0]["bet_type"] == "単勝"


def test_save_and_load_race_result(tmp_path, monkeypatch):
    _tmp_store(monkeypatch)
    pipeline_store.save_pipeline_race_result(
        "202501050811",
        {"finish_order": ["馬A", "馬B", "馬C"], "dividends": {"単勝": 250}},
    )
    r = pipeline_store.load_pipeline_race_result("202501050811")
    assert r["finish_order"][0] == "馬A"


def test_save_and_load_bet_outcomes(tmp_path, monkeypatch):
    _tmp_store(monkeypatch)
    outcomes = [
        {"bet_type": "単勝", "bet_combination": ["馬A"], "stake": 100,
         "hit": True, "payout": 250, "roi": 2.5},
    ]
    pipeline_store.save_bet_outcomes("202501050811", outcomes)
    loaded = pipeline_store.load_bet_outcomes("202501050811")
    assert loaded[0]["roi"] == 2.5


# ── Task 2 tests ──────────────────────────────────────────────

import daily_pipeline


def test_generate_all_bets_returns_all_types():
    """recommend_betmaster_plans の出力から全券種が変換されること。"""
    plans = [
        {"bet_type": "単勝",         "tickets": [{"combination": ["馬A"], "stake": 100}],
         "confidence_ok": True, "reason": "テスト", "confidence_score": 0.8,
         "no_pick_reason": ""},
        {"bet_type": "複勝",         "tickets": [{"combination": ["馬A"], "stake": 100}],
         "confidence_ok": True, "reason": "テスト", "confidence_score": 0.7,
         "no_pick_reason": ""},
        {"bet_type": "ワイド",       "tickets": [{"combination": ["馬A", "馬B"], "stake": 100}],
         "confidence_ok": True, "reason": "テスト", "confidence_score": 0.6,
         "no_pick_reason": ""},
        {"bet_type": "馬連（流し）",  "tickets": [{"combination": ["馬A", "馬B"], "stake": 100}],
         "confidence_ok": False, "reason": "テスト", "confidence_score": 0.4,
         "no_pick_reason": "自信不足"},
        {"bet_type": "馬単フォーメーション", "tickets": [], "confidence_ok": False,
         "reason": "", "confidence_score": 0.0, "no_pick_reason": "自信不足"},
        {"bet_type": "三連複フォーメーション（AI絞り）", "tickets": [],
         "confidence_ok": False, "reason": "", "confidence_score": 0.0, "no_pick_reason": ""},
        {"bet_type": "三連複フォーメーション（全頭）", "tickets": [],
         "confidence_ok": False, "reason": "", "confidence_score": 0.0, "no_pick_reason": ""},
        {"bet_type": "三連単フォーメーション（AI絞り）", "tickets": [],
         "confidence_ok": False, "reason": "", "confidence_score": 0.0, "no_pick_reason": ""},
        {"bet_type": "三連単フォーメーション（全頭）", "tickets": [],
         "confidence_ok": False, "reason": "", "confidence_score": 0.0, "no_pick_reason": ""},
    ]
    bets = daily_pipeline.generate_all_bets("202501050811", plans)
    # tickets が空でないプランはすべて含まれる（単勝・複勝・ワイド・馬連（流し）の4件）
    assert len(bets) == 4
    for bet in bets:
        assert "bet_type" in bet
        assert "bet_combination" in bet
        assert "stake_amount" in bet
        assert "confidence" in bet
        assert bet["stake_amount"] == 100


def test_generate_all_bets_skips_empty_tickets():
    """tickets=[] のプランは出力に含まれないこと。"""
    plans = [
        {"bet_type": "単勝", "tickets": [], "confidence_ok": False,
         "reason": "", "confidence_score": 0.0, "no_pick_reason": "自信不足"},
    ]
    bets = daily_pipeline.generate_all_bets("202501050811", plans)
    assert bets == []


# ── Task 3 tests ──────────────────────────────────────────────

def test_evaluate_bets_tansho_hit(tmp_path, monkeypatch):
    """単勝が的中した場合に hit=True, payout が正しく計算されること。"""
    _tmp_store(monkeypatch)

    bets = [
        {"bet_type": "tansho", "bet_type_label": "単勝",
         "bet_combination": ["馬A"], "stake_amount": 100,
         "selection_reason": "", "confidence": 0.8,
         "expected_value": None, "implied_probability": None},
    ]
    pipeline_store.save_bet_suggestions("202501050811", bets)

    race_result = {
        "finish_order": ["馬A", "馬B", "馬C"],
        "dividends": {"単勝": 320, "複勝": [130, 180, 220]},
        "runners": [],
    }
    pipeline_store.save_pipeline_race_result("202501050811", race_result)

    outcomes = daily_pipeline.evaluate_single_race("202501050811")
    assert len(outcomes) == 1
    assert outcomes[0]["hit"] is True
    assert outcomes[0]["payout"] == 320
    assert abs(outcomes[0]["roi"] - 3.2) < 0.01


def test_evaluate_bets_tansho_miss(tmp_path, monkeypatch):
    """単勝が外れた場合に hit=False, payout=0 であること。"""
    _tmp_store(monkeypatch)

    bets = [
        {"bet_type": "tansho", "bet_type_label": "単勝",
         "bet_combination": ["馬B"], "stake_amount": 100,
         "selection_reason": "", "confidence": 0.6,
         "expected_value": None, "implied_probability": None},
    ]
    pipeline_store.save_bet_suggestions("202501050811", bets)

    race_result = {
        "finish_order": ["馬A", "馬B", "馬C"],
        "dividends": {"単勝": 320},
        "runners": [],
    }
    pipeline_store.save_pipeline_race_result("202501050811", race_result)

    outcomes = daily_pipeline.evaluate_single_race("202501050811")
    assert outcomes[0]["hit"] is False
    assert outcomes[0]["payout"] == 0
    assert outcomes[0]["roi"] == 0.0


def test_evaluate_bets_wide_hit(tmp_path, monkeypatch):
    """ワイドが的中した場合に hit=True であること。"""
    _tmp_store(monkeypatch)

    bets = [
        {"bet_type": "wide", "bet_type_label": "ワイド",
         "bet_combination": ["馬A", "馬B"], "stake_amount": 100,
         "selection_reason": "", "confidence": 0.6,
         "expected_value": None, "implied_probability": None},
    ]
    pipeline_store.save_bet_suggestions("202501050811", bets)

    race_result = {
        "finish_order": ["馬A", "馬B", "馬C"],
        "dividends": {"ワイド": [210, 350, 480]},
        "runners": [],
    }
    pipeline_store.save_pipeline_race_result("202501050811", race_result)

    outcomes = daily_pipeline.evaluate_single_race("202501050811")
    assert outcomes[0]["hit"] is True
    assert outcomes[0]["payout"] == 210  # 最小払戻（保守的）
