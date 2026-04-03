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
