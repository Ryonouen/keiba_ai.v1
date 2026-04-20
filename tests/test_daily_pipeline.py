"""tests/test_daily_pipeline.py — パイプラインCRUD + 評価ロジックのユニットテスト"""
import sys, os, json, tempfile
from datetime import datetime, timedelta
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


def test_mark_prediction_stale_writes_marker(tmp_path, monkeypatch):
    _tmp_store(monkeypatch)
    pipeline_store.save_prediction(
        race_id="202501050811",
        race_meta={"race_title": "テストレース", "race_date": "2025-01-05"},
        features=[{"horse_name": "馬A", "win_prob": 0.3}],
        analysis_date="20250105",
    )

    saved = pipeline_store.mark_prediction_stale(
        "202501050811",
        ["empty_horse_number_map", "empty_horse_number_map", "result_runner_count_mismatch"],
        source="daily_pipeline_audit",
        expected_count=14,
        saved_count=15,
    )

    pred = pipeline_store.load_prediction("202501050811")
    marker = pred["stale_prediction"]
    assert saved is True
    assert marker["is_stale"] is True
    assert marker["reasons"] == ["empty_horse_number_map", "result_runner_count_mismatch"]
    assert marker["source"] == "daily_pipeline_audit"
    assert marker["expected_count"] == 14
    assert marker["saved_count"] == 15
    assert marker["checked_at"]


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


def test_save_result_fetch_status_does_not_overwrite_finished_result(tmp_path, monkeypatch):
    _tmp_store(monkeypatch)
    pipeline_store.save_pipeline_race_result(
        "202501050811",
        {"finish_order": ["馬A", "馬B"], "runners": [{"horse_name": "馬A"}]},
    )

    pipeline_store.save_pipeline_result_fetch_status(
        "202501050811",
        "fetch_failed",
        reason="blocked",
    )

    r = pipeline_store.load_pipeline_race_result("202501050811")
    assert r["finish_order"] == ["馬A", "馬B"]
    assert "result_fetch_status" not in r


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
    assert abs(outcomes[0]["roi"] - 2.1) < 0.01


def test_evaluate_uses_saved_prediction_ids_when_race_list_is_empty(tmp_path, monkeypatch):
    _tmp_store(monkeypatch)
    race_id = "202606030811"
    past_start = (datetime.now() - timedelta(hours=2)).isoformat(timespec="seconds")
    pipeline_store._save(pipeline_store.PREDICTIONS_FILE, {
        race_id: {
            "race_id": race_id,
            "race_name": "皐月賞(G1)",
            "analysis_date": "20260419",
            "start_time": past_start[11:16],
            "start_datetime": past_start,
            "horses": [{"horse_name": "馬A", "ai_win_prob": 0.2}],
        }
    })
    pipeline_store.save_bet_suggestions(race_id, [
        {
            "bet_type": "tansho",
            "bet_type_label": "単勝",
            "bet_combination": ["馬A"],
            "stake_amount": 100,
        }
    ])

    monkeypatch.setattr(daily_pipeline, "get_race_ids_by_date", lambda _date: [])
    import dividend_scraper
    monkeypatch.setattr(dividend_scraper, "scrape_race_result", lambda _race_id: None)

    summary = daily_pipeline.evaluate_prediction_for_day("20260419")

    assert summary["total"] == 1
    assert summary["skipped"] == 1
    marker = pipeline_store.load_pipeline_race_result(race_id)
    assert marker["result_fetch_status"] == "fetch_failed"
    assert marker["result_fetch_attempt_count"] == 1


def test_audit_stale_predictions_marks_count_mismatches(tmp_path, monkeypatch):
    _tmp_store(monkeypatch)
    race_id = "202606030602"
    pipeline_store._save(pipeline_store.PREDICTIONS_FILE, {
        race_id: {
            "race_id": race_id,
            "race_name": "３歳未勝利",
            "analysis_date": "20260412",
            "start_time": "10:20",
            "horse_number_map": {},
            "horses": [
                {"horse_name": "馬A", "ai_win_prob": 0.3},
                {"horse_name": "馬B", "ai_win_prob": 0.2},
                {"horse_name": "取消馬", "ai_win_prob": 0.1},
            ],
        }
    })
    pipeline_store.save_pipeline_race_result(
        race_id,
        {
            "finish_order": ["馬A", "馬B"],
            "runners": [{"horse_name": "馬A"}, {"horse_name": "馬B"}],
        },
    )

    import dashboard_loader
    monkeypatch.setattr(
        dashboard_loader,
        "_fetch_schedule_map_for_date",
        lambda _date: {race_id: {"head_count": 2}},
    )

    summary = daily_pipeline.audit_stale_predictions_for_date("20260412")

    pred = pipeline_store.load_prediction(race_id)
    marker = pred["stale_prediction"]
    assert summary["checked"] == 1
    assert summary["stale"] == 1
    assert marker["is_stale"] is True
    assert marker["reasons"] == [
        "empty_horse_number_map",
        "latest_head_count_mismatch",
        "result_runner_count_mismatch",
    ]
    assert marker["expected_count"] == 2
    assert marker["saved_count"] == 3


def test_audit_stale_predictions_marks_horse_number_map_count_mismatch(tmp_path, monkeypatch):
    _tmp_store(monkeypatch)
    race_id = "202606030603"
    pipeline_store._save(pipeline_store.PREDICTIONS_FILE, {
        race_id: {
            "race_id": race_id,
            "race_name": "３歳未勝利",
            "analysis_date": "20260412",
            "start_time": "10:50",
            "horse_number_map": {"1": "馬A", "2": "馬B"},
            "horses": [
                {"horse_name": "馬A", "ai_win_prob": 0.3},
                {"horse_name": "馬B", "ai_win_prob": 0.2},
                {"horse_name": "馬C", "ai_win_prob": 0.1},
            ],
        }
    })

    import dashboard_loader
    monkeypatch.setattr(dashboard_loader, "_fetch_schedule_map_for_date", lambda _date: {})

    summary = daily_pipeline.audit_stale_predictions_for_date("20260412")

    marker = pipeline_store.load_prediction(race_id)["stale_prediction"]
    assert summary["stale"] == 1
    assert marker["reasons"] == ["horse_number_map_count_mismatch"]
    assert marker["expected_count"] == 2
    assert marker["saved_count"] == 3


def test_reanalyze_stale_predictions_only_targets_stale_and_clears_on_success(tmp_path, monkeypatch):
    _tmp_store(monkeypatch)
    stale_race_id = "202606030602"
    normal_race_id = "202606030603"
    other_date_race_id = "202606030604"
    pipeline_store._save(pipeline_store.PREDICTIONS_FILE, {
        stale_race_id: {
            "race_id": stale_race_id,
            "race_name": "stale",
            "analysis_date": "20260412",
            "stale_prediction": {"is_stale": True, "reasons": ["result_runner_count_mismatch"]},
            "horses": [{"horse_name": "古い馬"}],
        },
        normal_race_id: {
            "race_id": normal_race_id,
            "race_name": "normal",
            "analysis_date": "20260412",
            "horses": [{"horse_name": "正常馬"}],
        },
        other_date_race_id: {
            "race_id": other_date_race_id,
            "race_name": "other",
            "analysis_date": "20260413",
            "stale_prediction": {"is_stale": True, "reasons": ["empty_horse_number_map"]},
            "horses": [{"horse_name": "別日馬"}],
        },
    })
    calls = []

    def fake_reanalyze(date_str, race_id, per_race_timeout_sec=300, force_refresh_cache=False, use_requests=False):
        calls.append((date_str, race_id, per_race_timeout_sec, force_refresh_cache))
        pipeline_store.save_prediction(
            race_id=race_id,
            race_meta={"race_title": "reanalyzed", "race_date": "2026-04-12"},
            features=[{"horse_name": "新しい馬", "win_prob": 0.4}],
            analysis_date=date_str,
        )
        return {"date": date_str, "total": 1, "success": 1, "skipped": 0, "errors": []}

    monkeypatch.setattr(daily_pipeline, "run_daily_race_analysis_one_safe", fake_reanalyze)

    summary = daily_pipeline.reanalyze_stale_predictions_for_date("20260412", per_race_timeout_sec=123)

    assert summary["total"] == 1
    assert summary["success"] == 1
    assert calls == [("20260412", stale_race_id, 123, True)]
    assert "stale_prediction" not in pipeline_store.load_prediction(stale_race_id)
    assert pipeline_store.load_prediction(normal_race_id)["race_name"] == "normal"
    assert pipeline_store.load_prediction(other_date_race_id)["stale_prediction"]["is_stale"] is True


def test_reanalyze_stale_predictions_clears_dashboard_needs_reanalysis(tmp_path, monkeypatch):
    _tmp_store(monkeypatch)
    race_id = "202606030602"
    pipeline_store._save(pipeline_store.PREDICTIONS_FILE, {
        race_id: {
            "race_id": race_id,
            "race_name": "stale",
            "analysis_date": "20260412",
            "stale_prediction": {"is_stale": True, "reasons": ["result_runner_count_mismatch"]},
            "horses": [{"horse_name": "古い馬", "ai_win_prob": 0.2}],
        },
    })

    def fake_reanalyze(date_str, race_id, per_race_timeout_sec=300, force_refresh_cache=False, use_requests=False):
        pipeline_store.save_prediction(
            race_id=race_id,
            race_meta={"race_title": "reanalyzed", "race_date": "2026-04-12"},
            features=[{"horse_name": "新しい馬", "win_prob": 0.4}],
            analysis_date=date_str,
        )
        return {"date": date_str, "total": 1, "success": 1, "skipped": 0, "errors": []}

    monkeypatch.setattr(daily_pipeline, "run_daily_race_analysis_one_safe", fake_reanalyze)

    daily_pipeline.reanalyze_stale_predictions_for_date("20260412")

    import dashboard_loader as dl
    monkeypatch.setattr(dl, "_PREDICTIONS_FILE", pipeline_store.PREDICTIONS_FILE)
    monkeypatch.setattr(dl, "_BET_SUGGESTIONS_FILE", pipeline_store.BET_SUGGESTIONS_FILE)
    monkeypatch.setattr(dl, "_BET_OUTCOMES_FILE", pipeline_store.BET_OUTCOMES_FILE)
    monkeypatch.setattr(dl, "_RACE_RESULTS_FILE", pipeline_store.RACE_RESULTS_FILE)
    monkeypatch.setattr(dl, "_fetch_schedule_map_for_date", lambda _date: {})
    races = dl.load_races_for_date("20260412")
    assert races[0]["needs_reanalysis"] is False
    assert races[0]["stale_prediction"] == {}


def test_reanalyze_stale_predictions_keeps_marker_on_failure(tmp_path, monkeypatch):
    _tmp_store(monkeypatch)
    race_id = "202606030602"
    pipeline_store._save(pipeline_store.PREDICTIONS_FILE, {
        race_id: {
            "race_id": race_id,
            "race_name": "stale",
            "analysis_date": "20260412",
            "stale_prediction": {"is_stale": True, "reasons": ["result_runner_count_mismatch"]},
            "horses": [{"horse_name": "古い馬"}],
        },
    })

    def fake_reanalyze(date_str, race_id, per_race_timeout_sec=300, force_refresh_cache=False, use_requests=False):
        return {
            "date": date_str,
            "total": 1,
            "success": 0,
            "skipped": 1,
            "errors": [{"race_id": race_id, "error": "timeout"}],
        }

    monkeypatch.setattr(daily_pipeline, "run_daily_race_analysis_one_safe", fake_reanalyze)

    summary = daily_pipeline.reanalyze_stale_predictions_for_date("20260412")

    assert summary["total"] == 1
    assert summary["success"] == 0
    assert summary["skipped"] == 1
    assert summary["errors"] == [{"race_id": race_id, "error": "timeout"}]
    assert pipeline_store.load_prediction(race_id)["stale_prediction"]["is_stale"] is True


def test_reanalyze_stale_predictions_treats_success_return_with_marker_left_as_failure(tmp_path, monkeypatch):
    _tmp_store(monkeypatch)
    race_id = "202606030602"
    pipeline_store._save(pipeline_store.PREDICTIONS_FILE, {
        race_id: {
            "race_id": race_id,
            "race_name": "stale",
            "analysis_date": "20260412",
            "stale_prediction": {"is_stale": True, "reasons": ["result_runner_count_mismatch"]},
            "horses": [{"horse_name": "古い馬"}],
        },
    })

    def fake_reanalyze(date_str, race_id, per_race_timeout_sec=300, force_refresh_cache=False, use_requests=False):
        return {"date": date_str, "total": 1, "success": 1, "skipped": 0, "errors": []}

    monkeypatch.setattr(daily_pipeline, "run_daily_race_analysis_one_safe", fake_reanalyze)

    summary = daily_pipeline.reanalyze_stale_predictions_for_date("20260412")

    assert summary["total"] == 1
    assert summary["success"] == 0
    assert summary["skipped"] == 1
    assert summary["errors"] == [{"race_id": race_id, "error": "stale_marker_still_present"}]
    assert pipeline_store.load_prediction(race_id)["stale_prediction"]["is_stale"] is True


# ── Task 4 tests ──────────────────────────────────────────────

def test_summarize_empty_returns_zeros(tmp_path, monkeypatch):
    """データなしの場合にゼロ値で返ること。"""
    _tmp_store(monkeypatch)
    summary = daily_pipeline.summarize_weekend_performance(["20250405", "20250406"])
    assert summary["total_races"] == 0
    assert summary["total_stake"] == 0
    assert summary["total_payout"] == 0
    assert summary["by_bet_type"] == {}


def test_summarize_accumulates_outcomes(tmp_path, monkeypatch):
    """複数レースの的中・回収を正しく集計すること。"""
    _tmp_store(monkeypatch)

    # analysis_date="20250405" で予測を保存（集計フィルタのために必要）
    pipeline_store.save_prediction(
        "202501050811",
        {"race_title": "テスト1", "race_date": "2025-04-05"},
        [{"horse_name": "馬A", "win_prob": 0.3}],
        analysis_date="20250405",
    )
    pipeline_store.save_prediction(
        "202501050812",
        {"race_title": "テスト2", "race_date": "2025-04-05"},
        [{"horse_name": "馬B", "win_prob": 0.2}],
        analysis_date="20250405",
    )

    # Race 1: 単勝的中
    pipeline_store.save_bet_outcomes("202501050811", [
        {"bet_type": "tansho", "bet_type_label": "単勝",
         "bet_combination": ["馬A"], "stake": 100, "hit": True, "payout": 300, "roi": 3.0},
    ])
    # Race 2: 単勝外れ
    pipeline_store.save_bet_outcomes("202501050812", [
        {"bet_type": "tansho", "bet_type_label": "単勝",
         "bet_combination": ["馬B"], "stake": 100, "hit": False, "payout": 0, "roi": 0.0},
    ])

    summary = daily_pipeline.summarize_weekend_performance(["20250405"])
    assert summary["total_races"] == 2
    assert summary["total_stake"] == 200
    assert summary["total_payout"] == 300
    assert abs(summary["total_roi"] - 1.5) < 0.01
    bt = summary["by_bet_type"]["tansho"]
    assert bt["bets"] == 2
    assert bt["hits"] == 1
    assert abs(bt["hit_rate"] - 0.5) < 0.01
    assert abs(bt["roi"] - 1.5) < 0.01


# ── Task 6 (v2 integration) ──────────────────────────────

def test_run_daily_saves_v2_fields(tmp_path, monkeypatch):
    """run_daily_race_analysis が start_time / horse_number_map / feature_dict を保存する"""
    import sys
    _tmp_store(monkeypatch)

    fake_result = {
        "race_meta": {
            "race_title": "テストレース",
            "race_info_text": "10:00発走 / 芝1200m",
            "race_date": "2026-04-05",
        },
        "features": [
            {
                "horse_name": "テスト馬A", "win_prob": 0.25,
                "win_odds": 4.0, "place_odds": 1.8,
                "running_style": "front", "records_source": "newspaper",
                "horse_number": 1,
                "link": "https://db.netkeiba.com/horse/2022100001/",
                "feat_gate": 1, "feat_age": 4, "feat_popularity": 1,
                "feat_win_odds_log": 1.386, "feat_last3f": 33.0,
                "feat_jockey_weight": 55.0, "feat_n_runners": 8,
                "feat_running_style_enc": 0, "feat_track_condition_enc": 0,
                "feat_signal_total_adjust": 0.0,
                "feat_cond_diff_age": 0.0, "feat_cond_diff_gate": 0.0,
                "feat_cond_diff_style": 0.0, "feat_cond_diff_popularity": 0.0,
                "feat_cond_diff_last3f": 0.0, "feat_cond_diff_weight": 0.0,
                "feat_cond_diff_jockey": 0.0, "feat_cond_diff_track": 0.0,
                "feat_recent_form": 0.5, "feat_trend_index": 0.6,
                "feat_consistency_index": 0.7,
            }
        ],
        "race_structure": {"pace": "medium"},
        "ev_table": [],
        "danger_favorites_v2": [],
    }

    def fake_analyze(url, headless=True, **kwargs):
        return fake_result

    def fake_get_ids(date_str):
        return ["202609020401"]

    def fake_assign_roles(features, ev_table, race_structure=None, danger_horses=None):
        return []

    def fake_recommend(features, race_structure, horse_roles=None, race_pace="medium"):
        return []

    # Monkeypatch sys.modules to fake race_ai_engine and value_ai
    fake_engine = type(sys)("race_ai_engine_fake")
    fake_engine.analyze_race = fake_analyze
    monkeypatch.setitem(sys.modules, "race_ai_engine", fake_engine)

    fake_value = type(sys)("value_ai_fake")
    fake_value.recommend_betmaster_plans = fake_recommend
    fake_value.assign_roles = fake_assign_roles
    monkeypatch.setitem(sys.modules, "value_ai", fake_value)

    monkeypatch.setattr(daily_pipeline, "get_race_ids_by_date", fake_get_ids)

    daily_pipeline.run_daily_race_analysis("20260405")

    pred = pipeline_store.load_prediction("202609020401")
    assert pred is not None
    assert pred.get("start_time") == "10:00"
    assert pred.get("horse_number_map") == {"1": "テスト馬A"}
    assert pred["horses"][0]["feature_dict"]["feat_gate"] == 1
    assert pred["prediction_version"] == 1


def test_generate_all_bets_tansho_ev_filled():
    """単勝のbetに expected_value が計算されること"""
    from daily_pipeline import generate_all_bets
    plans = [{
        "bet_type": "単勝",
        "confidence_score": 0.8,
        "reason": "test",
        "tickets": [{"combination": ["ホワイトホース"], "stake": 100}],
        "_horse_win_prob": 0.25,
        "_horse_win_odds": 4.0,
    }]
    bets = generate_all_bets("race_001", plans)
    assert len(bets) == 1
    ev = bets[0].get("expected_value")
    # EV = 0.25 * 4.0 - 1 = 0.0
    assert ev is not None
    assert abs(ev - 0.0) < 0.01

def test_generate_all_bets_non_tansho_ev_none():
    """単勝以外のbetはexpected_valueがNoneのまま"""
    from daily_pipeline import generate_all_bets
    plans = [{
        "bet_type": "馬連（流し）",
        "confidence_score": 0.6,
        "reason": "test",
        "tickets": [{"combination": ["A", "B"], "stake": 100}],
    }]
    bets = generate_all_bets("race_001", plans)
    assert len(bets) == 1
    assert bets[0]["expected_value"] is None


def test_run_lgbm_prediction_falls_back_when_expanded_missing(monkeypatch, tmp_path):
    """拡張モデルがない場合は既存モデルにフォールバックすること"""
    from daily_pipeline import _run_lgbm_prediction
    from race_ai_engine import ML_FEATURE_COLUMNS
    import daily_pipeline as dp

    monkeypatch.setattr(dp, "EXPANDED_MODEL_FILE", str(tmp_path / "nonexistent.txt"))
    features = [{col: 1.0 for col in ML_FEATURE_COLUMNS} for _ in range(4)]
    monkeypatch.setattr("race_ai_engine.predict_win_probability_with_model", lambda *a: None)
    result = _run_lgbm_prediction(features)
    assert len(result) == 4
    assert abs(sum(result) - 1.0) < 1e-6
