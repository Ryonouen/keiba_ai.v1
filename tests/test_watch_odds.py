# tests/test_watch_odds.py
import sys, os, tempfile, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import pipeline_store
import daily_pipeline


def _tmp_store(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_store, "PREDICTIONS_FILE",
                        str(tmp_path / "pred.json"))
    monkeypatch.setattr(pipeline_store, "BET_SUGGESTIONS_FILE",
                        str(tmp_path / "bets.json"))


def _seed_prediction(race_id: str):
    """テスト用の v2 prediction を store に保存する"""
    features = [
        {
            "horse_name": "ショウヘイ", "win_prob": 0.069, "win_odds": None,
            "place_odds": None, "running_style": "front", "records_source": "newspaper",
            "horse_number": 1,
            "link": "https://db.netkeiba.com/horse/2022105123/",
            "feat_gate": 3, "feat_age": 5, "feat_popularity": 0,
            "feat_win_odds_log": 0.0, "feat_last3f": 34.5,
            "feat_jockey_weight": 57.0, "feat_n_runners": 15,
            "feat_running_style_enc": 0, "feat_track_condition_enc": 0,
            "feat_signal_total_adjust": 0.12,
            "feat_cond_diff_age": 0.0, "feat_cond_diff_gate": 0.0,
            "feat_cond_diff_style": 0.0, "feat_cond_diff_popularity": 0.0,
            "feat_cond_diff_last3f": 0.0, "feat_cond_diff_weight": 0.0,
            "feat_cond_diff_jockey": 0.0, "feat_cond_diff_track": 0.0,
            "feat_recent_form": 0.5, "feat_trend_index": 0.6,
            "feat_consistency_index": 0.7,
        }
    ]
    pipeline_store.save_prediction_v2(
        race_id=race_id,
        race_meta={
            "race_title": "大阪杯", "race_info_text": "15:45発走",
            "race_date": "2026-04-05",
        },
        features=features, ev_table=[], race_structure={}, danger_v2=[],
        analysis_date="20260405",
    )


def test_not_open_no_version_bump(tmp_path, monkeypatch):
    """`not_open` ステータスでは prediction_version が増えない"""
    _tmp_store(monkeypatch, tmp_path)
    _seed_prediction("202609020411")

    import odds_fetcher
    monkeypatch.setattr(odds_fetcher, "fetch_win_odds",
                        lambda race_id, horse_number_map: ("not_open", None))

    result = daily_pipeline.update_race_odds("202609020411")
    pred = pipeline_store.load_prediction("202609020411")
    assert pred["prediction_version"] == 1
    assert result["status"] == "not_open"


def test_version_increments_on_success(tmp_path, monkeypatch):
    """成功時に prediction_version が 2 になり、odds_after.status が success"""
    _tmp_store(monkeypatch, tmp_path)
    _seed_prediction("202609020411")

    import odds_fetcher
    monkeypatch.setattr(odds_fetcher, "fetch_win_odds",
                        lambda race_id, horse_number_map: ("success", {"ショウヘイ": 5.6}))

    # LightGBM 予測をモック（モデルファイルなし環境でもテストが通るよう）
    monkeypatch.setattr(daily_pipeline, "_run_lgbm_prediction",
                        lambda features: [0.142])

    result = daily_pipeline.update_race_odds("202609020411")
    pred = pipeline_store.load_prediction("202609020411")
    assert pred["prediction_version"] == 2
    assert result["status"] == "success"
    assert pred["horses"][0]["win_odds"] == 5.6
    assert pred["horses"][0]["feature_dict"]["feat_win_odds_log"] == pytest.approx(
        __import__("math").log(5.6), abs=1e-3
    )


from datetime import datetime, timedelta


def test_window_detection():
    """T-30min 付近のレースのみが更新対象として選ばれる"""
    now = datetime(2026, 4, 5, 15, 15, 0)   # 15:15
    start_times = {
        "202609020401": "2026-04-05T10:00:00",   # 発走済み（対象外）
        "202609020410": "2026-04-05T15:40:00",   # T-25min（対象）
        "202609020411": "2026-04-05T15:45:00",   # T-30min（対象）
        "202609020412": "2026-04-05T16:25:00",   # T-70min（対象外）
    }
    # 発走時刻 - 35min ≤ now ≤ 発走時刻 - 20min
    targets = daily_pipeline._get_update_targets(start_times, now, updated_ids=set())
    assert "202609020410" in targets
    assert "202609020411" in targets
    assert "202609020401" not in targets   # 発走済み
    assert "202609020412" not in targets   # まだ早すぎる


def test_already_updated_skip():
    """`updated_ids` に入っているレースはスキップされる"""
    now = datetime(2026, 4, 5, 15, 15, 0)
    start_times = {
        "202609020411": "2026-04-05T15:45:00",
    }
    targets = daily_pipeline._get_update_targets(
        start_times, now, updated_ids={"202609020411"}
    )
    assert "202609020411" not in targets


def test_exit_condition_all_past():
    """全レースが start + 90min を過ぎていれば _should_exit が True"""
    now = datetime(2026, 4, 5, 20, 0, 0)
    start_times = {
        "202609020401": "2026-04-05T10:00:00",
        "202609020412": "2026-04-05T17:00:00",
    }
    assert daily_pipeline._should_exit(start_times, updated_ids=set(), now=now) is True


def test_no_exit_when_races_remain():
    """まだ発走前のレースがあれば _should_exit が False"""
    now = datetime(2026, 4, 5, 15, 0, 0)
    start_times = {
        "202609020411": "2026-04-05T15:45:00",
    }
    assert daily_pipeline._should_exit(start_times, updated_ids=set(), now=now) is False
