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
