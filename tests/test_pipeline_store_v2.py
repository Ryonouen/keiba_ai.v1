# tests/test_pipeline_store_v2.py
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import pipeline_store


def _tmp(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setattr(pipeline_store, "PREDICTIONS_FILE",
                        os.path.join(d, "pred.json"))
    return d


def _make_features():
    return [
        {
            "horse_name": "ショウヘイ", "win_prob": 0.142, "win_odds": 5.6,
            "place_odds": 2.1, "running_style": "front",
            "feat_gate": 3, "feat_age": 5, "feat_popularity": 1,
            "feat_win_odds_log": 1.7228, "feat_last3f": 34.5,
            "feat_jockey_weight": 57.0, "feat_n_runners": 15,
            "feat_running_style_enc": 0, "feat_track_condition_enc": 0,
            "feat_signal_total_adjust": 0.12,
            "feat_cond_diff_age": 0.0, "feat_cond_diff_gate": 0.0,
            "feat_cond_diff_style": 0.0, "feat_cond_diff_popularity": 0.0,
            "feat_cond_diff_last3f": 0.0, "feat_cond_diff_weight": 0.0,
            "feat_cond_diff_jockey": 0.0, "feat_cond_diff_track": 0.0,
            "feat_recent_form": 0.5, "feat_trend_index": 0.6,
            "feat_consistency_index": 0.7,
            "horse_number": 1,
            "link": "https://db.netkeiba.com/horse/2022105123/",
            "records_source": "newspaper",
        }
    ]


def test_start_time_parsing(monkeypatch):
    """`race_info_text` から start_time と start_datetime が正しく抽出される"""
    _tmp(monkeypatch)
    pipeline_store.save_prediction_v2(
        race_id="202609020411",
        race_meta={
            "race_title": "大阪杯",
            "race_info_text": "15:45発走 / 芝2000m / 良",
            "race_date": "2026-04-05",
        },
        features=_make_features(),
        ev_table=[], race_structure={}, danger_v2=[],
        analysis_date="20260405",
    )
    pred = pipeline_store.load_prediction("202609020411")
    assert pred["start_time"] == "15:45"
    assert pred["start_datetime"] == "2026-04-05T15:45:00"


def test_horse_id_parse_from_link(monkeypatch):
    """horse_link から horse_id が正しく抽出される"""
    _tmp(monkeypatch)
    pipeline_store.save_prediction_v2(
        race_id="202609020411",
        race_meta={"race_title": "大阪杯", "race_info_text": "15:45発走", "race_date": ""},
        features=_make_features(),
        ev_table=[], race_structure={}, danger_v2=[],
    )
    pred = pipeline_store.load_prediction("202609020411")
    assert pred["horse_id_map"]["ショウヘイ"] == "2022105123"


def test_feature_dict_roundtrip(monkeypatch):
    """feature_dict が欠損なく保存・ロードされる"""
    _tmp(monkeypatch)
    pipeline_store.save_prediction_v2(
        race_id="202609020411",
        race_meta={"race_title": "大阪杯", "race_info_text": "15:45発走", "race_date": ""},
        features=_make_features(),
        ev_table=[], race_structure={}, danger_v2=[],
    )
    pred = pipeline_store.load_prediction("202609020411")
    horse = pred["horses"][0]
    assert horse["feature_dict"]["feat_win_odds_log"] == pytest.approx(1.7228)
    assert horse["feature_dict"]["feat_gate"] == 3


def test_prediction_history_initial(monkeypatch):
    """初回保存で prediction_version=1、prediction_history に 1 エントリ"""
    _tmp(monkeypatch)
    pipeline_store.save_prediction_v2(
        race_id="202609020411",
        race_meta={"race_title": "大阪杯", "race_info_text": "15:45発走", "race_date": ""},
        features=_make_features(),
        ev_table=[], race_structure={}, danger_v2=[],
    )
    pred = pipeline_store.load_prediction("202609020411")
    assert pred["prediction_version"] == 1
    assert len(pred["prediction_history"]) == 1
    assert pred["prediction_history"][0]["source"] == "initial_analysis"


def test_odds_status_roundtrip(monkeypatch):
    """odds_before の status フィールドが正確に保存・ロードされる"""
    _tmp(monkeypatch)
    pipeline_store.save_prediction_v2(
        race_id="202609020411",
        race_meta={"race_title": "大阪杯", "race_info_text": "15:45発走", "race_date": ""},
        features=_make_features(),
        ev_table=[], race_structure={}, danger_v2=[],
    )
    pred = pipeline_store.load_prediction("202609020411")
    assert pred["odds_before"]["status"] == "not_open"
    assert pred["odds_after"]["status"] == "not_open"
