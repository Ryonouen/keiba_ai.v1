"""
pipeline_store.py
パイプライン用 4テーブル JSON ストア

テーブル:
  pipeline_predictions.json       race_id → 予測データ（馬リスト）
  pipeline_bet_suggestions.json   race_id → 全券種買い目候補リスト
  pipeline_race_results.json      race_id → レース結果キャッシュ
  pipeline_bet_outcomes.json      race_id → 的中・払戻結果リスト
"""
from __future__ import annotations

import json
import os
import re as _re
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))

PREDICTIONS_FILE     = os.path.join(_HERE, "pipeline_predictions.json")
BET_SUGGESTIONS_FILE = os.path.join(_HERE, "pipeline_bet_suggestions.json")
RACE_RESULTS_FILE    = os.path.join(_HERE, "pipeline_race_results.json")
BET_OUTCOMES_FILE    = os.path.join(_HERE, "pipeline_bet_outcomes.json")


# =========================================================
# 共通ユーティリティ
# =========================================================

def _load(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save(path: str, data: Dict[str, Any]) -> None:
    dir_ = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


# =========================================================
# predictions
# =========================================================

def save_prediction(
    race_id: str,
    race_meta: Dict[str, Any],
    features: List[Dict[str, Any]],
    analysis_date: str = "",
) -> None:
    """
    analyze_race() の結果から予測データを保存する。

    Parameters
    ----------
    analysis_date : run_daily_race_analysis() に渡した日付文字列（YYYYMMDD）。
                    summarize_weekend_performance() のフィルタに使用する。
    """
    data = _load(PREDICTIONS_FILE)
    horses = [
        {
            "horse_name":      str(f.get("horse_name") or ""),
            "ai_win_prob":     round(float(f.get("win_prob") or 0.0), 4),
            "win_odds":        f.get("win_odds"),
            "popularity":      f.get("popularity"),
        }
        for f in features
    ]
    data[race_id] = {
        "race_id":       race_id,
        "race_name":     race_meta.get("race_title", ""),
        "race_date":     race_meta.get("race_date", ""),
        "analysis_date": analysis_date,   # YYYYMMDD — 集計フィルタ用
        "analyzed_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "horses":        horses,
    }
    _save(PREDICTIONS_FILE, data)


def load_prediction(race_id: str) -> Optional[Dict[str, Any]]:
    return _load(PREDICTIONS_FILE).get(race_id)


def load_all_predictions() -> Dict[str, Any]:
    return _load(PREDICTIONS_FILE)


def is_prediction_incomplete(pred: Dict[str, Any]) -> bool:
    """
    明らかに不完全な prediction レコードを検出する。

    現状の主な壊れ方は「全頭取得できず 1〜3頭だけ保存される」ケースなので、
    JRA 前提で 3頭以下は不完全とみなす。
    併せて horse_number_map が存在する場合は件数不一致も不完全扱いにする。
    """
    if not isinstance(pred, dict):
        return True

    horses = pred.get("horses") or []
    if not isinstance(horses, list) or not horses:
        return True

    if len(horses) <= 3:
        return True

    if not str(pred.get("start_time") or "").strip():
        return True

    horse_number_map = pred.get("horse_number_map") or {}
    if not isinstance(horse_number_map, dict) or not horse_number_map:
        return True
    if len(horses) < len(horse_number_map):
        return True

    all_no_odds = all(h.get("win_odds") in (None, "", 0) for h in horses)
    all_no_popularity = all(h.get("popularity") in (None, "", 0) for h in horses)
    if all_no_odds and all_no_popularity:
        return True

    return False


def load_race_start_times(date_str: str) -> Dict[str, str]:
    """
    指定 analysis_date の全レースの {race_id: start_datetime} を返す。
    start_datetime が空のエントリは除外する。
    """
    data = _load(PREDICTIONS_FILE)
    result: Dict[str, str] = {}
    for race_id, pred in data.items():
        if pred.get("analysis_date") != date_str:
            continue
        sdt = pred.get("start_datetime", "")
        if sdt:
            result[race_id] = sdt
    return result


def update_prediction_odds_in_store(
    race_id: str,
    new_odds_by_name: Dict[str, float],     # {horse_name: win_odds}
    updated_horses: List[Dict[str, Any]],   # 更新済み horses リスト
    odds_status: str,
    odds_source: str,
    coverage_ratio: float,
) -> None:
    """
    オッズ更新後の予測データを store に書き戻す。
    prediction_version を +1 し、prediction_history・odds_after・odds_update_history を更新する。
    odds_status が "not_open" / "failed" 系の場合は呼ばない（呼び出し元が制御）。
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = _load(PREDICTIONS_FILE)
    pred = data.get(race_id)
    if pred is None:
        return

    old_version = pred.get("prediction_version", 1)
    new_version = old_version + 1

    # odds_after を horse_number ベースで構築
    horse_number_map = pred.get("horse_number_map", {})
    name_to_no = {v: k for k, v in horse_number_map.items()}  # 逆引き
    tansho_after = {
        name_to_no.get(name, name): odds
        for name, odds in new_odds_by_name.items()
    }
    # 未取得馬は null
    for no in horse_number_map:
        tansho_after.setdefault(no, None)

    pred["prediction_version"] = new_version
    pred["horses"] = updated_horses
    pred["prediction_history"].append({
        "version":    new_version,
        "created_at": now_str,
        "source":     f"odds_update_{odds_source}",
        "horses":     [{"horse_name": h["horse_name"], "ai_win_prob": h["ai_win_prob"]}
                       for h in updated_horses],
    })
    pred["odds_after"] = {
        "status":         odds_status,
        "source":         odds_source,
        "coverage_ratio": round(coverage_ratio, 4),
        "tansho":         tansho_after,
        "fukusho":        {no: None for no in horse_number_map},  # 今回は単勝のみ
    }
    pred["odds_update_history"].append({
        "at":                       now_str,
        "source":                   odds_source,
        "coverage_ratio":           round(coverage_ratio, 4),
        "prediction_version_after": new_version,
    })
    data[race_id] = pred
    _save(PREDICTIONS_FILE, data)


def _extract_start_time(race_info_text: str) -> str:
    """
    "15:45発走 / 芝..." または "15時45分発走 ..." から "HH:MM" を抽出。
    取得できなければ "" を返す。
    """
    m = _re.search(r'(\d{1,2})[時:](\d{2})分?発走', race_info_text or "")
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    return ""


def _parse_horse_id(link: str) -> str:
    """
    "https://db.netkeiba.com/horse/2022105123/" → "2022105123"
    取得できなければ "" を返す。
    """
    m = _re.search(r'/horse/(\d+)', link or "")
    return m.group(1) if m else ""


def _build_odds_shell(horse_number_map: Dict[str, Any]) -> Dict[str, Any]:
    """
    初期状態の odds_before / odds_after シェル（全馬 null、status=not_open）を返す。
    """
    nulls = {no: None for no in horse_number_map}
    return {
        "status": "not_open",
        "tansho": dict(nulls),
        "fukusho": dict(nulls),
    }


def save_prediction_v2(
    race_id: str,
    race_meta: Dict[str, Any],
    features: List[Dict[str, Any]],
    ev_table: List[Dict[str, Any]],
    race_structure: Dict[str, Any],
    danger_v2: List[Dict[str, Any]],
    analysis_date: str = "",
) -> None:
    """
    analyze_race() の結果を v2 スキーマで pipeline_predictions.json に保存する。

    既存 save_prediction() と共存。load_prediction() はどちらのフォーマットも読める。
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    race_info_text = race_meta.get("race_info_text", "")
    race_date = race_meta.get("race_date", "")
    # analysis_date を YYYYMMDD 形式に正規化（"2026-04-19" → "20260419"）
    analysis_date = analysis_date.replace("-", "") if analysis_date else ""

    start_time = _extract_start_time(race_info_text)
    start_datetime = f"{race_date}T{start_time}:00" if start_time and race_date else ""

    def _feature_key(feature: Dict[str, Any]) -> str:
        name = str(feature.get("horse_name") or "")
        if name:
            return f"name:{name}"
        no = feature.get("horse_number")
        return f"no:{no}"

    def _feature_score(feature: Dict[str, Any]) -> int:
        score = 0
        for key in [
            "horse_name", "horse_url", "horse_number", "gate", "jockey",
            "popularity", "win_odds", "ability_score", "raw_ability_score",
        ]:
            value = feature.get(key)
            if value not in (None, "", 0):
                score += 1
        records = feature.get("past_races") or feature.get("records") or []
        if isinstance(records, list):
            score += len(records)
        return score

    deduped_features: List[Dict[str, Any]] = []
    deduped_by_key: Dict[str, Dict[str, Any]] = {}
    ordered_keys: List[str] = []
    for feature in features:
        key = _feature_key(feature)
        prev = deduped_by_key.get(key)
        if prev is None:
            deduped_by_key[key] = feature
            ordered_keys.append(key)
            continue
        if _feature_score(feature) > _feature_score(prev):
            deduped_by_key[key] = feature
    deduped_features = [deduped_by_key[key] for key in ordered_keys]
    features = deduped_features

    # horse_number_map と horse_id_map を features から構築
    horse_number_map: Dict[str, str] = {}
    horse_id_map: Dict[str, str] = {}
    for f in features:
        no = f.get("horse_number")
        name = str(f.get("horse_name") or "")
        link = str(f.get("link") or "")
        if no is not None and name:
            horse_number_map[str(no)] = name
        if name and link:
            hid = _parse_horse_id(link)
            if hid:
                horse_id_map[name] = hid

    # 馬ごとのデータ
    horses = []
    for f in features:
        name = str(f.get("horse_name") or "")
        running_style = str(f.get("running_style") or "unknown")
        records_source = str(f.get("records_source") or "none")
        # running_style_source の決定
        if records_source == "newspaper":
            rs_source = "newspaper"
        elif running_style == "unknown":
            rs_source = "unknown"
        else:
            rs_source = "inferred"

        horses.append({
            "horse_name":             name,
            "horse_id":               horse_id_map.get(name, ""),
            "ai_win_prob":            round(float(f.get("win_prob") or 0.0), 4),
            "win_odds":               f.get("win_odds"),
            "popularity":             f.get("popularity"),
            "ability_score":          f.get("ability_score"),
            "raw_ability_score":      f.get("raw_ability_score"),
            "ai_power_index":         f.get("ai_power_index"),
            "running_style":          running_style,
            "running_style_source":   rs_source,
            "running_style_missing":  running_style == "unknown",
            "feature_dict":           dict(f),   # 全フィールドを保存
        })

    initial_history_entry = {
        "version":    1,
        "created_at": now_str,
        "source":     "initial_analysis",
        "horses":     [{"horse_name": h["horse_name"], "ai_win_prob": h["ai_win_prob"]}
                       for h in horses],
    }
    odds_shell = _build_odds_shell(horse_number_map)

    data = _load(PREDICTIONS_FILE)
    data[race_id] = {
        "race_id":        race_id,
        "race_name":      race_meta.get("race_title", ""),
        "race_date":      race_date,
        "analysis_date":  analysis_date,
        "analyzed_at":    now_str,
        "start_time":     start_time,
        "start_datetime": start_datetime,
        "horse_number_map": horse_number_map,
        "horse_id_map":     horse_id_map,
        "prediction_version": 1,
        "prediction_history": [initial_history_entry],
        "horses":             horses,
        "ev_table":           ev_table,
        "race_structure":     race_structure,
        "danger_v2":          danger_v2,
        "odds_update_history": [],
        "odds_before":        dict(odds_shell),
        "odds_after":         dict(odds_shell),
    }
    _save(PREDICTIONS_FILE, data)


# =========================================================
# bet_suggestions
# =========================================================

def save_bet_suggestions(race_id: str, bets: List[Dict[str, Any]]) -> None:
    """全券種買い目リストを保存する。"""
    data = _load(BET_SUGGESTIONS_FILE)
    data[race_id] = bets
    _save(BET_SUGGESTIONS_FILE, data)


def load_bet_suggestions(race_id: str) -> List[Dict[str, Any]]:
    return _load(BET_SUGGESTIONS_FILE).get(race_id, [])


def load_all_bet_suggestions() -> Dict[str, Any]:
    return _load(BET_SUGGESTIONS_FILE)


# =========================================================
# pipeline_race_results（既存の race_results.json とは別）
# =========================================================

def save_pipeline_race_result(race_id: str, result: Dict[str, Any]) -> None:
    data = _load(RACE_RESULTS_FILE)
    data[race_id] = result
    _save(RACE_RESULTS_FILE, data)


def load_pipeline_race_result(race_id: str) -> Optional[Dict[str, Any]]:
    return _load(RACE_RESULTS_FILE).get(race_id)


# =========================================================
# bet_outcomes
# =========================================================

def save_bet_outcomes(race_id: str, outcomes: List[Dict[str, Any]]) -> None:
    data = _load(BET_OUTCOMES_FILE)
    data[race_id] = outcomes
    _save(BET_OUTCOMES_FILE, data)


def load_bet_outcomes(race_id: str) -> List[Dict[str, Any]]:
    return _load(BET_OUTCOMES_FILE).get(race_id, [])


def load_all_bet_outcomes() -> Dict[str, Any]:
    return _load(BET_OUTCOMES_FILE)
