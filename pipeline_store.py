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
