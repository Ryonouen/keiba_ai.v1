# dashboard_loader.py
"""
パイプラインダッシュボード用データ読み込み・集計モジュール。
Streamlit に非依存。pipeline_*.json を読み取り専用で参照する。
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_PREDICTIONS_FILE     = os.path.join(_HERE, "pipeline_predictions.json")
_BET_SUGGESTIONS_FILE = os.path.join(_HERE, "pipeline_bet_suggestions.json")
_BET_OUTCOMES_FILE    = os.path.join(_HERE, "pipeline_bet_outcomes.json")

# running_style（英語キー）→ 日本語表示
STYLE_MAP: Dict[str, str] = {
    "front":   "逃げ",
    "stalker": "先行",
    "mid":     "差し",
    "closer":  "追込",
    "unknown": "不明",
}


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _parse_venue_race_number(race_name: str) -> Tuple[str, str]:
    """
    race_name から会場名と回次を抽出する。
    例: "３歳未勝利 出馬表 | 2026年4月5日 中山1R レース情報" → ("中山", "1R")
    """
    m = re.search(r"([^\d\s|]+)(\d+R)", race_name)
    if m:
        return m.group(1), m.group(2)
    return "", ""


def _race_status(start_datetime: Optional[str], outcomes: List[Dict]) -> str:
    """
    レースの表示ステータスを返す。
    - outcomes あり → "result"
    - start_datetime が 30分以上前 → "awaiting"（evaluate 待ち）
    - それ以外 → "prerace"
    """
    if outcomes:
        return "result"
    if start_datetime:
        try:
            t = datetime.fromisoformat(start_datetime)
            if datetime.now() > t + timedelta(minutes=30):
                return "awaiting"
        except (ValueError, TypeError):
            pass
    return "prerace"


def _build_horse_row(horse: Dict) -> Dict:
    """
    v1 / v2 スキーマ両対応で馬情報を正規化する。
    v2: horse["feature_dict"]["win_odds"] / ["running_style"]
    v1: horse["win_odds"] = null
    """
    fd = horse.get("feature_dict")
    is_v2 = fd is not None
    win_odds_raw      = fd.get("win_odds")        if is_v2 else horse.get("win_odds")
    running_style_raw = fd.get("running_style")   if is_v2 else horse.get("running_style")
    popularity_raw    = fd.get("feat_popularity") if is_v2 else horse.get("popularity")
    return {
        "horse_name":    horse.get("horse_name", ""),
        "ai_win_prob":   horse.get("ai_win_prob"),
        "win_odds":      win_odds_raw,
        "popularity":    int(popularity_raw) if popularity_raw is not None else None,
        "running_style": STYLE_MAP.get(running_style_raw or "", None),
    }


def get_available_dates() -> List[str]:
    """
    pipeline_predictions.json に存在する analysis_date の一覧を降順で返す。
    """
    preds = _load_json(_PREDICTIONS_FILE)
    dates = sorted(
        {v.get("analysis_date", "") for v in preds.values() if v.get("analysis_date")},
        reverse=True,
    )
    return dates


def load_races_for_date(date_str: str) -> List[Dict]:
    """
    指定日のレース一覧を predictions + bets + outcomes を合体して返す。

    Returns
    -------
    List[Dict] — 各要素のキー:
        race_id, race_name, venue, race_number,
        start_time, start_datetime, status,
        horses, bets, outcomes
    """
    preds    = _load_json(_PREDICTIONS_FILE)
    bets_all = _load_json(_BET_SUGGESTIONS_FILE)
    outs_all = _load_json(_BET_OUTCOMES_FILE)

    races: List[Dict] = []
    for race_id, pred in preds.items():
        if pred.get("analysis_date") != date_str:
            continue

        race_name     = pred.get("race_name", "")
        venue, r_num  = _parse_venue_race_number(race_name)
        start_dt      = pred.get("start_datetime")
        start_time    = pred.get("start_time")
        outcomes      = outs_all.get(race_id) or []
        bets          = bets_all.get(race_id) or []
        horses        = [_build_horse_row(h) for h in (pred.get("horses") or [])]
        horses.sort(key=lambda h: -(h["ai_win_prob"] or 0))

        races.append(
            {
                "race_id":        race_id,
                "race_name":      race_name,
                "venue":          venue,
                "race_number":    r_num,
                "start_time":     start_time,
                "start_datetime": start_dt,
                "status":         _race_status(start_dt, outcomes),
                "horses":         horses,
                "bets":           bets,
                "outcomes":       outcomes,
            }
        )

    # 発走時刻順（不明は末尾）
    races.sort(key=lambda r: r.get("start_time") or "99:99")
    return races


def calc_kpi(races: List[Dict]) -> Dict:
    """
    レース一覧から全体KPIを計算する。

    Returns
    -------
    {total_stake, total_payout, roi, hit_count, total_bets}
    """
    total_stake = total_payout = hit_count = total_bets = 0

    for race in races:
        for o in race.get("outcomes", []):
            stake         = o.get("stake", 100)
            total_stake  += stake
            total_payout += o.get("payout", 0)
            total_bets   += 1
            if o.get("hit"):
                hit_count += 1

    roi = round(total_payout / total_stake * 100, 1) if total_stake > 0 else 0.0
    return {
        "total_stake":  total_stake,
        "total_payout": total_payout,
        "roi":          roi,
        "hit_count":    hit_count,
        "total_bets":   total_bets,
    }


def calc_kpi_by_bet_type(races: List[Dict]) -> List[Dict]:
    """
    券種別KPIを計算する。

    Returns
    -------
    List[Dict] — 各要素: {bet_type, label, count, hit, hit_rate, roi}
    """
    stats: Dict[str, Dict] = defaultdict(
        lambda: {"label": "", "count": 0, "hit": 0, "stake": 0, "payout": 0}
    )

    for race in races:
        for o in race.get("outcomes", []):
            bt = o.get("bet_type", "other")
            s  = stats[bt]
            s["label"]   = o.get("bet_type_label", bt)
            s["count"]  += 1
            s["stake"]  += o.get("stake", 100)
            s["payout"] += o.get("payout", 0)
            if o.get("hit"):
                s["hit"] += 1

    rows = []
    for bt, s in stats.items():
        hit_rate = round(s["hit"] / s["count"] * 100, 1) if s["count"] > 0 else 0.0
        roi      = round(s["payout"] / s["stake"] * 100, 1) if s["stake"] > 0 else 0.0
        rows.append(
            {
                "bet_type": bt,
                "label":    s["label"],
                "count":    s["count"],
                "hit":      s["hit"],
                "hit_rate": hit_rate,
                "roi":      roi,
            }
        )
    rows.sort(key=lambda r: r["bet_type"])
    return rows
