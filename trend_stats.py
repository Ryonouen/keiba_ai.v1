"""
trend_stats.py
条件別集計テーブルビルダー

責務:
- 全走者付き過去レース履歴から条件別統計テーブルを構築
- 全体基準値（overall_win_rate / overall_top3_rate）を計算
- 各条件値に diff_win / diff_top3 を付与
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

# =========================================================
# バケット境界定数
# =========================================================

AGE_SENIOR_THRESHOLD: int = 7    # 7歳以上はまとめてグループ化

GATE_INNER_MAX:  int = 3          # 内枠: 1〜3
GATE_MIDDLE_MAX: int = 6          # 中枠: 4〜6
                                   # 外枠: 7〜8 (それ以降)

# 集計対象条件（辞書キー）
CONDITION_KEYS = ("年齢", "枠", "脚質", "人気帯")


# =========================================================
# バケット関数
# =========================================================

def bucket_age(age: Optional[int]) -> Optional[str]:
    """年齢 → バケットキー。7歳以上はまとめて '7歳以上'。"""
    if age is None:
        return None
    if age >= AGE_SENIOR_THRESHOLD:
        return "7歳以上"
    return f"{age}歳"


def bucket_gate(gate: Optional[int]) -> Optional[str]:
    """枠番 → バケットキー。"""
    if gate is None:
        return None
    if gate <= GATE_INNER_MAX:
        return "内枠(1〜3)"
    if gate <= GATE_MIDDLE_MAX:
        return "中枠(4〜6)"
    return "外枠(7〜)"


def bucket_style(running_style: Optional[str]) -> Optional[str]:
    """脚質（英語キー） → 日本語バケットキー。"""
    _MAP = {"front": "逃げ", "stalker": "先行", "closer": "差し"}
    if not running_style or running_style == "unknown":
        return None
    return _MAP.get(running_style)


def bucket_popularity_rank(popularity: Optional[int]) -> Optional[str]:
    """人気番号 → 人気帯バケットキー。"""
    if popularity is None:
        return None
    if popularity == 1:
        return "1番人気"
    if 2 <= popularity <= 3:
        return "2〜3番人気"
    if 4 <= popularity <= 6:
        return "4〜6番人気"
    if 7 <= popularity <= 9:
        return "7〜9番人気"
    return "10番人気以下"


def bucket_popularity_odds(win_odds: Optional[float]) -> Optional[str]:
    """単勝オッズ → 人気帯バケットキー（人気番号なしの場合に使用）。"""
    if win_odds is None:
        return None
    odds_f = float(win_odds)
    if odds_f <= 3.0:
        return "1番人気"
    if odds_f <= 8.0:
        return "2〜3番人気"
    if odds_f <= 15.0:
        return "4〜6番人気"
    if odds_f <= 30.0:
        return "7〜9番人気"
    return "10番人気以下"


def bucket_distance(distance: Optional[int]) -> Optional[str]:
    """距離 → 距離帯バケットキー。"""
    if distance is None:
        return None
    if distance <= 1399:
        return "短距離(〜1400)"
    if distance <= 1799:
        return "マイル(1400〜1800)"
    if distance <= 2199:
        return "中距離(1800〜2200)"
    return "長距離(2200〜)"


def get_condition_keys(runner: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """runner dict から各条件のバケットキーを返す。"""
    pop_key = (
        bucket_popularity_rank(runner.get("popularity"))
        or bucket_popularity_odds(runner.get("odds"))
    )
    return {
        "年齢":  bucket_age(runner.get("age")),
        "枠":    bucket_gate(runner.get("gate")),
        "脚質":  bucket_style(runner.get("running_style")),
        "人気帯": pop_key,
    }


# =========================================================
# 集計テーブル構築
# =========================================================

def build_condition_stats(
    history_enriched: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    全走者付き過去レース履歴から条件別統計テーブルを構築する。

    Parameters
    ----------
    history_enriched : fetch_race_history_enriched() の戻り値
        各要素: {
            "race_id":   str,
            "n_runners": int,
            "runners":   List[{"rank": int, "gate": int, "age": int,
                               "popularity": int, "odds": float,
                               "running_style": str}]
        }

    Returns
    -------
    {
        "_overall": {
            "n_races":           int,
            "avg_field_size":    float,
            "overall_win_rate":  float,
            "overall_top3_rate": float,
        },
        "年齢": {
            "4歳": {
                "sample_size": int,
                "wins":        int,
                "top3":        int,
                "win_rate":    float,
                "top3_rate":   float,
                "diff_win":    float,   # win_rate  - overall_win_rate
                "diff_top3":   float,   # top3_rate - overall_top3_rate
            },
            "5歳": {...},
            "7歳以上": {...},
            ...
        },
        "枠": {...},
        "脚質": {...},
        "人気帯": {...},
    }
    空の history_enriched が渡されたら空 dict を返す。
    """
    if not history_enriched:
        return {}

    n_races       = len(history_enriched)
    total_runners = sum(r.get("n_runners", 0) for r in history_enriched)
    avg_field     = total_runners / n_races if n_races > 0 else 18.0
    overall_win   = round(1.0 / avg_field, 4)
    overall_top3  = round(min(1.0, 3.0 / avg_field), 4)

    # { cond_name: { bucket_key: {"sample": int, "wins": int, "top3": int} } }
    counters: Dict[str, Any] = {
        k: defaultdict(lambda: {"sample": 0, "wins": 0, "top3": 0})
        for k in CONDITION_KEYS
    }

    for race in history_enriched:
        for runner in race.get("runners", []):
            rank = runner.get("rank")
            if rank is None:
                continue
            keys = get_condition_keys(runner)
            for cond, key in keys.items():
                if not key:
                    continue
                c = counters[cond][key]
                c["sample"] += 1
                if rank <= 3:
                    c["top3"] += 1
                if rank == 1:
                    c["wins"] += 1

    result: Dict[str, Any] = {
        "_overall": {
            "n_races":           n_races,
            "avg_field_size":    round(avg_field, 2),
            "overall_win_rate":  overall_win,
            "overall_top3_rate": overall_top3,
        }
    }

    for cond, buckets in counters.items():
        result[cond] = {}
        for key, c in sorted(buckets.items()):
            sample = c["sample"]
            wins   = c["wins"]
            top3   = c["top3"]
            wr  = round(wins / sample, 4) if sample > 0 else 0.0
            t3r = round(top3 / sample, 4) if sample > 0 else 0.0
            result[cond][key] = {
                "sample_size": sample,
                "wins":        wins,
                "top3":        top3,
                "win_rate":    wr,
                "top3_rate":   t3r,
                "diff_win":    round(wr  - overall_win,  4),
                "diff_top3":   round(t3r - overall_top3, 4),
            }

    return result
