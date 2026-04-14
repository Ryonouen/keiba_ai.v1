# dashboard_loader.py
"""
パイプラインダッシュボード用データ読み込み・集計モジュール。
Streamlit に非依存。pipeline_*.json を読み取り専用で参照する。
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from datetime import date as _date
from datetime import datetime, timedelta
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

from dividend_scraper import _fetch_html
from netkeiba_scrape_helpers import parse_today_races_from_html

_HERE = os.path.dirname(os.path.abspath(__file__))
_PREDICTIONS_FILE     = os.path.join(_HERE, "pipeline_predictions.json")
_BET_SUGGESTIONS_FILE = os.path.join(_HERE, "pipeline_bet_suggestions.json")
_BET_OUTCOMES_FILE    = os.path.join(_HERE, "pipeline_bet_outcomes.json")
_RACE_RESULTS_FILE    = os.path.join(_HERE, "pipeline_race_results.json")
_RACE_LIST_URL        = "https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date}"
_SHUTUBA_URL          = "https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"

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


_TITLE_PATTERN = re.compile(r"^(.+?)\s*出馬表")


def _extract_race_title(race_name: str) -> str:
    """
    race_name からレースカテゴリ名を抽出する。
    例: "天満橋S・3勝 出馬表 | ..." → "天満橋S・3勝"
         "大阪杯(G1) 出馬表 | ..."  → "大阪杯(G1)"
    """
    if not race_name:
        return ""
    m = _TITLE_PATTERN.match(race_name)
    return m.group(1).strip() if m else ""


_GRADE_PATTERN = _TITLE_PATTERN  # 同じパターンを共有


def _extract_grade_title(race_name: str) -> Optional[str]:
    """
    race_name が重賞（G1/G2/G3）の場合はレースタイトルを返す。
    例: "大阪杯(G1) 出馬表 | ..." → "大阪杯(G1)"
    非重賞または空文字は None を返す。
    """
    if not race_name:
        return None
    if not any(g in race_name for g in ("G1", "G2", "G3", "Ｇ１", "Ｇ２", "Ｇ３")):
        return None
    m = _GRADE_PATTERN.match(race_name)
    if m:
        return m.group(1).strip()
    return None


def _race_status(
    start_datetime: Optional[str],
    outcomes: List[Dict],
    finish_order: Optional[List] = None,
) -> str:
    """
    レースの表示ステータスを返す。
    - outcomes あり、または finish_order あり → "result"
    - start_datetime が 30分以上前 → "awaiting"（evaluate 待ち）
    - それ以外 → "prerace"
    """
    if outcomes or finish_order:
        return "result"
    if start_datetime:
        try:
            t = datetime.fromisoformat(start_datetime)
            if datetime.now() > t + timedelta(minutes=30):
                return "awaiting"
        except (ValueError, TypeError):
            pass
    return "prerace"


def _build_start_datetime(date_str: str, start_time: Optional[str]) -> str:
    if not date_str or len(date_str) != 8 or not start_time:
        return ""
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T{start_time}:00"


def _fetch_schedule_map_for_date(date_str: str) -> Dict[str, Dict[str, Any]]:
    html = _fetch_html(_RACE_LIST_URL.format(date=date_str))
    if not html:
        return {}

    races = parse_today_races_from_html(html)
    result: Dict[str, Dict[str, Any]] = {}
    for race in races:
        race_id = str(race.get("race_id") or "")
        if not race_id:
            continue
        post_time = str(race.get("post_time") or "").strip()
        result[race_id] = {
            "start_time": post_time,
            "start_datetime": _build_start_datetime(date_str, post_time) if post_time else "",
            "head_count": race.get("head_count"),
        }
    return result


def _fill_popularity_from_odds(horses: List[Dict[str, Any]]) -> None:
    ranked = [
        (idx, float(h["win_odds"]))
        for idx, h in enumerate(horses)
        if h.get("win_odds") not in (None, "", 0)
    ]
    ranked.sort(key=lambda item: item[1])
    for rank, (idx, _) in enumerate(ranked, 1):
        if horses[idx].get("popularity") in (None, "", 0):
            horses[idx]["popularity"] = rank


def _extract_first_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    m = re.search(r"(?<!\d)(\d{1,2})(?!\d)", str(value).replace(",", ""))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _extract_first_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", str(value).replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _fetch_entry_fallback_for_race(race_id: str, date_str: str) -> Dict[str, Any]:
    html = _fetch_html(_SHUTUBA_URL.format(race_id=race_id))
    if not html:
        return {}

    result: Dict[str, Any] = {"horses_by_name": {}}
    flat_text = re.sub(r"\s+", " ", html)
    m = re.search(r"(\d{1,2}:\d{2})\s*発走", flat_text)
    if m:
        start_time = m.group(1)
        result["start_time"] = start_time
        result["start_datetime"] = _build_start_datetime(date_str, start_time)

    try:
        tables = pd.read_html(StringIO(html))
    except Exception:
        return result

    for tbl in tables:
        df_tbl = tbl.copy()
        df_tbl.columns = [str(c) for c in df_tbl.columns]

        horse_col = None
        number_col = None
        gate_col = None
        odds_col = None
        popularity_col = None

        for c in df_tbl.columns:
            c_str = str(c)
            c_lower = c_str.lower()
            if horse_col is None and any(k in c_str for k in ["馬名", "Horse"]):
                horse_col = c
            if number_col is None and any(k in c_str for k in ["馬番", "馬 番", "Umaban"]):
                number_col = c
            if gate_col is None and "枠" in c_str:
                gate_col = c
            if odds_col is None and any(k in c_lower for k in ["odds", "単勝", "予想オッズ"]):
                odds_col = c
            if popularity_col is None and "人気" in c_str:
                popularity_col = c

        if horse_col is None:
            continue

        horses_by_name: Dict[str, Dict[str, Any]] = {}
        for _, row in df_tbl.iterrows():
            horse_name = str(row.get(horse_col, "")).strip()
            if not horse_name or horse_name == "nan":
                continue
            horses_by_name[horse_name] = {
                "horse_no": _extract_first_int(row.get(number_col, "")) if number_col is not None else None,
                "gate": _extract_first_int(row.get(gate_col, "")) if gate_col is not None else None,
                "win_odds": _extract_first_float(row.get(odds_col, "")) if odds_col is not None else None,
                "popularity": _extract_first_int(row.get(popularity_col, "")) if popularity_col is not None else None,
            }

        if horses_by_name:
            result["horses_by_name"] = horses_by_name
            result["head_count"] = len(horses_by_name)
            return result

    return result


def _needs_entry_fallback(pred: Dict[str, Any], horses: List[Dict[str, Any]]) -> bool:
    horse_number_map = pred.get("horse_number_map") or {}
    if not horse_number_map:
        return True
    if any(h.get("horse_no") in (None, "", 0) for h in horses):
        return True
    if all(h.get("win_odds") in (None, "", 0) for h in horses):
        return True
    if all(h.get("popularity") in (None, "", 0) for h in horses):
        return True
    return False


def _merge_entry_fallback(horses: List[Dict[str, Any]], fallback: Dict[str, Any]) -> None:
    horses_by_name = fallback.get("horses_by_name") or {}
    if not horses_by_name:
        return

    for h in horses:
        entry = horses_by_name.get(str(h.get("horse_name") or ""))
        if not entry:
            continue
        if h.get("horse_no") in (None, "", 0):
            h["horse_no"] = entry.get("horse_no")
        if h.get("gate") in (None, "", 0):
            h["gate"] = entry.get("gate")
        if h.get("win_odds") in (None, "", 0):
            h["win_odds"] = entry.get("win_odds")
        if h.get("popularity") in (None, "", 0):
            h["popularity"] = entry.get("popularity")


def _build_horse_row(horse: Dict) -> Dict:
    """
    v1 / v2 スキーマ両対応で馬情報を正規化する。

    新形式 (v2): feature_dict に place_prob / win_ev が直接入る。
    旧形式 (v1/sparse): 以下のフォールバック計算を適用する。
      - win_ev     = ai_win_prob × win_odds  （exact: EV の定義通り）
      - place_prob = 1 - (1 - p)^(n/3)  （Harville 独立近似: n=出走頭数）
      - top2_prob  = win_prob + (place_prob - win_prob) × 2/3  （補間）
    """
    fd = horse.get("feature_dict")
    is_v2 = fd is not None
    win_odds_raw      = fd.get("win_odds")        if is_v2 else horse.get("win_odds")
    running_style_raw = fd.get("running_style")   if is_v2 else horse.get("running_style")
    popularity_raw    = fd.get("feat_popularity") if is_v2 else horse.get("popularity")
    place_prob_raw    = fd.get("place_prob")       if is_v2 else None
    win_ev_raw        = fd.get("win_ev")           if is_v2 else None
    n_runners_raw     = fd.get("feat_n_runners")   if is_v2 else None

    win_prob  = horse.get("ai_win_prob")
    win_odds  = float(win_odds_raw) if win_odds_raw is not None else None
    n_runners = int(float(n_runners_raw)) if n_runners_raw is not None else 16

    # ── win_ev ──
    if win_ev_raw is not None:
        win_ev: Optional[float] = float(win_ev_raw)
    elif win_prob is not None and win_odds is not None:
        win_ev = win_prob * win_odds   # EV の定義: 勝率 × 配当倍率
    else:
        win_ev = None

    # ── place_prob (3着内率) ──
    if place_prob_raw is not None:
        place_prob: Optional[float] = float(place_prob_raw)
    elif win_prob is not None:
        # Harville 独立近似: P(top-3) ≈ 1 - (1-p)^(n/3)
        # n/3 は「3頭を順に選ぶ試行の有効回数」。n=16 のとき ≈ 5.33
        k = max(n_runners / 3, 1.0)
        place_prob = min(1.0 - (1.0 - win_prob) ** k, 0.99)
    else:
        place_prob = None

    # ── top2_prob (2着内率) ── 勝率〜3着内率の 2/3 補間
    top2_prob: Optional[float] = None
    if win_prob is not None and place_prob is not None:
        top2_prob = win_prob + (place_prob - win_prob) * 2 / 3

    return {
        "horse_name":    horse.get("horse_name", ""),
        "ai_win_prob":   win_prob,
        "place_prob":    place_prob,
        "top2_prob":     top2_prob,
        "win_ev":        win_ev,
        "win_odds":      win_odds,
        "popularity":    int(popularity_raw) if popularity_raw is not None else None,
        "running_style": STYLE_MAP.get(running_style_raw or "", None),
        "ability_score": float(fd.get("ability_score")) if is_v2 and fd.get("ability_score") is not None else None,
        "raw_ability_score": float(fd.get("raw_ability_score")) if is_v2 and fd.get("raw_ability_score") is not None else None,
        "odds_is_estimated": bool(fd.get("odds_is_estimated")) if is_v2 else bool(horse.get("odds_is_estimated")),
        "feature_dict":  fd if is_v2 else None,
        # enriched fields (filled by _enrich_horses_from_result)
        "horse_no":      None,
        "gate":          None,
        "jockey":        None,
        "actual_rank":   None,
    }


def _enrich_horses_from_result_dict(race_result: Dict, horses: List[Dict]) -> None:
    """
    race_result dict（pipeline_race_results.json の 1 レース分）から
    騎手・馬番・枠番・実際の着順を horses に付与する（in-place）。
    race_result が空 dict の場合は何もしない。
    """
    if not race_result:
        return
    runner_map = {r["horse_name"]: r for r in (race_result.get("runners") or [])}
    finish_order = race_result.get("finish_order") or []
    finish_rank_map = {name: i + 1 for i, name in enumerate(finish_order)}
    for h in horses:
        name = h.get("horse_name", "")
        runner = runner_map.get(name, {})
        h["horse_no"]    = runner.get("horse_no")
        h["gate"]        = runner.get("gate")
        h["jockey"]      = runner.get("jockey")
        h["actual_rank"] = finish_rank_map.get(name)


def _enrich_horses_from_result(race_id: str, horses: List[Dict]) -> None:
    """後方互換ラッパー。テスト等から直接呼ばれる場合向け。"""
    results = _load_json(_RACE_RESULTS_FILE)
    _enrich_horses_from_result_dict(results.get(str(race_id), {}), horses)


# ──────────────────────────────────────────────────────────────
# 荒れスコア
# ──────────────────────────────────────────────────────────────

_UPSET_LEVELS = [
    (30,  "堅い",     "#27ae60"),
    (45,  "やや堅い", "#8bc34a"),
    (60,  "中間",     "#ff9800"),
    (75,  "やや荒れ", "#e64a19"),
    (101, "荒れ",     "#c0392b"),
]


def calc_upset_score(horses: List[Dict]) -> Dict[str, Any]:
    """
    馬リストから荒れスコア (0–100 整数) を計算する。

    Returns
    -------
    {"score": int, "label": str, "color": str}
    """
    probs = [float(h.get("ai_win_prob") or 0.0) for h in horses]
    probs = [p for p in probs if p > 0.0]

    if not probs:
        return {"score": 50, "label": "中間", "color": "#ff9800"}

    # Shannon エントロピー → 0–100 に正規化
    total = sum(probs)
    norm_p = [p / total for p in probs]
    entropy = -sum(p * math.log2(p) for p in norm_p if p > 0.0)
    max_entropy = math.log2(len(norm_p)) if len(norm_p) > 1 else 1.0
    entropy_score = (entropy / max_entropy * 100.0) if max_entropy > 0.0 else 0.0

    # 1番人気オッズ → 0–100 に変換（最大 30 倍で cap）
    odds_list = [
        float(h.get("win_odds"))
        for h in horses
        if h.get("win_odds") is not None
    ]
    if odds_list:
        top_odds = min(odds_list)          # 最低オッズ = 1番人気
        odds_score = min(top_odds, 30.0) / 30.0 * 100.0
        w_e, w_o = 0.6, 0.4
    else:
        odds_score = 0.0
        w_e, w_o = 1.0, 0.0               # オッズなし → エントロピーのみ

    score = max(0, min(100, round(w_e * entropy_score + w_o * odds_score)))

    for threshold, label, color in _UPSET_LEVELS:
        if score < threshold:
            return {"score": score, "label": label, "color": color}

    return {"score": score, "label": "荒れ", "color": "#c0392b"}


def _float_or_none(value: Any, digits: int = 4) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _eval_group_from_prob(ai_win_prob: Optional[float]) -> str:
    ai_p = float(ai_win_prob or 0.0)
    if ai_p >= 0.20:
        return "S"
    if ai_p >= 0.12:
        return "A"
    if ai_p >= 0.07:
        return "B"
    return "C"


def _build_score_debug_row(horse: Dict[str, Any]) -> Dict[str, Any]:
    fd = horse.get("feature_dict") or {}

    def g(key: str, default: float = 0.0) -> float:
        try:
            value = fd.get(key, default)
            if value in (None, ""):
                return float(default)
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    base_score = _float_or_none(fd.get("model_score_before_trend"))
    final_score = _float_or_none(fd.get("model_score"))
    softmax_input_score = final_score

    draw_bonus = 0.03 * g("gate_index", 0.0)
    pace_bonus = (
        0.12 * g("style_suitability_index", 0.0)
        + 0.07 * g("pace_bias_index", 0.0)
        + 0.04 * g("pace_advantage", 0.0)
        + 0.06 * g("lap_suitability_index", 0.0)
    )
    course_bonus = (
        0.16 * g("distance_course_suitability_index", 0.0)
        + 0.03 * g("ground_match_index", 0.0)
        + 0.03 * g("ground_fit_index", 0.0)
    )
    distance_bonus = 0.03 * g("distance_fit_index", 0.0) + 0.02 * g("distance_change_index", 0.0)
    jockey_bonus = 0.06 * g("jockey_index", 0.0) + 0.02 * g("trainer_jockey_synergy_index", 0.0)
    recent_form_bonus = (
        0.13 * g("recent_form_index", 0.0)
        + 0.02 * g("last_margin_index", 0.0)
        + 0.07 * g("last3f_index", 0.0)
        + 0.06 * g("trend_index", 0.0)
        + 0.03 * g("consistency_index", 0.0)
        + 0.01 * g("expectation_deviation_index", 0.0)
    )
    class_bonus = 0.05 * g("race_level_index", 0.0) + 0.02 * g("class_change_index", 0.0)
    rotation_bonus = 0.02 * g("rotation_index", 0.0) + 0.01 * g("weight_change_index", 0.0)
    track_bias_bonus = 0.0

    return {
        "horse_name": horse.get("horse_name", ""),
        "base_score": base_score,
        "ability_score_raw": _float_or_none(horse.get("raw_ability_score")),
        "ability_score_display": _float_or_none(horse.get("ability_score")),
        "draw_bonus": round(draw_bonus, 4),
        "pace_bonus": round(pace_bonus, 4),
        "course_bonus": round(course_bonus, 4),
        "distance_bonus": round(distance_bonus, 4),
        "jockey_bonus": round(jockey_bonus, 4),
        "recent_form_bonus": round(recent_form_bonus, 4),
        "class_bonus": round(class_bonus, 4),
        "rotation_bonus": round(rotation_bonus, 4),
        "track_bias_bonus": round(track_bias_bonus, 4),
        "final_score_before_prob": final_score,
        "softmax_input_score": softmax_input_score,
        "ai_win_prob": _float_or_none(horse.get("ai_win_prob")),
        "ai_place2_prob": _float_or_none(horse.get("top2_prob")),
        "ai_place3_prob": _float_or_none(horse.get("place_prob")),
        "market_win_prob": _float_or_none(horse.get("market_win_prob")),
        "gap_vs_market": _float_or_none(horse.get("prob_gap")),
        "rating_rank": horse.get("eval_group") or _eval_group_from_prob(horse.get("ai_win_prob")),
    }


def _build_top_vs_bottom_debug(score_debug_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not score_debug_rows:
        return {}

    top = score_debug_rows[0]
    bottom = score_debug_rows[-1]
    diff: Dict[str, Any] = {}
    for key, value in top.items():
        if key in ("horse_name", "rating_rank"):
            continue
        top_val = top.get(key)
        bottom_val = bottom.get(key)
        if isinstance(top_val, (int, float)) and isinstance(bottom_val, (int, float)):
            diff[key] = round(float(top_val) - float(bottom_val), 4)
        else:
            diff[key] = None

    return {
        "top_horse_name": top.get("horse_name", ""),
        "bottom_horse_name": bottom.get("horse_name", ""),
        "top": top,
        "bottom": bottom,
        "diff": diff,
    }



def calc_hot_bets(bets: List[Dict]) -> List[Dict]:
    """
    confidence >= 0.75 かつ expected_value > 1.1（None は条件スキップ）
    の買い目リストを返す。

    Parameters
    ----------
    bets : pipeline_bet_suggestions.json から読んだ買い目リスト

    Returns
    -------
    条件を満たした bet dict のリスト
    """
    result: List[Dict] = []
    for bet in bets:
        conf = float(bet.get("confidence") or 0.0)
        if conf < 0.75:
            continue
        ev = bet.get("expected_value")
        if ev is not None and float(ev) <= 1.1:
            continue
        result.append(bet)
    return result


# ──────────────────────────────────────────────────────────────
# ダッシュボード用補助シグナル
# ──────────────────────────────────────────────────────────────

def _enrich_dashboard_signals(horses: List[Dict], upset: Dict[str, Any]) -> Dict[str, Any]:
    """
    UI表示用の補助シグナルを horses / race 単位で付与する。

    horses には以下を in-place で追加する:
      - market_win_prob
      - prob_gap
      - eval_group
      - is_value
      - is_danger

    Returns
    -------
    {
      "race_type_label": str,
      "race_type_color": str,
      "axis_confidence": str,
      "has_value_horse": bool,
      "has_danger_favorite": bool,
    }
    """
    if not horses:
        return {
            "race_type_label": "中間",
            "race_type_color": "#ff9800",
            "axis_confidence": "低",
            "has_value_horse": False,
            "has_danger_favorite": False,
        }

    top_win_prob = max(float(h.get("ai_win_prob") or 0.0) for h in horses)
    second_win_prob = 0.0
    if len(horses) >= 2:
        second_win_prob = float(horses[1].get("ai_win_prob") or 0.0)
    top_gap = top_win_prob - second_win_prob

    has_value_horse = False
    has_danger_favorite = False

    for h in horses:
        ai_win_prob = h.get("ai_win_prob")
        win_odds = h.get("win_odds")
        market_win_prob: Optional[float]
        if win_odds is not None and float(win_odds) > 0:
            market_win_prob = min(1.0 / float(win_odds), 0.99)
        else:
            market_win_prob = None

        prob_gap: Optional[float]
        if ai_win_prob is not None and market_win_prob is not None:
            prob_gap = float(ai_win_prob) - float(market_win_prob)
        else:
            prob_gap = None

        eval_group = _eval_group_from_prob(ai_win_prob)

        is_value = prob_gap is not None and prob_gap >= 0.03
        is_danger = prob_gap is not None and prob_gap <= -0.04

        h["market_win_prob"] = market_win_prob
        h["prob_gap"] = prob_gap
        h["eval_group"] = eval_group
        h["is_value"] = is_value
        h["is_danger"] = is_danger

        if is_value:
            has_value_horse = True
        if is_danger:
            has_danger_favorite = True

    upset_score = int(upset.get("score", 50))
    if upset_score >= 75:
        race_type_label = "波乱"
        race_type_color = "#c0392b"
    elif upset_score >= 60:
        race_type_label = "やや荒れ"
        race_type_color = "#e64a19"
    elif upset_score >= 45:
        race_type_label = "混戦"
        race_type_color = "#ff9800"
    else:
        race_type_label = "堅い"
        race_type_color = "#2e7d32"

    if top_win_prob >= 0.28 and top_gap >= 0.06:
        axis_confidence = "高"
    elif top_win_prob >= 0.20 and top_gap >= 0.03:
        axis_confidence = "中"
    else:
        axis_confidence = "低"

    return {
        "race_type_label": race_type_label,
        "race_type_color": race_type_color,
        "axis_confidence": axis_confidence,
        "has_value_horse": has_value_horse,
        "has_danger_favorite": has_danger_favorite,
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
        race_id, race_name, grade_title, venue, race_number,
        start_time, start_datetime, status,
        horses, bets, outcomes,
        upset_score, upset_label, upset_color, hot_bets
    """
    preds    = _load_json(_PREDICTIONS_FILE)
    bets_all = _load_json(_BET_SUGGESTIONS_FILE)
    outs_all = _load_json(_BET_OUTCOMES_FILE)
    rr_all   = _load_json(_RACE_RESULTS_FILE)   # 1回だけ読む
    needs_schedule_fallback = any(
        pred.get("analysis_date") == date_str
        and (not pred.get("start_time") or not pred.get("start_datetime"))
        for pred in preds.values()
    )
    schedule_map = _fetch_schedule_map_for_date(date_str) if needs_schedule_fallback else {}

    races: List[Dict] = []
    for race_id, pred in preds.items():
        if pred.get("analysis_date") != date_str:
            continue

        race_name     = pred.get("race_name", "")
        venue, r_num  = _parse_venue_race_number(race_name)
        fallback_schedule = schedule_map.get(str(race_id), {})
        start_time    = pred.get("start_time") or fallback_schedule.get("start_time")
        start_dt      = pred.get("start_datetime") or fallback_schedule.get("start_datetime")
        outcomes      = outs_all.get(race_id) or []
        bets          = bets_all.get(race_id) or []
        horses        = [_build_horse_row(h) for h in (pred.get("horses") or [])]
        horses.sort(key=lambda h: -(h["ai_win_prob"] or 0))

        race_result = rr_all.get(str(race_id), {})
        _enrich_horses_from_result_dict(race_result, horses)
        entry_fallback: Dict[str, Any] = {}
        if _needs_entry_fallback(pred, horses):
            entry_fallback = _fetch_entry_fallback_for_race(str(race_id), date_str)
            _merge_entry_fallback(horses, entry_fallback)
            if not start_time:
                start_time = entry_fallback.get("start_time") or start_time
            if not start_dt:
                start_dt = entry_fallback.get("start_datetime") or start_dt
        _fill_popularity_from_odds(horses)

        upset = calc_upset_score(horses)
        hot   = calc_hot_bets(bets)
        dashboard_signals = _enrich_dashboard_signals(horses, upset)
        score_debug_rows = [_build_score_debug_row(h) for h in horses]
        top_vs_bottom_debug = _build_top_vs_bottom_debug(score_debug_rows)

        races.append(
            {
                "race_id":        race_id,
                "race_name":      race_name,
                "race_title":     _extract_race_title(race_name),
                "grade_title":    _extract_grade_title(race_name),
                "venue":          venue,
                "race_number":    r_num,
                "start_time":     start_time,
                "start_datetime": start_dt,
                "status":         _race_status(start_dt, outcomes, race_result.get("finish_order")),
                "horses":         horses,
                "bets":           bets,
                "outcomes":       outcomes,
                "upset_score":    upset["score"],
                "upset_label":    upset["label"],
                "upset_color":    upset["color"],
                "hot_bets":       hot,
                "race_type_label": dashboard_signals["race_type_label"],
                "race_type_color": dashboard_signals["race_type_color"],
                "axis_confidence": dashboard_signals["axis_confidence"],
                "has_value_horse": dashboard_signals["has_value_horse"],
                "has_danger_favorite": dashboard_signals["has_danger_favorite"],
                "score_debug_rows": score_debug_rows,
                "top_vs_bottom_debug": top_vs_bottom_debug,
                "distance":       race_result.get("distance"),
                "surface":        race_result.get("surface"),
                "n_runners":      len(race_result.get("runners") or []) or fallback_schedule.get("head_count") or entry_fallback.get("head_count") or None,
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


def get_races_by_venue(date_str: str) -> Dict[str, List[Dict]]:
    """
    load_races_for_date() の結果を競馬場名でグループ化して返す。

    Returns
    -------
    {"中山": [...], "阪神": [...]}  発走時刻順を維持。
    """
    races = load_races_for_date(date_str)
    result: Dict[str, List[Dict]] = {}
    for race in races:
        venue = race.get("venue") or "不明"
        result.setdefault(venue, []).append(race)
    return result


def get_weekend_date_strs(_today: Optional[_date] = None) -> Tuple[str, str, str]:
    """
    今日・直近土曜・直近日曜の YYYYMMDD 文字列を返す。

    ルール:
      月〜金 → 次の週末（土/日）
      土     → 今日（土）+ 明日（日）
      日     → 昨日（土）+ 今日（日）

    Parameters
    ----------
    _today : テスト用の日付注入。省略時は datetime.now().date() を使用。
    """
    today = _today if _today is not None else datetime.now().date()
    wd = today.weekday()  # Mon=0, Tue=1, ..., Sat=5, Sun=6
    if wd == 5:  # Saturday
        saturday = today
        sunday = today + timedelta(days=1)
    elif wd == 6:  # Sunday
        saturday = today - timedelta(days=1)
        sunday = today
    else:  # Mon–Fri: next weekend
        saturday = today + timedelta(days=(5 - wd) % 7)
        sunday = saturday + timedelta(days=1)
    fmt = lambda d: d.strftime("%Y%m%d")
    return fmt(today), fmt(saturday), fmt(sunday)


def get_available_months() -> List[str]:
    """
    利用可能な年月リストを YYYYMM 形式で降順に返す。

    例: ["202604", "202603", "202512"]
    """
    dates = get_available_dates()
    months = sorted({d[:6] for d in dates if len(d) >= 6}, reverse=True)
    return months


def get_available_years() -> List[str]:
    """
    利用可能な年リストを YYYY 形式で降順に返す。

    例: ["2026", "2025"]
    """
    dates = get_available_dates()
    years = sorted({d[:4] for d in dates if len(d) >= 4}, reverse=True)
    return years


def get_daily_kpi_for_month(year: int, month: int) -> List[Dict]:
    """
    指定年月の日別KPIリストを返す。total_stake == 0 の日はスキップ。

    Returns
    -------
    [{"date": "20260405", "total_stake": int, "total_payout": int,
      "roi": float, "hit_count": int, "total_bets": int}, ...]
    昇順（日付順）。
    """
    prefix = f"{year:04d}{month:02d}"
    dates = sorted(d for d in get_available_dates() if d.startswith(prefix))
    rows: List[Dict] = []
    for d in dates:
        races = load_races_for_date(d)
        kpi = calc_kpi(races)
        if kpi["total_stake"] > 0:
            rows.append({"date": d, **kpi})
    return rows


def get_monthly_kpi_for_year(year: int) -> List[Dict]:
    """
    指定年の月別KPIリストを返す。total_stake == 0 の月はスキップ。

    Returns
    -------
    [{"month": "202604", "total_stake": int, "total_payout": int,
      "roi": float, "hit_count": int, "total_bets": int}, ...]
    昇順（月順）。
    """
    prefix = f"{year:04d}"
    all_dates = get_available_dates()  # 1回だけ読む
    months = sorted({d[:6] for d in all_dates if d.startswith(prefix)})
    rows: List[Dict] = []
    for m in months:
        dates = [d for d in all_dates if d.startswith(m)]
        all_races: List[Dict] = []
        for d in dates:
            all_races.extend(load_races_for_date(d))
        kpi = calc_kpi(all_races)
        if kpi["total_stake"] > 0:
            rows.append({"month": m, **kpi})
    return rows
