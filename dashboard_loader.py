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
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_PREDICTIONS_FILE     = os.path.join(_HERE, "pipeline_predictions.json")
_BET_SUGGESTIONS_FILE = os.path.join(_HERE, "pipeline_bet_suggestions.json")
_BET_OUTCOMES_FILE    = os.path.join(_HERE, "pipeline_bet_outcomes.json")
_RACE_RESULTS_FILE    = os.path.join(_HERE, "pipeline_race_results.json")

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
    v2: horse["feature_dict"] に win_odds / running_style 等が入る。
    v1: horse["win_odds"] = null
    """
    fd = horse.get("feature_dict")
    is_v2 = fd is not None
    win_odds_raw      = fd.get("win_odds")        if is_v2 else horse.get("win_odds")
    running_style_raw = fd.get("running_style")   if is_v2 else horse.get("running_style")
    popularity_raw    = fd.get("feat_popularity") if is_v2 else horse.get("popularity")
    place_prob_raw    = fd.get("place_prob")       if is_v2 else None
    win_ev_raw        = fd.get("win_ev")           if is_v2 else None

    win_prob   = horse.get("ai_win_prob")
    place_prob = float(place_prob_raw) if place_prob_raw is not None else None

    # 2着内率: 勝率と複勝率の間を 2:1 で補間（近似値）
    top2_prob: Optional[float] = None
    if win_prob is not None and place_prob is not None:
        top2_prob = win_prob + (place_prob - win_prob) * 2 / 3

    return {
        "horse_name":    horse.get("horse_name", ""),
        "ai_win_prob":   win_prob,
        "place_prob":    place_prob,    # ≈ 3着内率
        "top2_prob":     top2_prob,     # 2着内率 (近似)
        "win_ev":        float(win_ev_raw) if win_ev_raw is not None else None,
        "win_odds":      win_odds_raw,
        "popularity":    int(popularity_raw) if popularity_raw is not None else None,
        "running_style": STYLE_MAP.get(running_style_raw or "", None),
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

        race_result = rr_all.get(str(race_id), {})
        _enrich_horses_from_result_dict(race_result, horses)

        upset = calc_upset_score(horses)
        hot   = calc_hot_bets(bets)

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
                "status":         _race_status(start_dt, outcomes),
                "horses":         horses,
                "bets":           bets,
                "outcomes":       outcomes,
                "upset_score":    upset["score"],
                "upset_label":    upset["label"],
                "upset_color":    upset["color"],
                "hot_bets":       hot,
                "distance":       race_result.get("distance"),
                "surface":        race_result.get("surface"),
                "n_runners":      len(race_result.get("runners") or []) or None,
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
