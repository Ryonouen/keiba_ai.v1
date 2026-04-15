from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ============================================================
# Phase2 確定設定（2026-04-15 採用）
# 変更履歴:
#   Phase0: TOKEN_SCORE_CAP=0.45, race_any weight=0.50, min_starts=2
#   Phase1: race_any weight=0.00（小サンプルnoise無効化）、min_starts=5 に引き上げ
#   Phase2: TOKEN_SCORE_CAP=0.30（±0.40 飽和過多を緩和）
# 次の再評価ポイント: hist_applied レースが 30〜50 件蓄積後
# アルトラムス注意: Phase1 で min_starts 引き上げにより
#   race_top3:毎日杯・シンザン記念 が除去され -0.40 → +0.40 に急変。
#   急変監視フラグ付きで経過観察中（HIST_SUDDEN_CHANGE_THRESHOLD 参照）。
# ============================================================
TOKEN_SCORE_CAP = 0.30
FEATURE_SCORE_CAP = 0.40

# 監視用定数
# |diff| がこの閾値以上の馬を「急変馬」として [hist_audit] ログに出力する
HIST_SUDDEN_CHANGE_THRESHOLD = 0.50
# append_hist_distribution() のデフォルト出力先
HIST_DISTRIBUTION_LOG = "hist_distribution_log.jsonl"

TOKEN_WEIGHT_PREFIXES: Tuple[Tuple[str, float], ...] = (
    ("race_top3:",     0.70),
    ("race_any:",      0.00),  # disabled: low-sample noise; restore if needed
    ("distance_top3:", 0.60),
    ("distance_any:",  0.40),
    ("grade_top3:",    0.55),
    ("grade_any:",     0.35),
    ("month_top3:",    0.30),
    ("month_any:",     0.20),
    ("age:",           0.40),
    ("gate:",          0.35),
    ("body_weight:",   0.45),
    ("trial_group:",   0.55),
)


def normalize_race_name(name: Any) -> str:
    text = str(name or "").strip()
    if not text:
        return ""
    text = re.sub(r"[\(（][^）\)]*[\)）]", "", text)
    text = re.sub(r"\s+", "", text)
    text = text.replace("ステークス", "S")
    return text


def bucket_age(age: Any) -> str:
    try:
        value = int(age)
    except Exception:
        return "unknown"
    if value <= 2:
        return "2以下"
    if value == 3:
        return "3"
    if value == 4:
        return "4"
    if value == 5:
        return "5"
    return "6以上"


def bucket_gate(gate: Any) -> str:
    try:
        value = int(gate)
    except Exception:
        return "unknown"
    if value <= 3:
        return "inner"
    if value <= 6:
        return "middle"
    return "outer"


def bucket_distance(distance: Any) -> str:
    try:
        value = int(float(distance))
    except Exception:
        return "unknown"
    if value <= 1400:
        return "le_1400"
    if value == 1600:
        return "1600"
    if value == 1800:
        return "1800"
    return "ge_2000"


def bucket_body_weight(weight: Any) -> str:
    try:
        value = int(float(weight))
    except Exception:
        return "unknown"
    if value < 440:
        return "under_440"
    if value < 460:
        return "440_459"
    if value < 480:
        return "460_479"
    return "480_plus"


_GI_PREP_RACES = frozenset({
    "チューリップ賞", "フィリーズレビュー", "アネモネS", "クイーンC",
    "弥生賞", "スプリングS", "共同通信杯", "ホープフルS",
    "アーリントンC", "ニュージーランドT",
})


def classify_trial_group(race_names: List[str]) -> str:
    """Returns 'gi_prep' if any race name is a known GI prep trial, else 'direct'."""
    normalized = {normalize_race_name(n) for n in race_names}
    for name in normalized:
        if name in _GI_PREP_RACES:
            return "gi_prep"
    return "direct"


def extract_month(date_text: Any) -> str:
    text = str(date_text or "").strip()
    m = re.match(r"^\d{4}[/-](\d{1,2})[/-]\d{1,2}$", text)
    if not m:
        return "unknown"
    return str(int(m.group(1)))


def infer_grade(race_name: Any) -> str:
    text = str(race_name or "")
    if "(GI)" in text or "G1" in text:
        return "GI"
    if "(GII)" in text or "G2" in text:
        return "GII"
    if "(GIII)" in text or "G3" in text:
        return "GIII"
    if "(L)" in text or "Listed" in text:
        return "L"
    if "OP" in text or "オープン" in text:
        return "OP"
    return "OTHER"


def build_feature_pattern_tokens(feature: Dict[str, Any], max_races: int = 5) -> List[str]:
    tokens: List[str] = []
    seen = set()

    def add(token: str) -> None:
        if not token or token in seen:
            return
        seen.add(token)
        tokens.append(token)

    add(f"age:{bucket_age(feature.get('age'))}")
    add(f"gate:{bucket_gate(feature.get('gate'))}")

    for rec in list(feature.get("past_races") or [])[:max_races]:
        race_name = normalize_race_name(rec.get("race_name"))
        rank = rec.get("rank")
        distance_bucket = bucket_distance(rec.get("distance"))
        month = extract_month(rec.get("date"))
        grade = infer_grade(rec.get("race_name"))

        if race_name:
            add(f"race_any:{race_name}")
        add(f"distance_any:{distance_bucket}")
        add(f"month_any:{month}")
        add(f"grade_any:{grade}")

        try:
            rank_val = int(rank)
        except Exception:
            rank_val = None

        if rank_val is not None and rank_val <= 3:
            if race_name:
                add(f"race_top3:{race_name}")
            add(f"distance_top3:{distance_bucket}")
            add(f"month_top3:{month}")
            add(f"grade_top3:{grade}")

        body_weight = rec.get("body_weight")
        body_weight_change = rec.get("body_weight_change")
        bw_bucket = bucket_body_weight(body_weight)
        if bw_bucket != "unknown":
            add(f"body_weight:{bw_bucket}")
        if body_weight_change is not None:
            try:
                chg = int(body_weight_change)
                if chg > 2:
                    add("body_weight:gaining")
                elif chg < -2:
                    add("body_weight:losing")
            except Exception:
                pass

    all_race_names = [rec.get("race_name", "") for rec in list(feature.get("past_races") or [])[:max_races]]
    trial_group = classify_trial_group(all_race_names)
    add(f"trial_group:{trial_group}")

    return tokens


def _log_lift(value: float, baseline: float) -> float:
    eps = 1e-6
    return math.log((value + eps) / (baseline + eps))


def _token_weight(token: str) -> float:
    for prefix, weight in TOKEN_WEIGHT_PREFIXES:
        if token.startswith(prefix):
            return weight
    return 0.25


def build_historical_pattern_profile(samples: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    sample_list = list(samples)
    total = len(sample_list)
    if total == 0:
        return {"sample_size": 0, "token_stats": {}}

    baseline_win = sum(1 for s in sample_list if int(s.get("target_rank") or 99) == 1) / total
    baseline_top3 = sum(1 for s in sample_list if int(s.get("target_rank") or 99) <= 3) / total
    token_buckets: Dict[str, Dict[str, int]] = defaultdict(lambda: {"starts": 0, "wins": 0, "top3": 0})

    for sample in sample_list:
        tokens = build_feature_pattern_tokens(sample)
        target_rank = int(sample.get("target_rank") or 99)
        is_win = target_rank == 1
        is_top3 = target_rank <= 3
        for token in tokens:
            bucket = token_buckets[token]
            bucket["starts"] += 1
            bucket["wins"] += 1 if is_win else 0
            bucket["top3"] += 1 if is_top3 else 0

    token_stats: Dict[str, Dict[str, float]] = {}
    for token, row in token_buckets.items():
        starts = row["starts"]
        if starts < 5:
            continue
        win_rate = row["wins"] / starts
        top3_rate = row["top3"] / starts
        lift = 0.6 * _log_lift(win_rate, baseline_win) + 0.4 * _log_lift(top3_rate, baseline_top3)
        support = starts / (starts + 5)
        score = max(-TOKEN_SCORE_CAP, min(TOKEN_SCORE_CAP, lift * support))
        token_stats[token] = {
            "starts": starts,
            "wins": row["wins"],
            "top3": row["top3"],
            "win_rate": round(win_rate, 5),
            "top3_rate": round(top3_rate, 5),
            "score": round(score, 5),
        }

    return {
        "sample_size": total,
        "baseline_win_rate": round(baseline_win, 5),
        "baseline_top3_rate": round(baseline_top3, 5),
        "token_stats": token_stats,
    }


def score_feature_patterns(
    feature: Dict[str, Any],
    profile: Dict[str, Any],
    top_k: int = 6,
) -> Tuple[float, List[str]]:
    token_stats = (profile or {}).get("token_stats") or {}
    matched: List[Tuple[float, str, float]] = []

    for token in build_feature_pattern_tokens(feature):
        row = token_stats.get(token)
        if not row:
            continue
        score = float(row.get("score") or 0.0)
        if score == 0:
            continue
        weighted = score * _token_weight(token)
        matched.append((abs(weighted), token, score))

    matched.sort(reverse=True)
    chosen = matched[:top_k]

    total = sum(score * _token_weight(token) for _, token, score in chosen)
    total = max(-FEATURE_SCORE_CAP, min(FEATURE_SCORE_CAP, total))
    reasons = [f"{token}({score:+.3f})" for _, token, score in chosen]
    return round(total, 5), reasons


def apply_historical_pattern_bias(
    probs: List[float],
    features: List[Dict[str, Any]],
    weight: float = 0.20,
) -> List[float]:
    if not probs or len(probs) != len(features):
        return probs

    adjusted: List[float] = []
    for prob, feature in zip(probs, features):
        score = float(feature.get("historical_pattern_score") or 0.0)
        multiplier = 1.0 + score * weight
        multiplier = max(0.78, min(1.22, multiplier))
        adjusted.append(max(prob * multiplier, 1e-9))

    total = sum(adjusted)
    if total <= 0:
        return probs
    return [value / total for value in adjusted]


# ============================================================
# 監視ユーティリティ（Phase2 以降の追跡用・ロジック変更なし）
# ============================================================


def audit_hist_scores(
    race_id: str,
    features: List[Dict[str, Any]],
    profile: Dict[str, Any],
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    """
    各馬の hist_score・寄与トークン・飽和状態を辞書リストで返す。

    返却フィールド:
        race_id, horse_name, hist_score, raw_total,
        is_capped, cap_direction ("+"/"-"/"none"),
        top_tokens: [{token, score, weight, contribution, starts}]
    """
    token_stats = (profile or {}).get("token_stats") or {}
    result: List[Dict[str, Any]] = []

    for feature in features:
        name = str(feature.get("horse_name") or "")
        tokens = build_feature_pattern_tokens(feature)

        # (|weighted|, token, raw_score, token_weight, weighted, starts)
        matched: List[Tuple[float, str, float, float, float, int]] = []
        for tok in tokens:
            row = token_stats.get(tok)
            if not row:
                continue
            score = float(row.get("score") or 0.0)
            if score == 0:
                continue
            tw = _token_weight(tok)
            if tw == 0.0:
                continue
            weighted = score * tw
            matched.append((abs(weighted), tok, score, tw, weighted, int(row.get("starts") or 0)))

        matched.sort(reverse=True)
        top = matched[:top_k]

        raw_total = sum(x[4] for x in top)
        clamped = max(-FEATURE_SCORE_CAP, min(FEATURE_SCORE_CAP, raw_total))
        is_capped = abs(clamped - raw_total) > 0.001
        if raw_total > FEATURE_SCORE_CAP:
            cap_dir = "+"
        elif raw_total < -FEATURE_SCORE_CAP:
            cap_dir = "-"
        else:
            cap_dir = "none"

        result.append({
            "race_id": race_id,
            "horse_name": name,
            "hist_score": round(clamped, 5),
            "raw_total": round(raw_total, 5),
            "is_capped": is_capped,
            "cap_direction": cap_dir,
            "top_tokens": [
                {
                    "token": tok,
                    "score": round(s, 5),
                    "weight": tw,
                    "contribution": round(w, 5),
                    "starts": st,
                }
                for _, tok, s, tw, w, st in top
            ],
        })

    return result


def detect_hist_sudden_changes(
    before: Dict[str, float],
    after: Dict[str, float],
    threshold: float = HIST_SUDDEN_CHANGE_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    旧スコアと新スコアの差分が threshold 以上の馬を返す。

    before / after は {horse_name: hist_score} の辞書。
    アルトラムスのような急変ケース（min_starts 変更で -0.40 → +0.40）の
    自動検出に使う。結果は |diff| 降順でソートされる。
    """
    all_names = set(before) | set(after)
    changes: List[Dict[str, Any]] = []
    for name in sorted(all_names):
        s_b = before.get(name, 0.0)
        s_a = after.get(name, 0.0)
        diff = s_a - s_b
        if abs(diff) >= threshold:
            changes.append({
                "horse_name": name,
                "hist_before": round(s_b, 5),
                "hist_after": round(s_a, 5),
                "diff": round(diff, 5),
            })
    changes.sort(key=lambda x: abs(x["diff"]), reverse=True)
    return changes


def append_hist_distribution(
    race_id: str,
    race_name: str,
    race_date: str,
    features: List[Dict[str, Any]],
    profile: Dict[str, Any],
    log_path: Optional[str] = None,
) -> str:
    """
    hist_score 分布を JSONL 形式でファイルに追記する。

    hist_applied レースが 30〜50 件蓄積した時点での再監査用。
    各行の構造:
        {race_id, race_name, race_date, profile_sample_size,
         token_score_cap, feature_score_cap,
         horses: [audit_hist_scores() の出力]}

    呼び出し例:
        from historical_pattern_engine import append_hist_distribution
        append_hist_distribution(race_id, race_name, race_date, features, profile)
    """
    import json as _json
    from pathlib import Path as _Path

    path = _Path(log_path or HIST_DISTRIBUTION_LOG)
    record = {
        "race_id": race_id,
        "race_name": race_name,
        "race_date": race_date,
        "profile_sample_size": int((profile or {}).get("sample_size") or 0),
        "token_score_cap": TOKEN_SCORE_CAP,
        "feature_score_cap": FEATURE_SCORE_CAP,
        "horses": audit_hist_scores(race_id, features, profile),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(_json.dumps(record, ensure_ascii=False) + "\n")
    return str(path)
