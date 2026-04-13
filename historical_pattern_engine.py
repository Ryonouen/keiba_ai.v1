from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple


TOKEN_SCORE_CAP = 0.45
FEATURE_SCORE_CAP = 0.40

TOKEN_WEIGHT_PREFIXES: Tuple[Tuple[str, float], ...] = (
    ("race_top3:", 1.00),
    ("race_any:", 0.75),
    ("distance_top3:", 0.65),
    ("distance_any:", 0.45),
    ("grade_top3:", 0.55),
    ("grade_any:", 0.35),
    ("month_top3:", 0.30),
    ("month_any:", 0.20),
    ("age:", 0.45),
    ("gate:", 0.40),
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
        if starts < 2:
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
