from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


_ROUTE_REASON_RE = re.compile(r"^([^=]+)=(.+)\(([+-]\d+(?:\.\d+)?)\)$")
_GRADE_LABEL_REPLACEMENTS = (
    (re.compile(r"\bGIII(?=で|の|組|$)"), "G3"),
    (re.compile(r"\bGII(?=で|の|組|$)"), "G2"),
    (re.compile(r"\bGI(?=で|の|組|$)"), "G1"),
    (re.compile(r"\bOP(?=で|の|組|$)"), "オープン級"),
    (re.compile(r"\bL(?=で|の|組|$)"), "リステッド"),
)
_DISTANCE_TOP3_POSITIVE_RE = re.compile(r"(\d{3,4}m(?:以下|以上)?)で3着以内の実績は好材料")
_DISTANCE_TOP3_NEGATIVE_RE = re.compile(r"(\d{3,4}m(?:以下|以上)?)で3着以内の履歴は近年傾向ではやや割引")
_GRADE_TOP3_POSITIVE_RE = re.compile(r"\b(G[123])で3着以内の実績は好材料")
_GRADE_TOP3_NEGATIVE_RE = re.compile(r"\b(G[123])で3着以内の履歴は近年傾向ではやや割引")
_RAW_TOKEN_RE = re.compile(
    r"(?:race|distance|grade|month|gate|body_weight|trial_group)_[a-z0-9_]*:|prev_[a-z_]+=",
    re.IGNORECASE,
)
_MECHANICAL_PHRASE_RE = re.compile(r"(?:\d{3,4}m(?:以下|以上)?|G[123])で3着以内")


def _polish_historical_pattern_text(text: str) -> str:
    reason = str(text or "").strip()
    for race_label in ("2歳新馬", "2歳未勝利"):
        if reason.startswith(f"{race_label}で3着以内"):
            reason = reason.replace(
                "実績は好材料",
                "履歴は補助的な好材料（参考度はやや控えめ）",
            )
            reason = reason.replace(
                "履歴は好材料",
                "履歴は補助的な好材料（参考度はやや控えめ）",
            )
            reason = reason.replace(
                "履歴は近年傾向ではやや割引",
                "履歴は近年傾向ではやや割引（参考度はやや控えめ）",
            )

    for pattern, label in _GRADE_LABEL_REPLACEMENTS:
        reason = pattern.sub(label, reason)
    reason = _DISTANCE_TOP3_POSITIVE_RE.sub(r"\1での好走実績は距離面の好材料", reason)
    reason = _DISTANCE_TOP3_NEGATIVE_RE.sub(r"\1での好走履歴は近年傾向ではやや割引", reason)
    reason = _GRADE_TOP3_POSITIVE_RE.sub(r"\1級での好走実績は近年傾向で好材料", reason)
    reason = _GRADE_TOP3_NEGATIVE_RE.sub(r"\1級での好走履歴は近年傾向ではやや割引", reason)
    return reason


def _clean_display_reason_prefix(text: str) -> Tuple[str, str]:
    reason = str(text or "").strip()
    if reason.startswith("プラス要因:"):
        return "positive", _polish_historical_pattern_text(reason.split(":", 1)[1])
    if reason.startswith("マイナス要因:"):
        return "negative", _polish_historical_pattern_text(reason.split(":", 1)[1])
    if "(+" in reason:
        return "positive", _polish_historical_pattern_text(reason)
    if "(-" in reason:
        return "negative", _polish_historical_pattern_text(reason)
    return "positive", _polish_historical_pattern_text(reason)


def _group_item_text(item: Any) -> str:
    if isinstance(item, dict):
        text = item.get("text") or item.get("display_text") or item.get("reason") or ""
        return _polish_historical_pattern_text(str(text))
    return _polish_historical_pattern_text(str(item or ""))


def get_historical_pattern_ui_reason_groups(feature: Dict[str, Any]) -> Dict[str, List[str]]:
    """UI表示用の historical pattern 理由を取得する。

    優先順位:
    1. structured reason_groups
    2. historical_pattern_display_reasons
    3. legacy historical_pattern_reasons
    """
    groups: Dict[str, List[str]] = {"positive": [], "negative": []}
    structured = (feature or {}).get("historical_pattern_reason_groups") or {}
    if isinstance(structured, dict):
        for key in ("positive", "negative"):
            for item in structured.get(key) or []:
                text = _group_item_text(item)
                if text:
                    groups[key].append(text)
        if groups["positive"] or groups["negative"]:
            return groups

    display_reasons = (feature or {}).get("historical_pattern_display_reasons") or []
    for reason in display_reasons:
        key, text = _clean_display_reason_prefix(str(reason or ""))
        if text:
            groups[key].append(text)
    if groups["positive"] or groups["negative"]:
        return groups

    legacy_reasons = (feature or {}).get("historical_pattern_reasons") or []
    for reason in legacy_reasons:
        key, text = _clean_display_reason_prefix(str(reason or ""))
        if text:
            groups[key].append(text)
    return groups


def has_historical_pattern_ui_reasons(feature: Dict[str, Any]) -> bool:
    groups = get_historical_pattern_ui_reason_groups(feature or {})
    return bool(groups["positive"] or groups["negative"])


def _route_distance_label(bucket: str) -> str:
    return {
        "le_1400": "1400m以下",
        "1600": "1600m",
        "1800": "1800m",
        "ge_2000": "2000m以上",
    }.get(str(bucket), str(bucket))


def _route_rank_label(bucket: str) -> str:
    return {
        "1": "前走1着",
        "2_3": "前走2〜3着",
        "4_5": "前走4〜5着",
        "6_plus": "前走6着以下",
    }.get(str(bucket), str(bucket))


def _route_reason_text(raw_reason: str) -> str:
    raw = str(raw_reason or "").strip()
    match = _ROUTE_REASON_RE.match(raw)
    if not match:
        return raw

    key, bucket, score_text = match.groups()
    if bucket == "unknown":
        return ""

    try:
        score = float(score_text)
    except Exception:
        score = 0.0
    positive = score > 0
    suffix = "ローテ傾向で好材料" if positive else "ローテ傾向ではやや割引"

    if key == "prev_race_name":
        return f"前走{bucket}組は{suffix}"
    if key == "prev_distance_bucket":
        return f"前走{_route_distance_label(bucket)}組は{suffix}"
    if key == "prev_month":
        return f"{bucket}月からの臨戦は{suffix}"
    if key == "prev_rank_bucket":
        return f"{_route_rank_label(bucket)}は{suffix}"
    return raw


def get_route_profile_display_reasons(feature: Dict[str, Any], limit: int = 2) -> List[str]:
    display: List[str] = []
    for raw_reason in (feature or {}).get("route_profile_reasons") or []:
        text = _route_reason_text(str(raw_reason or ""))
        if not text:
            continue
        display.append(text)
        if len(display) >= limit:
            break
    return display


def _collect_ui_reason_texts(feature: Dict[str, Any]) -> List[Tuple[str, str]]:
    texts: List[Tuple[str, str]] = []
    for reason in get_route_profile_display_reasons(feature or {}):
        texts.append(("route", reason))
    hist_groups = get_historical_pattern_ui_reason_groups(feature or {})
    for key in ("positive", "negative"):
        for reason in hist_groups.get(key) or []:
            texts.append((f"historical_{key}", reason))
    return texts


def _has_strong_young_reason(text: str) -> bool:
    if "2歳新馬" not in text and "2歳未勝利" not in text:
        return False
    return "補助的" not in text and "参考度" not in text


def audit_historical_pattern_ui_reasons(
    features: List[Dict[str, Any]],
    example_limit: int = 5,
) -> Dict[str, Any]:
    """実レース表示理由の軽量監査用サマリーを返す。"""
    issue_keys = ("raw_token", "raw_grade", "strong_young_reason", "mechanical_phrase")
    issue_counts = {key: 0 for key in issue_keys}
    examples: Dict[str, List[Dict[str, str]]] = {key: [] for key in issue_keys}

    def add_issue(issue: str, horse_name: str, text: str, source: str) -> None:
        issue_counts[issue] += 1
        if len(examples[issue]) < example_limit:
            examples[issue].append({
                "horse_name": horse_name,
                "source": source,
                "text": text,
            })

    for feature in features or []:
        horse_name = str((feature or {}).get("horse_name") or "-")
        for source, text in _collect_ui_reason_texts(feature or {}):
            if _RAW_TOKEN_RE.search(text):
                add_issue("raw_token", horse_name, text, source)
            if any(marker in text for marker in ("Lで", "OPで", "GIで", "GIIで", "GIIIで")):
                add_issue("raw_grade", horse_name, text, source)
            if _has_strong_young_reason(text):
                add_issue("strong_young_reason", horse_name, text, source)
            if _MECHANICAL_PHRASE_RE.search(text):
                add_issue("mechanical_phrase", horse_name, text, source)

    return {
        "checked_horses": len(features or []),
        "issue_counts": issue_counts,
        "examples": examples,
    }
