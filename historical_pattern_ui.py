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
