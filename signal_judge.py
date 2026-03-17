"""
signal_judge.py
傾向シグナル判定・補正値計算・馬別詳細生成

責務:
- sample_size + diff_top3 から 7段階シグナルを判定
- シグナル強度 → model_score 補正値に変換
- 各馬の全条件シグナル詳細リストを生成
- 詳細リストを集約して最終補正値・サマリーを返す
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# =========================================================
# シグナル判定閾値
# =========================================================

# (min_sample_size, min_abs_diff_top3)
SIGNAL_THRESHOLDS: Dict[str, Tuple[int, float]] = {
    "strong": (10, 0.15),
    "medium": ( 8, 0.10),
    "weak":   ( 5, 0.05),
}

# =========================================================
# 補正値レンジ
# =========================================================

# (extreme_end, moderate_end)
# negatives: extreme_end < 0 < moderate_end の絶対値が小さい方
# positives: moderate_end が小さく extreme_end が大きい
CORRECTION_RANGES: Dict[str, Tuple[float, float]] = {
    "strong_negative": (-0.10, -0.06),   # 強い懸念: -0.06 〜 -0.10
    "medium_negative": (-0.05, -0.03),   # 懸念:    -0.03 〜 -0.05
    "weak_negative":   (-0.02, -0.01),   # 弱い懸念: -0.01 〜 -0.02
    "neutral":         ( 0.00,  0.00),
    "weak_positive":   ( 0.01,  0.02),   # 弱い追い風: +0.01 〜 +0.02
    "medium_positive": ( 0.02,  0.04),   # 追い風:    +0.02 〜 +0.04
    "strong_positive": ( 0.04,  0.07),   # 強い追い風: +0.04 〜 +0.07
}

# diff_top3 の絶対値がこの値に達したら補正レンジ上限 (t=1.0)
DIFF_SCALE_MAX: Dict[str, float] = {
    "strong": 0.30,
    "medium": 0.20,
    "weak":   0.12,
}

# サンプルサイズがこの値以上で信頼度 1.0
CONFIDENCE_FULL_SAMPLE: int = 20

# 最終合計補正値の上下限（単一馬への過剰補正を防ぐ）
MIN_TOTAL_ADJUST: float = -0.15
MAX_TOTAL_ADJUST: float = +0.10

# 負シグナル積み上げ: weak_negative がこの数以上で昇格
WEAK_NEGATIVE_ESCALATION: Dict[int, str] = {
    2: "medium_negative",
    3: "strong_negative",
}

# シグナル日本語ラベル
SIGNAL_JP: Dict[str, str] = {
    "strong_positive": "強い追い風",
    "medium_positive": "追い風",
    "weak_positive":   "弱い追い風",
    "neutral":         "中立",
    "weak_negative":   "弱い懸念",
    "medium_negative": "懸念",
    "strong_negative": "強い懸念",
}

# 条件名 → 日本語表示
COND_JP: Dict[str, str] = {
    "年齢":  "年齢",
    "枠":    "枠順",
    "脚質":  "脚質",
    "人気帯": "人気帯",
}


# =========================================================
# シグナル判定
# =========================================================

def judge_signal(sample_size: int, diff_top3: float) -> str:
    """
    sample_size と diff_top3 から 7段階シグナル文字列を返す。

    Returns one of:
      strong_negative, medium_negative, weak_negative,
      neutral,
      weak_positive, medium_positive, strong_positive
    """
    abs_diff = abs(diff_top3)
    sign     = "positive" if diff_top3 >= 0 else "negative"

    for level in ("strong", "medium", "weak"):
        min_n, min_d = SIGNAL_THRESHOLDS[level]
        if sample_size >= min_n and abs_diff >= min_d:
            return f"{level}_{sign}"

    return "neutral"


# =========================================================
# 補正値計算
# =========================================================

def calc_correction(
    signal: str,
    diff_top3: float,
    sample_size: int,
) -> float:
    """
    シグナル強度・diff_top3 の大きさ・サンプルサイズから補正値を計算する。

    設計:
    - diff_top3 の大きさに応じてレンジ内を補間（大きいほど極端な側へ）
    - sample_size < CONFIDENCE_FULL_SAMPLE なら補正を圧縮
    - 不利補正はやや強め（負の方向はそのまま）
    - 有利補正はやや控えめ（正の方向は 0.95 倍）
    """
    if signal == "neutral":
        return 0.0

    low, high = CORRECTION_RANGES[signal]
    if low == high == 0.0:
        return 0.0

    level     = signal.split("_")[0]            # "strong", "medium", "weak"
    scale_max = DIFF_SCALE_MAX.get(level, 0.20)
    t         = min(1.0, abs(diff_top3) / scale_max)

    # diff_top3 が大きいほど極端な側へ補間
    if diff_top3 >= 0:
        # positive: low=moderate, high=extreme
        raw = low + (high - low) * t
        raw *= 0.95     # 有利補正はやや控えめ
    else:
        # negative: high=moderate, low=extreme
        raw = high + (low - high) * t

    # サンプルサイズ信頼度によって圧縮
    confidence = min(1.0, sample_size / CONFIDENCE_FULL_SAMPLE)
    return round(raw * confidence, 5)


# =========================================================
# 馬ごとのシグナル詳細生成
# =========================================================

def _get_horse_cond_keys(feature: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """feature dict から各条件のバケットキーを返す。"""
    from trend_stats import (
        bucket_age, bucket_gate, bucket_style,
        bucket_popularity_rank, bucket_popularity_odds,
    )
    pop_key = (
        bucket_popularity_rank(feature.get("popularity"))
        or bucket_popularity_odds(feature.get("win_odds"))
    )
    return {
        "年齢":  bucket_age(feature.get("age")),
        "枠":    bucket_gate(feature.get("gate")),
        "脚質":  bucket_style(feature.get("running_style")),
        "人気帯": pop_key,
    }


def build_horse_signal_details(
    feature: Dict[str, Any],
    condition_stats: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    1頭分の全条件シグナル詳細リストを生成する（neutral は除外）。

    Parameters
    ----------
    feature         : race_ai_engine の features 要素
    condition_stats : build_condition_stats() の戻り値

    Returns
    -------
    [
        {
            "factor":            str,    # 条件名（日本語）
            "value":             str,    # バケットキー
            "sample_size":       int,
            "top3_rate":         float,
            "overall_top3_rate": float,
            "diff_top3":         float,
            "signal":            str,    # e.g. "medium_negative"
            "signal_jp":         str,    # 日本語ラベル
            "score_adjust":      float,  # model_score 補正値
            "reason":            str,    # 自然言語説明
        },
        ...
    ]
    """
    if not condition_stats:
        return []

    overall      = condition_stats.get("_overall", {})
    overall_top3 = overall.get("overall_top3_rate", 0.0)
    cond_keys    = _get_horse_cond_keys(feature)
    details: List[Dict[str, Any]] = []

    for cond_name, bucket_key in cond_keys.items():
        if not bucket_key:
            continue
        bucket_data = condition_stats.get(cond_name, {}).get(bucket_key)
        if not bucket_data:
            continue

        sample = bucket_data["sample_size"]
        t3r    = bucket_data["top3_rate"]
        diff   = bucket_data["diff_top3"]
        signal = judge_signal(sample, diff)
        if signal == "neutral":
            continue

        adj    = calc_correction(signal, diff, sample)
        reason = _build_reason(cond_name, bucket_key, t3r, overall_top3, signal)

        details.append({
            "factor":            COND_JP.get(cond_name, cond_name),
            "value":             bucket_key,
            "sample_size":       sample,
            "top3_rate":         t3r,
            "overall_top3_rate": overall_top3,
            "diff_top3":         diff,
            "signal":            signal,
            "signal_jp":         SIGNAL_JP.get(signal, ""),
            "score_adjust":      adj,
            "reason":            reason,
        })

    return details


def _build_reason(
    cond_name: str,
    bucket_key: str,
    top3_rate: float,
    overall_top3_rate: float,
    signal: str,
) -> str:
    direction = "高い" if top3_rate >= overall_top3_rate else "低い"
    pct_t3  = f"{top3_rate * 100:.1f}%"
    pct_all = f"{overall_top3_rate * 100:.1f}%"
    label   = COND_JP.get(cond_name, cond_name)
    return (
        f"このレースでは{label}「{bucket_key}」の好走率が{pct_t3}"
        f"（全体{pct_all}比 {direction}）"
    )


# =========================================================
# 負シグナル積み上げ（エスカレーション）
# =========================================================

def _apply_escalation(details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    weak_negative が 2個以上ある場合に 1個の強いシグナルへ昇格する。
    - 2個 → medium_negative
    - 3個以上 → strong_negative
    昇格後は元の weak_negative を全て削除し、代表1個に置き換える。
    """
    weak_neg = [d for d in details if d["signal"] == "weak_negative"]
    n = len(weak_neg)
    if n < 2:
        return details

    escalated_str = WEAK_NEGATIVE_ESCALATION.get(min(n, 3), "medium_negative")

    # diff_top3 が最も低い（最も悪い）要素を代表に使う
    rep        = min(weak_neg, key=lambda d: d["diff_top3"])
    avg_sample = int(sum(d["sample_size"] for d in weak_neg) / n)
    avg_diff   = sum(d["diff_top3"] for d in weak_neg) / n
    new_adj    = calc_correction(escalated_str, avg_diff, avg_sample)

    merged = {
        **rep,
        "signal":       escalated_str,
        "signal_jp":    SIGNAL_JP.get(escalated_str, ""),
        "score_adjust": new_adj,
        "reason":       rep["reason"] + f"（弱い懸念 {n}件が昇格）",
        "escalated":    True,
    }

    result = [d for d in details if d["signal"] != "weak_negative"]
    result.append(merged)
    return result


# =========================================================
# 詳細集約 → 最終補正値
# =========================================================

def aggregate_signal_result(
    details: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    build_horse_signal_details の結果を集約して最終補正値・サマリーを返す。

    Returns
    -------
    {
        "total_trend_adjust":    float,      # model_score 合計補正（上下限クランプ済み）
        "positive_count":        int,
        "negative_count":        int,
        "strong_negative_count": int,
        "strong_concerns":       List[Dict],  # medium/strong_negative
        "weak_concerns":         List[Dict],  # weak_negative
        "strong_tailwinds":      List[Dict],  # medium/strong_positive
        "weak_tailwinds":        List[Dict],  # weak_positive
        "details":               List[Dict],  # 全シグナル（escalation 適用済み）
        "summary_jp":            str,
    }
    """
    escalated = _apply_escalation(details)

    total = sum(d["score_adjust"] for d in escalated)
    total = max(MIN_TOTAL_ADJUST, min(MAX_TOTAL_ADJUST, total))

    pos_count   = sum(1 for d in escalated if "_positive" in d["signal"])
    neg_count   = sum(1 for d in escalated if "_negative" in d["signal"])
    strong_neg  = sum(1 for d in escalated if d["signal"] in ("strong_negative", "medium_negative"))

    strong_concerns  = [d for d in escalated if d["signal"] in ("strong_negative", "medium_negative")]
    weak_concerns    = [d for d in escalated if d["signal"] == "weak_negative"]
    strong_tailwinds = [d for d in escalated if d["signal"] in ("strong_positive", "medium_positive")]
    weak_tailwinds   = [d for d in escalated if d["signal"] == "weak_positive"]

    summary = _build_summary(
        strong_concerns, weak_concerns,
        strong_tailwinds, weak_tailwinds,
        total,
    )

    return {
        "total_trend_adjust":    round(total, 5),
        "positive_count":        pos_count,
        "negative_count":        neg_count,
        "strong_negative_count": strong_neg,
        "strong_concerns":       strong_concerns,
        "weak_concerns":         weak_concerns,
        "strong_tailwinds":      strong_tailwinds,
        "weak_tailwinds":        weak_tailwinds,
        "details":               escalated,
        "summary_jp":            summary,
    }


def _build_summary(
    strong_concerns:  List[Dict],
    weak_concerns:    List[Dict],
    strong_tailwinds: List[Dict],
    weak_tailwinds:   List[Dict],
    total: float,
) -> str:
    parts = []
    if strong_tailwinds:
        labels = [f"{d['factor']}({d['value']})" for d in strong_tailwinds]
        parts.append("強い追い風: " + " / ".join(labels))
    if weak_tailwinds:
        labels = [f"{d['factor']}({d['value']})" for d in weak_tailwinds]
        parts.append("弱い追い風: " + " / ".join(labels))
    if strong_concerns:
        labels = [f"{d['factor']}({d['value']})" for d in strong_concerns]
        parts.append("懸念: " + " / ".join(labels))
    if weak_concerns:
        labels = [f"{d['factor']}({d['value']})" for d in weak_concerns]
        parts.append("弱い懸念: " + " / ".join(labels))
    sign = "+" if total >= 0 else ""
    parts.append(f"補正合計: {sign}{total:.3f}")
    return " | ".join(parts) if parts else "傾向シグナルなし"
