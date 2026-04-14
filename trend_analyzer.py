"""
trend_analyzer.py
過去10年傾向の構造化分析モジュール（Phase 1）

責務:
- 5要素（前走クラス / 前走着順 / 脚質 / 人気帯 / 枠）の傾向スコアを計算
- 各馬に trend_adjustment（乗数）と UI 表示用テキストを返す
- signal_judge が標準パスで補正済みの要素（脚質 / 枠 / 人気帯）は
  display-only とし、trend_adjustment には含めない（二重計上防止）

設計方針:
- trend_adjustment の有効範囲: 0.90 〜 1.12
- 1要素あたりの最大寄与: ±0.05
- サンプル信頼度で縮小（shrink_by_sample）
- 純粋関数のみ（副作用なし）
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from trend_stats import bucket_popularity_odds

# =========================================================
# 定数
# =========================================================

# trend_adjustment の上下限
TREND_ADJ_MAX: float = 1.12
TREND_ADJ_MIN: float = 0.90

# apply_trend_analyzer_bias の weight 上限
BIAS_WEIGHT_DEFAULT: float = 0.50   # probs への適用強度（full=1.0）

# =========================================================
# サンプル信頼度縮小
# =========================================================

def shrink_by_sample(raw_edge: float, n: int) -> float:
    """サンプル数に応じて edge を縮小する。"""
    if n >= 30:
        return raw_edge
    if n >= 15:
        return raw_edge * 0.70
    if n >= 8:
        return raw_edge * 0.40
    return raw_edge * 0.20


# =========================================================
# 要素別スコアリング
# =========================================================

def score_prev_class(
    feature: Dict[str, Any],
) -> Tuple[float, Optional[str], Optional[str]]:
    """
    前走クラス補正。
    Returns (edge, match_item_or_None, risk_item_or_None)
    edge = 0.0 のときは両方 None。
    """
    prev_class = float(feature.get("prev_race_class_index") or 0.0)
    prev_name  = str(feature.get("prev_race_name") or "")

    if prev_class <= 0.0:
        return 0.0, None, None

    if prev_class >= 0.95:
        raw = 0.040
        n_eff = 50
        label = prev_name if prev_name else "G1"
        return shrink_by_sample(raw, n_eff), f"前走G1組（{label}）", None

    if prev_class >= 0.75:
        raw = 0.015
        n_eff = 35
        return shrink_by_sample(raw, n_eff), "前走重賞組（G2/G3）", None

    if prev_class >= 0.60:
        return 0.0, None, None

    # 条件戦前走
    raw = -0.025
    n_eff = 35
    return shrink_by_sample(raw, n_eff), None, "前走条件戦（クラス差あり）"


def score_prev_rank(
    feature: Dict[str, Any],
) -> Tuple[float, Optional[str], Optional[str]]:
    """
    前走着順補正。G1距離差がある場合の大敗は減点しない。
    Returns (edge, match_item_or_None, risk_item_or_None)
    """
    prev_rank  = feature.get("prev_rank")
    prev_class = float(feature.get("prev_race_class_index") or 0.0)
    past_races = feature.get("past_races") or []
    pr0        = past_races[0] if past_races else {}
    prev_dist  = int(pr0.get("distance") or 0)
    curr_dist  = int(feature.get("target_distance") or 0)
    dist_diff  = abs(curr_dist - prev_dist) if curr_dist and prev_dist else 0

    if prev_rank is None:
        return 0.0, None, None

    try:
        r = int(prev_rank)
    except (ValueError, TypeError):
        return 0.0, None, None
    if r == 1:
        return shrink_by_sample(0.030, 40), "前走1着", None
    if r <= 3:
        return shrink_by_sample(0.015, 35), "前走3着以内", None
    if r <= 5:
        return 0.0, None, None

    # r >= 6
    if prev_class >= 0.95 and dist_diff >= 400:
        return shrink_by_sample(0.010, 30), "G1大敗も距離条件差あり", None

    return shrink_by_sample(-0.020, 35), None, "前走6着以下"


def score_running_style(
    feature: Dict[str, Any],
    race_trend_10y: Dict[str, Any],
) -> Tuple[float, Optional[str], Optional[str]]:
    """
    脚質傾向。edge = 0（display-only）。
    signal_judge の標準パスが同要素をすでに補正しているため、
    trend_adjustment には寄与しない。
    """
    if not race_trend_10y:
        return 0.0, None, None

    style_counts: Dict[str, int] = race_trend_10y.get("style") or {}
    if not style_counts:
        return 0.0, None, None

    running_style = str(feature.get("running_style") or "")
    style_label_map = {
        "front":   (["逃げ"], "逃げ"),
        "stalker": (["先行"], "先行"),
        "closer":  (["差し", "追込"], "差し"),
    }
    info = style_label_map.get(running_style)
    if not info:
        return 0.0, None, None

    keys, jp = info
    total = sum(style_counts.values()) or 1
    ratio = sum(style_counts.get(k, 0) for k in keys) / total

    if ratio >= 0.50:
        return 0.0, f"脚質({jp})が過去10年傾向と一致 ({ratio*100:.0f}%)", None
    if ratio <= 0.15:
        return 0.0, None, f"脚質({jp})は過去10年で不振傾向 ({ratio*100:.0f}%)"
    return 0.0, None, None


def score_popularity_band(
    feature: Dict[str, Any],
) -> Tuple[float, Optional[str], Optional[str]]:
    """
    人気帯表示。edge = 0（display-only）。
    """
    win_odds = feature.get("win_odds")
    if win_odds is None:
        return 0.0, None, None

    band = bucket_popularity_odds(float(win_odds))
    if band == "1番人気":
        return 0.0, "1番人気帯（市場の本命）", None
    if band == "10番人気以下":
        return 0.0, None, "10番人気以下（穴人気）"
    return 0.0, None, None


def score_gate(
    feature: Dict[str, Any],
    race_trend_10y: Dict[str, Any],
) -> Tuple[float, Optional[str], Optional[str]]:
    """
    枠傾向表示。edge = 0（display-only）。
    gate_counts のキーは trend_stats.bucket_gate が返す文字列形式を前提とする:
      "内枠(1〜3)" / "中枠(4〜6)" / "外枠(7〜)"
    """
    if not race_trend_10y:
        return 0.0, None, None

    gate_counts: Dict[str, int] = race_trend_10y.get("gate") or {}
    gate = feature.get("gate")
    if not gate_counts or gate is None:
        return 0.0, None, None

    g = int(gate)
    if g <= 3:
        bucket_key = "内枠(1〜3)"
        label = f"内枠({g}枠)"
    elif g <= 6:
        bucket_key = "中枠(4〜6)"
        label = f"中枠({g}枠)"
    else:
        bucket_key = "外枠(7〜)"
        label = f"外枠({g}枠)"

    total = sum(gate_counts.values()) or 1
    matching = gate_counts.get(bucket_key, 0)
    ratio = matching / total

    if ratio >= 0.45:
        return 0.0, f"{label}が過去10年傾向と一致 ({ratio*100:.0f}%)", None
    if ratio <= 0.15:
        return 0.0, None, f"{label}は過去10年で不振傾向 ({ratio*100:.0f}%)"
    return 0.0, None, None


# =========================================================
# Phase 2 要素（display-only）
# =========================================================

_TOP_JOCKEYS = frozenset({
    "川田将雅", "福永祐一", "武豊", "ルメール", "デムーロ",
    "戸崎圭太", "横山武史", "松山弘平", "岩田望来", "坂井瑠星",
})


def score_body_weight(
    feature: Dict[str, Any],
) -> Tuple[float, Optional[str], Optional[str]]:
    """馬体重変化の表示。edge = 0（display-only）。"""
    past_races = feature.get("past_races") or []
    if not past_races:
        return 0.0, None, None

    pr0 = past_races[0]
    change = pr0.get("body_weight_change")
    if change is None:
        return 0.0, None, None

    try:
        chg = int(change)
    except (ValueError, TypeError):
        return 0.0, None, None

    if chg > 12:
        return 0.0, None, f"馬体重大幅増加(+{chg}kg)"
    if chg < -10:
        return 0.0, None, f"馬体重大幅減少({chg}kg)"
    if 2 <= chg <= 8:
        return 0.0, f"馬体重微増({chg}kg)で状態良好", None
    return 0.0, None, None


def score_distance_change(
    feature: Dict[str, Any],
) -> Tuple[float, Optional[str], Optional[str]]:
    """距離延長/短縮の表示。edge = 0（display-only）。"""
    past_races = feature.get("past_races") or []
    curr_dist  = int(feature.get("target_distance") or 0)
    if not curr_dist or not past_races:
        return 0.0, None, None

    pr0       = past_races[0]
    prev_dist = int(pr0.get("distance") or 0)
    if not prev_dist:
        return 0.0, None, None

    diff = curr_dist - prev_dist
    if diff >= 400:
        return 0.0, None, f"大幅距離延長(+{diff}m)"
    if diff <= -400:
        return 0.0, None, f"大幅距離短縮({diff}m)"
    if diff >= 200:
        return 0.0, None, f"距離延長({diff}m)"
    if diff <= -200:
        return 0.0, None, f"距離短縮({diff}m)"
    return 0.0, f"前走と同距離圏", None


def score_sex(
    feature: Dict[str, Any],
) -> Tuple[float, Optional[str], Optional[str]]:
    """性別表示（牝馬の混合戦など）。edge = 0（display-only）。"""
    sex = str(feature.get("sex") or "").strip()
    if sex in ("牝", "F"):
        return 0.0, "牝馬（混合戦）", None
    return 0.0, None, None


def score_age(
    feature: Dict[str, Any],
) -> Tuple[float, Optional[str], Optional[str]]:
    """年齢傾向表示。edge = 0（display-only）。"""
    try:
        age = int(feature.get("age") or 0)
    except (ValueError, TypeError):
        return 0.0, None, None

    if age == 3:
        return 0.0, "3歳馬（成長力あり）", None
    if age >= 7:
        return 0.0, None, f"{age}歳（高齢）"
    return 0.0, None, None


def score_jockey(
    feature: Dict[str, Any],
) -> Tuple[float, Optional[str], Optional[str]]:
    """騎手表示。edge = 0（display-only）。"""
    jockey = str(feature.get("jockey") or "").strip()
    if not jockey:
        return 0.0, None, None
    if jockey in _TOP_JOCKEYS:
        return 0.0, f"主要騎手({jockey})", None
    return 0.0, None, None


# =========================================================
# メイン分析関数
# =========================================================

def analyze_horse_trend(
    feature: Dict[str, Any],
    race_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    1頭分の傾向分析を返す。

    race_context:
      condition_stats_available: bool  — signal_judge 標準パスが通ったか
      race_trend_10y: dict             — fetch_past_10y_results で取得した傾向集計

    Returns:
      trend_score:       float  — 生 edge 合計 × 100（表示用、-10〜+10）
      trend_confidence:  float  — 縮小係数の平均（0.0〜1.0）
      trend_match_items: list[str]
      trend_risk_items:  list[str]
      trend_adjustment:  float  — 乗数 0.90〜1.12（前走クラス+前走着順 のみ寄与）
      trend_summary:     str    — UI 表示用短文
    """
    ctx = race_context or {}
    race_trend_10y = ctx.get("race_trend_10y") or {}

    match_items: List[str] = []
    risk_items:  List[str] = []
    total_adj_edge = 0.0
    shrinkage_factors: List[float] = []

    # ── 前走クラス（trend_adjustment に寄与）──────────────────
    pc_edge, pc_match, pc_risk = score_prev_class(feature)
    if pc_match:
        match_items.append(pc_match)
    if pc_risk:
        risk_items.append(pc_risk)
    total_adj_edge += pc_edge
    # 信頼度: 実際に補正が効いていれば 1.0（n_eff >= 30 なら shrink なし）
    if pc_edge != 0.0:
        shrinkage_factors.append(1.0)

    # ── 前走着順（trend_adjustment に寄与）────────────────────
    pr_edge, pr_match, pr_risk = score_prev_rank(feature)
    if pr_match:
        match_items.append(pr_match)
    if pr_risk:
        risk_items.append(pr_risk)
    total_adj_edge += pr_edge
    if pr_edge != 0.0:
        shrinkage_factors.append(1.0)

    # ── 脚質（display-only）────────────────────────────────────
    _, st_match, st_risk = score_running_style(feature, race_trend_10y)
    if st_match:
        match_items.append(st_match)
    if st_risk:
        risk_items.append(st_risk)

    # ── 人気帯（display-only）──────────────────────────────────
    _, pop_match, pop_risk = score_popularity_band(feature)
    if pop_match:
        match_items.append(pop_match)
    if pop_risk:
        risk_items.append(pop_risk)

    # ── 枠（display-only）──────────────────────────────────────
    _, gate_match, gate_risk = score_gate(feature, race_trend_10y)
    if gate_match:
        match_items.append(gate_match)
    if gate_risk:
        risk_items.append(gate_risk)

    # ── 馬体重変化（display-only）──────────────────────────────────
    _, bw_match, bw_risk = score_body_weight(feature)
    if bw_match:
        match_items.append(bw_match)
    if bw_risk:
        risk_items.append(bw_risk)

    # ── 距離延長/短縮（display-only）───────────────────────────────
    _, dc_match, dc_risk = score_distance_change(feature)
    if dc_match:
        match_items.append(dc_match)
    if dc_risk:
        risk_items.append(dc_risk)

    # ── 性別（display-only）────────────────────────────────────────
    _, sx_match, sx_risk = score_sex(feature)
    if sx_match:
        match_items.append(sx_match)
    if sx_risk:
        risk_items.append(sx_risk)

    # ── 年齢（display-only）────────────────────────────────────────
    _, ag_match, ag_risk = score_age(feature)
    if ag_match:
        match_items.append(ag_match)
    if ag_risk:
        risk_items.append(ag_risk)

    # ── 騎手（display-only）────────────────────────────────────────
    _, jk_match, jk_risk = score_jockey(feature)
    if jk_match:
        match_items.append(jk_match)
    if jk_risk:
        risk_items.append(jk_risk)

    # ── 乗数計算 ──────────────────────────────────────────────
    adjustment = max(TREND_ADJ_MIN, min(TREND_ADJ_MAX, 1.0 + total_adj_edge))

    # ── 表示用スコア ───────────────────────────────────────────
    trend_score = round(total_adj_edge * 100, 1)
    confidence  = round(sum(shrinkage_factors) / len(shrinkage_factors), 3) if shrinkage_factors else 0.0

    # ── サマリー文生成 ─────────────────────────────────────────
    if total_adj_edge >= 0.03:
        trend_summary = "過去10年傾向との一致度が高く、本命信頼度を補強。"
    elif total_adj_edge >= 0.01:
        trend_summary = "傾向的な追い風あり。"
    elif total_adj_edge <= -0.03:
        trend_summary = "過去10年傾向に逆行する要素が複数あり、注意が必要。"
    elif total_adj_edge <= -0.01:
        trend_summary = "傾向的な懸念あり。"
    else:
        trend_summary = "傾向的には中立。"

    if match_items:
        trend_summary += " 好材料: " + "、".join(match_items[:2]) + "。"
    if risk_items:
        trend_summary += " 懸念: " + "、".join(risk_items[:2]) + "。"

    return {
        "trend_score":       trend_score,
        "trend_confidence":  confidence,
        "trend_match_items": match_items,
        "trend_risk_items":  risk_items,
        "trend_adjustment":  round(adjustment, 4),
        "trend_summary":     trend_summary,
    }


# =========================================================
# 確率リストへのバイアス適用
# =========================================================

def apply_trend_analyzer_bias(
    probs: List[float],
    features: List[Dict[str, Any]],
    weight: float = BIAS_WEIGHT_DEFAULT,
) -> List[float]:
    """
    trend_analyzer_result.trend_adjustment を各馬の win_prob に反映する。
    weight=1.0 でフル適用、weight=0.5 で半適用。
    再正規化して返す。
    """
    if not probs or len(probs) != len(features):
        return probs

    adjusted: List[float] = []
    for prob, feature in zip(probs, features):
        result = feature.get("trend_analyzer_result") or {}
        adj    = float(result.get("trend_adjustment") or 1.0)
        blended = 1.0 + (adj - 1.0) * weight
        blended = max(0.78, min(1.22, blended))
        adjusted.append(max(prob * blended, 1e-9))

    total = sum(adjusted)
    if total <= 0:
        return probs
    return [v / total for v in adjusted]
