"""
value_ai.py
馬券期待値AI

責務:
- AI勝率 vs 市場期待勝率の比較テーブル構築
- 妙味馬（value_gap > 0）の自動抽出
- 危険人気馬（市場過大評価馬）の自動抽出
- 過去10年傾向スコア補正
- レース構造分類（展開ベース）
- 最適買い目を1案に絞る
- 印（◎○▲☆△×）の割り当て
"""
from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

from trend_stats import bucket_popularity_odds

# =========================================================
# 閾値・重みの定数（後から調整しやすくする）
# =========================================================

VALUE_GAP_STRONG: float = 0.06      # 強妙味の閾値（AI勝率 - 市場勝率）
VALUE_GAP_MIN: float = 0.025        # 妙味の最小閾値
DANGER_ODDS_MAX: float = 7.0        # 危険人気馬の対象オッズ上限
DANGER_GAP_MIN: float = 0.03        # 危険判定の市場-AI乖離閾値
DOMINANT_PROB_GAP: float = 0.12     # 本命信頼型の勝率差
CONTESTED_PROB_GAP: float = 0.05    # 波乱型の勝率差
MAX_BET_TICKETS: int = 8            # 買い目最大点数
MAX_TRIFECTA_TICKETS: int = 2       # 三連単最大点数
EV_SKIP_THRESHOLD: float = 1.02     # この期待値未満は見送り候補
BANKROLL_RATIO_BASE: float = 0.05   # 軍資金基本配分率
BANKROLL_UNIT: int = 100            # 掛け金単位（100円）

# 過去傾向補正の重み（model_scoreへの加減算の最大値）
TREND_WEIGHTS: Dict[str, float] = {
    "style":      0.025,   # 脚質傾向補正
    "popularity": 0.015,   # 人気帯傾向補正
    "gate":       0.010,   # 枠傾向補正
    "age":        0.012,   # 年齢傾向補正
}

# 券種別期待値エンジン定数
JRA_RETURN_RATES: Dict[str, float] = {
    "単勝":   0.800,
    "複勝":   0.800,
    "馬連":   0.775,
    "ワイド": 0.775,
    "3連複":  0.675,
}
MARKET_PLACE_MULT: float = 2.8   # 市場複勝オッズ推定: 単勝オッズ / MARKET_PLACE_MULT
WIDE_CORR: float    = 0.65       # ワイドヒット確率補正（相関考慮）
TRIO_CORR: float    = 0.50       # 3連複ヒット確率補正（相関考慮）
UMAREN_COEFF: float = 2.0        # 馬連ヒット確率補正（Harville近似）
EV_COMPOUND_SKIP: float = 0.95   # 複合馬券の見送り閾値（単勝より緩め）

# 人気帯別複勝オッズ推定係数（推定複勝オッズ = 単勝オッズ / factor）
# JRA実態: 人気馬ほど複勝が集中するため単勝との比率が縮まる
# 参考: 1番人気 単勝2.5→複勝1.6 / 4-6人気 単勝10→複勝4.0 / 10以下 単勝40→複勝10
PLACE_ODDS_FACTORS: Dict[str, float] = {
    "1番人気":      1.55,   # 単勝3倍以下   → 複勝 ÷1.55 ≒ 1.6–1.9倍
    "2〜3番人気":   1.90,   # 単勝3〜7倍    → 複勝 ÷1.90 ≒ 2.1–3.2倍
    "4〜6番人気":   2.50,   # 単勝7〜15倍   → 複勝 ÷2.50 ≒ 3.0–5.0倍
    "7〜9番人気":   3.10,   # 単勝15〜30倍  → 複勝 ÷3.10 ≒ 5.0–9.5倍
    "10番人気以下": 4.00,   # 単勝30倍超    → 複勝 ÷4.00 ≒ 8.0–倍
}

# 推奨プラン安定化定数
MIN_STAKE_PER_TICKET: int  = 100   # 1点最低掛け金（円）—— これを割る案は除外
MAX_PRACTICAL_TICKETS: int = 5     # 点数上限（EVよりも少点数・安定性を優先）
EV_TIE_MARGIN: float       = 0.10  # EV差がこれ以下なら少点数・低リスク券種を優先

# ── フェーズ3: 着順分布推定定数 ──────────────────────────────────
# p_top2 = p1 + TOP2_FRAC × (p_top3 - p1)
TOP2_FRAC_FRONT:   float = 0.68   # 逃げ・先行: ポジション保持しやすい → p_top2 寄り
TOP2_FRAC_CLOSER:  float = 0.58   # 差し馬: 2着以内はブレやすい
TOP2_FRAC_DEFAULT: float = 0.63   # 不明・その他

# 人気帯別ワイドヒット補正係数（両人気は割り引く、中穴混じりは優遇）
WIDE_CORR_BOTH_POP:  float = 0.58   # 両馬オッズ ≤7: 人気同士は売れすぎ
WIDE_CORR_MIX:       float = 0.65   # 片方オッズ >7: 標準
WIDE_CORR_BOTH_LONG: float = 0.70   # 両馬オッズ >12: 中穴混じり優遇

# 組み合わせ人気構成補正
POPULAR_COMBO_PENALTY: float = 0.94   # 全馬オッズ ≤7: 割り引く
MIXED_ODDS_BONUS:      float = 1.04   # 中穴馬（7<odds≤20）含む: 優遇

# ── フェーズ3.5: 当たるAI強化 ──────────────────────────────────────
WIN_PROB_HEAD_MIN: float   = 0.15    # 頭候補: AI勝率の最低値
AXIS_SCORE_HEAD_MIN: float = 0.50    # 頭候補: axis_score の最低値
AXIS_SCORE_AXIS_MIN: float = 0.38    # 軸候補: axis_score の最低値
STABLE_SCORE_AXIS_MIN: float = 0.48  # 軸候補: stable_score の最低値
TOP2_AXIS_MIN: float       = 0.22    # 軸候補: top2_prob の最低値
TOP3_HIMO_MIN: float       = 0.28    # ヒモ候補: top3_prob の最低値
TOP3_RESCUE_MIN: float     = 0.30    # 取りこぼし注意馬: top3_prob の最低値
UPSIDE_SCORE_HIGH: float   = 0.55    # 上振れ余地が高い閾値
RESCUE_AI_RANK_MAX: int    = 3       # 取りこぼし注意馬: AI上位N頭を除外
DANGER_V3_AXIS_MAX: float  = 0.35    # 「真に危険」判定: axis_score 上限
DANGER_V3_TOP3_MAX: float  = 0.25    # 「真に危険」判定: top3_prob 上限


# =========================================================
# EV テーブル構築
# =========================================================

def build_ev_table(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    各馬の AI勝率・市場期待勝率・value_gap を含む行リストを返す。

    市場期待勝率 = 1 / 単勝オッズ（簡易版。JRA控除25%の簡易補正は任意）
    value_gap > 0 → 妙味候補
    value_gap < 0 かつ人気上位 → 危険人気馬候補

    必須列:
        horse_name, ai_score, ai_win_prob, win_odds,
        market_win_prob, value_gap, popularity_rank
    """
    if not features:
        return []

    # 人気順位: 単勝オッズ昇順で付番
    sortable = [(i, float(f.get("win_odds") or 9999)) for i, f in enumerate(features)]
    sortable.sort(key=lambda x: x[1])
    popularity_map: Dict[int, int] = {idx: rank + 1 for rank, (idx, _) in enumerate(sortable)}

    rows = []
    for i, f in enumerate(features):
        win_odds = f.get("win_odds")
        ai_win_prob = float(f.get("win_prob") or 0.0)
        ai_score = float(f.get("ai_power_index") or f.get("model_score") or 0.0)

        if win_odds is not None and float(win_odds) > 0:
            market_win_prob: Optional[float] = round(1.0 / float(win_odds), 4)
        else:
            market_win_prob = None

        value_gap: Optional[float] = (
            round(ai_win_prob - market_win_prob, 4)
            if market_win_prob is not None
            else None
        )

        rows.append({
            "horse_name":      str(f.get("horse_name") or ""),
            "ai_score":        round(ai_score, 2),
            "ai_win_prob":     round(ai_win_prob, 4),
            "win_odds":        win_odds,
            "market_win_prob": market_win_prob,
            "value_gap":       value_gap,
            "popularity_rank": popularity_map.get(i),
        })

    return rows


# =========================================================
# 妙味馬検出
# =========================================================

def _value_reason(feature: Dict[str, Any], ev_row: Dict[str, Any], race_pace: str) -> str:
    parts = []
    vg = float(ev_row.get("value_gap") or 0.0)

    if vg >= VALUE_GAP_STRONG:
        parts.append("オッズに対して強妙味")
    else:
        parts.append("市場より高いAI評価")

    running_style = feature.get("running_style") or ""
    if running_style == "closer" and race_pace in ("fast", "very_fast"):
        parts.append("ハイペース差し有利")
    elif running_style in ("front", "stalker") and race_pace == "slow":
        parts.append("スロー先行有利")

    if float(feature.get("trend_index") or 0.5) >= 0.62:
        parts.append("上昇傾向")
    if float(feature.get("distance_fit_index") or 0.5) >= 0.65:
        parts.append("距離適性◎")

    return "・".join(parts[:3]) if parts else "AI評価 > 市場評価"


def detect_value_horses(
    ev_table: List[Dict[str, Any]],
    features: List[Dict[str, Any]],
    race_pace: str = "medium",
    min_value_gap: float = VALUE_GAP_MIN,
) -> List[Dict[str, Any]]:
    """
    value_gap >= min_value_gap の馬を妙味候補として返す（理由付き）。
    """
    if not ev_table or not features:
        return []

    features_by_name = {str(f.get("horse_name") or ""): f for f in features}
    candidates = []

    for row in ev_table:
        vg = row.get("value_gap")
        if vg is None or vg < min_value_gap:
            continue
        horse_name = row["horse_name"]
        f = features_by_name.get(horse_name, {})
        reason = _value_reason(f, row, race_pace)
        candidates.append({**row, "reason": reason})

    candidates.sort(key=lambda x: float(x.get("value_gap") or 0), reverse=True)
    return candidates


# =========================================================
# 危険人気馬検出
# =========================================================

def _danger_reason(feature: Dict[str, Any], ev_row: Dict[str, Any], race_pace: str) -> str:
    parts = []
    vg = float(ev_row.get("value_gap") or 0.0)

    if vg <= -DANGER_GAP_MIN * 2:
        parts.append("想定より人気先行")

    running_style = feature.get("running_style") or ""
    if running_style in ("front", "stalker") and race_pace in ("fast", "very_fast"):
        parts.append("先行有利条件に対し差し依存")
    elif running_style == "closer" and race_pace == "slow":
        parts.append("スロー差し不利な展開")

    if float(feature.get("trend_index") or 0.5) <= 0.40:
        parts.append("近走下降傾向")
    if float(feature.get("consistency_index") or 0.5) <= 0.35:
        parts.append("成績にムラあり")
    if float(feature.get("distance_fit_index") or 0.5) <= 0.35:
        parts.append("距離適性に不安")

    age = feature.get("age")
    if isinstance(age, (int, float)) and age >= 7:
        parts.append("年齢面がやや不利")

    return "・".join(parts[:3]) if parts else "AI評価 < 市場評価"


def detect_danger_favorites_v2(
    ev_table: List[Dict[str, Any]],
    features: List[Dict[str, Any]],
    race_pace: str = "medium",
    danger_odds_max: float = DANGER_ODDS_MAX,
    danger_gap_min: float = DANGER_GAP_MIN,
) -> List[Dict[str, Any]]:
    """
    市場人気は高いがAI評価が低い馬（危険人気馬）を抽出する（理由付き）。
    """
    if not ev_table or not features:
        return []

    features_by_name = {str(f.get("horse_name") or ""): f for f in features}
    candidates = []

    for row in ev_table:
        vg = row.get("value_gap")
        win_odds = row.get("win_odds")
        if vg is None or win_odds is None:
            continue
        if float(win_odds) > danger_odds_max:
            continue  # 人気馬のみ対象
        if vg >= -danger_gap_min:
            continue  # 乖離が小さければ対象外

        horse_name = row["horse_name"]
        f = features_by_name.get(horse_name, {})
        reason = _danger_reason(f, row, race_pace)
        candidates.append({**row, "reason": reason})

    # 最も乖離が大きい（value_gapが最小）順
    candidates.sort(key=lambda x: float(x.get("value_gap") or 0))
    return candidates[:3]


# =========================================================
# 過去10年傾向スコア補正
# =========================================================

def trend_score_adjustment(
    feature: Dict[str, Any],
    race_trend_10y: Dict[str, Any],
) -> float:
    """
    過去10年傾向データをもとに各馬のmodel_scoreへの補正値(±delta)を返す。

    【役割と位置づけ】
    apply_trend_adjustments() のフォールバックルートから呼び出される。
    condition_stats が取得できなかった場合（Selenium失敗・データ不足など）に
    race_trend_10y だけで補正を行うための補完関数。

    【削除禁止】
    この関数は "legacy" だが意図的に残している。
    condition_stats なしで race_trend_10y のみ存在するケースは運用上発生するため、
    削除すると当該ケースで補正が完全にゼロになる。

    【新ルート (signal_judge 経由) との主な差分】
    - サンプルサイズ信頼度補正なし
    - エスカレーション（weak×2 → medium 昇格）なし
    - trend_signal_details フィールドを生成しない
    - 枠補正はトークン文字列マッチ（bucket_gate キー完全一致ではない）

    設計方針:
    - 補正は強くしすぎない（TREND_WEIGHTS で上限管理）
    - 年齢だけで足切りしない（減点方式・微減のみ）
    - 傾向不一致も大きく引かない（最大 -WEIGHT * 0.6 程度）
    """
    if not race_trend_10y or not feature:
        return 0.0

    delta = 0.0

    # ----- 脚質傾向 -----
    style_counts: Dict[str, int] = race_trend_10y.get("style") or {}
    if style_counts:
        total = sum(style_counts.values()) or 1
        running_style = feature.get("running_style") or ""
        style_label_map = {
            "front":   ["逃げ"],
            "stalker": ["先行"],
            "closer":  ["差し", "追込"],
        }
        keys = style_label_map.get(running_style, [])
        style_ratio = sum(style_counts.get(k, 0) for k in keys) / total
        if style_ratio >= 0.50:
            delta += TREND_WEIGHTS["style"] * style_ratio
        elif style_ratio <= 0.20:
            delta -= TREND_WEIGHTS["style"] * (0.20 - style_ratio) * 2

    # ----- 人気帯傾向 -----
    # キー書式は bucket_popularity() に依存:
    #   "1番人気" / "2〜3番人気" / "4〜6番人気" / "7〜9番人気" / "10番人気以下"
    pop_counts: Dict[str, int] = race_trend_10y.get("popularity") or {}
    win_odds = feature.get("win_odds")
    if pop_counts and win_odds is not None:
        total = sum(pop_counts.values()) or 1
        target_pop_key = bucket_popularity_odds(float(win_odds))

        matching = pop_counts.get(target_pop_key, 0)
        pop_ratio = matching / total
        if pop_ratio >= 0.40:
            delta += TREND_WEIGHTS["popularity"] * pop_ratio
        elif pop_ratio <= 0.10:
            delta -= TREND_WEIGHTS["popularity"] * 0.5

    # ----- 枠傾向 -----
    gate_counts: Dict[str, int] = race_trend_10y.get("gate") or {}
    gate = feature.get("gate")
    if gate_counts and gate is not None:
        total = sum(gate_counts.values()) or 1
        g = int(gate)
        if g <= 3:
            bucket_tokens = ["1", "2", "3", "内"]
        elif g <= 6:
            bucket_tokens = ["4", "5", "6", "中"]
        else:
            bucket_tokens = ["7", "8", "外"]

        matching = sum(
            v for k, v in gate_counts.items()
            if any(t in str(k) for t in bucket_tokens)
        )
        gate_ratio = matching / total
        if gate_ratio >= 0.45:
            delta += TREND_WEIGHTS["gate"] * gate_ratio
        elif gate_ratio <= 0.15:
            delta -= TREND_WEIGHTS["gate"] * 0.5

    # ----- 年齢傾向 -----
    age_counts: Dict[str, int] = race_trend_10y.get("age") or {}
    age = feature.get("age")
    if age_counts and age is not None:
        total = sum(age_counts.values()) or 1
        age_i = int(age)
        # キー書式は f"{a}歳"（例: "4歳"）。intキーは存在しない。
        matching = sum(v for k, v in age_counts.items() if str(age_i) in str(k))
        age_ratio = matching / total
        if age_ratio >= 0.35:
            delta += TREND_WEIGHTS["age"] * age_ratio
        elif age_ratio <= 0.05 and age_i >= 7:
            # 7歳以上かつ過去傾向と合わない場合のみ微減
            delta -= TREND_WEIGHTS["age"] * 0.6

    return round(delta, 5)


def apply_trend_adjustments(
    features: List[Dict[str, Any]],
    race_trend_10y: Dict[str, Any],
    condition_stats: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    全馬に傾向補正を適用して model_score を更新する。

    【ルート選択の優先順位】
    1. 標準ルート（condition_stats が非空の場合）:
       signal_judge.build_horse_signal_details + aggregate_signal_result を使用。
       走者個別の統計に基づく証拠ベース補正。サンプルサイズ信頼度・エスカレーションあり。

    2. フォールバックルート（condition_stats が空 or None の場合）:
       trend_score_adjustment() を使用。
       race_trend_10y の集計値のみで補正する従来ロジック。
       Selenium取得失敗・fetch_race_history_enriched 例外・データ不足時に通る。
       このルートは意図的に維持しており、統合対象外。

    各馬に追加されるフィールド:
    - trend_delta:          float  — model_score 補正値（両ルート共通）
    - trend_signal_details: Dict   — 条件別シグナル詳細（標準ルートのみ。フォールバック時は未設定）
      ※ trend_signal_details が未設定でも UI 側は .get() でデフォルト処理済みのため実害なし。
    """
    if not features:
        return features

    # --- 標準ルート: condition_stats による証拠ベース補正 ---
    if condition_stats:
        from signal_judge import build_horse_signal_details, aggregate_signal_result
        for f in features:
            details    = build_horse_signal_details(f, condition_stats)
            sig_result = aggregate_signal_result(details)
            delta      = sig_result["total_trend_adjust"]
            f["trend_delta"]          = delta
            f["trend_signal_details"] = sig_result
            f["model_score"]          = round(float(f.get("model_score") or 0.0) + delta, 6)
        return features

    # --- フォールバックルート: race_trend_10y のみ使用 ---
    # condition_stats が取得できなかった場合（Selenium失敗・データ不足など）に通る。
    # trend_signal_details は設定しないが、UI側で .get() によるデフォルト処理が入っているため問題なし。
    if not race_trend_10y:
        return features

    for f in features:
        delta = trend_score_adjustment(f, race_trend_10y)
        f["trend_delta"] = delta
        f["model_score"] = round(float(f.get("model_score") or 0.0) + delta, 6)

    return features


# =========================================================
# 券種別期待値エンジン（プライベートヘルパー）
# =========================================================

def _safe_place_prob(f: Dict[str, Any]) -> float:
    """
    AI複勝確率の推定。

    【race_ai_engine.py の estimate_place_prob と式は同一】
      0.12 + 1.75 * win_prob、clamp [0.05, 0.85]

    【意図的な複製】
    value_ai → race_ai_engine の import を追加すると循環 import が発生する
    （race_ai_engine がすでに value_ai を import しているため）。
    この循環を回避するために同式をここに複製している。

    【将来の整理方針】
    共通関数として trend_stats.py 等の第三モジュールへ切り出す余地はあるが、
    現時点ではロジックの乖離がなく実害もないため、保守上このまま維持する。
    式を変更する場合は race_ai_engine.estimate_place_prob も必ず同期すること。
    """
    win_prob = float(f.get("win_prob") or 0.0)
    return round(min(0.85, max(0.05, 0.12 + 1.75 * win_prob)), 4)


def _market_place_prob(win_odds: float) -> float:
    """
    単勝オッズから市場複勝確率を人気帯別係数で推定。
    推定複勝オッズ = 単勝オッズ / factor → 市場複勝確率 = 1 / 推定複勝オッズ
    """
    if win_odds <= 0:
        return 0.05
    if win_odds <= 3.0:
        factor = PLACE_ODDS_FACTORS["1番人気"]
    elif win_odds <= 7.0:
        factor = PLACE_ODDS_FACTORS["2〜3番人気"]
    elif win_odds <= 15.0:
        factor = PLACE_ODDS_FACTORS["4〜6番人気"]
    elif win_odds <= 30.0:
        factor = PLACE_ODDS_FACTORS["7〜9番人気"]
    else:
        factor = PLACE_ODDS_FACTORS["10番人気以下"]
    est_place_odds = win_odds / factor
    return min(0.85, max(0.05, 1.0 / est_place_odds))


# =========================================================
# フェーズ3: 着順分布推定
# =========================================================

def estimate_placement_probs(
    f: Dict[str, Any],
    race_structure: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    各馬の着順分布確率を推定する。

    p1     = AI勝率（1着確率）
    p_top3 = 3着以内確率（既存 _safe_place_prob 式を利用）
    p_top2 = 2着以内確率（脚質・レース構造で分配比を補正）
    p2     = 2着確率近似 = p_top2 - p1
    p3     = 3着確率近似 = p_top3 - p_top2

    設計方針:
    - 欠損に強い（win_prob=0 でも下限 0.05 で機能）
    - 極端な値になりにくい（各値を [0, 0.95] にクリップ）
    - running_style と race_structure の組み合わせで分配比を微補正
    """
    p1     = float(f.get("win_prob") or 0.0)
    p_top3 = _safe_place_prob(f)   # 0.12 + 1.75 * p1 をベース

    running_style  = f.get("running_style") or ""
    structure_type = (race_structure or {}).get("structure_type", "標準型")

    # 脚質による分配比の基本値
    if running_style in ("front", "stalker"):
        top2_frac = TOP2_FRAC_FRONT
    elif running_style == "closer":
        top2_frac = TOP2_FRAC_CLOSER
    else:
        top2_frac = TOP2_FRAC_DEFAULT

    # 展開 × 脚質の補正
    if structure_type in ("本命信頼型", "先行有利型") and running_style in ("front", "stalker"):
        top2_frac = min(0.80, top2_frac + 0.05)
    elif structure_type in ("差し届く型", "波乱型") and running_style == "closer":
        top2_frac = min(0.80, top2_frac + 0.05)

    p_top2 = p1 + top2_frac * (p_top3 - p1)
    p2     = p_top2 - p1
    p3     = p_top3 - p_top2

    return {
        "p1":     round(min(0.95, max(0.0, p1)), 4),
        "p2":     round(min(0.70, max(0.0, p2)), 4),
        "p3":     round(min(0.60, max(0.0, p3)), 4),
        "p_top2": round(min(0.95, max(0.0, p_top2)), 4),
        "p_top3": round(min(0.95, max(0.0, p_top3)), 4),
    }


def _enrich_placement_probs(
    features: List[Dict[str, Any]],
    race_structure: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    features に着順分布推定値（_pp_ prefix）を付与して返す。
    build_ticket_ev_table の前処理として呼ぶ。
    """
    for f in features:
        pp = estimate_placement_probs(f, race_structure)
        f["_pp_p1"]     = pp["p1"]
        f["_pp_p2"]     = pp["p2"]
        f["_pp_p3"]     = pp["p3"]
        f["_pp_p_top2"] = pp["p_top2"]
        f["_pp_p_top3"] = pp["p_top3"]
    return features


# =========================================================
# 券種別EV補正ヘルパー（構造・脚質・妙味/危険馬）
# =========================================================

# 券種 × レース構造 の補正乗数テーブル（1.0 = 補正なし）
_STRUCTURE_EV_CORR: Dict[str, Dict[str, float]] = {
    "単勝": {
        "本命信頼型":    1.06,
        "先行有利型":    1.03,
        "波乱型":        0.88,
        "差し届く型":    0.92,
    },
    "複勝": {
        "本命信頼型":    1.04,
        "先行有利型":    1.03,
        "波乱型":        0.92,
    },
    "馬連": {
        "本命信頼型":    1.06,
        "1強相手混戦型": 1.03,
        "先行有利型":    1.04,
        "波乱型":        0.88,
        "差し届く型":    0.92,
    },
    "ワイド": {
        "混戦型":        1.06,
        "波乱型":        1.10,
        "差し届く型":    1.05,
        "1強相手混戦型": 0.96,
        "本命信頼型":    0.94,
    },
    "3連複": {
        "混戦型":        1.06,
        "差し届く型":    1.05,
        "波乱型":        0.90,
        "本命信頼型":    0.90,
    },
}


def _structure_ev_corr(bet_type: str, structure_type: str) -> float:
    """券種 × レース構造の補正係数（デフォルト 1.0）"""
    return _STRUCTURE_EV_CORR.get(bet_type, {}).get(structure_type, 1.0)


def _combo_value_corr(
    horse_names: List[str],
    value_names: Set[str],
    danger_names: Set[str],
) -> float:
    """
    組み合わせの価値補正。
    - 妙味馬1頭 → +4%  / 危険馬1頭 → -12%
    - 結果を [0.70, 1.15] にクリップ
    """
    mult = 1.0
    for h in horse_names:
        if h in value_names:
            mult *= 1.04
        if h in danger_names:
            mult *= 0.88
    return round(min(1.15, max(0.70, mult)), 4)


def _style_fit_corr(
    horse_names: List[str],
    features_by_name: Dict[str, Dict[str, Any]],
    favorable_style: str,
) -> float:
    """
    組み合わせの脚質適性補正。
    有利脚質の馬を2頭以上含む → +4%、1頭 → +2%、0頭 → -2%
    """
    if not favorable_style or favorable_style == "unknown":
        return 1.0
    match = sum(
        1 for h in horse_names
        if features_by_name.get(h, {}).get("running_style") == favorable_style
    )
    if match >= 2:
        return 1.04
    if match == 1:
        return 1.02
    return 0.98


def _harville_umaren_hit(pa: float, pb: float, eps: float = 0.001) -> float:
    """
    馬連ヒット確率の Harville 近似。
    P(A1着,B2着) + P(B1着,A2着) = pa*pb/(1-pa) + pb*pa/(1-pb)
    """
    return pa * pb / (1.0 - pa + eps) + pb * pa / (1.0 - pb + eps)


def _wide_corr_factor(odds_a: float, odds_b: float) -> float:
    """
    ワイドのヒット確率補正係数を人気帯に応じて返す。
    EV計算では ai_hit/mkt_hit の比を取るため、分子・分母に同一係数を適用すると
    キャンセルされる。ただし、AIと市場の評価ズレを反映させるため保持する。
    """
    if odds_a <= 7.0 and odds_b <= 7.0:
        return WIDE_CORR_BOTH_POP
    if odds_a > 12.0 and odds_b > 12.0:
        return WIDE_CORR_BOTH_LONG
    return WIDE_CORR_MIX


def _popularity_combo_corr(
    horse_names: List[str],
    ev_table_rows: List[Dict[str, Any]],
) -> float:
    """
    組み合わせの人気構成による市場歪み補正。
    - 全馬人気（オッズ≤7）→ 売れすぎ → -6%
    - 中穴馬（7<オッズ≤20）含む → +4%
    - それ以外 → 補正なし
    """
    if not ev_table_rows or not horse_names:
        return 1.0
    odds_map = {r["horse_name"]: float(r.get("win_odds") or 99.0) for r in ev_table_rows}
    odds_in  = [odds_map.get(h, 99.0) for h in horse_names]
    if all(o <= 7.0 for o in odds_in):
        return POPULAR_COMBO_PENALTY
    if any(7.0 < o <= 20.0 for o in odds_in):
        return MIXED_ODDS_BONUS
    return 1.0


# =========================================================
# 券種別期待値計算（パブリック関数）
# =========================================================

def calc_tansho_ev(f: Dict[str, Any]) -> Optional[float]:
    """単勝EV = AI勝率 × 単勝オッズ"""
    win_odds = f.get("win_odds")
    if win_odds is None:
        return None
    return round(float(f.get("win_prob") or 0.0) * float(win_odds), 3)


def calc_fukusho_ev(f: Dict[str, Any]) -> Optional[float]:
    """
    複勝EV = AI複勝確率(p_top3) × 複勝オッズ
    - place_odds 入力済み → 最優先で使用
    - 未入力 → 人気帯別係数で単勝オッズから推定
    - _enrich_placement_probs 済みなら _pp_p_top3 を使用（より精度が高い）
    """
    win_odds = f.get("win_odds")
    if win_odds is None:
        return None
    # 着順分布推定済みなら _pp_p_top3 を使う
    ai_place_prob = float(f.get("_pp_p_top3") or _safe_place_prob(f))
    place_odds = f.get("place_odds")
    if place_odds is not None and float(place_odds) > 0:
        est_place_odds = float(place_odds)
    else:
        mkt_prob = _market_place_prob(float(win_odds))
        est_place_odds = 1.0 / mkt_prob if mkt_prob > 0 else float(win_odds) / MARKET_PLACE_MULT
    return round(ai_place_prob * est_place_odds, 3)


def calc_umaren_ev_pair(fa: Dict[str, Any], fb: Dict[str, Any]) -> Optional[float]:
    """
    馬連EV — Harville近似（着順分布推定対応）。
    P(A1着,B2着) + P(B1着,A2着) = pa*pb/(1-pa) + pb*pa/(1-pb)
    _enrich_placement_probs 済みなら _pp_p1 を使用。
    """
    odds_a = fa.get("win_odds")
    odds_b = fb.get("win_odds")
    if odds_a is None or odds_b is None:
        return None
    pa_ai  = float(fa.get("_pp_p1") or fa.get("win_prob") or 0.0)
    pb_ai  = float(fb.get("_pp_p1") or fb.get("win_prob") or 0.0)
    pa_mkt = 1.0 / float(odds_a)
    pb_mkt = 1.0 / float(odds_b)
    ai_hit  = _harville_umaren_hit(pa_ai, pb_ai)
    mkt_hit = _harville_umaren_hit(pa_mkt, pb_mkt)
    if mkt_hit <= 0:
        return None
    return round(ai_hit / mkt_hit * JRA_RETURN_RATES["馬連"], 3)


def calc_wide_ev_pair(fa: Dict[str, Any], fb: Dict[str, Any]) -> Optional[float]:
    """
    ワイドEV — p_top3 ベース、人気帯別補正係数。
    _enrich_placement_probs 済みなら _pp_p_top3 を使用。
    """
    odds_a = fa.get("win_odds")
    odds_b = fb.get("win_odds")
    if odds_a is None or odds_b is None:
        return None
    pa_ai  = float(fa.get("_pp_p_top3") or _safe_place_prob(fa))
    pb_ai  = float(fb.get("_pp_p_top3") or _safe_place_prob(fb))
    pa_mkt = _market_place_prob(float(odds_a))
    pb_mkt = _market_place_prob(float(odds_b))
    if pa_mkt * pb_mkt <= 0:
        return None
    corr = _wide_corr_factor(float(odds_a), float(odds_b))
    # corr は ai/mkt 両側に同一適用 → 比では影響するため保持
    ai_hit  = pa_ai  * pb_ai  * corr
    mkt_hit = pa_mkt * pb_mkt * corr
    return round(ai_hit / mkt_hit * JRA_RETURN_RATES["ワイド"], 3)


def calc_sanrenpuku_ev_trio(
    fa: Dict[str, Any],
    fb: Dict[str, Any],
    fc: Dict[str, Any],
) -> Optional[float]:
    """
    3連複EV — p_top3 積近似。
    _enrich_placement_probs 済みなら _pp_p_top3 を使用。
    TRIO_CORR は ai/mkt 両側に作用するため EV 計算ではキャンセルされる。
    """
    odds_a = fa.get("win_odds")
    odds_b = fb.get("win_odds")
    odds_c = fc.get("win_odds")
    if odds_a is None or odds_b is None or odds_c is None:
        return None
    pa_ai  = float(fa.get("_pp_p_top3") or _safe_place_prob(fa))
    pb_ai  = float(fb.get("_pp_p_top3") or _safe_place_prob(fb))
    pc_ai  = float(fc.get("_pp_p_top3") or _safe_place_prob(fc))
    pa_mkt = _market_place_prob(float(odds_a))
    pb_mkt = _market_place_prob(float(odds_b))
    pc_mkt = _market_place_prob(float(odds_c))
    denom  = pa_mkt * pb_mkt * pc_mkt
    if denom <= 0:
        return None
    return round(pa_ai * pb_ai * pc_ai / denom * JRA_RETURN_RATES["3連複"], 3)


def build_ticket_ev_table(
    features: List[Dict[str, Any]],
    top_n: int = 6,
    race_structure: Optional[Dict[str, Any]] = None,
    ev_table: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    上位馬の全券種×組み合わせの期待値テーブルを返す（EV降順ソート済み）。

    フェーズ3 改善:
        - 着順分布推定（estimate_placement_probs）を前処理として適用
        - 馬連は Harville 近似に変更
        - ワイドは人気帯別 WIDE_CORR に変更
        - 人気構成補正（POPULAR_COMBO_PENALTY / MIXED_ODDS_BONUS）を追加

    race_structure / ev_table を与えると以下の補正を適用:
        - 構造補正 / 価値補正 / 脚質補正 / 人気構成補正

    Returns:
        List of {bet_type, horses, ev, ev_raw, correction, ai_hit_prob,
                 has_value_horse, has_danger_horse,
                 struct_corr, value_corr, style_corr, pop_corr}
    """
    if not features:
        return []

    sorted_f = sorted(
        features,
        key=lambda x: float(x.get("win_prob") or 0.0),
        reverse=True,
    )
    # shallow copy でオリジナル features を汚染しない
    top_f = [dict(f) for f in sorted_f[:top_n]]

    # 着順分布推定値を付与（_pp_* キー）
    _enrich_placement_probs(top_f, race_structure)

    rows: List[Dict[str, Any]] = []

    for f in top_f:
        ev = calc_tansho_ev(f)
        if ev is not None:
            rows.append({
                "bet_type":    "単勝",
                "horses":      [str(f.get("horse_name") or "")],
                "ev":          ev,
                "ai_hit_prob": round(float(f.get("_pp_p1") or f.get("win_prob") or 0.0), 4),
            })

    for f in top_f:
        ev = calc_fukusho_ev(f)
        if ev is not None:
            rows.append({
                "bet_type":    "複勝",
                "horses":      [str(f.get("horse_name") or "")],
                "ev":          ev,
                "ai_hit_prob": round(float(f.get("_pp_p_top3") or _safe_place_prob(f)), 4),
            })

    top_f_by_name = {str(f.get("horse_name") or ""): f for f in top_f}
    for fa, fb in combinations(top_f, 2):
        ev = calc_umaren_ev_pair(fa, fb)
        if ev is not None:
            pa = float(fa.get("_pp_p1") or fa.get("win_prob") or 0.0)
            pb = float(fb.get("_pp_p1") or fb.get("win_prob") or 0.0)
            rows.append({
                "bet_type":    "馬連",
                "horses":      [str(fa.get("horse_name") or ""), str(fb.get("horse_name") or "")],
                "ev":          ev,
                "ai_hit_prob": round(_harville_umaren_hit(pa, pb), 4),
            })

    for fa, fb in combinations(top_f, 2):
        ev = calc_wide_ev_pair(fa, fb)
        if ev is not None:
            pa = float(fa.get("_pp_p_top3") or _safe_place_prob(fa))
            pb = float(fb.get("_pp_p_top3") or _safe_place_prob(fb))
            corr = _wide_corr_factor(float(fa.get("win_odds") or 99), float(fb.get("win_odds") or 99))
            rows.append({
                "bet_type":    "ワイド",
                "horses":      [str(fa.get("horse_name") or ""), str(fb.get("horse_name") or "")],
                "ev":          ev,
                "ai_hit_prob": round(pa * pb * corr, 4),
            })

    for fa, fb, fc in combinations(top_f, 3):
        ev = calc_sanrenpuku_ev_trio(fa, fb, fc)
        if ev is not None:
            pa = float(fa.get("_pp_p_top3") or _safe_place_prob(fa))
            pb = float(fb.get("_pp_p_top3") or _safe_place_prob(fb))
            pc = float(fc.get("_pp_p_top3") or _safe_place_prob(fc))
            rows.append({
                "bet_type":    "3連複",
                "horses":      [
                    str(fa.get("horse_name") or ""),
                    str(fb.get("horse_name") or ""),
                    str(fc.get("horse_name") or ""),
                ],
                "ev":          ev,
                "ai_hit_prob": round(pa * pb * pc * TRIO_CORR, 4),
            })

    # ── 補正フェーズ ──────────────────────────────────────────────
    structure_type   = (race_structure or {}).get("structure_type", "標準型")
    favorable_style  = (race_structure or {}).get("favorable_style", "unknown")
    features_by_name = {str(f.get("horse_name") or ""): f for f in top_f}

    if ev_table:
        value_names: Set[str] = {
            r["horse_name"] for r in ev_table
            if (r.get("value_gap") or 0) >= VALUE_GAP_MIN
        }
        danger_names: Set[str] = {
            r["horse_name"] for r in ev_table
            if (r.get("value_gap") or 0) <= -DANGER_GAP_MIN
            and float(r.get("win_odds") or 99) <= DANGER_ODDS_MAX
        }
    else:
        value_names  = set()
        danger_names = set()

    ev_table_for_pop = ev_table or []

    for row in rows:
        ev_raw = row["ev"]
        bet_t  = row["bet_type"]
        hs     = row["horses"]
        sc  = _structure_ev_corr(bet_t, structure_type) if race_structure else 1.0
        vc  = _combo_value_corr(hs, value_names, danger_names) if ev_table else 1.0
        stc = _style_fit_corr(hs, features_by_name, favorable_style) if race_structure else 1.0
        pc  = _popularity_combo_corr(hs, ev_table_for_pop) if ev_table else 1.0
        correction = round(sc * vc * stc * pc, 4)

        row["ev_raw"]          = ev_raw
        row["ev"]              = round(ev_raw * correction, 3)
        row["correction"]      = correction
        row["struct_corr"]     = round(sc, 4)
        row["value_corr"]      = round(vc, 4)
        row["style_corr"]      = round(stc, 4)
        row["pop_corr"]        = round(pc, 4)
        row["has_value_horse"] = any(h in value_names  for h in hs)
        row["has_danger_horse"]= any(h in danger_names for h in hs)

    rows.sort(key=lambda x: x["ev"], reverse=True)
    return rows


# =========================================================
# レース構造分類
# =========================================================

def classify_race_structure(
    features: List[Dict[str, Any]],
    pace_balance: Dict[str, int],
) -> Dict[str, Any]:
    """
    展開予測とAI勝率分布からレース構造を分類する。

    Returns:
        structure_type: str
        description: str
        suitable_bet_types: List[str]
        upset_risk: float (0〜1)
        favorable_style: str  (展開有利な脚質)
    """
    FALLBACK = {
        "structure_type": "判定不可",
        "description": "データ不足",
        "suitable_bet_types": [],
        "upset_risk": 0.5,
        "favorable_style": "unknown",
    }
    if not features:
        return FALLBACK

    probs = sorted(
        [float(f.get("win_prob") or 0.0) for f in features],
        reverse=True,
    )
    top    = probs[0] if len(probs) >= 1 else 0.0
    second = probs[1] if len(probs) >= 2 else 0.0
    third  = probs[2] if len(probs) >= 3 else 0.0
    gap12  = top - second
    gap13  = top - third

    front   = pace_balance.get("逃げ") or 0
    stalker = pace_balance.get("先行") or 0

    # 展開有利な脚質
    if front >= 3:
        favorable_style = "closer"    # ハイペース → 差し有利
    elif front == 0:
        favorable_style = "stalker"   # スロー → 先行有利
    else:
        favorable_style = "unknown"

    # 分類ロジック
    if gap12 >= DOMINANT_PROB_GAP and gap13 >= DOMINANT_PROB_GAP * 1.4:
        return {
            "structure_type":    "本命信頼型",
            "description":       "1強が濃厚。軸に据えやすい。",
            "suitable_bet_types": ["単勝", "馬連", "ワイド"],
            "upset_risk":        0.20,
            "favorable_style":   favorable_style,
        }

    if gap12 >= DOMINANT_PROB_GAP * 0.65 and gap13 < DOMINANT_PROB_GAP:
        return {
            "structure_type":    "1強相手混戦型",
            "description":       "軸は明確だが相手が絞れない。軸1頭流し有効。",
            "suitable_bet_types": ["馬連流し", "3連複1頭軸", "ワイド流し"],
            "upset_risk":        0.35,
            "favorable_style":   favorable_style,
        }

    if front >= 3:
        return {
            "structure_type":    "差し届く型",
            "description":       "ハイペース濃厚。差し馬の浮上に警戒。",
            "suitable_bet_types": ["ワイド", "3連複", "複勝"],
            "upset_risk":        0.45,
            "favorable_style":   "closer",
        }

    if front == 0 and stalker >= 3:
        return {
            "structure_type":    "先行有利型",
            "description":       "スローペース。先行馬が有利。",
            "suitable_bet_types": ["馬連", "ワイド", "単勝"],
            "upset_risk":        0.30,
            "favorable_style":   "stalker",
        }

    if gap12 < CONTESTED_PROB_GAP:
        return {
            "structure_type":    "波乱型",
            "description":       "実力拮抗・荒れ模様。妙味馬重視の手広い買い方が有効。",
            "suitable_bet_types": ["ワイドBOX", "3連複BOX", "複勝"],
            "upset_risk":        0.65,
            "favorable_style":   favorable_style,
        }

    if gap12 < DOMINANT_PROB_GAP:
        return {
            "structure_type":    "混戦型",
            "description":       "中穴決着の可能性あり。ワイド・3連複が狙い目。",
            "suitable_bet_types": ["ワイド", "3連複", "馬連"],
            "upset_risk":        0.45,
            "favorable_style":   favorable_style,
        }

    return {
        "structure_type":    "標準型",
        "description":       "標準的なレース構成。馬連・ワイドが基本。",
        "suitable_bet_types": ["馬連", "ワイド"],
        "upset_risk":        0.35,
        "favorable_style":   favorable_style,
    }


# =========================================================
# 推奨買い目 (1案に絞る)
# =========================================================

def _round_stake(amount: float, unit: int = BANKROLL_UNIT) -> int:
    return max(unit, int(amount // unit * unit))


def recommend_bet_plan(
    features: List[Dict[str, Any]],
    ev_table: List[Dict[str, Any]],
    race_structure: Dict[str, Any],
    bankroll: int,
    race_pace: str = "medium",
) -> Dict[str, Any]:
    """
    券種別EVテーブルを優先し、オッズ未入力時はレース構造ベースにフォールバック。

    優先度:
        1. 券種別EV比較（オッズ入力済みの場合）
        2. レース構造ヒューリスティック（オッズ未入力 or EV全滅時）

    Returns dict:
        bet_type, horses, tickets, total_stake, ticket_count,
        reason, risk_level, ev_type, skip, skip_reason
    """
    EMPTY: Dict[str, Any] = {
        "bet_type":     "-",
        "horses":       [],
        "tickets":      [],
        "total_stake":  0,
        "ticket_count": 0,
        "reason":       "",
        "risk_level":   "-",
        "ev_type":      "-",
        "skip":         True,
        "skip_reason":  "データ不足",
    }

    if not features or not ev_table:
        return EMPTY

    # ── EV ベース選択 ──────────────────────────────────────────────
    has_odds = any(
        f.get("win_odds") is not None and float(f.get("win_odds") or 0) > 0
        for f in features
    )

    if has_odds:
        ticket_evs = build_ticket_ev_table(
            features, top_n=6, race_structure=race_structure, ev_table=ev_table
        )
        if ticket_evs:
            plan = _build_ev_plan(ticket_evs, bankroll)
            if plan:
                return plan
            # 全券種が閾値未満 → 見送り
            return {
                **EMPTY,
                "skip":        True,
                "skip_reason": f"全券種のEV不足（最高EV={ticket_evs[0]['ev']:.2f}）。見送り推奨。",
            }

    # ── 構造ベース fallback ────────────────────────────────────────
    plan = _recommend_by_structure(features, ev_table, race_structure, bankroll, race_pace)
    if not plan.get("skip") and "selection_detail" not in plan:
        stype = race_structure.get("structure_type", "標準型")
        plan["selection_detail"] = {
            "why_bet_type":  f"オッズ未入力のためレース構造（{stype}）から選定",
            "why_combo":     "AI上位馬・妙味馬を優先選択",
            "why_not_other": "オッズ未入力のためEV比較不可",
            "best_by_type":  {},
        }
    return plan


def _score_bet_plan(
    ev: float,
    ai_hit_prob: float,
    n_tickets: int,
    per_stake: int,
) -> float:
    """
    買い目プランの複合スコア（EV × ヒット率重み × 点数ペナルティ）。
    スコアが高いほど「再現性があり少額でも運用しやすい」プランと判断する。

    - per_stake が MIN_STAKE_PER_TICKET を下回るプランは即除外（-1.0）
    - ヒット率が高いほど微加点（arctan近似）
    - 点数が多いほどペナルティ
    """
    if per_stake < MIN_STAKE_PER_TICKET:
        return -1.0
    hit_weight = 1.0 + math.log1p(ai_hit_prob * 10.0) * 0.08
    if n_tickets <= 3:
        ticket_factor = 1.00
    elif n_tickets <= MAX_PRACTICAL_TICKETS:
        ticket_factor = 0.97
    else:
        ticket_factor = 0.93
    return ev * hit_weight * ticket_factor


def _select_best_plan(
    ticket_evs: List[Dict[str, Any]],
    bankroll: int,
) -> Optional[Dict[str, Any]]:
    """
    EVテーブルから最適プランを選定し {best_entry, group, per_stake} を返す。

    選定ルール:
    1. 券種別EV閾値（単勝/複勝 >= EV_SKIP_THRESHOLD、その他 >= EV_COMPOUND_SKIP）を満たす
    2. 同一券種内で MAX_PRACTICAL_TICKETS まで集約
    3. per_stake >= MIN_STAKE_PER_TICKET を保証
    4. 複合スコアで券種を選択
    5. EV差が EV_TIE_MARGIN 以内の券種が複数あれば、少点数・低リスクを優先
    """
    def ev_threshold(bet_type: str) -> float:
        return EV_SKIP_THRESHOLD if bet_type in ("単勝", "複勝") else EV_COMPOUND_SKIP

    candidates = [r for r in ticket_evs if r["ev"] >= ev_threshold(r["bet_type"])]
    if not candidates:
        return None

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in candidates:
        groups[r["bet_type"]].append(r)

    best_score  = -1.0
    best_result: Optional[Dict[str, Any]] = None

    for bet_type, group in groups.items():
        group_sorted = sorted(group, key=lambda x: x["ev"], reverse=True)
        selected     = group_sorted[:MAX_PRACTICAL_TICKETS]
        n            = len(selected)
        raw_per      = bankroll / max(n, 1)
        per_stake    = max(MIN_STAKE_PER_TICKET, (int(raw_per) // BANKROLL_UNIT) * BANKROLL_UNIT)

        top_ev       = selected[0]["ev"]
        top_hit_prob = selected[0]["ai_hit_prob"]
        score = _score_bet_plan(top_ev, top_hit_prob, n, per_stake)

        if score > best_score:
            best_score  = score
            best_result = {"best_entry": selected[0], "group": selected, "per_stake": per_stake}

    return best_result


def _build_ev_plan(
    ticket_evs: List[Dict[str, Any]],
    bankroll: int,
) -> Optional[Dict[str, Any]]:
    """
    EVテーブルから最適プランを選択してフォーマットして返す。
    内部で _select_best_plan を呼ぶ。selection_detail も付与する。
    """
    selection = _select_best_plan(ticket_evs, bankroll)
    if selection is None:
        return None

    best_entry = selection["best_entry"]
    group      = selection["group"]
    per_stake  = selection["per_stake"]
    bet_type   = best_entry["bet_type"]

    tickets = [{"combination": r["horses"], "stake": per_stake} for r in group]

    horses_flat: List[str] = []
    for r in group:
        for h in r["horses"]:
            if h not in horses_flat:
                horses_flat.append(h)

    n        = len(group)
    ev_str   = f"{best_entry['ev']:.2f}"
    corr     = best_entry.get("correction", 1.0)
    corr_note = f" 補正×{corr:.2f}" if corr is not None and abs(corr - 1.0) >= 0.01 else ""
    reason = (
        f"EV最高券種: {bet_type}（EV={ev_str}{corr_note}）。"
        f"{horses_flat[0]}中心{n}点。"
        if horses_flat else f"{bet_type} EV={ev_str}"
    )

    # ── selection_detail ──────────────────────────────────────────
    # 券種別ベストEV
    best_by_type: Dict[str, float] = {}
    for r in ticket_evs:
        bt = r["bet_type"]
        if bt not in best_by_type or r["ev"] > best_by_type[bt]:
            best_by_type[bt] = r["ev"]

    # なぜこの券種か
    why_bt = f"全券種中EV最高（{bet_type} EV={ev_str}）"

    # なぜこの組み合わせか
    why_parts: List[str] = []
    if best_entry.get("has_value_horse"):
        why_parts.append("妙味馬含む")
    if not best_entry.get("has_danger_horse"):
        why_parts.append("危険人気馬なし")
    if corr is not None and corr > 1.03:
        why_parts.append(f"レース構造適合（補正+{(corr - 1) * 100:.0f}%）")
    if best_entry.get("pop_corr", 1.0) >= MIXED_ODDS_BONUS:
        why_parts.append("中穴混じり優遇")
    why_combo = "・".join(why_parts) if why_parts else "EV上位組み合わせ"

    # なぜ他券種ではないか
    others = [(bt, ev) for bt, ev in best_by_type.items() if bt != bet_type]
    others.sort(key=lambda x: -x[1])
    why_not = (
        "他券種EV: " + " / ".join(f"{bt}={ev:.2f}" for bt, ev in others[:4])
        if others else "他に有効候補なし"
    )

    result = _plan(bet_type, horses_flat, tickets, reason, _risk_level_by_type(bet_type), "EV比較型")
    result["selection_detail"] = {
        "why_bet_type":  why_bt,
        "why_combo":     why_combo,
        "why_not_other": why_not,
        "best_by_type":  best_by_type,
    }
    return result


def _risk_level_by_type(bet_type: str) -> str:
    _MAP: Dict[str, str] = {
        "単勝":      "低",
        "複勝":      "低",
        "馬連":      "中",
        "馬連流し":  "中",
        "ワイド":    "中",
        "ワイドBOX": "高",
        "3連複":     "高",
        "3連複BOX":  "高",
    }
    return _MAP.get(bet_type, "中")


def _recommend_by_structure(
    features: List[Dict[str, Any]],
    ev_table: List[Dict[str, Any]],
    race_structure: Dict[str, Any],
    bankroll: int,
    race_pace: str,
) -> Dict[str, Any]:
    """レース構造ベースの推奨買い目（オッズ未入力時 or EV全滅時のフォールバック）"""
    EMPTY: Dict[str, Any] = {
        "bet_type":     "-",
        "horses":       [],
        "tickets":      [],
        "total_stake":  0,
        "ticket_count": 0,
        "reason":       "",
        "risk_level":   "-",
        "ev_type":      "-",
        "skip":         True,
        "skip_reason":  "データ不足",
    }

    structure_type = race_structure.get("structure_type", "標準型")

    sorted_features = sorted(
        features,
        key=lambda x: float(x.get("win_prob") or 0.0),
        reverse=True,
    )

    value_horses  = detect_value_horses(ev_table, features, race_pace)
    danger_horses = detect_danger_favorites_v2(ev_table, features, race_pace)
    danger_names  = {d["horse_name"] for d in danger_horses}

    axis_candidates = [f for f in sorted_features if f.get("horse_name") not in danger_names]
    if not axis_candidates:
        axis_candidates = sorted_features

    axis      = axis_candidates[0]
    axis_name = str(axis.get("horse_name") or "")

    has_value = len(value_horses) > 0
    if not has_value and not axis_candidates:
        return {**EMPTY, "skip": True, "skip_reason": "期待値条件を満たす買い目がありません。見送り推奨。"}

    # ----- ① 本命信頼型 -----
    if structure_type == "本命信頼型":
        second_name = _pick_second(axis_candidates, axis_name, value_horses)
        if second_name:
            per = _round_stake(bankroll * BANKROLL_RATIO_BASE)
            return _plan("馬連", [axis_name, second_name],
                         [{"combination": [axis_name, second_name], "stake": per}],
                         f"本命信頼型。{axis_name}〜{second_name}の馬連。",
                         "低", "構造型")

    # ----- ② 1強相手混戦型 -----
    elif structure_type == "1強相手混戦型":
        others = _build_others(axis_candidates, axis_name, value_horses, max_count=4)
        if len(others) >= 1:
            per     = _round_stake(bankroll / min(len(others) + 1, MAX_BET_TICKETS))
            tickets = [{"combination": [axis_name, o], "stake": per} for o in others]
            return _plan("馬連流し",
                         [axis_name] + others, tickets,
                         f"{axis_name}軸・馬連流し{len(tickets)}点。妙味馬優先。",
                         "中", "構造型")

    # ----- ③ 波乱型 -----
    elif structure_type == "波乱型":
        box_names = [v["horse_name"] for v in value_horses[:2]]
        for f in axis_candidates:
            fn = str(f.get("horse_name") or "")
            if fn not in box_names:
                box_names.append(fn)
            if len(box_names) >= 4:
                break
        if len(box_names) >= 2:
            combos  = list(combinations(box_names[:4], 2))[:MAX_BET_TICKETS]
            per     = _round_stake(bankroll / max(len(combos), 1))
            tickets = [{"combination": list(c), "stake": per} for c in combos]
            return _plan("ワイドBOX",
                         box_names[:4], tickets,
                         f"波乱型。ワイドBOX{len(tickets)}点。妙味馬重視。",
                         "高", "構造型")

    # ----- ④ 差し届く型（ハイペース） -----
    elif structure_type == "差し届く型":
        closer_names = [
            str(f.get("horse_name") or "")
            for f in axis_candidates
            if f.get("running_style") == "closer"
        ]
        for v in value_horses:
            if v["horse_name"] not in closer_names:
                closer_names.append(v["horse_name"])
        pool = closer_names + [str(f.get("horse_name") or "") for f in axis_candidates]
        pool = list(dict.fromkeys(pool))[:4]
        if len(pool) >= 3:
            combos  = list(combinations(pool[:4], 3))[:MAX_BET_TICKETS]
            per     = _round_stake(bankroll / max(len(combos), 1))
            tickets = [{"combination": list(c), "stake": per} for c in combos]
            return _plan("3連複BOX", pool[:4], tickets,
                         f"差し届く型。3連複BOX{len(tickets)}点。",
                         "中", "構造型")

    # ----- ⑤ 先行有利型 -----
    elif structure_type == "先行有利型":
        front_names = [
            str(f.get("horse_name") or "")
            for f in axis_candidates
            if f.get("running_style") in ("front", "stalker")
        ]
        pool = front_names + [str(f.get("horse_name") or "") for f in axis_candidates]
        pool = list(dict.fromkeys(pool))[:2]
        if len(pool) >= 2:
            per = _round_stake(bankroll * BANKROLL_RATIO_BASE)
            return _plan("馬連", pool,
                         [{"combination": pool, "stake": per}],
                         f"先行有利型。{pool[0]}〜{pool[1]}の馬連。",
                         "低", "構造型")

    # ----- ⑥ 混戦型 / 標準型 -----
    else:
        pool = [str(axis_candidates[0].get("horse_name") or "")]
        for v in value_horses:
            if v["horse_name"] not in pool:
                pool.append(v["horse_name"])
        for f in axis_candidates[1:]:
            fn = str(f.get("horse_name") or "")
            if fn not in pool:
                pool.append(fn)
            if len(pool) >= 4:
                break

        if len(pool) >= 3:
            combos  = list(combinations(pool[:4], 3))[:MAX_BET_TICKETS]
            per     = _round_stake(bankroll / max(len(combos), 1))
            tickets = [{"combination": list(c), "stake": per} for c in combos]
            return _plan("3連複BOX", pool[:4], tickets,
                         f"混戦型。3連複BOX{len(tickets)}点。妙味馬を優先選択。",
                         "中", "構造型")
        elif len(pool) >= 2:
            per = _round_stake(bankroll * BANKROLL_RATIO_BASE)
            return _plan("ワイド", pool[:2],
                         [{"combination": pool[:2], "stake": per}],
                         f"混戦型。{pool[0]}〜{pool[1]}のワイド。",
                         "中", "構造型")

    # ----- fallback -----
    stake = _round_stake(bankroll * BANKROLL_RATIO_BASE)
    plan  = _plan("単勝", [axis_name],
                  [{"combination": [axis_name], "stake": stake}],
                  f"{axis_name}の単勝（構造型fallback）。",
                  "低", "構造型")
    plan["selection_detail"] = {
        "why_bet_type":  f"オッズ未入力のためレース構造（{structure_type}）から選定",
        "why_combo":     "AI上位馬・妙味馬を優先選択",
        "why_not_other": "オッズ未入力のためEV比較不可",
        "best_by_type":  {},
    }
    return plan


def _plan(
    bet_type: str,
    horses: List[str],
    tickets: List[Dict[str, Any]],
    reason: str,
    risk_level: str,
    ev_type: str,
) -> Dict[str, Any]:
    total = sum(t["stake"] for t in tickets)
    return {
        "bet_type":     bet_type,
        "horses":       horses,
        "tickets":      tickets,
        "total_stake":  total,
        "ticket_count": len(tickets),
        "reason":       reason,
        "risk_level":   risk_level,
        "ev_type":      ev_type,
        "skip":         False,
        "skip_reason":  "",
    }


def _pick_second(
    axis_candidates: List[Dict[str, Any]],
    axis_name: str,
    value_horses: List[Dict[str, Any]],
) -> Optional[str]:
    """軸の次の候補を返す（妙味馬があれば優先）"""
    for v in value_horses:
        vn = v["horse_name"]
        if vn != axis_name:
            return vn
    for f in axis_candidates[1:]:
        fn = str(f.get("horse_name") or "")
        if fn != axis_name:
            return fn
    return None


def _build_others(
    axis_candidates: List[Dict[str, Any]],
    axis_name: str,
    value_horses: List[Dict[str, Any]],
    max_count: int = 4,
) -> List[str]:
    """軸以外の相手馬リストを構築（妙味馬を優先）"""
    result: List[str] = []
    for v in value_horses:
        vn = v["horse_name"]
        if vn != axis_name and vn not in result:
            result.append(vn)
    for f in axis_candidates[1:]:
        fn = str(f.get("horse_name") or "")
        if fn != axis_name and fn not in result:
            result.append(fn)
        if len(result) >= max_count:
            break
    return result[:max_count]


# =========================================================
# 印（◎○▲☆△×）の割り当て
# =========================================================

def assign_marks(
    features: List[Dict[str, Any]],
    ev_table: List[Dict[str, Any]],
    danger_horses: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    各馬に印を付けて返す。

    ◎: AI勝率1位（危険人気馬でなければ）
    ○: AI勝率2位
    ▲: AI勝率3位
    △: AI勝率4〜5位
    ☆: 妙味上位（◎〜△のついていない馬）
    ×: 危険人気馬フラグ
    """
    if not features:
        return []

    danger_names = {d["horse_name"] for d in danger_horses}
    value_horses  = detect_value_horses(ev_table, features)
    value_names   = [v["horse_name"] for v in value_horses]

    sorted_f = sorted(
        features,
        key=lambda x: float(x.get("win_prob") or 0.0),
        reverse=True,
    )

    mark_order = ["◎", "○", "▲", "△"]
    marks: Dict[str, str] = {}
    rank = 0

    for f in sorted_f:
        name = str(f.get("horse_name") or "")
        if name in danger_names:
            marks[name] = "×"
            continue
        if rank < len(mark_order):
            marks[name] = mark_order[rank]
            rank += 1

    # 妙味馬のうち印がない馬に ☆
    for vn in value_names:
        if vn not in marks:
            marks[vn] = "☆"

    return [
        {"horse_name": str(f.get("horse_name") or ""), "mark": marks.get(str(f.get("horse_name") or ""), "")}
        for f in features
    ]


# =========================================================
# フェーズ3.5: 当たるAI強化
# =========================================================

def calc_horse_scores(
    f: Dict[str, Any],
    race_structure: Optional[Dict[str, Any]] = None,
    ev_row: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    各馬の派生指標を計算する。

    top2_prob, top3_prob  : 着順分布推定値（estimate_placement_probs 利用）
    stable_score          : 安定指数 = (consistency_index + trend_index) / 2
    upside_score          : 上振れ余地 = フォーム × バリュー合成
    axis_score            : 軸適性 = stable × top2 × ペース適合の加重和
    pace_fit              : 脚質 × 展開一致度（0.5〜1.0）
    """
    pp        = estimate_placement_probs(f, race_structure)
    top2_prob = pp["p_top2"]
    top3_prob = pp["p_top3"]

    consistency  = float(f.get("consistency_index") or 0.5)
    trend        = float(f.get("trend_index") or 0.5)
    stable_score = min(1.0, (consistency + trend) / 2.0)

    recent_form  = float(f.get("recent_form_index") or 0.5)
    last3f       = float(f.get("last3f_index") or 0.5)
    form_base    = (recent_form + last3f) / 2.0
    value_gap    = float((ev_row or {}).get("value_gap") or 0.0)
    value_comp   = max(0.0, value_gap * 3.0)
    upside_score = min(1.0, form_base * 0.6 + value_comp * 0.4)

    # ペース適合: 有利脚質との一致度
    running_style   = f.get("running_style") or ""
    favorable_style = (race_structure or {}).get("favorable_style", "")
    if favorable_style and running_style == favorable_style:
        pace_fit = 1.0
    elif favorable_style == "front" and running_style == "stalker":
        pace_fit = 0.75  # 先行は逃げ有利展開でもそれなり
    elif favorable_style == "closer" and running_style == "stalker":
        pace_fit = 0.60
    else:
        pace_fit = 0.50

    axis_score = min(1.0, stable_score * 0.45 + top2_prob * 0.35 + pace_fit * 0.20)

    return {
        "top2_prob":    round(top2_prob, 4),
        "top3_prob":    round(top3_prob, 4),
        "stable_score": round(stable_score, 4),
        "upside_score": round(upside_score, 4),
        "axis_score":   round(axis_score, 4),
        "pace_fit":     round(pace_fit, 4),
    }


def _role_reason_head(f: Dict[str, Any], scores: Dict[str, float]) -> str:
    parts = ["AI勝率上位"]
    if scores["axis_score"] >= 0.55:
        parts.append("軸適性高")
    rs = f.get("running_style") or ""
    if rs in ("front", "stalker"):
        parts.append("先行安定")
    if scores["upside_score"] >= UPSIDE_SCORE_HIGH:
        parts.append("上振れ余地大")
    return "・".join(parts[:3])


def _role_reason_axis(f: Dict[str, Any], scores: Dict[str, float]) -> str:
    parts = ["安定した着順分布"]
    if scores["stable_score"] >= 0.58:
        parts.append("安定指数高")
    if scores["top2_prob"] >= 0.30:
        parts.append("2着以内確率高")
    if scores["pace_fit"] >= 0.75:
        parts.append("展開適性◎")
    return "・".join(parts[:3])


def _role_reason_himo(f: Dict[str, Any], scores: Dict[str, float]) -> str:
    parts = ["3着内圏内"]
    if float(f.get("distance_fit_index") or 0.5) >= 0.65:
        parts.append("距離適性◎")
    if float(f.get("style_suitability_index") or 0.5) >= 0.65:
        parts.append("コース適性◎")
    if scores["upside_score"] >= 0.45:
        parts.append("上振れ期待")
    return "・".join(parts[:3])


def _role_reason_fade(f: Dict[str, Any], scores: Dict[str, float]) -> str:
    parts = []
    if float(f.get("trend_index") or 0.5) <= 0.40:
        parts.append("近走下降")
    if scores["top3_prob"] < 0.20:
        parts.append("着順分布低調")
    if scores["stable_score"] < 0.40:
        parts.append("成績にムラ")
    return "・".join(parts[:3]) if parts else "AI評価低調"


def assign_roles(
    features: List[Dict[str, Any]],
    ev_table: List[Dict[str, Any]],
    race_structure: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    各馬に役割を割り当てる。

    head : 頭向き（勝ち筋あり — 単勝・馬連軸向き）
    axis : 軸向き（連対安定 — 馬連・3連複軸向き）
    himo : ヒモ向き（3着付け候補）
    fade : 消し寄り（危険人気馬または期待薄）

    Returns list of dicts with keys:
        horse_name, role, reason, win_prob, top2_prob, top3_prob,
        stable_score, upside_score, axis_score
    """
    if not features:
        return []

    ev_by_name: Dict[str, Dict[str, Any]] = (
        {row["horse_name"]: row for row in ev_table} if ev_table else {}
    )

    # 危険人気馬名セット（v2 と同基準）
    danger_names: Set[str] = set()
    for row in ev_table:
        vg = row.get("value_gap")
        wo = row.get("win_odds")
        if vg is not None and wo is not None:
            if float(wo) <= DANGER_ODDS_MAX and vg <= -DANGER_GAP_MIN:
                danger_names.add(row["horse_name"])

    sorted_by_win = sorted(
        features,
        key=lambda x: float(x.get("win_prob") or 0.0),
        reverse=True,
    )
    top2_names: Set[str] = {str(f.get("horse_name") or "") for f in sorted_by_win[:2]}

    results = []
    for f in features:
        name   = str(f.get("horse_name") or "")
        ev_row = ev_by_name.get(name)
        scores = calc_horse_scores(f, race_structure, ev_row)

        axis_s  = scores["axis_score"]
        stable  = scores["stable_score"]
        top2_p  = scores["top2_prob"]
        top3_p  = scores["top3_prob"]
        win_p   = float(f.get("win_prob") or 0.0)

        # 騎手補正による閾値の微調整（apply_jockey_adjustments 適用済みの場合に有効）
        jockey_delta  = float(f.get("jockey_delta") or 0.0)
        jockey_codes  = f.get("jockey_reason_codes") or []
        jockey_summary = f.get("jockey_summary") or ""

        # 騎手ボーナス: FAVORITE_TRUST → 頭候補の有効性を微引き上げ
        #               FAVORITE_LOW_TRUST → 危険馬方向に傾ける
        #               LONGSHOT_UPSIDE → ヒモ候補の有効性を微引き上げ
        head_threshold_adj  = -0.03 if "FAVORITE_TRUST"    in jockey_codes else (
                               0.04 if "FAVORITE_LOW_TRUST" in jockey_codes else 0.0)
        himo_threshold_adj  = -0.02 if "LONGSHOT_UPSIDE"   in jockey_codes else 0.0

        if name in danger_names and "FAVORITE_LOW_TRUST" not in jockey_codes:
            # 危険人気馬だが騎手補正がニュートラルな場合はそのまま fade
            role   = "fade"
            reason = "危険人気馬（AI<市場評価）"
        elif name in danger_names and "FAVORITE_LOW_TRUST" in jockey_codes:
            role   = "fade"
            reason = "危険人気馬（AI<市場評価）・" + (jockey_summary or "人気馬信頼度低")
        elif (
            name in top2_names
            and win_p >= WIN_PROB_HEAD_MIN
            and axis_s >= (AXIS_SCORE_HEAD_MIN + head_threshold_adj)
        ):
            role   = "head"
            reason = _role_reason_head(f, scores)
            if jockey_summary:
                reason = reason + "・" + jockey_summary
        elif (
            axis_s  >= AXIS_SCORE_AXIS_MIN
            and stable >= STABLE_SCORE_AXIS_MIN
            and top2_p >= TOP2_AXIS_MIN
        ):
            role   = "axis"
            reason = _role_reason_axis(f, scores)
            if jockey_summary and "STYLE_FIT" in jockey_codes:
                reason = reason + "・" + jockey_summary
        elif top3_p >= (TOP3_HIMO_MIN + himo_threshold_adj):
            role   = "himo"
            reason = _role_reason_himo(f, scores)
            if jockey_summary and "LONGSHOT_UPSIDE" in jockey_codes:
                reason = reason + "・" + jockey_summary
        else:
            role   = "fade"
            reason = _role_reason_fade(f, scores)

        results.append({
            "horse_name":   name,
            "role":         role,
            "reason":       reason,
            "win_prob":     win_p,
            "top2_prob":    top2_p,
            "top3_prob":    top3_p,
            "stable_score": stable,
            "upside_score": scores["upside_score"],
            "axis_score":   axis_s,
            "jockey_delta": jockey_delta,
        })

    return results


def detect_rescue_candidates(
    features: List[Dict[str, Any]],
    ev_table: List[Dict[str, Any]],
    race_structure: Optional[Dict[str, Any]] = None,
    race_pace: str = "medium",
) -> List[Dict[str, Any]]:
    """
    AIランク上位ではないが3着以内の可能性が高い「取りこぼし注意馬」を返す。

    条件:
    - AI順位が RESCUE_AI_RANK_MAX より下（上位は除外）
    - top3_prob >= TOP3_RESCUE_MIN
    - ペース適合 または 傾向上昇 のいずれかを満たす
    """
    if not features:
        return []

    ev_by_name = {row["horse_name"]: row for row in ev_table} if ev_table else {}
    favorable_style = (race_structure or {}).get("favorable_style", "")

    sorted_by_win = sorted(
        features,
        key=lambda x: float(x.get("win_prob") or 0.0),
        reverse=True,
    )
    top_names: Set[str] = {str(f.get("horse_name") or "") for f in sorted_by_win[:RESCUE_AI_RANK_MAX]}

    candidates = []
    for f in features:
        name = str(f.get("horse_name") or "")
        if name in top_names:
            continue

        ev_row = ev_by_name.get(name)
        scores = calc_horse_scores(f, race_structure, ev_row)
        top3_p = scores["top3_prob"]

        if top3_p < TOP3_RESCUE_MIN:
            continue

        running_style = f.get("running_style") or ""
        trend_up      = float(f.get("trend_index") or 0.5) >= 0.60

        # ペース適合チェック
        pace_match = False
        if favorable_style:
            if favorable_style == "closer" and running_style == "closer":
                pace_match = race_pace in ("fast", "very_fast")
            elif favorable_style in ("front", "stalker") and running_style in ("front", "stalker"):
                pace_match = race_pace in ("slow", "very_slow")
            elif running_style == favorable_style:
                pace_match = True

        if not (pace_match or trend_up):
            continue

        parts = []
        if pace_match:
            style_jp = {"front": "逃げ", "stalker": "先行", "closer": "差し"}.get(
                running_style, running_style
            )
            parts.append(f"{style_jp}有利の展開")
        if trend_up:
            parts.append("上昇傾向")
        if float(f.get("distance_fit_index") or 0.5) >= 0.65:
            parts.append("距離適性◎")

        candidates.append({
            "horse_name": name,
            "reason":     "・".join(parts[:3]) if parts else "着圏内可能性あり",
            "top3_prob":  top3_p,
            "top2_prob":  scores["top2_prob"],
            "win_prob":   float(f.get("win_prob") or 0.0),
            "axis_score": scores["axis_score"],
        })

    candidates.sort(key=lambda x: x["top3_prob"], reverse=True)
    return candidates[:4]


def _danger_reason_v3(
    f: Dict[str, Any],
    ev_row: Dict[str, Any],
    race_pace: str,
    scores: Dict[str, float],
    pace_conflict: bool,
) -> str:
    parts = []
    vg = float(ev_row.get("value_gap") or 0.0)

    if vg <= -DANGER_GAP_MIN * 2:
        parts.append("人気先行(AI低評価)")

    if pace_conflict:
        running_style = f.get("running_style") or ""
        style_jp = {"front": "逃げ", "stalker": "先行", "closer": "差し"}.get(
            running_style, running_style
        )
        parts.append(f"{style_jp}で展開不利")

    if float(f.get("trend_index") or 0.5) <= 0.40:
        parts.append("近走下降傾向")
    if float(f.get("consistency_index") or 0.5) <= 0.35:
        parts.append("成績にムラあり")
    if float(f.get("distance_fit_index") or 0.5) <= 0.35:
        parts.append("距離適性に不安")

    age = f.get("age")
    if isinstance(age, (int, float)) and age >= 7:
        parts.append("年齢面がやや不利")

    if scores["top3_prob"] >= DANGER_V3_TOP3_MAX:
        parts.append("※3着圏内は残る可能性")

    return "・".join(parts[:4]) if parts else "AI評価 < 市場評価"


def detect_danger_favorites_v3(
    ev_table: List[Dict[str, Any]],
    features: List[Dict[str, Any]],
    race_structure: Optional[Dict[str, Any]] = None,
    race_pace: str = "medium",
    danger_odds_max: float = DANGER_ODDS_MAX,
    danger_gap_min: float = DANGER_GAP_MIN,
) -> List[Dict[str, Any]]:
    """
    危険人気馬を「真に危険（消し）」と「相手なら残る」に分けて返す。

    is_truly_dangerous=True  : 単純に消せる危険馬
    is_truly_dangerous=False : EVは低いが3着は残る可能性あり（ヒモとして残す選択肢）
    """
    if not ev_table or not features:
        return []

    features_by_name = {str(f.get("horse_name") or ""): f for f in features}
    candidates = []

    for row in ev_table:
        vg       = row.get("value_gap")
        win_odds = row.get("win_odds")
        if vg is None or win_odds is None:
            continue
        if float(win_odds) > danger_odds_max:
            continue
        if vg >= -danger_gap_min:
            continue

        horse_name = row["horse_name"]
        f          = features_by_name.get(horse_name, {})
        scores     = calc_horse_scores(f, race_structure, row)

        axis_s = scores["axis_score"]
        top3_p = scores["top3_prob"]

        running_style   = f.get("running_style") or ""
        favorable_style = (race_structure or {}).get("favorable_style", "")
        pace_conflict   = (
            (favorable_style == "closer" and running_style in ("front", "stalker"))
            or (favorable_style in ("front", "stalker") and running_style == "closer")
        )

        is_truly_dangerous = (axis_s < DANGER_V3_AXIS_MAX and top3_p < DANGER_V3_TOP3_MAX)
        if pace_conflict:
            is_truly_dangerous = True

        reason = _danger_reason_v3(f, row, race_pace, scores, pace_conflict)
        candidates.append({
            **row,
            "reason":             reason,
            "is_truly_dangerous": is_truly_dangerous,
            "axis_score":         axis_s,
            "top3_prob":          top3_p,
        })

    candidates.sort(key=lambda x: float(x.get("value_gap") or 0))
    return candidates[:3]
