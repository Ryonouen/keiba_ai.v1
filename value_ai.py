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

# 妙味候補の最低条件（「妙味だけで飛びやすい馬」の過大評価抑制）
VALUE_MIN_WIN_PROB: float = 0.07    # AI勝率がこれ未満の馬は value_gap があっても妙味候補に入れない
VALUE_MIN_STABLE_SCORE: float = 0.42  # (consistency_index + trend_index)/2 がこれ未満の馬も除外

# 乖離分析: 能力ランクが市場ランクより何位以上良ければ「市場が過小評価」と判断するか
ABILITY_UNDERRATED_GAP_MIN: int = 3   # 市場10番人気→能力7位以内 でフラグ
ABILITY_SCORE_MIN: float = 45.0        # 最低限の能力値がある馬のみ対象（雑魚を除外）

# 主推奨昇格のための安定性下限（妙味候補に残りつつも主推奨に上がりにくくする 3 段階設計）
# stable < 0.35 → 妙味候補に入れない（VALUE_MIN_STABLE_SCORE）
# 0.35 ≤ stable < 0.38 → 候補入りするが単勝/複勝ともペナルティ
# 0.38 ≤ stable < 0.45 → 複勝 OK、単勝はペナルティ
# 0.45 ≤ stable → 制約なし
TANSHO_STABLE_MIN: float = 0.45   # 単勝の主推奨昇格に必要な安定性（高め：1着が必要なため）
FUKUSHO_STABLE_MIN: float = 0.38  # 複勝の主推奨昇格に必要な安定性（やや緩め：3着圏内でよいため）
UMAREN_AXIS_STABLE_MIN: float = 0.43  # 馬連の軸馬安定性下限（軸が安定しないと組み合わせ全体が破綻しやすい）
WIDE_AXIS_STABLE_MIN: float = 0.38   # ワイド/3連複の安定性下限（3着圏内なので複勝と同水準）

# 過去傾向補正の重み（model_scoreへの加減算の最大値）
TREND_WEIGHTS: Dict[str, float] = {
    "style":       0.025,   # 脚質傾向補正
    "popularity":  0.015,   # 人気帯傾向補正
    "gate":        0.010,   # 枠傾向補正
    "age":         0.022,   # 年齢傾向補正
    "prev_class":  0.025,   # 前走クラス補正（G1前走 + / 条件戦前走 −）
    "prev_rank":   0.018,   # 前走着順補正（G1距離違い考慮）
    "prev_course": 0.010,   # 前走コース継続補正（同一競馬場 +）
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
MAX_PRACTICAL_TICKETS: int = 10    # 点数上限（旧5点から緩和 — EVがあれば10点まで許容）
EV_TIE_MARGIN: float       = 0.10  # EV差がこれ以下なら少点数・低リスク券種を優先

# Kelly 賭け金計算パラメータ
# half-Kelly を採用（フルKellyは分散が大きすぎるため）
# f_kelly = KELLY_FRACTION × p × (EV-1) / (EV-p)
KELLY_FRACTION: float   = 0.5    # half-Kelly
KELLY_MAX_RATIO: float  = 0.10   # bankroll に対する上限（10%）
KELLY_MIN_RATIO: float  = 0.01   # bankroll に対する下限（1%、最低掛け金保証用）

# ── フェーズ3: 着順分布推定定数 ──────────────────────────────────
# p_top2 = p1 + TOP2_FRAC × (p_top3 - p1)
TOP2_FRAC_FRONT:   float = 0.68   # 逃げ・先行: ポジション保持しやすい → p_top2 寄り
TOP2_FRAC_CLOSER:  float = 0.58   # 差し馬: 2着以内はブレやすい
TOP2_FRAC_DEFAULT: float = 0.63   # 不明・その他

# ── JRA経験的3着内率（人気順位別）──────────────────────────────
# 出典: JRA公式統計 過去10年・中央競馬全レース集計の近似値
# 3連複のp_top3推定にAI推定値とブレンドして使用する
EMPIRICAL_TOP3_RATES: Dict[int, float] = {
    1:  0.65,   # 1番人気: 約65%（圧倒的に高い）
    2:  0.52,   # 2番人気: 約52%
    3:  0.40,   # 3番人気: 約40%
    4:  0.32,   # 4番人気: 約32%
    5:  0.26,   # 5番人気: 約26%
    6:  0.22,   # 6番人気: 約22%
    7:  0.18,   # 7番人気: 約18%
    8:  0.15,   # 8番人気: 約15%
    9:  0.13,   # 9番人気: 約13%
}
EMPIRICAL_TOP3_DEFAULT: float = 0.10  # 10番人気以下

# 人気順位別ブレンド係数（経験データの重み）
# 人気馬ほど経験データが信頼できる → 重みを上げる
# AI能力推定が高い場合は自然にAI側が勝つ（加重平均なので）
EMPIRICAL_BLEND_WEIGHTS: Dict[int, float] = {
    1:  0.40,   # 1番人気: 経験データ40% + AI推定60%
    2:  0.35,   # 2番人気: 経験データ35%
    3:  0.28,   # 3番人気: 経験データ28%
    4:  0.20,   # 4番人気: 経験データ20%
    5:  0.15,   # 5番人気: 経験データ15%
    6:  0.12,   # 6番人気: 経験データ12%
}

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
STABLE_SCORE_AXIS_MIN: float = 0.52  # 軸候補: stable_score の最低値
TOP2_AXIS_MIN: float       = 0.28    # 軸候補: top2_prob の最低値
TOP3_HIMO_MIN: float       = 0.28    # ヒモ候補: top3_prob の最低値
TOP3_RESCUE_MIN: float     = 0.30    # 取りこぼし注意馬: top3_prob の最低値
UPSIDE_SCORE_HIGH: float   = 0.55    # 上振れ余地が高い閾値
RESCUE_AI_RANK_MAX: int    = 3       # 取りこぼし注意馬: AI上位N頭を除外
DANGER_V3_AXIS_MAX: float  = 0.35    # 「真に危険」判定: axis_score 上限
DANGER_V3_TOP3_MAX: float  = 0.25    # 「真に危険」判定: top3_prob 上限

# 人気帯別 危険判定乖離閾値（実力馬を過剰排除しないよう人気馬は閾値を大きくする）
# AI win_prob は絶対値推定なので人気馬ほど market_win_prob との乖離が構造的に拡大する
DANGER_GAP_POP1: float = 0.15   # 1番人気: 15pt以上の乖離がないと危険判定しない
DANGER_GAP_POP3: float = 0.10   # 2〜3番人気: 10pt以上
DANGER_GAP_POP6: float = 0.05   # 4〜6番人気: 5pt以上
# 7番人気以下: DANGER_GAP_MIN (0.03) をそのまま使用

# ── v2 買い目生成 定数 ─────────────────────────────────────────────
# 閾値変更は下記定数だけ触れば全関数に反映される。
V2_AXIS_BAN_GAP: float      = -0.08   # prob_gap <= this → axis_ban=True（軸禁止）
V2_PARTNER_BAN_GAP: float   = -0.10   # prob_gap <= this → partner_ban 判定入口
V2_PARTNER_BAN_EV: float    = 0.75    # win_ev <= this かつ gap 条件満足 → partner_ban=True
V2_AXIS_W_WIN_PROB: float   = 0.35    # axis_score_v2: ai_win_prob の重み
V2_AXIS_W_TOP2: float       = 0.25    # axis_score_v2: top2_prob の重み
V2_AXIS_W_TOP3: float       = 0.10    # axis_score_v2: top3_prob の重み
V2_AXIS_W_MARKET: float     = 0.10    # axis_score_v2: market_win_prob の重み（能力/市場差分）
V2_AXIS_W_GAP: float        = 0.10    # axis_score_v2: prob_gap 正規化値の重み
V2_AXIS_W_EV: float         = 0.05    # axis_score_v2: win_ev 正規化値の重み
V2_AXIS_W_TREND: float      = 0.05    # axis_score_v2: trend_delta 正規化値の重み
V2_PARTNER_W_TOP3: float    = 0.45    # partner_score: top3_prob の重み
V2_PARTNER_W_GAP: float     = 0.25    # partner_score: prob_gap 正規化値の重み
V2_PARTNER_W_EV: float      = 0.20    # partner_score: win_ev 正規化値の重み
V2_PARTNER_W_TREND: float   = 0.10    # partner_score: trend_delta 正規化値の重み
V2_TANSHO_EV_MIN: float     = 1.0     # 単勝推奨: win_ev の最低値
V2_PARTNER_MAX: int         = 4       # 馬連流し: 相手の最大点数
V2_WIDE_BAN_BOTH_GAP: float = -0.05  # ワイド: 両馬ともこれ以下なら対象外
V2_RACE_WARN_GAP: float     = -0.08  # race-level warning: 人気馬の gap 閾値
V2_RACE_WARN_POP: int       = 3      # race-level warning: 対象人気帯（1〜N位）
V2_RACE_WARN_COUNT: int     = 2      # race-level warning: 該当頭数の下限

# ── v3 買い目生成 定数 ─────────────────────────────────────────────
# キャリブレーション（人気帯補正）
V3_CALIB_POP1_3_DELTA: float    = +0.03  # 1-3番人気: AI過小評価補正（+3pt）
V3_CALIB_POP8_PLUS_DELTA: float = -0.02  # 8番人気以下: AI過大評価補正（-2pt）
# マーケット不一致ガード
V3_GUARD_GAP_THRESH: float      = -0.08  # prob_gap <= この値 → underestimated 判定
V3_GUARD_POP_BAND: int          = 3      # 1〜N番人気を監視対象
V3_GUARD_HIGH_COUNT: int        = 2      # この頭数以上 → high alert
V3_GUARD_MED_COUNT: int         = 1      # この頭数以上 → medium alert
# ベットサイズポリシー
V3_POLICY_HIGH_BRL_MULT: float  = 0.50   # high alert: 予算50%に圧縮
V3_POLICY_MED_BRL_MULT: float   = 0.75   # medium alert: 予算75%に圧縮
V3_POLICY_HIGH_WIDE_MAX: int    = 2      # high alert: ワイド最大点数
V3_POLICY_HIGH_UMAREN: bool     = False  # high alert: 馬連を停止
# レース形状分類
V3_SHAPE_SOLID_CONC: float      = 0.55   # 上位3頭の合計確率 > この値 → solid
V3_SHAPE_BALANCED_LOW: float    = 0.40   # balanced の下限
V3_SHAPE_EV_COUNT_CHAOS: int    = 4      # EV>=1.0 の馬がこれ以上 → chaotic 方向
V3_SHAPE_CONFLICT_COUNT: int    = 2      # prob_gap<=-0.08 かつ人気上位がこれ以上 → conflict
# スコア重み: 単勝
V3_WIN_W_CALIB: float           = 0.50   # キャリブレーション済み勝率
V3_WIN_W_WIN_EV: float          = 0.30   # win EV
V3_WIN_W_GAP: float             = 0.20   # prob_gap 正規化値
# スコア重み: 複勝
V3_PLACE_W_TOP3: float          = 0.55   # top3_prob
V3_PLACE_W_GAP: float           = 0.25   # prob_gap 正規化値
V3_PLACE_W_TREND: float         = 0.20   # trend_delta 正規化値
# スコア重み: ワイド（ペア平均）
V3_WIDE_W_TOP3_AVG: float       = 0.50   # 両馬 top3_prob の平均
V3_WIDE_W_GAP_AVG: float        = 0.30   # 両馬 prob_gap 正規化値の平均
V3_WIDE_W_EV_AVG: float         = 0.20   # 両馬 win_ev 正規化値の平均
# スコア重み: 馬連（軸×相手）
V3_UMAREN_W_AXIS_CALIB: float   = 0.40   # 軸のキャリブレーション済み勝率
V3_UMAREN_W_AXIS_GAP: float     = 0.20   # 軸の prob_gap 正規化値
V3_UMAREN_W_PART_TOP3: float    = 0.25   # 相手の top3_prob
V3_UMAREN_W_PART_GAP: float     = 0.15   # 相手の prob_gap 正規化値
# その他
V3_TANSHO_EV_MIN: float         = 1.0    # 単勝: win_ev の最低値
V3_PARTNER_MAX: int             = 4      # 馬連流し: 相手最大点数
V3_WIDE_MAX: int                = 3      # ワイド: 最大点数（通常時）
V3_AXIS_BAN_GAP: float          = -0.08  # 軸禁止: prob_gap <= この値
V3_PARTNER_BAN_GAP: float       = -0.10  # 相手禁止入口
V3_PARTNER_BAN_EV: float        = 0.75   # 相手禁止: win_ev <= この値（gap 条件と AND）


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

        ability_rank  = f.get("ability_rank")    # attach_ability_scores で付与
        winprob_rank  = f.get("winprob_rank")    # 同上
        market_rank   = popularity_map.get(i)     # 単勝オッズ昇順の人気順位

        # 市場評価 vs AI評価の乖離指標
        # 正 = 市場より能力が高い（アンダーレイ候補）/ 負 = 市場より能力が低い（オーバーレイ候補）
        ability_vs_market_gap = (
            (market_rank - ability_rank)
            if ability_rank is not None and market_rank is not None
            else None
        )
        winprob_vs_market_gap = (
            (market_rank - winprob_rank)
            if winprob_rank is not None and market_rank is not None
            else None
        )

        rows.append({
            "horse_name":             str(f.get("horse_name") or ""),
            "ai_score":               round(ai_score, 2),
            "ai_win_prob":            round(ai_win_prob, 4),
            "win_odds":               win_odds,
            "market_win_prob":        market_win_prob,
            "value_gap":              value_gap,
            # popularity_rank は win_odds の昇順で毎回再計算される動的順位。
            # features["popularity"]（スクレイピング時点の固定人気）とは別物であり、
            # オッズ更新後は両者がズレる場合がある。assign_roles での扱い注意。
            "popularity_rank":        market_rank,
            # --- 将来の妙味・危険馬判定強化用 ---
            "ability_score":          round(float(f.get("ability_score") or 50.0), 1),
            "ability_rank":           ability_rank,
            "winprob_rank":           winprob_rank,
            "market_rank":            market_rank,
            "ability_vs_market_gap":  ability_vs_market_gap,   # 正=能力>市場評価
            "winprob_vs_market_gap":  winprob_vs_market_gap,   # 正=AI勝率>市場評価
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

    # 能力 vs 市場乖離を理由に追記
    ability_gap = ev_row.get("ability_vs_market_gap")
    if ability_gap is not None and ability_gap >= ABILITY_UNDERRATED_GAP_MIN:
        ab_rank = ev_row.get("ability_rank")
        mk_rank = ev_row.get("market_rank")
        parts.append(f"能力{ab_rank}位↑市場{mk_rank}位（能力過小評価）")

    return "・".join(parts[:4]) if parts else "AI評価 > 市場評価"


def detect_underrated_by_ability(
    ev_table: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    「市場が能力を過小評価している馬」を返す。

    value_gap（オッズ乖離）がなくても、
        market_rank - ability_rank >= ABILITY_UNDERRATED_GAP_MIN
        かつ ability_score >= ABILITY_SCORE_MIN
    の馬は「能力的に市場より高い評価」として別リストで提示する。

    これにより、以下のケースを捕捉できる:
        - 実力はあるが展開不利で win_prob が低い → value_gap なし
        - ただし ability_rank は市場より大幅に上位
        → 「能力はあるが買いにくい」穴馬候補

    返り値フィールド:
        horse_name, ability_rank, market_rank, ability_vs_market_gap,
        ability_score, ai_win_prob, win_odds, underrate_reason
    """
    if not ev_table:
        return []

    result: List[Dict[str, Any]] = []
    for row in ev_table:
        gap = row.get("ability_vs_market_gap")
        if gap is None or gap < ABILITY_UNDERRATED_GAP_MIN:
            continue
        score = float(row.get("ability_score") or 0.0)
        if score < ABILITY_SCORE_MIN:
            continue

        ab_rank = row.get("ability_rank")
        mk_rank = row.get("market_rank")
        result.append({
            "horse_name":             row.get("horse_name"),
            "ability_rank":           ab_rank,
            "market_rank":            mk_rank,
            "ability_vs_market_gap":  gap,
            "ability_score":          round(score, 1),
            "ai_win_prob":            row.get("ai_win_prob"),
            "win_odds":               row.get("win_odds"),
            "underrate_reason": (
                f"能力{ab_rank}位 vs 市場{mk_rank}位人気 "
                f"（乖離{gap}位分）"
            ),
        })

    result.sort(key=lambda x: x.get("ability_vs_market_gap", 0), reverse=True)
    return result


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

        # AI勝率下限チェック: 勝率が極めて低い馬はオッズ妙味だけで上位に来るのを抑制
        ai_win_prob = float(row.get("ai_win_prob") or 0.0)
        if ai_win_prob < VALUE_MIN_WIN_PROB:
            continue

        horse_name = row["horse_name"]
        f = features_by_name.get(horse_name, {})

        # 安定性下限チェック: 成績が極端にバラバラな馬（飛びやすい）を妙味候補から除外
        # consistency_index / trend_index が未取得の場合は 0.5（中程度）扱いで通す。
        # これは「データ不足 = 危険とは断定しない」という意図的な設計であり、
        # デフォルトを VALUE_MIN_STABLE_SCORE(0.42) 以下に下げると
        # データ未取得の馬を一律除外することになるため変更しないこと。
        consistency = float(f.get("consistency_index") or 0.5)
        trend = float(f.get("trend_index") or 0.5)
        stable = (consistency + trend) / 2.0
        if stable < VALUE_MIN_STABLE_SCORE:
            continue

        # 年齢シグナルが明確な懸念（medium/strong_negative）の場合、
        # 妙味候補からは除外しないが「強妙味」への昇格を禁止する。
        # 超長オッズ馬は value_gap が大きくなりやすいため、
        # そのレースの年齢別3着内率が明確に低い馬は推しすぎを抑制する。
        age_signal = f.get("age_signal", "neutral")
        age_label_cap = "妙味" if age_signal in ("medium_negative", "strong_negative") else None

        reason = _value_reason(f, row, race_pace)
        entry = {**row, "reason": reason}
        if age_label_cap:
            entry["age_label_cap"] = age_label_cap
        candidates.append(entry)

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

    # 年齢の固定テキストは廃止。レース別傾向データを持つ場合は
    # trend_signal_details 経由で表示されるため、ここでは一律表記しない。

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
        # 人気帯別乖離閾値: 実力馬ほど AI と市場の差が構造的に大きくなるため閾値を引き上げる
        _pop = int(row.get("popularity_rank") or 99)
        _gap_thresh = (
            -DANGER_GAP_POP1 if _pop == 1 else
            -DANGER_GAP_POP3 if _pop <= 3 else
            -DANGER_GAP_POP6 if _pop <= 6 else
            -danger_gap_min
        )
        if vg >= _gap_thresh:
            continue

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

def _calc_age_delta_from_trend(
    feature: Dict[str, Any],
    race_trend_10y: Dict[str, Any],
) -> float:
    """
    race_trend_10y の年齢別勝者カウントから年齢補正値を計算する。
    signal_judge が年齢 neutral のときに標準ルートの補完として使う。

    【評価軸の統一】
    旧実装は固定閾値（ratio <= 0.05 → -0.025 等）を使っていたが、
    標準ルート（condition_stats）の diff_top3 と評価軸を統一するため、
    「このレースの年齢別平均勝率との相対差分」で補正量を算出する。

    具体的には:
      age_ratio  = この年齢の勝者数 / 全勝者数
      avg_ratio  = 1 / 出走年齢グループ数（均等配分の期待値）
      diff_ratio = age_ratio - avg_ratio（相対偏差）
      adj        = TREND_WEIGHTS["age"] * (diff_ratio / avg_ratio)

    例:
      全4年齢グループで5歳が5/10年優勝 → ratio=0.5, avg=0.25 → adj=+0.022
      7歳以上が0/10年優勝             → ratio=0.0, avg=0.25 → adj=-0.022
    """
    age_counts: Dict[str, int] = race_trend_10y.get("age") or {}
    age = feature.get("age")
    if age is None:
        return 0.0
    age_i = int(age)
    # bucket_age と同じ: 7歳以上はまとめる
    age_key = "7歳以上" if age_i >= 7 else f"{age_i}歳"
    if not age_counts:
        return 0.0

    total = sum(age_counts.values()) or 1
    count = age_counts.get(age_key, 0)
    age_ratio = count / total

    # 平均比率: 観測された年齢グループ数で均等配分
    n_groups = len([v for v in age_counts.values() if v > 0]) or 1
    avg_ratio = 1.0 / n_groups

    diff_ratio = age_ratio - avg_ratio
    # 相対偏差をスコア補正に変換（標準ルートの diff_top3 と同じ方向性）
    adj = TREND_WEIGHTS["age"] * (diff_ratio / avg_ratio)
    # 過大補正を防止
    return round(max(-0.025, min(TREND_WEIGHTS["age"], adj)), 5)


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
    if not feature:
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
    # このレース過去10年の勝者年齢分布と照合して補正。
    # 評価軸: 年齢グループ別の勝者比率 vs 均等配分期待値の相対差分。
    # _calc_age_delta_from_trend と同一ロジックで統一。
    age_counts: Dict[str, int] = race_trend_10y.get("age") or {}
    age = feature.get("age")
    age_delta = 0.0
    if age is not None and age_counts:
        age_i   = int(age)
        age_key = "7歳以上" if age_i >= 7 else f"{age_i}歳"
        total   = sum(age_counts.values()) or 1
        count   = age_counts.get(age_key, 0)
        age_ratio = count / total
        n_groups  = len([v for v in age_counts.values() if v > 0]) or 1
        avg_ratio = 1.0 / n_groups
        diff_ratio = age_ratio - avg_ratio
        age_delta  = TREND_WEIGHTS["age"] * (diff_ratio / avg_ratio)
        age_delta  = max(-0.025, min(TREND_WEIGHTS["age"], age_delta))
    delta += age_delta

    # ----- 前走クラス補正 -----
    # G1前走: 大幅加点 / G2・G3重賞前走: 小加点 / 条件戦前走: 減点
    prev_class = float(feature.get("prev_race_class_index") or 0.0)
    prev_name  = str(feature.get("prev_race_name") or "")
    if prev_class > 0:
        prev_delta = 0.0
        if prev_class >= 0.95:          # G1前走
            prev_delta = TREND_WEIGHTS["prev_class"]
            if "有馬記念" in prev_name or "天皇賞" in prev_name:
                prev_delta += 0.010     # 長距離G1組は特別加点
        elif prev_class >= 0.75:        # G2/G3重賞前走
            prev_delta = TREND_WEIGHTS["prev_class"] * 0.40
        elif prev_class < 0.60:         # OP未満（条件戦）前走
            prev_delta = -TREND_WEIGHTS["prev_class"] * 0.60
        delta += prev_delta

    # ----- 前走着順補正（G1 × 距離差 考慮） -----
    # 設計思想:
    #   前走 G1 かつ今走との距離差が 400m 以上 → 条件が大きく違うため大敗でもペナルティなし、
    #   むしろ「強い相手と戦った」証左として小加点する（メイショウタバル型のケース）。
    #   非 G1 重賞以下で 6 着以下 → 近走不振として減点。
    #   前走 1 着 / 3 着以内 → 上昇フラグとして加点。
    _prev_rank = feature.get("prev_rank")
    _pr_list   = feature.get("past_races") or []
    _pr0       = _pr_list[0] if _pr_list else {}
    _prev_dist = int(_pr0.get("distance") or 0)
    _curr_dist = int(feature.get("target_distance") or 0)
    _dist_diff = abs(_curr_dist - _prev_dist) if (_curr_dist and _prev_dist) else 0
    _prc       = float(feature.get("prev_race_class_index") or 0.0)

    if _prev_rank is not None:
        if _prev_rank == 1:
            delta += TREND_WEIGHTS["prev_rank"]
        elif _prev_rank <= 3:
            delta += TREND_WEIGHTS["prev_rank"] * 0.40
        elif _prev_rank >= 6:
            if _prc >= 0.95 and _dist_diff >= 400:
                # G1 かつ 400m 以上の距離差での大敗 → 条件差補正として加点
                delta += TREND_WEIGHTS["prev_rank"] * 0.25
            elif _prc < 0.90:
                # 非 G1 重賞以下で 6 着以下 → 近走不振として減点
                delta -= TREND_WEIGHTS["prev_rank"] * 0.50

    # ----- 前走コース補正 -----
    # 前走と今走が同一競馬場 → コース形状・馬場を把握済みとして加点。
    _prev_course = str(_pr0.get("course_name") or "")
    _curr_course = str(feature.get("target_course") or "")
    if _prev_course and _curr_course and _prev_course == _curr_course:
        delta += TREND_WEIGHTS["prev_course"]

    return round(delta, 5)


def apply_trend_adjustments(
    features: List[Dict[str, Any]],
    race_trend_10y: Dict[str, Any],
    condition_stats: Optional[Dict[str, Any]] = None,
    combo_condition_stats: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    全馬に傾向補正を適用して model_score を更新する。

    【ルート選択の優先順位】
    1. 標準ルート（condition_stats が非空の場合）:
       signal_judge.build_horse_signal_details + aggregate_signal_result を使用。
       走者個別の統計に基づく証拠ベース補正。サンプルサイズ信頼度・エスカレーションあり。
       combo_condition_stats が渡された場合はコンボシグナルも追加（60% 圧縮で二重計上を抑制）。

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
        from signal_judge import (
            build_horse_signal_details,
            build_horse_combo_signal_details,
            aggregate_signal_result,
        )
        for f in features:
            score_before  = float(f.get("model_score") or 0.0)
            details       = build_horse_signal_details(f, condition_stats)
            combo_details = build_horse_combo_signal_details(f, combo_condition_stats or {})
            all_details   = details + combo_details
            sig_result    = aggregate_signal_result(all_details)
            delta         = sig_result["total_trend_adjust"]

            # 年齢シグナルが neutral（サンプル不足等）だった場合は
            # race_trend_10y の年齢分布データをフォールバックとして補完する。
            age_detail = next(
                (d for d in details if d.get("factor") == "年齢"), None
            )
            age_signal_applied = age_detail is not None
            if not age_signal_applied and race_trend_10y:
                age_delta = _calc_age_delta_from_trend(f, race_trend_10y)
                if age_delta != 0.0:
                    delta += age_delta
                    f["age_delta_fallback"] = age_delta

            # --- デバッグフィールド ---
            f["model_score_before_trend"] = score_before
            f["combo_signal_count"] = len(combo_details)
            if age_detail:
                f["age_bucket"]        = age_detail.get("value")
                f["age_sample_size"]   = age_detail.get("sample_size")
                f["age_top3_rate"]     = age_detail.get("top3_rate")
                f["overall_top3_rate"] = age_detail.get("overall_top3_rate")
                f["age_diff_top3"]     = age_detail.get("diff_top3")
                f["age_signal"]        = age_detail.get("signal")
                f["age_adjustment"]    = age_detail.get("score_adjust")
            else:
                f["age_bucket"]        = f.get("age_bucket")  # 既存値保持
                f["age_sample_size"]   = None
                f["age_top3_rate"]     = None
                f["overall_top3_rate"] = None
                f["age_diff_top3"]     = None
                f["age_signal"]        = "neutral"
                f["age_adjustment"]    = f.get("age_delta_fallback", 0.0)

            f["trend_delta"]          = delta
            f["trend_signal_details"] = sig_result
            f["model_score"]          = round(score_before + delta, 6)
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
    p_top3_ai = _safe_place_prob(f)   # 0.12 + 1.75 * p1 をベース

    # ── JRA経験的3着内率とのブレンド ──────────────────────────────
    # 人気馬は経験的3着内率が高く信頼性があるため、AI推定と加重平均する。
    # AIが能力を高く評価している馬は自然にAI側の値が効く（p_top3_aiが大きくなる）。
    # AIが過小評価していても経験データが下支えするため、人気馬が過度に除外されなくなる。
    _pop_rank = int(f.get("popularity") or 99)
    _empirical = EMPIRICAL_TOP3_RATES.get(_pop_rank, EMPIRICAL_TOP3_DEFAULT)
    _blend_w   = EMPIRICAL_BLEND_WEIGHTS.get(_pop_rank, 0.0)  # 7番人気以下は経験ブレンドなし
    p_top3 = round(
        (1.0 - _blend_w) * p_top3_ai + _blend_w * _empirical, 4
    )

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
    - 妙味馬1頭 → +4%  / 危険馬1頭 → -7%
    - 結果を [0.75, 1.15] にクリップ
    （かつての -12% は人気馬を組み合わせから過剰に締め出す原因だったため緩和）
    """
    mult = 1.0
    for h in horse_names:
        if h in value_names:
            mult *= 1.04
        if h in danger_names:
            mult *= 0.93
    return round(min(1.15, max(0.75, mult)), 4)


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
        # _market_place_prob は max(0.05, ...) を保証するため mkt_prob > 0 は常に True。
        # else 分岐（MARKET_PLACE_MULT 使用）には現在到達しない。
        # _market_place_prob の実装が変わった場合のセーフティとして残している。
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
    top_n: int = 8,
    race_structure: Optional[Dict[str, Any]] = None,
    ev_table: Optional[List[Dict[str, Any]]] = None,
    forced_axis: Optional[str] = None,
    horse_roles: Optional[List[Dict[str, Any]]] = None,
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

    # ── ロールベースのプール構築 ───────────────────────────────────────
    # horse_roles が提供された場合: assign_roles の結果でプールを構成する
    # → fade 馬を除外し、head/axis を軸候補、himo をヒモ候補として分離
    # horse_roles が未提供の場合: 従来通り win_prob 上位 top_n を使用（後方互換）

    _AXIS_ROLES = {"head", "axis"}

    if horse_roles:
        _role_map: Dict[str, str] = {
            r["horse_name"]: r.get("role", "himo") for r in horse_roles
        }
        # fade 以外を全プールに入れる
        _all_non_fade = [
            dict(f) for f in features
            if _role_map.get(str(f.get("horse_name") or ""), "himo") != "fade"
        ]
        if not _all_non_fade:
            _all_non_fade = [dict(f) for f in features]  # fallback

        # 各馬に _role を付与
        for _f in _all_non_fade:
            _f["_role"] = _role_map.get(str(_f.get("horse_name") or ""), "himo")

        # 軸候補 / ヒモ候補 を分離
        axis_pool = [f for f in _all_non_fade if f["_role"] in _AXIS_ROLES]
        himo_pool = [f for f in _all_non_fade if f["_role"] == "himo"]

        # 軸候補が2頭未満の場合: win_prob 上位の himo を軸に昇格（買い目が成立しなくなるのを防ぐ）
        if len(axis_pool) < 2 and himo_pool:
            _himo_sorted = sorted(himo_pool, key=lambda x: -float(x.get("win_prob") or 0))
            _need = min(2 - len(axis_pool), len(_himo_sorted))
            for _pf in _himo_sorted[:_need]:
                _pf["_role"] = "axis"
            axis_pool = axis_pool + _himo_sorted[:_need]
            himo_pool = _himo_sorted[_need:]

        # forced_axis: 指定馬を軸プールに強制追加
        if forced_axis:
            _in_axis = any(str(f.get("horse_name") or "") == forced_axis for f in axis_pool)
            if not _in_axis:
                _fa_f = next(
                    (dict(f) for f in features if str(f.get("horse_name") or "") == forced_axis),
                    None,
                )
                if _fa_f:
                    _fa_f["_role"] = "axis"
                    axis_pool = [_fa_f] + axis_pool

        # 全プール（軸+ヒモ）を上限で絞る
        axis_pool = axis_pool[:top_n]
        himo_pool = himo_pool[:top_n]
        top_f     = axis_pool + himo_pool   # 単勝・複勝・馬連・ワイド用（以後も参照）
        trio_pool = top_f                   # 3連複も同じプール（role 制約で組み合わせを制限）

        # フォールバック: プールが3頭未満で3連複が組めない場合
        # fade 馬の中から p_top3（経験ブレンド済み）が高い馬を "himo_fallback" として追加
        if len(trio_pool) < 3:
            _all_names_in_pool = {str(f.get("horse_name") or "") for f in trio_pool}
            _fade_candidates = [
                {**dict(f), "_role": "himo"}
                for f in sorted(
                    features,
                    key=lambda x: -float(x.get("_pp_p_top3") or _safe_place_prob(x)),
                )
                if str(f.get("horse_name") or "") not in _all_names_in_pool
            ]
            _need = 3 - len(trio_pool)
            trio_pool = trio_pool + _fade_candidates[:_need]
            top_f     = trio_pool  # 複勝・馬連も拡張

    else:
        # ── 従来モード（role 情報なし）──────────────────────────────
        sorted_f = sorted(features, key=lambda x: float(x.get("win_prob") or 0.0), reverse=True)
        top_f    = [dict(f) for f in sorted_f[:top_n]]

        # 全馬に _role="axis" を付与（制約なし）
        for _f in top_f:
            _f["_role"] = "axis"

        # 強制軸追加
        if forced_axis:
            if not any(str(f.get("horse_name") or "") == forced_axis for f in top_f):
                _fa_f = next(
                    (dict(f) for f in features if str(f.get("horse_name") or "") == forced_axis),
                    None,
                )
                if _fa_f:
                    _fa_f["_role"] = "axis"
                    top_f = [_fa_f] + top_f[:top_n - 1]

        axis_pool = top_f
        himo_pool = []

        # 3連複プール: 人気上位馬を強制追加（従来互換）
        _top_f_names = {str(f.get("horse_name") or "") for f in top_f}
        _pop_sorted  = sorted(features, key=lambda x: float(x.get("popularity") or 99))
        _extra_trio: List[Dict[str, Any]] = []
        for _pf in _pop_sorted:
            if len(_extra_trio) >= 2:
                break
            if str(_pf.get("horse_name") or "") not in _top_f_names:
                _extra_trio.append({**dict(_pf), "_role": "himo"})
        trio_pool = top_f + _extra_trio

    # 着順分布推定値を付与（_pp_* キー）
    _enrich_placement_probs(top_f, race_structure)
    # trio_pool に top_f 以外の馬がいれば追加エンリッチ
    _top_f_set = {str(f.get("horse_name") or "") for f in top_f}
    _trio_extras = [f for f in trio_pool if str(f.get("horse_name") or "") not in _top_f_set]
    if _trio_extras:
        _enrich_placement_probs(_trio_extras, race_structure)

    # 人気上位2頭の馬名セット（EV閾値緩和用）
    _pop_sorted_all = sorted(features, key=lambda x: float(x.get("popularity") or 99))
    _top2_pop_names: Set[str] = {str(f.get("horse_name") or "") for f in _pop_sorted_all[:2]}

    rows: List[Dict[str, Any]] = []

    def _stable(f: Dict[str, Any]) -> float:
        return (float(f.get("consistency_index") or 0.5) + float(f.get("trend_index") or 0.5)) / 2.0

    def _is_axis(f: Dict[str, Any]) -> bool:
        return f.get("_role", "axis") in _AXIS_ROLES

    # 単勝: 軸役（head/axis）のみ対象
    for f in top_f:
        if not _is_axis(f):
            continue
        ev = calc_tansho_ev(f)
        if ev is not None:
            rows.append({
                "bet_type":    "単勝",
                "horses":      [str(f.get("horse_name") or "")],
                "ev":          ev,
                "ai_hit_prob": round(float(f.get("_pp_p1") or f.get("win_prob") or 0.0), 4),
                "stable_score": round(_stable(f), 4),
            })

    # 複勝: 全プール（ヒモも3着に来るので対象）
    for f in top_f:
        ev = calc_fukusho_ev(f)
        if ev is not None:
            rows.append({
                "bet_type":    "複勝",
                "horses":      [str(f.get("horse_name") or "")],
                "ev":          ev,
                "ai_hit_prob": round(float(f.get("_pp_p_top3") or _safe_place_prob(f)), 4),
                "stable_score": round(_stable(f), 4),
            })

    # 馬連: 少なくとも一方が軸役
    for fa, fb in combinations(top_f, 2):
        if not (_is_axis(fa) or _is_axis(fb)):
            continue  # ヒモ同士の馬連は生成しない
        ev = calc_umaren_ev_pair(fa, fb)
        if ev is not None:
            pa = float(fa.get("_pp_p1") or fa.get("win_prob") or 0.0)
            pb = float(fb.get("_pp_p1") or fb.get("win_prob") or 0.0)
            _axis_stable = round(_stable(fa) if pa >= pb else _stable(fb), 4)
            rows.append({
                "bet_type":    "馬連",
                "horses":      [str(fa.get("horse_name") or ""), str(fb.get("horse_name") or "")],
                "ev":          ev,
                "ai_hit_prob": round(_harville_umaren_hit(pa, pb), 4),
                "stable_score": _axis_stable,
            })

    # ワイド: 少なくとも一方が軸役
    for fa, fb in combinations(top_f, 2):
        if not (_is_axis(fa) or _is_axis(fb)):
            continue
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
                "stable_score": round(min(_stable(fa), _stable(fb)), 4),
            })

    # 3連複: 少なくとも1頭が軸役
    for fa, fb, fc in combinations(trio_pool, 3):
        if not any(_is_axis(f) for f in (fa, fb, fc)):
            continue  # ヒモ3頭の組み合わせは生成しない
        ev = calc_sanrenpuku_ev_trio(fa, fb, fc)
        if ev is not None:
            pa = float(fa.get("_pp_p_top3") or _safe_place_prob(fa))
            pb = float(fb.get("_pp_p_top3") or _safe_place_prob(fb))
            pc = float(fc.get("_pp_p_top3") or _safe_place_prob(fc))
            _trio_names = [
                str(fa.get("horse_name") or ""),
                str(fb.get("horse_name") or ""),
                str(fc.get("horse_name") or ""),
            ]
            rows.append({
                "bet_type":       "3連複",
                "horses":         _trio_names,
                "ev":             ev,
                "ai_hit_prob":    round(pa * pb * pc * TRIO_CORR, 4),
                "stable_score":   round(min(_stable(fa), _stable(fb), _stable(fc)), 4),
                "has_top2_popular": any(h in _top2_pop_names for h in _trio_names),
            })

    # ── 補正フェーズ ──────────────────────────────────────────────
    structure_type   = (race_structure or {}).get("structure_type", "標準型")
    favorable_style  = (race_structure or {}).get("favorable_style", "unknown")
    features_by_name = {str(f.get("horse_name") or ""): f for f in trio_pool}

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

    # 軸馬フィルタ: 指定された馬が含まれる組み合わせのみ残す
    if forced_axis:
        filtered = [r for r in rows if forced_axis in r.get("horses", [])]
        # フィルタ後にEVが基準を超える行が最低1件あれば採用、なければ全件返す
        if filtered:
            return filtered

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
    forced_axis: Optional[str] = None,
    horse_roles: Optional[List[Dict[str, Any]]] = None,
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

    # ── 常に構造型推奨を使用 ──────────────────────────────────────
    # LightGBMモデル未訓練の段階ではwin_probが不正確なため、
    # EV計算（= win_prob × odds）は推奨の根拠として信頼できない。
    # オッズ入力は危険馬・妙味馬の検出には活用するが、
    # 推奨馬券はレース構造 + horse_roles（役割）ベースで決定する。
    _ = has_odds  # オッズ有無は推奨経路に影響させない

    plan = _recommend_by_structure(
        features, ev_table, race_structure, bankroll, race_pace, horse_roles=horse_roles,
    )
    if not plan.get("skip") and "selection_detail" not in plan:
        stype = race_structure.get("structure_type", "標準型")
        has_odds_note = "（オッズ参照済み: 危険馬・妙味馬判定に使用）" if has_odds else ""
        import os as _os
        _model_trained = _os.path.exists("keiba_lgbm_model.txt")
        _ev_note = "LightGBMモデルによるEV計算を適用中" if _model_trained else "LightGBMモデル未訓練のためEV計算は参考値として表示"
        plan["selection_detail"] = {
            "why_bet_type":  f"レース構造（{stype}）から選定{has_odds_note}",
            "why_combo":     "AI上位馬・役割（head/axis/himo）・妙味馬を優先選択",
            "why_not_other": _ev_note,
            "best_by_type":  {},
        }
    return plan


def recommend_all_bet_types(
    features: List[Dict[str, Any]],
    ev_table: List[Dict[str, Any]],
    race_structure: Dict[str, Any],
    bankroll: int,
    race_pace: str = "medium",
    horse_roles: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    ワイド・馬連・三連複・三連単それぞれの推奨買い目を能力ベースで生成して返す。

    馬の選定は全券種共通（horse_roles の role 順 + win_prob 順）。
    オッズは使わない。bankroll は券種ごと独立。
    """
    from itertools import permutations as _perms

    # ── 共通: 軸候補構築（_recommend_by_structure と同じロジック） ──
    if horse_roles:
        _role_map = {r["horse_name"]: r.get("role", "himo") for r in horse_roles}
        _ROLE_ORDER = {"head": 0, "axis": 1, "himo": 2, "fade": 3}
        _sorted = sorted(
            features,
            key=lambda x: (
                _ROLE_ORDER.get(_role_map.get(str(x.get("horse_name") or ""), "himo"), 2),
                -float(x.get("win_prob") or 0.0),
            ),
        )
        _non_fade = [f for f in _sorted
                     if _role_map.get(str(f.get("horse_name") or ""), "himo") != "fade"]
        if len(_non_fade) < 2:
            _non_fade += [f for f in _sorted if f not in _non_fade][:2 - len(_non_fade)]
        candidates = _non_fade
    else:
        candidates = sorted(features, key=lambda x: float(x.get("win_prob") or 0.0), reverse=True)

    names = [str(f.get("horse_name") or "") for f in candidates if f.get("horse_name")][:4]
    if len(names) < 2:
        return []

    plans: List[Dict[str, Any]] = []
    n1, n2 = names[0], names[1]
    n3 = names[2] if len(names) >= 3 else None
    n4 = names[3] if len(names) >= 4 else None
    top3 = [n for n in [n1, n2, n3] if n]
    top4 = [n for n in [n1, n2, n3, n4] if n]

    # ── ワイドBOX（上位3頭, 3点） ──
    if len(top3) >= 3:
        combos = list(combinations(top3, 2))
        s = _round_stake(bankroll / len(combos))
        plans.append(_plan(
            "ワイドBOX", top3,
            [{"combination": list(c), "stake": s} for c in combos],
            f"能力上位3頭ワイドBOX {len(combos)}点", "低", "構造型",
        ))

    # ── 馬連（上位2頭, 1点） ──
    plans.append(_plan(
        "馬連", [n1, n2],
        [{"combination": [n1, n2], "stake": _round_stake(bankroll)}],
        f"{n1}〜{n2} 馬連", "低", "構造型",
    ))

    # ── 三連複BOX（上位4頭, 最大4点） ──
    if len(top4) >= 3:
        combos3 = list(combinations(top4, 3))[:MAX_BET_TICKETS]
        s = _round_stake(bankroll / len(combos3))
        plans.append(_plan(
            "三連複BOX", top4,
            [{"combination": list(c), "stake": s} for c in combos3],
            f"能力上位{len(top4)}頭 三連複BOX {len(combos3)}点", "中", "構造型",
        ))

    # ── 三連単（1着固定: names[0], 2・3着: names[1:3] の全順列） ──
    if n3:
        perms = list(_perms([n2, n3], 2))
        s = _round_stake(bankroll / len(perms))
        plans.append(_plan(
            "三連単", top3,
            [{"combination": [n1] + list(p), "stake": s} for p in perms],
            f"{n1} 1着固定 三連単 {len(perms)}通り", "高", "構造型",
        ))

    return plans


def _score_bet_plan(
    ev: float,
    ai_hit_prob: float,
    n_tickets: int,
    per_stake: int,
    stable_score: float = 0.5,
    bet_type: str = "",
) -> float:
    """
    買い目プランの複合スコア（EV × ヒット率重み × 点数ペナルティ × 安定性ペナルティ）。
    スコアが高いほど「再現性があり少額でも運用しやすい」プランと判断する。

    - per_stake が MIN_STAKE_PER_TICKET を下回るプランは即除外（-1.0）
    - ヒット率が高いほど微加点（arctan近似）
    - 点数が多いほどペナルティ
    - 単勝: stable_score < TANSHO_STABLE_MIN → ×0.88 ペナルティ（1着が必要なため安定性を重視）
    - 複勝: stable_score < FUKUSHO_STABLE_MIN → ×0.92 ペナルティ（3着圏内でよいため緩め）
    - 馬連: stable_score（軸馬） < UMAREN_AXIS_STABLE_MIN → ×0.90（軸不安定＝組み合わせ全崩れリスク）
    - ワイド/3連複: stable_score（min） < WIDE_AXIS_STABLE_MIN → ×0.93（複数馬依存のためやや緩め）
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
    if bet_type == "単勝" and stable_score < TANSHO_STABLE_MIN:
        stable_factor = 0.88
    elif bet_type == "複勝" and stable_score < FUKUSHO_STABLE_MIN:
        stable_factor = 0.92
    elif bet_type == "馬連" and stable_score < UMAREN_AXIS_STABLE_MIN:
        stable_factor = 0.90
    elif bet_type in ("ワイド", "3連複") and stable_score < WIDE_AXIS_STABLE_MIN:
        stable_factor = 0.93
    else:
        stable_factor = 1.0
    return ev * hit_weight * ticket_factor * stable_factor


def _select_best_plan(
    ticket_evs: List[Dict[str, Any]],
    bankroll: int,
) -> Optional[Dict[str, Any]]:
    """
    EVテーブルから最適プランを選定し {best_entry, group, per_stake} を返す。

    選定ルール:
    1. 券種別EV閾値（単勝/複勝 >= EV_SKIP_THRESHOLD、その他 >= EV_COMPOUND_SKIP）を満たす
    2. 同一券種内で EV 降順に最大 MAX_PRACTICAL_TICKETS 点まで集約（現在10点上限）
    3. per_stake >= MIN_STAKE_PER_TICKET を保証
    4. 複合スコア（_score_bet_plan）で券種を選択
       — 点数が多いほどペナルティ: ≤3点×1.00 / ≤5点×0.97 / >5点×0.93
    5. EV差が EV_TIE_MARGIN 以内の券種が複数あれば、少点数・低リスクを優先
    """
    def ev_threshold(r: Dict[str, Any]) -> float:
        base = EV_SKIP_THRESHOLD if r["bet_type"] in ("単勝", "複勝") else EV_COMPOUND_SKIP
        # 人気1〜2番を含む3連複は閾値を緩める（AIが過小評価しても市場人気馬を候補に残す）
        if r["bet_type"] == "3連複" and r.get("has_top2_popular"):
            base = min(base, 0.82)
        return base

    # 複勝は主推奨から除外（回収率上限が低く、ユーザーの期待に応えられないため）
    _EXCLUDE_AS_PRIMARY = {"複勝"}

    candidates = [
        r for r in ticket_evs
        if r["ev"] >= ev_threshold(r) and r["bet_type"] not in _EXCLUDE_AS_PRIMARY
    ]
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

        top_ev       = selected[0]["ev"]
        top_hit_prob = selected[0]["ai_hit_prob"]
        top_stable   = float(selected[0].get("stable_score") or 0.5)

        # half-Kelly による賭け金計算
        # f = KELLY_FRACTION × p × (EV-1) / (EV-p)
        # EV = p × payout → payout = EV/p, net_odds = EV/p - 1
        _p   = max(0.001, top_hit_prob)
        _ev  = top_ev
        _kelly_f = KELLY_FRACTION * _p * (_ev - 1.0) / max(0.001, _ev - _p)
        _kelly_f = max(KELLY_MIN_RATIO, min(KELLY_MAX_RATIO, _kelly_f))
        # 総配分を点数で均等割り、100円単位に丸め
        _total_kelly = bankroll * _kelly_f
        per_stake    = max(MIN_STAKE_PER_TICKET,
                           (int(_total_kelly / max(n, 1)) // BANKROLL_UNIT) * BANKROLL_UNIT)

        score = _score_bet_plan(top_ev, top_hit_prob, n, per_stake, top_stable, bet_type)

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

    # 3連複で人気1〜2番が1枚も含まれていない場合、末尾チケットと入れ替える
    if bet_type == "3連複" and not any(r.get("has_top2_popular") for r in group):
        _pop_candidates = [
            r for r in ticket_evs
            if r["bet_type"] == "3連複"
            and r.get("has_top2_popular")
            and r["ev"] >= 0.75
        ]
        if _pop_candidates:
            _best_pop = max(_pop_candidates, key=lambda x: x["ev"])
            # 末尾（最低EV）を置き換え、既存グループに重複がなければ差し込む
            _group_combos = [frozenset(r["horses"]) for r in group]
            if frozenset(_best_pop["horses"]) not in _group_combos:
                group = group[:-1] + [_best_pop]

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
    horse_roles: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """レース構造ベースの推奨買い目。horse_roles があれば消し馬を除外する。"""
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

    # 妙味馬を事前計算（himo層内での優先ソートに使用）
    value_horses = detect_value_horses(ev_table, features, race_pace)
    _value_names_set: Set[str] = {str(v.get("horse_name") or "") for v in value_horses}

    # horse_roles が渡された場合: 消し馬を除外し、head/axis を優先順位上位に
    # himo 層の妙味馬は通常 himo より優先（1.5）して相手候補の上位に来るようにする
    if horse_roles:
        _role_map = {r["horse_name"]: r.get("role", "himo") for r in horse_roles}
        _ROLE_ORDER = {"head": 0, "axis": 1, "himo": 2, "fade": 3}
        sorted_features = sorted(
            features,
            key=lambda x: (
                (1.5 if (
                    _role_map.get(str(x.get("horse_name") or ""), "himo") == "himo"
                    and str(x.get("horse_name") or "") in _value_names_set
                ) else _ROLE_ORDER.get(_role_map.get(str(x.get("horse_name") or ""), "himo"), 2)),
                -float(x.get("win_prob") or 0.0),
            ),
        )
        # 消し馬は除外（ただし非消し馬が2頭未満なら最高win_prob消し馬を追加）
        _non_fade = [f for f in sorted_features
                     if _role_map.get(str(f.get("horse_name") or ""), "himo") != "fade"]
        if len(_non_fade) < 2:
            _fade_sorted = [f for f in sorted_features if f not in _non_fade]
            _non_fade = _non_fade + _fade_sorted[:2 - len(_non_fade)]
        sorted_features = _non_fade
    else:
        sorted_features = sorted(
            features,
            key=lambda x: float(x.get("win_prob") or 0.0),
            reverse=True,
        )

    # horse_roles がある場合: fade除外はすでに sorted_features に反映済み。
    # オッズ由来の detect_danger_favorites_v2 を二重適用しない。
    # horse_roles がない場合（フォールバック）のみ、オッズベースの danger 除外を行う。
    if horse_roles:
        axis_candidates = sorted_features
    else:
        _danger_h = detect_danger_favorites_v2(ev_table, features, race_pace)
        _danger_n = {d["horse_name"] for d in _danger_h if d.get("is_truly_dangerous", True)}
        axis_candidates = [f for f in sorted_features if f.get("horse_name") not in _danger_n]
        if not axis_candidates:
            axis_candidates = sorted_features

    if not axis_candidates:
        return {**EMPTY, "skip": True, "skip_reason": "推奨できる馬がいません。見送り推奨。"}

    axis      = axis_candidates[0]
    axis_name = str(axis.get("horse_name") or "")

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
        box_names = []
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
        pool = []
        for f in axis_candidates:
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
    value_horses: List[Dict[str, Any]],  # 引数は互換性のため残す（未使用）
) -> Optional[str]:
    """軸の次の候補を返す。win_prob/role順のaxis_candidatesから選択（オッズ非依存）"""
    for f in axis_candidates[1:]:
        fn = str(f.get("horse_name") or "")
        if fn != axis_name:
            return fn
    return None


def _build_others(
    axis_candidates: List[Dict[str, Any]],
    axis_name: str,
    value_horses: List[Dict[str, Any]],  # 引数は互換性のため残す（未使用）
    max_count: int = 4,
) -> List[str]:
    """軸以外の相手馬リストを構築。win_prob/role順のaxis_candidatesから選択（オッズ非依存）"""
    result: List[str] = []
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
    # upside_score は能力ベースのみ（value_gap=オッズ由来を除外）
    upside_score = min(1.0, form_base)

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
    danger_horses: Optional[List[Dict[str, Any]]] = None,
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

    # 危険人気馬名セット（人気帯別乖離閾値を使用）
    # ここで使う popularity_rank は ev_table の動的順位（現在の win_odds 昇順）。
    # features["popularity"]（スクレイピング時の固定人気）ではない。
    danger_names: Set[str] = set()
    for row in ev_table:
        vg  = row.get("value_gap")
        wo  = row.get("win_odds")
        if vg is None or wo is None:
            continue
        if float(wo) > DANGER_ODDS_MAX:
            continue
        _pop = int(row.get("popularity_rank") or 99)
        _gap_thresh = (
            -DANGER_GAP_POP1 if _pop == 1 else
            -DANGER_GAP_POP3 if _pop <= 3 else
            -DANGER_GAP_POP6 if _pop <= 6 else
            -DANGER_GAP_MIN
        )
        if vg <= _gap_thresh:
            danger_names.add(row["horse_name"])

    # is_truly_dangerous マップ
    # danger_horses 未提供時は False（能力チェックなしで fade にしない）。
    # 通常フロー（recommend_bet_plan）は detect_danger_favorites_v3 の結果を必ず渡す。
    _truly_map: Dict[str, bool] = {
        d["horse_name"]: bool(d.get("is_truly_dangerous", True))
        for d in (danger_horses or [])
    }
    _truly_default = False  # danger_horses 未提供時: 能力証拠なしで fade にしない

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

        _is_truly_danger = _truly_map.get(name, _truly_default) if name in danger_names else False
        # _pop は features["popularity"]（スクレイピング時点の固定アンカー）。
        # ev_table の popularity_rank（動的）とはオッズ更新後にズレる場合がある。
        # 危険軟化条件（下記）はこの乖離があるときに意味を持つ設計。
        _pop = int(f.get("popularity") or 99)
        _ev_vg = float((ev_by_name.get(name) or {}).get("value_gap") or 0.0)

        # 1〜3番人気の危険判定軟化条件。
        #
        # なぜ必要か:
        #   danger_names は ev_table の popularity_rank（現在の win_odds 昇順）で構築する。
        #   _pop は features の popularity フィールド（スクレイピング時点の人気）。
        #   オッズ更新により両者がズレる場合がある。例:
        #     スクレイピング時 2番人気(popularity=2) → オッズ更新後 5番人気(popularity_rank=5)
        #     → danger_names の閾値は -0.05（4〜6番人気）→ vg=-0.06 で入る
        #     → _pop=2, _ev_vg=-0.06 > -0.08 → この条件が発動し、fadeではなくhimoに軟化
        #
        # 発動しないケース（データに乖離がない通常フロー）:
        #   popularity と popularity_rank が一致している場合、pop=1-3 で danger_names 入りには
        #   vg <= -0.10 or -0.15 が必要となり、_ev_vg > -0.08 は常に False になる。
        #
        # DANGER_GAP_POP1/POP3 を変更する場合はこの条件との整合を確認すること。
        if name in danger_names and _is_truly_danger and _pop <= 3 and _ev_vg > -0.08:
            # 危険判定を軟化: fadeではなくhimoに留める
            _is_truly_danger = False

        if name in danger_names and _is_truly_danger and "FAVORITE_LOW_TRUST" not in jockey_codes:
            # 真に危険な人気馬（消し）
            role   = "fade"
            reason = "危険人気馬（AI<市場評価）"
        elif name in danger_names and _is_truly_danger and "FAVORITE_LOW_TRUST" in jockey_codes:
            role   = "fade"
            reason = "危険人気馬（AI<市場評価）・" + (jockey_summary or "人気馬信頼度低")
        elif name in danger_names and not _is_truly_danger:
            # 相手残り危険馬: 1着は怪しいが3着残りの可能性あり → himo 上限で判定
            if top3_p >= (TOP3_HIMO_MIN + himo_threshold_adj):
                role   = "himo"
                reason = "相手残り（上位人気・3着内残り）"
                if jockey_summary and "LONGSHOT_UPSIDE" in jockey_codes:
                    reason = reason + "・" + jockey_summary
            else:
                role   = "fade"
                reason = "相手残り候補だが3着内確率不足"
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

    # ── 大頭数フォールバック ─────────────────────────────────────────
    # 全馬/ほぼ全馬がfadeになった場合（大頭数レースで絶対閾値を超えられないケース）、
    # 勝率上位から相対評価で head/axis/himo を強制割り当てする。
    n_non_fade = sum(1 for r in results if r["role"] != "fade")
    if n_non_fade < 3:
        _sorted = sorted(results, key=lambda r: -r["win_prob"])
        _rank_role = {0: "head", 1: "axis", 2: "himo", 3: "himo"}
        _assigned  = 0
        for _r in _sorted:
            if _r["role"] == "fade" and _assigned in _rank_role:
                _r["role"]   = _rank_role[_assigned]
                _r["reason"] = f"相対評価{_assigned+1}位（大頭数フィールド自動調整）"
                _assigned   += 1
            elif _r["role"] != "fade":
                _assigned   += 1
            if _assigned >= 4:
                break

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
        # 人気帯別乖離閾値: 実力馬ほど AI と市場の差が構造的に大きくなるため閾値を引き上げる
        _pop = int(row.get("popularity_rank") or 99)
        _gap_thresh = (
            -DANGER_GAP_POP1 if _pop == 1 else
            -DANGER_GAP_POP3 if _pop <= 3 else
            -DANGER_GAP_POP6 if _pop <= 6 else
            -danger_gap_min
        )
        if vg >= _gap_thresh:
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
        # 展開不利でも top3_prob が高い馬は「相手残り」扱い — 実力馬を過剰に消さない
        if pace_conflict and top3_p < DANGER_V3_TOP3_MAX:
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


# =========================================================
# AI馬券師推奨機能 — 全券種・1枚100円固定
# =========================================================

BETMASTER_TICKET_UNIT: int = 100  # 1枚100円固定

# 自信度閾値
CONFIDENCE_TANSHO: float = 0.60    # 単勝: head の stable_score
CONFIDENCE_FUKUSHO: float = 0.50   # 複勝: head の top3_prob
CONFIDENCE_BATAN: float = 0.65     # 馬単・三連単: head の stable_score（厳しめ）


def _bm_stable(f: Dict[str, Any]) -> float:
    """stable_score = (consistency_index + trend_index) / 2"""
    return (float(f.get("consistency_index") or 0.5) + float(f.get("trend_index") or 0.5)) / 2.0


def _bm_sorted_candidates(
    features: List[Dict[str, Any]],
    horse_roles: Optional[List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    horse_roles に従って馬をソートし、role_map も返す。
    horse_roles が None の場合は win_prob 順で自動割当。
    fade 馬は non_fade リストから除外する。
    """
    _ROLE_ORDER = {"head": 0, "axis": 1, "himo": 2, "fade": 3}

    if horse_roles:
        role_map: Dict[str, str] = {r["horse_name"]: r.get("role", "himo") for r in horse_roles}
        sorted_f = sorted(
            features,
            key=lambda x: (
                _ROLE_ORDER.get(role_map.get(str(x.get("horse_name") or ""), "himo"), 2),
                -float(x.get("win_prob") or 0.0),
            ),
        )
        non_fade = [f for f in sorted_f
                    if role_map.get(str(f.get("horse_name") or ""), "himo") != "fade"]
    else:
        sorted_f = sorted(features, key=lambda x: -float(x.get("win_prob") or 0.0))
        non_fade = sorted_f
        role_map = {}
        for i, f in enumerate(non_fade):
            name = str(f.get("horse_name") or "")
            if i == 0:
                role_map[name] = "head"
            elif i <= 2:
                role_map[name] = "axis"
            else:
                role_map[name] = "himo"

    if len(non_fade) < 2:
        fade_horses = [f for f in sorted_f if f not in non_fade]
        to_add = fade_horses[:2 - len(non_fade)]
        for f in to_add:
            name = str(f.get("horse_name") or "")
            role_map[name] = "himo"
        non_fade = non_fade + to_add

    return non_fade, role_map


def _bm_formation_trio_tickets(
    leg1: List[str], leg2: List[str], leg3: List[str]
) -> List[Dict[str, Any]]:
    """
    三連複フォーメーション: leg1/leg2/leg3 から3頭の組み合わせを生成。
    各レグから少なくとも1頭を含み、重複しないユニークな3頭組み合わせを返す。
    """
    seen: Set[frozenset] = set()
    tickets: List[Dict[str, Any]] = []
    for a in leg1:
        for b in leg2:
            for c in leg3:
                combo = frozenset([a, b, c])
                if len(combo) == 3 and combo not in seen:
                    seen.add(combo)
                    tickets.append({"combination": sorted([a, b, c]), "stake": BETMASTER_TICKET_UNIT})
    return tickets


def _bm_formation_trifecta_tickets(
    leg1: List[str], leg2: List[str], leg3: List[str]
) -> List[Dict[str, Any]]:
    """
    三連単フォーメーション: (1着, 2着, 3着) の順列を生成。
    各馬は1回のみ使用。
    """
    seen: Set[Tuple[str, str, str]] = set()
    tickets: List[Dict[str, Any]] = []
    for a in leg1:
        for b in leg2:
            for c in leg3:
                if len({a, b, c}) == 3 and (a, b, c) not in seen:
                    seen.add((a, b, c))
                    tickets.append({"combination": [a, b, c], "stake": BETMASTER_TICKET_UNIT})
    return tickets


def _bm_plan(
    bet_type: str,
    formation_legs: Optional[Dict[str, List[str]]],
    tickets: List[Dict[str, Any]],
    risk_level: str,
    reason: str,
    confidence_ok: bool,
    no_pick_reason: str,
    confidence_score: float,
) -> Dict[str, Any]:
    """AI馬券師プランの標準辞書を返す。tickets が空の場合は confidence_ok を False に矯正する。"""
    count = len(tickets)
    _ok = confidence_ok and count > 0
    _reason = no_pick_reason if _ok or not confidence_ok else "買い目が生成できませんでした"
    return {
        "bet_type":         bet_type,
        "formation_legs":   formation_legs,
        "tickets":          tickets,
        "ticket_count":     count,
        "budget":           count * BETMASTER_TICKET_UNIT,
        "risk_level":       risk_level,
        "reason":           reason,
        "confidence_ok":    _ok,
        "no_pick_reason":   _reason,
        "confidence_score": confidence_score,
    }


def recommend_betmaster_plans(
    features: List[Dict[str, Any]],
    race_structure: Dict[str, Any],
    horse_roles: Optional[List[Dict[str, Any]]] = None,
    race_pace: str = "medium",
) -> List[Dict[str, Any]]:
    """
    全券種（単勝・複勝・ワイド・馬連・馬単・三連複×2・三連単×2）を評価し、
    AI馬券師として1枚100円固定で推奨する買い目リストを返す。

    各プランに confidence_ok フラグを付与し、
    自信がない券種は tickets=[] / confidence_ok=False / no_pick_reason=理由 を返す。
    """
    non_fade, role_map = _bm_sorted_candidates(features, horse_roles)
    if not non_fade:
        return []

    head_horses = [f for f in non_fade if role_map.get(str(f.get("horse_name") or ""), "himo") == "head"]
    axis_horses = [f for f in non_fade if role_map.get(str(f.get("horse_name") or ""), "himo") == "axis"]
    himo_horses = [f for f in non_fade if role_map.get(str(f.get("horse_name") or ""), "himo") == "himo"]

    if not head_horses:
        head_horses = non_fade[:1]
    if not axis_horses:
        axis_horses = non_fade[1:3]
    if not himo_horses:
        himo_horses = non_fade[3:]

    head = head_horses[0]
    head_name = str(head.get("horse_name") or "")
    head_stable = _bm_stable(head)

    pp = estimate_placement_probs(head, race_structure)
    head_top3_prob = pp["p_top3"]

    axis_names = [str(f.get("horse_name") or "") for f in axis_horses[:3]
                  if str(f.get("horse_name") or "") != head_name]
    himo_names = [str(f.get("horse_name") or "") for f in himo_horses
                  if str(f.get("horse_name") or "") != head_name]
    all_names = [str(f.get("horse_name") or "") for f in non_fade]

    plans: List[Dict[str, Any]] = []

    # ── 1. 単勝 ───────────────────────────────────────────────────────────
    ok_tansho = head_stable >= CONFIDENCE_TANSHO
    plans.append(_bm_plan(
        bet_type="単勝",
        formation_legs=None,
        tickets=[{"combination": [head_name], "stake": BETMASTER_TICKET_UNIT}] if ok_tansho else [],
        risk_level="低",
        reason=f"{head_name} の安定指数 {head_stable:.2f}（閾値{CONFIDENCE_TANSHO}）",
        confidence_ok=ok_tansho,
        no_pick_reason="" if ok_tansho else f"軸馬の安定指数不足（{head_stable:.2f} < {CONFIDENCE_TANSHO}）",
        confidence_score=head_stable,
    ))

    # ── 2. 複勝 ───────────────────────────────────────────────────────────
    ok_fukusho = head_top3_prob >= CONFIDENCE_FUKUSHO
    plans.append(_bm_plan(
        bet_type="複勝",
        formation_legs=None,
        tickets=[{"combination": [head_name], "stake": BETMASTER_TICKET_UNIT}] if ok_fukusho else [],
        risk_level="最低",
        reason=f"{head_name} の3着圏内確率 {head_top3_prob:.2f}（閾値{CONFIDENCE_FUKUSHO}）",
        confidence_ok=ok_fukusho,
        no_pick_reason="" if ok_fukusho else f"3着圏内確率不足（{head_top3_prob:.2f} < {CONFIDENCE_FUKUSHO}）",
        confidence_score=head_top3_prob,
    ))

    # ── 3. ワイド ────────────────────────────────────────────────────────
    wide_legs = list(dict.fromkeys([head_name] + axis_names[:2]))  # 重複除去・順序保持
    wide_combos = list(combinations(wide_legs, 2))
    wide_tickets = [{"combination": list(c), "stake": BETMASTER_TICKET_UNIT} for c in wide_combos]
    plans.append(_bm_plan(
        bet_type="ワイド",
        formation_legs={"組み合わせ": wide_legs},
        tickets=wide_tickets,
        risk_level="低",
        reason=f"能力上位{len(wide_legs)}頭のワイドBOX",
        confidence_ok=True,
        no_pick_reason="",
        confidence_score=head_stable,
    ))

    # ── 4. 馬連（流し）────────────────────────────────────────────────────
    umaren_partners = axis_names + [str(f.get("horse_name") or "") for f in himo_horses[:3]]
    umaren_partners = [n for n in umaren_partners if n != head_name][:5]
    umaren_tickets = [{"combination": sorted([head_name, p]), "stake": BETMASTER_TICKET_UNIT}
                      for p in umaren_partners]
    plans.append(_bm_plan(
        bet_type="馬連（流し）",
        formation_legs={"軸": [head_name], "相手": umaren_partners},
        tickets=umaren_tickets,
        risk_level="低",
        reason=f"{head_name} 軸・馬連流し {len(umaren_tickets)}点",
        confidence_ok=True,
        no_pick_reason="",
        confidence_score=head_stable,
    ))

    # ── 5. 馬単フォーメーション ──────────────────────────────────────────
    ok_batan = head_stable >= CONFIDENCE_BATAN
    batan_partners = axis_names[:3]
    batan_tickets = [{"combination": [head_name, p], "stake": BETMASTER_TICKET_UNIT}
                     for p in batan_partners] if ok_batan else []
    plans.append(_bm_plan(
        bet_type="馬単フォーメーション",
        formation_legs={"1着": [head_name], "2着": batan_partners} if batan_partners else None,
        tickets=batan_tickets,
        risk_level="中",
        reason=f"{head_name} 1着固定・馬単 {len(batan_tickets)}点",
        confidence_ok=ok_batan,
        no_pick_reason="" if ok_batan else f"1着固定の確度不足（安定指数 {head_stable:.2f} < {CONFIDENCE_BATAN}）",
        confidence_score=head_stable,
    ))

    # ── 6. 三連複フォーメーション（AI絞り）────────────────────────────────
    trio_leg1 = [head_name]
    trio_leg2 = axis_names[:2]
    trio_leg3_ai = [str(f.get("horse_name") or "") for f in
                    sorted(himo_horses, key=lambda x: -float(x.get("win_prob") or 0.0))[:6]]
    trio_ai_tickets = _bm_formation_trio_tickets(trio_leg1, trio_leg2, trio_leg3_ai)
    plans.append(_bm_plan(
        bet_type="三連複フォーメーション（AI絞り）",
        formation_legs={"馬1": trio_leg1, "馬2": trio_leg2, "馬3": trio_leg3_ai},
        tickets=trio_ai_tickets,
        risk_level="中",
        reason=f"馬1:{head_name} / 馬2:{len(trio_leg2)}頭 / 馬3:AI絞り{len(trio_leg3_ai)}頭",
        confidence_ok=True,
        no_pick_reason="",
        confidence_score=head_stable,
    ))

    # ── 7. 三連複フォーメーション（全頭）─────────────────────────────────
    trio_leg3_all = [n for n in all_names if n != head_name and n not in trio_leg2]
    trio_all_tickets = _bm_formation_trio_tickets(trio_leg1, trio_leg2, trio_leg3_all)
    plans.append(_bm_plan(
        bet_type="三連複フォーメーション（全頭）",
        formation_legs={"馬1": trio_leg1, "馬2": trio_leg2, "馬3": trio_leg3_all},
        tickets=trio_all_tickets,
        risk_level="中",
        reason=f"馬1:{head_name} / 馬2:{len(trio_leg2)}頭 / 馬3:全{len(trio_leg3_all)}頭",
        confidence_ok=True,
        no_pick_reason="",
        confidence_score=head_stable,
    ))

    # ── 8. 三連単フォーメーション（AI絞り）────────────────────────────────
    trifecta_leg1 = [head_name]
    trifecta_leg2 = axis_names[:3]
    trifecta_leg3_ai = [str(f.get("horse_name") or "") for f in
                        sorted(himo_horses, key=lambda x: -float(x.get("win_prob") or 0.0))[:6]]
    trifecta_ai_tickets = _bm_formation_trifecta_tickets(
        trifecta_leg1, trifecta_leg2, trifecta_leg3_ai
    ) if ok_batan else []
    plans.append(_bm_plan(
        bet_type="三連単フォーメーション（AI絞り）",
        formation_legs={"1着": trifecta_leg1, "2着": trifecta_leg2, "3着": trifecta_leg3_ai},
        tickets=trifecta_ai_tickets,
        risk_level="高",
        reason=f"1着:{head_name} / 2着:{len(trifecta_leg2)}頭 / 3着:AI絞り{len(trifecta_leg3_ai)}頭",
        confidence_ok=ok_batan,
        no_pick_reason="" if ok_batan else f"1着固定の確度不足（安定指数 {head_stable:.2f} < {CONFIDENCE_BATAN}）",
        confidence_score=head_stable,
    ))

    # ── 9. 三連単フォーメーション（全頭）─────────────────────────────────
    trifecta_leg3_all = [n for n in all_names if n != head_name and n not in trifecta_leg2]
    trifecta_all_tickets = _bm_formation_trifecta_tickets(
        trifecta_leg1, trifecta_leg2, trifecta_leg3_all
    ) if ok_batan else []
    plans.append(_bm_plan(
        bet_type="三連単フォーメーション（全頭）",
        formation_legs={"1着": trifecta_leg1, "2着": trifecta_leg2, "3着": trifecta_leg3_all},
        tickets=trifecta_all_tickets,
        risk_level="高",
        reason=f"1着:{head_name} / 2着:{len(trifecta_leg2)}頭 / 3着:全{len(trifecta_leg3_all)}頭",
        confidence_ok=ok_batan,
        no_pick_reason="" if ok_batan else f"1着固定の確度不足（安定指数 {head_stable:.2f} < {CONFIDENCE_BATAN}）",
        confidence_score=head_stable,
    ))

    return plans


def select_primary_betmaster(
    plans: List[Dict[str, Any]],
    race_structure: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    recommend_betmaster_plans() の結果から主推奨を1つ選定して返す。
    confidence_ok=True の中からレース構造に基づいてスコアリングし最高値を返す。
    全て confidence_ok=False の場合は None を返す。
    """
    structure_type = (race_structure or {}).get("structure_type", "標準型")

    _BASE_PRIORITY: Dict[str, float] = {
        "単勝":                         0.50,
        "複勝":                         0.20,
        "ワイド":                        0.40,
        "馬連（流し）":                   0.55,
        "馬単フォーメーション":            0.60,
        "三連複フォーメーション（AI絞り）":  0.80,
        "三連複フォーメーション（全頭）":    0.65,
        "三連単フォーメーション（AI絞り）":  0.70,
        "三連単フォーメーション（全頭）":    0.55,
    }

    _STRUCTURE_BONUS: Dict[str, Dict[str, float]] = {
        "本命信頼型": {
            "馬単フォーメーション":            0.20,
            "三連単フォーメーション（AI絞り）":  0.15,
            "馬連（流し）":                   0.10,
        },
        "標準型": {
            "三連複フォーメーション（AI絞り）":  0.15,
            "馬連（流し）":                   0.05,
        },
        "1強相手混戦型": {
            "三連複フォーメーション（AI絞り）":  0.10,
            "三連複フォーメーション（全頭）":    0.15,
        },
        "混戦型": {
            "三連複フォーメーション（全頭）":    0.20,
            "ワイド":                        0.10,
        },
        "波乱型": {
            "ワイド":                        0.25,
            "複勝":                         0.15,
            "三連複フォーメーション（全頭）":    0.10,
        },
        "差し届く型": {
            "三連複フォーメーション（AI絞り）":  0.10,
            "三連単フォーメーション（AI絞り）":  0.10,
        },
    }

    bonus_map = _STRUCTURE_BONUS.get(structure_type, {})

    best_plan: Optional[Dict[str, Any]] = None
    best_score: float = -1.0

    for plan in plans:
        if not plan.get("confidence_ok"):
            continue
        if not plan.get("tickets"):
            continue

        base = _BASE_PRIORITY.get(plan["bet_type"], 0.4)
        bonus = bonus_map.get(plan["bet_type"], 0.0)
        ticket_penalty = min(0.20, plan["ticket_count"] / 500.0)
        score = base + bonus - ticket_penalty

        if score > best_score:
            best_score = score
            best_plan = plan

    return best_plan


# =========================================================
# v2 買い目生成: 期待値・乖離ベース (bet_recommendations_v2)
# =========================================================
# 既存ロジック（recommend_bet_plan 系）は無変更。
# v2 は axis_ban / partner_ban で市場乖離の大きい馬を事前除外し、
# 軸スコア(axis_score_v2) / 相手スコア(partner_score_v2) で馬を評価する。
#
# 主な変更点:
#   - axis_score_v2: ai_win_prob + top2/3_prob + market_win_prob + prob_gap + win_ev + trend_delta
#   - partner_score_v2: top3_prob + prob_gap + win_ev + trend_delta
#   - axis_ban: prob_gap <= V2_AXIS_BAN_GAP の馬は軸に使わない
#   - partner_ban: prob_gap <= V2_PARTNER_BAN_GAP かつ win_ev 低い馬は相手にも使わない
#   - 券種ごとに独立した関数で生成
#   - race-level warning: 人気上位の危険馬が複数いる場合にアラートを出す
# =========================================================


def _v2_norm_gap(prob_gap: float) -> float:
    """prob_gap を [0,1] に正規化（-0.20 → 0, 0 → 0.5, +0.20 → 1）"""
    return min(1.0, max(0.0, (prob_gap + 0.20) / 0.40))


def _v2_norm_ev(win_ev: float) -> float:
    """win_ev を [0,1] に正規化（0.5 → 0, 1.5 → 1）"""
    return min(1.0, max(0.0, (win_ev - 0.5) / 1.0))


def _v2_norm_trend(trend_delta: float) -> float:
    """trend_delta を [0,1] に正規化（-0.10 → 0, 0 → 0.5, +0.10 → 1）"""
    return min(1.0, max(0.0, (trend_delta + 0.10) / 0.20))


def _enrich_horses_v2(
    features: List[Dict[str, Any]],
    ev_table: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    各馬に v2 スコアとban フラグを付与して返す。

    追加フィールド:
        axis_score_v2  : 軸適性スコア（0〜1）
        axis_ban       : True = 軸不可（prob_gap 過大マイナス）
        partner_score_v2 : 相手適性スコア（0〜1）
        partner_ban    : True = 相手不可（gap + EV 両方低い）
        prob_gap_v2    : ev_table から取得した prob_gap（=value_gap）
        win_ev_v2      : features の win_ev（またはフォールバック値）
        trend_delta_v2 : features の trend_delta
    """
    ev_by_name: Dict[str, Dict[str, Any]] = {
        row["horse_name"]: row for row in (ev_table or [])
    }

    enriched: List[Dict[str, Any]] = []
    for f in features:
        h = dict(f)  # 元の feature を壊さないコピー
        name      = str(h.get("horse_name") or "")
        ev_row    = ev_by_name.get(name) or {}

        # ── 基礎値取得 ──────────────────────────────────────────
        ai_win_prob     = float(h.get("ai_win_prob") or h.get("win_prob") or 0.0)
        market_win_prob = float(ev_row.get("market_win_prob") or 0.0)
        prob_gap        = float(ev_row.get("value_gap") or 0.0)   # = ai_win_prob - market
        _win_ev_raw     = h.get("win_ev")
        if _win_ev_raw is not None:
            win_ev = float(_win_ev_raw)
        else:
            # win_ev が未保存の場合は win_prob × win_odds で再計算
            _wp  = float(h.get("win_prob") or h.get("ai_win_prob") or 0.0)
            _wo  = float(h.get("win_odds") or 0.0)
            win_ev = round(_wp * _wo, 4) if _wp > 0 and _wo > 0 else 0.0
        trend_delta     = float(h.get("trend_delta") or 0.0)

        # top2_prob / top3_prob: 既に feature に入っている場合はそれを使う、なければ推定
        top2_prob = float(h.get("top2_prob") or 0.0)
        top3_prob = float(h.get("top3_prob") or 0.0)
        if top2_prob == 0.0 or top3_prob == 0.0:
            _race_structure: Dict[str, Any] = {}
            _pp = estimate_placement_probs(h, _race_structure)
            top2_prob = top2_prob or _pp.get("p_top2", 0.0)
            top3_prob = top3_prob or _pp.get("p_top3", 0.0)

        # ── 正規化 ────────────────────────────────────────────
        gap_norm   = _v2_norm_gap(prob_gap)
        ev_norm    = _v2_norm_ev(win_ev)
        trend_norm = _v2_norm_trend(trend_delta)

        # ── axis_score_v2 ─────────────────────────────────────
        axis_score_v2 = (
            ai_win_prob    * V2_AXIS_W_WIN_PROB
            + top2_prob    * V2_AXIS_W_TOP2
            + top3_prob    * V2_AXIS_W_TOP3
            + market_win_prob * V2_AXIS_W_MARKET
            + gap_norm     * V2_AXIS_W_GAP
            + ev_norm      * V2_AXIS_W_EV
            + trend_norm   * V2_AXIS_W_TREND
        )
        axis_score_v2 = round(min(1.0, axis_score_v2), 4)

        # ── partner_score_v2 ──────────────────────────────────
        partner_score_v2 = (
            top3_prob      * V2_PARTNER_W_TOP3
            + gap_norm     * V2_PARTNER_W_GAP
            + ev_norm      * V2_PARTNER_W_EV
            + trend_norm   * V2_PARTNER_W_TREND
        )
        partner_score_v2 = round(min(1.0, partner_score_v2), 4)

        # ── ban フラグ ────────────────────────────────────────
        axis_ban    = prob_gap <= V2_AXIS_BAN_GAP
        partner_ban = (prob_gap <= V2_PARTNER_BAN_GAP) and (win_ev <= V2_PARTNER_BAN_EV)

        h.update({
            "axis_score_v2":    axis_score_v2,
            "axis_ban":         axis_ban,
            "partner_score_v2": partner_score_v2,
            "partner_ban":      partner_ban,
            "prob_gap_v2":      round(prob_gap, 4),
            "win_ev_v2":        round(win_ev, 4),
            "trend_delta_v2":   round(trend_delta, 4),
        })
        enriched.append(h)

    return enriched


def _gen_tansho_v2(
    enriched: List[Dict[str, Any]],
    bankroll: int,
) -> Dict[str, Any]:
    """
    単勝 v2: axis_ban=False かつ win_ev >= V2_TANSHO_EV_MIN の馬を
    axis_score_v2 降順で最大1頭推奨。
    """
    candidates = [
        h for h in enriched
        if not h.get("axis_ban")
        and float(h.get("win_ev_v2") or 0.0) >= V2_TANSHO_EV_MIN
    ]
    candidates.sort(key=lambda x: float(x.get("axis_score_v2") or 0.0), reverse=True)

    if not candidates:
        return {"bet_type": "単勝", "tickets": [], "skip": True, "skip_reason": "EV>=1.0 の軸候補なし"}

    top = candidates[0]
    name = str(top.get("horse_name") or "")
    ev   = float(top.get("win_ev_v2") or 0.0)
    gap  = float(top.get("prob_gap_v2") or 0.0)
    stake = max(100, (bankroll // 10) // 100 * 100)
    return {
        "bet_type":  "単勝",
        "horses":    [name],
        "tickets":   [{"combination": [name], "stake": stake}],
        "total_stake": stake,
        "ticket_count": 1,
        "reason":    f"{name} EV={ev:.2f} gap={gap:+.2f}pt",
        "skip":      False,
    }


def _gen_fukusho_v2(
    enriched: List[Dict[str, Any]],
    bankroll: int,
) -> Dict[str, Any]:
    """
    複勝 v2: partner_ban=False の馬を partner_score_v2 降順で最大2頭推奨。
    EV 要件なし（複勝はEV計算精度が低いため）。
    """
    candidates = [h for h in enriched if not h.get("partner_ban")]
    candidates.sort(key=lambda x: float(x.get("partner_score_v2") or 0.0), reverse=True)
    picks = candidates[:2]

    if not picks:
        return {"bet_type": "複勝", "tickets": [], "skip": True, "skip_reason": "partner_ban 除外後に候補なし"}

    stake_each = max(100, (bankroll // (len(picks) * 10)) // 100 * 100)
    tickets = [{"combination": [str(h.get("horse_name") or "")], "stake": stake_each} for h in picks]
    names   = [str(h.get("horse_name") or "") for h in picks]
    return {
        "bet_type":  "複勝",
        "horses":    names,
        "tickets":   tickets,
        "total_stake": stake_each * len(picks),
        "ticket_count": len(picks),
        "reason":    f"partner_score上位: {' / '.join(names)}",
        "skip":      False,
    }


def _gen_wide_v2(
    enriched: List[Dict[str, Any]],
    bankroll: int,
) -> Dict[str, Any]:
    """
    ワイド v2: partner_ban=False の馬を partner_score_v2 降順で上位3頭選出し
    組み合わせを作成。ただし両馬ともprob_gap <= V2_WIDE_BAN_BOTH_GAP の組は除外。
    """
    candidates = [h for h in enriched if not h.get("partner_ban")]
    candidates.sort(key=lambda x: float(x.get("partner_score_v2") or 0.0), reverse=True)
    pool = candidates[:4]

    if len(pool) < 2:
        return {"bet_type": "ワイド", "tickets": [], "skip": True, "skip_reason": "ワイド候補が2頭未満"}

    valid_pairs = []
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            a, b = pool[i], pool[j]
            gap_a = float(a.get("prob_gap_v2") or 0.0)
            gap_b = float(b.get("prob_gap_v2") or 0.0)
            # 両馬ともマイナス乖離大の組み合わせは除外
            if gap_a <= V2_WIDE_BAN_BOTH_GAP and gap_b <= V2_WIDE_BAN_BOTH_GAP:
                continue
            valid_pairs.append((a, b))

    if not valid_pairs:
        return {"bet_type": "ワイド", "tickets": [], "skip": True, "skip_reason": "有効なワイドペアなし（両馬マイナス乖離）"}

    stake_each = max(100, (bankroll // len(valid_pairs)) // 100 * 100)
    tickets = [
        {"combination": [str(a.get("horse_name") or ""), str(b.get("horse_name") or "")], "stake": stake_each}
        for a, b in valid_pairs
    ]
    names = list(dict.fromkeys(
        str(h.get("horse_name") or "")
        for pair in valid_pairs for h in pair
    ))
    return {
        "bet_type":  "ワイド",
        "horses":    names,
        "tickets":   tickets,
        "total_stake": stake_each * len(tickets),
        "ticket_count": len(tickets),
        "reason":    f"partner_score上位ペア {len(tickets)}点（両マイナス乖離組除外）",
        "skip":      False,
    }


def _gen_umaren_v2(
    enriched: List[Dict[str, Any]],
    bankroll: int,
) -> Dict[str, Any]:
    """
    馬連 v2: axis_score_v2 最上位の非 axis_ban 馬を軸とし、
    partner_score_v2 上位の非 partner_ban 馬を相手にして流し。
    """
    axes = [h for h in enriched if not h.get("axis_ban")]
    axes.sort(key=lambda x: float(x.get("axis_score_v2") or 0.0), reverse=True)

    if not axes:
        return {"bet_type": "馬連", "tickets": [], "skip": True, "skip_reason": "axis_ban 除外後に軸候補なし"}

    axis_horse = axes[0]
    axis_name  = str(axis_horse.get("horse_name") or "")

    partners = [
        h for h in enriched
        if not h.get("partner_ban") and str(h.get("horse_name") or "") != axis_name
    ]
    partners.sort(key=lambda x: float(x.get("partner_score_v2") or 0.0), reverse=True)
    picks = partners[:V2_PARTNER_MAX]

    if not picks:
        return {"bet_type": "馬連", "tickets": [], "skip": True, "skip_reason": "partner_ban 除外後に相手候補なし"}

    stake_each = max(100, (bankroll // len(picks)) // 100 * 100)
    tickets = [
        {"combination": [axis_name, str(p.get("horse_name") or "")], "stake": stake_each}
        for p in picks
    ]
    partner_names = [str(p.get("horse_name") or "") for p in picks]
    gap  = float(axis_horse.get("prob_gap_v2") or 0.0)
    ascore = float(axis_horse.get("axis_score_v2") or 0.0)
    return {
        "bet_type":  "馬連",
        "horses":    [axis_name] + partner_names,
        "tickets":   tickets,
        "total_stake": stake_each * len(tickets),
        "ticket_count": len(tickets),
        "reason":    f"軸: {axis_name}(axis={ascore:.3f}, gap={gap:+.2f}pt) 流し{len(tickets)}点",
        "skip":      False,
    }


def race_warning_v2(
    features: List[Dict[str, Any]],
    ev_table: List[Dict[str, Any]],
) -> Optional[str]:
    """
    人気上位馬（1〜V2_RACE_WARN_POP 位）に prob_gap <= V2_RACE_WARN_GAP の馬が
    V2_RACE_WARN_COUNT 頭以上いる場合に警告文字列を返す。それ以外は None。
    """
    ev_by_name: Dict[str, Dict[str, Any]] = {
        row["horse_name"]: row for row in (ev_table or [])
    }
    flagged = []
    for f in features:
        name   = str(f.get("horse_name") or "")
        ev_row = ev_by_name.get(name) or {}
        pop    = int(ev_row.get("popularity_rank") or 99)
        gap    = float(ev_row.get("value_gap") or 0.0)
        if pop <= V2_RACE_WARN_POP and gap <= V2_RACE_WARN_GAP:
            flagged.append(f"{name}({pop}人気, gap={gap:+.2f}pt)")
    if len(flagged) >= V2_RACE_WARN_COUNT:
        return f"⚠️ 人気上位に乖離大の馬が複数: {' / '.join(flagged)}"
    return None


def bet_recommendations_v2(
    features: List[Dict[str, Any]],
    ev_table: List[Dict[str, Any]],
    bankroll: int = 10000,
) -> Dict[str, Any]:
    """
    v2 買い目生成エントリポイント。既存 recommend_bet_plan とは独立。

    Returns:
        {
            "enriched":   List[Dict]  — axis_score_v2 / partner_score_v2 付き馬リスト（降順）
            "tansho":     Dict        — 単勝推奨
            "fukusho":    Dict        — 複勝推奨
            "wide":       Dict        — ワイド推奨
            "umaren":     Dict        — 馬連推奨
            "warning":    str|None    — race-level warning
        }
    """
    if not features:
        empty = {"bet_type": "-", "tickets": [], "skip": True, "skip_reason": "データなし"}
        return {
            "enriched": [],
            "tansho":   empty,
            "fukusho":  empty,
            "wide":     empty,
            "umaren":   empty,
            "warning":  None,
        }

    enriched = _enrich_horses_v2(features, ev_table)
    # axis_score_v2 降順で整列（UI 表示用）
    enriched_sorted = sorted(enriched, key=lambda x: float(x.get("axis_score_v2") or 0.0), reverse=True)

    return {
        "enriched": enriched_sorted,
        "tansho":   _gen_tansho_v2(enriched_sorted, bankroll),
        "fukusho":  _gen_fukusho_v2(enriched_sorted, bankroll),
        "wide":     _gen_wide_v2(enriched_sorted, bankroll),
        "umaren":   _gen_umaren_v2(enriched_sorted, bankroll),
        "warning":  race_warning_v2(features, ev_table),
    }


# =========================================================
# v3 買い目生成: キャリブレーション + ガード + ポリシー連動
# (bet_recommendations_v3)
# =========================================================
# v2 との主な違い:
#   - calibrated_ai_win_prob: 人気帯補正（1-3番人気↑, 8番人気以下↓）+ 再正規化
#   - market_disagreement_guard: レース全体で人気馬が過小評価されているか監視
#   - bet_size_policy: アラートレベルに応じた予算・券種制御
#   - race_shape_classifier: キャリブ済み確率・EV分布・乖離から形状分類
#   - 券種別スコア: win_bet_score_v3 / place_bet_score_v3 / wide_pair / umaren_pair
#   - trend_delta は補助要素として複勝スコアに反映（主軸ではない）
# v2 / 既存ロジックは無変更。
# =========================================================


def _calibrate_win_probs_v3(
    features: List[Dict[str, Any]],
    ev_table: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    人気帯別補正で calibrated_ai_win_prob を計算し、全体を再正規化して返す。

    補正ルール（V3_CALIB_* 定数参照）:
        1-3番人気  : +V3_CALIB_POP1_3_DELTA    (AIの過小評価を上方補正)
        4-7番人気  : 補正なし
        8番人気以下: +V3_CALIB_POP8_PLUS_DELTA  (AIの過大評価を下方補正)

    追加フィールド:
        calibrated_ai_win_prob : 補正・正規化後の勝率
        calibration_delta      : 補正量（正規化前）
    """
    ev_by_name: Dict[str, Dict[str, Any]] = {
        row["horse_name"]: row for row in (ev_table or [])
    }

    pre_norm: List[Dict[str, Any]] = []
    for f in features:
        h = dict(f)
        name    = str(h.get("horse_name") or "")
        ev_row  = ev_by_name.get(name) or {}
        pop     = int(ev_row.get("popularity_rank") or h.get("popularity") or 99)
        ai_prob = float(h.get("ai_win_prob") or h.get("win_prob") or 0.0)

        if pop <= 3:
            delta = V3_CALIB_POP1_3_DELTA
        elif pop >= 8:
            delta = V3_CALIB_POP8_PLUS_DELTA
        else:
            delta = 0.0

        h["calibration_delta"] = round(delta, 4)
        h["_pre_norm_calib"]   = max(0.0, ai_prob + delta)
        pre_norm.append(h)

    total = sum(h["_pre_norm_calib"] for h in pre_norm)
    for h in pre_norm:
        if total > 0:
            h["calibrated_ai_win_prob"] = round(h["_pre_norm_calib"] / total, 4)
        else:
            h["calibrated_ai_win_prob"] = 0.0
        del h["_pre_norm_calib"]

    return pre_norm


def market_disagreement_guard_v3(
    features: List[Dict[str, Any]],
    ev_table: List[Dict[str, Any]],
    enriched_list: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    1〜V3_GUARD_POP_BAND 番人気のうち gap <= V3_GUARD_GAP_THRESH の馬数を集計し、
    アラートレベルを返す。

    enriched_list が渡された場合は calibrated_prob_gap_v3 でも集計し、
    アラート判定を calibrated ベースで行う（より精度の高い判定）。
    raw ベースの件数は raw_underestimated_count として保持。

    Returns:
        {
            is_alert                     : bool
            alert_level                  : "high" | "medium" | "normal"
            top_pop_underestimated_count : int   — calibrated ベース（enriched_list あり）
                                                   または raw ベース（なし）
            avg_negative_gap_top3_pop    : float — calibrated gap の平均（同上）
            raw_underestimated_count     : int   — raw value_gap ベースの件数
            calibrated_underestimated_count: int — calibrated_prob_gap_v3 ベース（なければ -1）
            message                      : str
        }
    """
    ev_by_name: Dict[str, Dict[str, Any]] = {
        row["horse_name"]: row for row in (ev_table or [])
    }

    # ── raw gap ベース集計（常時） ──────────────────────────────────
    raw_under: List[tuple] = []   # (name, pop, gap)
    for f in features:
        name   = str(f.get("horse_name") or "")
        ev_row = ev_by_name.get(name) or {}
        pop    = int(ev_row.get("popularity_rank") or f.get("popularity") or 99)
        gap    = float(ev_row.get("value_gap") or 0.0)
        if pop <= V3_GUARD_POP_BAND and gap <= V3_GUARD_GAP_THRESH:
            raw_under.append((name, pop, gap))
    raw_count = len(raw_under)

    # ── calibrated gap ベース集計（enriched_list あり時） ──────────
    calib_count = -1
    calib_under: List[tuple] = []
    if enriched_list is not None:
        enr_by_name = {h["horse_name"]: h for h in enriched_list}
        for f in features:
            name   = str(f.get("horse_name") or "")
            ev_row = ev_by_name.get(name) or {}
            enr    = enr_by_name.get(name) or {}
            pop    = int(ev_row.get("popularity_rank") or f.get("popularity") or 99)
            cgap   = float(enr.get("calibrated_prob_gap_v3") or ev_row.get("value_gap") or 0.0)
            if pop <= V3_GUARD_POP_BAND and cgap <= V3_GUARD_GAP_THRESH:
                calib_under.append((name, pop, cgap))
        calib_count = len(calib_under)

    # ── アラート判定: calibrated 優先、なければ raw ────────────────
    active_under = calib_under if enriched_list is not None else raw_under
    count   = len(active_under)
    avg_neg = sum(g for _, _, g in active_under) / count if count > 0 else 0.0

    if count >= V3_GUARD_HIGH_COUNT:
        alert_level = "high"
        is_alert    = True
        detail = ", ".join(f"{n}({p}人気, {g:+.2f}pt)" for n, p, g in active_under)
        msg = f"⚠️ 高アラート: 1-3番人気に乖離大が{count}頭 [{detail}]"
    elif count >= V3_GUARD_MED_COUNT:
        alert_level = "medium"
        is_alert    = True
        detail = ", ".join(f"{n}({p}人気, {g:+.2f}pt)" for n, p, g in active_under)
        msg = f"⚡ 中アラート: 1-3番人気に乖離大が{count}頭 [{detail}]"
    else:
        alert_level = "normal"
        is_alert    = False
        msg         = "市場乖離は正常範囲内"

    return {
        "is_alert":                       is_alert,
        "alert_level":                    alert_level,
        "top_pop_underestimated_count":   count,
        "avg_negative_gap_top3_pop":      round(avg_neg, 4),
        "raw_underestimated_count":       raw_count,
        "calibrated_underestimated_count": calib_count,
        "message":                        msg,
    }


def bet_size_policy_v3(guard_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    ガード結果からベットサイズ・券種ポリシーを返す。

    Returns:
        {
            bankroll_multiplier : float  — 予算乗数
            umaren_enabled      : bool   — 馬連を購入するか
            wide_max_tickets    : int    — ワイド最大点数
            fukusho_max         : int    — 複勝最大頭数
            note                : str
        }
    """
    level = guard_result.get("alert_level", "normal")
    if level == "high":
        return {
            "bankroll_multiplier": V3_POLICY_HIGH_BRL_MULT,
            "umaren_enabled":      V3_POLICY_HIGH_UMAREN,
            "wide_max_tickets":    V3_POLICY_HIGH_WIDE_MAX,
            "fukusho_max":         2,
            "note":                "高アラート: 馬連停止・予算50%・ワイド上限2点",
        }
    elif level == "medium":
        return {
            "bankroll_multiplier": V3_POLICY_MED_BRL_MULT,
            "umaren_enabled":      True,
            "wide_max_tickets":    V3_WIDE_MAX,
            "fukusho_max":         2,
            "note":                "中アラート: 予算75%",
        }
    else:
        return {
            "bankroll_multiplier": 1.0,
            "umaren_enabled":      True,
            "wide_max_tickets":    V3_WIDE_MAX,
            "fukusho_max":         2,
            "note":                "通常: ポリシー変更なし",
        }


def race_shape_classifier_v3(
    features: List[Dict[str, Any]],
    ev_table: List[Dict[str, Any]],
    enriched_list: List[Dict[str, Any]],
) -> str:
    """
    レース形状を分類する。enriched_list から calibrated_prob_gap_v3 と win_ev_v3 を使う。

    Returns: "solid" | "balanced" | "chaotic" | "ai_market_conflict"

    優先順位:
      1. ai_market_conflict: calibrated_prob_gap_v3 <= -0.08 かつ人気上位が
                             V3_SHAPE_CONFLICT_COUNT 以上
      2. solid             : キャリブ済み上位3頭の合計確率 > V3_SHAPE_SOLID_CONC
      3. chaotic           : win_ev_v3>=1.0 の馬が V3_SHAPE_EV_COUNT_CHAOS 以上、
                             または上位3頭合計 < V3_SHAPE_BALANCED_LOW
      4. balanced          : それ以外
    """
    ev_by_name: Dict[str, Dict[str, Any]] = {
        row["horse_name"]: row for row in (ev_table or [])
    }
    enr_by_name: Dict[str, Dict[str, Any]] = {
        h["horse_name"]: h for h in (enriched_list or [])
    }

    # 1. ai_market_conflict チェック（calibrated_prob_gap_v3 ベース）
    conflict_count = 0
    for f in features:
        name   = str(f.get("horse_name") or "")
        ev_row = ev_by_name.get(name) or {}
        enr    = enr_by_name.get(name) or {}
        pop    = int(ev_row.get("popularity_rank") or f.get("popularity") or 99)
        cgap   = float(enr.get("calibrated_prob_gap_v3") or ev_row.get("value_gap") or 0.0)
        if pop <= V3_GUARD_POP_BAND and cgap <= V3_GUARD_GAP_THRESH:
            conflict_count += 1
    if conflict_count >= V3_SHAPE_CONFLICT_COUNT:
        return "ai_market_conflict"

    # 2. 上位3頭集中度（calibrated_ai_win_prob）
    calib_probs = sorted(
        [float(h.get("calibrated_ai_win_prob") or 0.0) for h in enriched_list],
        reverse=True,
    )
    top3_conc = sum(calib_probs[:3]) if len(calib_probs) >= 3 else sum(calib_probs)

    # 3. EV>=1.0 の馬の数（win_ev_v3 = calibrated EV）
    ev_count = sum(
        1 for h in enriched_list
        if float(h.get("win_ev_v3") or 0.0) >= 1.0
    )

    if top3_conc > V3_SHAPE_SOLID_CONC:
        return "solid"
    elif ev_count >= V3_SHAPE_EV_COUNT_CHAOS or top3_conc < V3_SHAPE_BALANCED_LOW:
        return "chaotic"
    else:
        return "balanced"


def _calc_win_bet_score_v3(
    calibrated_prob: float,
    prob_gap: float,
    win_ev: float,
) -> float:
    """単勝スコア v3: キャリブレーション済み勝率を主軸とする。"""
    gap_norm = _v2_norm_gap(prob_gap)
    ev_norm  = _v2_norm_ev(win_ev)
    return round(min(1.0, max(0.0,
        calibrated_prob * V3_WIN_W_CALIB
        + ev_norm       * V3_WIN_W_WIN_EV
        + gap_norm      * V3_WIN_W_GAP
    )), 4)


def _calc_place_bet_score_v3(
    top3_prob: float,
    prob_gap: float,
    trend_delta: float,
) -> float:
    """複勝スコア v3: top3_prob + gap補正 + trend補正（trend は補助）"""
    gap_norm   = _v2_norm_gap(prob_gap)
    trend_norm = _v2_norm_trend(trend_delta)
    return round(min(1.0, max(0.0,
        top3_prob    * V3_PLACE_W_TOP3
        + gap_norm   * V3_PLACE_W_GAP
        + trend_norm * V3_PLACE_W_TREND
    )), 4)


def _calc_wide_pair_score_v3(
    h_a: Dict[str, Any],
    h_b: Dict[str, Any],
) -> float:
    """ワイドペアスコア v3: 両馬の top3/gap/EV の平均で評価。gap は calibrated 優先。"""
    top3_a = float(h_a.get("top3_prob_v3") or 0.0)
    top3_b = float(h_b.get("top3_prob_v3") or 0.0)
    gap_a  = float(h_a.get("calibrated_prob_gap_v3") or h_a.get("prob_gap_v3") or 0.0)
    gap_b  = float(h_b.get("calibrated_prob_gap_v3") or h_b.get("prob_gap_v3") or 0.0)
    ev_a   = float(h_a.get("win_ev_v3") or 0.0)
    ev_b   = float(h_b.get("win_ev_v3") or 0.0)

    top3_avg     = (top3_a + top3_b) / 2.0
    gap_norm_avg = (_v2_norm_gap(gap_a) + _v2_norm_gap(gap_b)) / 2.0
    ev_norm_avg  = (_v2_norm_ev(ev_a) + _v2_norm_ev(ev_b)) / 2.0

    return round(min(1.0, max(0.0,
        top3_avg     * V3_WIDE_W_TOP3_AVG
        + gap_norm_avg * V3_WIDE_W_GAP_AVG
        + ev_norm_avg  * V3_WIDE_W_EV_AVG
    )), 4)


def _calc_umaren_pair_score_v3(
    axis: Dict[str, Any],
    partner: Dict[str, Any],
) -> float:
    """馬連ペアスコア v3: 軸のキャリブ勝率 + 相手の top3_prob で評価。gap は calibrated 優先。"""
    axis_calib    = float(axis.get("calibrated_ai_win_prob") or 0.0)
    axis_gap      = float(axis.get("calibrated_prob_gap_v3") or axis.get("prob_gap_v3") or 0.0)
    part_top3     = float(partner.get("top3_prob_v3") or 0.0)
    part_gap      = float(partner.get("calibrated_prob_gap_v3") or partner.get("prob_gap_v3") or 0.0)

    axis_gap_norm = _v2_norm_gap(axis_gap)
    part_gap_norm = _v2_norm_gap(part_gap)

    return round(min(1.0, max(0.0,
        axis_calib    * V3_UMAREN_W_AXIS_CALIB
        + axis_gap_norm * V3_UMAREN_W_AXIS_GAP
        + part_top3   * V3_UMAREN_W_PART_TOP3
        + part_gap_norm * V3_UMAREN_W_PART_GAP
    )), 4)


def _enrich_horses_v3(
    features: List[Dict[str, Any]],
    ev_table: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    各馬に v3 スコア・フラグを付与して返す。

    追加フィールド:
        market_win_prob          : 1/win_odds（市場勝率）
        calibrated_ai_win_prob   : 人気帯補正・正規化済み勝率
        calibration_delta        : 補正量（正規化前）
        prob_gap_v3              : raw gap = ai_win_prob - market_win_prob（= value_gap）
        calibrated_prob_gap_v3   : calibrated gap = calibrated_ai_win_prob - market_win_prob
        raw_win_ev_v3            : raw EV = ai_win_prob × win_odds
        win_ev_v3                : calibrated EV = calibrated_ai_win_prob × win_odds
        trend_delta_v3           : trend_delta
        top3_prob_v3             : top3_prob
        axis_ban_v3              : True = 軸禁止（calibrated_prob_gap_v3 ベース）
        partner_ban_v3           : True = 相手禁止（calibrated_prob_gap_v3 + win_ev_v3 ベース）
        win_bet_score_v3         : 単勝適性スコア（calibrated gap + calibrated EV）
        place_bet_score_v3       : 複勝適性スコア（calibrated gap + trend）
    """
    calibrated_list = _calibrate_win_probs_v3(features, ev_table)
    ev_by_name: Dict[str, Dict[str, Any]] = {
        row["horse_name"]: row for row in (ev_table or [])
    }

    enriched: List[Dict[str, Any]] = []
    for h in calibrated_list:
        name    = str(h.get("horse_name") or "")
        ev_row  = ev_by_name.get(name) or {}

        calibrated_prob  = float(h.get("calibrated_ai_win_prob") or 0.0)
        market_win_prob  = float(ev_row.get("market_win_prob") or 0.0)
        raw_prob_gap     = float(ev_row.get("value_gap") or 0.0)           # ai_win_prob - market
        calib_prob_gap   = round(calibrated_prob - market_win_prob, 4)     # calibrated - market

        win_odds = float(h.get("win_odds") or ev_row.get("win_odds") or 0.0)

        # raw EV: stored win_ev 優先 → ai_win_prob × win_odds
        _stored_ev = h.get("win_ev")
        if _stored_ev is not None:
            raw_win_ev = float(_stored_ev)
        else:
            _wp_raw = float(h.get("win_prob") or h.get("ai_win_prob") or 0.0)
            raw_win_ev = round(_wp_raw * win_odds, 4) if _wp_raw > 0 and win_odds > 0 else 0.0

        # calibrated EV: calibrated_prob × win_odds（v3 の主軸）
        win_ev_v3 = round(calibrated_prob * win_odds, 4) if calibrated_prob > 0 and win_odds > 0 else 0.0

        trend_delta = float(h.get("trend_delta") or 0.0)

        top2_prob = float(h.get("top2_prob") or 0.0)
        top3_prob = float(h.get("top3_prob") or 0.0)
        if top2_prob == 0.0 or top3_prob == 0.0:
            _pp       = estimate_placement_probs(h, {})
            top2_prob = top2_prob or _pp.get("p_top2", 0.0)
            top3_prob = top3_prob or _pp.get("p_top3", 0.0)

        # ban フラグ: calibrated gap + calibrated EV を使用
        axis_ban_v3    = calib_prob_gap <= V3_AXIS_BAN_GAP
        partner_ban_v3 = (calib_prob_gap <= V3_PARTNER_BAN_GAP) and (win_ev_v3 <= V3_PARTNER_BAN_EV)

        h.update({
            "market_win_prob":        round(market_win_prob, 4),
            "prob_gap_v3":            round(raw_prob_gap, 4),       # raw gap（保持）
            "calibrated_prob_gap_v3": calib_prob_gap,               # calibrated gap（v3 主軸）
            "raw_win_ev_v3":          round(raw_win_ev, 4),         # raw EV（保持）
            "win_ev_v3":              win_ev_v3,                    # calibrated EV（v3 主軸）
            "trend_delta_v3":         round(trend_delta, 4),
            "top3_prob_v3":           round(top3_prob, 4),
            "axis_ban_v3":            axis_ban_v3,
            "partner_ban_v3":         partner_ban_v3,
            # スコア: calibrated gap / calibrated EV を使用
            "win_bet_score_v3":       _calc_win_bet_score_v3(calibrated_prob, calib_prob_gap, win_ev_v3),
            "place_bet_score_v3":     _calc_place_bet_score_v3(top3_prob, calib_prob_gap, trend_delta),
        })
        enriched.append(h)

    return enriched


def _gen_tansho_v3(
    enriched: List[Dict[str, Any]],
    bankroll: int,
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    """単勝 v3: axis_ban_v3=False かつ win_ev >= V3_TANSHO_EV_MIN の馬を win_bet_score_v3 降順で最大1頭。"""
    candidates = [
        h for h in enriched
        if not h.get("axis_ban_v3")
        and float(h.get("win_ev_v3") or 0.0) >= V3_TANSHO_EV_MIN
    ]
    candidates.sort(key=lambda x: float(x.get("win_bet_score_v3") or 0.0), reverse=True)

    if not candidates:
        return {"bet_type": "単勝", "tickets": [], "skip": True, "skip_reason": "EV>=1.0 の軸候補なし"}

    effective_brl = int(bankroll * policy["bankroll_multiplier"])
    top   = candidates[0]
    name  = str(top.get("horse_name") or "")
    ev    = float(top.get("win_ev_v3") or 0.0)
    cgap  = float(top.get("calibrated_prob_gap_v3") or top.get("prob_gap_v3") or 0.0)
    calib = float(top.get("calibrated_ai_win_prob") or 0.0)
    score = float(top.get("win_bet_score_v3") or 0.0)
    stake = max(100, (effective_brl // 10) // 100 * 100)
    return {
        "bet_type":     "単勝",
        "horses":       [name],
        "tickets":      [{"combination": [name], "stake": stake}],
        "total_stake":  stake,
        "ticket_count": 1,
        "reason":       f"{name} score={score:.3f} EV={ev:.2f} calib={calib:.3f} cgap={cgap:+.2f}pt",
        "skip":         False,
    }


def _gen_fukusho_v3(
    enriched: List[Dict[str, Any]],
    bankroll: int,
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    """複勝 v3: partner_ban_v3=False の馬を place_bet_score_v3 降順で policy['fukusho_max'] 頭。"""
    candidates = [h for h in enriched if not h.get("partner_ban_v3")]
    candidates.sort(key=lambda x: float(x.get("place_bet_score_v3") or 0.0), reverse=True)
    picks = candidates[:policy["fukusho_max"]]

    if not picks:
        return {"bet_type": "複勝", "tickets": [], "skip": True, "skip_reason": "partner_ban_v3 除外後に候補なし"}

    effective_brl = int(bankroll * policy["bankroll_multiplier"])
    stake_each    = max(100, (effective_brl // (len(picks) * 10)) // 100 * 100)
    tickets = [{"combination": [str(h.get("horse_name") or "")], "stake": stake_each} for h in picks]
    names   = [str(h.get("horse_name") or "") for h in picks]
    return {
        "bet_type":     "複勝",
        "horses":       names,
        "tickets":      tickets,
        "total_stake":  stake_each * len(picks),
        "ticket_count": len(picks),
        "reason":       f"複勝スコア上位: {' / '.join(names)}",
        "skip":         False,
    }


def _gen_wide_v3(
    enriched: List[Dict[str, Any]],
    bankroll: int,
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    """
    ワイド v3: place_bet_score_v3 上位5頭からペアスコアで上位 policy['wide_max_tickets'] 点。
    両馬ともに prob_gap_v3 <= V3_AXIS_BAN_GAP の組み合わせは除外。
    """
    max_tickets = policy["wide_max_tickets"]
    candidates  = [h for h in enriched if not h.get("partner_ban_v3")]
    candidates.sort(key=lambda x: float(x.get("place_bet_score_v3") or 0.0), reverse=True)
    pool = candidates[:5]

    if len(pool) < 2:
        return {"bet_type": "ワイド", "tickets": [], "skip": True, "skip_reason": "ワイド候補が2頭未満"}

    pair_scores: List[tuple] = []
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            a, b = pool[i], pool[j]
            gap_a = float(a.get("prob_gap_v3") or 0.0)
            gap_b = float(b.get("prob_gap_v3") or 0.0)
            if gap_a <= V3_AXIS_BAN_GAP and gap_b <= V3_AXIS_BAN_GAP:
                continue  # 両馬マイナス乖離大は除外
            pair_scores.append((a, b, _calc_wide_pair_score_v3(a, b)))

    if not pair_scores:
        return {"bet_type": "ワイド", "tickets": [], "skip": True, "skip_reason": "有効なワイドペアなし（両馬マイナス乖離）"}

    pair_scores.sort(key=lambda x: x[2], reverse=True)
    top_pairs = pair_scores[:max_tickets]

    effective_brl = int(bankroll * policy["bankroll_multiplier"])
    stake_each    = max(100, (effective_brl // len(top_pairs)) // 100 * 100)
    tickets = [
        {"combination": [str(a.get("horse_name") or ""), str(b.get("horse_name") or "")], "stake": stake_each}
        for a, b, _ in top_pairs
    ]
    names = list(dict.fromkeys(
        str(h.get("horse_name") or "")
        for a, b, _ in top_pairs
        for h in [a, b]
    ))
    return {
        "bet_type":     "ワイド",
        "horses":       names,
        "tickets":      tickets,
        "total_stake":  stake_each * len(tickets),
        "ticket_count": len(tickets),
        "reason":       f"ワイドスコア上位ペア {len(tickets)}点",
        "skip":         False,
    }


def _gen_umaren_v3(
    enriched: List[Dict[str, Any]],
    bankroll: int,
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    """
    馬連 v3: win_bet_score_v3 最上位の非 axis_ban_v3 馬を軸とし、
    umaren_pair_score でソートした非 partner_ban_v3 馬を相手に V3_PARTNER_MAX 点。
    高アラート時（policy['umaren_enabled']=False）はスキップ。
    """
    if not policy.get("umaren_enabled", True):
        return {"bet_type": "馬連", "tickets": [], "skip": True, "skip_reason": "高アラートにより馬連停止"}

    axes = [h for h in enriched if not h.get("axis_ban_v3")]
    axes.sort(key=lambda x: float(x.get("win_bet_score_v3") or 0.0), reverse=True)

    if not axes:
        return {"bet_type": "馬連", "tickets": [], "skip": True, "skip_reason": "axis_ban_v3 除外後に軸候補なし"}

    axis_horse = axes[0]
    axis_name  = str(axis_horse.get("horse_name") or "")

    partners = [
        h for h in enriched
        if not h.get("partner_ban_v3") and str(h.get("horse_name") or "") != axis_name
    ]
    partner_scored = [
        (h, _calc_umaren_pair_score_v3(axis_horse, h)) for h in partners
    ]
    partner_scored.sort(key=lambda x: x[1], reverse=True)
    picks = [h for h, _ in partner_scored[:V3_PARTNER_MAX]]

    if not picks:
        return {"bet_type": "馬連", "tickets": [], "skip": True, "skip_reason": "partner_ban_v3 除外後に相手候補なし"}

    effective_brl = int(bankroll * policy["bankroll_multiplier"])
    stake_each    = max(100, (effective_brl // len(picks)) // 100 * 100)
    tickets = [
        {"combination": [axis_name, str(p.get("horse_name") or "")], "stake": stake_each}
        for p in picks
    ]
    partner_names = [str(p.get("horse_name") or "") for p in picks]
    calib  = float(axis_horse.get("calibrated_ai_win_prob") or 0.0)
    cgap   = float(axis_horse.get("calibrated_prob_gap_v3") or axis_horse.get("prob_gap_v3") or 0.0)
    score  = float(axis_horse.get("win_bet_score_v3") or 0.0)
    return {
        "bet_type":     "馬連",
        "horses":       [axis_name] + partner_names,
        "tickets":      tickets,
        "total_stake":  stake_each * len(tickets),
        "ticket_count": len(tickets),
        "reason":       f"軸: {axis_name}(score={score:.3f}, calib={calib:.3f}, cgap={cgap:+.2f}pt) 流し{len(tickets)}点",
        "skip":         False,
    }


def bet_recommendations_v3(
    features: List[Dict[str, Any]],
    ev_table: List[Dict[str, Any]],
    bankroll: int = 10000,
) -> Dict[str, Any]:
    """
    v3 買い目生成エントリポイント。v2 / 既存ロジックは無変更。

    Returns:
        {
            "enriched":   List[Dict] — v3 スコア付き馬リスト（win_bet_score_v3 降順）
            "guard":      Dict       — market_disagreement_guard_v3 結果
            "policy":     Dict       — bet_size_policy_v3 結果
            "race_shape": str        — solid/balanced/chaotic/ai_market_conflict
            "tansho":     Dict       — 単勝推奨
            "fukusho":    Dict       — 複勝推奨
            "wide":       Dict       — ワイド推奨
            "umaren":     Dict       — 馬連推奨
            "summary":    Dict       — レース全体サマリ（audit 用）
        }
    """
    if not features:
        _empty = {"bet_type": "-", "tickets": [], "skip": True, "skip_reason": "データなし"}
        _no_guard = {
            "is_alert": False, "alert_level": "normal",
            "top_pop_underestimated_count": 0, "avg_negative_gap_top3_pop": 0.0,
            "raw_underestimated_count": 0, "calibrated_underestimated_count": -1,
            "message": "データなし",
        }
        return {
            "enriched":   [],
            "guard":      _no_guard,
            "policy":     {"bankroll_multiplier": 1.0, "umaren_enabled": True,
                           "wide_max_tickets": V3_WIDE_MAX, "fukusho_max": 2, "note": "データなし"},
            "race_shape": "balanced",
            "tansho":     _empty,
            "fukusho":    _empty,
            "wide":       _empty,
            "umaren":     _empty,
            "summary":    {},
        }

    # enrich 先行 → guard に enriched を渡して calibrated gap ベースで判定
    enriched = _enrich_horses_v3(features, ev_table)
    guard    = market_disagreement_guard_v3(features, ev_table, enriched_list=enriched)
    policy   = bet_size_policy_v3(guard)

    # race_shape は enriched list を直接渡す（calibrated_prob_gap_v3 / win_ev_v3 を内部使用）
    race_shape = race_shape_classifier_v3(features, ev_table, enriched)

    enriched_sorted = sorted(
        enriched,
        key=lambda x: float(x.get("win_bet_score_v3") or 0.0),
        reverse=True,
    )

    summary = {
        "alert_level":                    guard["alert_level"],
        "race_shape":                     race_shape,
        "bankroll_mult":                  policy["bankroll_multiplier"],
        "umaren_enabled":                 policy["umaren_enabled"],
        "top_pop_underestimated_count":   guard["top_pop_underestimated_count"],
        "avg_negative_gap_top3_pop":      guard["avg_negative_gap_top3_pop"],
        "raw_underestimated_count":       guard["raw_underestimated_count"],
        "calibrated_underestimated_count": guard["calibrated_underestimated_count"],
        "policy_note":                    policy["note"],
    }

    return {
        "enriched":   enriched_sorted,
        "guard":      guard,
        "policy":     policy,
        "race_shape": race_shape,
        "tansho":     _gen_tansho_v3(enriched_sorted, bankroll, policy),
        "fukusho":    _gen_fukusho_v3(enriched_sorted, bankroll, policy),
        "wide":       _gen_wide_v3(enriched_sorted, bankroll, policy),
        "umaren":     _gen_umaren_v3(enriched_sorted, bankroll, policy),
        "summary":    summary,
    }
