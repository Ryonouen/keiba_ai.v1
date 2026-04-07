"""
confidence_scorer.py
レースの「勝負度スコア」（0.0〜1.0）を算出する。

スコア構成:
  40% — EVスコア: max(win_ev) を sigmoid で正規化
  30% — market_edge スコア: max(win_market_edge) を sigmoid で正規化
  30% — 予測集中度: 上位3頭の win_prob 合計（AIが本命を絞っているか）

閾値以上のレースを「勝負レース」として選別できる。

使い方:
  from confidence_scorer import compute_race_confidence, filter_races_by_confidence
  score = compute_race_confidence(features)          # features = predict_win_probability の出力と同形式
  races = filter_races_by_confidence(all_races, threshold=0.6)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

# スコア重み（合計 1.0）
_W_EV    = 0.40
_W_EDGE  = 0.30
_W_CONC  = 0.30

# sigmoid スケール係数
_EV_SCALE   = 2.0   # EV=0.5 のとき sigmoid(0.5*2)=0.73 程度
_EDGE_SCALE = 3.0   # market_edge=0.3 のとき sigmoid(0.9)=0.71 程度


def _sigmoid(x: float) -> float:
    """標準 sigmoid 関数。オーバーフロー対策あり。"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def compute_race_confidence(features: List[Dict[str, Any]]) -> float:
    """
    1レース分の features リストから勝負度スコア（0.0〜1.0）を返す。

    Parameters
    ----------
    features : 馬ごとの辞書リスト。以下のキーを使用:
      - win_prob        : float (AI勝率。合計1.0に正規化済み)
      - win_ev          : float | None (単勝EV = win_prob × win_odds - 1)
      - win_market_edge : float | None (market_edge = win_prob - 1/win_odds)
    """
    if not features:
        return 0.0

    # ── EV スコア ──────────────────────────────────────────────────
    evs = [float(f.get("win_ev") or 0.0) for f in features]
    max_ev  = max(evs) if evs else 0.0
    ev_score = _sigmoid(max_ev * _EV_SCALE)

    # ── market_edge スコア ──────────────────────────────────────────
    edges = [float(f.get("win_market_edge") or 0.0) for f in features]
    max_edge   = max(edges) if edges else 0.0
    edge_score = _sigmoid(max_edge * _EDGE_SCALE)

    # ── 予測集中度 ──────────────────────────────────────────────────
    probs = sorted(
        [float(f.get("win_prob") or 0.0) for f in features],
        reverse=True,
    )
    top3_sum = sum(probs[:3]) if len(probs) >= 3 else sum(probs)
    # top3_sum は 0〜1 の範囲。0.5 を超えると高集中とみなす。
    conc_score = _sigmoid((top3_sum - 0.5) * 4.0)

    score = _W_EV * ev_score + _W_EDGE * edge_score + _W_CONC * conc_score
    return round(max(0.0, min(1.0, score)), 4)


def filter_races_by_confidence(
    races: List[Dict[str, Any]],
    threshold: float = 0.6,
) -> List[Dict[str, Any]]:
    """
    confidence_score が threshold 以上のレースのみ返す。

    Parameters
    ----------
    races     : 各要素に "confidence_score" キーを持つ辞書リスト
    threshold : 勝負レースと判断する閾値（デフォルト 0.6）
    """
    return [r for r in races if r.get("confidence_score", 0.0) >= threshold]


def score_and_annotate(
    races: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    races リストの各要素に "confidence_score" を追記して返す（in-place 変更）。
    各 race は "horses" または "features" キーに馬リストを持つことを想定する。

    dashboard_loader.py の load_races_for_date() 出力に直接適用できる。
    """
    for race in races:
        features = race.get("horses") or race.get("features") or []
        race["confidence_score"] = compute_race_confidence(features)
    return races
