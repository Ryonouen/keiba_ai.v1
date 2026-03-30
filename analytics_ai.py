"""
analytics_ai.py
成績集計モジュール

責務:
- 保存済みレースデータから各種パフォーマンス指標を集計
- 将来の閾値最適化・バックテストに向けたデータ基盤

集計対象: result フィールドが入力済みのレコードのみ
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from backtest_analyzer import (
    summarize_role_performance,
    summarize_pass_judgment,
    aggregate_review_tags,
    generate_improvement_comments,
)


# =========================================================
# 定数
# =========================================================

# セグメント別集計の統計的最小サンプル数。
# これ未満のグループは hit_rate / roi を "insufficient_data" で返す。
# 3 では信頼区間が広すぎるため 8 に設定（二項分布 95%CI ≒ ±0.25 程度）
MIN_STAT_SAMPLES: int = 8

# =========================================================
# 内部ヘルパー
# =========================================================

def _field_size_bucket(n: int) -> str:
    """頭数をバケットに変換（集計キーとして使用）。"""
    if n <= 0:   return "不明"
    if n <= 8:   return "少頭数(〜8)"
    if n <= 12:  return "中頭数(9-12)"
    if n <= 16:  return "大頭数(13-16)"
    return "フルゲート(17+)"


def _value_horse_core_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """妙味馬の基本精度指標。セグメント別集計・summarize_value_horse 内部で共用。"""
    completed = [r for r in records if _has_result(r)]
    total = in_money = wrong = bought_hit = 0
    for r in completed:
        finish_order = (r.get("result") or {}).get("finish_order", [])
        top3   = set(finish_order[:3])
        bought: set = set()
        for t in r.get("recommended_tickets", []):
            bought.update(t.get("combination", []))
        for h in r.get("horses", []):
            if not h.get("is_value_horse"):
                continue
            total += 1
            name = h["horse_name"]
            if name in top3:
                in_money += 1
                if name in bought:
                    bought_hit += 1
            else:
                wrong += 1
    return {
        "value_total":           total,
        "value_in_money":        in_money,
        "value_wrong_count":     wrong,                                      # 拾いすぎ回数
        "value_in_money_rate":   round(in_money    / max(total, 1), 4),
        "value_wrong_rate":      round(wrong        / max(total, 1), 4),     # 拾いすぎ率
        "value_bought_hit_rate": round(bought_hit   / max(total, 1), 4),
    }


def _danger_core_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """危険人気馬判定精度の基本指標。セグメント別集計・summarize_danger_cutoff 内部で共用。"""
    completed = [r for r in records if _has_result(r)]
    truly_total = truly_correct = 0
    for r in completed:
        finish_order = (r.get("result") or {}).get("finish_order", [])
        top3 = set(finish_order[:3])
        for h in r.get("horses", []):
            if not h.get("is_danger_favorite"):
                continue
            if h.get("is_truly_dangerous", True):
                truly_total += 1
                if h["horse_name"] not in top3:
                    truly_correct += 1
    return {
        "truly_total":         truly_total,
        "truly_correct":       truly_correct,
        "danger_correct_rate": round(truly_correct / max(truly_total, 1), 4),
    }


def _segment_rows(
    records: List[Dict[str, Any]],
    segment_key: str,
    stats_fn,
) -> List[Dict[str, Any]]:
    """任意のセグメントキーでグループ化して stats_fn を適用する共通処理。"""
    completed = [r for r in records if _has_result(r)]
    groups: Dict[str, List] = defaultdict(list)
    for r in completed:
        k = (
            _field_size_bucket(r.get("race_field_size") or 0)
            if segment_key == "field_size"
            else str(r.get(segment_key) or "不明")
        )
        groups[k].append(r)
    return [
        {segment_key: k, **stats_fn(group)}
        for k, group in sorted(groups.items())
    ]


def _has_result(record: Dict[str, Any]) -> bool:
    """result フィールドが入力済みか確認する。"""
    r = record.get("result")
    return isinstance(r, dict) and r.get("finish_order") is not None


def _base_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """共通集計処理。records は全て result 入力済みであること。"""
    n            = len(records)
    n_hit        = sum(1 for r in records if r.get("result", {}).get("hit") is True)
    n_pass       = sum(1 for r in records if r.get("is_pass", False))
    total_invest = sum(int(r.get("result", {}).get("investment_amount") or 0) for r in records)
    total_return = sum(int(r.get("result", {}).get("return_amount") or 0)     for r in records)
    ticket_counts = [
        len(r.get("recommended_tickets", []))
        for r in records
        if not r.get("is_pass", False)
    ]
    avg_tickets  = round(sum(ticket_counts) / max(len(ticket_counts), 1), 2)

    hit_rate  = round(n_hit / max(n - n_pass, 1), 4)
    pass_rate = round(n_pass / max(n, 1), 4)
    roi       = round(total_return / max(total_invest, 1), 4) if total_invest > 0 else None

    return {
        "n_races":       n,
        "n_hit":         n_hit,
        "n_pass":        n_pass,
        "hit_rate":      hit_rate,
        "pass_rate":     pass_rate,
        "total_invest":  total_invest,
        "total_return":  total_return,
        "roi":           roi,
        "avg_tickets":   avg_tickets,
    }


# =========================================================
# 全体集計
# =========================================================

def summarize_performance(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """全レース通算成績を返す。"""
    completed = [r for r in records if _has_result(r)]
    if not completed:
        return _base_stats([])
    return _base_stats(completed)


# =========================================================
# 軸別集計（共通パターン）
# =========================================================

def _summarize_by_key(
    records: List[Dict[str, Any]],
    key_fn,
    key_label: str = "key",
) -> List[Dict[str, Any]]:
    """任意のキー関数でグループ化して集計する。"""
    completed = [r for r in records if _has_result(r)]
    groups: Dict[str, List] = defaultdict(list)
    for r in completed:
        k = key_fn(r)
        if k:
            groups[k].append(r)

    rows = []
    for k, group in sorted(groups.items()):
        stats = _base_stats(group)
        # サンプル不足のセグメントは統計的に信頼できないフラグを付与
        if stats["n_races"] < MIN_STAT_SAMPLES:
            stats["insufficient_data"] = True
        rows.append({key_label: k, **stats})
    return rows


def summarize_by_bet_type(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """券種別成績を返す。"""
    return _summarize_by_key(
        records,
        lambda r: r.get("recommended_bet_type", "") or "不明",
        key_label="bet_type",
    )


def summarize_by_race_structure(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """レース構造別成績を返す。"""
    return _summarize_by_key(
        records,
        lambda r: r.get("structure_type", "") or "不明",
        key_label="structure_type",
    )


def summarize_by_ev_type(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """ev_type 別（EV比較型 / 構造型）成績を返す。"""
    return _summarize_by_key(
        records,
        lambda r: r.get("ev_type", "") or "不明",
        key_label="ev_type",
    )


def summarize_by_grade(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """グレード別（G1/G2/G3）成績を返す。"""
    return _summarize_by_key(
        records,
        lambda r: r.get("race_grade", "") or "不明",
        key_label="grade",
    )


# =========================================================
# 特定テーマ集計
# =========================================================

def summarize_danger_cutoff(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    危険人気馬を切ったケースの成績。

    Returns
    -------
    {
        "n_races_with_danger": int,   # 危険馬がいたレース数
        "danger_truly_correct_rate": float,  # 消し推奨が正解だった割合
        "danger_soft_placed_rate":   float,  # 「相手なら残る」が3着内に来た割合
    }
    """
    completed = [r for r in records if _has_result(r)]
    n_with_danger    = 0
    truly_correct    = 0
    truly_total      = 0
    soft_placed      = 0
    soft_total       = 0

    for r in completed:
        result_data  = r.get("result", {})
        finish_order = result_data.get("finish_order", [])
        top3         = set(finish_order[:3])
        has_danger   = False

        for h in r.get("horses", []):
            if not h.get("is_danger_favorite"):
                continue
            has_danger = True
            name = h["horse_name"]
            if h.get("is_truly_dangerous", True):
                truly_total += 1
                if name not in top3:
                    truly_correct += 1
            else:
                soft_total += 1
                if name in top3:
                    soft_placed += 1

        if has_danger:
            n_with_danger += 1

    return {
        "n_races_with_danger":   n_with_danger,
        "danger_truly_correct_rate": round(truly_correct / max(truly_total, 1), 4),
        "danger_soft_placed_rate":   round(soft_placed   / max(soft_total, 1),  4),
        "truly_total":               truly_total,
        "soft_total":                soft_total,
    }


def summarize_value_horse(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    妙味馬を含む買い目の成績と、妙味馬が馬券内に来た頻度を集計する。
    """
    completed = [r for r in records if _has_result(r)]
    with_value    = []
    without_value = []

    value_in_money     = 0
    value_bought_hit   = 0
    value_total        = 0

    for r in completed:
        result_data  = r.get("result", {})
        finish_order = result_data.get("finish_order", [])
        top3         = set(finish_order[:3])
        tickets      = r.get("recommended_tickets", [])

        bought: set = set()
        for t in tickets:
            bought.update(t.get("combination", []))

        has_value_in_ticket = any(
            h.get("is_value_horse") and h["horse_name"] in bought
            for h in r.get("horses", [])
        )

        if has_value_in_ticket:
            with_value.append(r)
        else:
            without_value.append(r)

        for h in r.get("horses", []):
            if not h.get("is_value_horse"):
                continue
            value_total += 1
            name = h["horse_name"]
            if name in top3:
                value_in_money += 1
                if name in bought:
                    value_bought_hit += 1

    with_stats    = _base_stats(with_value)
    without_stats = _base_stats(without_value)

    value_wrong = value_total - value_in_money  # 妙味候補が着外に終わった回数（拾いすぎ指標）

    return {
        "value_in_money_rate":   round(value_in_money   / max(value_total, 1), 4),
        "value_wrong_rate":      round(value_wrong       / max(value_total, 1), 4),  # 拾いすぎ率
        "value_wrong_count":     value_wrong,                                         # 拾いすぎ回数
        "value_bought_hit_rate": round(value_bought_hit / max(value_total, 1), 4),
        "value_total":           value_total,
        "with_value_bet":        with_stats,
        "without_value_bet":     without_stats,
    }


def summarize_rescue_horse(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """取りこぼし注意馬の精度を集計する。"""
    completed = [r for r in records if _has_result(r)]
    in_money  = 0
    total     = 0
    ignored   = 0

    for r in completed:
        result_data  = r.get("result", {})
        finish_order = result_data.get("finish_order", [])
        top3         = set(finish_order[:3])
        tickets      = r.get("recommended_tickets", [])
        bought: set  = set()
        for t in tickets:
            bought.update(t.get("combination", []))

        for h in r.get("horses", []):
            if not h.get("is_rescue_candidate"):
                continue
            total += 1
            name = h["horse_name"]
            if name in top3:
                in_money += 1
                if name not in bought:
                    ignored += 1

    return {
        "rescue_in_money_rate": round(in_money / max(total, 1), 4),
        "rescue_ignored_rate":  round(ignored  / max(in_money, 1), 4) if in_money > 0 else 0.0,
        "rescue_total":         total,
        "rescue_in_money":      in_money,
        "rescue_ignored":       ignored,
    }


# =========================================================
# セグメント別集計（グレード / 頭数）
# =========================================================

def summarize_by_field_size(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """頭数区分別の全体成績（hit_rate / roi / n_races など）。"""
    return _summarize_by_key(
        records,
        lambda r: _field_size_bucket(r.get("race_field_size") or 0),
        key_label="field_size",
    )


def summarize_value_by_segment(
    records: List[Dict[str, Any]],
    segment_key: str = "race_grade",
) -> List[Dict[str, Any]]:
    """
    グレード・頭数などセグメント別に妙味馬精度を集計する。

    segment_key:
        "race_grade"   → G1/G2/G3/重賞別
        "field_size"   → 頭数バケット別（race_field_size から自動変換）
    """
    return _segment_rows(records, segment_key, _value_horse_core_stats)


def summarize_danger_by_segment(
    records: List[Dict[str, Any]],
    segment_key: str = "race_grade",
) -> List[Dict[str, Any]]:
    """
    グレード・頭数などセグメント別に危険人気馬判定精度を集計する。

    戻り値の danger_correct_rate が高いほど「切りの精度が高い」。
    """
    return _segment_rows(records, segment_key, _danger_core_stats)


def summarize_role_by_segment(
    records: List[Dict[str, Any]],
    segment_key: str = "race_grade",
) -> List[Dict[str, Any]]:
    """
    グレード・頭数などセグメント別に役割判定精度（head/axis/himo/fade）を集計する。

    戻り値: [{segment_key: k, "head": {...}, "axis": {...}, ...}, ...]
    """
    return _segment_rows(records, segment_key, summarize_role_performance)


# =========================================================
# バックテスト感度分析
# =========================================================

# 馬レベルのフィールド（horses 配列内のキー）
_HORSE_LEVEL_KEYS = {"value_gap", "ai_win_prob", "stable_score", "win_odds", "top3_prob"}


def compare_threshold_sensitivity(
    records: List[Dict[str, Any]],
    threshold_key: str,
    values: List[float],
) -> List[Dict[str, Any]]:
    """
    閾値感度分析: threshold_key を異なる閾値で絞り込んだときの成績比較。

    Parameters
    ----------
    records       : 全レース記録
    threshold_key : フィルタ対象フィールド名
                    - 馬レベル (horses 配列内): "value_gap", "ai_win_prob",
                      "stable_score", "win_odds", "top3_prob"
                    - レースレベル: "upset_risk"（この場合は <= でフィルタ）,
                      その他は >= でフィルタ
    values        : 試す閾値リスト（昇順推奨）

    NOTE: 馬レベルキーの場合、フィルタは `is_value_horse=True` かつ
    `threshold_key >= thr` の馬が 1頭以上存在するレースを抽出する。
    `is_value_horse` フラグは value_ai.detect_value_horses() の結果に依存するため、
    閾値の実効範囲は value_ai 側の VALUE_GAP_MIN 等の設定にも影響を受ける点に注意。

    Returns
    -------
    各閾値ごとに {"threshold": ..., n_races, n_hit, hit_rate, roi, ...} を格納したリスト
    """
    completed = [r for r in records if _has_result(r)]

    rows = []
    for thr in values:
        if threshold_key in _HORSE_LEVEL_KEYS:
            # 妙味馬フラグを持ち threshold_key >= thr の馬が1頭以上いるレースを抽出
            filtered = [
                r for r in completed
                if any(
                    h.get("is_value_horse")
                    and float(h.get(threshold_key) or 0) >= thr
                    for h in r.get("horses", [])
                )
            ]
        elif threshold_key == "upset_risk":
            # upset_risk は低い方が安全 → <= でフィルタ
            filtered = [r for r in completed if float(r.get("upset_risk") or 1.0) <= thr]
        else:
            filtered = [r for r in completed if float(r.get(threshold_key) or 0) >= thr]

        stats = _base_stats(filtered)
        rows.append({"threshold": thr, **stats})

    return rows


# =========================================================
# シグナル別的中率トラッキング（4-A）
# =========================================================

def summarize_signal_hit_rates(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    シグナル条件ごとの的中率（top3到達率）を集計する。

    追跡対象:
      - positive_factor別: 「脚質」シグナルが positive だった馬のうち top3 に入った割合
      - negative_factor別: 「年齢」シグナルが negative だった馬のうち top3 を外した割合
      - signal_total_adjust 帯別: 補正合計値の大きさと的中率の相関

    Returns
    -------
    {
        "positive_by_factor": [{"factor": str, "n_horses": int, "top3_rate": float, "lift": float}, ...],
        "negative_by_factor": [{"factor": str, "n_horses": int, "outside_rate": float, "lift": float}, ...],
        "by_adjust_band":     [{"band": str, "n_horses": int, "top3_rate": float}, ...],
        "base_top3_rate":     float,
        "data_horses":        int,
    }
    """
    completed = [r for r in records if _has_result(r)]
    if not completed:
        return {
            "positive_by_factor": [],
            "negative_by_factor": [],
            "by_adjust_band": [],
            "base_top3_rate": 0.0,
            "data_horses": 0,
        }

    # 全馬のフラットなリストを構築
    all_horse_rows: List[Dict[str, Any]] = []
    for rec in completed:
        finish_order = (rec.get("result") or {}).get("finish_order", [])
        top3_names   = set(finish_order[:3])
        for h in rec.get("horses", []):
            all_horse_rows.append({
                **h,
                "_is_top3": h.get("horse_name", "") in top3_names,
            })

    if not all_horse_rows:
        return {
            "positive_by_factor": [],
            "negative_by_factor": [],
            "by_adjust_band": [],
            "base_top3_rate": 0.0,
            "data_horses": 0,
        }

    total_n     = len(all_horse_rows)
    total_top3  = sum(1 for h in all_horse_rows if h["_is_top3"])
    base_top3   = round(total_top3 / max(total_n, 1), 4)

    # ── ① positive factor 別 ────────────────────────────────
    pos_buckets: Dict[str, List[bool]] = defaultdict(list)
    for h in all_horse_rows:
        for fac in h.get("signal_positive_factors", []):
            if fac:
                pos_buckets[fac].append(h["_is_top3"])

    positive_by_factor = []
    for fac, hits in sorted(pos_buckets.items()):
        n    = len(hits)
        if n < MIN_STAT_SAMPLES:
            continue
        rate = round(sum(hits) / n, 4)
        positive_by_factor.append({
            "factor":    fac,
            "n_horses":  n,
            "top3_rate": rate,
            "lift":      round(rate - base_top3, 4),
        })
    positive_by_factor.sort(key=lambda x: -x["lift"])

    # ── ② negative factor 別 ────────────────────────────────
    neg_buckets: Dict[str, List[bool]] = defaultdict(list)
    for h in all_horse_rows:
        for fac in h.get("signal_negative_factors", []):
            if fac:
                neg_buckets[fac].append(h["_is_top3"])

    negative_by_factor = []
    for fac, hits in sorted(neg_buckets.items()):
        n    = len(hits)
        if n < MIN_STAT_SAMPLES:
            continue
        rate = round(sum(hits) / n, 4)
        outside_rate = round(1.0 - rate, 4)
        negative_by_factor.append({
            "factor":       fac,
            "n_horses":     n,
            "top3_rate":    rate,
            "outside_rate": outside_rate,      # top3 を外した率（高いほど消し精度が高い）
            "lift":         round(base_top3 - rate, 4),  # positive = base より低い（良い）
        })
    negative_by_factor.sort(key=lambda x: -x["lift"])

    # ── ③ signal_total_adjust 帯別 ───────────────────────────
    def _adjust_band(adj: float) -> str:
        if adj >= 0.06:   return "+0.06以上（強い追い風）"
        if adj >= 0.03:   return "+0.03〜+0.06（追い風）"
        if adj >= 0.01:   return "+0.01〜+0.03（弱い追い風）"
        if adj >= -0.01:  return "±0.01（中立）"
        if adj >= -0.03:  return "-0.01〜-0.03（弱い懸念）"
        if adj >= -0.06:  return "-0.03〜-0.06（懸念）"
        return "-0.06以下（強い懸念）"

    band_buckets: Dict[str, List[bool]] = defaultdict(list)
    for h in all_horse_rows:
        adj = float(h.get("signal_total_adjust") or 0.0)
        band_buckets[_adjust_band(adj)].append(h["_is_top3"])

    by_adjust_band = []
    band_order = [
        "+0.06以上（強い追い風）", "+0.03〜+0.06（追い風）", "+0.01〜+0.03（弱い追い風）",
        "±0.01（中立）",
        "-0.01〜-0.03（弱い懸念）", "-0.03〜-0.06（懸念）", "-0.06以下（強い懸念）",
    ]
    for band in band_order:
        hits = band_buckets.get(band, [])
        if not hits:
            continue
        n    = len(hits)
        rate = round(sum(hits) / n, 4)
        by_adjust_band.append({
            "band":      band,
            "n_horses":  n,
            "top3_rate": rate,
            "lift":      round(rate - base_top3, 4),
        })

    return {
        "positive_by_factor": positive_by_factor,
        "negative_by_factor": negative_by_factor,
        "by_adjust_band":     by_adjust_band,
        "base_top3_rate":     base_top3,
        "data_horses":        total_n,
    }


# =========================================================
# ドリフト検出（4-B）
# =========================================================

# 直近ウィンドウの最小サンプル数
DRIFT_MIN_WINDOW: int = 5

# ドリフト警告の ROI 閾値（直近がこれ以下で警告）
DRIFT_ROI_WARN: float = 0.80

# 的中率の急落検出しきい値
DRIFT_HITRATE_DROP: float = 0.15  # 全体比で -15%p 以上の落ち込み


def detect_performance_drift(
    records: List[Dict[str, Any]],
    window: int = 10,
) -> Dict[str, Any]:
    """
    直近N件レースの成績が全体平均から有意に悪化していないかを検出する。

    Parameters
    ----------
    records : 保存済みレース一覧（結果入力済みを優先）
    window  : 直近ウィンドウサイズ（デフォルト10件）

    Returns
    -------
    {
        "status":          "ok" | "warning" | "alert",
        "message":         str,
        "overall_roi":     float | None,
        "recent_roi":      float | None,
        "overall_hitrate": float,
        "recent_hitrate":  float,
        "n_recent":        int,
        "n_overall":       int,
        "drift_detected":  bool,
    }
    """
    completed = sorted(
        [r for r in records if _has_result(r)],
        key=lambda r: r.get("race_date", ""),
    )

    if len(completed) < DRIFT_MIN_WINDOW:
        return {
            "status":          "insufficient_data",
            "message":         f"ドリフト検出には最低{DRIFT_MIN_WINDOW}件の結果入力済みレースが必要です（現在{len(completed)}件）",
            "overall_roi":     None,
            "recent_roi":      None,
            "overall_hitrate": 0.0,
            "recent_hitrate":  0.0,
            "n_recent":        len(completed),
            "n_overall":       len(completed),
            "drift_detected":  False,
        }

    # 全体統計
    overall_stats = _base_stats(completed)
    overall_roi   = overall_stats.get("roi")
    overall_hr    = overall_stats.get("hit_rate", 0.0)

    # 直近ウィンドウ
    recent = completed[-window:]
    recent_stats = _base_stats(recent)
    recent_roi   = recent_stats.get("roi")
    recent_hr    = recent_stats.get("hit_rate", 0.0)

    drift_detected = False
    warnings: List[str] = []

    # ROI 急落チェック
    if recent_roi is not None and overall_roi is not None:
        if recent_roi < DRIFT_ROI_WARN:
            drift_detected = True
            warnings.append(
                f"直近{len(recent)}件 ROI: {recent_roi*100:.1f}% "
                f"（全体: {overall_roi*100:.1f}%）― 回収率が低下しています"
            )

    # 的中率急落チェック
    hr_drop = recent_hr - overall_hr
    if hr_drop < -DRIFT_HITRATE_DROP:
        drift_detected = True
        warnings.append(
            f"直近{len(recent)}件 的中率: {recent_hr*100:.1f}% "
            f"（全体: {overall_hr*100:.1f}%）― 的中率が {abs(hr_drop)*100:.1f}%p 低下"
        )

    if drift_detected:
        status  = "alert" if len(warnings) >= 2 else "warning"
        message = " / ".join(warnings)
    else:
        status  = "ok"
        message = f"直近{len(recent)}件の成績は安定しています（ROI: {recent_roi*100:.1f}%）" if recent_roi else "成績は安定しています"

    return {
        "status":          status,
        "message":         message,
        "overall_roi":     overall_roi,
        "recent_roi":      recent_roi,
        "overall_hitrate": round(overall_hr, 4),
        "recent_hitrate":  round(recent_hr, 4),
        "n_recent":        len(recent),
        "n_overall":       len(completed),
        "drift_detected":  drift_detected,
    }


# =========================================================
# まとめて全集計（UI から呼びやすいショートカット）
# =========================================================

def build_full_analytics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """全集計を1つの dict にまとめて返す。"""
    base = {
        "overall":              summarize_performance(records),
        "by_bet_type":          summarize_by_bet_type(records),
        "by_structure":         summarize_by_race_structure(records),
        "by_ev_type":           summarize_by_ev_type(records),
        "by_grade":             summarize_by_grade(records),
        "by_field_size":        summarize_by_field_size(records),        # 頭数別全体成績
        "danger_cutoff":        summarize_danger_cutoff(records),
        "value_horse":          summarize_value_horse(records),
        "rescue_horse":         summarize_rescue_horse(records),
        # フェーズ5-2追加
        "role_performance":     summarize_role_performance(records),
        "pass_judgment":        summarize_pass_judgment(records),
        "tag_ranking":          aggregate_review_tags(records),
        # セグメント別詳細（改善比較用）
        "value_by_grade":       summarize_value_by_segment(records, "race_grade"),
        "value_by_field_size":  summarize_value_by_segment(records, "field_size"),
        "danger_by_grade":      summarize_danger_by_segment(records, "race_grade"),
        "danger_by_field_size": summarize_danger_by_segment(records, "field_size"),
        "role_by_grade":        summarize_role_by_segment(records, "race_grade"),
    }
    base["improvement_comments"] = generate_improvement_comments(base)
    base["signal_hit_rates"]     = summarize_signal_hit_rates(records)
    base["drift_detection"]      = detect_performance_drift(records)
    return base
