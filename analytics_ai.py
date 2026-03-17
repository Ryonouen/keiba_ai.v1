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
# 内部ヘルパー
# =========================================================

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

    return {
        "value_in_money_rate":   round(value_in_money   / max(value_total, 1), 4),
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
# まとめて全集計（UI から呼びやすいショートカット）
# =========================================================

def build_full_analytics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """全集計を1つの dict にまとめて返す。"""
    base = {
        "overall":          summarize_performance(records),
        "by_bet_type":      summarize_by_bet_type(records),
        "by_structure":     summarize_by_race_structure(records),
        "by_ev_type":       summarize_by_ev_type(records),
        "by_grade":         summarize_by_grade(records),
        "danger_cutoff":    summarize_danger_cutoff(records),
        "value_horse":      summarize_value_horse(records),
        "rescue_horse":     summarize_rescue_horse(records),
        # フェーズ5-2追加
        "role_performance": summarize_role_performance(records),
        "pass_judgment":    summarize_pass_judgment(records),
        "tag_ranking":      aggregate_review_tags(records),
    }
    base["improvement_comments"] = generate_improvement_comments(base)
    return base
