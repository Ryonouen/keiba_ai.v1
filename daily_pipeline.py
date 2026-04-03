"""
daily_pipeline.py
週末全レース自動分析パイプライン

主要API:
  get_race_ids_by_date(date_str)         → List[str]
  run_daily_race_analysis(date_str)      → Dict[str, Any]  （分析 + 買い目保存）
  generate_all_bets(race_id, plans)      → List[Dict]
  evaluate_prediction_for_day(date_str)  → Dict[str, Any]  （結果取得 + 照合）
  summarize_weekend_performance(dates)   → Dict[str, Any]  （券種別集計）

使い方:
  python3 daily_pipeline.py --analyze 20250406
  python3 daily_pipeline.py --evaluate 20250406
  python3 daily_pipeline.py --summarize 20250405,20250406
"""
from __future__ import annotations

import argparse
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pipeline_store

# =========================================================
# 定数
# =========================================================

SHUTUBA_URL_TEMPLATE = "https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
SLEEP_MIN = 5.0   # analyze_race はSeleniumを使うため長めに
SLEEP_MAX = 10.0

# 券種名の正規化マップ（recommend_betmaster_plans → 保存用キー）
BET_TYPE_KEY_MAP: Dict[str, str] = {
    "単勝":                         "tansho",
    "複勝":                         "fukusho",
    "ワイド":                       "wide",
    "馬連（流し）":                 "umaren",
    "馬単フォーメーション":         "umatan",
    "三連複フォーメーション（AI絞り）": "sanrenpuku_ai",
    "三連複フォーメーション（全頭）":   "sanrenpuku_all",
    "三連単フォーメーション（AI絞り）": "sanrentan_ai",
    "三連単フォーメーション（全頭）":   "sanrentan_all",
}

# 払戻ルックアップ用: 保存bet_type → dividends dict キー
DIVIDEND_KEY_MAP: Dict[str, str] = {
    "tansho":        "単勝",
    "fukusho":       "複勝",
    "wide":          "ワイド",
    "umaren":        "馬連",
    "umatan":        "馬単",
    "sanrenpuku_ai": "3連複",
    "sanrenpuku_all":"3連複",
    "sanrentan_ai":  "3連単",
    "sanrentan_all": "3連単",
}

# check_bet_hit で使う bet_type 名（result_store.check_bet_hit の引数）
HIT_CHECK_TYPE_MAP: Dict[str, str] = {
    "tansho":        "単勝",
    "fukusho":       "複勝",
    "wide":          "ワイド",
    "umaren":        "馬連",
    "umatan":        "馬単",
    "sanrenpuku_ai": "3連複",
    "sanrenpuku_all":"3連複",
    "sanrentan_ai":  "3連単",
    "sanrentan_all": "3連単",
}


# =========================================================
# Phase 1: レースID取得
# =========================================================

def get_race_ids_by_date(date_str: str) -> List[str]:
    """
    指定日（YYYYMMDD）のJRA全レースIDを返す。

    Parameters
    ----------
    date_str : "20250406" 形式

    Returns
    -------
    List[str] — 12桁の race_id リスト
    """
    from dividend_scraper import fetch_race_ids_by_date
    return fetch_race_ids_by_date(date_str)


def _race_url(race_id: str) -> str:
    return SHUTUBA_URL_TEMPLATE.format(race_id=race_id)


# =========================================================
# Phase 2: 全レース分析 + 買い目生成・保存
# =========================================================

def generate_all_bets(race_id: str, plans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    recommend_betmaster_plans() の出力を保存用スキーマに変換する。

    Parameters
    ----------
    race_id : レースID（参照用）
    plans   : recommend_betmaster_plans() の戻り値

    Returns
    -------
    List[Dict] — bet_typeごと・チケットごとに分解されたフラットなリスト
    """
    result: List[Dict[str, Any]] = []
    for plan in plans:
        bet_type_raw = str(plan.get("bet_type") or "")
        bet_key = BET_TYPE_KEY_MAP.get(bet_type_raw, bet_type_raw)
        tickets = plan.get("tickets") or []
        if not tickets:
            continue  # confidence_ok=False の場合は tickets=[] → スキップ

        confidence = float(plan.get("confidence_score") or 0.0)
        reason     = str(plan.get("reason") or "")

        for ticket in tickets:
            combo = ticket.get("combination") or []
            stake = int(ticket.get("stake") or 100)
            result.append({
                "bet_type":           bet_key,
                "bet_type_label":     bet_type_raw,
                "bet_combination":    list(combo),
                "stake_amount":       stake,
                "selection_reason":   reason,
                "confidence":         round(confidence, 4),
                "expected_value":     None,   # 将来拡張用
                "implied_probability": None,  # 将来拡張用
            })
    return result


def run_daily_race_analysis(date_str: str) -> Dict[str, Any]:
    """
    指定日の全JRAレースを分析し、予測と買い目をストアに保存する。

    Parameters
    ----------
    date_str : "20250406" 形式

    Returns
    -------
    {
        "date":    str,
        "total":   int,   # 対象レース数
        "success": int,   # 分析成功数
        "skipped": int,   # スキップ数（エラー等）
        "errors":  List[{"race_id": str, "error": str}],
    }
    """
    from race_ai_engine import analyze_race
    from value_ai import recommend_betmaster_plans, assign_roles

    race_ids = get_race_ids_by_date(date_str)
    summary: Dict[str, Any] = {
        "date": date_str, "total": len(race_ids),
        "success": 0, "skipped": 0, "errors": [],
    }

    print(f"[{date_str}] 対象: {len(race_ids)} レース", flush=True)

    for i, race_id in enumerate(race_ids, 1):
        print(f"  [{i}/{len(race_ids)}] {race_id} 分析中...", flush=True)
        try:
            url    = _race_url(race_id)
            result = analyze_race(url, headless=True)

            features       = result.get("features") or []
            race_meta      = result.get("race_meta") or {}
            race_structure = result.get("race_structure") or {}
            ev_table       = result.get("ev_table") or []
            danger_v2      = result.get("danger_favorites_v2") or []

            if not features:
                summary["skipped"] += 1
                summary["errors"].append({"race_id": race_id, "error": "features empty"})
                print(f"    → features 空。スキップ", flush=True)
                continue

            # 役割割当（recommend_betmaster_plans に必要）
            horse_roles = assign_roles(features, ev_table, race_structure, danger_v2)

            # 予測保存（analysis_date を渡す → summarize のフィルタに使用）
            pipeline_store.save_prediction(race_id, race_meta, features,
                                           analysis_date=date_str)

            # 全券種買い目生成・保存
            plans = recommend_betmaster_plans(features, race_structure, horse_roles)
            bets  = generate_all_bets(race_id, plans)
            pipeline_store.save_bet_suggestions(race_id, bets)

            summary["success"] += 1
            print(f"    → 完了（{len(bets)} 買い目）", flush=True)

        except Exception as e:
            summary["skipped"] += 1
            summary["errors"].append({"race_id": race_id, "error": str(e)})
            print(f"    → エラー: {e}", flush=True)

        if i < len(race_ids):
            time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

    print(f"[{date_str}] 完了: 成功 {summary['success']}/{summary['total']}", flush=True)
    return summary


# =========================================================
# Phase 3: 結果取得・的中照合
# =========================================================

def _lookup_payout(
    bet_key: str,
    dividends: Dict[str, Any],
) -> int:
    """
    的中した買い目の払戻額（100円あたり）を dividends から取得する。
    取得できない場合は 0 を返す。

    ワイド・複勝のように複数値がある場合は最小値（保守的）を使う。
    """
    div_key = DIVIDEND_KEY_MAP.get(bet_key)
    if not div_key:
        return 0
    val = dividends.get(div_key)
    if val is None:
        return 0
    if isinstance(val, list):
        positives = [v for v in val if isinstance(v, (int, float)) and v > 0]
        return int(min(positives)) if positives else 0
    return int(val)


def evaluate_single_race(race_id: str) -> List[Dict[str, Any]]:
    """
    1レース分の予測照合を行い、bet_outcomes リストを返す（ストアには保存しない）。

    Parameters
    ----------
    race_id : レースID

    Returns
    -------
    List[Dict] with keys:
        bet_type, bet_combination, stake, hit, payout, roi
    """
    from result_store import check_bet_hit

    bets        = pipeline_store.load_bet_suggestions(race_id)
    race_result = pipeline_store.load_pipeline_race_result(race_id)

    if not bets or not race_result:
        return []

    finish_order = race_result.get("finish_order") or []
    dividends    = race_result.get("dividends") or {}

    outcomes: List[Dict[str, Any]] = []
    for bet in bets:
        bet_key   = str(bet.get("bet_type") or "")
        hit_type  = HIT_CHECK_TYPE_MAP.get(bet_key, bet_key)
        combo     = bet.get("bet_combination") or []
        stake     = int(bet.get("stake_amount") or 100)

        # combination を result_store.check_bet_hit の ticket 形式に変換
        ticket = {"combination": combo}
        hit    = check_bet_hit(hit_type, [ticket], finish_order)
        payout = _lookup_payout(bet_key, dividends) if hit else 0
        roi    = round(payout / stake, 4) if stake > 0 else 0.0

        outcomes.append({
            "bet_type":        bet_key,
            "bet_type_label":  bet.get("bet_type_label", bet_key),
            "bet_combination": combo,
            "stake":           stake,
            "hit":             hit,
            "payout":          payout,
            "roi":             roi,
        })

    return outcomes


def evaluate_prediction_for_day(date_str: str) -> Dict[str, Any]:
    """
    指定日のレース結果を scrape_race_result() で取得し、
    予測と照合して bet_outcomes に保存する。

    Parameters
    ----------
    date_str : "20250406" 形式

    Returns
    -------
    {
        "date":    str,
        "total":   int,
        "success": int,
        "skipped": int,
        "errors":  List[{"race_id": str, "error": str}],
    }
    """
    from dividend_scraper import scrape_race_result

    race_ids = get_race_ids_by_date(date_str)
    summary: Dict[str, Any] = {
        "date": date_str, "total": len(race_ids),
        "success": 0, "skipped": 0, "errors": [],
    }

    print(f"[{date_str}] 結果照合: {len(race_ids)} レース", flush=True)

    for i, race_id in enumerate(race_ids, 1):
        print(f"  [{i}/{len(race_ids)}] {race_id} 結果取得中...", flush=True)
        try:
            # 買い目が保存されていないレースはスキップ
            if not pipeline_store.load_bet_suggestions(race_id):
                summary["skipped"] += 1
                print(f"    → 買い目データなし。スキップ", flush=True)
                continue

            result = scrape_race_result(race_id)
            if result is None:
                summary["skipped"] += 1
                print(f"    → 結果なし（未開催/取得失敗）。スキップ", flush=True)
                continue

            pipeline_store.save_pipeline_race_result(race_id, result)

            outcomes = evaluate_single_race(race_id)
            pipeline_store.save_bet_outcomes(race_id, outcomes)

            hit_count = sum(1 for o in outcomes if o["hit"])
            summary["success"] += 1
            print(f"    → 完了（{len(outcomes)} 買い目 / {hit_count} 的中）", flush=True)

        except Exception as e:
            summary["errors"].append({"race_id": race_id, "error": str(e)})
            print(f"    → エラー: {e}", flush=True)

        if i < len(race_ids):
            time.sleep(random.uniform(1.5, 3.0))

    return summary
