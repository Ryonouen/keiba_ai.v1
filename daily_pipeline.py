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
import math as _math
import logging as _logging
import random
import signal as _signal
import sys as _sys
import time
from datetime import datetime, timedelta as _td
from datetime import datetime as _dt
from typing import Any, Dict, List, Optional

import pipeline_store

_logger = _logging.getLogger(__name__)

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
            pipeline_store.save_prediction_v2(
                race_id=race_id,
                race_meta=race_meta,
                features=features,
                ev_table=ev_table,
                race_structure=race_structure,
                danger_v2=danger_v2,
                analysis_date=date_str,
            )

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


# =========================================================
# Phase 4: 週末集計
# =========================================================

def summarize_weekend_performance(date_list: List[str]) -> Dict[str, Any]:
    """
    複数日の bet_outcomes を集計し、券種別の的中率・回収率を返す。

    Parameters
    ----------
    date_list : ["20250405", "20250406"] 形式

    Returns
    -------
    {
        "dates":         List[str],
        "total_races":   int,          # 買い目が存在したレース数
        "total_bets":    int,          # 総買い目数
        "total_stake":   int,          # 総投資額（円）
        "total_payout":  int,          # 総払戻額（円）
        "total_roi":     float,        # 総回収率（払戻/投資）
        "by_bet_type":   {             # 券種別
            "<bet_key>": {
                "label":    str,
                "bets":     int,
                "hits":     int,
                "stake":    int,
                "payout":   int,
                "hit_rate": float,
                "roi":      float,
            },
            ...
        },
    }
    """
    all_outcomes    = pipeline_store.load_all_bet_outcomes()
    all_predictions = pipeline_store.load_all_predictions()

    # predictions の analysis_date（YYYYMMDD）を使って対象レースをフィルタする。
    # race_id のフォーマットは YYYY+venue(2)+round(2)+day(2)+race(2) なので
    # race_id[:8] は日付ではない点に注意。
    target_dates = set(date_list)
    target_race_ids: List[str] = [
        race_id
        for race_id, outcomes in all_outcomes.items()
        if all_predictions.get(race_id, {}).get("analysis_date", "") in target_dates
        and outcomes  # 空リストは除外
    ]

    summary: Dict[str, Any] = {
        "dates":        date_list,
        "total_races":  len(target_race_ids),
        "total_bets":   0,
        "total_stake":  0,
        "total_payout": 0,
        "total_roi":    0.0,
        "by_bet_type":  {},
    }

    for race_id in target_race_ids:
        for outcome in all_outcomes.get(race_id, []):
            bet_key = str(outcome.get("bet_type") or "")
            label   = str(outcome.get("bet_type_label") or bet_key)
            stake   = int(outcome.get("stake") or 0)
            payout  = int(outcome.get("payout") or 0)
            hit     = bool(outcome.get("hit"))

            summary["total_bets"]   += 1
            summary["total_stake"]  += stake
            summary["total_payout"] += payout

            if bet_key not in summary["by_bet_type"]:
                summary["by_bet_type"][bet_key] = {
                    "label": label, "bets": 0, "hits": 0,
                    "stake": 0, "payout": 0, "hit_rate": 0.0, "roi": 0.0,
                }
            bt = summary["by_bet_type"][bet_key]
            bt["bets"]   += 1
            bt["stake"]  += stake
            bt["payout"] += payout
            if hit:
                bt["hits"] += 1

    # 比率の計算
    if summary["total_stake"] > 0:
        summary["total_roi"] = round(summary["total_payout"] / summary["total_stake"], 4)

    for bt in summary["by_bet_type"].values():
        if bt["bets"] > 0:
            bt["hit_rate"] = round(bt["hits"] / bt["bets"], 4)
        if bt["stake"] > 0:
            bt["roi"] = round(bt["payout"] / bt["stake"], 4)

    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    """券種別集計をターミナルに出力する。"""
    print(f"\n{'='*60}", flush=True)
    print(f"  週末集計レポート: {', '.join(summary['dates'])}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  総レース数  : {summary['total_races']}", flush=True)
    print(f"  総買い目数  : {summary['total_bets']}", flush=True)
    print(f"  総投資額    : ¥{summary['total_stake']:,}", flush=True)
    print(f"  総払戻額    : ¥{summary['total_payout']:,}", flush=True)
    print(f"  総回収率    : {summary['total_roi']:.1%}", flush=True)
    print(f"\n  [券種別]", flush=True)
    print(f"  {'券種':<30} {'点数':>5} {'的中':>5} {'的中率':>7} {'回収率':>7}", flush=True)
    print(f"  {'-'*56}", flush=True)
    for bt in sorted(summary["by_bet_type"].values(), key=lambda x: -x["roi"]):
        print(
            f"  {bt['label']:<30} {bt['bets']:>5} {bt['hits']:>5}"
            f" {bt['hit_rate']:>7.1%} {bt['roi']:>7.1%}",
            flush=True,
        )
    print(f"{'='*60}\n", flush=True)


# =========================================================
# CSV エクスポート
# =========================================================

def export_summary_csv(summary: Dict[str, Any], out_path: str = "") -> str:
    """
    週末集計をCSVに出力する。

    Parameters
    ----------
    summary  : summarize_weekend_performance() の戻り値
    out_path : 出力パス。省略時は pipeline_summary_YYYYMMDD.csv

    Returns
    -------
    出力ファイルパス
    """
    import csv
    import os

    dates_str = "_".join(summary.get("dates") or ["unknown"])
    if not out_path:
        out_path = f"pipeline_summary_{dates_str}.csv"

    rows = []
    for bet_key, bt in sorted(summary["by_bet_type"].items()):
        rows.append({
            "date":       dates_str,
            "bet_type":   bet_key,
            "label":      bt["label"],
            "bets":       bt["bets"],
            "hits":       bt["hits"],
            "stake":      bt["stake"],
            "payout":     bt["payout"],
            "hit_rate":   round(bt["hit_rate"], 4),
            "roi":        round(bt["roi"], 4),
        })

    if not rows:
        print("[export_summary_csv] 集計データなし。出力スキップ。", flush=True)
        return out_path

    fieldnames = ["date", "bet_type", "label", "bets", "hits",
                  "stake", "payout", "hit_rate", "roi"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[export_summary_csv] 保存: {out_path} ({len(rows)}件)", flush=True)
    return out_path


def get_race_meta_by_date(date_str: str) -> List[Dict[str, Any]]:
    """
    指定日の全レースID＋メタ情報（開催場・R番号）を返す。

    Parameters
    ----------
    date_str : "20250406" 形式

    Returns
    -------
    List of {
        "race_id":   str,
        "venue":     str,   # 開催場名（例: "阪神"）
        "race_no":   int,   # レース番号
        "venue_code": str,  # 場所コード（例: "09"）
    }
    """
    _VENUE_NAME: Dict[str, str] = {
        "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
        "05": "東京", "06": "中山", "07": "中京", "08": "京都",
        "09": "阪神", "10": "小倉",
    }
    race_ids = get_race_ids_by_date(date_str)
    result = []
    for rid in race_ids:
        venue_code = rid[4:6]
        race_no    = int(rid[10:12]) if rid[10:12].isdigit() else 0
        result.append({
            "race_id":    rid,
            "venue":      _VENUE_NAME.get(venue_code, f"場所{venue_code}"),
            "race_no":    race_no,
            "venue_code": venue_code,
        })
    return result


# =========================================================
# Phase 2b: オッズ更新・再予測
# =========================================================

def _run_lgbm_prediction(features: List[Dict[str, Any]]) -> List[float]:
    """
    LightGBM で win_prob を再計算し、正規化された確率リストを返す。
    モデルがない場合はフォールバック（一様分布）を返す。
    テストで monkeypatch しやすいよう分離。
    """
    from race_ai_engine import predict_win_probability_with_model, MODEL_FILE
    probs = predict_win_probability_with_model(features, MODEL_FILE)
    if probs is not None:
        return probs
    n = len(features)
    return [1.0 / n] * n if n > 0 else []


def _recalc_downstream(features: List[Dict[str, Any]], new_probs: List[float]) -> None:
    """
    オッズ更新後に win_prob とその downstream フィールドを in-place 更新する。

    更新する: win_prob, place_prob, fair_win_odds, fair_place_odds,
              win_ev, place_ev, win_market_edge, place_market_edge,
              odds_distortion_index, value_flag, win_value_label,
              place_value_label, expected_value_score, bet_suitability
    更新しない: feat_gate/feat_age 等の発走当日不変フィールド、place_odds
    """
    try:
        from race_ai_engine import (
            estimate_place_prob, fair_odds, calc_expected_value, calc_market_edge,
            calc_odds_distortion, calc_expected_value_score, classify_bet_suitability,
            classify_value_label,
        )
    except ImportError:
        # フォールバック: win_prob のみ更新
        for f, p in zip(features, new_probs):
            f["win_prob"] = round(p, 4)
        return

    for f, p in zip(features, new_probs):
        f["win_prob"]          = round(p, 4)
        f["place_prob"]        = estimate_place_prob(p)
        f["fair_win_odds"]     = fair_odds(p)
        f["fair_place_odds"]   = fair_odds(f["place_prob"])
        f["win_ev"]            = calc_expected_value(p, f.get("win_odds"))
        f["place_ev"]          = calc_expected_value(f["place_prob"], f.get("place_odds"))
        f["win_market_edge"]   = calc_market_edge(p, f.get("win_odds"))
        f["place_market_edge"] = calc_market_edge(f["place_prob"], f.get("place_odds"))
        f["odds_distortion_index"] = calc_odds_distortion(f)
        odi = f["odds_distortion_index"]
        f["value_flag"] = (
            "SUPER_VALUE" if odi >= 1.4 else
            "VALUE"       if odi >= 1.15 else
            "NORMAL"
        )
        f["win_value_label"]       = classify_value_label(f["win_market_edge"], f["win_ev"])
        f["place_value_label"]     = classify_value_label(f["place_market_edge"], f["place_ev"])
        f["expected_value_score"]  = calc_expected_value_score(f)
        f["bet_suitability"]       = classify_bet_suitability(f)


def update_race_odds(race_id: str) -> Dict[str, Any]:
    """
    1 レースのオッズを取得し、予測と買い目を更新する。

    Returns
    -------
    {
        "race_id":  str,
        "status":   str,   # fetch_win_odds のステータス
        "coverage": float,
        "version_before": int,
        "version_after":  int,
    }
    """
    import odds_fetcher
    from value_ai import assign_roles, recommend_betmaster_plans

    pred = pipeline_store.load_prediction(race_id)
    if pred is None:
        _logger.error("[update_race_odds] %s | 予測データなし", race_id)
        return {"race_id": race_id, "status": "failed", "coverage": 0.0,
                "version_before": 0, "version_after": 0}

    horse_number_map: Dict[str, str] = pred.get("horse_number_map") or {}
    ev_table         = pred.get("ev_table") or []
    race_structure   = pred.get("race_structure") or {}
    danger_v2        = pred.get("danger_v2") or []
    version_before   = pred.get("prediction_version", 1)

    # オッズ取得
    status, new_odds = odds_fetcher.fetch_win_odds(race_id, horse_number_map)
    _logger.info("[update_race_odds] %s | status=%s | v%d",
                 race_id, status, version_before)

    if status in ("not_open", "api_failed", "selenium_failed", "failed"):
        _logger.info("[update_race_odds] %s | スキップ（%s）", race_id, status)
        return {"race_id": race_id, "status": status, "coverage": 0.0,
                "version_before": version_before, "version_after": version_before}

    # feature_dict を horses から復元
    features: List[Dict[str, Any]] = [
        dict(h["feature_dict"]) for h in pred.get("horses", [])
        if h.get("feature_dict")
    ]
    if not features:
        _logger.error("[update_race_odds] %s | feature_dict なし", race_id)
        return {"race_id": race_id, "status": "failed", "coverage": 0.0,
                "version_before": version_before, "version_after": version_before}

    # feat_win_odds_log と win_odds を更新
    for f in features:
        name = str(f.get("horse_name") or "")
        odds = new_odds.get(name) if new_odds else None
        if odds is not None:
            f["win_odds"] = odds
            f["feat_win_odds_log"] = round(_math.log(max(odds, 1.0)), 4)

    # 脚質 missing 馬の再試行
    missing_styles = [
        h["horse_name"] for h in pred.get("horses", [])
        if h.get("running_style_missing")
    ]
    if missing_styles:
        _logger.info("[update_race_odds] %s | 脚質 missing %d頭 → 再取得試行",
                     race_id, len(missing_styles))
        st, styles = odds_fetcher.fetch_newspaper_styles(race_id, horse_number_map)
        if styles:
            for f in features:
                name = str(f.get("horse_name") or "")
                if name in missing_styles and name in styles:
                    f["running_style"] = styles[name]
                    from race_ai_engine import _RUNNING_STYLE_ENC
                    f["feat_running_style_enc"] = _RUNNING_STYLE_ENC.get(
                        styles[name], 3)

    # LightGBM 再予測
    new_probs = _run_lgbm_prediction(features)
    if not new_probs:
        return {"race_id": race_id, "status": "failed", "coverage": 0.0,
                "version_before": version_before, "version_after": version_before}

    # downstream 再計算
    _recalc_downstream(features, new_probs)

    # ログ: top 馬の変化
    old_top = max(pred.get("horses", []),
                  key=lambda h: h.get("ai_win_prob", 0),
                  default=None)
    new_top = max(features, key=lambda f: f.get("win_prob", 0), default=None)
    if old_top and new_top:
        _logger.info(
            "[update_race_odds] %s | top: %s %.1f%%→%.1f%% @ %.1f倍",
            race_id,
            new_top.get("horse_name"),
            old_top.get("ai_win_prob", 0) * 100,
            new_top.get("win_prob", 0) * 100,
            new_top.get("win_odds") or 0,
        )

    # 役割・買い目 再計算
    horse_roles = assign_roles(features, ev_table, race_structure, danger_v2)
    plans = recommend_betmaster_plans(features, race_structure, horse_roles)
    bets  = generate_all_bets(race_id, plans)

    # 保存用 horses リスト（feature_dict も更新）
    old_horses_by_name = {h["horse_name"]: h for h in pred.get("horses", [])}
    updated_horses = []
    for f in features:
        name = str(f.get("horse_name") or "")
        old_h = old_horses_by_name.get(name, {})
        updated_horses.append({
            **old_h,
            "ai_win_prob": round(float(f.get("win_prob", 0)), 4),
            "win_odds":    f.get("win_odds"),
            "popularity":  f.get("feat_popularity"),
            "running_style":         f.get("running_style", "unknown"),
            "running_style_missing": f.get("running_style", "unknown") == "unknown",
            "feature_dict": dict(f),
        })

    total = len(horse_number_map)
    ok    = len(new_odds) if new_odds else 0
    coverage = ok / total if total > 0 else 0.0

    # store 更新
    pipeline_store.update_prediction_odds_in_store(
        race_id=race_id,
        new_odds_by_name=new_odds or {},
        updated_horses=updated_horses,
        odds_status=status,
        odds_source="api" if "selenium" not in status else "selenium",
        coverage_ratio=coverage,
    )
    pipeline_store.save_bet_suggestions(race_id, bets)

    _logger.info("[update_race_odds] %s | 完了 | v%d→v%d | coverage=%.0f%%",
                 race_id, version_before, version_before + 1, coverage * 100)

    return {
        "race_id":       race_id,
        "status":        status,
        "coverage":      coverage,
        "version_before": version_before,
        "version_after":  version_before + 1,
    }


# =========================================================
# Phase 2c: オッズ監視ループ
# =========================================================

def _get_update_targets(
    start_times: Dict[str, str],
    now: Any,
    updated_ids: set,
) -> List[str]:
    """
    更新ウィンドウ（発走 -35min ≤ now ≤ 発走 -20min）内にあり、
    まだ updated_ids に入っていないレース ID のリストを返す。
    """
    targets = []
    for race_id, sdt_str in start_times.items():
        if race_id in updated_ids:
            continue
        try:
            start = _dt.fromisoformat(sdt_str)
        except ValueError:
            continue
        lo = start - _td(minutes=35)
        hi = start - _td(minutes=20)
        if lo <= now <= hi:
            targets.append(race_id)
    return targets


def _should_exit(
    start_times: Dict[str, str],
    updated_ids: set,
    now: Any,
) -> bool:
    """
    停止条件:
      (A) 全レースが updated_ids に入り、かつ全発走時刻 < now
      (B) 全レースの start + 90min < now
    どちらかを満たせば True。
    """
    if not start_times:
        return True

    all_done = all(rid in updated_ids for rid in start_times)
    all_past = all(
        _dt.fromisoformat(sdt) < now
        for sdt in start_times.values()
        if sdt
    )
    if all_done and all_past:
        return True

    all_expired = all(
        _dt.fromisoformat(sdt) + _td(minutes=90) < now
        for sdt in start_times.values()
        if sdt
    )
    return all_expired


def watch_odds(date_str: str, poll_interval: int = 60) -> None:
    """
    指定日の全レースを監視し、発走 30 分前にオッズを自動更新する。

    フォアグラウンドブロッキング。SIGINT で正常終了。

    Parameters
    ----------
    date_str      : "20260405" 形式
    poll_interval : ポーリング間隔（秒）
    """
    start_times = pipeline_store.load_race_start_times(date_str)
    if not start_times:
        print(f"[watch_odds] {date_str} に分析済みレースがありません。"
              " 先に --analyze を実行してください。", flush=True)
        return

    total = len(start_times)
    print(f"[watch_odds] {date_str} | 監視対象: {total} レース", flush=True)
    for rid, sdt in sorted(start_times.items(), key=lambda x: x[1]):
        print(f"  {rid}  発走: {sdt[11:16]}", flush=True)
    print(flush=True)

    updated_ids: set = set()
    failed_ids:  set = set()

    # SIGINT ハンドラ
    def _on_sigint(*_):
        print(
            f"\n[watch_odds] 中断。更新済み {len(updated_ids)}/{total}件"
            f" | 失敗 {len(failed_ids)}件",
            flush=True,
        )
        _sys.exit(0)

    _signal.signal(_signal.SIGINT, _on_sigint)

    import time as _time

    while True:
        now = _dt.now()

        if _should_exit(start_times, updated_ids | failed_ids, now):
            break

        targets = _get_update_targets(start_times, now, updated_ids | failed_ids)
        for race_id in targets:
            sdt_str = start_times.get(race_id, "")
            print(f"[watch_odds] {now.strftime('%H:%M')} | {race_id}"
                  f" (発走 {sdt_str[11:16]}) | オッズ更新中...", flush=True)

            result = update_race_odds(race_id)
            status = result.get("status", "failed")

            print(
                f"[watch_odds] {now.strftime('%H:%M')} | {race_id}"
                f" | {status}"
                f" | coverage={result.get('coverage', 0):.0%}"
                f" | v{result.get('version_before')}→v{result.get('version_after')}",
                flush=True,
            )

            if status in ("success", "partial"):
                updated_ids.add(race_id)
            elif status == "not_open":
                pass   # 次サイクルで再試行
            else:
                failed_ids.add(race_id)

        _time.sleep(poll_interval)

    print(
        f"[watch_odds] {date_str} | 完了"
        f" | 更新済み {len(updated_ids)}/{total}件"
        f" | 失敗 {len(failed_ids)}件",
        flush=True,
    )


# =========================================================
# CLI エントリポイント
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="週末全レース自動分析パイプライン")
    parser.add_argument("--analyze",    metavar="DATE",  help="全レース分析（例: 20250406）")
    parser.add_argument("--evaluate",   metavar="DATE",  help="結果照合（例: 20250406）")
    parser.add_argument("--summarize",  metavar="DATES",
                        help="週末集計（カンマ区切り例: 20250405,20250406）")
    parser.add_argument("--export-csv", metavar="DATES",
                        help="週末集計をCSV出力（カンマ区切り例: 20250405,20250406）")
    parser.add_argument("--list",       metavar="DATE",  help="指定日のレース一覧を表示")
    parser.add_argument("--watch-odds",    metavar="DATE",
                        help="オッズ自動監視（例: 20250406）")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="watch-odds のポーリング間隔（秒、デフォルト: 60）")
    args = parser.parse_args()

    if args.analyze:
        run_daily_race_analysis(args.analyze)

    elif args.evaluate:
        evaluate_prediction_for_day(args.evaluate)

    elif args.summarize:
        dates   = [d.strip() for d in args.summarize.split(",")]
        summary = summarize_weekend_performance(dates)
        print_summary(summary)

    elif args.export_csv:
        dates   = [d.strip() for d in args.export_csv.split(",")]
        summary = summarize_weekend_performance(dates)
        print_summary(summary)
        export_summary_csv(summary)

    elif args.list:
        races = get_race_meta_by_date(args.list)
        print(f"{args.list} のレース一覧 ({len(races)}件):")
        for r in races:
            print(f"  {r['venue']} {r['race_no']:>2}R  {r['race_id']}")

    elif args.watch_odds:
        watch_odds(args.watch_odds, poll_interval=args.poll_interval)

    else:
        parser.print_help()
