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
  python3 daily_pipeline.py --analyze-missing 20250406 --analyze-timeout 420
  python3 daily_pipeline.py --evaluate 20250406
  python3 daily_pipeline.py --summarize 20250405,20250406
"""
from __future__ import annotations

import argparse
import math as _math
import logging as _logging
import random
import signal as _signal
import subprocess as _subprocess
import sys as _sys
import time
from datetime import datetime as _dt, timedelta as _td
from typing import Any, Dict, List, Optional

import pipeline_store
from kelly_staking import apply_kelly_to_bets

_logger = _logging.getLogger(__name__)

# =========================================================
# 定数
# =========================================================

SHUTUBA_URL_TEMPLATE = "https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
SLEEP_MIN = 5.0   # analyze_race はSeleniumを使うため長めに
SLEEP_MAX = 10.0

import os as _os_dp
_HERE_DP = _os_dp.path.dirname(_os_dp.path.abspath(__file__))
EXPANDED_MODEL_FILE: str = _os_dp.path.join(_HERE_DP, "keiba_lgbm_model_expanded.txt")

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
HIT_CHECK_TYPE_MAP = DIVIDEND_KEY_MAP


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

MAX_RECOMMENDED_BET_TYPES = 2  # 1レースで推奨する券種の最大数


def generate_all_bets(
    race_id: str,
    plans: List[Dict[str, Any]],
    kelly_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    recommend_betmaster_plans() の出力を保存用スキーマに変換する。
    confidence_ok=True の券種を confidence_score 降順で最大 MAX_RECOMMENDED_BET_TYPES 種に絞る。

    Parameters
    ----------
    race_id : レースID（参照用）
    plans   : recommend_betmaster_plans() の戻り値

    Returns
    -------
    List[Dict] — bet_typeごと・チケットごとに分解されたフラットなリスト
    """
    # confidence_ok=True のプランのみ、confidence_score 降順で上位2種に絞る
    valid_plans = [p for p in plans if (p.get("tickets") or [])]
    valid_plans.sort(key=lambda p: float(p.get("confidence_score") or 0.0), reverse=True)
    selected_plans = valid_plans[:MAX_RECOMMENDED_BET_TYPES]

    result: List[Dict[str, Any]] = []
    for plan in selected_plans:
        bet_type_raw = str(plan.get("bet_type") or "")
        bet_key = BET_TYPE_KEY_MAP.get(bet_type_raw, bet_type_raw)
        tickets = plan.get("tickets") or []
        if not tickets:
            continue  # confidence_ok=False の場合は tickets=[] → スキップ

        confidence = float(plan.get("confidence_score") or 0.0)
        reason     = str(plan.get("reason") or "")

        # 単勝EVは plan に _horse_win_prob / _horse_win_odds が付いている場合のみ計算
        _win_prob = plan.get("_horse_win_prob")
        _win_odds = plan.get("_horse_win_odds")
        if bet_type_raw == "単勝" and _win_prob and _win_odds:
            _ev = round(float(_win_prob) * float(_win_odds) - 1.0, 4)
        else:
            _ev = None

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
                "expected_value":     _ev,
                "implied_probability": None,
                "_win_prob":          plan.get("_horse_win_prob"),
                "_win_odds":          plan.get("_horse_win_odds"),
            })

    # 同一券種・同一組み合わせの重複排除
    # sanrenpuku_ai/sanrenpuku_all など実際に同じ馬券になるものを DIVIDEND_KEY_MAP で正規化
    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for b in result:
        actual_type = DIVIDEND_KEY_MAP.get(b["bet_type"], b["bet_type"])
        key = (actual_type, tuple(b["bet_combination"]))
        if key not in seen:
            seen.add(key)
            deduped.append(b)
    return apply_kelly_to_bets(deduped, kelly_config)


def _inject_estimated_odds_recalc(features: List[Dict[str, Any]]) -> None:
    """
    推定オッズを features に注入した後に odds-dependent フィールドを再計算する。
    win_prob はそのまま保持し、win_ev / win_market_edge / value_flag 等だけ更新する。
    """
    try:
        from race_ai_engine import (
            calc_expected_value, calc_market_edge,
            calc_odds_distortion, calc_expected_value_score,
            classify_bet_suitability, classify_value_label,
        )
    except ImportError:
        return

    for f in features:
        p = float(f.get("win_prob") or 0.0)
        f["win_ev"]                = calc_expected_value(p, f.get("win_odds"))
        f["win_market_edge"]       = calc_market_edge(p, f.get("win_odds"))
        f["odds_distortion_index"] = calc_odds_distortion(f)
        odi = f["odds_distortion_index"]
        f["value_flag"] = (
            "SUPER_VALUE" if odi >= 1.4 else
            "VALUE"       if odi >= 1.15 else
            "NORMAL"
        )
        f["win_value_label"]       = classify_value_label(f["win_market_edge"], f["win_ev"])
        f["expected_value_score"]  = calc_expected_value_score(f)
        f["bet_suitability"]       = classify_bet_suitability(f)


def run_daily_race_analysis(
    date_str: str,
    force_refresh_cache: bool = False,
    use_requests: bool = False,
) -> Dict[str, Any]:
    """
    指定日の全JRAレースを分析し、予測と買い目をストアに保存する。

    Parameters
    ----------
    date_str : "20250406" 形式
    force_refresh_cache : True の場合、既存キャッシュを無視して再分析する
    use_requests : True の場合、Selenium を使わず requests + BeautifulSoup で取得する（高速・要確認）

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
    race_ids = get_race_ids_by_date(date_str)
    return _run_daily_race_analysis_for_ids(
        date_str,
        race_ids,
        analyze_timeout_sec=0,
        force_refresh_cache=force_refresh_cache,
        use_requests=use_requests,
    )


def _run_daily_race_analysis_for_ids(
    date_str: str,
    race_ids: List[str],
    analyze_timeout_sec: int = 0,
    force_refresh_cache: bool = False,
    use_requests: bool = False,
) -> Dict[str, Any]:
    """指定された race_id 群のみ分析して保存する（既存 analyze ロジック共通化）。"""
    from race_ai_engine import analyze_race
    from value_ai import recommend_betmaster_plans, assign_roles
    from kelly_staking import load_kelly_config
    kelly_config = load_kelly_config()

    summary: Dict[str, Any] = {
        "date": date_str, "total": len(race_ids),
        "success": 0, "skipped": 0, "errors": [],
    }

    print(f"[{date_str}] 対象: {len(race_ids)} レース", flush=True)

    for i, race_id in enumerate(race_ids, 1):
        print(f"  [{i}/{len(race_ids)}] {race_id} 分析中...", flush=True)
        try:
            url    = _race_url(race_id)
            _alarm_supported = (
                analyze_timeout_sec > 0
                and hasattr(_signal, "SIGALRM")
                and hasattr(_signal, "alarm")
            )
            if _alarm_supported:
                class _AnalyzeTimeoutError(Exception):
                    pass

                def _on_alarm(_sig, _frame):
                    raise _AnalyzeTimeoutError(f"analyze timeout ({analyze_timeout_sec}s)")

                _prev_handler = _signal.getsignal(_signal.SIGALRM)
                _signal.signal(_signal.SIGALRM, _on_alarm)
                _signal.alarm(int(analyze_timeout_sec))
                try:
                    result = analyze_race(
                        url,
                        headless=True,
                        force_refresh_cache=force_refresh_cache,
                        use_requests=use_requests,
                    )
                finally:
                    _signal.alarm(0)
                    _signal.signal(_signal.SIGALRM, _prev_handler)
            else:
                result = analyze_race(
                    url,
                    headless=True,
                    force_refresh_cache=force_refresh_cache,
                    use_requests=use_requests,
                )

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

            # オッズ・人気が欠損している場合の補完（確定オッズ優先、未開催なら予想オッズ）
            _odds_missing = sum(1 for f in features if not f.get("win_odds"))
            _pop_missing  = sum(1 for f in features if not f.get("popularity"))
            _odds_coverage = 1.0 - (_odds_missing / len(features)) if features else 0.0
            if _odds_coverage < 0.70 or _pop_missing > len(features) * 0.3:
                from dividend_scraper import scrape_race_result as _srl
                _result = None
                try:
                    _result = _srl(race_id)
                except Exception as _srl_err:
                    print(f"    → scrape_race_result 失敗: {_srl_err}", flush=True)

                if _result and _result.get("runners"):
                    # レース終了後：結果ページから確定オッズ・人気を注入
                    _runner_map = {r["horse_name"]: r for r in _result["runners"]}
                    injected_odds = 0
                    injected_pop  = 0
                    for f in features:
                        name   = f.get("horse_name")
                        runner = _runner_map.get(name, {})
                        if not f.get("win_odds") and runner.get("win_odds"):
                            f["win_odds"]          = runner["win_odds"]
                            f["odds_is_estimated"] = False
                            injected_odds += 1
                        if not f.get("popularity") and runner.get("popularity"):
                            f["popularity"] = runner["popularity"]
                            injected_pop   += 1
                    if injected_odds:
                        _inject_estimated_odds_recalc(features)
                    print(
                        f"    → 確定オッズ注入 {injected_odds}頭・人気注入 {injected_pop}頭（レース結果より）",
                        flush=True,
                    )
                else:
                    # レース前：予想オッズを補完
                    import odds_fetcher as _of
                    horse_number_map = {
                        str(int(f.get("horse_number") or (idx + 1))): f["horse_name"]
                        for idx, f in enumerate(features)
                        if f.get("horse_name")
                    }
                    est_status, est_odds = _of.fetch_estimated_odds(race_id, horse_number_map)
                    if est_odds:
                        injected = 0
                        for f in features:
                            name = f.get("horse_name")
                            if name and not f.get("win_odds") and name in est_odds:
                                f["win_odds"]          = est_odds[name]
                                f["odds_is_estimated"] = True
                                injected += 1
                        if injected:
                            _inject_estimated_odds_recalc(features)
                        print(
                            f"    → 予想オッズ補完 {injected}頭 (coverage {_odds_coverage:.0%}→補完後, status={est_status})",
                            flush=True,
                        )
                    else:
                        print(f"    → 予想オッズ取得不可 (status={est_status})。オッズなしで継続", flush=True)

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
            bets  = generate_all_bets(race_id, plans, kelly_config=kelly_config)
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


def run_daily_race_analysis_missing(
    date_str: str,
    analyze_timeout_sec: int = 420,
) -> Dict[str, Any]:
    """
    指定日のうち pipeline_predictions.json に未保存の race_id のみ再分析する。
    analyze_timeout_sec > 0 の場合、1レースごとにタイムアウトを適用する。
    """
    race_ids_all = get_race_ids_by_date(date_str)
    all_preds = pipeline_store.load_all_predictions()
    existing_ids = {
        rid for rid, pred in all_preds.items()
        if (
            isinstance(pred, dict)
            and pred.get("analysis_date") == date_str
            and not pipeline_store.is_prediction_incomplete(pred)
        )
    }
    missing_ids = [rid for rid in race_ids_all if rid not in existing_ids]

    print(
        f"[{date_str}] 欠損再分析: 全{len(race_ids_all)} / 既存{len(existing_ids)} / 対象{len(missing_ids)}",
        flush=True,
    )
    if not missing_ids:
        return {"date": date_str, "total": 0, "success": 0, "skipped": 0, "errors": []}

    return _run_daily_race_analysis_for_ids(
        date_str,
        missing_ids,
        analyze_timeout_sec=analyze_timeout_sec,
        force_refresh_cache=False,
    )


def run_daily_race_analysis_one(
    date_str: str,
    race_id: str,
    force_refresh_cache: bool = False,
) -> Dict[str, Any]:
    """1レースだけ分析して保存する（safe実行用の内部API）。"""
    return _run_daily_race_analysis_for_ids(
        date_str,
        [race_id],
        analyze_timeout_sec=0,
        force_refresh_cache=force_refresh_cache,
    )


def run_daily_race_analysis_one_safe(
    date_str: str,
    race_id: str,
    per_race_timeout_sec: int = 300,
    force_refresh_cache: bool = False,
) -> Dict[str, Any]:
    """
    1レースだけを別プロセスで安全実行する。
    ハング時は timeout で強制終了し、親プロセス側で状況を返す。
    """
    script_path = _os_dp.path.abspath(__file__)
    cmd = [_sys.executable, script_path, "--analyze-one", date_str, race_id]
    if force_refresh_cache:
        cmd.append("--force-refresh-cache")
    summary: Dict[str, Any] = {
        "date": date_str,
        "total": 1,
        "success": 0,
        "skipped": 0,
        "errors": [],
    }

    print(
        f"[{date_str}] 1レース安全再分析: {race_id} (timeout={int(per_race_timeout_sec)}s)",
        flush=True,
    )
    try:
        proc = _subprocess.Popen(
            cmd,
            start_new_session=True,
        )
        try:
            returncode = proc.wait(timeout=max(1, int(per_race_timeout_sec)))
        except _subprocess.TimeoutExpired:
            try:
                _os_dp.killpg(proc.pid, _signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
            raise

        if returncode == 0:
            summary["success"] = 1
            print("    → 完了", flush=True)
        else:
            summary["skipped"] = 1
            summary["errors"].append({"race_id": race_id, "error": f"exit_code={returncode}"})
            print(f"    → エラー: exit_code={returncode}", flush=True)
    except _subprocess.TimeoutExpired:
        summary["skipped"] = 1
        summary["errors"].append({"race_id": race_id, "error": f"timeout({per_race_timeout_sec}s)"})
        print(f"    → タイムアウト: {per_race_timeout_sec}s", flush=True)

    return summary


def run_daily_race_analysis_missing_safe(
    date_str: str,
    per_race_timeout_sec: int = 300,
) -> Dict[str, Any]:
    """
    未取得レースだけを1レースずつ別プロセスで実行する安全モード。
    1レースがハングしても timeout で打ち切り、残りを継続する。
    """
    race_ids_all = get_race_ids_by_date(date_str)
    all_preds = pipeline_store.load_all_predictions()
    existing_ids = {
        rid for rid, pred in all_preds.items()
        if (
            isinstance(pred, dict)
            and pred.get("analysis_date") == date_str
            and not pipeline_store.is_prediction_incomplete(pred)
        )
    }
    missing_ids = [rid for rid in race_ids_all if rid not in existing_ids]
    summary: Dict[str, Any] = {
        "date": date_str,
        "total": len(missing_ids),
        "success": 0,
        "skipped": 0,
        "errors": [],
    }

    print(
        f"[{date_str}] 欠損再分析(safe): 全{len(race_ids_all)} / 既存{len(existing_ids)} / 対象{len(missing_ids)}",
        flush=True,
    )
    if not missing_ids:
        return summary

    script_path = _os_dp.path.abspath(__file__)
    for i, race_id in enumerate(missing_ids, 1):
        print(f"  [{i}/{len(missing_ids)}] {race_id} 分析中...(safe)", flush=True)
        cmd = [_sys.executable, script_path, "--analyze-one", date_str, race_id]
        try:
            proc = _subprocess.Popen(
                cmd,
                start_new_session=True,
            )
            try:
                returncode = proc.wait(timeout=max(1, int(per_race_timeout_sec)))
            except _subprocess.TimeoutExpired:
                try:
                    _os_dp.killpg(proc.pid, _signal.SIGKILL)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
                raise

            if returncode == 0:
                summary["success"] += 1
                print("    → 完了", flush=True)
            else:
                summary["skipped"] += 1
                summary["errors"].append({"race_id": race_id, "error": f"exit_code={returncode}"})
                print(f"    → エラー: exit_code={returncode}", flush=True)
        except _subprocess.TimeoutExpired:
            summary["skipped"] += 1
            summary["errors"].append({"race_id": race_id, "error": f"timeout({per_race_timeout_sec}s)"})
            print(f"    → タイムアウト: {per_race_timeout_sec}s", flush=True)

        if i < len(missing_ids):
            time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

    print(
        f"[{date_str}] 欠損再分析(safe) 完了: 成功 {summary['success']}/{summary['total']}",
        flush=True,
    )
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
            summary["skipped"] += 1
            summary["errors"].append({"race_id": race_id, "error": str(e)})
            print(f"    → エラー: {e}", flush=True)

        if i < len(race_ids):
            time.sleep(random.uniform(1.5, 3.0))

    return summary


# =========================================================
# Phase 3.5: リアルタイム結果監視
# =========================================================

def watch_results(
    date_str: str,
    poll_interval: int = 300,
    result_delay_minutes: int = 10,
) -> None:
    """
    指定日のレース結果をリアルタイムで監視し、終了後に自動で取得・保存する。

    Parameters
    ----------
    date_str            : "20260412" 形式
    poll_interval       : チェック間隔（秒）。デフォルト 300 秒 (5分)
    result_delay_minutes: 発走時刻から何分後に結果スクレイプを試みるか（デフォルト 10分）
    """
    from dividend_scraper import scrape_race_result
    from datetime import datetime as _dt, timedelta as _td

    race_ids = get_race_ids_by_date(date_str)
    if not race_ids:
        print(f"[watch_results] {date_str}: レースIDなし。終了。", flush=True)
        return

    # レースのスケジュール情報を取得（発走時刻）
    from dashboard_loader import _fetch_schedule_map_for_date
    schedule = _fetch_schedule_map_for_date(date_str)

    done: set = set()  # 取得済み race_id

    print(
        f"[watch_results] {date_str}: {len(race_ids)}レース監視開始 "
        f"(間隔={poll_interval}s, 結果待ち={result_delay_minutes}分)",
        flush=True,
    )

    while True:
        now = _dt.now()
        pending = [rid for rid in race_ids if rid not in done]

        if not pending:
            print(f"[watch_results] 全レース完了。終了。", flush=True)
            break

        newly_fetched = 0
        for race_id in pending:
            # 既に結果が保存済みならスキップ
            existing = pipeline_store.load_pipeline_race_result(race_id)
            if existing and existing.get("finish_order"):
                done.add(race_id)
                continue

            # 発走時刻 + result_delay_minutes を過ぎているか確認
            meta = schedule.get(race_id, {})
            start_time_str = meta.get("start_time", "")
            if start_time_str:
                try:
                    start_dt = _dt.strptime(
                        f"{date_str} {start_time_str}", "%Y%m%d %H:%M"
                    )
                    if now < start_dt + _td(minutes=result_delay_minutes):
                        continue  # まだ早い
                except (ValueError, TypeError):
                    pass
            else:
                continue  # 時刻不明はスキップ

            # 結果スクレイプ
            try:
                result = scrape_race_result(race_id)
                if result and result.get("finish_order"):
                    pipeline_store.save_pipeline_race_result(race_id, result)
                    outcomes = evaluate_single_race(race_id)
                    pipeline_store.save_bet_outcomes(race_id, outcomes)
                    hit_count = sum(1 for o in outcomes if o.get("hit"))
                    print(
                        f"[watch_results] {race_id} 結果取得 "
                        f"({len(result['finish_order'])}頭 / {hit_count}的中)",
                        flush=True,
                    )
                    done.add(race_id)
                    newly_fetched += 1
                else:
                    print(f"[watch_results] {race_id} 結果未確定。次回再試行。", flush=True)
            except Exception as e:
                print(f"[watch_results] {race_id} エラー: {e}", flush=True)

            time.sleep(random.uniform(1.5, 3.0))

        remaining = len(race_ids) - len(done)
        print(
            f"[watch_results] サイクル完了: 今回取得={newly_fetched} / "
            f"残り={remaining} / 完了={len(done)} "
            f"({now.strftime('%H:%M:%S')})",
            flush=True,
        )

        if remaining == 0:
            print(f"[watch_results] 全レース完了。終了。", flush=True)
            break

        print(f"[watch_results] {poll_interval}秒後に再チェック...", flush=True)
        time.sleep(poll_interval)


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
            "date":          dates_str,
            "bet_type":      bet_key,
            "label":         bt["label"],
            "bets":          bt["bets"],
            "hits":          bt["hits"],
            "stake":         bt["stake"],
            "payout":        bt["payout"],
            "hit_rate(%)":   round(bt["hit_rate"] * 100, 1),
            "roi(%)":        round(bt["roi"] * 100, 1),
        })

    if not rows:
        print("[export_summary_csv] 集計データなし。出力スキップ。", flush=True)
        return out_path

    fieldnames = ["date", "bet_type", "label", "bets", "hits",
                  "stake", "payout", "hit_rate(%)", "roi(%)"]
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
    拡張モデル (keiba_lgbm_model_expanded.txt) が存在する場合は優先使用し、
    ない場合は既存モデルにフォールバック。モデルがない場合は一様分布を返す。
    テストで monkeypatch しやすいよう分離。
    """
    import os as _os
    from race_ai_engine import predict_win_probability_with_model, MODEL_FILE

    # 拡張モデルが存在する場合は優先使用
    if _os.path.exists(EXPANDED_MODEL_FILE):
        try:
            from race_ai_engine import ML_FEATURE_COLUMNS_EXPANDED
            probs = predict_win_probability_with_model(features, EXPANDED_MODEL_FILE, feature_columns=ML_FEATURE_COLUMNS_EXPANDED)
            if probs is not None:
                return probs
        except Exception as _e:
            _logger.debug("拡張モデル推論失敗、既存モデルにフォールバック: %s", _e)

    # 既存モデルにフォールバック
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
        # place_odds is from the morning analysis snapshot and may be None (market not yet open)
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

    if status == "not_open":
        # 実オッズ未開放 → 予想オッズにフォールバック
        est_status, est_odds = odds_fetcher.fetch_estimated_odds(race_id, horse_number_map)
        if est_odds:
            _logger.info("[update_race_odds] %s | 予想オッズ使用 (est_status=%s)", race_id, est_status)
            new_odds = est_odds
            status = est_status  # "estimated" / "success" / "partial"
        else:
            _logger.info("[update_race_odds] %s | スキップ（not_open、予想オッズも取得不可）", race_id)
            return {"race_id": race_id, "status": "not_open", "coverage": 0.0,
                    "version_before": version_before, "version_after": version_before}

    if status in ("api_failed", "selenium_failed", "failed"):
        _logger.info("[update_race_odds] %s | スキップ（%s）", race_id, status)
        return {"race_id": race_id, "status": status, "coverage": 0.0,
                "version_before": version_before, "version_after": version_before}

    _is_estimated = (status == "estimated")

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
            if _is_estimated:
                f["odds_is_estimated"] = True

    # 予想オッズから人気順位を推定（実オッズ未開放時のみ）
    if _is_estimated and new_odds:
        sorted_by_odds = sorted(new_odds.items(), key=lambda x: x[1])
        est_pop_map = {name: rank for rank, (name, _) in enumerate(sorted_by_odds, 1)}
        for f in features:
            if not f.get("popularity"):
                f["popularity"] = est_pop_map.get(str(f.get("horse_name") or ""))

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

    # Ranker ブレンド（モデルが存在する場合のみ。失敗しても継続）
    try:
        from ranker_engine import predict_rank_score, blend_scores
        rank_scores = predict_rank_score(features, profile="balanced")
        if rank_scores is not None:
            new_probs = blend_scores(new_probs, rank_scores, weight_ranker=0.3)
    except Exception as _e:
        _logger.debug("ranker blend skipped: %s", _e)

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
    from kelly_staking import load_kelly_config as _load_kelly_config
    horse_roles = assign_roles(features, ev_table, race_structure, danger_v2)
    plans = recommend_betmaster_plans(features, race_structure, horse_roles)
    bets  = generate_all_bets(race_id, plans, kelly_config=_load_kelly_config())

    # 保存用 horses リスト（feature_dict も更新）
    old_horses_by_name = {h["horse_name"]: h for h in pred.get("horses", [])}
    updated_horses = []
    for f in features:
        name = str(f.get("horse_name") or "")
        old_h = old_horses_by_name.get(name, {})
        updated_horses.append({
            **old_h,
            "ai_win_prob":     round(float(f.get("win_prob", 0)), 4),
            "win_odds":        f.get("win_odds"),
            "popularity":      f.get("popularity"),
            "odds_is_estimated": f.get("odds_is_estimated", False),
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
        # NOTE: odds_source is "api" for both requests-based and Selenium-partial paths,
        # since fetch_win_odds merges both into the same status string ("success"/"partial").
        # Only explicit selenium_failed carries "selenium" in the status.
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
            elif status in ("not_open", "estimated"):
                pass   # 次サイクルで再試行（実オッズ確定を待つ）
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
    parser.add_argument("--analyze-one", nargs=2, metavar=("DATE", "RACE_ID"),
                        help="1レースのみ分析（内部用）")
    parser.add_argument("--analyze-one-safe", nargs=2, metavar=("DATE", "RACE_ID"),
                        help="1レースのみを別プロセスで安全再分析（ハング回避）")
    parser.add_argument("--analyze-missing", metavar="DATE",
                        help="未取得レースのみ再分析（例: 20250406）")
    parser.add_argument("--analyze-missing-safe", metavar="DATE",
                        help="未取得レースを1件ずつ別プロセスで再分析（ハング回避）")
    parser.add_argument("--analyze-timeout", type=int, default=420,
                        help="--analyze-missing 時の1レースあたりタイムアウト秒（0で無効, デフォルト: 420）")
    parser.add_argument("--force-refresh-cache", action="store_true",
                        help="レース分析キャッシュを使わず再取得する")
    parser.add_argument("--use-requests", action="store_true",
                        help="Selenium の代わりに requests+BeautifulSoup で高速取得する（実験的）")
    parser.add_argument("--evaluate",   metavar="DATE",  help="結果照合（例: 20250406）")
    parser.add_argument("--summarize",  metavar="DATES",
                        help="週末集計（カンマ区切り例: 20250405,20250406）")
    parser.add_argument("--export-csv", metavar="DATES",
                        help="週末集計をCSV出力（カンマ区切り例: 20250405,20250406）")
    parser.add_argument("--list",       metavar="DATE",  help="指定日のレース一覧を表示")
    parser.add_argument("--watch-odds",    metavar="DATE",
                        help="オッズ自動監視（例: 20250406）")
    parser.add_argument("--watch-results", metavar="DATE",
                        help="レース結果リアルタイム監視（例: 20260412）")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="watch-odds のポーリング間隔（秒、デフォルト: 60）")
    parser.add_argument("--result-interval", type=int, default=300,
                        help="watch-results のポーリング間隔（秒、デフォルト: 300）")
    parser.add_argument("--result-delay", type=int, default=10,
                        help="watch-results の発走後何分で結果取得を試みるか（デフォルト: 10）")
    args = parser.parse_args()

    if args.analyze:
        run_daily_race_analysis(
            args.analyze,
            force_refresh_cache=args.force_refresh_cache,
            use_requests=args.use_requests,
        )

    elif args.analyze_one:
        run_daily_race_analysis_one(
            args.analyze_one[0],
            args.analyze_one[1],
            force_refresh_cache=args.force_refresh_cache,
        )

    elif args.analyze_one_safe:
        run_daily_race_analysis_one_safe(
            args.analyze_one_safe[0],
            args.analyze_one_safe[1],
            per_race_timeout_sec=args.analyze_timeout,
            force_refresh_cache=args.force_refresh_cache,
        )

    elif args.analyze_missing_safe:
        run_daily_race_analysis_missing_safe(args.analyze_missing_safe, per_race_timeout_sec=args.analyze_timeout)

    elif args.analyze_missing:
        run_daily_race_analysis_missing(args.analyze_missing, analyze_timeout_sec=args.analyze_timeout)

    elif args.evaluate:
        evaluate_prediction_for_day(args.evaluate)

    elif args.summarize:
        dates   = [d.strip() for d in args.summarize.split(",")]
        summary = summarize_weekend_performance(dates)
        print_summary(summary)

        # ROI レポート自動出力
        try:
            from roi_reporter import aggregate_by_bet_type, generate_markdown_report, generate_csv_report, filter_outcomes_by_dates
            import os as _os

            all_outcomes = pipeline_store.load_all_bet_outcomes()
            all_preds    = pipeline_store.load_all_predictions()
            filtered     = filter_outcomes_by_dates(all_outcomes, all_preds, dates)
            roi_summary  = aggregate_by_bet_type(filtered)
            report_dir   = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "reports")
            Path(report_dir).mkdir(parents=True, exist_ok=True)
            date_label = "_".join(sorted(dates))
            md_path  = _os.path.join(report_dir, f"roi_{date_label}.md")
            csv_path = _os.path.join(report_dir, f"roi_{date_label}.csv")
            md_text  = generate_markdown_report(roi_summary, dates=dates)
            with open(md_path, "w", encoding="utf-8") as _f:
                _f.write(md_text)
            generate_csv_report(roi_summary, csv_path, dates=dates)
            print(f"ROIレポートを保存しました: {md_path} / {csv_path}")
        except Exception as _roi_err:
            print(f"[警告] ROIレポート生成に失敗しました: {_roi_err}")

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

    elif args.watch_results:
        watch_results(
            args.watch_results,
            poll_interval=args.result_interval,
            result_delay_minutes=args.result_delay,
        )

    else:
        parser.print_help()
