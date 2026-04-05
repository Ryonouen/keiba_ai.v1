# odds_fetcher.py
"""
odds_fetcher.py
確定前オッズ・脚質取得モジュール

優先順:
  1. requests — netkeiba JSON API（高速・Selenium不要）
  2. Selenium fallback — shutuba ページ .Odds スクレイピング（重い）

公開 API:
  fetch_win_odds(race_id, horse_number_map)     → (status, {horse_name: odds} | None)
  fetch_newspaper_styles(race_id, horse_number_map) → (status, {horse_name: style} | None)
"""
from __future__ import annotations

import logging
import re
from typing import Dict, Optional, Tuple

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────
# 定数
# ──────────────────────────────────────────────────────
ODDS_API_URL            = "https://race.netkeiba.com/api/api_get_jra_odds.html"
NEWSPAPER_URL_TEMPLATE  = "https://race.netkeiba.com/race/newspaper.html?race_id={race_id}"
SHUTUBA_URL_TEMPLATE    = "https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"

REQUEST_TIMEOUT         = 10        # seconds
BACKOFF_DELAYS          = [1.0, 2.0]
SELENIUM_WAIT_MAX       = 15        # seconds
ODDS_COVERAGE_THRESHOLD = 0.8
STYLE_COVERAGE_THRESHOLD = 0.7

_BLOCK_STATUS_CODES     = {403, 429}

_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "X-Requested-With": "XMLHttpRequest",
}

# ──────────────────────────────────────────────────────
# 内部ヘルパー
# ──────────────────────────────────────────────────────

def _request_get(url: str, **kwargs):
    """requests.get の薄いラッパー（テストで monkeypatch するための分離点）。"""
    return requests.get(url, **kwargs)


def _normalize_horse_no(no: str) -> str:
    """"01" → "1" のゼロパディング除去。"""
    try:
        return str(int(no))
    except ValueError:
        return no


def _parse_odds_response(data: dict) -> Optional[Dict[str, float]]:
    """
    netkeiba オッズ API の JSON レスポンスから {horse_no_str: odds_float} を返す。
    複数の既知スキーマを試みる。いずれも合わなければ None。

    既知パス（優先順）:
      1. data["data"]["Odds"]  — str 値の辞書
      2. data["data"]["WinOdds"]
      3. data["data"]["Odds"]["WinOdds"]
    """
    candidates = []
    try:
        candidates.append(data["data"]["Odds"])
    except (KeyError, TypeError):
        pass
    try:
        candidates.append(data["data"]["WinOdds"])
    except (KeyError, TypeError):
        pass
    try:
        candidates.append(data["data"]["Odds"]["WinOdds"])
    except (KeyError, TypeError):
        pass

    for raw in candidates:
        if not isinstance(raw, dict):
            continue
        result: Dict[str, float] = {}
        for k, v in raw.items():
            no = _normalize_horse_no(str(k))
            val_str = v if isinstance(v, str) else str(v)
            if val_str in ("–", "-", "---", "", "0"):
                continue
            try:
                result[no] = float(val_str)
            except ValueError:
                continue
        if result:   # 1件以上 parse できたら採用
            return result
    return None


def _eval_coverage(
    data: dict,
    horse_number_map: Dict[str, str],
) -> Tuple[str, Optional[Dict[str, float]]]:
    """
    レスポンスを解析してカバレッジを評価し (status, {horse_name: odds}) を返す。

    status:
      "not_open"  — 全馬が "–"（未発売）
      "success"   — coverage >= ODDS_COVERAGE_THRESHOLD
      "partial"   — 0 < coverage < ODDS_COVERAGE_THRESHOLD
    """
    total = len(horse_number_map)

    # 全オッズが "–" かチェック（not_open 判定用）
    raw_check = None
    for path_fn in [
        lambda d: d["data"]["Odds"],
        lambda d: d["data"]["WinOdds"],
        lambda d: d["data"]["Odds"]["WinOdds"],
    ]:
        try:
            raw_check = path_fn(data)
            break
        except (KeyError, TypeError):
            continue

    if raw_check and isinstance(raw_check, dict):
        all_dash = all(
            str(v) in ("–", "-", "---", "", "0")
            for v in raw_check.values()
        )
        if all_dash:
            return "not_open", None

    parsed = _parse_odds_response(data)
    if parsed is None:
        # Response received but schema unrecognized — not the same as "not open"
        logger.warning("[odds_fetcher] 未知レスポンス構造 → status=failed")
        return "failed", None

    # horse_name に変換
    named: Dict[str, float] = {}
    for no, odds in parsed.items():
        name = horse_number_map.get(no)
        if name:
            named[name] = odds

    ok = len(named)
    ratio = ok / total if total > 0 else 0.0
    logger.info(
        "[odds_fetcher] coverage=%.0f%% (%d/%d頭)",
        ratio * 100, ok, total,
    )

    if ok == 0:
        return "not_open", None
    if ratio >= ODDS_COVERAGE_THRESHOLD:
        return "success", named
    return "partial", named
