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
import time
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
      "failed"    — レスポンス構造が未知でパース不能
    """
    total = len(horse_number_map)

    # 「馬番→オッズ」のフラット辞書を特定する（ネスト辞書は除外）
    raw_odds: Optional[dict] = None
    for path_fn in [
        lambda d: d["data"]["Odds"],
        lambda d: d["data"]["WinOdds"],
        lambda d: d["data"]["Odds"]["WinOdds"],
    ]:
        try:
            candidate = path_fn(data)
            if isinstance(candidate, dict) and all(
                not isinstance(v, dict) for v in candidate.values()
            ):
                raw_odds = candidate
                break
        except (KeyError, TypeError):
            continue

    if raw_odds is None:
        logger.warning("[odds_fetcher] 未知レスポンス構造 → status=failed")
        return "failed", None

    # 全オッズが "–" かチェック（not_open 判定）
    all_dash = all(
        str(v) in ("–", "-", "---", "", "0") for v in raw_odds.values()
    )
    if all_dash:
        return "not_open", None

    parsed = _parse_odds_response(data)
    if parsed is None:
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


# ──────────────────────────────────────────────────────
# requests ベース取得
# ──────────────────────────────────────────────────────

def _build_session(race_id: str) -> requests.Session:
    s = requests.Session()
    s.headers.update(_REQUEST_HEADERS)
    s.headers["Referer"] = SHUTUBA_URL_TEMPLATE.format(race_id=race_id)
    return s


def _fetch_win_odds_by_requests(
    race_id: str,
    horse_number_map: Dict[str, str],
) -> Tuple[str, Optional[Dict[str, float]]]:
    """
    requests で単勝オッズを取得し (status, {horse_name: odds} | None) を返す。
    status: "success" | "partial" | "not_open" | "api_failed"
    """
    url = ODDS_API_URL
    params = {"race_id": race_id, "type": "b1", "action": "all"}
    session = _build_session(race_id)

    for attempt, delay in enumerate(BACKOFF_DELAYS, 1):
        try:
            resp = _request_get(url, params=params, timeout=REQUEST_TIMEOUT,
                                headers=dict(session.headers))
        except Exception as e:
            logger.warning("[odds_fetcher] %s | request exception: %s", race_id, e)
            time.sleep(delay)
            continue

        logger.info("[odds_fetcher] %s | attempt %d/%d | HTTP %d",
                    race_id, attempt, len(BACKOFF_DELAYS), resp.status_code)

        if resp.status_code in _BLOCK_STATUS_CODES:
            logger.warning("[odds_fetcher] %s | ブロック検知 HTTP %d → api_failed",
                           race_id, resp.status_code)
            return "api_failed", None

        if not resp.ok:
            if attempt < len(BACKOFF_DELAYS):
                time.sleep(delay)
            continue

        # 200 OK
        try:
            data = resp.json()
        except Exception:
            logger.warning("[odds_fetcher] %s | JSON parse 失敗 → api_failed", race_id)
            return "api_failed", None

        parsed = _parse_odds_response(data)
        if parsed is None:
            logger.warning(
                "[odds_fetcher] %s | 未知のレスポンス構造 → api_failed", race_id
            )
            return "api_failed", None

        # Rebuild as path-1 format for _eval_coverage
        fake_data = {"data": {"Odds": {k: str(v) for k, v in parsed.items()}}}
        status, named = _eval_coverage(fake_data, horse_number_map)
        logger.info("[odds_fetcher] %s | %s | coverage (from requests)", race_id, status)
        return status, named

    logger.warning("[odds_fetcher] %s | 全リトライ失敗 → api_failed", race_id)
    return "api_failed", None


# ──────────────────────────────────────────────────────
# Selenium フォールバック
# ──────────────────────────────────────────────────────

def _fetch_win_odds_by_selenium(
    race_id: str,
    horse_number_map: Dict[str, str],
) -> Tuple[str, Optional[Dict[str, float]]]:
    """
    Selenium で shutuba ページから単勝オッズを取得する（軽量版）。
    新聞・馬個別ページは開かない。ブラウザは即クローズ。
    status: "success" | "partial" | "not_open" | "selenium_failed"
    """
    try:
        from race_ai_engine import build_webdriver, safe_get, warmup_netkeiba_session
        from selenium.webdriver.common.by import By
        import time as _time
    except ImportError as e:
        logger.error("[odds_fetcher] Selenium import 失敗: %s", e)
        return "selenium_failed", None

    url = SHUTUBA_URL_TEMPLATE.format(race_id=race_id)
    driver = None
    try:
        driver = build_webdriver(headless=True)
        driver = warmup_netkeiba_session(driver, headless=True)
        driver = safe_get(driver, url, headless=True, retries=1)

        # オッズが描画されるまで待機（最大 SELENIUM_WAIT_MAX 秒）
        odds_raw: Dict[str, float] = {}
        for _ in range(SELENIUM_WAIT_MAX):
            rows = driver.find_elements(By.CSS_SELECTOR, "table.Shutuba_Table tbody tr")
            for row in rows:
                try:
                    no_text = row.find_element(By.CSS_SELECTOR, ".Umaban").text.strip()
                    odds_text = row.find_element(By.CSS_SELECTOR, ".Odds").text.strip()
                    if odds_text in ("–", "-", "", "---"):
                        continue
                    no = _normalize_horse_no(no_text)
                    odds_raw[no] = float(odds_text)
                except Exception:
                    continue
            if odds_raw:
                break
            _time.sleep(1)

        if not odds_raw:
            return "not_open", None

        named: Dict[str, float] = {
            horse_number_map[no]: v
            for no, v in odds_raw.items()
            if no in horse_number_map
        }
        total = len(horse_number_map)
        ratio = len(named) / total if total > 0 else 0.0
        logger.info("[odds_fetcher] %s | selenium | coverage=%.0f%% (%d/%d頭)",
                    race_id, ratio * 100, len(named), total)

        if len(named) == 0:
            return "not_open", None
        status = "success" if ratio >= ODDS_COVERAGE_THRESHOLD else "partial"
        return status, named

    except Exception as e:
        logger.error("[odds_fetcher] %s | Selenium 失敗: %s", race_id, e)
        return "selenium_failed", None
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass


# ──────────────────────────────────────────────────────
# 公開 API
# ──────────────────────────────────────────────────────

def fetch_win_odds(
    race_id: str,
    horse_number_map: Dict[str, str],
) -> Tuple[str, Optional[Dict[str, float]]]:
    """
    単勝オッズを取得。requests → Selenium fallback。

    Parameters
    ----------
    race_id          : 12桁 race_id
    horse_number_map : {"1": "ショウヘイ", ...}

    Returns
    -------
    (status, {horse_name: win_odds_float} | None)
    status: "success" | "partial" | "not_open" | "api_failed" | "selenium_failed" | "failed"
    """
    status, result = _fetch_win_odds_by_requests(race_id, horse_number_map)
    logger.info("[odds_fetcher] %s | requests → %s", race_id, status)

    if status in ("success", "partial", "not_open"):
        return status, result

    # requests 失敗 → Selenium fallback
    logger.info("[odds_fetcher] %s | Selenium fallback 開始", race_id)
    status2, result2 = _fetch_win_odds_by_selenium(race_id, horse_number_map)
    logger.info("[odds_fetcher] %s | selenium → %s", race_id, status2)

    if status2 in ("success", "partial", "not_open"):
        return status2, result2

    return "failed", None


# ──────────────────────────────────────────────────────
# 脚質取得
# ──────────────────────────────────────────────────────

def _parse_newspaper_html(html: str, horse_number_map: Dict[str, str]) -> Optional[Dict[str, str]]:
    """
    新聞ページ HTML から {horse_name: running_style} を抽出する。
    style_char_to_running_style() は race_ai_engine から借用する。
    取得できなかった馬は含まない（呼び出し元がカバレッジ判定）。
    """
    try:
        from race_ai_engine import style_char_to_running_style
    except ImportError:
        return None

    soup = BeautifulSoup(html, "html.parser")
    result: Dict[str, str] = {}
    all_names = set(horse_number_map.values())

    for row in soup.select("tr"):
        name_cell = row.select_one(".HorseName, .Horse_Name, td.name")
        style_cell = row.select_one(".RunningStyle, .Style, td.style")
        if not name_cell or not style_cell:
            continue
        name = name_cell.get_text(strip=True)
        style_char = style_cell.get_text(strip=True)
        if name in all_names and style_char:
            result[name] = style_char_to_running_style(style_char)

    return result if result else None


def _fetch_newspaper_styles_by_requests(
    race_id: str,
    horse_number_map: Dict[str, str],
) -> Tuple[str, Optional[Dict[str, str]]]:
    url = NEWSPAPER_URL_TEMPLATE.format(race_id=race_id)
    try:
        resp = _request_get(url, timeout=REQUEST_TIMEOUT, headers=_REQUEST_HEADERS)
    except Exception as e:
        logger.warning("[odds_fetcher] %s | newspaper requests 失敗: %s", race_id, e)
        return "api_failed", None

    if not resp.ok:
        logger.warning("[odds_fetcher] %s | newspaper HTTP %d", race_id, resp.status_code)
        return "api_failed", None

    result = _parse_newspaper_html(resp.text, horse_number_map)
    if result is None:
        logger.warning("[odds_fetcher] %s | newspaper HTML parse 失敗", race_id)
        return "api_failed", None

    total = len(horse_number_map)
    ratio = len(result) / total if total > 0 else 0.0
    status = "success" if ratio >= STYLE_COVERAGE_THRESHOLD else "partial"
    logger.info("[odds_fetcher] %s | newspaper style coverage=%.0f%%", race_id, ratio * 100)
    return status, result


def _fetch_newspaper_styles_by_selenium(
    race_id: str,
    horse_number_map: Dict[str, str],
) -> Tuple[str, Optional[Dict[str, str]]]:
    try:
        from race_ai_engine import (
            build_webdriver, safe_get, warmup_netkeiba_session,
            fetch_newspaper_records, style_char_to_running_style,
        )
    except ImportError as e:
        logger.error("[odds_fetcher] Selenium import 失敗: %s", e)
        return "selenium_failed", None

    url = NEWSPAPER_URL_TEMPLATE.format(race_id=race_id)
    driver = None
    try:
        driver = build_webdriver(headless=True)
        driver = warmup_netkeiba_session(driver, headless=True)
        driver = safe_get(driver, url, headless=True, retries=1)
        records = fetch_newspaper_records(driver)

        result: Dict[str, str] = {}
        all_names = set(horse_number_map.values())
        for name, entry in records.items():
            if name not in all_names:
                continue
            style_char = str(entry.get("style_char", "")) if isinstance(entry, dict) else ""
            result[name] = style_char_to_running_style(style_char) if style_char else "unknown"

        if not result:
            return "selenium_failed", None

        total = len(horse_number_map)
        ratio = len(result) / total if total > 0 else 0.0
        status = "success" if ratio >= STYLE_COVERAGE_THRESHOLD else "partial"
        return status, result

    except Exception as e:
        logger.error("[odds_fetcher] %s | newspaper Selenium 失敗: %s", race_id, e)
        return "selenium_failed", None
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass


def fetch_newspaper_styles(
    race_id: str,
    horse_number_map: Dict[str, str],
) -> Tuple[str, Optional[Dict[str, str]]]:
    """
    新聞ページから脚質を取得。requests → Selenium fallback。

    Returns
    -------
    (status, {horse_name: running_style_str} | None)
    status: "success" | "partial" | "api_failed" | "selenium_failed" | "failed"
    """
    status, result = _fetch_newspaper_styles_by_requests(race_id, horse_number_map)
    if status in ("success", "partial"):
        return status, result

    status2, result2 = _fetch_newspaper_styles_by_selenium(race_id, horse_number_map)
    if status2 in ("success", "partial"):
        return status2, result2

    return "failed", None
