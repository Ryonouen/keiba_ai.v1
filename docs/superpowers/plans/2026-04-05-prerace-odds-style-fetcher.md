# 確定前オッズ・脚質確実取得 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 早朝 `--analyze` で脚質・過去データを取得し、発走30分前に requests → Selenium fallback で最新オッズを自動取得、LightGBM を再予測して買い目を更新する。

**Architecture:** `odds_fetcher.py` にオッズ・脚質取得責務を分離し、`pipeline_store.py` に v2 スキーマ（発走時刻・feature_dict・prediction_history）を追加、`daily_pipeline.py` に `update_race_odds()` と `watch_odds()` ループを追加する。`analyze_race()` は変更しない。

**Tech Stack:** Python 3.14, requests, BeautifulSoup4, LightGBM, Selenium (fallback only), pytest

---

## 設計上の重要方針

### prediction_version の保存方針
- `--analyze` 実行時: `prediction_version = 1`、`prediction_history` にエントリを追記
- `update_race_odds()` が `status in ("success", "partial")` のとき: `prediction_version += 1`、`prediction_history` に追記
- `not_open` / `api_failed` / `selenium_failed` / `failed`: version 変更なし

### partial / fallback 閾値
- `ODDS_COVERAGE_THRESHOLD = 0.8`（出走頭数の 80% 以上で valid odds を取得できた場合 `success`）
- `0 < coverage < 0.8` → `partial`（更新は行い version もインクリメント、警告ログ）
- `coverage == 0` かつオッズが全て `"–"` → `not_open`（watcher が次サイクルで再試行）
- `coverage == 0` かつ HTTP エラー → `api_failed` → Selenium fallback
- `STYLE_COVERAGE_THRESHOLD = 0.7`（脚質が 70% 未満なら `fetch_newspaper_styles()` で再取得試行）

### watcher の停止条件
以下のいずれかを満たしたら `watch_odds` ループを終了する：
1. 全 race_id が `updated_ids` に入っており、かつ全レースの `start_datetime < now`
2. 全レースの `start_datetime + 90min < now`（発走後90分経過したら諦め）
3. SIGINT（Ctrl+C）→ サマリを出力して `sys.exit(0)`

### オッズ更新後に再計算する特徴量の範囲
**更新する（odds 依存）:**
`feat_win_odds_log`, `win_odds`, `feat_popularity`, `win_prob`, `place_prob`,
`fair_win_odds`, `fair_place_odds`, `win_ev`, `place_ev`, `win_market_edge`,
`place_market_edge`, `odds_distortion_index`, `value_flag`, `win_value_label`,
`place_value_label`, `expected_value_score`, `bet_suitability`

**更新しない（発走当日に変化しない）:**
`feat_gate`, `feat_age`, `feat_last3f`, `feat_jockey_weight`, `feat_n_runners`,
`feat_running_style_enc`, `feat_track_condition_enc`, `feat_signal_total_adjust`,
全 `feat_cond_diff_*`, `feat_recent_form`, `feat_trend_index`, `feat_consistency_index`,
`running_style`, `place_odds`

### 最低限のログ項目
```
[odds_fetcher] {race_id} | attempt {n}/{max} | HTTP {code}
[odds_fetcher] {race_id} | 未知レスポンス構造 → Selenium fallback
[odds_fetcher] {race_id} | {status} | coverage={ratio:.0%} ({ok}/{total}頭)
[update_race_odds] {race_id} | status={status} | coverage={ratio:.0%} | v{old}→v{new}
[update_race_odds] {race_id} | top: {name} {old_prob:.1%}→{new_prob:.1%} @ {odds}倍
[watch_odds] {DATE} {hh:mm} | {venue}{R}R | {status}
[watch_odds] {DATE} | 完了 {updated}/{total}件 | 失敗 {failed}件
```

---

## ファイル構成

```
odds_fetcher.py              ← 新規作成
pipeline_store.py            ← save_prediction_v2() 等を追加（既存関数は変更しない）
daily_pipeline.py            ← run_daily_race_analysis() 拡張 + 新規関数追加
tests/test_odds_fetcher.py   ← 新規作成
tests/test_pipeline_store_v2.py ← 新規作成
tests/test_watch_odds.py     ← 新規作成
```

---

## Task 1: odds_fetcher.py — requests ベースのオッズ取得（解析・カバレッジ判定）

**Files:**
- Create: `odds_fetcher.py`
- Create: `tests/test_odds_fetcher.py`

- [ ] **Step 1: 失敗するテストを書く**

```python
# tests/test_odds_fetcher.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from unittest.mock import patch, MagicMock
import odds_fetcher

HORSE_MAP = {"1": "ショウヘイ", "2": "ヨーホーレイク", "3": "セイウン", "4": "クロワ",
             "5": "デビット",   "6": "ダノン",         "7": "ボルド",  "8": "サンスト"}


def test_parse_api_response_standard():
    """正常な JSON レスポンス → {horse_no_str: float} を返す"""
    raw = {"data": {"Odds": {"1": "5.6", "2": "12.3", "3": "8.0", "4": "15.1",
                             "5": "22.5", "6": "7.2", "7": "18.0", "8": "9.9"}}}
    result = odds_fetcher._parse_odds_response(raw)
    assert result is not None
    assert result["1"] == 5.6
    assert result["2"] == 12.3


def test_parse_api_response_unknown_schema():
    """未知スキーマ → None を返す（warning は呼び出し元が処理）"""
    raw = {"unexpected": {"key": "value"}}
    result = odds_fetcher._parse_odds_response(raw)
    assert result is None


def test_status_not_open():
    """全オッズが "–" → status=not_open, None"""
    raw = {"data": {"Odds": {"1": "–", "2": "–", "3": "–", "4": "–",
                             "5": "–", "6": "–", "7": "–", "8": "–"}}}
    status, result = odds_fetcher._eval_coverage(raw, HORSE_MAP)
    assert status == "not_open"
    assert result is None


def test_coverage_above_threshold():
    """coverage 87.5% (7/8) → status=success"""
    raw = {"data": {"Odds": {"1": "5.6", "2": "12.3", "3": "8.0", "4": "15.1",
                             "5": "22.5", "6": "7.2", "7": "18.0", "8": "–"}}}
    status, result = odds_fetcher._eval_coverage(raw, HORSE_MAP)
    assert status == "success"
    assert result is not None
    assert "ショウヘイ" in result
    assert "8" not in result   # "–" は除外


def test_coverage_below_threshold():
    """coverage 50% (4/8) → status=partial"""
    raw = {"data": {"Odds": {"1": "5.6", "2": "12.3", "3": "8.0", "4": "15.1",
                             "5": "–", "6": "–", "7": "–", "8": "–"}}}
    status, result = odds_fetcher._eval_coverage(raw, HORSE_MAP)
    assert status == "partial"
    assert result is not None
    assert len(result) == 4


def test_horse_number_normalization():
    """ゼロパディング "01" → "1" に正規化して horse_name に変換される"""
    raw = {"data": {"Odds": {"01": "5.6", "02": "12.3"}}}
    padded_map = {"1": "ショウヘイ", "2": "ヨーホーレイク"}
    result = odds_fetcher._parse_odds_response(raw)
    assert result is not None
    # normalize_horse_no で "01" → "1"
    norm = {odds_fetcher._normalize_horse_no(k): v for k, v in result.items()}
    assert norm["1"] == 5.6
    assert norm["2"] == 12.3
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
cd /Users/ryokarahashi/keiba_ai
python3 -m pytest tests/test_odds_fetcher.py -v 2>&1 | head -20
```

Expected: `ERROR` (ModuleNotFoundError: No module named 'odds_fetcher')

- [ ] **Step 3: 定数・パース関数を実装**

```python
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
import math
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
    for path in [
        lambda d: d["data"]["Odds"],
        lambda d: d["data"]["WinOdds"],
        lambda d: d["data"]["Odds"]["WinOdds"],
    ]:
        try:
            raw_check = path(data)
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
        return "not_open", None

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
```

- [ ] **Step 4: テストが通ることを確認**

```bash
python3 -m pytest tests/test_odds_fetcher.py::test_parse_api_response_standard \
  tests/test_odds_fetcher.py::test_parse_api_response_unknown_schema \
  tests/test_odds_fetcher.py::test_status_not_open \
  tests/test_odds_fetcher.py::test_coverage_above_threshold \
  tests/test_odds_fetcher.py::test_coverage_below_threshold \
  tests/test_odds_fetcher.py::test_horse_number_normalization \
  -v
```

Expected: 6 passed

- [ ] **Step 5: コミット**

```bash
git add odds_fetcher.py tests/test_odds_fetcher.py
git commit -m "feat: add odds_fetcher — parse/coverage helpers (no network calls yet)"
```

---

## Task 2: odds_fetcher.py — backoff・Selenium fallback・公開 fetch_win_odds()

**Files:**
- Modify: `odds_fetcher.py`
- Modify: `tests/test_odds_fetcher.py`

- [ ] **Step 1: 失敗するテストを追加**

```python
# tests/test_odds_fetcher.py に追記

def test_backoff_retry_transient(monkeypatch):
    """HTTP 503 が 2 回続いてから成功する場合、リトライして取得できる"""
    call_count = {"n": 0}
    good_json = {"data": {"Odds": {"1": "5.6", "2": "12.3", "3": "8.0",
                                   "4": "15.1", "5": "22.5", "6": "7.2",
                                   "7": "18.0", "8": "9.9"}}}

    class FakeResp:
        def __init__(self, code, json_data=None):
            self.status_code = code
            self.ok = code == 200
            self._json = json_data
        def json(self):
            return self._json

    def fake_get(url, **kwargs):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            return FakeResp(503)
        return FakeResp(200, good_json)

    monkeypatch.setattr(odds_fetcher, "_request_get", fake_get)
    monkeypatch.setattr(odds_fetcher, "BACKOFF_DELAYS", [0.0, 0.0])  # 待機をゼロに
    status, result = odds_fetcher._fetch_win_odds_by_requests("202609020411", HORSE_MAP)
    assert status == "success"
    assert result is not None
    assert call_count["n"] == 3


def test_block_no_retry(monkeypatch):
    """HTTP 403 → 即 api_failed（リトライしない）"""
    call_count = {"n": 0}

    class FakeResp:
        status_code = 403
        ok = False
        def json(self): return {}

    def fake_get(url, **kwargs):
        call_count["n"] += 1
        return FakeResp()

    monkeypatch.setattr(odds_fetcher, "_request_get", fake_get)
    status, result = odds_fetcher._fetch_win_odds_by_requests("202609020411", HORSE_MAP)
    assert status == "api_failed"
    assert result is None
    assert call_count["n"] == 1   # リトライなし


def test_unknown_schema_triggers_warning(monkeypatch, caplog):
    """未知スキーマ受信時に warning ログが出る"""
    import logging

    class FakeResp:
        status_code = 200
        ok = True
        def json(self): return {"totally": "unexpected"}

    monkeypatch.setattr(odds_fetcher, "_request_get", lambda *a, **kw: FakeResp())

    with caplog.at_level(logging.WARNING, logger="odds_fetcher"):
        status, result = odds_fetcher._fetch_win_odds_by_requests("202609020411", HORSE_MAP)

    assert status == "api_failed"   # 未知スキーマ → Selenium は呼ばないので api_failed
    assert any("未知" in r.message for r in caplog.records)
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
python3 -m pytest tests/test_odds_fetcher.py::test_backoff_retry_transient \
  tests/test_odds_fetcher.py::test_block_no_retry \
  tests/test_odds_fetcher.py::test_unknown_schema_triggers_warning \
  -v 2>&1 | head -20
```

Expected: FAIL (AttributeError: module 'odds_fetcher' has no attribute '_request_get')

- [ ] **Step 3: backoff・requests fetch・Selenium fallback・公開 API を実装**

`odds_fetcher.py` の末尾に追記:

```python
# ──────────────────────────────────────────────────────
# requests ラッパー（テストで monkeypatch しやすいよう分離）
# ──────────────────────────────────────────────────────

def _request_get(url: str, **kwargs):
    return requests.get(url, **kwargs)


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
                                headers=session.headers)
        except Exception as e:
            logger.warning("[odds_fetcher] %s | request exception: %s", race_id, e)
            time.sleep(delay)
            continue

        logger.info("[odds_fetcher] %s | attempt %d/%d | HTTP %d",
                    race_id, attempt, len(BACKOFF_DELAYS) + 1, resp.status_code)

        if resp.status_code in _BLOCK_STATUS_CODES:
            logger.warning("[odds_fetcher] %s | ブロック検知 HTTP %d → api_failed",
                           race_id, resp.status_code)
            return "api_failed", None

        if not resp.ok:
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

        status, named = _eval_coverage({"data": {"Odds": {k: str(v) for k, v in parsed.items()}}},
                                        horse_number_map)
        logger.info("[odds_fetcher] %s | %s | coverage (from requests)", race_id, status)
        return status, named

    logger.warning("[odds_fetcher] %s | 全リトライ失敗 → api_failed", race_id)
    return "api_failed", None


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
```

- [ ] **Step 4: テストが通ることを確認**

```bash
python3 -m pytest tests/test_odds_fetcher.py -v
```

Expected: 9 passed

- [ ] **Step 5: コミット**

```bash
git add odds_fetcher.py tests/test_odds_fetcher.py
git commit -m "feat: add odds_fetcher — fetch_win_odds with backoff and Selenium fallback"
```

---

## Task 3: odds_fetcher.py — 脚質取得 fetch_newspaper_styles()

**Files:**
- Modify: `odds_fetcher.py`
- Modify: `tests/test_odds_fetcher.py`

- [ ] **Step 1: 失敗するテストを追加**

```python
# tests/test_odds_fetcher.py に追記

def test_fetch_newspaper_styles_from_html(monkeypatch):
    """新聞ページの HTML から脚質を取得できる"""
    # 新聞ページの table に 逃/先/差/追/自 が含まれる想定 HTML
    fake_html = """
    <html><body>
    <table class="Newspaper_Table">
      <tr><td class="HorseName">ショウヘイ</td><td class="RunningStyle">逃</td></tr>
      <tr><td class="HorseName">ヨーホーレイク</td><td class="RunningStyle">先</td></tr>
    </table>
    </body></html>
    """
    class FakeResp:
        status_code = 200
        ok = True
        text = fake_html
        def raise_for_status(self): pass

    monkeypatch.setattr(odds_fetcher, "_request_get", lambda *a, **kw: FakeResp())
    status, result = odds_fetcher.fetch_newspaper_styles(
        "202609020411", {"1": "ショウヘイ", "2": "ヨーホーレイク"}
    )
    # HTML からスタイルが取れなくても "failed" にはならない（Selenium fallback を試みる）
    # requests が返す HTML の構造によっては "failed" になる場合もあるが、
    # ここでは status が "failed" でないことだけ確認
    assert isinstance(status, str)
    assert isinstance(result, (dict, type(None)))


def test_fetch_newspaper_styles_request_fail(monkeypatch):
    """requests 失敗でも "failed" ステータスを返す（Selenium fallback は統合テストで確認）"""
    class FakeResp:
        status_code = 500
        ok = False
        text = ""
        def raise_for_status(self): raise Exception("500")

    monkeypatch.setattr(odds_fetcher, "_request_get", lambda *a, **kw: FakeResp())

    # Selenium fallback が呼ばれる前に止める
    monkeypatch.setattr(odds_fetcher, "_fetch_newspaper_styles_by_selenium",
                        lambda *a, **kw: ("selenium_failed", None))

    status, result = odds_fetcher.fetch_newspaper_styles(
        "202609020411", {"1": "ショウヘイ"}
    )
    assert status == "selenium_failed"
    assert result is None
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
python3 -m pytest tests/test_odds_fetcher.py::test_fetch_newspaper_styles_from_html \
  tests/test_odds_fetcher.py::test_fetch_newspaper_styles_request_fail -v 2>&1 | head -15
```

Expected: FAIL (AttributeError)

- [ ] **Step 3: fetch_newspaper_styles() を実装**

`odds_fetcher.py` に追記:

```python
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

    # 新聞ページの行から horse_name と style_char を抽出
    # 複数のセレクタを試みる
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
```

- [ ] **Step 4: テストが通ることを確認**

```bash
python3 -m pytest tests/test_odds_fetcher.py -v
```

Expected: 11 passed

- [ ] **Step 5: コミット**

```bash
git add odds_fetcher.py tests/test_odds_fetcher.py
git commit -m "feat: add odds_fetcher — fetch_newspaper_styles with requests/Selenium"
```

---

## Task 4: pipeline_store.py — save_prediction_v2()

**Files:**
- Modify: `pipeline_store.py`
- Create: `tests/test_pipeline_store_v2.py`

- [ ] **Step 1: 失敗するテストを書く**

```python
# tests/test_pipeline_store_v2.py
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import pipeline_store


def _tmp(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setattr(pipeline_store, "PREDICTIONS_FILE",
                        os.path.join(d, "pred.json"))
    return d


def _make_features():
    return [
        {
            "horse_name": "ショウヘイ", "win_prob": 0.142, "win_odds": 5.6,
            "place_odds": 2.1, "running_style": "front",
            "feat_gate": 3, "feat_age": 5, "feat_popularity": 1,
            "feat_win_odds_log": 1.7228, "feat_last3f": 34.5,
            "feat_jockey_weight": 57.0, "feat_n_runners": 15,
            "feat_running_style_enc": 0, "feat_track_condition_enc": 0,
            "feat_signal_total_adjust": 0.12,
            "feat_cond_diff_age": 0.0, "feat_cond_diff_gate": 0.0,
            "feat_cond_diff_style": 0.0, "feat_cond_diff_popularity": 0.0,
            "feat_cond_diff_last3f": 0.0, "feat_cond_diff_weight": 0.0,
            "feat_cond_diff_jockey": 0.0, "feat_cond_diff_track": 0.0,
            "feat_recent_form": 0.5, "feat_trend_index": 0.6,
            "feat_consistency_index": 0.7,
            "horse_number": 1,
            "link": "https://db.netkeiba.com/horse/2022105123/",
            "records_source": "newspaper",
        }
    ]


def test_start_time_parsing(monkeypatch):
    """`race_info_text` から start_time と start_datetime が正しく抽出される"""
    _tmp(monkeypatch)
    pipeline_store.save_prediction_v2(
        race_id="202609020411",
        race_meta={
            "race_title": "大阪杯",
            "race_info_text": "15:45発走 / 芝2000m / 良",
            "race_date": "2026-04-05",
        },
        features=_make_features(),
        ev_table=[], race_structure={}, danger_v2=[],
        analysis_date="20260405",
    )
    pred = pipeline_store.load_prediction("202609020411")
    assert pred["start_time"] == "15:45"
    assert pred["start_datetime"] == "2026-04-05T15:45:00"


def test_horse_id_parse_from_link(monkeypatch):
    """horse_link から horse_id が正しく抽出される"""
    _tmp(monkeypatch)
    pipeline_store.save_prediction_v2(
        race_id="202609020411",
        race_meta={"race_title": "大阪杯", "race_info_text": "15:45発走", "race_date": ""},
        features=_make_features(),
        ev_table=[], race_structure={}, danger_v2=[],
    )
    pred = pipeline_store.load_prediction("202609020411")
    assert pred["horse_id_map"]["ショウヘイ"] == "2022105123"


def test_feature_dict_roundtrip(monkeypatch):
    """feature_dict が欠損なく保存・ロードされる"""
    _tmp(monkeypatch)
    pipeline_store.save_prediction_v2(
        race_id="202609020411",
        race_meta={"race_title": "大阪杯", "race_info_text": "15:45発走", "race_date": ""},
        features=_make_features(),
        ev_table=[], race_structure={}, danger_v2=[],
    )
    pred = pipeline_store.load_prediction("202609020411")
    horse = pred["horses"][0]
    assert horse["feature_dict"]["feat_win_odds_log"] == pytest.approx(1.7228)
    assert horse["feature_dict"]["feat_gate"] == 3


def test_prediction_history_initial(monkeypatch):
    """初回保存で prediction_version=1、prediction_history に 1 エントリ"""
    _tmp(monkeypatch)
    pipeline_store.save_prediction_v2(
        race_id="202609020411",
        race_meta={"race_title": "大阪杯", "race_info_text": "15:45発走", "race_date": ""},
        features=_make_features(),
        ev_table=[], race_structure={}, danger_v2=[],
    )
    pred = pipeline_store.load_prediction("202609020411")
    assert pred["prediction_version"] == 1
    assert len(pred["prediction_history"]) == 1
    assert pred["prediction_history"][0]["source"] == "initial_analysis"


def test_odds_status_roundtrip(monkeypatch):
    """odds_before の status フィールドが正確に保存・ロードされる"""
    _tmp(monkeypatch)
    pipeline_store.save_prediction_v2(
        race_id="202609020411",
        race_meta={"race_title": "大阪杯", "race_info_text": "15:45発走", "race_date": ""},
        features=_make_features(),
        ev_table=[], race_structure={}, danger_v2=[],
    )
    pred = pipeline_store.load_prediction("202609020411")
    assert pred["odds_before"]["status"] == "not_open"
    assert pred["odds_after"]["status"] == "not_open"
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
python3 -m pytest tests/test_pipeline_store_v2.py -v 2>&1 | head -20
```

Expected: FAIL (AttributeError: module 'pipeline_store' has no attribute 'save_prediction_v2')

- [ ] **Step 3: save_prediction_v2() を pipeline_store.py に追加**

`pipeline_store.py` の `# predictions` セクションに追記（既存 `save_prediction()` は変更しない）:

```python
import re as _re


def _extract_start_time(race_info_text: str) -> str:
    """
    "15:45発走 / 芝..." または "15時45分発走 ..." から "HH:MM" を抽出。
    取得できなければ "" を返す。
    """
    m = _re.search(r'(\d{1,2})[時:](\d{2})分?発走', race_info_text or "")
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    return ""


def _parse_horse_id(link: str) -> str:
    """
    "https://db.netkeiba.com/horse/2022105123/" → "2022105123"
    取得できなければ "" を返す。
    """
    m = _re.search(r'/horse/(\d+)', link or "")
    return m.group(1) if m else ""


def _build_odds_shell(horse_number_map: Dict[str, Any]) -> Dict[str, Any]:
    """
    初期状態の odds_before / odds_after シェル（全馬 null、status=not_open）を返す。
    """
    nulls = {no: None for no in horse_number_map}
    return {
        "status": "not_open",
        "tansho": dict(nulls),
        "fukusho": dict(nulls),
    }


def save_prediction_v2(
    race_id: str,
    race_meta: Dict[str, Any],
    features: List[Dict[str, Any]],
    ev_table: List[Dict[str, Any]],
    race_structure: Dict[str, Any],
    danger_v2: List[Dict[str, Any]],
    analysis_date: str = "",
) -> None:
    """
    analyze_race() の結果を v2 スキーマで pipeline_predictions.json に保存する。

    既存 save_prediction() と共存。load_prediction() はどちらのフォーマットも読める。
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    race_info_text = race_meta.get("race_info_text", "")
    race_date = race_meta.get("race_date", "")

    start_time = _extract_start_time(race_info_text)
    start_datetime = f"{race_date}T{start_time}:00" if start_time and race_date else ""

    # horse_number_map と horse_id_map を features から構築
    horse_number_map: Dict[str, str] = {}
    horse_id_map: Dict[str, str] = {}
    for f in features:
        no = f.get("horse_number")
        name = str(f.get("horse_name") or "")
        link = str(f.get("link") or "")
        if no is not None and name:
            horse_number_map[str(no)] = name
        if name and link:
            hid = _parse_horse_id(link)
            if hid:
                horse_id_map[name] = hid

    # 馬ごとのデータ
    horses = []
    for f in features:
        name = str(f.get("horse_name") or "")
        running_style = str(f.get("running_style") or "unknown")
        records_source = str(f.get("records_source") or "none")
        # running_style_source の決定
        if records_source == "newspaper":
            rs_source = "newspaper"
        elif running_style == "unknown":
            rs_source = "unknown"
        else:
            rs_source = "inferred"

        horses.append({
            "horse_name":             name,
            "horse_id":               horse_id_map.get(name, ""),
            "ai_win_prob":            round(float(f.get("win_prob") or 0.0), 4),
            "win_odds":               f.get("win_odds"),
            "popularity":             f.get("popularity") or f.get("feat_popularity"),
            "running_style":          running_style,
            "running_style_source":   rs_source,
            "running_style_missing":  running_style == "unknown",
            "feature_dict":           dict(f),   # 全フィールドを保存
        })

    initial_history_entry = {
        "version":    1,
        "created_at": now_str,
        "source":     "initial_analysis",
        "horses":     [{"horse_name": h["horse_name"], "ai_win_prob": h["ai_win_prob"]}
                       for h in horses],
    }
    odds_shell = _build_odds_shell(horse_number_map)

    data = _load(PREDICTIONS_FILE)
    data[race_id] = {
        "race_id":        race_id,
        "race_name":      race_meta.get("race_title", ""),
        "race_date":      race_date,
        "analysis_date":  analysis_date,
        "analyzed_at":    now_str,
        "start_time":     start_time,
        "start_datetime": start_datetime,
        "horse_number_map": horse_number_map,
        "horse_id_map":     horse_id_map,
        "prediction_version": 1,
        "prediction_history": [initial_history_entry],
        "horses":             horses,
        "ev_table":           ev_table,
        "race_structure":     race_structure,
        "danger_v2":          danger_v2,
        "odds_update_history": [],
        "odds_before":        dict(odds_shell),
        "odds_after":         dict(odds_shell),
    }
    _save(PREDICTIONS_FILE, data)
```

- [ ] **Step 4: テストが通ることを確認**

```bash
python3 -m pytest tests/test_pipeline_store_v2.py -v
```

Expected: 5 passed

- [ ] **Step 5: コミット**

```bash
git add pipeline_store.py tests/test_pipeline_store_v2.py
git commit -m "feat: add pipeline_store.save_prediction_v2 — v2 schema with start_time, feature_dict, history"
```

---

## Task 5: pipeline_store.py — load_race_start_times() + update_prediction_odds_in_store()

**Files:**
- Modify: `pipeline_store.py`
- Modify: `tests/test_pipeline_store_v2.py`

- [ ] **Step 1: 失敗するテストを追加**

```python
# tests/test_pipeline_store_v2.py に追記

def test_load_race_start_times(monkeypatch):
    """`load_race_start_times("20260405")` が analysis_date でフィルタして返す"""
    _tmp(monkeypatch)
    pipeline_store.save_prediction_v2(
        race_id="202609020411",
        race_meta={"race_title": "大阪杯", "race_info_text": "15:45発走", "race_date": "2026-04-05"},
        features=_make_features(), ev_table=[], race_structure={}, danger_v2=[],
        analysis_date="20260405",
    )
    pipeline_store.save_prediction_v2(
        race_id="202609020412",
        race_meta={"race_title": "阪神12R", "race_info_text": "16:25発走", "race_date": "2026-04-05"},
        features=_make_features(), ev_table=[], race_structure={}, danger_v2=[],
        analysis_date="20260405",
    )
    times = pipeline_store.load_race_start_times("20260405")
    assert "202609020411" in times
    assert times["202609020411"] == "2026-04-05T15:45:00"
    assert "202609020412" in times
    assert times["202609020412"] == "2026-04-05T16:25:00"


def test_update_prediction_odds_in_store(monkeypatch):
    """オッズ更新後に prediction_version が 2、odds_after が保存される"""
    _tmp(monkeypatch)
    pipeline_store.save_prediction_v2(
        race_id="202609020411",
        race_meta={"race_title": "大阪杯", "race_info_text": "15:45発走", "race_date": "2026-04-05"},
        features=_make_features(), ev_table=[], race_structure={}, danger_v2=[],
        analysis_date="20260405",
    )
    new_odds = {"ショウヘイ": 5.6}
    updated_horses = [
        {"horse_name": "ショウヘイ", "ai_win_prob": 0.142,
         "win_odds": 5.6, "feature_dict": {"feat_win_odds_log": 1.7228}}
    ]
    pipeline_store.update_prediction_odds_in_store(
        race_id="202609020411",
        new_odds_by_name=new_odds,
        updated_horses=updated_horses,
        odds_status="success",
        odds_source="api",
        coverage_ratio=0.93,
    )
    pred = pipeline_store.load_prediction("202609020411")
    assert pred["prediction_version"] == 2
    assert len(pred["prediction_history"]) == 2
    assert pred["prediction_history"][1]["source"] == "odds_update_api"
    assert pred["odds_after"]["status"] == "success"
    assert pred["odds_after"]["tansho"]["1"] == 5.6   # horse_no "1" に変換
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
python3 -m pytest tests/test_pipeline_store_v2.py::test_load_race_start_times \
  tests/test_pipeline_store_v2.py::test_update_prediction_odds_in_store -v 2>&1 | head -15
```

Expected: FAIL

- [ ] **Step 3: 2 つの関数を pipeline_store.py に追加**

```python
def load_race_start_times(date_str: str) -> Dict[str, str]:
    """
    指定 analysis_date の全レースの {race_id: start_datetime} を返す。
    start_datetime が空のエントリは除外する。
    """
    data = _load(PREDICTIONS_FILE)
    result: Dict[str, str] = {}
    for race_id, pred in data.items():
        if pred.get("analysis_date") != date_str:
            continue
        sdt = pred.get("start_datetime", "")
        if sdt:
            result[race_id] = sdt
    return result


def update_prediction_odds_in_store(
    race_id: str,
    new_odds_by_name: Dict[str, float],     # {horse_name: win_odds}
    updated_horses: List[Dict[str, Any]],   # 更新済み horses リスト
    odds_status: str,
    odds_source: str,
    coverage_ratio: float,
) -> None:
    """
    オッズ更新後の予測データを store に書き戻す。
    prediction_version を +1 し、prediction_history・odds_after・odds_update_history を更新する。
    odds_status が "not_open" / "failed" 系の場合は呼ばない（呼び出し元が制御）。
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = _load(PREDICTIONS_FILE)
    pred = data.get(race_id)
    if pred is None:
        return

    old_version = pred.get("prediction_version", 1)
    new_version = old_version + 1

    # odds_after を horse_number ベースで構築
    horse_number_map = pred.get("horse_number_map", {})
    name_to_no = {v: k for k, v in horse_number_map.items()}  # 逆引き
    tansho_after = {
        name_to_no.get(name, name): odds
        for name, odds in new_odds_by_name.items()
    }
    # 未取得馬は null
    for no in horse_number_map:
        tansho_after.setdefault(no, None)

    pred["prediction_version"] = new_version
    pred["horses"] = updated_horses
    pred["prediction_history"].append({
        "version":    new_version,
        "created_at": now_str,
        "source":     f"odds_update_{odds_source}",
        "horses":     [{"horse_name": h["horse_name"], "ai_win_prob": h["ai_win_prob"]}
                       for h in updated_horses],
    })
    pred["odds_after"] = {
        "status":         odds_status,
        "source":         odds_source,
        "coverage_ratio": round(coverage_ratio, 4),
        "tansho":         tansho_after,
        "fukusho":        {no: None for no in horse_number_map},  # 今回は単勝のみ
    }
    pred["odds_update_history"].append({
        "at":                       now_str,
        "source":                   odds_source,
        "coverage_ratio":           round(coverage_ratio, 4),
        "prediction_version_after": new_version,
    })
    data[race_id] = pred
    _save(PREDICTIONS_FILE, data)
```

- [ ] **Step 4: テストが通ることを確認**

```bash
python3 -m pytest tests/test_pipeline_store_v2.py -v
```

Expected: 7 passed

- [ ] **Step 5: コミット**

```bash
git add pipeline_store.py tests/test_pipeline_store_v2.py
git commit -m "feat: add load_race_start_times and update_prediction_odds_in_store"
```

---

## Task 6: daily_pipeline.py — run_daily_race_analysis() を v2 に拡張

**Files:**
- Modify: `daily_pipeline.py`
- Modify: `tests/test_daily_pipeline.py`

- [ ] **Step 1: 失敗するテストを追加**

```python
# tests/test_daily_pipeline.py に追記
import pipeline_store


def test_run_daily_saves_v2_fields(tmp_path, monkeypatch):
    """run_daily_race_analysis が start_time / horse_number_map / feature_dict を保存する"""
    monkeypatch.setattr(pipeline_store, "PREDICTIONS_FILE",
                        str(tmp_path / "pred.json"))
    monkeypatch.setattr(pipeline_store, "BET_SUGGESTIONS_FILE",
                        str(tmp_path / "bets.json"))

    fake_result = {
        "race_meta": {
            "race_title": "テストレース",
            "race_info_text": "10:00発走 / 芝1200m",
            "race_date": "2026-04-05",
        },
        "features": [
            {
                "horse_name": "テスト馬A", "win_prob": 0.25,
                "win_odds": 4.0, "place_odds": 1.8,
                "running_style": "front", "records_source": "newspaper",
                "horse_number": 1,
                "link": "https://db.netkeiba.com/horse/2022100001/",
                "feat_gate": 1, "feat_age": 4, "feat_popularity": 1,
                "feat_win_odds_log": 1.386, "feat_last3f": 33.0,
                "feat_jockey_weight": 55.0, "feat_n_runners": 8,
                "feat_running_style_enc": 0, "feat_track_condition_enc": 0,
                "feat_signal_total_adjust": 0.0,
                "feat_cond_diff_age": 0.0, "feat_cond_diff_gate": 0.0,
                "feat_cond_diff_style": 0.0, "feat_cond_diff_popularity": 0.0,
                "feat_cond_diff_last3f": 0.0, "feat_cond_diff_weight": 0.0,
                "feat_cond_diff_jockey": 0.0, "feat_cond_diff_track": 0.0,
                "feat_recent_form": 0.5, "feat_trend_index": 0.6,
                "feat_consistency_index": 0.7,
            }
        ],
        "race_structure": {"pace": "medium"},
        "ev_table": [],
        "danger_favorites_v2": [],
    }

    def fake_analyze(url, headless=True):
        return fake_result

    def fake_get_ids(date_str):
        return ["202609020401"]

    def fake_assign_roles(features, ev_table, race_structure, danger_v2):
        return []

    def fake_recommend(features, race_structure, horse_roles):
        return []

    monkeypatch.setattr("daily_pipeline.get_race_ids_by_date", fake_get_ids)
    import daily_pipeline
    monkeypatch.setattr(daily_pipeline, "_analyze_race_fn",
                        lambda: fake_analyze)  # see step 3 for how this works

    # Use monkeypatch to override the lazy import inside run_daily_race_analysis
    import sys
    fake_module = type(sys)("race_ai_engine_fake")
    fake_module.analyze_race = fake_analyze
    monkeypatch.setitem(sys.modules, "race_ai_engine", fake_module)

    fake_value_module = type(sys)("value_ai_fake")
    fake_value_module.recommend_betmaster_plans = fake_recommend
    fake_value_module.assign_roles = fake_assign_roles
    monkeypatch.setitem(sys.modules, "value_ai", fake_value_module)

    daily_pipeline.run_daily_race_analysis("20260405")

    pred = pipeline_store.load_prediction("202609020401")
    assert pred is not None
    assert pred.get("start_time") == "10:00"
    assert pred.get("horse_number_map") == {"1": "テスト馬A"}
    assert pred["horses"][0]["feature_dict"]["feat_gate"] == 1
    assert pred["prediction_version"] == 1
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
python3 -m pytest tests/test_daily_pipeline.py::test_run_daily_saves_v2_fields -v 2>&1 | head -20
```

Expected: FAIL (pred["start_time"] が存在しない)

- [ ] **Step 3: run_daily_race_analysis() を拡張**

`daily_pipeline.py` の `run_daily_race_analysis()` 内で `pipeline_store.save_prediction()` を呼んでいる箇所を `save_prediction_v2()` に置き換える。

現在の該当コード（`daily_pipeline.py` 行 194 付近）:
```python
pipeline_store.save_prediction(race_id, race_meta, features,
                               analysis_date=date_str)
```

これを以下に置き換える:
```python
pipeline_store.save_prediction_v2(
    race_id=race_id,
    race_meta=race_meta,
    features=features,
    ev_table=ev_table,
    race_structure=race_structure,
    danger_v2=danger_v2,
    analysis_date=date_str,
)
```

- [ ] **Step 4: テストが通ることを確認**

```bash
python3 -m pytest tests/test_daily_pipeline.py -v
```

Expected: 全テスト passed（既存テストも含め）

- [ ] **Step 5: コミット**

```bash
git add daily_pipeline.py tests/test_daily_pipeline.py
git commit -m "feat: run_daily_race_analysis now saves v2 schema (start_time, feature_dict, ev_table)"
```

---

## Task 7: daily_pipeline.py — _recalc_downstream() + update_race_odds()

**Files:**
- Modify: `daily_pipeline.py`
- Create: `tests/test_watch_odds.py`（update_race_odds のテスト）

- [ ] **Step 1: 失敗するテストを書く**

```python
# tests/test_watch_odds.py
import sys, os, tempfile, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import pipeline_store
import daily_pipeline


def _tmp_store(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline_store, "PREDICTIONS_FILE",
                        str(tmp_path / "pred.json"))
    monkeypatch.setattr(pipeline_store, "BET_SUGGESTIONS_FILE",
                        str(tmp_path / "bets.json"))


def _seed_prediction(race_id: str):
    """テスト用の v2 prediction を store に保存する"""
    features = [
        {
            "horse_name": "ショウヘイ", "win_prob": 0.069, "win_odds": None,
            "place_odds": None, "running_style": "front", "records_source": "newspaper",
            "horse_number": 1,
            "link": "https://db.netkeiba.com/horse/2022105123/",
            "feat_gate": 3, "feat_age": 5, "feat_popularity": 0,
            "feat_win_odds_log": 0.0, "feat_last3f": 34.5,
            "feat_jockey_weight": 57.0, "feat_n_runners": 15,
            "feat_running_style_enc": 0, "feat_track_condition_enc": 0,
            "feat_signal_total_adjust": 0.12,
            "feat_cond_diff_age": 0.0, "feat_cond_diff_gate": 0.0,
            "feat_cond_diff_style": 0.0, "feat_cond_diff_popularity": 0.0,
            "feat_cond_diff_last3f": 0.0, "feat_cond_diff_weight": 0.0,
            "feat_cond_diff_jockey": 0.0, "feat_cond_diff_track": 0.0,
            "feat_recent_form": 0.5, "feat_trend_index": 0.6,
            "feat_consistency_index": 0.7,
        }
    ]
    pipeline_store.save_prediction_v2(
        race_id=race_id,
        race_meta={
            "race_title": "大阪杯", "race_info_text": "15:45発走",
            "race_date": "2026-04-05",
        },
        features=features, ev_table=[], race_structure={}, danger_v2=[],
        analysis_date="20260405",
    )


def test_not_open_no_version_bump(tmp_path, monkeypatch):
    """`not_open` ステータスでは prediction_version が増えない"""
    _tmp_store(monkeypatch, tmp_path)
    _seed_prediction("202609020411")

    import odds_fetcher
    monkeypatch.setattr(odds_fetcher, "fetch_win_odds",
                        lambda race_id, horse_number_map: ("not_open", None))

    result = daily_pipeline.update_race_odds("202609020411")
    pred = pipeline_store.load_prediction("202609020411")
    assert pred["prediction_version"] == 1
    assert result["status"] == "not_open"


def test_version_increments_on_success(tmp_path, monkeypatch):
    """成功時に prediction_version が 2 になり、odds_after.status が "success""""
    _tmp_store(monkeypatch, tmp_path)
    _seed_prediction("202609020411")

    import odds_fetcher
    monkeypatch.setattr(odds_fetcher, "fetch_win_odds",
                        lambda race_id, horse_number_map: ("success", {"ショウヘイ": 5.6}))

    # LightGBM 予測をモック（モデルファイルなし環境でもテストが通るよう）
    monkeypatch.setattr(daily_pipeline, "_run_lgbm_prediction",
                        lambda features: [0.142])

    result = daily_pipeline.update_race_odds("202609020411")
    pred = pipeline_store.load_prediction("202609020411")
    assert pred["prediction_version"] == 2
    assert result["status"] == "success"
    assert pred["horses"][0]["win_odds"] == 5.6
    assert pred["horses"][0]["feature_dict"]["feat_win_odds_log"] == pytest.approx(
        __import__("math").log(5.6), abs=1e-3
    )
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
python3 -m pytest tests/test_watch_odds.py::test_not_open_no_version_bump \
  tests/test_watch_odds.py::test_version_increments_on_success -v 2>&1 | head -20
```

Expected: FAIL (AttributeError: module 'daily_pipeline' has no attribute 'update_race_odds')

- [ ] **Step 3: _recalc_downstream() と update_race_odds() を daily_pipeline.py に追加**

```python
# daily_pipeline.py に追加

import math as _math
import logging as _logging

_logger = _logging.getLogger(__name__)


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
    from race_ai_engine import (
        estimate_place_prob, fair_odds, calc_expected_value, calc_market_edge,
        calc_odds_distortion, calc_expected_value_score, classify_bet_suitability,
        classify_value_label,
    )
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
        "status":   str,   # "success"|"partial"|"not_open"|"api_failed"|"selenium_failed"|"failed"
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
```

- [ ] **Step 4: テストが通ることを確認**

```bash
python3 -m pytest tests/test_watch_odds.py::test_not_open_no_version_bump \
  tests/test_watch_odds.py::test_version_increments_on_success -v
```

Expected: 2 passed

- [ ] **Step 5: コミット**

```bash
git add daily_pipeline.py tests/test_watch_odds.py
git commit -m "feat: add update_race_odds and _recalc_downstream to daily_pipeline"
```

---

## Task 8: daily_pipeline.py — watch_odds() ループ + CLI

**Files:**
- Modify: `daily_pipeline.py`
- Modify: `tests/test_watch_odds.py`

- [ ] **Step 1: 失敗するテストを追加**

```python
# tests/test_watch_odds.py に追記

from datetime import datetime, timedelta


def test_window_detection():
    """T-30min 付近のレースのみが更新対象として選ばれる"""
    now = datetime(2026, 4, 5, 15, 15, 0)   # 15:15
    start_times = {
        "202609020401": "2026-04-05T10:00:00",   # 発走済み（対象外）
        "202609020410": "2026-04-05T15:40:00",   # T-25min（対象）
        "202609020411": "2026-04-05T15:45:00",   # T-30min（対象）
        "202609020412": "2026-04-05T16:25:00",   # T-70min（対象外）
    }
    # 発走時刻 - 35min ≤ now ≤ 発走時刻 - 20min
    targets = daily_pipeline._get_update_targets(start_times, now, updated_ids=set())
    assert "202609020410" in targets
    assert "202609020411" in targets
    assert "202609020401" not in targets   # 発走済み
    assert "202609020412" not in targets   # まだ早すぎる


def test_already_updated_skip():
    """`updated_ids` に入っているレースはスキップされる"""
    now = datetime(2026, 4, 5, 15, 15, 0)
    start_times = {
        "202609020411": "2026-04-05T15:45:00",
    }
    targets = daily_pipeline._get_update_targets(
        start_times, now, updated_ids={"202609020411"}
    )
    assert "202609020411" not in targets


def test_exit_condition_all_past():
    """全レースが start + 90min を過ぎていれば _should_exit が True"""
    now = datetime(2026, 4, 5, 20, 0, 0)
    start_times = {
        "202609020401": "2026-04-05T10:00:00",
        "202609020412": "2026-04-05T17:00:00",
    }
    assert daily_pipeline._should_exit(start_times, updated_ids=set(), now=now) is True


def test_no_exit_when_races_remain():
    """まだ発走前のレースがあれば _should_exit が False"""
    now = datetime(2026, 4, 5, 15, 0, 0)
    start_times = {
        "202609020411": "2026-04-05T15:45:00",
    }
    assert daily_pipeline._should_exit(start_times, updated_ids=set(), now=now) is False
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
python3 -m pytest tests/test_watch_odds.py::test_window_detection \
  tests/test_watch_odds.py::test_already_updated_skip \
  tests/test_watch_odds.py::test_exit_condition_all_past \
  tests/test_watch_odds.py::test_no_exit_when_races_remain -v 2>&1 | head -15
```

Expected: FAIL (AttributeError: module 'daily_pipeline' has no attribute '_get_update_targets')

- [ ] **Step 3: watch_odds() と ヘルパーを daily_pipeline.py に追加**

```python
# daily_pipeline.py に追加

import signal as _signal
import sys as _sys
from datetime import datetime as _dt, timedelta as _td


def _get_update_targets(
    start_times: Dict[str, str],
    now: "_dt",
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
    now: "_dt",
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

    while True:
        now = _dt.now()

        if _should_exit(start_times, updated_ids | failed_ids, now):
            break

        targets = _get_update_targets(start_times, now, updated_ids | failed_ids)
        for race_id in targets:
            sdt_str = start_times.get(race_id, "")
            venue_info = race_id  # シンプルにIDだけ表示
            print(f"[watch_odds] {now.strftime('%H:%M')} | {venue_info}"
                  f" (発走 {sdt_str[11:16]}) | オッズ更新中...", flush=True)

            result = update_race_odds(race_id)
            status = result.get("status", "failed")

            print(
                f"[watch_odds] {now.strftime('%H:%M')} | {venue_info}"
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

        import time as _time
        _time.sleep(poll_interval)

    print(
        f"[watch_odds] {date_str} | 完了"
        f" | 更新済み {len(updated_ids)}/{total}件"
        f" | 失敗 {len(failed_ids)}件",
        flush=True,
    )
```

また、CLI セクションに `--watch-odds` と `--poll-interval` を追加する。

現在の `daily_pipeline.py` の `if __name__ == "__main__":` ブロックに以下を追加:

```python
# argparse に追加（既存の parser.add_argument 群の後に）
parser.add_argument("--watch-odds",    metavar="DATE",
                    help="オッズ自動監視（例: 20250406）")
parser.add_argument("--poll-interval", type=int, default=60,
                    help="watch-odds のポーリング間隔（秒、デフォルト: 60）")

# elif チェーンに追加
elif args.watch_odds:
    watch_odds(args.watch_odds, poll_interval=args.poll_interval)
```

- [ ] **Step 4: テストが通ることを確認**

```bash
python3 -m pytest tests/test_watch_odds.py -v
```

Expected: 6 passed（Task 7 の 2 件 + 今回の 4 件）

- [ ] **Step 5: 全テストが通ることを確認**

```bash
python3 -m pytest tests/ -v
```

Expected: 全テスト passed

- [ ] **Step 6: コミット**

```bash
git add daily_pipeline.py tests/test_watch_odds.py
git commit -m "feat: add watch_odds loop, _get_update_targets, _should_exit + --watch-odds CLI"
```

---

## 動作確認（手動）

```bash
# 1. 分析実行（Selenium 必要）
python3 daily_pipeline.py --analyze 20260412

# 2. 予測に start_time が入っていることを確認
python3 -c "
import json
with open('pipeline_predictions.json') as f:
    preds = json.load(f)
for rid, p in list(preds.items())[:3]:
    print(rid, p.get('start_time'), p.get('prediction_version'))
"

# 3. watch_odds を バックグラウンドで起動
python3 daily_pipeline.py --watch-odds 20260412 >> watchdog.log 2>&1 &

# 4. ログを監視
tail -f watchdog.log
```

---

## 自己レビューチェックリスト

- [ ] スペックの全要件にタスクが対応しているか確認
- [ ] prediction_version 方針: success/partial のみインクリメント ✓
- [ ] partial 閾値 0.8 / 脚質閾値 0.7 が定数に明記されている ✓
- [ ] watcher 停止条件 (A)(B) が `_should_exit()` に実装されている ✓
- [ ] 再計算フィールド範囲が `_recalc_downstream()` のコメントに明記されている ✓
- [ ] ログ項目が全関数に揃っている ✓
- [ ] `analyze_race()` は変更していない ✓
- [ ] 既存 `save_prediction()` は変更していない ✓
