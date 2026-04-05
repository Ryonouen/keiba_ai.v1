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
    assert "サンスト" not in result   # "–" は除外


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
    result = odds_fetcher._parse_odds_response(raw)
    assert result is not None
    # normalize_horse_no で "01" → "1"
    norm = {odds_fetcher._normalize_horse_no(k): v for k, v in result.items()}
    assert norm["1"] == 5.6
    assert norm["2"] == 12.3


def test_parse_api_response_win_odds_path():
    """data["data"]["WinOdds"] パス（パス2）から正常にパースできる"""
    raw = {"data": {"WinOdds": {"1": "5.6", "2": "12.3", "3": "8.0", "4": "15.1",
                                "5": "22.5", "6": "7.2", "7": "18.0", "8": "9.9"}}}
    result = odds_fetcher._parse_odds_response(raw)
    assert result is not None
    assert result["1"] == 5.6
    assert result["8"] == 9.9


def test_parse_api_response_nested_win_odds_path():
    """data["data"]["Odds"]["WinOdds"] パス（パス3）から正常にパースできる"""
    raw = {"data": {"Odds": {"WinOdds": {"1": "5.6", "2": "12.3", "3": "8.0", "4": "15.1",
                                         "5": "22.5", "6": "7.2", "7": "18.0", "8": "9.9"}}}}
    result = odds_fetcher._parse_odds_response(raw)
    assert result is not None
    assert result["1"] == 5.6
    assert result["8"] == 9.9


def test_status_not_open_nested_schema():
    """path-3 スキーマ（data["data"]["Odds"]["WinOdds"]）で全オッズ "–" → status=not_open"""
    raw = {"data": {"Odds": {"WinOdds": {"1": "–", "2": "–", "3": "–", "4": "–",
                                          "5": "–", "6": "–", "7": "–", "8": "–"}}}}
    status, result = odds_fetcher._eval_coverage(raw, HORSE_MAP)
    assert status == "not_open"
    assert result is None


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
    monkeypatch.setattr(odds_fetcher, "BACKOFF_DELAYS", [0.0, 0.0, 0.0])  # 3 attempts, zero wait
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

    assert status == "api_failed"   # 未知スキーマ → api_failed
    assert any("未知" in r.message for r in caplog.records)


def test_fetch_newspaper_styles_from_html(monkeypatch):
    """新聞ページの HTML から脚質を取得できる"""
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
    assert isinstance(status, str)
    assert isinstance(result, (dict, type(None)))


def test_fetch_newspaper_styles_request_fail(monkeypatch):
    """requests 失敗 → Selenium fallback が呼ばれ、その結果が返る"""
    class FakeResp:
        status_code = 500
        ok = False
        text = ""
        def raise_for_status(self): raise Exception("500")

    monkeypatch.setattr(odds_fetcher, "_request_get", lambda *a, **kw: FakeResp())
    monkeypatch.setattr(odds_fetcher, "_fetch_newspaper_styles_by_selenium",
                        lambda *a, **kw: ("selenium_failed", None))

    status, result = odds_fetcher.fetch_newspaper_styles(
        "202609020411", {"1": "ショウヘイ"}
    )
    assert status == "failed"  # both failed → "failed"
    assert result is None
