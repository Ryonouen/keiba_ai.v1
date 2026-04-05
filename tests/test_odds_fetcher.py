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
