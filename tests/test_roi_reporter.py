"""tests/test_roi_reporter.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import roi_reporter as rr


# ── テスト用データ ──────────────────────────────────────────────
def _mock_outcomes():
    return {
        "race_001": [
            {"bet_type": "tansho",      "bet_type_label": "単勝",     "stake": 100, "hit": True,  "payout": 500},
            {"bet_type": "wide",        "bet_type_label": "ワイド",    "stake": 100, "hit": False, "payout": 0},
            {"bet_type": "umaren",      "bet_type_label": "馬連（流し）", "stake": 100, "hit": True, "payout": 1200},
        ],
        "race_002": [
            {"bet_type": "tansho",      "bet_type_label": "単勝",     "stake": 100, "hit": False, "payout": 0},
            {"bet_type": "sanrenpuku_ai","bet_type_label": "三連複（AI絞り）", "stake": 100, "hit": True, "payout": 3000},
        ],
    }

def _mock_predictions():
    return {
        "race_001": {"analysis_date": "20260405"},
        "race_002": {"analysis_date": "20260405"},
    }


# ── aggregate_by_bet_type ────────────────────────────────────────
def test_aggregate_returns_correct_totals():
    outcomes = _mock_outcomes()
    result = rr.aggregate_by_bet_type(outcomes)
    assert result["tansho"]["bets"] == 2
    assert result["tansho"]["hits"] == 1
    assert result["tansho"]["stake"] == 200
    assert result["tansho"]["payout"] == 500
    assert result["tansho"]["hit_rate"] == 50.0
    assert result["tansho"]["roi"] == 250.0  # 500/200*100

def test_aggregate_wide_all_miss():
    outcomes = _mock_outcomes()
    result = rr.aggregate_by_bet_type(outcomes)
    assert result["wide"]["hits"] == 0
    assert result["wide"]["roi"] == 0.0

def test_aggregate_empty_outcomes():
    result = rr.aggregate_by_bet_type({})
    assert result == {}


# ── filter_by_dates ──────────────────────────────────────────────
def test_filter_by_dates_returns_matching_races():
    outcomes = _mock_outcomes()
    preds    = _mock_predictions()
    filtered = rr.filter_outcomes_by_dates(outcomes, preds, ["20260405"])
    assert set(filtered.keys()) == {"race_001", "race_002"}

def test_filter_by_dates_excludes_other_dates():
    outcomes = _mock_outcomes()
    preds    = _mock_predictions()
    filtered = rr.filter_outcomes_by_dates(outcomes, preds, ["20260406"])
    assert filtered == {}


# ── generate_markdown_report ─────────────────────────────────────
def test_markdown_report_contains_headers():
    outcomes = _mock_outcomes()
    summary = rr.aggregate_by_bet_type(outcomes)
    md = rr.generate_markdown_report(summary, dates=["20260405"])
    assert "# 券種別ROIレポート" in md
    assert "単勝" in md
    assert "ROI" in md

def test_markdown_report_shows_roi():
    outcomes = _mock_outcomes()
    summary = rr.aggregate_by_bet_type(outcomes)
    md = rr.generate_markdown_report(summary, dates=["20260405"])
    assert "250.0" in md  # tansho ROI


# ── generate_csv_report ──────────────────────────────────────────
def test_csv_report_has_header_and_rows(tmp_path):
    outcomes = _mock_outcomes()
    summary = rr.aggregate_by_bet_type(outcomes)
    csv_path = tmp_path / "test_report.csv"
    rr.generate_csv_report(summary, str(csv_path), dates=["20260405"])
    content = csv_path.read_text(encoding="utf-8")
    assert "券種" in content
    assert "単勝" in content
    assert "250.0" in content

def test_csv_report_creates_parent_dirs(tmp_path):
    outcomes = _mock_outcomes()
    summary = rr.aggregate_by_bet_type(outcomes)
    deep_path = tmp_path / "a" / "b" / "report.csv"
    rr.generate_csv_report(summary, str(deep_path), dates=["20260405"])
    assert deep_path.exists()
