# tests/test_daily_pipeline_kelly.py
"""Kelly integration tests for generate_all_bets."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


def _make_plan(bet_type="単勝", stake=100, win_prob=0.5, win_odds=3.0):
    return {
        "bet_type": bet_type,
        "tickets": [{"combination": ["馬A"], "stake": stake}],
        "selection_reason": "テスト",
        "confidence_score": 0.8,
        "_horse_win_prob": win_prob,
        "_horse_win_odds": win_odds,
    }


def test_generate_all_bets_kelly_disabled():
    from daily_pipeline import generate_all_bets
    plans = [_make_plan()]
    bets = generate_all_bets("dummy_race_id", plans, kelly_config={"enabled": False, "bankroll": 10000, "fraction": 0.25})
    assert bets[0]["stake_amount"] == 100


def test_generate_all_bets_kelly_enabled():
    from daily_pipeline import generate_all_bets
    plans = [_make_plan(win_prob=0.5, win_odds=3.0)]
    bets = generate_all_bets("dummy_race_id", plans, kelly_config={"enabled": True, "bankroll": 10000, "fraction": 0.25})
    # f=0.0625, 10000*0.0625=625 → round down to 600
    assert bets[0]["stake_amount"] == 600


def test_generate_all_bets_kelly_none_passthrough():
    from daily_pipeline import generate_all_bets
    plans = [_make_plan()]
    bets = generate_all_bets("dummy_race_id", plans, kelly_config=None)
    assert bets[0]["stake_amount"] == 100
