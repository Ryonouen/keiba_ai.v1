# tests/test_kelly_staking.py
"""Tests for kelly_staking module."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from kelly_staking import (
    compute_kelly_fraction,
    kelly_stake,
    apply_kelly_to_bets,
    load_kelly_config,
    save_kelly_config,
)


# ── compute_kelly_fraction ──────────────────────────────────────

def test_kelly_fraction_positive_ev():
    # b=2.0, p=0.5, q=0.5 → f* = (2.0*0.5 - 0.5)/2.0 = 0.25; fraction=0.25 → 0.0625
    f = compute_kelly_fraction(win_prob=0.5, win_odds=3.0, fraction=0.25)
    assert abs(f - 0.0625) < 1e-9


def test_kelly_fraction_negative_ev():
    # b=1.0, p=0.1 → f* = (1.0*0.1 - 0.9)/1.0 = -0.8 → clamp to 0.0
    f = compute_kelly_fraction(win_prob=0.1, win_odds=2.0, fraction=0.25)
    assert f == 0.0


def test_kelly_fraction_edge_odds_one():
    # win_odds=1.0 → b=0.0 → division by zero guard → 0.0
    f = compute_kelly_fraction(win_prob=0.5, win_odds=1.0, fraction=0.25)
    assert f == 0.0


def test_kelly_fraction_full_fraction():
    # fraction=1.0 (full Kelly): b=2.0, p=0.5 → f* = 0.25
    f = compute_kelly_fraction(win_prob=0.5, win_odds=3.0, fraction=1.0)
    assert abs(f - 0.25) < 1e-9


# ── kelly_stake ────────────────────────────────────────────────

def test_kelly_stake_normal():
    # f=0.0625, bankroll=10000 → 625 → round down to 600
    s = kelly_stake(win_prob=0.5, win_odds=3.0, bankroll=10000, fraction=0.25)
    assert s == 600


def test_kelly_stake_min_clamp():
    # negative EV → f=0 → stake clamped to min_stake=100
    s = kelly_stake(win_prob=0.1, win_odds=2.0, bankroll=10000, fraction=0.25)
    assert s == 100


def test_kelly_stake_max_clamp():
    # very high prob → stake clamped to max_stake=10000
    s = kelly_stake(win_prob=0.99, win_odds=10.0, bankroll=10_000_000, fraction=1.0,
                    max_stake=10000)
    assert s == 10000


# ── apply_kelly_to_bets ────────────────────────────────────────

def test_apply_kelly_to_bets_overrides_stake():
    bets = [
        {
            "bet_type": "tansho",
            "stake_amount": 100,
            "_win_prob": 0.5,
            "_win_odds": 3.0,
        }
    ]
    config = {"enabled": True, "bankroll": 10000, "fraction": 0.25}
    result = apply_kelly_to_bets(bets, config)
    assert result[0]["stake_amount"] == 600


def test_apply_kelly_to_bets_disabled_passthrough():
    bets = [{"bet_type": "tansho", "stake_amount": 100, "_win_prob": 0.5, "_win_odds": 3.0}]
    config = {"enabled": False, "bankroll": 10000, "fraction": 0.25}
    result = apply_kelly_to_bets(bets, config)
    assert result[0]["stake_amount"] == 100


# ── load/save config ───────────────────────────────────────────

def test_save_and_load_config(tmp_path):
    path = str(tmp_path / "kelly_config.json")
    cfg = {"enabled": True, "bankroll": 50000, "fraction": 0.25}
    save_kelly_config(cfg, path=path)
    loaded = load_kelly_config(path=path)
    assert loaded == cfg
