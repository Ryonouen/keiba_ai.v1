"""tests/test_confidence_scorer.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import confidence_scorer as cs


# ── _sigmoid ──────────────────────────────────────────────────────
def test_sigmoid_zero_returns_half():
    assert abs(cs._sigmoid(0.0) - 0.5) < 1e-9

def test_sigmoid_large_positive_approaches_1():
    assert cs._sigmoid(10.0) > 0.99

def test_sigmoid_large_negative_approaches_0():
    assert cs._sigmoid(-10.0) < 0.01


# ── compute_race_confidence ───────────────────────────────────────
def _make_features(n=8, top_win_prob=0.4, top_ev=0.5, top_edge=0.3):
    features = []
    for i in range(n):
        prob = top_win_prob if i == 0 else (1.0 - top_win_prob) / (n - 1)
        features.append({
            "horse_name": f"horse_{i}",
            "win_prob":   round(prob, 4),
            "win_ev":     top_ev if i == 0 else -0.1,
            "win_market_edge": top_edge if i == 0 else -0.05,
        })
    return features

def test_confidence_is_between_0_and_1():
    features = _make_features()
    score = cs.compute_race_confidence(features)
    assert 0.0 <= score <= 1.0

def test_high_ev_race_has_higher_score():
    low_ev  = _make_features(top_ev=0.0, top_edge=0.0, top_win_prob=0.15)
    high_ev = _make_features(top_ev=1.0, top_edge=0.5, top_win_prob=0.50)
    assert cs.compute_race_confidence(high_ev) > cs.compute_race_confidence(low_ev)

def test_empty_features_returns_0():
    assert cs.compute_race_confidence([]) == 0.0

def test_missing_win_ev_uses_0():
    features = [{"horse_name": "A", "win_prob": 0.5}]
    score = cs.compute_race_confidence(features)
    assert 0.0 <= score <= 1.0


# ── filter_races_by_confidence ────────────────────────────────────
def test_filter_returns_only_above_threshold():
    races = [
        {"race_id": "r1", "confidence_score": 0.8},
        {"race_id": "r2", "confidence_score": 0.3},
        {"race_id": "r3", "confidence_score": 0.6},
    ]
    result = cs.filter_races_by_confidence(races, threshold=0.5)
    assert {r["race_id"] for r in result} == {"r1", "r3"}

def test_filter_empty_returns_empty():
    assert cs.filter_races_by_confidence([], threshold=0.5) == []

def test_filter_threshold_0_returns_all():
    races = [{"race_id": "r1", "confidence_score": 0.1}]
    assert len(cs.filter_races_by_confidence(races, threshold=0.0)) == 1
