"""tests/test_ranker_engine.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

import ranker_engine as re_mod


# ── _build_relevance_labels ──────────────────────────────────────
def test_relevance_winner_gets_2():
    df = pd.DataFrame({"target_win": [1, 0, 0], "target_top3": [1, 1, 0]})
    labels = re_mod._build_relevance_labels(df)
    assert labels.tolist() == [2, 1, 0]

def test_relevance_clipped_at_2():
    df = pd.DataFrame({"target_win": [1, 0], "target_top3": [1, 0]})
    labels = re_mod._build_relevance_labels(df)
    assert labels.max() == 2

def test_relevance_handles_nan():
    df = pd.DataFrame({"target_win": [float("nan"), 0], "target_top3": [float("nan"), 1]})
    labels = re_mod._build_relevance_labels(df)
    assert labels.tolist() == [0, 1]


# ── blend_scores ──────────────────────────────────────────────────
def test_blend_returns_win_probs_when_rank_none():
    wp = [0.5, 0.3, 0.2]
    result = re_mod.blend_scores(wp, None)
    assert result == wp

def test_blend_returns_win_probs_when_length_mismatch():
    wp = [0.5, 0.3, 0.2]
    result = re_mod.blend_scores(wp, [0.8, 0.2])
    assert result == wp

def test_blend_normalized_sums_to_1():
    wp = [0.6, 0.3, 0.1]
    rs = [10.0, 5.0, 2.0]
    result = re_mod.blend_scores(wp, rs, weight_ranker=0.3)
    assert abs(sum(result) - 1.0) < 1e-9
    assert len(result) == 3

def test_blend_weight_0_returns_win_probs():
    wp = [0.6, 0.3, 0.1]
    rs = [10.0, 5.0, 2.0]
    result = re_mod.blend_scores(wp, rs, weight_ranker=0.0)
    assert abs(sum(result) - 1.0) < 1e-9

def test_blend_flat_rank_scores():
    wp = [0.5, 0.3, 0.2]
    rs = [1.0, 1.0, 1.0]
    result = re_mod.blend_scores(wp, rs, weight_ranker=0.3)
    assert abs(sum(result) - 1.0) < 1e-9


# ── RANKER_PROFILES 定義 ──────────────────────────────────────────
def test_all_profiles_defined():
    assert set(re_mod.RANKER_PROFILES.keys()) == {"conservative", "balanced", "aggressive"}

def test_each_profile_has_required_keys():
    for name, cfg in re_mod.RANKER_PROFILES.items():
        assert "model_file" in cfg, f"{name} missing model_file"
        assert "params" in cfg, f"{name} missing params"
        assert "num_boost_round" in cfg, f"{name} missing num_boost_round"
        assert cfg["params"]["objective"] == "lambdarank", f"{name} wrong objective"


# ── predict_rank_score (モデルなし) ──────────────────────────────
def test_predict_returns_none_when_model_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(re_mod, "RANKER_MODEL_DIR", str(tmp_path))
    result = re_mod.predict_rank_score([{"feat_gate": 1}], profile="balanced")
    assert result is None

def test_predict_returns_none_for_unknown_profile():
    result = re_mod.predict_rank_score([{"feat_gate": 1}], profile="unknown_profile")
    assert result is None
