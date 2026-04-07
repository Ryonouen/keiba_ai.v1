# tests/test_daily_pipeline_kelly_cli.py
"""Test that kelly_config defaults are accessible."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


def test_load_kelly_config_returns_defaults():
    """load_kelly_config が enabled/bankroll/fraction を持つことを確認。"""
    from kelly_staking import load_kelly_config
    cfg = load_kelly_config()
    assert "enabled" in cfg
    assert "bankroll" in cfg
    assert "fraction" in cfg


def test_kelly_config_defaults_when_missing():
    from kelly_staking import load_kelly_config
    cfg = load_kelly_config(path="/nonexistent/path.json")
    assert cfg["enabled"] is False
    assert cfg["bankroll"] == 10_000
    assert cfg["fraction"] == 0.25
