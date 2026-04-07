# tests/test_kelly_config_io.py
"""Test Kelly config save/load round-trip."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from kelly_staking import save_kelly_config, load_kelly_config


def test_save_and_load_round_trip(tmp_path):
    path = str(tmp_path / "kelly_config.json")
    cfg_in = {"enabled": True, "bankroll": 30000, "fraction": 0.5}
    save_kelly_config(cfg_in, path=path)
    cfg_out = load_kelly_config(path=path)
    assert cfg_out["enabled"] is True
    assert cfg_out["bankroll"] == 30000
    assert cfg_out["fraction"] == 0.5


def test_load_missing_file_returns_defaults(tmp_path):
    path = str(tmp_path / "missing.json")
    cfg = load_kelly_config(path=path)
    assert cfg["enabled"] is False


def test_save_creates_file(tmp_path):
    path = str(tmp_path / "kelly_config.json")
    save_kelly_config({"enabled": False, "bankroll": 10000, "fraction": 0.25}, path=path)
    assert (tmp_path / "kelly_config.json").exists()
