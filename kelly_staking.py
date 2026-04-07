"""
kelly_staking.py
----------------
ケリー基準に基づく賭け金自動調整ロジック。

使い方:
  from kelly_staking import compute_kelly_fraction, kelly_stake, apply_kelly_to_bets

ケリー計算式:
  b  = win_odds - 1          # net odds
  f* = (b * p - (1-p)) / b  # full Kelly fraction
  f  = fraction * f*         # fractional Kelly (デフォルト 0.25)
  stake = floor(f * bankroll / 100) * 100  # 100円単位
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG_PATH = os.path.join(_HERE, "kelly_config.json")

KELLY_UNIT: int = 100
KELLY_MIN_STAKE: int = 100
KELLY_MAX_STAKE: int = 10_000

_DEFAULT_KELLY_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "bankroll": 10_000,
    "fraction": 0.25,
}


def compute_kelly_fraction(
    win_prob: float,
    win_odds: float,
    fraction: float = 0.25,
) -> float:
    """フラクショナル・ケリー比率を返す（0.0〜fraction）。"""
    if win_odds <= 1.0:
        return 0.0
    b = win_odds - 1.0
    q = 1.0 - win_prob
    full_kelly = (b * win_prob - q) / b
    if full_kelly <= 0.0:
        return 0.0
    return fraction * full_kelly


def kelly_stake(
    win_prob: float,
    win_odds: float,
    bankroll: int,
    fraction: float = 0.25,
    min_stake: int = KELLY_MIN_STAKE,
    max_stake: int = KELLY_MAX_STAKE,
) -> int:
    """ケリー基準による賭け金（円）を返す。EV マイナスの場合は min_stake を返す。"""
    f = compute_kelly_fraction(win_prob, win_odds, fraction)
    if f <= 0.0:
        return min_stake
    raw = f * bankroll
    rounded = int(raw // KELLY_UNIT) * KELLY_UNIT
    return max(min_stake, min(max_stake, rounded))


def apply_kelly_to_bets(
    bets: List[Dict[str, Any]],
    kelly_config: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """bets リストの stake_amount をケリー基準で上書きする。disabled/None の場合はそのまま返す。"""
    if not kelly_config or not kelly_config.get("enabled"):
        return bets

    bankroll = int(kelly_config.get("bankroll") or 10_000)
    fraction = float(kelly_config.get("fraction") or 0.25)

    result = []
    for bet in bets:
        bet = dict(bet)
        win_prob = bet.get("_win_prob")
        win_odds = bet.get("_win_odds")
        if win_prob is not None and win_odds is not None:
            bet["stake_amount"] = kelly_stake(
                float(win_prob), float(win_odds), bankroll, fraction
            )
        result.append(bet)
    return result


def load_kelly_config(path: str = _DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """kelly_config.json を読み込む。ファイルが存在しない場合はデフォルトを返す。"""
    if not os.path.exists(path):
        return dict(_DEFAULT_KELLY_CONFIG)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return dict(_DEFAULT_KELLY_CONFIG)


def save_kelly_config(config: Dict[str, Any], path: str = _DEFAULT_CONFIG_PATH) -> None:
    """kelly_config.json に書き込む。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
