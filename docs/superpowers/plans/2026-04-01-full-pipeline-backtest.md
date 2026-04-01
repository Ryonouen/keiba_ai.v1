# フルパイプライン バックテスト 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** LightGBM + value_ai.py のフルパイプラインを 2021〜2024 年の過去レースに適用し、年別・券種別の的中率と回収率を検証する単体スクリプトを作成する。

**Architecture:** 年ごとに LightGBM を前年以前のデータで再訓練（時系列リーク防止）し、各レースに value_ai.py フルパイプラインを適用。見送り判定されたレースはスキップし、払戻を近似計算して集計する。

**Tech Stack:** Python 3.14, pandas, lightgbm, scikit-learn, value_ai.py（既存）

---

### Task 1: CSV ロード・レースグループ化ヘルパー

**Files:**
- Create: `backtest_full_pipeline.py`
- Test: `tests/test_backtest_full_pipeline.py`

- [ ] **Step 1: テストファイルを作成して失敗させる**

```python
# tests/test_backtest_full_pipeline.py
import pandas as pd
import pytest
from backtest_full_pipeline import load_and_group_csv


def test_load_and_group_csv_returns_dict(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text(
        "race_id,race_date,horse_name,feat_win_odds_log,feat_popularity,"
        "feat_running_style_enc,feat_last3f,feat_gate,feat_age,feat_n_runners,"
        "feat_track_condition_enc,feat_signal_total_adjust,"
        "feat_cond_diff_age,feat_cond_diff_gate,feat_cond_diff_style,"
        "feat_cond_diff_popularity,feat_cond_diff_last3f,feat_cond_diff_weight,"
        "feat_cond_diff_jockey,feat_cond_diff_track,feat_recent_form,"
        "feat_trend_index,feat_consistency_index,target_win,target_top3\n"
        "202101010101,2021-01-05,HorseA,1.0,1,0,35.0,3,4,10,0,0.1,"
        "0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5,0.0,1.0,1,1\n"
        "202101010101,2021-01-05,HorseB,2.0,2,1,36.0,5,5,10,0,0.0,"
        "0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.3,0.0,0.8,0,0\n"
    )
    result = load_and_group_csv(str(csv), years=[2021])
    assert 2021 in result
    assert "202101010101" in result[2021]
    assert len(result[2021]["202101010101"]) == 2


def test_load_and_group_csv_filters_year(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text(
        "race_id,race_date,horse_name,feat_win_odds_log,feat_popularity,"
        "feat_running_style_enc,feat_last3f,feat_gate,feat_age,feat_n_runners,"
        "feat_track_condition_enc,feat_signal_total_adjust,"
        "feat_cond_diff_age,feat_cond_diff_gate,feat_cond_diff_style,"
        "feat_cond_diff_popularity,feat_cond_diff_last3f,feat_cond_diff_weight,"
        "feat_cond_diff_jockey,feat_cond_diff_track,feat_recent_form,"
        "feat_trend_index,feat_consistency_index,target_win,target_top3\n"
        "202101010101,2021-01-05,HorseA,1.0,1,0,35.0,3,4,10,0,0.1,"
        "0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5,0.0,1.0,1,1\n"
        "202201010101,2022-01-05,HorseB,2.0,2,1,36.0,5,5,10,0,0.0,"
        "0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.3,0.0,0.8,0,0\n"
    )
    result = load_and_group_csv(str(csv), years=[2021])
    assert 2021 in result
    assert 2022 not in result
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
cd /Users/ryokarahashi/keiba_ai && python3 -m pytest tests/test_backtest_full_pipeline.py -v
```

Expected: `ModuleNotFoundError: No module named 'backtest_full_pipeline'`

- [ ] **Step 3: backtest_full_pipeline.py を作成し load_and_group_csv を実装**

```python
# backtest_full_pipeline.py
"""
フルパイプライン バックテスト
LightGBM + value_ai.py を 2021〜2024 年に適用し、年別・券種別 ROI を検証する。
"""
from __future__ import annotations

import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── 定数 ──────────────────────────────────────────────────────────────
TRAINING_CSV = "keiba_training_data.csv"
TARGET_YEARS = [2021, 2022, 2023, 2024]
BANKROLL = 10_000        # 1レース当たり仮想軍資金（円）
STAKE_UNIT = 100         # 最小掛け金単位

ML_FEATURE_COLUMNS = [
    "feat_gate", "feat_age", "feat_popularity", "feat_win_odds_log",
    "feat_last3f", "feat_jockey_weight", "feat_n_runners",
    "feat_running_style_enc", "feat_track_condition_enc",
    "feat_signal_total_adjust",
    "feat_cond_diff_age", "feat_cond_diff_gate", "feat_cond_diff_style",
    "feat_cond_diff_popularity", "feat_cond_diff_last3f",
    "feat_cond_diff_weight", "feat_cond_diff_jockey", "feat_cond_diff_track",
]

# running_style_enc の逆引き（pace_balance 用）
_ENC_TO_STYLE = {0: "front", 1: "stalker", 2: "closer", 3: "unknown"}

# PLACE_ODDS_FACTORS（value_ai.py と同値）
_PLACE_FACTORS = {1: 1.55, 2: 1.90, 3: 1.90, 4: 2.50, 5: 2.50, 6: 2.50,
                  7: 3.10, 8: 3.10, 9: 3.10}
_PLACE_FACTOR_DEFAULT = 4.00


def load_and_group_csv(
    csv_path: str = TRAINING_CSV,
    years: Optional[List[int]] = None,
) -> Dict[int, Dict[str, List[Dict[str, Any]]]]:
    """
    CSVを読み込み {year: {race_id: [row_dict, ...]}} 形式で返す。

    Args:
        csv_path: CSVファイルパス
        years: 対象年リスト。None の場合は TARGET_YEARS を使用

    Returns:
        {year(int): {race_id(str): [row(dict), ...]}}
    """
    if years is None:
        years = TARGET_YEARS

    df = pd.read_csv(csv_path, low_memory=False)
    df["year"] = pd.to_datetime(df["race_date"], errors="coerce").dt.year
    df = df[df["year"].isin(years)].copy()

    # ML特徴量の欠損を 0 で埋める
    for col in ML_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    result: Dict[int, Dict[str, List[Dict[str, Any]]]] = defaultdict(dict)
    for _, row in df.iterrows():
        year = int(row["year"])
        race_id = str(row["race_id"])
        if race_id not in result[year]:
            result[year][race_id] = []
        result[year][race_id].append(row.to_dict())

    return dict(result)
```

- [ ] **Step 4: テストを実行して通過確認**

```bash
cd /Users/ryokarahashi/keiba_ai && python3 -m pytest tests/test_backtest_full_pipeline.py -v
```

Expected: `2 passed`

- [ ] **Step 5: コミット**

```bash
git add backtest_full_pipeline.py tests/test_backtest_full_pipeline.py
git commit -m "feat: add load_and_group_csv for full pipeline backtest"
```

---

### Task 2: 特徴量 dict ビルダー（CSV行 → value_ai 形式）

**Files:**
- Modify: `backtest_full_pipeline.py`
- Modify: `tests/test_backtest_full_pipeline.py`

- [ ] **Step 1: テストを追加**

```python
# tests/test_backtest_full_pipeline.py に追記
from backtest_full_pipeline import build_feature_dict, build_pace_balance


def test_build_feature_dict_win_odds():
    row = {
        "horse_name": "HorseA",
        "feat_win_odds_log": math.log(5.0),
        "feat_popularity": 2,
        "feat_running_style_enc": 1,
        "feat_last3f": 35.5,
        "feat_gate": 3,
        "feat_age": 4,
        "feat_n_runners": 12,
        "feat_track_condition_enc": 0,
        "feat_signal_total_adjust": 0.1,
        "feat_cond_diff_age": 0.0,
        "feat_cond_diff_gate": 0.0,
        "feat_cond_diff_style": 0.0,
        "feat_cond_diff_popularity": 0.0,
        "feat_cond_diff_last3f": 0.0,
        "feat_cond_diff_weight": 0.0,
        "feat_cond_diff_jockey": 0.0,
        "feat_cond_diff_track": 0.0,
        "feat_recent_form": 0.4,
        "target_win": 0,
        "target_top3": 1,
    }
    f = build_feature_dict(row, win_prob=0.15)
    assert abs(f["win_odds"] - 5.0) < 0.01
    assert f["win_prob"] == 0.15
    assert f["horse_name"] == "HorseA"
    assert f["popularity_rank"] == 2
    assert f["running_style"] == "stalker"


def test_build_pace_balance():
    rows = [
        {"feat_running_style_enc": 0},  # front（逃げ）
        {"feat_running_style_enc": 0},  # front
        {"feat_running_style_enc": 1},  # stalker（先行）
        {"feat_running_style_enc": 2},  # closer（差し）
    ]
    pb = build_pace_balance(rows)
    assert pb["逃げ"] == 2
    assert pb["先行"] == 1
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
cd /Users/ryokarahashi/keiba_ai && python3 -m pytest tests/test_backtest_full_pipeline.py::test_build_feature_dict_win_odds tests/test_backtest_full_pipeline.py::test_build_pace_balance -v
```

Expected: `ImportError`

- [ ] **Step 3: backtest_full_pipeline.py に 2 関数を追加**

```python
# backtest_full_pipeline.py の load_and_group_csv の後に追記


def build_feature_dict(row: Dict[str, Any], win_prob: float) -> Dict[str, Any]:
    """
    CSV の 1行 dict を value_ai.py が期待するキー形式に変換する。

    - win_odds: exp(feat_win_odds_log) で復元
    - running_style: enc 値を文字列に変換
    - jockey_delta / place_odds: 未収録のためデフォルト値
    """
    raw_odds_log = float(row.get("feat_win_odds_log") or 0.0)
    win_odds = math.exp(raw_odds_log) if raw_odds_log != 0.0 else None

    enc = int(float(row.get("feat_running_style_enc") or 3))
    running_style = _ENC_TO_STYLE.get(enc, "unknown")

    pop = int(float(row.get("feat_popularity") or 99))
    place_factor = _PLACE_FACTORS.get(pop, _PLACE_FACTOR_DEFAULT)
    place_odds = round(win_odds * place_factor / 100, 2) if win_odds else None

    return {
        "horse_name":       str(row.get("horse_name") or ""),
        "win_prob":         win_prob,
        "model_score":      win_prob,
        "win_odds":         win_odds,
        "place_odds":       place_odds,
        "popularity_rank":  pop,
        "running_style":    running_style,
        "last3f":           float(row.get("feat_last3f") or 0.0),
        "gate":             int(float(row.get("feat_gate") or 0)),
        "age":              int(float(row.get("feat_age") or 0)),
        "n_runners":        int(float(row.get("feat_n_runners") or 0)),
        "jockey_delta":     0.0,   # CSV未収録
        "jockey_reason_codes": [],
        # 正解ラベル（払戻計算用、value_ai には渡さない）
        "_target_win":      int(float(row.get("target_win") or 0)),
        "_target_top3":     int(float(row.get("target_top3") or 0)),
        "_win_odds_log":    raw_odds_log,
    }


def build_pace_balance(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    レース内の脚質カウントを返す（classify_race_structure の pace_balance 用）。

    Returns: {"逃げ": n, "先行": n, "差し": n, "追込": n}
    """
    pb = {"逃げ": 0, "先行": 0, "差し": 0, "追込": 0}
    style_map = {0: "逃げ", 1: "先行", 2: "差し", 3: "追込"}
    for row in rows:
        enc = int(float(row.get("feat_running_style_enc") or 3))
        key = style_map.get(enc)
        if key:
            pb[key] += 1
    return pb
```

- [ ] **Step 4: テストを実行して通過確認**

```bash
cd /Users/ryokarahashi/keiba_ai && python3 -m pytest tests/test_backtest_full_pipeline.py -v
```

Expected: `4 passed`

- [ ] **Step 5: コミット**

```bash
git add backtest_full_pipeline.py tests/test_backtest_full_pipeline.py
git commit -m "feat: add build_feature_dict and build_pace_balance"
```

---

### Task 3: 年次 LightGBM 訓練ループ

**Files:**
- Modify: `backtest_full_pipeline.py`
- Modify: `tests/test_backtest_full_pipeline.py`

- [ ] **Step 1: テストを追加**

```python
# tests/test_backtest_full_pipeline.py に追記
import math
from backtest_full_pipeline import train_lgbm_for_year


def _make_df(years: List[int]) -> pd.DataFrame:
    import numpy as np
    rows = []
    for y in years:
        for i in range(20):
            rows.append({
                "race_id": f"{y}01010101",
                "race_date": f"{y}-01-05",
                "horse_name": f"H{i}",
                "feat_gate": i % 8 + 1,
                "feat_age": 3,
                "feat_popularity": i + 1,
                "feat_win_odds_log": math.log(i + 1.5),
                "feat_last3f": 35.0 + i * 0.1,
                "feat_jockey_weight": 55.0,
                "feat_n_runners": 20,
                "feat_running_style_enc": i % 3,
                "feat_track_condition_enc": 0,
                "feat_signal_total_adjust": 0.0,
                "feat_cond_diff_age": 0.0,
                "feat_cond_diff_gate": 0.0,
                "feat_cond_diff_style": 0.0,
                "feat_cond_diff_popularity": 0.0,
                "feat_cond_diff_last3f": 0.0,
                "feat_cond_diff_weight": 0.0,
                "feat_cond_diff_jockey": 0.0,
                "feat_cond_diff_track": 0.0,
                "target_win": 1 if i == 0 else 0,
                "target_top3": 1 if i < 3 else 0,
                "year": y,
            })
    return pd.DataFrame(rows)


def test_train_lgbm_for_year_returns_model():
    df = _make_df([2019, 2020, 2021])
    model = train_lgbm_for_year(df, test_year=2021)
    assert model is not None
    # 予測できること
    import numpy as np
    X = df[df["year"] < 2021][ML_FEATURE_COLUMNS].fillna(0)
    preds = model.predict(X.values)
    assert len(preds) == len(X)
    assert all(0 <= p <= 1 for p in preds)


# ファイル先頭の import に追加が必要
# from backtest_full_pipeline import ML_FEATURE_COLUMNS
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
cd /Users/ryokarahashi/keiba_ai && python3 -m pytest tests/test_backtest_full_pipeline.py::test_train_lgbm_for_year_returns_model -v
```

Expected: `ImportError`

- [ ] **Step 3: train_lgbm_for_year を実装**

```python
# backtest_full_pipeline.py に追記


def train_lgbm_for_year(df: pd.DataFrame, test_year: int):
    """
    test_year より前のデータで LightGBM を訓練して返す。

    Args:
        df: 全年データ（'year' 列を含む）
        test_year: テスト対象年（この年より前のみ訓練に使用）

    Returns:
        訓練済み lightgbm.Booster
    """
    import lightgbm as lgb

    train_df = df[df["year"] < test_year].copy()
    if len(train_df) == 0:
        raise ValueError(f"訓練データが0件: test_year={test_year}")

    for col in ML_FEATURE_COLUMNS:
        if col not in train_df.columns:
            train_df[col] = 0.0

    X = train_df[ML_FEATURE_COLUMNS].fillna(0.0)
    y = train_df["target_win"].fillna(0).astype(int)

    params = {
        "objective":        "binary",
        "metric":           "binary_logloss",
        "verbosity":        -1,
        "learning_rate":    0.03,
        "num_leaves":       31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "seed":             42,
    }
    dataset = lgb.Dataset(X, label=y)
    model = lgb.train(params, dataset, num_boost_round=200)
    return model
```

- [ ] **Step 4: テストを実行して通過確認**

```bash
cd /Users/ryokarahashi/keiba_ai && python3 -m pytest tests/test_backtest_full_pipeline.py -v
```

Expected: `5 passed`

- [ ] **Step 5: コミット**

```bash
git add backtest_full_pipeline.py tests/test_backtest_full_pipeline.py
git commit -m "feat: add year-by-year LightGBM training function"
```

---

### Task 4: value_ai パイプライン実行関数

**Files:**
- Modify: `backtest_full_pipeline.py`
- Modify: `tests/test_backtest_full_pipeline.py`

- [ ] **Step 1: テストを追加**

```python
# tests/test_backtest_full_pipeline.py に追記
from backtest_full_pipeline import run_value_ai_pipeline


def _make_features(n: int = 8) -> List[Dict[str, Any]]:
    import math
    feats = []
    for i in range(n):
        odds = 2.0 + i * 3.0
        feats.append({
            "horse_name": f"H{i}",
            "win_prob": max(0.01, 0.4 - i * 0.04),
            "model_score": max(0.01, 0.4 - i * 0.04),
            "win_odds": odds,
            "place_odds": round(odds * 1.9 / 100, 2),
            "popularity_rank": i + 1,
            "running_style": ["front", "stalker", "closer"][i % 3],
            "last3f": 35.0,
            "gate": i + 1,
            "age": 4,
            "n_runners": n,
            "jockey_delta": 0.0,
            "jockey_reason_codes": [],
            "_target_win": 1 if i == 0 else 0,
            "_target_top3": 1 if i < 3 else 0,
            "_win_odds_log": math.log(odds),
        })
    return feats


def test_run_value_ai_pipeline_returns_plan():
    features = _make_features(8)
    pace_balance = {"逃げ": 2, "先行": 3, "差し": 2, "追込": 1}
    plan = run_value_ai_pipeline(features, pace_balance, bankroll=10000)
    # skip または有効なプランが返る
    assert "skip" in plan
    assert "bet_type" in plan


def test_run_value_ai_pipeline_skip_when_empty():
    plan = run_value_ai_pipeline([], {}, bankroll=10000)
    assert plan["skip"] is True
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
cd /Users/ryokarahashi/keiba_ai && python3 -m pytest tests/test_backtest_full_pipeline.py::test_run_value_ai_pipeline_returns_plan tests/test_backtest_full_pipeline.py::test_run_value_ai_pipeline_skip_when_empty -v
```

Expected: `ImportError`

- [ ] **Step 3: run_value_ai_pipeline を実装**

```python
# backtest_full_pipeline.py に追記


def run_value_ai_pipeline(
    features: List[Dict[str, Any]],
    pace_balance: Dict[str, int],
    bankroll: int = BANKROLL,
) -> Dict[str, Any]:
    """
    value_ai.py フルパイプラインを実行して推奨買い目を返す。

    Args:
        features: build_feature_dict で作成した馬リスト
        pace_balance: build_pace_balance で作成した脚質カウント
        bankroll: 仮想軍資金（円）

    Returns:
        recommend_bet_plan の戻り値 dict。skip=True なら見送り。
    """
    from value_ai import (
        build_ev_table,
        classify_race_structure,
        recommend_bet_plan,
    )

    EMPTY = {
        "bet_type": "-", "horses": [], "tickets": [],
        "total_stake": 0, "ticket_count": 0,
        "reason": "", "risk_level": "-", "ev_type": "-",
        "skip": True, "skip_reason": "データ不足",
    }

    if not features:
        return EMPTY

    ev_table = build_ev_table(features)
    if not ev_table:
        return EMPTY

    race_structure = classify_race_structure(features, pace_balance)
    plan = recommend_bet_plan(
        features=features,
        ev_table=ev_table,
        race_structure=race_structure,
        bankroll=bankroll,
        race_pace="medium",  # CSVに展開情報なし → 固定
    )
    return plan
```

- [ ] **Step 4: テストを実行して通過確認**

```bash
cd /Users/ryokarahashi/keiba_ai && python3 -m pytest tests/test_backtest_full_pipeline.py -v
```

Expected: `7 passed`

- [ ] **Step 5: コミット**

```bash
git add backtest_full_pipeline.py tests/test_backtest_full_pipeline.py
git commit -m "feat: add run_value_ai_pipeline for backtest"
```

---

### Task 5: 払戻シミュレーター

**Files:**
- Modify: `backtest_full_pipeline.py`
- Modify: `tests/test_backtest_full_pipeline.py`

- [ ] **Step 1: テストを追加**

```python
# tests/test_backtest_full_pipeline.py に追記
from backtest_full_pipeline import simulate_payout


def test_simulate_payout_tansho_hit():
    features = _make_features(8)  # H0 が勝ち馬（_target_win=1）
    plan = {
        "bet_type": "単勝",
        "horses": ["H0"],
        "total_stake": 100,
        "skip": False,
    }
    result = simulate_payout(plan, features)
    import math
    expected_odds = math.exp(math.log(2.0))  # H0 の win_odds
    assert result["hit"] is True
    assert abs(result["payout"] - expected_odds * 100) < 1.0


def test_simulate_payout_tansho_miss():
    features = _make_features(8)  # H0 が勝ち馬
    plan = {
        "bet_type": "単勝",
        "horses": ["H1"],  # 外れ
        "total_stake": 100,
        "skip": False,
    }
    result = simulate_payout(plan, features)
    assert result["hit"] is False
    assert result["payout"] == 0


def test_simulate_payout_skip():
    features = _make_features(8)
    plan = {"skip": True, "total_stake": 0}
    result = simulate_payout(plan, features)
    assert result is None
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
cd /Users/ryokarahashi/keiba_ai && python3 -m pytest tests/test_backtest_full_pipeline.py::test_simulate_payout_tansho_hit tests/test_backtest_full_pipeline.py::test_simulate_payout_tansho_miss tests/test_backtest_full_pipeline.py::test_simulate_payout_skip -v
```

Expected: `ImportError`

- [ ] **Step 3: simulate_payout を実装**

```python
# backtest_full_pipeline.py に追記


def simulate_payout(
    plan: Dict[str, Any],
    features: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    推奨買い目と実際の結果から払戻を計算する。

    Args:
        plan: run_value_ai_pipeline の戻り値
        features: build_feature_dict で作成した馬リスト

    Returns:
        {"hit": bool, "payout": float, "invest": int} または
        None（見送りの場合）
    """
    if plan.get("skip"):
        return None

    bet_type = plan.get("bet_type", "-")
    horses = plan.get("horses", [])
    invest = int(plan.get("total_stake") or STAKE_UNIT)

    # 名前 → feature dict
    fmap = {f["horse_name"]: f for f in features}

    # ── 単勝 ──────────────────────────────────────────────────────────
    if bet_type == "単勝" and len(horses) >= 1:
        h = fmap.get(horses[0])
        if h is None:
            return {"hit": False, "payout": 0, "invest": invest}
        if h["_target_win"] == 1:
            payout = math.exp(h["_win_odds_log"]) * invest
            return {"hit": True, "payout": payout, "invest": invest}
        return {"hit": False, "payout": 0, "invest": invest}

    # ── 複勝 ──────────────────────────────────────────────────────────
    if bet_type == "複勝" and len(horses) >= 1:
        h = fmap.get(horses[0])
        if h is None:
            return {"hit": False, "payout": 0, "invest": invest}
        if h["_target_top3"] == 1:
            pop = h.get("popularity_rank", 99)
            factor = _PLACE_FACTORS.get(pop, _PLACE_FACTOR_DEFAULT)
            win_odds = math.exp(h["_win_odds_log"])
            payout = win_odds * factor * invest / 100
            return {"hit": True, "payout": payout, "invest": invest}
        return {"hit": False, "payout": 0, "invest": invest}

    # ── 馬連 ──────────────────────────────────────────────────────────
    if bet_type == "馬連" and len(horses) >= 2:
        ha, hb = fmap.get(horses[0]), fmap.get(horses[1])
        if ha is None or hb is None:
            return {"hit": False, "payout": 0, "invest": invest}
        # 的中: 両馬が top3 かつ 1着・2着（上位2頭）
        top2 = sorted(
            [f for f in features if f["_target_top3"] == 1],
            key=lambda x: x.get("popularity_rank", 99)
        )[:2]
        top2_names = {f["horse_name"] for f in top2}
        if horses[0] in top2_names and horses[1] in top2_names:
            oa = math.exp(ha["_win_odds_log"])
            ob = math.exp(hb["_win_odds_log"])
            payout = oa * ob * 0.75 * invest / 10
            return {"hit": True, "payout": payout, "invest": invest}
        return {"hit": False, "payout": 0, "invest": invest}

    # ── ワイド ─────────────────────────────────────────────────────────
    if bet_type == "ワイド" and len(horses) >= 2:
        ha, hb = fmap.get(horses[0]), fmap.get(horses[1])
        if ha is None or hb is None:
            return {"hit": False, "payout": 0, "invest": invest}
        if ha["_target_top3"] == 1 and hb["_target_top3"] == 1:
            oa = math.exp(ha["_win_odds_log"])
            ob = math.exp(hb["_win_odds_log"])
            payout = oa * ob * 0.25 * invest / 10
            return {"hit": True, "payout": payout, "invest": invest}
        return {"hit": False, "payout": 0, "invest": invest}

    # ── 3連複 ──────────────────────────────────────────────────────────
    if bet_type in ("3連複", "三連複") and len(horses) >= 3:
        hs = [fmap.get(h) for h in horses[:3]]
        if any(h is None for h in hs):
            return {"hit": False, "payout": 0, "invest": invest}
        if all(h["_target_top3"] == 1 for h in hs):
            odds_product = math.prod(math.exp(h["_win_odds_log"]) for h in hs)
            payout = odds_product * 0.70 * invest / 10
            return {"hit": True, "payout": payout, "invest": invest}
        return {"hit": False, "payout": 0, "invest": invest}

    # ── 対応外 ────────────────────────────────────────────────────────
    return {"hit": False, "payout": 0, "invest": invest}
```

- [ ] **Step 4: テストを実行して通過確認**

```bash
cd /Users/ryokarahashi/keiba_ai && python3 -m pytest tests/test_backtest_full_pipeline.py -v
```

Expected: `10 passed`

- [ ] **Step 5: コミット**

```bash
git add backtest_full_pipeline.py tests/test_backtest_full_pipeline.py
git commit -m "feat: add payout simulator for full pipeline backtest"
```

---

### Task 6: メインバックテストループと集計

**Files:**
- Modify: `backtest_full_pipeline.py`

- [ ] **Step 1: run_backtest 関数を実装**

```python
# backtest_full_pipeline.py に追記


def run_backtest(
    csv_path: str = TRAINING_CSV,
    target_years: List[int] = TARGET_YEARS,
    bankroll: int = BANKROLL,
) -> Dict[str, Any]:
    """
    フルパイプラインバックテストを実行する。

    Returns:
        {
            "total_races": int,        # 対象レース総数
            "recommended": int,        # 推奨あり（見送りでない）レース数
            "hits": int,               # 的中数
            "total_invest": float,     # 総投資額
            "total_payout": float,     # 総払戻額
            "roi": float,              # 総ROI（1.0 = トントン）
            "by_year": {year: {...}},  # 年別集計
            "by_bet_type": {...},      # 券種別集計
            "records": [...],          # レース別詳細
        }
    """
    print(f"CSVを読み込み中: {csv_path}")
    df_full = pd.read_csv(csv_path, low_memory=False)
    df_full["year"] = pd.to_datetime(df_full["race_date"], errors="coerce").dt.year

    for col in ML_FEATURE_COLUMNS:
        if col not in df_full.columns:
            df_full[col] = 0.0
        else:
            df_full[col] = pd.to_numeric(df_full[col], errors="coerce").fillna(0.0)
    if "feat_jockey_weight" not in df_full.columns:
        df_full["feat_jockey_weight"] = 55.0
    else:
        df_full["feat_jockey_weight"] = df_full["feat_jockey_weight"].fillna(55.0)

    print(f"総行数: {len(df_full):,}  対象年: {target_years}\n")

    records = []
    by_year: Dict[int, Dict] = {}
    by_bet_type: Dict[str, Dict] = defaultdict(lambda: {"races": 0, "hits": 0, "invest": 0.0, "payout": 0.0})

    for test_year in target_years:
        print(f"[{test_year}] モデル訓練中（{test_year - 1}年以前）...", end=" ", flush=True)
        model = train_lgbm_for_year(df_full, test_year=test_year)
        print("完了")

        year_df = df_full[df_full["year"] == test_year].copy()
        race_groups = year_df.groupby("race_id")
        race_ids = list(race_groups.groups.keys())
        print(f"[{test_year}] レース数: {len(race_ids):,}")

        y_total = y_recommended = y_hits = 0
        y_invest = y_payout = 0.0

        for idx, race_id in enumerate(race_ids):
            group = race_groups.get_group(race_id)
            rows = group.to_dict(orient="records")

            # LightGBM 予測
            X = group[ML_FEATURE_COLUMNS].fillna(0.0)
            probs = model.predict(X.values)

            # 特徴量 dict 組み立て
            features = [
                build_feature_dict(row, win_prob=float(prob))
                for row, prob in zip(rows, probs)
            ]
            pace_balance = build_pace_balance(rows)

            # value_ai パイプライン
            plan = run_value_ai_pipeline(features, pace_balance, bankroll=bankroll)

            y_total += 1

            # 払戻シミュレーション
            result = simulate_payout(plan, features)
            if result is None:
                continue  # 見送り

            y_recommended += 1
            hit = result["hit"]
            payout = result["payout"]
            invest = result["invest"]

            if hit:
                y_hits += 1
            y_invest += invest
            y_payout += payout

            bet_type = plan.get("bet_type", "-")
            by_bet_type[bet_type]["races"] += 1
            by_bet_type[bet_type]["hits"] += int(hit)
            by_bet_type[bet_type]["invest"] += invest
            by_bet_type[bet_type]["payout"] += payout

            records.append({
                "race_id": race_id,
                "year": test_year,
                "bet_type": bet_type,
                "horses": ",".join(plan.get("horses", [])),
                "hit": hit,
                "invest": invest,
                "payout": round(payout, 0),
            })

            if (idx + 1) % 200 == 0:
                roi_so_far = y_payout / y_invest if y_invest > 0 else 0.0
                print(f"  [{test_year}] {idx+1}/{len(race_ids)} 推奨率:{y_recommended/(idx+1)*100:.0f}% ROI:{roi_so_far*100:.1f}%", flush=True)

        y_roi = y_payout / y_invest if y_invest > 0 else 0.0
        by_year[test_year] = {
            "total_races": y_total,
            "recommended": y_recommended,
            "hits": y_hits,
            "invest": y_invest,
            "payout": y_payout,
            "roi": y_roi,
        }
        print(f"[{test_year}] 完了 推奨:{y_recommended}/{y_total} 的中:{y_hits} ROI:{y_roi*100:.1f}%\n")

    total_invest = sum(v["invest"] for v in by_year.values())
    total_payout = sum(v["payout"] for v in by_year.values())
    total_hits   = sum(v["hits"]   for v in by_year.values())
    total_rec    = sum(v["recommended"] for v in by_year.values())
    total_races  = sum(v["total_races"] for v in by_year.values())

    return {
        "total_races":   total_races,
        "recommended":   total_rec,
        "hits":          total_hits,
        "total_invest":  total_invest,
        "total_payout":  total_payout,
        "roi":           total_payout / total_invest if total_invest > 0 else 0.0,
        "by_year":       by_year,
        "by_bet_type":   dict(by_bet_type),
        "records":       records,
    }
```

- [ ] **Step 2: 結果出力・CSV保存関数を追加**

```python
# backtest_full_pipeline.py に追記


def print_summary(result: Dict[str, Any]) -> None:
    """バックテスト結果をコンソールに整形して表示する。"""
    total_races  = result["total_races"]
    recommended  = result["recommended"]
    hits         = result["hits"]
    roi          = result["roi"]
    by_year      = result["by_year"]
    by_bet_type  = result["by_bet_type"]

    hit_rate = hits / recommended if recommended > 0 else 0.0
    sel_rate = recommended / total_races if total_races > 0 else 0.0

    print("\n" + "=" * 60)
    print("  フルパイプライン バックテスト結果")
    print("=" * 60)
    print(f"\n【全体サマリー】")
    print(f"  対象レース  : {total_races:,} 件")
    print(f"  推奨あり    : {recommended:,} 件 ({sel_rate*100:.1f}%)")
    print(f"  的中        : {hits:,} 件 ({hit_rate*100:.1f}%)")
    print(f"  総ROI       : {roi*100:.1f}%  (100%=トントン)")

    print(f"\n【年別ROI】")
    for year, v in sorted(by_year.items()):
        y_roi = v["roi"] * 100
        y_sel = v["recommended"] / v["total_races"] * 100 if v["total_races"] > 0 else 0
        print(f"  {year}: ROI {y_roi:6.1f}%  推奨率 {y_sel:.0f}%  的中 {v['hits']}/{v['recommended']}")

    print(f"\n【券種別】")
    for bet_type, v in sorted(by_bet_type.items(), key=lambda x: -x[1]["invest"]):
        if v["races"] == 0:
            continue
        b_hit  = v["hits"] / v["races"] * 100
        b_roi  = v["payout"] / v["invest"] * 100 if v["invest"] > 0 else 0.0
        note   = "  ※近似" if bet_type not in ("単勝", "複勝") else ""
        print(f"  {bet_type:6s}: 的中率 {b_hit:5.1f}%  ROI {b_roi:6.1f}%  ({v['races']}件){note}")

    print("\n" + "=" * 60)


def save_records_csv(result: Dict[str, Any], out_path: str = "backtest_full_pipeline_result.csv") -> None:
    """レース別詳細を CSV に保存する。"""
    if not result["records"]:
        print("出力レコードなし")
        return
    df = pd.DataFrame(result["records"])
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n詳細CSV保存: {out_path} ({len(df)} 件)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="フルパイプライン バックテスト")
    parser.add_argument("--csv", default=TRAINING_CSV, help="学習データCSVパス")
    parser.add_argument("--years", nargs="+", type=int, default=TARGET_YEARS,
                        help="対象年（例: 2021 2022 2023 2024）")
    parser.add_argument("--bankroll", type=int, default=BANKROLL,
                        help="1レース当たり仮想軍資金（円）")
    parser.add_argument("--save-csv", action="store_true",
                        help="レース別詳細をCSVに保存する")
    args = parser.parse_args()

    result = run_backtest(csv_path=args.csv, target_years=args.years, bankroll=args.bankroll)
    print_summary(result)
    if args.save_csv:
        save_records_csv(result)
```

- [ ] **Step 3: 構文エラーがないか確認**

```bash
cd /Users/ryokarahashi/keiba_ai && python3 -c "import backtest_full_pipeline" && echo "OK"
```

Expected: `OK`

- [ ] **Step 4: 全テストを実行**

```bash
cd /Users/ryokarahashi/keiba_ai && python3 -m pytest tests/test_backtest_full_pipeline.py -v
```

Expected: `10 passed`

- [ ] **Step 5: コミット**

```bash
git add backtest_full_pipeline.py
git commit -m "feat: add main backtest loop, summary printer, and CSV export"
```

---

### Task 7: エンドツーエンド実行

**Files:**
- 読み取り専用（既存ファイルを変更しない）

- [ ] **Step 1: まず1年分だけ試走（速度確認）**

```bash
cd /Users/ryokarahashi/keiba_ai && python3 backtest_full_pipeline.py --years 2024 2>&1 | head -50
```

Expected: モデル訓練メッセージ → レース処理ログ → 年別結果

- [ ] **Step 2: 問題がなければフル4年実行（バックグラウンド）**

```bash
cd /Users/ryokarahashi/keiba_ai && python3 backtest_full_pipeline.py --years 2021 2022 2023 2024 --save-csv >> backtest_full.log 2>&1 &
echo "PID: $!"
```

- [ ] **Step 3: ログで進捗確認**

```bash
tail -f /Users/ryokarahashi/keiba_ai/backtest_full.log
```

- [ ] **Step 4: 完了後に結果を確認**

```bash
tail -40 /Users/ryokarahashi/keiba_ai/backtest_full.log
```

- [ ] **Step 5: 最終コミット**

```bash
cd /Users/ryokarahashi/keiba_ai && git add backtest_full_pipeline.py tests/test_backtest_full_pipeline.py
git commit -m "feat: complete full pipeline backtest (LightGBM + value_ai)" 2>/dev/null || echo "変更なし"
```
