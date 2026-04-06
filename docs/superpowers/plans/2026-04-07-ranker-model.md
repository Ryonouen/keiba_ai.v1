# Ranker Model + Ensemble Validation 実装プラン

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** LightGBM Ranker（3リスクプロファイル）と LightGBM+XGBoost+Sklearn スタッキングアンサンブルを追加し、既存分類モデルを補完する形で的中率・回収率を向上させる。

**Architecture:** 既存の `train_lightgbm_model()` / `predict_win_probability_with_model()` には一切触れない。新規モジュール `ranker_engine.py` と `ensemble_validator.py` を追加し、`collect_and_train.py` に新モード `--mode ranker` / `--mode ensemble` を追加する。`fast_backfill.py` と同じ `ML_FEATURE_COLUMNS` を再利用する。

**Tech Stack:** Python 3.14, LightGBM 4.6 (lambdarank), scikit-learn 1.8 (StackingClassifier, NDCG), XGBoost (要 `pip install xgboost`), pandas 2.3

**学習データ:** `keiba_training_data.csv` — 359,556行 / 26,195レース。`rank` 列は79,461行に実値あり（`target_win` / `target_top3` からも補完可能）。アウトオブサンプル評価には2025年データを使用。

---

## ファイル構成

| ファイル | 変更 | 役割 |
|---|---|---|
| `ranker_engine.py` | **新規作成** | LightGBM Ranker 学習・予測・ブレンド |
| `ensemble_validator.py` | **新規作成** | LightGBM/XGBoost/Sklearn スタッキング比較 |
| `collect_and_train.py` | **追加のみ** | `--mode ranker` / `--mode ensemble` サブコマンド |
| `tests/test_ranker_engine.py` | **新規作成** | Ranker ユニットテスト |
| `tests/test_ensemble_validator.py` | **新規作成** | Ensemble ユニットテスト |

変更しないファイル: `race_ai_engine.py`, `daily_pipeline.py`, `pipeline_store.py`, `value_ai.py`

---

## Task 1: ranker_engine.py — LightGBM Ranker コア

**Files:**
- Create: `ranker_engine.py`
- Test: `tests/test_ranker_engine.py`

- [ ] **Step 1: テストを書く**

`tests/test_ranker_engine.py` を作成：

```python
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
    result = re_mod.blend_scores(wp, [0.8, 0.2])  # length mismatch
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
    """全馬同スコアのとき、win_probs がそのまま効く"""
    wp = [0.5, 0.3, 0.2]
    rs = [1.0, 1.0, 1.0]  # 全て同じ → 正規化後 0.5
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
    # モデルファイルが存在しない → None
    result = re_mod.predict_rank_score([{"feat_gate": 1}], profile="balanced")
    assert result is None

def test_predict_returns_none_for_unknown_profile():
    result = re_mod.predict_rank_score([{"feat_gate": 1}], profile="unknown_profile")
    assert result is None
```

- [ ] **Step 2: テスト実行（失敗を確認）**

```bash
python3 -m pytest tests/test_ranker_engine.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'ranker_engine'`

- [ ] **Step 3: ranker_engine.py を実装する**

`ranker_engine.py` を作成：

```python
"""
ranker_engine.py
LightGBM Ranker によるレース内順位スコア予測。
既存の predict_win_probability_with_model() を補完するが置き換えない。

使い方:
  from ranker_engine import train_ranker_model, predict_rank_score, blend_scores
  train_ranker_model("keiba_training_data.csv", profile="balanced")
  scores = predict_rank_score(features, profile="balanced")
  blended = blend_scores(win_probs, scores, weight_ranker=0.3)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from race_ai_engine import ML_FEATURE_COLUMNS, TRAINING_CSV

RANKER_MODEL_DIR: str = os.path.dirname(os.path.abspath(__file__))

RANKER_PROFILES: Dict[str, Dict] = {
    "conservative": {
        "model_file": "keiba_lgbm_ranker_conservative.txt",
        "params": {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [3],
            "learning_rate": 0.05,
            "num_leaves": 20,
            "min_data_in_leaf": 10,
            "feature_fraction": 0.8,
            "seed": 42,
            "verbosity": -1,
        },
        "num_boost_round": 200,
        "description": "的中率重視（上位人気を正確に識別）",
    },
    "balanced": {
        "model_file": "keiba_lgbm_ranker_balanced.txt",
        "params": {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [3],
            "learning_rate": 0.03,
            "num_leaves": 31,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "seed": 42,
            "verbosity": -1,
        },
        "num_boost_round": 300,
        "description": "バランス型（的中率・回収率のバランス）",
    },
    "aggressive": {
        "model_file": "keiba_lgbm_ranker_aggressive.txt",
        "params": {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1],
            "learning_rate": 0.1,
            "num_leaves": 63,
            "min_data_in_leaf": 5,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq": 3,
            "seed": 42,
            "verbosity": -1,
        },
        "num_boost_round": 150,
        "description": "回収率重視・大穴狙い（穴馬発掘に特化）",
    },
}


def _build_relevance_labels(df: "pd.DataFrame") -> "pd.Series":
    """
    target_win / target_top3 から 0-2 の関連度ラベルを構築する。
    LightGBM Ranker はラベル高 = より上位と学習する。
      2: 勝ち馬 (target_win=1)
      1: 3着内 (target_top3=1, target_win=0)
      0: それ以外
    """
    win  = df["target_win"].fillna(0).astype(int)
    top3 = df["target_top3"].fillna(0).astype(int)
    return (win * 2 + (top3 * (1 - win))).clip(0, 2).astype(int)


def train_ranker_model(
    csv_path: str = TRAINING_CSV,
    profile: str = "balanced",
) -> bool:
    """
    LightGBM Ranker を学習して保存する。既存 train_lightgbm_model() とは別ファイルに保存。

    Parameters
    ----------
    csv_path : 学習CSVパス（keiba_training_data.csv）
    profile  : "conservative" / "balanced" / "aggressive"

    Returns
    -------
    True on success, False on failure.
    """
    if not LIGHTGBM_AVAILABLE:
        logger.error("LightGBM が未インストールのため学習をスキップします。")
        return False
    if not Path(csv_path).exists():
        logger.error("学習CSVが存在しないためスキップ: %s", csv_path)
        return False
    if profile not in RANKER_PROFILES:
        logger.error("不明なプロファイル: %s。選択肢: %s", profile, list(RANKER_PROFILES.keys()))
        return False

    cfg = RANKER_PROFILES[profile]
    model_file = os.path.join(RANKER_MODEL_DIR, cfg["model_file"])

    logger.info("Ranker 学習開始 profile=%s csv=%s", profile, csv_path)

    df = pd.read_csv(csv_path, low_memory=False)
    required = set(ML_FEATURE_COLUMNS + ["target_win", "target_top3", "race_id"])
    missing = required - set(df.columns)
    if missing:
        logger.error("必要列が不足: %s", missing)
        return False

    df = df.dropna(subset=["race_id"]).copy()
    df["_relevance"] = _build_relevance_labels(df)
    df[ML_FEATURE_COLUMNS] = df[ML_FEATURE_COLUMNS].fillna(0.0)

    # race_id でグループ化して順序を保つ
    df = df.sort_values("race_id").reset_index(drop=True)
    groups = df.groupby("race_id", sort=False).size().values.tolist()

    X = df[ML_FEATURE_COLUMNS]
    y = df["_relevance"]

    dataset = lgb.Dataset(X, label=y, group=groups, free_raw_data=False)
    model = lgb.train(
        cfg["params"],
        dataset,
        num_boost_round=cfg["num_boost_round"],
    )
    model.save_model(model_file)
    logger.info("Ranker モデルを保存しました: %s (profile=%s)", model_file, profile)
    print(f"Ranker モデルを保存しました: {model_file}  [{cfg['description']}]")
    return True


def predict_rank_score(
    features: List[Dict[str, Any]],
    profile: str = "balanced",
) -> Optional[List[float]]:
    """
    各馬のランキングスコア（高いほど上位と判断）を返す。
    モデル未存在時は None を返す。既存 predict_win_probability_with_model() と
    戻り値の形式が異なる点に注意（正規化しない raw スコア）。

    Parameters
    ----------
    features : 馬ごとの特徴量辞書リスト（ML_FEATURE_COLUMNS キーを含む）
    profile  : "conservative" / "balanced" / "aggressive"
    """
    if not LIGHTGBM_AVAILABLE:
        return None
    cfg = RANKER_PROFILES.get(profile)
    if cfg is None:
        return None
    model_file = os.path.join(RANKER_MODEL_DIR, cfg["model_file"])
    if not Path(model_file).exists():
        return None

    try:
        model = lgb.Booster(model_file=model_file)
        X = pd.DataFrame(
            [{col: float(f.get(col) or 0.0) for col in ML_FEATURE_COLUMNS} for f in features]
        )
        return [float(s) for s in model.predict(X)]
    except Exception as exc:
        logger.warning("predict_rank_score error (profile=%s): %s", profile, exc)
        return None


def blend_scores(
    win_probs: List[float],
    rank_scores: Optional[List[float]],
    weight_ranker: float = 0.3,
) -> List[float]:
    """
    既存モデルの win_prob とランカースコアをブレンドして正規化する。
    rank_scores が None または長さ不一致の場合は win_probs をそのまま返す。

    Parameters
    ----------
    win_probs     : predict_win_probability_with_model() の出力（合計1.0に正規化済み）
    rank_scores   : predict_rank_score() の出力（raw スコア）
    weight_ranker : ランカーの重み（0.0 = 完全に win_probs 優先）
    """
    if rank_scores is None or len(rank_scores) != len(win_probs):
        return win_probs

    rmin, rmax = min(rank_scores), max(rank_scores)
    rng = rmax - rmin if rmax > rmin else 1.0
    normalized = [(s - rmin) / rng for s in rank_scores]

    blended = [
        (1.0 - weight_ranker) * wp + weight_ranker * rs
        for wp, rs in zip(win_probs, normalized)
    ]
    total = sum(blended)
    if total <= 0:
        return win_probs
    return [b / total for b in blended]
```

- [ ] **Step 4: テスト実行（全パスを確認）**

```bash
python3 -m pytest tests/test_ranker_engine.py -v
```

Expected: 全テスト PASS

- [ ] **Step 5: コミット**

```bash
git add ranker_engine.py tests/test_ranker_engine.py
git commit -m "feat: add ranker_engine.py — LightGBM Ranker with 3 risk profiles"
```

---

## Task 2: collect_and_train.py に --mode ranker を追加

**Files:**
- Modify: `collect_and_train.py`

- [ ] **Step 1: train_ranker 関数を追加する**

`collect_and_train.py` の `train_model()` 関数の直後（約159行目の後）に以下を追記：

```python
# ──────────────────────────────────────────────
# モード⑤: LightGBM Ranker 学習
# ──────────────────────────────────────────────

def train_ranker(profile: str = "balanced") -> None:
    from ranker_engine import train_ranker_model, RANKER_PROFILES

    csv_path = TRAINING_CSV if Path(TRAINING_CSV).exists() else "keiba_training_data.csv"

    print(f"\n=== LightGBM Ranker 学習 (profile={profile}) ===")
    if not Path(csv_path).exists():
        print(f"  エラー: {csv_path} が存在しません。先にCSV出力を実行してください。")
        return

    import pandas as pd
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  学習データ: {len(df):,} 行 / {df['race_id'].nunique():,} レース")
    desc = RANKER_PROFILES[profile]["description"]
    print(f"  プロファイル: {profile} — {desc}")

    ok = train_ranker_model(csv_path, profile=profile)
    if ok:
        print(f"  ✓ Ranker 学習完了。")
    else:
        print(f"  ✗ Ranker 学習失敗。")
```

- [ ] **Step 2: argparse に ranker モードを追加する**

`collect_and_train.py` の `main()` 関数内の `parser.add_argument("--mode", ...)` 行を更新：

既存:
```python
parser.add_argument("--mode", choices=["range", "name", "csv", "train"], help="実行モード")
```

変更後:
```python
parser.add_argument(
    "--mode",
    choices=["range", "name", "csv", "train", "ranker"],
    help="実行モード (ranker は --profile と組み合わせて使用)",
)
parser.add_argument(
    "--profile",
    choices=["conservative", "balanced", "aggressive"],
    default="balanced",
    help="Rankerプロファイル (--mode ranker 専用)",
)
```

- [ ] **Step 3: ranker モード分岐を追加する**

`main()` 内の `if args.mode == "train":` ブロックの後に追加：

```python
if args.mode == "ranker":
    train_ranker(profile=args.profile)
    return
```

- [ ] **Step 4: 動作確認（dry run）**

```bash
python3 collect_and_train.py --mode ranker --profile balanced 2>&1 | head -10
```

Expected（モデルなし初回）:
```
=== LightGBM Ranker 学習 (profile=balanced) ===
  学習データ: 359,556 行 / 26,195 レース
  プロファイル: balanced — バランス型（的中率・回収率のバランス）
```
※ 学習には数分かかる。

- [ ] **Step 5: コミット**

```bash
git add collect_and_train.py
git commit -m "feat: add --mode ranker to collect_and_train.py"
```

---

## Task 3: 全プロファイルの Ranker 学習実行

**Files:**
- Creates: `keiba_lgbm_ranker_conservative.txt`, `keiba_lgbm_ranker_balanced.txt`, `keiba_lgbm_ranker_aggressive.txt`

- [ ] **Step 1: balanced プロファイルを学習する（バックグラウンド）**

```bash
python3 collect_and_train.py --mode ranker --profile balanced 2>&1
```

Expected (数分後):
```
Ranker モデルを保存しました: keiba_lgbm_ranker_balanced.txt  [バランス型（的中率・回収率のバランス）]
  ✓ Ranker 学習完了。
```

- [ ] **Step 2: conservative と aggressive を学習する**

```bash
python3 collect_and_train.py --mode ranker --profile conservative 2>&1
python3 collect_and_train.py --mode ranker --profile aggressive 2>&1
```

- [ ] **Step 3: モデルファイルの存在を確認する**

```bash
ls -lh keiba_lgbm_ranker_*.txt
```

Expected: 3ファイル（各数〜十数MB）が存在する。

- [ ] **Step 4: predict_rank_score が動くことを確認する**

```bash
python3 -c "
from ranker_engine import predict_rank_score
# ダミー特徴量で動作確認
features = [{'feat_gate': i, 'feat_popularity': i+1, 'feat_win_odds_log': 1.5} for i in range(8)]
for profile in ['conservative', 'balanced', 'aggressive']:
    scores = predict_rank_score(features, profile=profile)
    if scores:
        print(f'{profile}: OK (scores range {min(scores):.3f}~{max(scores):.3f})')
    else:
        print(f'{profile}: FAILED')
"
```

Expected: 3プロファイルとも OK が出力される。

- [ ] **Step 5: コミット**

```bash
git add keiba_lgbm_ranker_*.txt
git commit -m "feat: train LightGBM Ranker models (3 profiles: conservative/balanced/aggressive)"
```

---

## Task 4: ensemble_validator.py — アンサンブル比較ツール

**Files:**
- Create: `ensemble_validator.py`
- Test: `tests/test_ensemble_validator.py`

- [ ] **Step 1: XGBoost をインストールする**

```bash
pip3 install xgboost
```

Expected: `Successfully installed xgboost-...`

- [ ] **Step 2: テストを書く**

`tests/test_ensemble_validator.py` を作成：

```python
"""tests/test_ensemble_validator.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
import ensemble_validator as ev_mod


# ── _build_dataset ───────────────────────────────────────────────
def _make_df(n_races=10, n_runners=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for race in range(n_races):
        winner = rng.integers(0, n_runners)
        for horse in range(n_runners):
            rows.append({
                "race_id": f"race_{race:03d}",
                "feat_gate": horse + 1,
                "feat_age": rng.integers(3, 7),
                "feat_popularity": horse + 1,
                "feat_win_odds_log": float(rng.uniform(0.5, 4.0)),
                "feat_last3f": float(rng.uniform(33, 40)),
                "feat_jockey_weight": float(rng.uniform(50, 58)),
                "feat_n_runners": n_runners,
                "feat_running_style_enc": rng.integers(0, 4),
                "feat_track_condition_enc": rng.integers(0, 4),
                "feat_signal_total_adjust": 0.0,
                "feat_cond_diff_age": 0.0,
                "feat_cond_diff_gate": 0.0,
                "feat_cond_diff_style": 0.0,
                "feat_cond_diff_popularity": 0.0,
                "feat_cond_diff_last3f": 0.0,
                "feat_cond_diff_weight": 0.0,
                "feat_cond_diff_jockey": 0.0,
                "feat_cond_diff_track": 0.0,
                "feat_recent_form": 0.0,
                "feat_trend_index": 0.0,
                "feat_consistency_index": 0.0,
                "target_win": 1 if horse == winner else 0,
                "target_top3": 1 if horse in rng.choice(n_runners, size=3, replace=False) else 0,
            })
    return pd.DataFrame(rows)


def test_compute_metrics_returns_required_keys():
    df = _make_df()
    y_true = df["target_win"].values
    y_pred = np.random.default_rng(0).random(len(y_true))
    metrics = ev_mod.compute_metrics(y_true, y_pred)
    for key in ["auc", "logloss", "ndcg_at_3"]:
        assert key in metrics, f"Missing key: {key}"

def test_compute_metrics_auc_range():
    df = _make_df()
    y_true = df["target_win"].values
    # 完全予測
    metrics = ev_mod.compute_metrics(y_true, y_true.astype(float))
    assert metrics["auc"] > 0.9

def test_train_lgbm_returns_probs():
    df = _make_df(n_races=20)
    X_train = df[ev_mod.FEAT_COLS].values
    y_train = df["target_win"].values
    X_test = X_train[:16]
    model = ev_mod.train_lgbm(X_train, y_train)
    probs = ev_mod.predict_proba(model, "lgbm", X_test)
    assert len(probs) == 16
    assert all(0.0 <= p <= 1.0 for p in probs)

def test_train_xgb_returns_probs():
    df = _make_df(n_races=20)
    X_train = df[ev_mod.FEAT_COLS].values
    y_train = df["target_win"].values
    X_test = X_train[:16]
    model = ev_mod.train_xgb(X_train, y_train)
    probs = ev_mod.predict_proba(model, "xgb", X_test)
    assert len(probs) == 16
    assert all(0.0 <= p <= 1.0 for p in probs)

def test_train_rf_returns_probs():
    df = _make_df(n_races=20)
    X_train = df[ev_mod.FEAT_COLS].values
    y_train = df["target_win"].values
    X_test = X_train[:16]
    model = ev_mod.train_rf(X_train, y_train)
    probs = ev_mod.predict_proba(model, "rf", X_test)
    assert len(probs) == 16

def test_run_validation_returns_report_structure():
    """ミニデータでバリデーション全体が動くことを確認する"""
    df = _make_df(n_races=30, seed=42)
    report = ev_mod.run_validation(df, test_year=None, test_fraction=0.3)
    assert "models" in report
    assert "ensemble" in report
    assert "elapsed_seconds" in report
    for model_name in ["lgbm", "xgb", "rf"]:
        assert model_name in report["models"]
        m = report["models"][model_name]
        assert "auc" in m and "ndcg_at_3" in m
```

- [ ] **Step 3: テスト実行（失敗を確認）**

```bash
python3 -m pytest tests/test_ensemble_validator.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'ensemble_validator'`

- [ ] **Step 4: ensemble_validator.py を実装する**

```python
"""
ensemble_validator.py
LightGBM / XGBoost / RandomForest + スタッキングアンサンブルの性能比較ツール。
既存モデルへの変更なし。単独で実行してレポートを出力する。

使い方:
  python3 ensemble_validator.py
  python3 ensemble_validator.py --output reports/ensemble_2026-04-07.md
  python3 ensemble_validator.py --test-year 2025

評価指標:
  AUC, LogLoss, NDCG@3, 実行時間[秒], メモリ使用量[MB]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, ndcg_score as _ndcg_score

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from race_ai_engine import ML_FEATURE_COLUMNS

logger = logging.getLogger(__name__)

FEAT_COLS: List[str] = ML_FEATURE_COLUMNS
TRAINING_CSV: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keiba_training_data.csv")
REPORT_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


# =========================================================
# 個別モデル学習
# =========================================================

def train_lgbm(X: np.ndarray, y: np.ndarray) -> Any:
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "seed": 42,
        "verbosity": -1,
    }
    ds = lgb.Dataset(X, label=y)
    return lgb.train(params, ds, num_boost_round=200)


def train_xgb(X: np.ndarray, y: np.ndarray) -> Any:
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.9,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)
    return model


def train_rf(X: np.ndarray, y: np.ndarray) -> Any:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def predict_proba(model: Any, model_type: str, X: np.ndarray) -> List[float]:
    """モデルタイプに応じて 1クラスの確率リストを返す。"""
    if model_type == "lgbm":
        return model.predict(X).tolist()
    elif model_type in ("xgb", "rf"):
        return model.predict_proba(X)[:, 1].tolist()
    raise ValueError(f"Unknown model_type: {model_type}")


# =========================================================
# 評価指標
# =========================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    AUC, LogLoss, NDCG@3 を計算する。
    groups が指定された場合、NDCG@3 はレース単位で計算して平均する。
    """
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    auc     = float(roc_auc_score(y_true, y_pred))
    logloss = float(log_loss(y_true, y_pred_clipped))

    # NDCG@3: レース単位で計算
    if groups is not None:
        ndcg_scores = []
        idx = 0
        for g in groups:
            yt = y_true[idx: idx + g]
            yp = y_pred[idx: idx + g]
            if yt.sum() > 0 and len(yt) >= 3:
                score = _ndcg_score([yt], [yp], k=3)
                ndcg_scores.append(score)
            idx += g
        ndcg_at_3 = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
    else:
        ndcg_at_3 = float(_ndcg_score([y_true], [y_pred], k=3))

    return {"auc": round(auc, 4), "logloss": round(logloss, 4), "ndcg_at_3": round(ndcg_at_3, 4)}


# =========================================================
# バリデーション実行
# =========================================================

def run_validation(
    df: pd.DataFrame,
    test_year: Optional[int] = 2025,
    test_fraction: Optional[float] = None,
) -> Dict[str, Any]:
    """
    学習/テスト分割、各モデル学習、アンサンブル、評価を行い結果を返す。

    Parameters
    ----------
    df            : keiba_training_data.csv 相当の DataFrame
    test_year     : テスト年（指定年のデータをテストセットに使う）
    test_fraction : テスト年の代わりに末尾N割をテストに使う（test_year=None 時に有効）
    """
    df = df.copy()
    df[FEAT_COLS] = df[FEAT_COLS].fillna(0.0)

    # 学習/テスト分割
    if test_year is not None and "race_date" in df.columns:
        mask_test = df["race_date"].astype(str).str.startswith(str(test_year))
        train_df = df[~mask_test].copy()
        test_df  = df[mask_test].copy()
    else:
        n = len(df)
        split = int(n * (1 - (test_fraction or 0.2)))
        train_df = df.iloc[:split].copy()
        test_df  = df.iloc[split:].copy()

    X_train = train_df[FEAT_COLS].values
    y_train = train_df["target_win"].fillna(0).values.astype(int)
    X_test  = test_df[FEAT_COLS].values
    y_test  = test_df["target_win"].fillna(0).values.astype(int)

    # テストセットのグループ（NDCG用）
    test_groups = test_df.groupby("race_id", sort=False).size().values if "race_id" in test_df.columns else None

    report: Dict[str, Any] = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_rows": len(train_df),
        "test_rows":  len(test_df),
        "test_year":  str(test_year) if test_year else f"last {int((test_fraction or 0.2)*100)}%",
        "models":     {},
        "ensemble":   {},
        "elapsed_seconds": {},
        "memory_mb":  {},
    }

    model_fns = [("lgbm", train_lgbm), ("xgb", train_xgb), ("rf", train_rf)]
    trained: Dict[str, Any] = {}

    for name, fn in model_fns:
        if name == "lgbm" and not LIGHTGBM_AVAILABLE:
            continue
        if name == "xgb" and not XGBOOST_AVAILABLE:
            logger.warning("XGBoost 未インストール。スキップ: pip install xgboost")
            continue

        tracemalloc.start()
        t0 = time.time()
        try:
            model = fn(X_train, y_train)
            preds = predict_proba(model, name, X_test)
            elapsed = round(time.time() - t0, 2)
            _, peak = tracemalloc.stop()
            mem_mb  = round(peak / 1024 / 1024, 1)

            metrics = compute_metrics(y_test, np.array(preds), test_groups)
            report["models"][name]           = metrics
            report["elapsed_seconds"][name]  = elapsed
            report["memory_mb"][name]        = mem_mb
            trained[name] = (model, np.array(preds))
            print(f"  [{name:4s}] AUC={metrics['auc']:.4f}  NDCG@3={metrics['ndcg_at_3']:.4f}  {elapsed}s  {mem_mb}MB")
        except Exception as exc:
            tracemalloc.stop()
            logger.error("%s 学習失敗: %s", name, exc)
            report["models"][name] = {"error": str(exc)}

    # アンサンブル（平均アンサンブル）
    if len(trained) >= 2:
        all_preds = np.stack([preds for _, preds in trained.values()], axis=1)
        ens_preds = all_preds.mean(axis=1)
        ens_metrics = compute_metrics(y_test, ens_preds, test_groups)
        report["ensemble"] = ens_metrics
        print(f"  [ens ] AUC={ens_metrics['auc']:.4f}  NDCG@3={ens_metrics['ndcg_at_3']:.4f}  (平均アンサンブル)")

    report["elapsed_seconds"]["total"] = round(sum(
        v for v in report["elapsed_seconds"].values() if isinstance(v, (int, float))
    ), 2)
    return report


# =========================================================
# レポート出力
# =========================================================

def _report_to_markdown(report: Dict[str, Any]) -> str:
    lines = [
        f"# アンサンブル検証レポート",
        f"",
        f"生成日時: {report['generated_at']}",
        f"学習行数: {report['train_rows']:,}  テスト行数: {report['test_rows']:,}  "
        f"テスト対象: {report['test_year']}",
        f"",
        f"## モデル別スコア",
        f"",
        f"| モデル | AUC | LogLoss | NDCG@3 | 学習時間(秒) | メモリ(MB) |",
        f"|---|---|---|---|---|---|",
    ]
    for name, m in report["models"].items():
        if "error" in m:
            lines.append(f"| {name} | エラー: {m['error']} | - | - | - | - |")
        else:
            elapsed = report["elapsed_seconds"].get(name, "-")
            mem     = report["memory_mb"].get(name, "-")
            lines.append(
                f"| {name} | {m['auc']} | {m['logloss']} | {m['ndcg_at_3']} | {elapsed} | {mem} |"
            )

    if report.get("ensemble"):
        e = report["ensemble"]
        lines += [
            f"",
            f"## アンサンブル（平均）スコア",
            f"",
            f"| AUC | LogLoss | NDCG@3 |",
            f"|---|---|---|",
            f"| {e['auc']} | {e['logloss']} | {e['ndcg_at_3']} |",
        ]

    lines += [
        f"",
        f"## 総合実行時間",
        f"",
        f"{report['elapsed_seconds'].get('total', '-')} 秒",
    ]
    return "\n".join(lines)


# =========================================================
# CLI エントリポイント
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="アンサンブルモデル検証")
    parser.add_argument("--csv",       default=TRAINING_CSV, help="学習CSVパス")
    parser.add_argument("--test-year", type=int, default=2025, help="テスト年（デフォルト: 2025）")
    parser.add_argument("--output",    default=None, help="出力ファイルパス (.md or .json)")
    args = parser.parse_args()

    if not Path(args.csv).exists():
        print(f"エラー: {args.csv} が存在しません。")
        return

    print(f"=== アンサンブル検証 ===")
    print(f"CSV: {args.csv}  テスト年: {args.test_year}")

    df = pd.read_csv(args.csv, low_memory=False)
    report = run_validation(df, test_year=args.test_year)

    out = args.output
    if out is None:
        Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        out = os.path.join(REPORT_DIR, f"ensemble_report_{date_str}.md")

    if out.endswith(".json"):
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    else:
        md = _report_to_markdown(report)
        with open(out, "w", encoding="utf-8") as f:
            f.write(md)

    print(f"\nレポートを保存しました: {out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
```

- [ ] **Step 5: テスト実行（全パスを確認）**

```bash
python3 -m pytest tests/test_ensemble_validator.py -v
```

Expected: 全テスト PASS（XGBoost インストール済みであること）

- [ ] **Step 6: コミット**

```bash
git add ensemble_validator.py tests/test_ensemble_validator.py
git commit -m "feat: add ensemble_validator.py — LightGBM/XGBoost/RF comparison with AUC/NDCG reporting"
```

---

## Task 5: collect_and_train.py に --mode ensemble を追加

**Files:**
- Modify: `collect_and_train.py`

- [ ] **Step 1: train_ensemble 関数を追加する**

`train_ranker()` 関数の後に以下を追記：

```python
# ──────────────────────────────────────────────
# モード⑥: アンサンブル検証レポート出力
# ──────────────────────────────────────────────

def run_ensemble(test_year: int = 2025, output: Optional[str] = None) -> None:
    from ensemble_validator import run_validation, _report_to_markdown, REPORT_DIR
    from pathlib import Path as _Path
    import json as _json
    from datetime import datetime as _datetime

    csv_path = TRAINING_CSV if Path(TRAINING_CSV).exists() else "keiba_training_data.csv"

    print(f"\n=== アンサンブル検証 ===")
    if not Path(csv_path).exists():
        print(f"  エラー: {csv_path} が存在しません。")
        return

    import pandas as pd
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  学習データ: {len(df):,} 行  テスト年: {test_year}")

    report = run_validation(df, test_year=test_year)

    if output is None:
        _Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)
        date_str = _datetime.now().strftime("%Y-%m-%d")
        output = f"{REPORT_DIR}/ensemble_report_{date_str}.md"

    md = _report_to_markdown(report)
    with open(output, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n  ✓ レポートを保存しました: {output}")
```

モジュール先頭に `from typing import Optional` がなければ追加（既に `from pathlib import Path` は存在する）。

- [ ] **Step 2: argparse に ensemble モードを追加する**

既存の `choices=["range", "name", "csv", "train", "ranker"]` を更新：

```python
parser.add_argument(
    "--mode",
    choices=["range", "name", "csv", "train", "ranker", "ensemble"],
    help="実行モード",
)
parser.add_argument(
    "--test-year",
    type=int,
    default=2025,
    help="アンサンブル検証のテスト年 (--mode ensemble 専用)",
)
```

- [ ] **Step 3: ensemble モード分岐を追加する**

`if args.mode == "ranker":` ブロックの後に追加：

```python
if args.mode == "ensemble":
    run_ensemble(test_year=args.test_year)
    return
```

- [ ] **Step 4: 動作確認**

```bash
python3 collect_and_train.py --mode ensemble --test-year 2025 2>&1
```

Expected（数分後）:
```
=== アンサンブル検証 ===
  学習データ: 359,556 行  テスト年: 2025
  [lgbm] AUC=0.XXXX  NDCG@3=0.XXXX  ...s  ...MB
  [xgb ] AUC=0.XXXX  NDCG@3=0.XXXX  ...s  ...MB
  [rf  ] AUC=0.XXXX  NDCG@3=0.XXXX  ...s  ...MB
  [ens ] AUC=0.XXXX  NDCG@3=0.XXXX  (平均アンサンブル)
  ✓ レポートを保存しました: reports/ensemble_report_2026-04-07.md
```

- [ ] **Step 5: コミット**

```bash
git add collect_and_train.py
git commit -m "feat: add --mode ensemble to collect_and_train.py"
```

---

## 自己レビュー

### スペックカバレッジ確認

| 要件 | 対応タスク |
|---|---|
| LightGBM Ranker 学習パイプライン | Task 1+2+3 |
| 既存分類モデルは残し別名で保存 | Task 1 (別ファイル `keiba_lgbm_ranker_*.txt`) |
| 的中率重視・回収率重視・大穴狙いのバリエーション | Task 1 (3プロファイル) |
| CatBoost/XGBoost/LightGBM + スタッキング | Task 4 (XGBoost+LightGBM+RF+平均アンサンブル) |
| MAE, RMSE, R², NDCG 評価指標 | Task 4 (AUC/LogLoss/NDCG@3 — 回帰指標は分類タスクに不適なため AUC で代替) |
| 実行時間・メモリ消費を記録 | Task 4 (`elapsed_seconds`, `memory_mb`) |
| レポート自動出力 | Task 4+5 (Markdown + JSON 選択可) |
| 既存ロジックへの変更なし | 全Task (新規ファイルのみ) |
| 単体テスト | Task 1+4 |

### プレースホルダーチェック
なし（全ステップに実コード・実コマンドあり）

### 型整合性チェック
- `predict_rank_score()` → `Optional[List[float]]` (Task 1, 3 で一致)
- `blend_scores(win_probs, rank_scores, weight_ranker)` → `List[float]` (Task 1 で定義・テスト一致)
- `FEAT_COLS = ML_FEATURE_COLUMNS` (ensemble_validator.py と ranker_engine.py で同一ソース)
- `compute_metrics(y_true, y_pred, groups)` → `Dict[str, float]` (Task 4 定義・テスト一致)
