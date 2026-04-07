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
