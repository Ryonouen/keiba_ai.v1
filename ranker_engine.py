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
    各馬のランキングスコア（高いほど上位と判断）を返す。モデル未存在時は None。
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
