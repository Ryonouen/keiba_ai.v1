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
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            mem_mb  = round(peak / 1024 / 1024, 1)

            metrics = compute_metrics(y_test, np.array(preds), test_groups)
            report["models"][name]           = metrics
            report["elapsed_seconds"][name]  = elapsed
            report["memory_mb"][name]        = mem_mb
            trained[name] = (model, np.array(preds))
            print(f"  [{name:4s}] AUC={metrics['auc']:.4f}  NDCG@3={metrics['ndcg_at_3']:.4f}  {elapsed}s  {mem_mb}MB")
        except Exception as exc:
            if tracemalloc.is_tracing():
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
