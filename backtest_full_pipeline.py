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

    Raises:
        FileNotFoundError: csv_path が存在しない場合（pandas から伝播）
    """
    if years is None:
        years = TARGET_YEARS

    df = pd.read_csv(csv_path, low_memory=False)
    df["year"] = pd.to_datetime(df["race_date"], errors="coerce").dt.year
    na_count = df["year"].isna().sum()
    if na_count > 0:
        print(f"[警告] race_date パース失敗: {na_count}/{len(df)} 行を除外します")
    df = df[df["year"].isin(years)].copy()
    if len(df) == 0:
        print(f"[警告] 指定年 {years} に該当するデータが0件です。CSVパスとyears引数を確認してください。")

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


def build_feature_dict(row: Dict[str, Any], win_prob: float) -> Dict[str, Any]:
    """
    CSV の 1行 dict を value_ai.py が期待するキー形式に変換する。

    - win_odds: exp(feat_win_odds_log) で復元
    - running_style: enc 値を文字列に変換
    - jockey_delta / place_odds: 未収録のためデフォルト値

    Raises:
        なし（不正値はデフォルト値にフォールバック）
    """
    val_odds_log = row.get("feat_win_odds_log")
    raw_odds_log = float(val_odds_log) if val_odds_log is not None else 0.0
    win_odds = math.exp(raw_odds_log) if raw_odds_log != 0.0 else None

    val_enc = row.get("feat_running_style_enc")
    enc = int(float(val_enc)) if val_enc is not None else 3
    running_style = _ENC_TO_STYLE.get(enc, "unknown")

    val_pop = row.get("feat_popularity")
    pop = int(float(val_pop)) if val_pop is not None else 99
    place_factor = _PLACE_FACTORS.get(pop, _PLACE_FACTOR_DEFAULT)
    place_odds = round(win_odds * place_factor / 100, 2) if win_odds else None

    val_last3f = row.get("feat_last3f")
    last3f = float(val_last3f) if val_last3f is not None else 0.0

    val_gate = row.get("feat_gate")
    gate = int(float(val_gate)) if val_gate is not None else 0

    val_age = row.get("feat_age")
    age = int(float(val_age)) if val_age is not None else 0

    val_runners = row.get("feat_n_runners")
    n_runners = int(float(val_runners)) if val_runners is not None else 0

    val_target_win = row.get("target_win")
    target_win = int(float(val_target_win)) if val_target_win is not None else 0

    val_target_top3 = row.get("target_top3")
    target_top3 = int(float(val_target_top3)) if val_target_top3 is not None else 0

    return {
        "horse_name":       str(row.get("horse_name") or ""),
        "win_prob":         win_prob,
        "model_score":      win_prob,
        "win_odds":         win_odds,
        "place_odds":       place_odds,
        "popularity_rank":  pop,
        "running_style":    running_style,
        "last3f":           last3f,
        "gate":             gate,
        "age":              age,
        "n_runners":        n_runners,
        "jockey_delta":     0.0,   # CSV未収録
        "jockey_reason_codes": [],
        # 正解ラベル（払戻計算用、value_ai には渡さない）
        "_target_win":      target_win,
        "_target_top3":     target_top3,
        "_win_odds_log":    raw_odds_log,
    }


def build_pace_balance(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    レース内の脚質カウントを返す（classify_race_structure の pace_balance 用）。

    Returns:
        {"逃げ": n, "先行": n, "差し": n, "追込": n}
    """
    pb: Dict[str, int] = {"逃げ": 0, "先行": 0, "差し": 0, "追込": 0}
    style_map = {0: "逃げ", 1: "先行", 2: "差し", 3: "追込"}
    for row in rows:
        val = row.get("feat_running_style_enc")
        enc = int(float(val)) if val is not None else 3
        key = style_map.get(enc)
        if key:
            pb[key] += 1
    return pb


def train_lgbm_for_year(df: pd.DataFrame, test_year: int):
    """
    test_year より前のデータで LightGBM を訓練して返す。

    Args:
        df: 全年データ（'year' 列を含む）
        test_year: テスト対象年（この年より前のみ訓練に使用）

    Returns:
        訓練済み lightgbm.Booster

    Raises:
        ValueError: 訓練データが0件の場合
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

    EMPTY: Dict[str, Any] = {
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
        race_pace="medium",
    )
    return plan


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

    Note:
        馬連・ワイド・3連複の払戻は理論近似値（実際の JRA 払戻とは乖離する場合あり）
    """
    if plan.get("skip"):
        return None

    bet_type = plan.get("bet_type", "-")
    horses = plan.get("horses", [])
    invest = int(plan.get("total_stake") or STAKE_UNIT)

    fmap = {f["horse_name"]: f for f in features}

    # ── 単勝 ──
    if bet_type == "単勝" and len(horses) >= 1:
        h = fmap.get(horses[0])
        if h is None:
            return {"hit": False, "payout": 0, "invest": invest}
        if h["_target_win"] == 1:
            payout = math.exp(h["_win_odds_log"]) * invest
            return {"hit": True, "payout": payout, "invest": invest}
        return {"hit": False, "payout": 0, "invest": invest}

    # ── 複勝 ──
    # factor は複勝オッズそのもの（例: 1番人気=1.55倍）
    if bet_type == "複勝" and len(horses) >= 1:
        h = fmap.get(horses[0])
        if h is None:
            return {"hit": False, "payout": 0, "invest": invest}
        if h["_target_top3"] == 1:
            pop = h.get("popularity_rank", 99)
            factor = _PLACE_FACTORS.get(pop, _PLACE_FACTOR_DEFAULT)
            payout = factor * invest
            return {"hit": True, "payout": payout, "invest": invest}
        return {"hit": False, "payout": 0, "invest": invest}

    # ── 馬連 ──
    # 的中: 推奨2頭が上位2頭（_target_win=1 の馬 + もう1頭の top3）
    # 払戻近似: oa * ob * 0.75 * invest（理論値）
    if bet_type in ("馬連", "馬連流し") and len(horses) >= 2:
        winner_name = next(
            (f["horse_name"] for f in features if f["_target_win"] == 1), None
        )
        top3_names = {f["horse_name"] for f in features if f["_target_top3"] == 1}
        # 軸 = horses[0]、相手 = horses[1:]
        axis = horses[0]
        partners = horses[1:]
        # 的中: 軸が top3 かつ 相手のいずれかが top3、かつ両馬が 1着・2着
        hit = False
        payout_horse_a = payout_horse_b = None
        if axis in top3_names:
            for partner in partners:
                if partner in top3_names:
                    # 両馬が top3 に含まれる（馬連の近似的中）
                    ha = fmap.get(axis)
                    hb = fmap.get(partner)
                    if ha is not None and hb is not None:
                        hit = True
                        payout_horse_a, payout_horse_b = ha, hb
                        break
        if hit and payout_horse_a and payout_horse_b:
            oa = math.exp(payout_horse_a["_win_odds_log"])
            ob = math.exp(payout_horse_b["_win_odds_log"])
            ticket_count = max(len(partners), 1)
            payout = oa * ob * 0.75 * (invest / ticket_count)
            return {"hit": True, "payout": payout, "invest": invest}
        return {"hit": False, "payout": 0, "invest": invest}

    # ── ワイド / ワイドBOX ──
    # 払戻近似: oa * ob * 0.25 * invest（理論値）
    if bet_type in ("ワイド", "ワイドBOX") and len(horses) >= 2:
        top3_names = {f["horse_name"] for f in features if f["_target_top3"] == 1}
        from itertools import combinations
        ticket_count = max(len(list(combinations(horses, 2))), 1)
        for ha_name, hb_name in combinations(horses, 2):
            if ha_name in top3_names and hb_name in top3_names:
                ha = fmap.get(ha_name)
                hb = fmap.get(hb_name)
                if ha and hb:
                    oa = math.exp(ha["_win_odds_log"])
                    ob = math.exp(hb["_win_odds_log"])
                    payout = oa * ob * 0.25 * (invest / ticket_count)
                    return {"hit": True, "payout": payout, "invest": invest}
        return {"hit": False, "payout": 0, "invest": invest}

    # ── 3連複 / 3連複BOX ──
    # 払戻近似: 的中3頭のオッズ積 * 0.70 * (invest / ticket_count)（理論値）
    if bet_type in ("3連複", "三連複", "3連複BOX") and len(horses) >= 3:
        top3_names = {f["horse_name"] for f in features if f["_target_top3"] == 1}
        from itertools import combinations
        combos = list(combinations(horses, 3))
        ticket_count = max(len(combos), 1)
        for trio in combos:
            if all(h in top3_names for h in trio):
                hs = [fmap.get(h) for h in trio]
                if all(h is not None for h in hs):
                    odds_product = math.prod(math.exp(h["_win_odds_log"]) for h in hs)
                    payout = odds_product * 0.70 * (invest / ticket_count)
                    return {"hit": True, "payout": payout, "invest": invest}
        return {"hit": False, "payout": 0, "invest": invest}

    # ── 対応外 ──
    return {"hit": False, "payout": 0, "invest": invest}


def run_backtest(
    csv_path: str = TRAINING_CSV,
    target_years: List[int] = TARGET_YEARS,
    bankroll: int = BANKROLL,
) -> Dict[str, Any]:
    """
    フルパイプラインバックテストを実行する。

    Returns:
        {
            "total_races": int,
            "recommended": int,
            "hits": int,
            "total_invest": float,
            "total_payout": float,
            "roi": float,
            "by_year": {year: {"total_races", "recommended", "hits", "invest", "payout", "roi"}},
            "by_bet_type": {bet_type: {"races", "hits", "invest", "payout"}},
            "records": [{"race_id", "year", "bet_type", "horses", "hit", "invest", "payout"}],
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

    records: List[Dict[str, Any]] = []
    by_year: Dict[int, Dict] = {}
    by_bet_type: Dict[str, Dict] = defaultdict(
        lambda: {"races": 0, "hits": 0, "invest": 0.0, "payout": 0.0}
    )

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

            X = group[ML_FEATURE_COLUMNS].fillna(0.0)
            probs = model.predict(X.values)

            features = [
                build_feature_dict(row, win_prob=float(prob))
                for row, prob in zip(rows, probs)
            ]
            pace_balance = build_pace_balance(rows)

            plan = run_value_ai_pipeline(features, pace_balance, bankroll=bankroll)
            y_total += 1

            result = simulate_payout(plan, features)
            if result is None:
                continue

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
                "race_id":  race_id,
                "year":     test_year,
                "bet_type": bet_type,
                "horses":   ",".join(plan.get("horses", [])),
                "hit":      hit,
                "invest":   invest,
                "payout":   round(payout, 0),
            })

            if (idx + 1) % 200 == 0:
                roi_so_far = y_payout / y_invest if y_invest > 0 else 0.0
                print(
                    f"  [{test_year}] {idx+1}/{len(race_ids)} "
                    f"推奨率:{y_recommended/(idx+1)*100:.0f}% "
                    f"ROI:{roi_so_far*100:.1f}%",
                    flush=True,
                )

        y_roi = y_payout / y_invest if y_invest > 0 else 0.0
        by_year[test_year] = {
            "total_races": y_total,
            "recommended": y_recommended,
            "hits":        y_hits,
            "invest":      y_invest,
            "payout":      y_payout,
            "roi":         y_roi,
        }
        print(
            f"[{test_year}] 完了 推奨:{y_recommended}/{y_total} "
            f"的中:{y_hits} ROI:{y_roi*100:.1f}%\n"
        )

    total_invest = sum(v["invest"] for v in by_year.values())
    total_payout = sum(v["payout"] for v in by_year.values())
    total_hits   = sum(v["hits"]   for v in by_year.values())
    total_rec    = sum(v["recommended"] for v in by_year.values())
    total_races  = sum(v["total_races"] for v in by_year.values())

    return {
        "total_races":  total_races,
        "recommended":  total_rec,
        "hits":         total_hits,
        "total_invest": total_invest,
        "total_payout": total_payout,
        "roi":          total_payout / total_invest if total_invest > 0 else 0.0,
        "by_year":      by_year,
        "by_bet_type":  dict(by_bet_type),
        "records":      records,
    }


def print_summary(result: Dict[str, Any]) -> None:
    """バックテスト結果をコンソールに整形して表示する。"""
    total_races = result["total_races"]
    recommended = result["recommended"]
    hits        = result["hits"]
    roi         = result["roi"]
    by_year     = result["by_year"]
    by_bet_type = result["by_bet_type"]

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
        print(
            f"  {year}: ROI {y_roi:6.1f}%  "
            f"推奨率 {y_sel:.0f}%  "
            f"的中 {v['hits']}/{v['recommended']}"
        )

    print(f"\n【券種別】")
    for bet_type, v in sorted(by_bet_type.items(), key=lambda x: -x[1]["invest"]):
        if v["races"] == 0:
            continue
        b_hit = v["hits"] / v["races"] * 100
        b_roi = v["payout"] / v["invest"] * 100 if v["invest"] > 0 else 0.0
        note  = "  ※近似" if bet_type not in ("単勝", "複勝") else ""
        print(f"  {bet_type:6s}: 的中率 {b_hit:5.1f}%  ROI {b_roi:6.1f}%  ({v['races']}件){note}")

    print("\n" + "=" * 60)


def save_records_csv(
    result: Dict[str, Any],
    out_path: str = "backtest_full_pipeline_result.csv",
) -> None:
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
    parser.add_argument("--csv", default=TRAINING_CSV)
    parser.add_argument("--years", nargs="+", type=int, default=TARGET_YEARS)
    parser.add_argument("--bankroll", type=int, default=BANKROLL)
    parser.add_argument("--save-csv", action="store_true")
    args = parser.parse_args()

    result = run_backtest(csv_path=args.csv, target_years=args.years, bankroll=args.bankroll)
    print_summary(result)
    if args.save_csv:
        save_records_csv(result)
