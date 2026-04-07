"""
fast_backfill.py
----------------
keiba_training_data.csv の 2026 年データを使って、
pipeline_predictions.json / pipeline_bet_suggestions.json / pipeline_bet_outcomes.json
を高速に生成する。

  スクレイピング不要 → 28日分を数分で処理できる。
  実際のオッズ (feat_win_odds_log) と脚質 (feat_running_style_enc) を使うため予測精度も正確。

使い方:
  python3 fast_backfill.py             # keiba_training_data.csv の全2026年日程
  python3 fast_backfill.py 20260101 20260331  # 期間指定
  python3 fast_backfill.py 20260329    # 1日のみ
"""
from __future__ import annotations

import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

import pandas as pd

# ── 定数 ──────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

STYLE_DEC: Dict[int, str] = {0: "front", 1: "stalker", 2: "closer", 3: "unknown"}
VENUE_NAME: Dict[str, str] = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}
FEAT_COLS = [
    "feat_gate", "feat_age", "feat_popularity", "feat_win_odds_log",
    "feat_last3f", "feat_jockey_weight", "feat_n_runners",
    "feat_running_style_enc", "feat_track_condition_enc",
    "feat_signal_total_adjust",
    "feat_cond_diff_age", "feat_cond_diff_gate", "feat_cond_diff_style",
    "feat_cond_diff_popularity", "feat_cond_diff_last3f",
    "feat_cond_diff_weight", "feat_cond_diff_jockey", "feat_cond_diff_track",
    "feat_recent_form", "feat_trend_index", "feat_consistency_index",
]


# ── ヘルパー ──────────────────────────────────────────────────
def _win_odds_from_log(log_val: float) -> Optional[float]:
    if log_val and log_val > 0:
        return round(math.exp(log_val), 1)
    return None


def _build_features(group: pd.DataFrame, probs: List[float]) -> List[Dict]:
    """CSV の1レース分の行からfeaturesリストを構築する。"""
    from race_ai_engine import estimate_place_prob, fair_odds, calc_expected_value, calc_market_edge

    features = []
    for i, (_, row) in enumerate(group.iterrows()):
        win_prob   = float(probs[i])
        win_odds   = _win_odds_from_log(float(row.get("feat_win_odds_log") or 0))
        place_prob = estimate_place_prob(win_prob)

        feat: Dict[str, Any] = {
            "horse_name":           row["horse_name"],
            "horse_number":         str(i + 1),
            "running_style":        STYLE_DEC.get(int(row["feat_running_style_enc"]) if pd.notna(row.get("feat_running_style_enc")) else 3, "unknown"),
            "running_style_source": "csv",
            "running_style_missing": False,
            "win_prob":    round(win_prob, 4),
            "place_prob":  round(place_prob, 4),
            "win_odds":    win_odds,
            "place_odds":  None,
            "popularity":  int(row.get("feat_popularity") or 0) or None,
            "fair_win_odds":    fair_odds(win_prob),
            "fair_place_odds":  fair_odds(place_prob),
            "win_ev":           calc_expected_value(win_prob, win_odds),
            "place_ev":         None,
            "win_market_edge":  calc_market_edge(win_prob, win_odds),
            "place_market_edge": None,
        }

        # 全 feat_* をそのままコピー
        for col in FEAT_COLS:
            val = row.get(col)
            feat[col] = float(val) if pd.notna(val) else 0.0

        features.append(feat)

    return features


def _build_ev_table(features: List[Dict]) -> List[Dict]:
    return [
        {
            "horse_name": f["horse_name"],
            "win_prob":   f["win_prob"],
            "win_odds":   f.get("win_odds"),
            "win_ev":     f.get("win_ev"),
            "value_gap":  (f["win_prob"] * f["win_odds"] - 1.0)
                          if f.get("win_odds") and f.get("win_prob") else None,
        }
        for f in features
    ]


def _build_race_structure(group: pd.DataFrame) -> Dict:
    n = int(group["feat_n_runners"].iloc[0]) if "feat_n_runners" in group else len(group)
    return {
        "n_runners": n,
        "pace": "medium",
        "pace_distribution": {"front": 0.3, "mid": 0.4, "late": 0.3},
    }


def _build_race_meta(race_id: str, race_date: str, analysis_date: str) -> Dict:
    venue_code = race_id[4:6]
    venue = VENUE_NAME.get(venue_code, f"場所{venue_code}")
    race_no = int(race_id[10:12]) if race_id[10:12].isdigit() else 0
    return {
        "race_id":        race_id,
        "race_name":      f"{venue}{race_no}R ({race_date})",
        "race_date":      race_date,
        "analysis_date":  analysis_date,
        "race_info_text": f"{venue}{race_no}R",
        "venue":          venue,
        "race_no":        race_no,
    }


# ── メイン処理 ────────────────────────────────────────────────
def backfill_dates(date_strs: List[str]) -> None:
    from race_ai_engine import predict_win_probability_with_model, MODEL_FILE
    from value_ai import assign_roles, recommend_betmaster_plans
    import pipeline_store
    from daily_pipeline import generate_all_bets, evaluate_prediction_for_day

    print(f"CSV読み込み中...")
    df = pd.read_csv(os.path.join(_HERE, "keiba_training_data.csv"), low_memory=False)
    df["race_date_str"] = df["race_date"].str.replace("-", "")
    print(f"  総行数: {len(df):,}")

    for date_str in date_strs:
        race_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        day_df = df[df["race_date_str"] == date_str]
        race_ids = day_df["race_id"].unique().tolist()

        if not race_ids:
            print(f"[{date_str}] CSV にデータなし。スキップ")
            continue

        print(f"\n[{date_str}] {len(race_ids)} レース処理開始")
        success = skipped = 0

        for i, race_id in enumerate(race_ids, 1):
            race_id = str(race_id)
            group = day_df[day_df["race_id"].astype(str) == race_id].reset_index(drop=True)
            print(f"  [{i}/{len(race_ids)}] {race_id} ({len(group)}頭)", end=" ", flush=True)

            try:
                # LightGBM 予測
                feat_matrix = group[FEAT_COLS].fillna(0.0)
                raw_probs = predict_win_probability_with_model(
                    feat_matrix.to_dict("records"), MODEL_FILE
                )
                if raw_probs is None or len(raw_probs) != len(group):
                    n = len(group)
                    raw_probs = [1.0 / n] * n

                # Ranker ブレンド（モデルが存在する場合のみ）
                try:
                    from ranker_engine import predict_rank_score, blend_scores
                    feat_dicts = feat_matrix.to_dict("records")
                    rank_scores = predict_rank_score(feat_dicts, profile="balanced")
                    if rank_scores is not None:
                        raw_probs = blend_scores(list(raw_probs), rank_scores, weight_ranker=0.3)
                except Exception:
                    pass

                # features リスト構築
                features = _build_features(group, list(raw_probs))
                ev_table = _build_ev_table(features)
                race_structure = _build_race_structure(group)
                race_meta = _build_race_meta(race_id, race_date, date_str)

                # 予測保存
                pipeline_store.save_prediction_v2(
                    race_id=race_id,
                    race_meta=race_meta,
                    features=features,
                    ev_table=ev_table,
                    race_structure=race_structure,
                    danger_v2=[],
                    analysis_date=date_str,
                )

                # 買い目生成・保存
                horse_roles = assign_roles(features, ev_table, race_structure, [])
                plans = recommend_betmaster_plans(features, race_structure, horse_roles)
                # 単勝EV計算用: 最高win_prob馬のオッズをplanに付加
                best_horse = max(features, key=lambda h: h.get("win_prob", 0))
                for plan in plans:
                    if plan.get("bet_type") == "単勝":
                        plan["_horse_win_prob"] = best_horse.get("win_prob")
                        plan["_horse_win_odds"] = best_horse.get("win_odds")
                bets = generate_all_bets(race_id, plans)
                pipeline_store.save_bet_suggestions(race_id, bets)

                print(f"→ {len(bets)} 買い目")
                success += 1

            except Exception as e:
                print(f"→ エラー: {e}")
                skipped += 1

        print(f"[{date_str}] 予測完了: 成功 {success} / スキップ {skipped}")

        # evaluate（払戻・的中照合）
        print(f"[{date_str}] --evaluate 実行中...")
        try:
            result = evaluate_prediction_for_day(date_str)
            print(f"[{date_str}] evaluate 完了: {result.get('success',0)}/{result.get('total',0)}")
        except Exception as e:
            print(f"[{date_str}] evaluate エラー: {e}")

        time.sleep(1)

    print("\n=== バックフィル完了 ===")


def _parse_args() -> List[str]:
    """コマンドライン引数から処理対象日付リストを返す。"""
    args = sys.argv[1:]

    if not args:
        # デフォルト: CSV の 2026 年全日程
        df = pd.read_csv(os.path.join(_HERE, "keiba_training_data.csv"),
                         usecols=["race_date"], low_memory=False)
        return sorted(
            {d.replace("-", "") for d in df["race_date"].dropna()
             if str(d).startswith("2026")}
        )

    if len(args) == 1:
        # 1日のみ
        return [args[0]]

    # 期間指定
    start, end = args[0], args[1]
    df = pd.read_csv(os.path.join(_HERE, "keiba_training_data.csv"),
                     usecols=["race_date"], low_memory=False)
    start_ymd = f"{start[:4]}-{start[4:6]}-{start[6:8]}"
    end_ymd   = f"{end[:4]}-{end[4:6]}-{end[6:8]}"
    return sorted({
        d.replace("-", "") for d in df["race_date"].dropna()
        if start_ymd <= str(d) <= end_ymd
    })


if __name__ == "__main__":
    dates = _parse_args()
    if not dates:
        print("処理対象日なし。CSV に対象データがありません。")
        sys.exit(0)
    print(f"処理対象: {len(dates)} 日  {dates[0]} 〜 {dates[-1]}")
    backfill_dates(dates)
