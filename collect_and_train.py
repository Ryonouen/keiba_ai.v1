"""
collect_and_train.py
====================
レースIDを収集して学習データCSVを出力し、LightGBMを訓練するスクリプト。

使い方:
  python collect_and_train.py               # メインメニューを表示
  python collect_and_train.py --mode range  # 日付範囲でID収集
  python collect_and_train.py --mode name   # レース名でID収集
  python collect_and_train.py --mode train  # 既存CSVからLightGBM訓練
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

RACE_IDS_FILE = "collected_race_ids.json"
TRAINING_CSV  = "keiba_training_data.csv"


# ──────────────────────────────────────────────
# 進捗表示
# ──────────────────────────────────────────────

def _progress(current: int, total: int, label: str = "") -> None:
    pct = int(current / total * 100) if total else 0
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    print(f"\r  [{bar}] {pct:3d}% ({current}/{total}) {label:<20}", end="", flush=True)
    if current >= total:
        print()


# ──────────────────────────────────────────────
# ID保存・読み込み
# ──────────────────────────────────────────────

def _save_ids(ids: list, path: str = RACE_IDS_FILE) -> None:
    existing = _load_ids(path)
    merged = list(dict.fromkeys(existing + ids))  # 重複除去・順序維持
    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"  → {path} に {len(merged)} 件保存（新規追加: {len(merged)-len(existing)} 件）")


def _load_ids(path: str = RACE_IDS_FILE) -> list:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────
# モード①: 日付範囲でID収集（平場・重賞まとめて）
# ──────────────────────────────────────────────

def collect_by_date_range():
    """
    例: 2024年の阪神・東京・中山の全レースIDを収集する
    """
    from dividend_scraper import fetch_race_ids_by_date_range, VENUE_CODE_MAP

    print("\n=== 日付範囲でレースID収集 ===")
    print("開催場所コード:", {v: k for k, v in VENUE_CODE_MAP.items()})
    print()

    start = input("開始日 (YYYYMMDD, 例: 20240101): ").strip()
    end   = input("終了日 (YYYYMMDD, 例: 20241231): ").strip()

    print("開催場所を選択 (複数可, カンマ区切り)")
    print("  05=東京, 06=中山, 08=京都, 09=阪神, 07=中京, 空白=全場")
    venues_input = input("開催場所コード: ").strip()
    venue_codes = [v.strip() for v in venues_input.split(",") if v.strip()] or None

    print(f"\n{start} ～ {end} のレースIDを収集中...")
    result = fetch_race_ids_by_date_range(
        start, end, venue_codes,
        progress_callback=lambda c, t, d: _progress(c, t, d),
    )

    print(f"\n取得: {result['n_races']} レース（{result['n_dates']} 日間）")
    _save_ids(result["race_ids"])
    return result["race_ids"]


# ──────────────────────────────────────────────
# モード②: レース名でID収集（阪神大賞典など特定重賞）
# ──────────────────────────────────────────────

def collect_by_race_name():
    """
    例: 阪神大賞典 2015〜2024 を収集する
    """
    from dividend_scraper import fetch_race_ids_by_name

    print("\n=== レース名でレースID収集 ===")
    print("※ 部分一致で検索します（例: '阪神大賞典', '天皇賞', '有馬記念'）")
    print()

    race_name = input("レース名: ").strip()
    start_year = int(input("開始年 (例: 2015): ").strip())
    end_year   = int(input("終了年 (例: 2024): ").strip())

    print("開催月を入力（カンマ区切り。例: 阪神大賞典=3, 天皇賞春=5）")
    months_input = input("月 (空白=全月): ").strip()
    search_months = [int(m.strip()) for m in months_input.split(",") if m.strip()] or None

    print(f"\n「{race_name}」を {start_year}〜{end_year} 年で検索中...")
    if search_months:
        print(f"  対象月: {search_months}（月を絞ると大幅に高速化）")

    result = fetch_race_ids_by_name(
        race_name, start_year, end_year, search_months,
        progress_callback=lambda c, t, d: _progress(c, t, d),
    )

    print(f"\n取得: {result['n_races']} レース")
    for year, ids in sorted(result["by_year"].items()):
        print(f"  {year}: {ids}")

    if result["race_ids"]:
        _save_ids(result["race_ids"])
    else:
        print("  ※ レースが見つかりませんでした。レース名・月の指定を確認してください。")

    return result["race_ids"]


# ──────────────────────────────────────────────
# モード③: 学習データCSV出力
# ──────────────────────────────────────────────

def export_csv():
    from backtest_runner import export_training_csv_noselenium

    ids = _load_ids()
    if not ids:
        print(f"  エラー: {RACE_IDS_FILE} が空です。先にID収集を実行してください。")
        return

    print(f"\n=== 学習データCSV出力 ===")
    print(f"対象: {len(ids)} レース → {TRAINING_CSV}")
    print()

    result = export_training_csv_noselenium(
        ids, TRAINING_CSV, append=True,
        progress_callback=lambda c, t, rid: _progress(c, t, rid),
    )

    print(f"\n完了: {result.get('n_rows', 0)} 行出力")
    if result.get("errors"):
        print(f"  スキップ: {len(result['errors'])} レース（データ取得失敗）")


# ──────────────────────────────────────────────
# モード④: LightGBM訓練
# ──────────────────────────────────────────────

def train_model():
    from race_ai_engine import train_lightgbm_model, TRAINING_CSV as DEFAULT_CSV

    csv_path = TRAINING_CSV if Path(TRAINING_CSV).exists() else DEFAULT_CSV

    print(f"\n=== LightGBM 訓練 ===")
    if not Path(csv_path).exists():
        print(f"  エラー: {csv_path} が存在しません。先にCSV出力を実行してください。")
        return

    import pandas as pd
    df = pd.read_csv(csv_path)
    print(f"  学習データ: {len(df)} 行, 勝ち馬: {df['target_win'].sum()} 頭")
    print(f"  訓練開始...")

    ok = train_lightgbm_model(csv_path)
    if ok:
        print("  ✓ 訓練完了。次回のレース予測から自動的にモデルが使われます。")
    else:
        print("  ✗ 訓練失敗。LightGBMがインストールされているか確認してください: pip install lightgbm pandas")


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


# ──────────────────────────────────────────────
# モード⑥: アンサンブル検証レポート出力
# ──────────────────────────────────────────────

def run_ensemble(test_year: int = 2025, output: Optional[str] = None) -> None:
    from datetime import datetime
    from ensemble_validator import run_validation, _report_to_markdown, REPORT_DIR

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
        Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        output = f"{REPORT_DIR}/ensemble_report_{date_str}.md"

    md = _report_to_markdown(report)
    with open(output, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n  ✓ レポートを保存しました: {output}")


# ──────────────────────────────────────────────
# メインメニュー
# ──────────────────────────────────────────────

MENU = """
=====================================
  競馬AI 学習データ収集・訓練ツール
=====================================
現在の収集済みID: {n_ids} 件 ({ids_file})
CSVファイル:      {csv_status}

1. 日付範囲でレースID収集（平場・重賞まとめて）
2. レース名でレースID収集（阪神大賞典など特定重賞）
3. 学習データCSV出力（収集済みIDを使って特徴量生成）
4. LightGBM訓練（CSVからモデルを作成）
5. 全部まとめて実行（1→3→4 または 2→3→4）
q. 終了

選択: """

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["range", "name", "csv", "train", "ranker", "ensemble"],
        help="実行モード (ranker は --profile と組み合わせて使用)",
    )
    parser.add_argument(
        "--profile",
        choices=["conservative", "balanced", "aggressive"],
        default="balanced",
        help="Rankerプロファイル (--mode ranker 専用)",
    )
    parser.add_argument(
        "--test-year",
        type=int,
        default=2025,
        help="アンサンブル検証のテスト年 (--mode ensemble 専用)",
    )
    args = parser.parse_args()

    if args.mode == "range":
        collect_by_date_range()
        return
    if args.mode == "name":
        collect_by_race_name()
        return
    if args.mode == "csv":
        export_csv()
        return
    if args.mode == "train":
        train_model()
        return
    if args.mode == "ranker":
        train_ranker(profile=args.profile)
        return
    if args.mode == "ensemble":
        run_ensemble(test_year=args.test_year)
        return

    # インタラクティブメニュー
    while True:
        n_ids = len(_load_ids())
        csv_status = f"{TRAINING_CSV} ({Path(TRAINING_CSV).stat().st_size // 1024}KB)" if Path(TRAINING_CSV).exists() else "なし"
        choice = input(MENU.format(n_ids=n_ids, ids_file=RACE_IDS_FILE, csv_status=csv_status)).strip()

        if choice == "1":
            collect_by_date_range()
        elif choice == "2":
            collect_by_race_name()
        elif choice == "3":
            export_csv()
        elif choice == "4":
            train_model()
        elif choice == "5":
            print("\nどちらのID収集方法ですか？")
            print("  a. 日付範囲（平場レース多数）")
            print("  b. レース名（特定重賞のみ）")
            sub = input("選択 (a/b): ").strip()
            if sub == "a":
                collect_by_date_range()
            else:
                collect_by_race_name()
            export_csv()
            train_model()
        elif choice == "q":
            break
        else:
            print("  1〜5 または q を入力してください")


if __name__ == "__main__":
    main()
