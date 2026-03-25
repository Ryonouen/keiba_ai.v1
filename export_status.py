"""
export_status.py
================
学習データCSVエクスポートの進捗を確認するCLIツール。

使い方:
  python3 export_status.py
"""
import csv
import json
from pathlib import Path

RACE_IDS_FILE = "collected_race_ids.json"
TRAINING_CSV  = "keiba_training_data.csv"


def main():
    total_ids = 0
    years: dict = {}
    if Path(RACE_IDS_FILE).exists():
        with open(RACE_IDS_FILE) as f:
            ids = json.load(f)
        total_ids = len(ids)
        for rid in ids:
            y = rid[:4]
            years[y] = years.get(y, 0) + 1

    csv_rows  = 0
    csv_races: set = set()
    if Path(TRAINING_CSV).exists():
        with open(TRAINING_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                csv_rows += 1
                if row.get("race_id"):
                    csv_races.add(row["race_id"])

    csv_size = Path(TRAINING_CSV).stat().st_size // 1024 if Path(TRAINING_CSV).exists() else 0
    done = len(csv_races)
    pct  = int(done / total_ids * 100) if total_ids else 0
    bar  = "█" * (pct // 5) + "░" * (20 - pct // 5)

    print("=" * 50)
    print("  競馬AI 学習データ収集状況")
    print("=" * 50)
    print(f"  収集済みレースID: {total_ids:,} 件")
    if years:
        print(f"  年別内訳: { {y: c for y, c in sorted(years.items())} }")
    print()
    print(f"  CSVエクスポート: [{bar}] {pct}%")
    print(f"    完了レース: {done:,} / {total_ids:,}")
    print(f"    行数:       {csv_rows:,} 行")
    print(f"    サイズ:     {csv_size:,} KB")
    remaining = total_ids - done
    if remaining > 0:
        est_min = int(remaining * 3.0 / 60)
        print(f"    残り推定:   約 {est_min} 分（{remaining:,} レース × 平均3秒）")
    print()

    if done == 0:
        print("  ⚠ CSVが空です。エクスポートを実行してください:")
        print("    python3 collect_and_train.py --mode csv")
    elif done < total_ids:
        print("  ▶ エクスポート未完了。再実行で続きから再開できます:")
        print("    python3 collect_and_train.py --mode csv")
    else:
        print("  ✓ エクスポート完了！次は訓練を実行:")
        print("    python3 collect_and_train.py --mode train")


if __name__ == "__main__":
    main()
