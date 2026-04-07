"""
roi_reporter.py
pipeline_bet_outcomes から券種別 ROI を集計し CSV / Markdown で出力する。
pipeline_store には直接依存せず、呼び出し元が outcomes/predictions を渡す設計。

使い方:
  python3 roi_reporter.py --dates 20260405 20260406
  python3 roi_reporter.py --all --output reports/roi_2026-04.csv

CLI では pipeline_bet_outcomes.json と pipeline_predictions.json を自動読み込みする。
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(_HERE, "reports")

BET_TYPE_LABEL_MAP: Dict[str, str] = {
    "tansho":         "単勝",
    "fukusho":        "複勝",
    "wide":           "ワイド",
    "umaren":         "馬連",
    "umatan":         "馬単",
    "sanrenpuku_ai":  "三連複（AI絞り）",
    "sanrenpuku_all": "三連複（全頭）",
    "sanrentan_ai":   "三連単（AI絞り）",
    "sanrentan_all":  "三連単（全頭）",
}

BET_TYPE_ORDER: List[str] = [
    "tansho", "fukusho", "wide", "umaren", "umatan",
    "sanrenpuku_ai", "sanrenpuku_all", "sanrentan_ai", "sanrentan_all",
]


# =========================================================
# 集計
# =========================================================

def aggregate_by_bet_type(
    outcomes: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    """
    race_id → outcomes の辞書から券種別集計を返す。

    Returns
    -------
    {
      "<bet_type>": {
        "label": str, "bets": int, "hits": int, "stake": int,
        "payout": int, "hit_rate": float, "roi": float
      }, ...
    }
    """
    result: Dict[str, Dict[str, Any]] = {}

    for race_id, race_outcomes in outcomes.items():
        for o in race_outcomes:
            bet_type = str(o.get("bet_type") or "")
            if not bet_type:
                continue
            label  = str(o.get("bet_type_label") or BET_TYPE_LABEL_MAP.get(bet_type, bet_type))
            stake  = int(o.get("stake") or 0)
            payout = int(o.get("payout") or 0)
            hit    = bool(o.get("hit"))

            if bet_type not in result:
                result[bet_type] = {
                    "label": label, "bets": 0, "hits": 0,
                    "stake": 0, "payout": 0, "hit_rate": 0.0, "roi": 0.0,
                }
            r = result[bet_type]
            r["bets"]   += 1
            r["stake"]  += stake
            r["payout"] += payout
            if hit:
                r["hits"] += 1

    for bet_type, r in result.items():
        r["hit_rate"] = round(r["hits"] / r["bets"] * 100, 1) if r["bets"] else 0.0
        r["roi"]      = round(r["payout"] / r["stake"] * 100, 1) if r["stake"] else 0.0

    return result


def filter_outcomes_by_dates(
    outcomes: Dict[str, List[Dict[str, Any]]],
    predictions: Dict[str, Any],
    dates: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    predictions の analysis_date が dates に含まれる race_id のみ返す。
    """
    target = set(dates)
    return {
        race_id: race_outcomes
        for race_id, race_outcomes in outcomes.items()
        if predictions.get(race_id, {}).get("analysis_date", "") in target
        and race_outcomes
    }


# =========================================================
# レポート生成
# =========================================================

def generate_markdown_report(
    summary: Dict[str, Dict[str, Any]],
    dates: Optional[List[str]] = None,
) -> str:
    date_str = "、".join(dates) if dates else "全期間"
    lines = [
        f"# 券種別ROIレポート",
        f"",
        f"対象日: {date_str}",
        f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"| 券種 | 買い目数 | 的中 | 的中率(%) | 投資(円) | 回収(円) | ROI(%) |",
        f"|---|---|---|---|---|---|---|",
    ]
    total_bets = total_hits = total_stake = total_payout = 0

    for bet_type in BET_TYPE_ORDER + [k for k in summary if k not in BET_TYPE_ORDER]:
        r = summary.get(bet_type)
        if r is None:
            continue
        lines.append(
            f"| {r['label']} | {r['bets']} | {r['hits']} | {r['hit_rate']} |"
            f" {r['stake']:,} | {r['payout']:,} | {r['roi']} |"
        )
        total_bets   += r["bets"]
        total_hits   += r["hits"]
        total_stake  += r["stake"]
        total_payout += r["payout"]

    total_hit_rate = round(total_hits / total_bets * 100, 1) if total_bets else 0.0
    total_roi      = round(total_payout / total_stake * 100, 1) if total_stake else 0.0
    lines += [
        f"| **合計** | **{total_bets}** | **{total_hits}** | **{total_hit_rate}** |"
        f" **{total_stake:,}** | **{total_payout:,}** | **{total_roi}** |",
    ]
    return "\n".join(lines)


def generate_csv_report(
    summary: Dict[str, Dict[str, Any]],
    output_path: str,
    dates: Optional[List[str]] = None,
) -> None:
    """
    集計結果を CSV ファイルとして保存する。出力先ディレクトリが存在しない場合は作成する。
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["券種", "買い目数", "的中", "的中率(%)", "投資(円)", "回収(円)", "ROI(%)"]
    rows = []
    for bet_type in BET_TYPE_ORDER + [k for k in summary if k not in BET_TYPE_ORDER]:
        r = summary.get(bet_type)
        if r is None:
            continue
        rows.append({
            "券種":       r["label"],
            "買い目数":   r["bets"],
            "的中":       r["hits"],
            "的中率(%)":  r["hit_rate"],
            "投資(円)":   r["stake"],
            "回収(円)":   r["payout"],
            "ROI(%)":    r["roi"],
        })
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("CSVレポートを保存しました: %s", output_path)


# =========================================================
# CLI エントリポイント
# =========================================================

def _load_json(path: str) -> Any:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="券種別ROIレポート出力")
    parser.add_argument("--dates", nargs="+", default=None, help="対象日付 (例: 20260405 20260406)")
    parser.add_argument("--all",   action="store_true", help="全期間を対象にする")
    parser.add_argument("--output", default=None, help="出力ファイルパス (.md or .csv)")
    args = parser.parse_args()

    outcomes    = _load_json(os.path.join(_HERE, "pipeline_bet_outcomes.json"))
    predictions = _load_json(os.path.join(_HERE, "pipeline_predictions.json"))

    if args.all:
        filtered = {rid: v for rid, v in outcomes.items() if v}
        dates_used = sorted({predictions.get(rid, {}).get("analysis_date", "") for rid in filtered})
    elif args.dates:
        filtered   = filter_outcomes_by_dates(outcomes, predictions, args.dates)
        dates_used = args.dates
    else:
        parser.print_help()
        return

    summary = aggregate_by_bet_type(filtered)

    out = args.output
    if out is None:
        Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        out = os.path.join(REPORT_DIR, f"roi_report_{date_str}.md")

    if out.endswith(".csv"):
        generate_csv_report(summary, out, dates=dates_used)
        md = generate_markdown_report(summary, dates=dates_used)
    else:
        md = generate_markdown_report(summary, dates=dates_used)
        with open(out, "w", encoding="utf-8") as f:
            f.write(md)

    print(f"レポートを保存しました: {out}")
    print(md)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
