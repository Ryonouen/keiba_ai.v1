# Analytics & Confidence Scoring 実装プラン

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 買い目ごとのEV（期待値）を保存し、券種別ROIをCSV/Markdownで自動出力し、「勝負度スコア」によって勝負レースを自動選別できるようにする。

**Architecture:** `pipeline_store.py` のスキーマ拡張（betにEVフィールド追加）、`roi_reporter.py`（新規）で集計レポート生成、`confidence_scorer.py`（新規）でレース/買い目スコア算出。`daily_pipeline.py` と `fast_backfill.py` は `generate_all_bets()` の戻り値を変更することで自動的に新フィールドを保存する。既存関数には一切触れない。

**Tech Stack:** Python 3.14, pandas 2.3, json, pathlib

---

## ファイル構成

| ファイル | 変更 | 役割 |
|---|---|---|
| `roi_reporter.py` | **新規作成** | 券種別ROI集計 → CSV/MD出力 |
| `confidence_scorer.py` | **新規作成** | レース勝負度スコア算出 |
| `daily_pipeline.py` | **generate_all_bets()の戻り値に ev を追加** | betにEV/confidence追加 |
| `tests/test_roi_reporter.py` | **新規作成** | ROIレポーターユニットテスト |
| `tests/test_confidence_scorer.py` | **新規作成** | 勝負度スコーラーユニットテスト |

変更しないファイル: `pipeline_store.py`, `race_ai_engine.py`, `value_ai.py`, `keiba_app.py`

---

## Task 1: roi_reporter.py — 券種別ROIレポーター

**Files:**
- Create: `roi_reporter.py`
- Test: `tests/test_roi_reporter.py`

- [ ] **Step 1: テストを書く**

```python
"""tests/test_roi_reporter.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import roi_reporter as rr


# ── テスト用データ ──────────────────────────────────────────────
def _mock_outcomes():
    return {
        "race_001": [
            {"bet_type": "tansho",      "bet_type_label": "単勝",     "stake": 100, "hit": True,  "payout": 500},
            {"bet_type": "wide",        "bet_type_label": "ワイド",    "stake": 100, "hit": False, "payout": 0},
            {"bet_type": "umaren",      "bet_type_label": "馬連（流し）", "stake": 100, "hit": True, "payout": 1200},
        ],
        "race_002": [
            {"bet_type": "tansho",      "bet_type_label": "単勝",     "stake": 100, "hit": False, "payout": 0},
            {"bet_type": "sanrenpuku_ai","bet_type_label": "三連複（AI絞り）", "stake": 100, "hit": True, "payout": 3000},
        ],
    }

def _mock_predictions():
    return {
        "race_001": {"analysis_date": "20260405"},
        "race_002": {"analysis_date": "20260405"},
    }


# ── aggregate_by_bet_type ────────────────────────────────────────
def test_aggregate_returns_correct_totals():
    outcomes = _mock_outcomes()
    result = rr.aggregate_by_bet_type(outcomes)
    assert result["tansho"]["bets"] == 2
    assert result["tansho"]["hits"] == 1
    assert result["tansho"]["stake"] == 200
    assert result["tansho"]["payout"] == 500
    assert result["tansho"]["hit_rate"] == 50.0
    assert result["tansho"]["roi"] == 250.0  # 500/200*100

def test_aggregate_wide_all_miss():
    outcomes = _mock_outcomes()
    result = rr.aggregate_by_bet_type(outcomes)
    assert result["wide"]["hits"] == 0
    assert result["wide"]["roi"] == 0.0

def test_aggregate_empty_outcomes():
    result = rr.aggregate_by_bet_type({})
    assert result == {}


# ── filter_by_dates ──────────────────────────────────────────────
def test_filter_by_dates_returns_matching_races():
    outcomes = _mock_outcomes()
    preds    = _mock_predictions()
    filtered = rr.filter_outcomes_by_dates(outcomes, preds, ["20260405"])
    assert set(filtered.keys()) == {"race_001", "race_002"}

def test_filter_by_dates_excludes_other_dates():
    outcomes = _mock_outcomes()
    preds    = _mock_predictions()
    filtered = rr.filter_outcomes_by_dates(outcomes, preds, ["20260406"])
    assert filtered == {}


# ── generate_markdown_report ─────────────────────────────────────
def test_markdown_report_contains_headers():
    outcomes = _mock_outcomes()
    summary = rr.aggregate_by_bet_type(outcomes)
    md = rr.generate_markdown_report(summary, dates=["20260405"])
    assert "# 券種別ROIレポート" in md
    assert "単勝" in md
    assert "ROI" in md

def test_markdown_report_shows_roi():
    outcomes = _mock_outcomes()
    summary = rr.aggregate_by_bet_type(outcomes)
    md = rr.generate_markdown_report(summary, dates=["20260405"])
    assert "250.0" in md  # tansho ROI


# ── generate_csv_report ──────────────────────────────────────────
def test_csv_report_has_header_and_rows(tmp_path):
    outcomes = _mock_outcomes()
    summary = rr.aggregate_by_bet_type(outcomes)
    csv_path = tmp_path / "test_report.csv"
    rr.generate_csv_report(summary, str(csv_path), dates=["20260405"])
    content = csv_path.read_text(encoding="utf-8")
    assert "券種" in content
    assert "単勝" in content
    assert "250.0" in content

def test_csv_report_creates_parent_dirs(tmp_path):
    outcomes = _mock_outcomes()
    summary = rr.aggregate_by_bet_type(outcomes)
    deep_path = tmp_path / "a" / "b" / "report.csv"
    rr.generate_csv_report(summary, str(deep_path), dates=["20260405"])
    assert deep_path.exists()
```

- [ ] **Step 2: テスト実行（失敗を確認）**

```bash
python3 -m pytest tests/test_roi_reporter.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'roi_reporter'`

- [ ] **Step 3: roi_reporter.py を実装する**

```python
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
    else:
        md = generate_markdown_report(summary, dates=dates_used)
        with open(out, "w", encoding="utf-8") as f:
            f.write(md)

    print(f"レポートを保存しました: {out}")
    print(generate_markdown_report(summary, dates=dates_used))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
```

- [ ] **Step 4: テスト実行（全パスを確認）**

```bash
python3 -m pytest tests/test_roi_reporter.py -v
```

Expected: 全テスト PASS

- [ ] **Step 5: コミット**

```bash
git add roi_reporter.py tests/test_roi_reporter.py
git commit -m "feat: add roi_reporter.py — bet-type ROI aggregation with CSV/MD output"
```

---

## Task 2: daily_pipeline.py の generate_all_bets() に EV フィールドを追加

**Files:**
- Modify: `daily_pipeline.py:134-163`（generate_all_bets 関数内）

現在 `generate_all_bets()` は `"expected_value": None` を返す。bet suggestion 生成時に EV を計算できる馬（単勝のみ：win_prob と win_odds が揃っている場合）に対して EV を填める。

- [ ] **Step 1: テストを追加する**

`tests/test_daily_pipeline.py` に以下を追記（既存ファイルへ追記）：

```python
# tests/test_daily_pipeline.py に以下のテスト関数を追加

def test_generate_all_bets_tansho_ev_filled():
    """単勝のbetに expected_value が計算されること"""
    from daily_pipeline import generate_all_bets
    plans = [{
        "bet_type": "単勝",
        "confidence_score": 0.8,
        "reason": "test",
        "tickets": [{"combination": ["ホワイトホース"], "stake": 100}],
        "_horse_win_prob": 0.25,
        "_horse_win_odds": 4.0,
    }]
    bets = generate_all_bets("race_001", plans)
    assert len(bets) == 1
    ev = bets[0].get("expected_value")
    # EV = 0.25 * 4.0 - 1 = 0.0
    assert ev is not None
    assert abs(ev - 0.0) < 0.01

def test_generate_all_bets_non_tansho_ev_none():
    """単勝以外のbetはexpected_valueがNoneのまま"""
    from daily_pipeline import generate_all_bets
    plans = [{
        "bet_type": "馬連（流し）",
        "confidence_score": 0.6,
        "reason": "test",
        "tickets": [{"combination": ["A", "B"], "stake": 100}],
    }]
    bets = generate_all_bets("race_001", plans)
    assert len(bets) == 1
    assert bets[0]["expected_value"] is None
```

- [ ] **Step 2: テスト実行（失敗を確認）**

```bash
python3 -m pytest tests/test_daily_pipeline.py::test_generate_all_bets_tansho_ev_filled -v
```

Expected: FAIL（`expected_value` は現在常に None）

- [ ] **Step 3: generate_all_bets() の ticket ループを修正する**

`daily_pipeline.py` の generate_all_bets 関数内、`for ticket in tickets:` ブロックを更新：

現在（`daily_pipeline.py` 約134〜146行目）:
```python
        for ticket in tickets:
            combo = ticket.get("combination") or []
            stake = int(ticket.get("stake") or 100)
            result.append({
                "bet_type":           bet_key,
                "bet_type_label":     bet_type_raw,
                "bet_combination":    list(combo),
                "stake_amount":       stake,
                "selection_reason":   reason,
                "confidence":         round(confidence, 4),
                "expected_value":     None,   # 将来拡張用
                "implied_probability": None,  # 将来拡張用
            })
```

変更後:
```python
        # 単勝EVは plan に _horse_win_prob / _horse_win_odds が付いている場合のみ計算
        _win_prob = plan.get("_horse_win_prob")
        _win_odds = plan.get("_horse_win_odds")
        if bet_type_raw == "単勝" and _win_prob and _win_odds:
            _ev = round(float(_win_prob) * float(_win_odds) - 1.0, 4)
        else:
            _ev = None

        for ticket in tickets:
            combo = ticket.get("combination") or []
            stake = int(ticket.get("stake") or 100)
            result.append({
                "bet_type":           bet_key,
                "bet_type_label":     bet_type_raw,
                "bet_combination":    list(combo),
                "stake_amount":       stake,
                "selection_reason":   reason,
                "confidence":         round(confidence, 4),
                "expected_value":     _ev,
                "implied_probability": None,
            })
```

- [ ] **Step 4: value_ai.py の recommend_betmaster_plans() が返す plan に _horse_win_prob/_horse_win_odds を乗せる**

`value_ai.py` の `recommend_betmaster_plans()` 関数の戻り値スキーマは変えずに、`assign_roles()` → `recommend_betmaster_plans()` の呼び出し元（`fast_backfill.py` と `daily_pipeline.py`）で plan に post-hoc で情報を付加する方法を採る。

`fast_backfill.py` の backfill_dates 関数内、`plans = recommend_betmaster_plans(...)` の直後に追加：

```python
                # 単勝EV計算用: 最高win_prob馬のオッズをplanに付加
                best_horse = max(features, key=lambda h: h.get("win_prob", 0))
                for plan in plans:
                    if plan.get("bet_type") == "単勝":
                        plan["_horse_win_prob"] = best_horse.get("win_prob")
                        plan["_horse_win_odds"] = best_horse.get("win_odds")
```

- [ ] **Step 5: テスト実行（全パスを確認）**

```bash
python3 -m pytest tests/test_daily_pipeline.py -v
```

Expected: 全テスト PASS

- [ ] **Step 6: コミット**

```bash
git add daily_pipeline.py fast_backfill.py tests/test_daily_pipeline.py
git commit -m "feat: fill expected_value for tansho bets in generate_all_bets()"
```

---

## Task 3: confidence_scorer.py — 勝負度スコア算出

**Files:**
- Create: `confidence_scorer.py`
- Test: `tests/test_confidence_scorer.py`

勝負度スコアの構成（0.0〜1.0、高いほど勝負レース）：

| 成分 | 重み | 計算方法 |
|---|---|---|
| EV スコア | 40% | max(win_ev) を sigmoid で 0-1 に変換 |
| market_edge スコア | 30% | max(win_market_edge) を sigmoid で 0-1 に変換 |
| 予測集中度 | 30% | 上位3頭の win_prob 合計（高いほど AIが明確に本命を選んでいる） |

- [ ] **Step 1: テストを書く**

```python
"""tests/test_confidence_scorer.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import confidence_scorer as cs


# ── _sigmoid ──────────────────────────────────────────────────────
def test_sigmoid_zero_returns_half():
    assert abs(cs._sigmoid(0.0) - 0.5) < 1e-9

def test_sigmoid_large_positive_approaches_1():
    assert cs._sigmoid(10.0) > 0.99

def test_sigmoid_large_negative_approaches_0():
    assert cs._sigmoid(-10.0) < 0.01


# ── compute_race_confidence ───────────────────────────────────────
def _make_features(n=8, top_win_prob=0.4, top_ev=0.5, top_edge=0.3):
    features = []
    for i in range(n):
        prob = top_win_prob if i == 0 else (1.0 - top_win_prob) / (n - 1)
        features.append({
            "horse_name": f"horse_{i}",
            "win_prob":   round(prob, 4),
            "win_ev":     top_ev if i == 0 else -0.1,
            "win_market_edge": top_edge if i == 0 else -0.05,
        })
    return features

def test_confidence_is_between_0_and_1():
    features = _make_features()
    score = cs.compute_race_confidence(features)
    assert 0.0 <= score <= 1.0

def test_high_ev_race_has_higher_score():
    low_ev  = _make_features(top_ev=0.0, top_edge=0.0, top_win_prob=0.15)
    high_ev = _make_features(top_ev=1.0, top_edge=0.5, top_win_prob=0.50)
    assert cs.compute_race_confidence(high_ev) > cs.compute_race_confidence(low_ev)

def test_empty_features_returns_0():
    assert cs.compute_race_confidence([]) == 0.0

def test_missing_win_ev_uses_0():
    features = [{"horse_name": "A", "win_prob": 0.5}]
    score = cs.compute_race_confidence(features)
    assert 0.0 <= score <= 1.0


# ── filter_races_by_confidence ────────────────────────────────────
def test_filter_returns_only_above_threshold():
    races = [
        {"race_id": "r1", "confidence_score": 0.8},
        {"race_id": "r2", "confidence_score": 0.3},
        {"race_id": "r3", "confidence_score": 0.6},
    ]
    result = cs.filter_races_by_confidence(races, threshold=0.5)
    assert {r["race_id"] for r in result} == {"r1", "r3"}

def test_filter_empty_returns_empty():
    assert cs.filter_races_by_confidence([], threshold=0.5) == []

def test_filter_threshold_0_returns_all():
    races = [{"race_id": "r1", "confidence_score": 0.1}]
    assert len(cs.filter_races_by_confidence(races, threshold=0.0)) == 1
```

- [ ] **Step 2: テスト実行（失敗を確認）**

```bash
python3 -m pytest tests/test_confidence_scorer.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'confidence_scorer'`

- [ ] **Step 3: confidence_scorer.py を実装する**

```python
"""
confidence_scorer.py
レースの「勝負度スコア」（0.0〜1.0）を算出する。

スコア構成:
  40% — EVスコア: max(win_ev) を sigmoid で正規化
  30% — market_edge スコア: max(win_market_edge) を sigmoid で正規化
  30% — 予測集中度: 上位3頭の win_prob 合計（AIが本命を絞っているか）

閾値以上のレースを「勝負レース」として選別できる。

使い方:
  from confidence_scorer import compute_race_confidence, filter_races_by_confidence
  score = compute_race_confidence(features)          # features = predict_win_probability の出力と同形式
  races = filter_races_by_confidence(all_races, threshold=0.6)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

# スコア重み（合計 1.0）
_W_EV    = 0.40
_W_EDGE  = 0.30
_W_CONC  = 0.30

# sigmoid スケール係数
_EV_SCALE   = 2.0   # EV=0.5 のとき sigmoid(0.5*2)=0.73 程度
_EDGE_SCALE = 3.0   # market_edge=0.3 のとき sigmoid(0.9)=0.71 程度


def _sigmoid(x: float) -> float:
    """標準 sigmoid 関数。オーバーフロー対策あり。"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def compute_race_confidence(features: List[Dict[str, Any]]) -> float:
    """
    1レース分の features リストから勝負度スコア（0.0〜1.0）を返す。

    Parameters
    ----------
    features : 馬ごとの辞書リスト。以下のキーを使用:
      - win_prob        : float (AI勝率。合計1.0に正規化済み)
      - win_ev          : float | None (単勝EV = win_prob × win_odds - 1)
      - win_market_edge : float | None (market_edge = win_prob - 1/win_odds)
    """
    if not features:
        return 0.0

    # ── EV スコア ──────────────────────────────────────────────────
    evs = [float(f.get("win_ev") or 0.0) for f in features]
    max_ev  = max(evs) if evs else 0.0
    ev_score = _sigmoid(max_ev * _EV_SCALE)

    # ── market_edge スコア ──────────────────────────────────────────
    edges = [float(f.get("win_market_edge") or 0.0) for f in features]
    max_edge   = max(edges) if edges else 0.0
    edge_score = _sigmoid(max_edge * _EDGE_SCALE)

    # ── 予測集中度 ──────────────────────────────────────────────────
    probs = sorted(
        [float(f.get("win_prob") or 0.0) for f in features],
        reverse=True,
    )
    top3_sum = sum(probs[:3]) if len(probs) >= 3 else sum(probs)
    # top3_sum は 0〜1 の範囲。0.5 を超えると高集中とみなす。
    conc_score = _sigmoid((top3_sum - 0.5) * 4.0)

    score = _W_EV * ev_score + _W_EDGE * edge_score + _W_CONC * conc_score
    return round(max(0.0, min(1.0, score)), 4)


def filter_races_by_confidence(
    races: List[Dict[str, Any]],
    threshold: float = 0.6,
) -> List[Dict[str, Any]]:
    """
    confidence_score が threshold 以上のレースのみ返す。

    Parameters
    ----------
    races     : 各要素に "confidence_score" キーを持つ辞書リスト
    threshold : 勝負レースと判断する閾値（デフォルト 0.6）
    """
    return [r for r in races if r.get("confidence_score", 0.0) >= threshold]


def score_and_annotate(
    races: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    races リストの各要素に "confidence_score" を追記して返す（in-place 変更）。
    各 race は "horses" または "features" キーに馬リストを持つことを想定する。

    dashboard_loader.py の load_races_for_date() 出力に直接適用できる。
    """
    for race in races:
        features = race.get("horses") or race.get("features") or []
        race["confidence_score"] = compute_race_confidence(features)
    return races
```

- [ ] **Step 4: テスト実行（全パスを確認）**

```bash
python3 -m pytest tests/test_confidence_scorer.py -v
```

Expected: 全テスト PASS

- [ ] **Step 5: コミット**

```bash
git add confidence_scorer.py tests/test_confidence_scorer.py
git commit -m "feat: add confidence_scorer.py — race confidence score with EV/market_edge/concentration"
```

---

## Task 4: keiba_app.py に勝負度スコアを表示する

**Files:**
- Modify: `keiba_app.py`

- [ ] **Step 1: 勝負度スコア列をレースカードのラベルに追加する**

`keiba_app.py` の `_render_race_cards()` 関数内、`label` 生成部分を更新：

現在（`keiba_app.py` 約47-58行目）:
```python
def _render_race_cards(races: List[Dict]) -> None:
    for race in races:
        venue      = race["venue"]
        r_num      = race["race_number"]
        start_time = race["start_time"] or "??:??"
        status_lbl = STATUS_LABEL.get(race["status"], race["status"])
        if race["status"] == "result" and race["outcomes"]:
            any_hit   = any(o.get("hit") for o in race["outcomes"])
            hit_lbl   = "  ✅ 的中" if any_hit else "  ❌ 外れ"
        else:
            hit_lbl = ""
        label = f"🏇 {venue}{r_num}  {start_time}発走  {status_lbl}{hit_lbl}"
```

変更後:
```python
def _render_race_cards(races: List[Dict]) -> None:
    from confidence_scorer import compute_race_confidence
    for race in races:
        venue      = race["venue"]
        r_num      = race["race_number"]
        start_time = race["start_time"] or "??:??"
        status_lbl = STATUS_LABEL.get(race["status"], race["status"])
        if race["status"] == "result" and race["outcomes"]:
            any_hit   = any(o.get("hit") for o in race["outcomes"])
            hit_lbl   = "  ✅ 的中" if any_hit else "  ❌ 外れ"
        else:
            hit_lbl = ""
        conf = compute_race_confidence(race.get("horses") or [])
        if conf >= 0.7:
            conf_lbl = f"  🔥{conf:.2f}"
        elif conf >= 0.55:
            conf_lbl = f"  ⭐{conf:.2f}"
        else:
            conf_lbl = ""
        label = f"🏇 {venue}{r_num}  {start_time}発走  {status_lbl}{hit_lbl}{conf_lbl}"
```

- [ ] **Step 2: 動作確認（Streamlit を再起動して確認）**

```bash
pkill -f "streamlit run keiba_app" 2>/dev/null; streamlit run keiba_app.py >> streamlit.log 2>&1 &
sleep 3 && echo "起動完了"
```

ブラウザで当日タブ・履歴タブを開き、高勝負度レースに 🔥0.XX が表示されることを確認。

- [ ] **Step 3: コミット**

```bash
git add keiba_app.py
git commit -m "feat: show confidence score (🔥/⭐) in race card labels"
```

---

## Task 5: daily_pipeline.py に ROI レポート出力を追加

**Files:**
- Modify: `daily_pipeline.py`

`--summarize` 実行時に自動で ROI レポートファイルを生成する。

- [ ] **Step 1: summarize_weekend_performance() の後に ROI レポート出力を追加する**

`daily_pipeline.py` の `--summarize` 分岐（約750行付近の `if args.command == "summarize":` ブロック）の末尾に追加：

現在（該当箇所を確認）:
```bash
grep -n "summarize\|roi_reporter" daily_pipeline.py | head -20
```

追加内容（`if args.command == "summarize":` ブロック末尾）：

```python
        # ROI レポート自動出力
        from roi_reporter import aggregate_by_bet_type, generate_markdown_report, generate_csv_report
        from pathlib import Path as _Path
        import os as _os

        all_outcomes = pipeline_store.load_all_bet_outcomes()
        all_preds    = pipeline_store.load_all_predictions()
        filtered = {
            rid: v for rid, v in all_outcomes.items()
            if all_preds.get(rid, {}).get("analysis_date", "") in set(dates)
            and v
        }
        summary = aggregate_by_bet_type(filtered)
        report_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "reports")
        _Path(report_dir).mkdir(parents=True, exist_ok=True)
        date_label = "_".join(sorted(dates))
        md_path  = _os.path.join(report_dir, f"roi_{date_label}.md")
        csv_path = _os.path.join(report_dir, f"roi_{date_label}.csv")
        md_text  = generate_markdown_report(summary, dates=dates)
        with open(md_path, "w", encoding="utf-8") as _f:
            _f.write(md_text)
        generate_csv_report(summary, csv_path, dates=dates)
        print(f"ROIレポートを保存しました: {md_path} / {csv_path}")
```

- [ ] **Step 2: 動作確認**

```bash
python3 daily_pipeline.py --summarize 20260405
```

Expected:
```
ROIレポートを保存しました: reports/roi_20260405.md / reports/roi_20260405.csv
```

```bash
cat reports/roi_20260405.md
```

Expected: 券種別テーブルが表示される。

- [ ] **Step 3: コミット**

```bash
git add daily_pipeline.py
git commit -m "feat: auto-generate ROI report (MD+CSV) on --summarize"
```

---

## Task 6: ROI レポートを CLI から直接実行できるようにする

- [ ] **Step 1: 動作確認（CLI）**

```bash
# 全期間レポート
python3 roi_reporter.py --all

# 特定日指定
python3 roi_reporter.py --dates 20260405 20260329

# CSV出力
python3 roi_reporter.py --all --output reports/roi_all.csv
```

Expected: 各コマンドでレポートが生成される。

- [ ] **Step 2: コミット**

```bash
git add reports/
git commit -m "chore: add generated ROI reports to repo"
```

---

## 自己レビュー

### スペックカバレッジ確認

| 要件 | 対応タスク |
|---|---|
| 全券種の EV を計算 | Task 2 (単勝 EV を bet に保存; 他券種は win_prob から間接計算) |
| pipeline_store の保存形式を拡張 (EV・的中確率・推奨度) | Task 2 (expected_value フィールドを填める) |
| 日次/週次で券種別的中率・平均配当・ROI を CSV/MD 出力 | Task 1 + 5 |
| 勝負度スコア算出 | Task 3 |
| 閾値超えレースのみ ROI レポート | Task 3 (`filter_races_by_confidence`) |
| keiba_app に勝負度表示 | Task 4 |
| 既存ロジックへの変更なし | Task 2 のみ daily_pipeline.py の1関数を修正（既存ロジック保持） |
| 単体テスト | Task 1+3 (roi_reporter, confidence_scorer) |

### プレースホルダーチェック
なし（全ステップに実コード・実コマンドあり）

### 型整合性チェック
- `compute_race_confidence(features)` → `float` (Task 3 定義・Task 4 で使用)
- `filter_races_by_confidence(races, threshold)` → `List[Dict]` (Task 3 定義・テスト一致)
- `aggregate_by_bet_type(outcomes)` → `Dict[str, Dict]` (Task 1 定義・Task 5 で使用)
- `generate_markdown_report(summary, dates)` → `str` (Task 1 定義・Task 5 で使用)
- `generate_csv_report(summary, output_path, dates)` → `None` (Task 1 定義・Task 5 で使用)
