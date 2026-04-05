# パイプラインダッシュボード Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `keiba_app.py` を URL貼り付け型の手動ツールから、`weekend_pipeline.sh` の自動実行結果を表示する読み取り専用 Streamlit ダッシュボードに完全刷新する。

**Architecture:** データ読み込み・集計ロジックを `dashboard_loader.py`（純粋関数、Streamlit 非依存）に分離し、`keiba_app.py` は UI 描画のみに専念する。`pipeline_predictions.json` / `pipeline_bet_suggestions.json` / `pipeline_bet_outcomes.json` の 3 ファイルを読み取り専用で参照する。

**Tech Stack:** Python 3.14, Streamlit 1.55, pandas, pytest

---

## ファイル構成

| ファイル | 変更 | 責務 |
|---|---|---|
| `dashboard_loader.py` | **新規作成** | 3つのJSONを読んでマージ・集計する純粋関数群 |
| `keiba_app.py` | **完全書き直し** | Streamlit UI のみ（Tab 1 当日 / Tab 2 履歴） |
| `tests/test_dashboard_loader.py` | **新規作成** | dashboard_loader の単体テスト 7件 |

---

## Task 1: tests/test_dashboard_loader.py（テスト先行）

**Files:**
- Create: `tests/test_dashboard_loader.py`

- [ ] **Step 1: テストファイルを作成する**

```python
# tests/test_dashboard_loader.py
import json
import os
import sys
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dashboard_loader as dl

# ── フィクスチャ ────────────────────────────────────────────
PRED_V2 = {
    "202606030401": {
        "race_id": "202606030401",
        "race_name": "３歳未勝利 出馬表 | 2026年4月5日 中山1R レース情報(JRA) - netkeiba",
        "analysis_date": "20260405",
        "start_time": "10:00",
        "start_datetime": "2026-04-05T10:00:00",
        "horses": [
            {
                "horse_name": "ウマA",
                "ai_win_prob": 0.15,
                "feature_dict": {
                    "win_odds": 5.6,
                    "feat_popularity": 1,
                    "running_style": "front",
                },
            },
            {
                "horse_name": "ウマB",
                "ai_win_prob": 0.10,
                "feature_dict": None,
            },
        ],
    }
}

PRED_V1 = {
    "202606030402": {
        "race_id": "202606030402",
        "race_name": "３歳未勝利 出馬表 | 2026年4月5日 中山2R レース情報(JRA) - netkeiba",
        "analysis_date": "20260405",
        "horses": [
            {
                "horse_name": "ウマC",
                "ai_win_prob": 0.12,
                "win_odds": None,
                "popularity": None,
            }
        ],
    }
}

BETS = {
    "202606030401": [
        {
            "bet_type": "tansho",
            "bet_type_label": "単勝",
            "bet_combination": ["ウマA"],
            "stake_amount": 100,
        }
    ]
}

OUTCOMES_HIT = {
    "202606030401": [
        {
            "bet_type": "tansho",
            "bet_type_label": "単勝",
            "bet_combination": ["ウマA"],
            "stake": 100,
            "hit": True,
            "payout": 560,
            "roi": 5.6,
        }
    ]
}

OUTCOMES_MISS = {
    "202606030401": [
        {
            "bet_type": "tansho",
            "bet_type_label": "単勝",
            "bet_combination": ["ウマA"],
            "stake": 100,
            "hit": False,
            "payout": 0,
            "roi": 0.0,
        }
    ]
}


def _mock_load(preds, bets, outcomes):
    """_load_json を差し替えるヘルパー。"""
    def side_effect(path):
        if "predictions" in path:
            return preds
        if "suggestions" in path:
            return bets
        if "outcomes" in path:
            return outcomes
        return {}
    return side_effect


# ── テスト ────────────────────────────────────────────────────
def test_load_today_races_filters_by_date():
    """analysis_date が一致するレースだけ返す。"""
    preds = {**PRED_V2, **PRED_V1}
    with patch.object(dl, "_load_json", side_effect=_mock_load(preds, {}, {})):
        races = dl.load_races_for_date("20260405")
    assert len(races) == 2
    ids = {r["race_id"] for r in races}
    assert "202606030401" in ids
    assert "202606030402" in ids


def test_kpi_calculation():
    """的中率・ROI・投資額・回収額が正確に計算される。"""
    with patch.object(dl, "_load_json", side_effect=_mock_load(PRED_V2, BETS, OUTCOMES_HIT)):
        races = dl.load_races_for_date("20260405")
    kpi = dl.calc_kpi(races)
    assert kpi["total_stake"] == 100
    assert kpi["total_payout"] == 560
    assert kpi["roi"] == 560.0
    assert kpi["hit_count"] == 1
    assert kpi["total_bets"] == 1


def test_bet_type_breakdown():
    """券種別集計が正確に計算される。"""
    with patch.object(dl, "_load_json", side_effect=_mock_load(PRED_V2, BETS, OUTCOMES_HIT)):
        races = dl.load_races_for_date("20260405")
    rows = dl.calc_kpi_by_bet_type(races)
    tansho = next(r for r in rows if r["bet_type"] == "tansho")
    assert tansho["count"] == 1
    assert tansho["hit"] == 1
    assert tansho["hit_rate"] == 100.0
    assert tansho["roi"] == 560.0


def test_horse_odds_fallback_v1():
    """v1スキーマ（feature_dict なし）で win_odds と running_style が None になる。"""
    with patch.object(dl, "_load_json", side_effect=_mock_load(PRED_V1, {}, {})):
        races = dl.load_races_for_date("20260405")
    horse = races[0]["horses"][0]
    assert horse["win_odds"] is None
    assert horse["running_style"] is None


def test_horse_running_style_mapped_v2():
    """v2スキーマで running_style が日本語にマッピングされる。"""
    with patch.object(dl, "_load_json", side_effect=_mock_load(PRED_V2, {}, {})):
        races = dl.load_races_for_date("20260405")
    horse_a = next(h for h in races[0]["horses"] if h["horse_name"] == "ウマA")
    assert horse_a["running_style"] == "逃げ"


def test_race_status_prerace():
    """outcomes なし・start_datetime が未来 → status == 'prerace'。"""
    future_pred = {
        "202606030401": {
            **PRED_V2["202606030401"],
            "start_datetime": (datetime.now() + timedelta(hours=2)).isoformat(),
        }
    }
    with patch.object(dl, "_load_json", side_effect=_mock_load(future_pred, {}, {})):
        races = dl.load_races_for_date("20260405")
    assert races[0]["status"] == "prerace"


def test_race_status_result():
    """outcomes あり → status == 'result'。"""
    with patch.object(dl, "_load_json", side_effect=_mock_load(PRED_V2, BETS, OUTCOMES_HIT)):
        races = dl.load_races_for_date("20260405")
    assert races[0]["status"] == "result"


def test_date_list_descending():
    """get_available_dates() が降順で返る。"""
    preds = {
        "A": {"analysis_date": "20260403"},
        "B": {"analysis_date": "20260405"},
        "C": {"analysis_date": "20260404"},
    }
    with patch.object(dl, "_load_json", return_value=preds):
        dates = dl.get_available_dates()
    assert dates == ["20260405", "20260404", "20260403"]
```

- [ ] **Step 2: テストを実行して全件 FAIL を確認する**

```bash
cd /Users/ryokarahashi/keiba_ai
python3 -m pytest tests/test_dashboard_loader.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'dashboard_loader'`（dashboard_loader.py 未作成のため）

---

## Task 2: dashboard_loader.py（実装）

**Files:**
- Create: `dashboard_loader.py`
- Test: `tests/test_dashboard_loader.py`

- [ ] **Step 1: dashboard_loader.py を作成する**

```python
# dashboard_loader.py
"""
パイプラインダッシュボード用データ読み込み・集計モジュール。
Streamlit に非依存。pipeline_*.json を読み取り専用で参照する。
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_PREDICTIONS_FILE     = os.path.join(_HERE, "pipeline_predictions.json")
_BET_SUGGESTIONS_FILE = os.path.join(_HERE, "pipeline_bet_suggestions.json")
_BET_OUTCOMES_FILE    = os.path.join(_HERE, "pipeline_bet_outcomes.json")

# running_style（英語キー）→ 日本語表示
STYLE_MAP: Dict[str, str] = {
    "front":   "逃げ",
    "stalker": "先行",
    "mid":     "差し",
    "closer":  "追込",
    "unknown": "不明",
}


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _parse_venue_race_number(race_name: str) -> Tuple[str, str]:
    """
    race_name から会場名と回次を抽出する。
    例: "３歳未勝利 出馬表 | 2026年4月5日 中山1R レース情報" → ("中山", "1R")
    """
    m = re.search(r"([^\d\s|]+)(\d+R)", race_name)
    if m:
        return m.group(1), m.group(2)
    return "", ""


def _race_status(start_datetime: Optional[str], outcomes: List[Dict]) -> str:
    """
    レースの表示ステータスを返す。
    - outcomes あり → "result"
    - start_datetime が 30分以上前 → "awaiting"（evaluate 待ち）
    - それ以外 → "prerace"
    """
    if outcomes:
        return "result"
    if start_datetime:
        try:
            t = datetime.fromisoformat(start_datetime)
            if datetime.now() > t + timedelta(minutes=30):
                return "awaiting"
        except ValueError:
            pass
    return "prerace"


def _build_horse_row(horse: Dict) -> Dict:
    """
    v1 / v2 スキーマ両対応で馬情報を正規化する。
    v2: horse["feature_dict"]["win_odds"] / ["running_style"]
    v1: horse["win_odds"] = null
    """
    fd = horse.get("feature_dict") or {}
    win_odds_raw       = fd.get("win_odds") if fd else horse.get("win_odds")
    running_style_raw  = fd.get("running_style") if fd else horse.get("running_style")
    popularity_raw     = fd.get("feat_popularity") if fd else horse.get("popularity")
    return {
        "horse_name":    horse.get("horse_name", ""),
        "ai_win_prob":   horse.get("ai_win_prob"),
        "win_odds":      win_odds_raw,
        "popularity":    int(popularity_raw) if popularity_raw is not None else None,
        "running_style": STYLE_MAP.get(running_style_raw or "", None),
    }


def get_available_dates() -> List[str]:
    """
    pipeline_predictions.json に存在する analysis_date の一覧を降順で返す。
    """
    preds = _load_json(_PREDICTIONS_FILE)
    dates = sorted(
        {v.get("analysis_date", "") for v in preds.values() if v.get("analysis_date")},
        reverse=True,
    )
    return dates


def load_races_for_date(date_str: str) -> List[Dict]:
    """
    指定日のレース一覧を predictions + bets + outcomes を合体して返す。

    Returns
    -------
    List[Dict] — 各要素のキー:
        race_id, race_name, venue, race_number,
        start_time, start_datetime, status,
        horses, bets, outcomes
    """
    preds    = _load_json(_PREDICTIONS_FILE)
    bets_all = _load_json(_BET_SUGGESTIONS_FILE)
    outs_all = _load_json(_BET_OUTCOMES_FILE)

    races: List[Dict] = []
    for race_id, pred in preds.items():
        if pred.get("analysis_date") != date_str:
            continue

        race_name     = pred.get("race_name", "")
        venue, r_num  = _parse_venue_race_number(race_name)
        start_dt      = pred.get("start_datetime")
        start_time    = pred.get("start_time")
        outcomes      = outs_all.get(race_id) or []
        bets          = bets_all.get(race_id) or []
        horses        = [_build_horse_row(h) for h in (pred.get("horses") or [])]
        horses.sort(key=lambda h: -(h["ai_win_prob"] or 0))

        races.append(
            {
                "race_id":        race_id,
                "race_name":      race_name,
                "venue":          venue,
                "race_number":    r_num,
                "start_time":     start_time,
                "start_datetime": start_dt,
                "status":         _race_status(start_dt, outcomes),
                "horses":         horses,
                "bets":           bets,
                "outcomes":       outcomes,
            }
        )

    # 発走時刻順（不明は末尾）
    races.sort(key=lambda r: r.get("start_time") or "99:99")
    return races


def calc_kpi(races: List[Dict]) -> Dict:
    """
    レース一覧から全体KPIを計算する。

    Returns
    -------
    {total_stake, total_payout, roi, hit_count, total_bets}
    """
    total_stake = total_payout = hit_count = total_bets = 0

    for race in races:
        for o in race.get("outcomes", []):
            stake         = o.get("stake", 100)
            total_stake  += stake
            total_payout += o.get("payout", 0)
            total_bets   += 1
            if o.get("hit"):
                hit_count += 1

    roi = round(total_payout / total_stake * 100, 1) if total_stake > 0 else 0.0
    return {
        "total_stake":  total_stake,
        "total_payout": total_payout,
        "roi":          roi,
        "hit_count":    hit_count,
        "total_bets":   total_bets,
    }


def calc_kpi_by_bet_type(races: List[Dict]) -> List[Dict]:
    """
    券種別KPIを計算する。

    Returns
    -------
    List[Dict] — 各要素: {bet_type, label, count, hit, hit_rate, roi}
    """
    stats: Dict[str, Dict] = defaultdict(
        lambda: {"label": "", "count": 0, "hit": 0, "stake": 0, "payout": 0}
    )

    for race in races:
        for o in race.get("outcomes", []):
            bt = o.get("bet_type", "other")
            s  = stats[bt]
            s["label"]   = o.get("bet_type_label", bt)
            s["count"]  += 1
            s["stake"]  += o.get("stake", 100)
            s["payout"] += o.get("payout", 0)
            if o.get("hit"):
                s["hit"] += 1

    rows = []
    for bt, s in stats.items():
        hit_rate = round(s["hit"] / s["count"] * 100, 1) if s["count"] > 0 else 0.0
        roi      = round(s["payout"] / s["stake"] * 100, 1) if s["stake"] > 0 else 0.0
        rows.append(
            {
                "bet_type": bt,
                "label":    s["label"],
                "count":    s["count"],
                "hit":      s["hit"],
                "hit_rate": hit_rate,
                "roi":      roi,
            }
        )
    rows.sort(key=lambda r: r["bet_type"])
    return rows
```

- [ ] **Step 2: テストを実行して全件 PASS を確認する**

```bash
cd /Users/ryokarahashi/keiba_ai
python3 -m pytest tests/test_dashboard_loader.py -v
```

Expected:
```
tests/test_dashboard_loader.py::test_load_today_races_filters_by_date PASSED
tests/test_dashboard_loader.py::test_kpi_calculation PASSED
tests/test_dashboard_loader.py::test_bet_type_breakdown PASSED
tests/test_dashboard_loader.py::test_horse_odds_fallback_v1 PASSED
tests/test_dashboard_loader.py::test_horse_running_style_mapped_v2 PASSED
tests/test_dashboard_loader.py::test_race_status_prerace PASSED
tests/test_dashboard_loader.py::test_race_status_result PASSED
tests/test_dashboard_loader.py::test_date_list_descending PASSED
7 passed
```

- [ ] **Step 3: 全既存テストも通ることを確認する**

```bash
python3 -m pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: 全 PASSED（既存テストに影響なし）

- [ ] **Step 4: コミットする**

```bash
git add dashboard_loader.py tests/test_dashboard_loader.py
git commit -m "feat: add dashboard_loader with data loading and KPI calc"
```

---

## Task 3: keiba_app.py — 完全書き直し

**Files:**
- Modify: `keiba_app.py`（完全書き直し）

`keiba_app.py` は 3702 行の URL 入力型アプリ。全て置き換える。
データ取得・集計は全て `dashboard_loader` に委譲し、UI 描画のみ担う。

- [ ] **Step 1: keiba_app.py を以下の内容で書き直す**

```python
# keiba_app.py
"""
競馬AI パイプライン ダッシュボード
weekend_pipeline.sh が生成したデータを読み取り専用で表示する。
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import pandas as pd
import streamlit as st

import dashboard_loader as dl

# ──────────────────────────────────────────────────────────────
# 定数
# ──────────────────────────────────────────────────────────────
STATUS_LABEL: Dict[str, str] = {
    "prerace":  "🕐 発走前",
    "awaiting": "⏳ 集計待ち",
    "result":   "✅ 結果済み",
}

BET_TYPE_ORDER = [
    "tansho", "fukusho", "wide",
    "umaren", "umatan",
    "sanrenpuku", "sanrenpuku_ai", "sanrentan",
]


# ──────────────────────────────────────────────────────────────
# 共通ウィジェット
# ──────────────────────────────────────────────────────────────
def _render_kpi(kpi: Dict) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("投資額",   f"¥{kpi['total_stake']:,}")
    col2.metric("回収額",   f"¥{kpi['total_payout']:,}")
    col3.metric("ROI",      f"{kpi['roi']}%")
    col4.metric("的中",     f"{kpi['hit_count']} / {kpi['total_bets']}")


def _render_bet_type_table(races: List[Dict]) -> None:
    rows = dl.calc_kpi_by_bet_type(races)
    if not rows:
        st.caption("まだ結果データがありません。")
        return
    df = pd.DataFrame(rows)[["label", "count", "hit", "hit_rate", "roi"]]
    df.columns = ["券種", "買い目数", "的中", "的中率(%)", "ROI(%)"]
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_race_cards(races: List[Dict]) -> None:
    for race in races:
        venue      = race["venue"]
        r_num      = race["race_number"]
        start_time = race["start_time"] or "??:??"
        status_lbl = STATUS_LABEL.get(race["status"], race["status"])
        label      = f"🏇 {venue}{r_num}  {start_time}発走  {status_lbl}"

        with st.expander(label, expanded=False):
            # 買い目
            bets = race["bets"]
            if bets:
                st.markdown("**買い目**")
                parts = []
                for b in bets:
                    combo = "・".join(b.get("bet_combination") or [])
                    stake = b.get("stake_amount", 100)
                    parts.append(f"{b.get('bet_type_label', '')} {combo} ¥{stake}")
                st.caption("  |  ".join(parts))
            else:
                st.caption("買い目なし")

            # 結果
            outcomes = race["outcomes"]
            if race["status"] == "result" and outcomes:
                total_stake  = sum(o.get("stake", 100) for o in outcomes)
                total_payout = sum(o.get("payout", 0) for o in outcomes)
                profit       = total_payout - total_stake
                roi          = round(total_payout / total_stake * 100, 1) if total_stake > 0 else 0.0
                any_hit      = any(o.get("hit") for o in outcomes)
                icon         = "✅" if any_hit else "❌"
                label_hit    = "的中" if any_hit else "外れ"
                st.markdown(
                    f"**結果:** {icon} {label_hit}  "
                    f"損益: ¥{profit:+,}  ROI: {roi}%"
                )

            # 馬別 AI 予測
            horses = race["horses"]
            if horses:
                st.markdown("**馬別AI予測**")
                rows = []
                for h in horses:
                    rows.append(
                        {
                            "馬名":   h["horse_name"],
                            "AI勝率": f"{h['ai_win_prob'] * 100:.1f}%" if h["ai_win_prob"] is not None else "-",
                            "オッズ": f"{h['win_odds']:.1f}" if h["win_odds"] is not None else "未取得",
                            "人気":   str(h["popularity"]) if h["popularity"] is not None else "未取得",
                            "脚質":   h["running_style"] or "未取得",
                        }
                    )
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────
# Tab 1「当日」— 60秒自動更新
# ──────────────────────────────────────────────────────────────
@st.fragment(run_every=60)
def _tab_today() -> None:
    today_str = datetime.now().strftime("%Y%m%d")
    races     = dl.load_races_for_date(today_str)

    st.caption(f"最終更新: {datetime.now().strftime('%H:%M:%S')}  （60秒ごとに自動更新）")

    if not races:
        st.info(
            "本日のレースデータがまだありません。\n"
            "`bash weekend_pipeline.sh` を実行してください。"
        )
        return

    # KPI
    kpi = dl.calc_kpi(races)
    _render_kpi(kpi)

    # 券種別集計
    st.subheader("券種別集計")
    _render_bet_type_table(races)

    # レースカード
    st.subheader(f"レース一覧（{len(races)} レース）")
    _render_race_cards(races)


# ──────────────────────────────────────────────────────────────
# Tab 2「履歴」
# ──────────────────────────────────────────────────────────────
def _tab_history() -> None:
    dates = dl.get_available_dates()

    if not dates:
        st.info("結果データがありません。`--evaluate` を実行してください。")
        return

    selected = st.selectbox("日付を選択", dates, index=0)

    # 全期間累計 KPI
    all_races: List[Dict] = []
    for d in dates:
        all_races.extend(dl.load_races_for_date(d))

    kpi_all = dl.calc_kpi(all_races)
    st.subheader("全期間累計")
    _render_kpi(kpi_all)

    # 券種別累計
    st.subheader("券種別累計")
    _render_bet_type_table(all_races)

    # ROI 推移グラフ
    roi_rows = []
    for d in sorted(dates):
        day_races = dl.load_races_for_date(d)
        k = dl.calc_kpi(day_races)
        if k["total_stake"] > 0:
            roi_rows.append({"日付": d, "ROI(%)": k["roi"]})

    if roi_rows:
        st.subheader("ROI 推移")
        df_roi = pd.DataFrame(roi_rows).set_index("日付")
        st.line_chart(df_roi)

    # 選択日のレース
    st.subheader(f"{selected} のレース")
    day_races = dl.load_races_for_date(selected)
    if day_races:
        kpi_day = dl.calc_kpi(day_races)
        _render_kpi(kpi_day)
        _render_race_cards(day_races)
    else:
        st.info("この日のデータがありません。")


# ──────────────────────────────────────────────────────────────
# エントリポイント
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="競馬AI ダッシュボード", layout="wide")
st.title("🏇 競馬AI パイプライン ダッシュボード")

tab_today, tab_history = st.tabs(["📅 当日", "📊 履歴"])

with tab_today:
    _tab_today()

with tab_history:
    _tab_history()
```

- [ ] **Step 2: Streamlit を起動して画面を確認する**

既存プロセスを停止してから再起動：

```bash
pkill -f "streamlit run keiba_app" 2>/dev/null || true
sleep 1
streamlit run keiba_app.py >> streamlit.log 2>&1 &
echo "PID: $!"
```

ブラウザで `http://localhost:8501` を開き、以下を確認する：
- 「📅 当日」タブが表示される
- 当日データがない場合 "本日のレースデータがまだありません" が表示される
- 「📊 履歴」タブに切り替えられる
- エラーが出ない

- [ ] **Step 3: エラーがないことをログで確認する**

```bash
sleep 3 && tail -20 streamlit.log
```

Expected: `You can now view your Streamlit app in your browser` が含まれ、Traceback なし

- [ ] **Step 4: コミットする**

```bash
git add keiba_app.py
git commit -m "feat: replace keiba_app.py with pipeline dashboard (2-tab read-only)"
```

---

## Task 4: 動作確認（実データ）

**Files:** なし（確認のみ）

今日 (`20260405`) の pipeline データが存在するため、実際の表示を確認する。

- [ ] **Step 1: 当日タブでデータ表示を確認する**

ブラウザで `http://localhost:8501` を開き「📅 当日」タブを確認：

1. レースカードが発走時刻順に並んでいる
2. 各レースを展開すると馬名・AI勝率・オッズ（未取得 or 実値）・脚質が表示される
3. `pipeline_bet_outcomes.json` がない場合は全レース「🕐 発走前」or「⏳ 集計待ち」となる
4. KPI は 投資額 ¥0 / 回収額 ¥0 / ROI 0% （outcomes がないため）

- [ ] **Step 2: 履歴タブでデータ表示を確認する**

「📊 履歴」タブを開き：
1. 日付ドロップダウンに `20260405` が表示される
2. `20260405` を選択するとレースカードが表示される
3. エラーが出ない

- [ ] **Step 3: 全テストが通ることを確認する**

```bash
python3 -m pytest tests/ -v --tb=short 2>&1 | tail -10
```

Expected: 全 PASSED

- [ ] **Step 4: 最終コミット**

```bash
git add -p  # 変更があれば
git commit -m "chore: verify pipeline dashboard with real data" 2>/dev/null || echo "no changes"
```
