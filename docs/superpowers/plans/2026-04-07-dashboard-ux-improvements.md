# Dashboard UX 改善 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** レースカードに常時サマリー・荒れスコア・激熱バッジを追加し、クリック不要で全レース状況を把握できるダッシュボードに刷新する。

**Architecture:** `dashboard_loader.py` に `calc_upset_score()` / `calc_hot_bets()` を追加して `load_races_for_date()` の出力を拡張。`keiba_app.py` の `_render_race_cards()` を「常時 HTML サマリー行 + `st.expander` 詳細」の2層構造に書き直す。

**Tech Stack:** Python 3.14, Streamlit 1.55, math (標準ライブラリ), pytest

---

## ファイル構成

| ファイル | 変更 |
|---|---|
| `dashboard_loader.py` | `calc_upset_score()` / `calc_hot_bets()` 追加。`load_races_for_date()` の各レース dict に `upset_score`, `upset_label`, `upset_color`, `hot_bets` を追加 |
| `keiba_app.py` | `MARKS` / `MARK_COLORS` 定数追加。`_render_race_summary()` 新規追加。`_render_race_cards()` を2層構造に書き直し |
| `tests/test_dashboard_loader.py` | 6テスト追加（既存143件は変更なし） |

---

## Task 1: `calc_upset_score()` — 荒れスコア計算関数

**Files:**
- Modify: `dashboard_loader.py`
- Modify: `tests/test_dashboard_loader.py`

- [ ] **Step 1: 失敗するテストを追加する**

`tests/test_dashboard_loader.py` の末尾に追記:

```python
# ── Task 1: calc_upset_score ──────────────────────────────────

def test_upset_score_concentrated():
    """1頭が勝率90%を占める → エントロピー低 → 低スコア（堅い or やや堅い）"""
    horses = [
        {"horse_name": "A", "ai_win_prob": 0.9,  "win_odds": 1.2},
        {"horse_name": "B", "ai_win_prob": 0.05, "win_odds": 20.0},
        {"horse_name": "C", "ai_win_prob": 0.05, "win_odds": 20.0},
    ]
    result = dl.calc_upset_score(horses)
    assert result["score"] < 40
    assert result["label"] in ("堅い", "やや堅い")
    assert "color" in result


def test_upset_score_uniform():
    """10頭均等分布 → エントロピー最大 → 高スコア（やや荒れ or 荒れ）"""
    horses = [
        {"horse_name": str(i), "ai_win_prob": 0.1, "win_odds": 10.0}
        for i in range(10)
    ]
    result = dl.calc_upset_score(horses)
    assert result["score"] >= 55
    assert result["label"] in ("やや荒れ", "荒れ", "中間")


def test_upset_score_no_odds():
    """オッズが全馬 None → エントロピーのみで計算、戻り値のキーが揃っている"""
    horses = [
        {"horse_name": "A", "ai_win_prob": 0.5, "win_odds": None},
        {"horse_name": "B", "ai_win_prob": 0.3, "win_odds": None},
        {"horse_name": "C", "ai_win_prob": 0.2, "win_odds": None},
    ]
    result = dl.calc_upset_score(horses)
    assert 0 <= result["score"] <= 100
    assert result["label"] in ("堅い", "やや堅い", "中間", "やや荒れ", "荒れ")
    assert result["color"].startswith("#")
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
cd /Users/ryokarahashi/keiba_ai
python3 -m pytest tests/test_dashboard_loader.py::test_upset_score_concentrated -v 2>&1 | head -15
```

Expected: `AttributeError: module 'dashboard_loader' has no attribute 'calc_upset_score'`

- [ ] **Step 3: `calc_upset_score()` を `dashboard_loader.py` に追加する**

`dashboard_loader.py` の `get_available_dates()` 関数の直前（既存の `_build_horse_row` の後）に追記:

```python
# ──────────────────────────────────────────────────────────────
# 荒れスコア
# ──────────────────────────────────────────────────────────────

_UPSET_LEVELS = [
    (30,  "堅い",     "#27ae60"),
    (45,  "やや堅い", "#8bc34a"),
    (60,  "中間",     "#ff9800"),
    (75,  "やや荒れ", "#e64a19"),
    (101, "荒れ",     "#c0392b"),
]


def calc_upset_score(horses: List[Dict]) -> Dict[str, Any]:
    """
    馬リストから荒れスコア (0–100 整数) を計算する。

    Returns
    -------
    {"score": int, "label": str, "color": str}
    """
    import math

    probs = [float(h.get("ai_win_prob") or 0.0) for h in horses]
    probs = [p for p in probs if p > 0.0]

    if not probs:
        return {"score": 50, "label": "中間", "color": "#ff9800"}

    # Shannon エントロピー → 0–100 に正規化
    total = sum(probs)
    norm_p = [p / total for p in probs]
    entropy = -sum(p * math.log2(p) for p in norm_p if p > 0.0)
    max_entropy = math.log2(len(norm_p)) if len(norm_p) > 1 else 1.0
    entropy_score = (entropy / max_entropy * 100.0) if max_entropy > 0.0 else 0.0

    # 1番人気オッズ → 0–100 に変換（最大 30 倍で cap）
    odds_list = [
        float(h.get("win_odds"))
        for h in horses
        if h.get("win_odds") is not None
    ]
    if odds_list:
        top_odds = min(odds_list)          # 最低オッズ = 1番人気
        odds_score = min(top_odds, 30.0) / 30.0 * 100.0
        w_e, w_o = 0.6, 0.4
    else:
        odds_score = 0.0
        w_e, w_o = 1.0, 0.0               # オッズなし → エントロピーのみ

    score = max(0, min(100, round(w_e * entropy_score + w_o * odds_score)))

    for threshold, label, color in _UPSET_LEVELS:
        if score < threshold:
            return {"score": score, "label": label, "color": color}

    return {"score": score, "label": "荒れ", "color": "#c0392b"}
```

- [ ] **Step 4: テストが通ることを確認**

```bash
python3 -m pytest tests/test_dashboard_loader.py::test_upset_score_concentrated \
  tests/test_dashboard_loader.py::test_upset_score_uniform \
  tests/test_dashboard_loader.py::test_upset_score_no_odds -v
```

Expected: `3 passed`

- [ ] **Step 5: コミット**

```bash
git add dashboard_loader.py tests/test_dashboard_loader.py
git commit -m "feat: add calc_upset_score() to dashboard_loader"
```

---

## Task 2: `calc_hot_bets()` — 激熱買い目判定関数

**Files:**
- Modify: `dashboard_loader.py`
- Modify: `tests/test_dashboard_loader.py`

- [ ] **Step 1: 失敗するテストを追加する**

`tests/test_dashboard_loader.py` の末尾に追記:

```python
# ── Task 2: calc_hot_bets ─────────────────────────────────────

def test_hot_bets_threshold():
    """confidence 0.74 → 激熱なし / 0.75 → 激熱あり（境界値）"""
    bet_below = [{"bet_type": "tansho", "confidence": 0.74, "expected_value": None}]
    bet_at    = [{"bet_type": "tansho", "confidence": 0.75, "expected_value": None}]
    assert dl.calc_hot_bets(bet_below) == []
    assert len(dl.calc_hot_bets(bet_at)) == 1


def test_hot_bets_ev_filter():
    """confidence >= 0.75 でも expected_value <= 1.1 なら除外"""
    bets = [{"bet_type": "tansho", "confidence": 0.80, "expected_value": 1.05}]
    assert dl.calc_hot_bets(bets) == []


def test_hot_bets_ev_none_passes():
    """expected_value が None なら EV 条件をスキップして通過"""
    bets = [{"bet_type": "tansho", "confidence": 0.80, "expected_value": None}]
    assert len(dl.calc_hot_bets(bets)) == 1
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
python3 -m pytest tests/test_dashboard_loader.py::test_hot_bets_threshold -v 2>&1 | head -15
```

Expected: `AttributeError: module 'dashboard_loader' has no attribute 'calc_hot_bets'`

- [ ] **Step 3: `calc_hot_bets()` を `dashboard_loader.py` に追加する**

`calc_upset_score()` の直後に追記:

```python
def calc_hot_bets(bets: List[Dict]) -> List[Dict]:
    """
    confidence >= 0.75 かつ expected_value > 1.1（None は条件スキップ）
    の買い目リストを返す。

    Parameters
    ----------
    bets : pipeline_bet_suggestions.json から読んだ買い目リスト

    Returns
    -------
    条件を満たした bet dict のリスト
    """
    result: List[Dict] = []
    for bet in bets:
        conf = float(bet.get("confidence") or 0.0)
        if conf < 0.75:
            continue
        ev = bet.get("expected_value")
        if ev is not None and float(ev) <= 1.1:
            continue
        result.append(bet)
    return result
```

- [ ] **Step 4: テストが通ることを確認**

```bash
python3 -m pytest tests/test_dashboard_loader.py::test_hot_bets_threshold \
  tests/test_dashboard_loader.py::test_hot_bets_ev_filter \
  tests/test_dashboard_loader.py::test_hot_bets_ev_none_passes -v
```

Expected: `3 passed`

- [ ] **Step 5: コミット**

```bash
git add dashboard_loader.py tests/test_dashboard_loader.py
git commit -m "feat: add calc_hot_bets() to dashboard_loader"
```

---

## Task 3: `load_races_for_date()` の出力拡張

**Files:**
- Modify: `dashboard_loader.py`
- Modify: `tests/test_dashboard_loader.py`

- [ ] **Step 1: 失敗するテストを追加する**

`tests/test_dashboard_loader.py` の末尾に追記:

```python
# ── Task 3: load_races_for_date 出力拡張 ──────────────────────

def test_load_races_includes_upset_and_hot():
    """load_races_for_date の出力に upset_score / upset_label / upset_color / hot_bets が含まれる"""
    preds = {
        "202606030401": {
            "race_id": "202606030401",
            "race_name": "テスト | 2026年4月5日 中山1R レース情報",
            "analysis_date": "20260405",
            "horses": [
                {"horse_name": "A", "ai_win_prob": 0.4,
                 "feature_dict": {"win_odds": 3.0, "feat_popularity": 1, "running_style": "front"}},
                {"horse_name": "B", "ai_win_prob": 0.3,
                 "feature_dict": {"win_odds": 5.0, "feat_popularity": 2, "running_style": "stalker"}},
            ],
        }
    }
    bets = {
        "202606030401": [
            {"bet_type": "tansho", "confidence": 0.80, "expected_value": None,
             "bet_combination": ["A"], "stake_amount": 100}
        ]
    }
    with patch.object(dl, "_load_json", side_effect=_mock_load(preds, bets, {})):
        races = dl.load_races_for_date("20260405")

    assert len(races) == 1
    race = races[0]
    assert "upset_score" in race
    assert "upset_label" in race
    assert "upset_color" in race
    assert "hot_bets" in race
    assert isinstance(race["upset_score"], int)
    assert 0 <= race["upset_score"] <= 100
    assert race["upset_label"] in ("堅い", "やや堅い", "中間", "やや荒れ", "荒れ")
    assert race["upset_color"].startswith("#")
    assert len(race["hot_bets"]) == 1
```

- [ ] **Step 2: テストが失敗することを確認**

```bash
python3 -m pytest tests/test_dashboard_loader.py::test_load_races_includes_upset_and_hot -v 2>&1 | head -15
```

Expected: `AssertionError` (upset_score キーが存在しない)

- [ ] **Step 3: `load_races_for_date()` を拡張する**

`dashboard_loader.py` の `load_races_for_date()` 内、`horses.sort(...)` の直後に2行追加し、`races.append({...})` の dict に4フィールドを追加する。

変更前（`races.append` 直前の部分）:
```python
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
```

変更後:
```python
        horses        = [_build_horse_row(h) for h in (pred.get("horses") or [])]
        horses.sort(key=lambda h: -(h["ai_win_prob"] or 0))

        upset = calc_upset_score(horses)
        hot   = calc_hot_bets(bets)

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
                "upset_score":    upset["score"],
                "upset_label":    upset["label"],
                "upset_color":    upset["color"],
                "hot_bets":       hot,
            }
        )
```

- [ ] **Step 4: テストが通ることを確認（既存テストも含む）**

```bash
python3 -m pytest tests/test_dashboard_loader.py -v
```

Expected: 全件 PASS（既存10件 + 新規7件 = 17件）

- [ ] **Step 5: コミット**

```bash
git add dashboard_loader.py tests/test_dashboard_loader.py
git commit -m "feat: extend load_races_for_date with upset_score and hot_bets"
```

---

## Task 4: `keiba_app.py` — サマリー行＋2層カード構造

**Files:**
- Modify: `keiba_app.py`

- [ ] **Step 1: 定数と `_render_race_summary()` を追加する**

`keiba_app.py` の `STATUS_LABEL` 定数の直後（既存定数ブロックの末尾）に追記:

```python
# 印と色
_MARKS = ["◎", "○", "▲"]
_MARK_COLORS = {"◎": "#e74c3c", "○": "#e67e22", "▲": "#27ae60"}
```

次に `_render_kpi()` の直前に新関数を追加:

```python
def _render_race_summary(race: Dict) -> None:
    """レースカードの常時表示サマリー行を HTML で描画する。"""
    venue      = race.get("venue") or ""
    r_num      = race.get("race_number") or ""
    start_time = race.get("start_time") or "??:??"
    status     = race.get("status", "prerace")

    # ── ステータスバッジ ──
    if status == "result" and race.get("outcomes"):
        any_hit = any(o.get("hit") for o in race["outcomes"])
        if any_hit:
            status_html = (
                '<span style="background:#1d6a27;color:#6fcf97;font-size:11px;'
                'padding:2px 8px;border-radius:10px">✅ 的中</span>'
            )
        else:
            status_html = (
                '<span style="background:#4a1a1a;color:#e57373;font-size:11px;'
                'padding:2px 8px;border-radius:10px">❌ 外れ</span>'
            )
    elif status == "awaiting":
        status_html = (
            '<span style="background:#4a3f00;color:#f1c40f;font-size:11px;'
            'padding:2px 8px;border-radius:10px">⏳ 集計待ち</span>'
        )
    else:
        status_html = (
            '<span style="background:#1a3a5c;color:#7fb3d3;font-size:11px;'
            'padding:2px 8px;border-radius:10px">🕐 発走前</span>'
        )

    # ── 上位3頭の印 ──
    marks_parts = []
    for i, h in enumerate(race.get("horses", [])[:3]):
        mark  = _MARKS[i]
        color = _MARK_COLORS[mark]
        prob  = h.get("ai_win_prob")
        pstr  = f"{prob * 100:.1f}%" if prob is not None else "-"
        name  = h.get("horse_name", "")
        marks_parts.append(
            f'<span style="color:{color};font-size:12px;margin-right:10px">'
            f'{mark} {name} <b>{pstr}</b></span>'
        )
    marks_html = "".join(marks_parts)

    # ── 荒れスコアバッジ ──
    upset_label = race.get("upset_label", "")
    upset_color = race.get("upset_color", "#ff9800")
    upset_score = race.get("upset_score", "")
    upset_html = (
        f'<span style="background:{upset_color};color:#fff;font-size:11px;'
        f'padding:2px 8px;border-radius:10px">{upset_label} {upset_score}</span>'
    )

    # ── 激熱バッジ ──
    hot_bets = race.get("hot_bets") or []
    hot_html = ""
    if hot_bets:
        hot_html = (
            f'<span style="background:#c0392b;color:#fff;font-size:11px;'
            f'padding:2px 8px;border-radius:10px;margin-left:4px">🔥 {len(hot_bets)}件</span>'
        )

    html = f"""
<div style="background:#16213e;border-radius:6px;padding:10px 14px;
            margin-bottom:2px;display:flex;align-items:center;gap:10px;
            font-family:sans-serif;">
  <div style="background:#293174;color:#fff;font-size:13px;font-weight:bold;
              width:40px;height:40px;border-radius:6px;display:flex;
              align-items:center;justify-content:center;flex-shrink:0;">{r_num}</div>
  <div style="flex:1;min-width:0;">
    <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:5px;">
      <span style="color:#e0e0e0;font-weight:bold;font-size:13px">{venue}</span>
      <span style="color:#888;font-size:12px">{start_time}発走</span>
      {status_html}
    </div>
    <div>{marks_html}</div>
  </div>
  <div style="display:flex;flex-direction:column;align-items:flex-end;gap:4px;flex-shrink:0;">
    {hot_html}
    {upset_html}
  </div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)
```

- [ ] **Step 2: `_render_race_cards()` を2層構造に書き直す**

`keiba_app.py` の `_render_race_cards()` 関数全体を以下で置き換える:

```python
def _render_race_cards(races: List[Dict]) -> None:
    for race in races:
        # ── 常時表示サマリー行 ──
        _render_race_summary(race)

        # ── 折りたたみ詳細 ──
        venue = race.get("venue") or ""
        r_num = race.get("race_number") or ""
        with st.expander(f"▼ {venue}{r_num} 詳細を開く", expanded=False):
            bets     = race["bets"]
            outcomes = race["outcomes"]
            outcome_map = {
                (o.get("bet_type", ""), "・".join(o.get("bet_combination") or [])): o
                for o in outcomes
            } if outcomes else {}

            if bets:
                st.markdown("**買い目**")
                rows = []
                for b in bets:
                    combo_list = b.get("bet_combination") or []
                    combo_str  = "・".join(combo_list)
                    bet_label  = b.get("bet_type_label", "")
                    stake      = b.get("stake_amount", 100)
                    o = outcome_map.get((b.get("bet_type", ""), combo_str))
                    if o is not None:
                        hit_str    = "✅" if o.get("hit") else "❌"
                        payout     = o.get("payout", 0)
                        payout_str = f"¥{payout:,}" if o.get("hit") else "-"
                    else:
                        hit_str    = "-"
                        payout_str = "-"
                    rows.append({
                        "券種":       bet_label,
                        "組み合わせ": combo_str,
                        "投資":       f"¥{stake}",
                        "的中":       hit_str,
                        "払戻":       payout_str,
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.caption("買い目なし")

            # 結果サマリー
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
                    rows.append({
                        "馬名":   h["horse_name"],
                        "AI勝率": f"{h['ai_win_prob'] * 100:.1f}%" if h["ai_win_prob"] is not None else "-",
                        "オッズ": f"{h['win_odds']:.1f}" if h["win_odds"] is not None else "未取得",
                        "人気":   str(h["popularity"]) if h["popularity"] is not None else "未取得",
                        "脚質":   h["running_style"] or "未取得",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
```

- [ ] **Step 3: `_render_race_cards()` から不要になった `confidence_scorer` import を削除する**

変更前（`_render_race_cards` 関数の先頭にある行）:
```python
def _render_race_cards(races: List[Dict]) -> None:
    from confidence_scorer import compute_race_confidence
    for race in races:
```

この `from confidence_scorer import compute_race_confidence` 行は Step 2 の書き直しで既に含まれていないことを確認する（Step 2 のコードには含まれていない）。

- [ ] **Step 4: Streamlit 構文チェック**

```bash
python3 -c "import keiba_app" 2>&1 | grep -v "ScriptRunContext\|WARNING\|streamlit run"
```

Expected: エラーなし（空出力 or ScriptRunContext 警告のみ）

- [ ] **Step 5: コミット**

```bash
git add keiba_app.py
git commit -m "feat: add race card summary row with upset score and hot badge"
```

---

## Task 5: 統合確認

**Files:** なし（確認のみ）

- [ ] **Step 1: 全テストを通す**

```bash
python3 -m pytest tests/ -v --tb=short 2>&1 | tail -15
```

Expected: 全 PASSED（旧143件 + 新7件 = 150件）

- [ ] **Step 2: モジュール構文チェック**

```bash
python3 -c "import dashboard_loader; import pipeline_store; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: `--help` で CLI 正常動作確認**

```bash
python3 daily_pipeline.py --help 2>&1 | head -5
```

Expected: usage 行が表示される

- [ ] **Step 4: 最終コミット**

```bash
git add -A
git status
```

未コミットの変更がなければ完了。あれば:

```bash
git commit -m "chore: dashboard UX improvements complete"
```

---

## 完成チェックリスト

- [ ] `calc_upset_score()` — テスト3件 PASS
- [ ] `calc_hot_bets()` — テスト3件 PASS
- [ ] `load_races_for_date()` 出力に `upset_score` / `upset_label` / `upset_color` / `hot_bets`
- [ ] `_render_race_summary()` がサマリー行を HTML で描画
- [ ] `_render_race_cards()` が2層構造（サマリー + expander）
- [ ] 全テスト PASS（150件）
- [ ] Streamlit 起動エラーなし
