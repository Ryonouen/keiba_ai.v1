# ダッシュボード UX 改善 設計書

**日付:** 2026-04-07  
**参考:** https://keiba-noodds.onrender.com/  
**ゴール:** レースカードを「開かなくても瞬時に状況を把握できる」設計に刷新する

---

## 実装する3機能

### A. レースカード常時サマリー

**現状の問題:**  
`st.expander` のタイトル行にテキストのみ。クリックしないと上位馬も買い目状況も見えない。

**変更後の設計:**  
各レースカードを「常時表示のサマリー行 + 折りたたみ詳細」の2層構造にする。

サマリー行（常時表示）の構成要素（左→右）:

| 要素 | 内容 |
|------|------|
| 番号ボックス | R数（例: `2R`）。ステータスに応じて背景色変化 |
| 会場・発走時刻 | 例: `中山 10:30` |
| ステータスバッジ | `🕐 発走前` / `✅ 的中` / `❌ 外れ` / `⏳ 集計待ち` |
| 印＋上位3頭 | `◎ ウマA 32.1%  ○ ウマB 18.4%  ▲ ウマC 12.7%` |
| 荒れスコアバッジ | 例: `やや荒れ 52`（色付き） |
| 激熱バッジ | `🔥 2件`（条件を満たすレースのみ） |

**印の割り当て:**

| 印 | 条件 |
|----|------|
| ◎ | `ai_win_prob` 1位 |
| ○ | 2位 |
| ▲ | 3位 |

**実装方法:**  
`st.markdown(unsafe_allow_html=True)` でサマリー HTML を描画し、その直下に `st.expander("▼ 詳細", expanded=False)` で既存の買い目テーブル・馬別予測を表示。

---

### B. 荒れスコア（upset_score）

**算出式:**

```
entropy_score = normalize(shannon_entropy(ai_win_prob_list), 0, 100)
odds_score    = min(top_horse_win_odds, 30) / 30 * 100   # オッズなし時は 0
upset_score   = round(0.6 * entropy_score + 0.4 * odds_score)  # 0–100 整数
```

- `shannon_entropy(p)` = -Σ(p_i * log2(p_i))（0確率はスキップ）
- `normalize`: 最大エントロピー = log2(n_horses) を 100 にマップ
- オッズが全馬 None の場合: `odds_score = 0`（エントロピーのみ）

**5段階分類:**

| スコア | ラベル | バッジ色 |
|--------|--------|---------|
| 0–30 | 堅い | `#27ae60`（緑） |
| 30–45 | やや堅い | `#8bc34a`（黄緑） |
| 45–60 | 中間 | `#ff9800`（橙） |
| 60–75 | やや荒れ | `#e64a19`（赤橙） |
| 75+ | 荒れ | `#c0392b`（赤） |

**実装場所:** `dashboard_loader.py` に `calc_upset_score(horses: List[Dict]) -> Dict` を追加。  
`load_races_for_date()` の出力各レース dict に `upset_score: int` と `upset_label: str` を追加。

---

### C. 激熱バッジ（hot_bets）

**判定条件:**  
レースの `bets`（`pipeline_bet_suggestions.json` から読んだリスト）のうち、以下を満たす買い目を「激熱」とみなす:

- `confidence >= 0.75`
- かつ `expected_value > 1.1`（`expected_value` が `None` の場合はこの条件を無視）

**出力:**  
`dashboard_loader.calc_hot_bets(bets: List[Dict]) -> List[Dict]`  
条件を満たした bet dict のリストを返す。

**サマリー行への表示:**  
- 0件 → バッジなし
- 1件以上 → `🔥 N件` バッジ（赤背景、白文字）

**実装場所:** `dashboard_loader.py` に `calc_hot_bets()` を追加。  
`load_races_for_date()` の出力に `hot_bets: List[Dict]` を追加。

---

## アーキテクチャ概要

```
dashboard_loader.py
  calc_upset_score(horses)  → {score: int, label: str, color: str}
  calc_hot_bets(bets)       → List[Dict]
  load_races_for_date()     → ... + upset_score, upset_label, upset_color, hot_bets

keiba_app.py
  _render_race_summary(race)  → st.markdown(HTML) でサマリー行を描画（新規）
  _render_race_cards(races)   → summary + expander の2層構造に変更

tests/test_dashboard_loader.py
  test_upset_score_concentrated()   → 1頭独占 → 低スコア
  test_upset_score_uniform()        → 均等分布 → 高スコア
  test_upset_score_no_odds()        → オッズなし → エントロピーのみ
  test_hot_bets_threshold()         → confidence 0.75 境界値
  test_hot_bets_ev_filter()         → EV フィルタの動作確認
  test_load_races_includes_upset()  → load_races_for_date の出力にフィールドが含まれる
```

---

## 変更ファイル一覧

| ファイル | 変更種別 |
|--------|---------|
| `dashboard_loader.py` | 関数追加、`load_races_for_date()` 出力拡張 |
| `keiba_app.py` | `_render_race_cards()` 書き直し、`_render_race_summary()` 新規追加 |
| `tests/test_dashboard_loader.py` | テスト追加（6件） |

**新規ファイル:** なし  
**既存テスト:** 全143件に影響なし（追加のみ）

---

## 対象外（スコープ外）

- Streamlit テーマのグローバル CSS カスタマイズ
- モバイル対応レイアウトの最適化
- 履歴タブのカード表示変更（当日タブと同じ関数を使うため自動適用）
