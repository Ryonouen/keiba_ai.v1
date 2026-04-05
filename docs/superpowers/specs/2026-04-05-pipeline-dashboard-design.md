# パイプラインダッシュボード 設計ドキュメント

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this spec.

**Goal:** `keiba_app.py` を URL貼り付け型の手動分析ツールから、週末パイプラインの自動実行結果を表示する読み取り専用ダッシュボードに完全刷新する。

**背景:** `weekend_pipeline.sh` + `daily_pipeline.py` による全自動化が完成したため、Streamlit UI もパイプラインデータ（`pipeline_predictions.json` / `pipeline_bet_suggestions.json` / `pipeline_bet_outcomes.json`）を読むだけのダッシュボードに移行する。URL入力・手動操作は廃止。

---

## アーキテクチャ概要

### 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `keiba_app.py` | 完全書き直し（3702行 → 約500行）。pipeline_store のデータを読み取り表示するのみ |

### 変更なし

- `pipeline_store.py` — 既存の `load_*` 関数をそのまま利用
- `pipeline_predictions.json` / `pipeline_bet_suggestions.json` / `pipeline_bet_outcomes.json` — 既存スキーマをそのまま読む
- `daily_pipeline.py` / `weekend_pipeline.sh` — パイプライン側は変更なし

---

## データフロー

```
weekend_pipeline.sh
  └─ daily_pipeline.py --analyze     → pipeline_predictions.json (v2スキーマ)
  └─ daily_pipeline.py --watch-odds  → pipeline_predictions.json (オッズ・脚質更新)
  └─ daily_pipeline.py --evaluate    → pipeline_bet_outcomes.json
  └─ daily_pipeline.py --summarize   → (ログ出力)

keiba_app.py [読み取り専用]
  └─ pipeline_predictions.json   → レース名・発走時刻・馬AI確率・オッズ・脚質
  └─ pipeline_bet_suggestions.json → 買い目・券種・投資額
  └─ pipeline_bet_outcomes.json   → 的中フラグ・払戻額
```

---

## 画面構成

### 2タブ構成

```
Tab 1「当日」    … 本日のKPI + レースカード一覧（60秒自動リフレッシュ）
Tab 2「履歴」   … 日付選択 + 累計KPI + 選択日のレースカード
```

---

## Tab 1「当日」詳細

### KPIカード（上部）

```
投資額: ¥3,600  |  回収額: ¥1,200  |  ROI: 33%  |  的中: 3 / 18
```

### 券種別集計テーブル

| 券種 | 買い目数 | 的中 | 的中率 | ROI |
|---|---|---|---|---|
| 単勝 | 12 | 2 | 16.7% | 45% |
| 馬連（流し） | 18 | 1 | 5.6% | 22% |
| ワイド | 12 | 3 | 25.0% | 88% |
| 三連複(AI) | 12 | 0 | 0.0% | 0% |

1点 = 100円固定。

### レースカード（発走時刻順）

**デフォルト表示（折りたたみ）:**
```
🏇 中山1R  3歳未勝利  15:45発走  [発走前 / 集計待ち / 結果済み]

買い目: 単勝 ギンケイ ¥100  |  馬連 ギンケイ-トラストレガート ¥100  ...
結果:  ❌ 外れ (-¥600)   or   ✅ 的中 (+¥1,200 / ROI 200%)
```

**展開時（馬詳細テーブル）:**

| 馬名 | AI勝率 | オッズ | 人気 | 脚質 |
|---|---|---|---|---|
| ギンケイ | 12.4% | 5.6 | 1 | 先行 |
| トラストレガート | 11.8% | 8.3 | 2 | 差し |
| ... | ... | 未取得 | 未取得 | 未取得 |

- オッズ・脚質は `watch_odds` 更新後に実値を表示。未更新なら「未取得」。
- v2スキーマの `feature_dict` から `win_odds`・`running_style` を読む。
- v1スキーマ（`watch_odds`未実行）の場合は「未取得」表示にフォールバック。

---

## Tab 2「履歴」詳細

### 日付選択

- ドロップダウン: `pipeline_predictions.json` に存在する `analysis_date` の一覧を降順で表示
- デフォルト: 最新日付

### 累計KPI（全期間）

```
【全期間累計】
投資額: ¥XX,XXX  |  回収額: ¥XX,XXX  |  ROI: XX%  |  的中率: XX%
```

### ROI推移グラフ

- X軸: 日付、Y軸: 日別ROI（%）
- Streamlit の `st.line_chart` で表示

### 券種別累計テーブル

当日タブと同形式、全期間の集計値。

### 選択日のレースカード

当日タブと同じカード形式。

---

## ステータス出し分け

| 状況 | 表示 |
|---|---|
| `pipeline_predictions.json` に当日データなし | 「本日のレースデータがまだありません。weekend_pipeline.sh を実行してください。」 |
| 発走前（evaluate未実行・当日） | 買い目のみ、結果欄なし、バッジ「発走前」 |
| 発走後・evaluate未実行 | バッジ「集計待ち」、ROIは `-` |
| evaluate完了 | バッジ「結果済み」、的中/外れ・回収額を表示 |
| オッズ未取得（v1スキーマまたはwatch_odds未実行） | オッズ・脚質列に「未取得」 |
| `pipeline_bet_outcomes.json` が存在しない | 履歴タブ「結果データがありません」 |

---

## 自動リフレッシュ

- 当日タブのみ 60秒ごとに `st.rerun()` で再読み込み
- 履歴タブはリフレッシュなし（手動で日付切り替え）

---

## pipeline_bet_outcomes.json スキーマ（既存）

`--evaluate` ステップが `pipeline_store.save_bet_outcomes()` で書き込む。`daily_pipeline.py` の変更は不要。

```json
{
  "202606030401": [
    {
      "bet_type":        "tansho",
      "bet_type_label":  "単勝",
      "bet_combination": ["ギンケイ"],
      "stake":           100,
      "hit":             true,
      "payout":          560,
      "roi":             5.6
    },
    {
      "bet_type":        "umaren",
      "bet_type_label":  "馬連（流し）",
      "bet_combination": ["ギンケイ", "トラストレガート"],
      "stake":           100,
      "hit":             false,
      "payout":          0,
      "roi":             0.0
    }
  ]
}
```

`total_stake` / `total_payout` / ROI はダッシュボード側で計算する（JSON には含まれない）。

---

## 実装除外スコープ

- URL手動入力による単発レース分析
- OpenAI API 呼び出し（ダッシュボードは読み取りのみ）
- レース後回顧・タグ付け（review_engine）
- バックテスト表示
- モバイル最適化

---

## テスト方針

### `tests/test_dashboard_loader.py`

| テスト | 内容 |
|---|---|
| `test_load_today_races` | 当日の `analysis_date` でレースが正しく絞り込まれる |
| `test_kpi_calculation` | 的中率・ROI・投資額・回収額が正確に計算される |
| `test_bet_type_breakdown` | 券種別集計が正確（単勝/馬連/ワイド/三連複） |
| `test_horse_odds_fallback` | v1スキーマ（オッズなし）で「未取得」にフォールバック |
| `test_race_status_prerace` | evaluate前は「発走前」ステータス |
| `test_race_status_result` | evaluate後は的中フラグ・払戻が表示される |
| `test_date_list_descending` | 日付ドロップダウンが降順で並ぶ |
