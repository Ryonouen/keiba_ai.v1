# 確定前オッズ・脚質確実取得 設計ドキュメント

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this spec.

**Goal:** 週末全レース分析において、確定前オッズと脚質を確実に取得し、発走30分前に自動でオッズ更新・予測再計算を行う。

**背景:** 早朝の `--analyze` 時点ではオッズが未発売のため `feat_win_odds_log = 0` 固定となり、LightGBM が馬を区別できなかった。

---

## アーキテクチャ概要

### 新規ファイル

| ファイル | 責務 |
|---|---|
| `odds_fetcher.py` | requests → Selenium fallback でオッズ・脚質を取得する単一責務モジュール |

### 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `pipeline_store.py` | `save_prediction_v2()` 追加（start_time / horse_number_map / feature_dict / odds diff 等） |
| `daily_pipeline.py` | `update_race_odds()` / `watch_odds()` 追加、`--watch-odds` CLI 追加 |

### 変更なし

- `analyze_race()` — 既存の返り値をラッパー側で活用するだけ。シグネチャ・内部ロジック変更なし。

---

## データフロー

```
[早朝] python3 daily_pipeline.py --analyze 20260405
  analyze_race() [Selenium・変更なし]
    └─ race_meta.race_info_text に "15:45発走" が含まれる
    └─ features[] に horse_number / horse_link / feat_* / running_style 等が含まれる

  ラッパー (run_daily_race_analysis) が追加抽出:
    start_time       "15:45"
    start_datetime   "2026-04-05T15:45:00"
    horse_number_map {"1": "ショウヘイ", "2": "ヨーホーレイク", ...}
    horse_id_map     {"ショウヘイ": "2022105123", ...}  ← horse_link から parse
    feature_dict[]   全 feat_* / running_style / running_style_source 等を包含
    running_style_missing フラグ

  save_prediction_v2() で pipeline_predictions.json に保存
    prediction_version = 1

[当日レース30分前ごと] python3 daily_pipeline.py --watch-odds 20260405
  ループ (poll_interval=60s):
    start_datetime - 35min ≤ now ≤ start_datetime - 20min のレースを検出
    → update_race_odds(race_id):
        1. odds_fetcher.fetch_win_odds()  [requests → Selenium fallback]
        2. running_style_missing 馬があれば fetch_newspaper_styles() 再試行
        3. feature_dict の feat_win_odds_log / win_odds / popularity を更新
        4. predict_win_probability_with_model() 再実行
        5. win_prob / place_prob / win_ev / place_ev 等を再計算
        6. assign_roles() + recommend_betmaster_plans() 再実行
        7. prediction_version++ / prediction_history 追記
        8. odds_before/odds_after / odds_update_history 保存
        9. pipeline_store.save_prediction_v2() + save_bet_suggestions()
    全レースが start_datetime + 90min を過ぎたら終了

[レース後] python3 daily_pipeline.py --evaluate 20260405
[集計]     python3 daily_pipeline.py --summarize 20260405
```

---

## pipeline_predictions.json スキーマ (v2)

```json
{
  "202609020411": {
    "race_id":       "202609020411",
    "race_name":     "大阪杯(G1)...",
    "race_date":     "2026-04-05",
    "analysis_date": "20260405",
    "analyzed_at":   "2026-04-05 05:04:00",

    "start_time":     "15:45",
    "start_datetime": "2026-04-05T15:45:00",

    "horse_number_map": {"1": "ショウヘイ", "2": "ヨーホーレイク"},
    "horse_id_map":     {"ショウヘイ": "2022105123", "ヨーホーレイク": "2020105456"},

    "prediction_version": 2,
    "prediction_history": [
      {
        "version": 1,
        "created_at": "2026-04-05 05:04:00",
        "source": "initial_analysis",
        "horses": [{"horse_name": "ショウヘイ", "ai_win_prob": 0.069}]
      },
      {
        "version": 2,
        "created_at": "2026-04-05 15:15:22",
        "source": "odds_update_api",
        "horses": [{"horse_name": "ショウヘイ", "ai_win_prob": 0.142}]
      }
    ],

    "horses": [
      {
        "horse_name":   "ショウヘイ",
        "horse_id":     "2022105123",
        "ai_win_prob":  0.142,
        "win_odds":     5.6,
        "popularity":   1,
        "running_style":        "front",
        "running_style_source": "newspaper",
        "running_style_missing": false,
        "feature_dict": {
          "feat_gate": 3,
          "feat_age":  5,
          "feat_popularity":          1,
          "feat_win_odds_log":        1.7228,
          "feat_last3f":              34.5,
          "feat_jockey_weight":       57.0,
          "feat_n_runners":           15,
          "feat_running_style_enc":   0,
          "feat_track_condition_enc": 0,
          "feat_signal_total_adjust": 0.12,
          "feat_cond_diff_age":        0.03,
          "feat_cond_diff_gate":      -0.01,
          "feat_cond_diff_style":      0.05,
          "feat_cond_diff_popularity": 0.02,
          "feat_cond_diff_last3f":     0.08,
          "feat_cond_diff_weight":    -0.01,
          "feat_cond_diff_jockey":     0.04,
          "feat_cond_diff_track":      0.01,
          "win_prob":   0.142,
          "place_prob": 0.361,
          "win_ev":     1.23,
          "place_ev":   0.98,
          "running_style": "front"
        }
      }
    ],

    "odds_update_history": [
      {
        "at":                       "2026-04-05 15:15:22",
        "source":                   "api",
        "coverage_ratio":           0.93,
        "prediction_version_after": 2
      }
    ],

    "odds_before": {
      "status":  "not_open",
      "tansho":  {"1": null, "2": null},
      "fukusho": {"1": null, "2": null}
    },
    "odds_after": {
      "status":           "success",
      "source":           "api",
      "coverage_ratio":   0.93,
      "tansho":  {"1": 5.6, "2": 8.3},
      "fukusho": {"1": [2.1, 3.4], "2": [2.6, 4.1]}
    }
  }
}
```

---

## odds_fetcher.py 設計

### 定数

```python
ODDS_API_URL             = "https://race.netkeiba.com/api/api_get_jra_odds.html"
NEWSPAPER_URL_TEMPLATE   = "https://race.netkeiba.com/race/newspaper.html?race_id={race_id}"
SHUTUBA_URL_TEMPLATE     = "https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"

REQUEST_TIMEOUT          = 10       # seconds
BACKOFF_DELAYS           = [1.0, 2.0]  # retry wait seconds
SELENIUM_WAIT_MAX        = 15       # seconds（オッズ描画待ち）
ODDS_COVERAGE_THRESHOLD  = 0.8     # 出走頭数に対する取得率閾値
STYLE_COVERAGE_THRESHOLD = 0.7
```

### オッズ取得ステータス

| 値 | 意味 |
|---|---|
| `not_open` | 馬券未発売（全オッズ `"–"` または null） |
| `success` | 取得成功（coverage ≥ 0.8） |
| `partial` | 取得できたが coverage < 0.8 |
| `api_failed` | requests 失敗（Selenium 未試行） |
| `selenium_failed` | requests + Selenium 両方失敗 |
| `failed` | その他の失敗 |

### 公開 API

```python
def fetch_win_odds(
    race_id: str,
    horse_number_map: Dict[str, str],   # {"1": "ショウヘイ", ...}
) -> Tuple[str, Optional[Dict[str, float]]]:
    """
    単勝オッズを取得。requests → Selenium fallback。

    Returns
    -------
    (status, {horse_name: win_odds_float} | None)
    status: OddsStatus 文字列
    """

def fetch_newspaper_styles(
    race_id: str,
    horse_number_map: Dict[str, str],
) -> Tuple[str, Optional[Dict[str, str]]]:
    """
    新聞ページから脚質を取得。requests → Selenium fallback。

    Returns
    -------
    (status, {horse_name: running_style_str} | None)
    """
```

### requests バックオフ

```python
BACKOFF_DELAYS = [1.0, 2.0]

for attempt, delay in enumerate(BACKOFF_DELAYS, 1):
    resp = session.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
    if resp.ok:
        break
    if resp.status_code in (403, 429):
        return "api_failed", None   # ブロック → 即返却、リトライしない
    time.sleep(delay)
else:
    return "api_failed", None
```

### 未知 JSON 構造の扱い

```python
try:
    win_odds_raw = data["data"]["Odds"]["WinOdds"]   # 想定パス
except (KeyError, TypeError):
    logger.warning(f"[odds_fetcher] 未知のレスポンス構造: race_id={race_id} → Selenium fallback")
    return _fetch_win_odds_selenium(race_id, horse_number_map)
```

### Selenium フォールバック（軽量）

- shutuba ページのみアクセス（新聞・馬個別ページは開かない）
- `.Odds` CSS セレクタが 1 件以上 `"–"` 以外になるまで最大 15 秒待機
- `{horse_number_str: float}` を返す
- ブラウザは即クローズ

---

## watch_odds ループ設計

### 更新ウィンドウ

```
start_datetime - 35min  ≤  now  ≤  start_datetime - 20min
```

ウィンドウ幅 15 分、poll_interval 60 秒なので必ず 1 回以上チェックされる。

### 終了条件

```
全レースの start_datetime + 90min < now
```

### ステータス別の動作

| fetch_win_odds の status | watch_odds の動作 |
|---|---|
| `not_open` | スキップ（次サイクルで再チェック） |
| `success` | 更新・`updated_ids` に追加 |
| `partial` | ログ警告・更新・`updated_ids` に追加 |
| `api_failed` / `selenium_failed` / `failed` | ログ記録・`updated_ids` に追加（諦め） |

### SIGINT ハンドリング

```python
signal.signal(signal.SIGINT, lambda *_: ...)
# 更新済み N / 総 M 件 を出力して sys.exit(0)
```

---

## 脚質取得失敗時の扱い

| `running_style_source` | 条件 | watcher の動作 |
|---|---|---|
| `"newspaper"` | 新聞ページから取得 | 何もしない |
| `"inferred"` | コーナー通過順から推定 | 何もしない（正常） |
| `"unknown"` | データなし（新馬・転入） | `fetch_newspaper_styles()` で再試行。失敗なら `unknown` 維持・フラグ保存 |

`running_style_missing: true` の馬は bet_suggestions の `selection_reason` に `[脚質不明]` を付記。

---

## CLI

```bash
# 分析（既存・拡張）
python3 daily_pipeline.py --analyze 20260405

# オッズ自動監視（新規・フォアグラウンドブロッキング）
python3 daily_pipeline.py --watch-odds 20260405
python3 daily_pipeline.py --watch-odds 20260405 --poll-interval 30

# 結果照合・集計（既存）
python3 daily_pipeline.py --evaluate  20260405
python3 daily_pipeline.py --summarize 20260405,20260406
```

バックグラウンド実行例：
```bash
python3 daily_pipeline.py --watch-odds 20260405 >> watchdog.log 2>&1 &
```

---

## テスト方針

### `tests/test_odds_fetcher.py`

| テスト | 内容 |
|---|---|
| `test_parse_api_response_standard` | 正常 JSON → `{horse_no: odds}` |
| `test_parse_api_response_unknown_schema` | 未知構造 → warning ログ出力 + Selenium fallback 呼び出し |
| `test_status_not_open` | 全オッズ `"–"` → status=`not_open` |
| `test_coverage_below_threshold` | coverage 60% → status=`partial` |
| `test_coverage_above_threshold` | coverage 85% → status=`success` |
| `test_backoff_retry_transient` | HTTP 503 mock × 2 → 成功（backoff 確認） |
| `test_block_no_retry` | HTTP 403 → 即 `api_failed`（リトライなし） |
| `test_horse_number_normalization` | `"01"` → `"1"` の正規化 |

### `tests/test_watch_odds.py`

| テスト | 内容 |
|---|---|
| `test_window_detection` | T-30min の race のみ選択 |
| `test_already_updated_skip` | `updated_ids` にある race はスキップ |
| `test_not_open_no_version_bump` | `not_open` では `prediction_version` が増えない |
| `test_version_increments_on_success` | 成功時に version が 1 上がる |
| `test_exit_when_all_races_past` | 全レース終了後にループを抜ける |

### `tests/test_pipeline_store_v2.py`

| テスト | 内容 |
|---|---|
| `test_start_time_parsing` | `"15:45発走 / 芝..."` → `"15:45"` + ISO datetime |
| `test_prediction_history_append` | 更新ごとに history が増える |
| `test_odds_status_roundtrip` | 各 status が正確に保存・ロードされる |
| `test_feature_dict_roundtrip` | 全 feat_* フィールドが欠損なく保存・ロードされる |
| `test_horse_id_parse_from_link` | horse_link URL から horse_id を正確に抽出 |

---

## 実装前提条件

- `collect_results_all.py` の全レース収集完了
- LightGBM モデル再学習完了（`keiba_lgbm_model.txt` が最新）
- 既存 `analyze_race()` は変更しない

## 実装除外スコープ

- 複勝・馬連等の全券種リアルタイムオッズ取得（`odds_after.fukusho` 等のスキーマは保持するが今回は単勝のみ実装）
- オッズ更新の複数回実行（T-30min での 1 回のみ）
