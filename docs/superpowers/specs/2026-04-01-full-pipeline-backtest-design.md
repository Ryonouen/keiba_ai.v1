# フルパイプライン バックテスト 設計書

**作成日**: 2026-04-01  
**対象ファイル**: `backtest_full_pipeline.py`（新規）

---

## 目的

LightGBM + value_ai.py のフルパイプラインを過去レースに適用し、実運用に近い条件で的中率・回収率を検証する。

---

## 要件

| 項目 | 内容 |
|------|------|
| 対象期間 | 2021〜2024年（CSV内の該当レース全件） |
| モデル | 時系列厳密版（test_year ごとに prior years で再訓練） |
| 見送り | value_ai が「ノーベット」と判定したレースはスキップ |
| 払戻 | 単勝・複勝は高精度、馬連・3連複は理論近似値 |

---

## アーキテクチャ

### 処理フロー

```
CSVデータロード（2021〜2024）
  ↓
race_id でグループ化 → レース辞書リスト
  ↓
for test_year in [2021, 2022, 2023, 2024]:
  ├─ LightGBM 再訓練（test_year より前のデータで学習）
  └─ for race in test_year_races:
       ├─ 特徴量 dict 組み立て（CSV列 → value_ai 期待形式）
       ├─ win_prob を各馬に付与
       ├─ value_ai フルパイプライン実行
       │    build_ev_table()
       │    classify_race_structure()
       │    detect_value_horses()
       │    detect_danger_favorites_v3()
       │    recommend_bet_plan()
       ├─ 見送り判定 → スキップ
       └─ 払戻シミュレーション → 結果記録

集計・出力（全体 / 年別 / 券種別）
```

---

## 特徴量マッピング

CSVの列名を value_ai.py が期待するキーに変換する。

| CSV列 | value_ai キー | 備考 |
|-------|--------------|------|
| `feat_win_odds_log` | `win_odds` | `exp(feat_win_odds_log)` で復元 |
| `feat_popularity` | `popularity_rank` | そのまま |
| `feat_running_style_enc` | `running_style` | そのまま |
| `feat_last3f` | `last3f` | そのまま |
| `feat_gate` | `gate` | そのまま |
| `feat_age` | `age` | そのまま |
| `feat_n_runners` | `n_runners` | そのまま |
| `horse_name` | `horse_name` | そのまま |
| LightGBM 出力 | `win_prob` | 推定勝率 |
| `target_win` | — | 正解ラベル（シミュレーション用） |
| `target_top3` | — | 正解ラベル（シミュレーション用） |

---

## 払戻シミュレーション

### 単勝
- 的中条件: 推奨馬の `target_win == 1`
- 払戻: `exp(win_odds_log) * 100` 円
- 精度: 高（実オッズの近似）

### 複勝
- 的中条件: 推奨馬の `target_top3 == 1`
- 払戻: value_ai.py の `PLACE_ODDS_FACTORS` を用いた人気帯別推定
  - 1番人気: オッズ × 1.55 / 2〜3番人気: × 1.90 / 4〜6: × 2.50 / 7〜9: × 3.10 / 10以下: × 4.00
- 精度: 中

### 馬連
- 的中条件: 推奨2頭がそれぞれ1着・2着（target_top3かつ上位2頭）
- 払戻近似: `win_odds_1 * win_odds_2 * 0.75 * 10` 円
  - JRA控除率25%を考慮、10円単位に丸め
- 精度: 低〜中（実際の払戻との誤差が大きい場合あり）

### ワイド
- 的中条件: 推奨2頭がともに `target_top3 == 1`
- 払戻近似: `win_odds_1 * win_odds_2 * 0.25 * 10` 円
- 精度: 低〜中

### 3連複
- 的中条件: 推奨3頭がすべて `target_top3 == 1`
- 払戻近似: `win_odds_1 * win_odds_2 * win_odds_3 * 0.70 * 10` 円
- 精度: 低〜中

> **注意**: 馬連・ワイド・3連複の払戻はあくまで理論近似値。実際のJRA払戻金とは乖離する。傾向・方向性の把握には有効だが、絶対値は参考値として扱うこと。

---

## 出力

### コンソール出力

```
## バックテスト結果（2021〜2024）

全体サマリー
  対象レース      : 12,xxx 件
  推奨あり        : x,xxx 件 (xx.x%)
  総的中          : x,xxx 件 (xx.x%)
  総ROI           : xxx.x%

年別ROI
  2021: xxx.x%  2022: xxx.x%  2023: xxx.x%  2024: xxx.x%

券種別
  単勝   : 的中率 xx.x% / ROI xxx.x%
  複勝   : 的中率 xx.x% / ROI xxx.x%
  馬連   : 的中率 xx.x% / ROI xxx.x%  ※近似
  ワイド : 的中率 xx.x% / ROI xxx.x%  ※近似
  3連複  : 的中率 xx.x% / ROI xxx.x%  ※近似
```

### CSV出力（任意）

`backtest_full_pipeline_result.csv` に各レースの詳細を保存。
列: `race_id, race_date, bet_type, horses, hit, invest, return, roi`

---

## 新規ファイル

| ファイル | 内容 |
|---------|------|
| `backtest_full_pipeline.py` | バックテスト本体（単体実行スクリプト） |

既存ファイルは一切変更しない。

---

## 制約・注意事項

- `jockey_delta` など CSVに存在しない列はデフォルト値（0）で埋める
- `place_odds` は CSVに存在しないため PLACE_ODDS_FACTORS で代替
- `race_pace` は CSVに存在しないため `"medium"` 固定とする
- value_ai.py の一部関数（`detect_rescue_candidates` 等）はバックテストでは省略可

---

## 成功基準

- 2021〜2024の4年分が完走すること
- 券種別ROI・的中率が出力されること
- 「見送り率（推奨なしレースの割合）」が把握できること
