# Historical Pattern Handoff

このメモは 2026-04-13 時点の旧 handoff を、直近コミット後の現行実装ベースに更新したものです。

## 概要

直近3コミット（`8841320`、`c990a02`、`767421a`）は表示品質更新です。`race_ai_engine.py` が `historical_pattern_display_reasons` / `historical_pattern_reason_groups` を付与し、`historical_pattern_ui.py` が自然文変換と表示監査を担当します。

## 固定

`historical_pattern_engine.py` は未変更です。スコア計算、閾値、確率補正、`model_score` は触らないでください。仕様は `min_starts=5`、`race_any=0.00`、`TOKEN_SCORE_CAP=0.30`、`FEATURE_SCORE_CAP=0.40` です。

## 表示/入力

UI は `historical_pattern_reason_groups` → `historical_pattern_display_reasons` → `historical_pattern_reasons` の順で表示します。body_weight / body_weight_change は requests / Selenium 両方で供給し、キャッシュも更新済みです。

## 確認/次

historical_pattern / body_weight / display UI 関連テストは通過済みで、`tests/test_historical_pattern_ui.py` も追加済みです。主要4レースでは raw token 漏れはなく、body_weight reasons は表示され、低支持・重複表示は UI 側で弱めています。route / historical の軽い重複は許容範囲です。次はスコア調整ではなく、UI文言や監査 helper の小改善を優先してください。重み・閾値調整は当面不要です。
