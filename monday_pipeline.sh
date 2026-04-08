#!/bin/bash
# monday_pipeline.sh
# 毎週月曜朝に実行:
#   1. 先週末（土日）の結果照合・集計を完了させる（idempotent）
#   2. 今週末（土日）のレースID一覧をプローブ取得する
#
# 使い方:
#   bash monday_pipeline.sh
#
# cron 設定（毎週月曜 7:00）:
#   0 7 * * 1 cd /Users/ryokarahashi/keiba_ai && bash monday_pipeline.sh >> monday_pipeline.log 2>&1

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DIR/monday_pipeline_$(date +%Y%m%d).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"
}

cd "$DIR"

# macOS / Linux 両対応の日付計算
_date_offset() {
    local offset=$1
    if date -v"${offset}d" +%Y%m%d 2>/dev/null; then
        return
    fi
    date -d "${offset} days" +%Y%m%d
}

TODAY=$(date +%Y%m%d)
LAST_SAT=$(_date_offset -2)
LAST_SUN=$(_date_offset -1)
NEXT_SAT=$(_date_offset +5)
NEXT_SUN=$(_date_offset +6)

log "============================================"
log "月曜パイプライン開始: $TODAY"
log "  先週末: $LAST_SAT (土) / $LAST_SUN (日)"
log "  今週末: $NEXT_SAT (土) / $NEXT_SUN (日)"
log "============================================"

log "Step1: --evaluate $LAST_SAT 開始"
python3 daily_pipeline.py --evaluate "$LAST_SAT" >> "$LOG" 2>&1 && log "Step1 完了" || log "Step1 エラー（スキップ）"

log "Step2: --evaluate $LAST_SUN 開始"
python3 daily_pipeline.py --evaluate "$LAST_SUN" >> "$LOG" 2>&1 && log "Step2 完了" || log "Step2 エラー（スキップ）"

log "Step3: --summarize $LAST_SAT,$LAST_SUN"
python3 daily_pipeline.py --summarize "$LAST_SAT,$LAST_SUN" 2>&1 | tee -a "$LOG"
log "Step3 完了"

log "Step4: 今週末レースID プローブ"
export NEXT_SAT NEXT_SUN
python3 - <<'PYEOF' 2>&1 | tee -a "$LOG"
import sys, os
from daily_pipeline import get_race_ids_by_date

next_sat = os.environ.get("NEXT_SAT", "")
next_sun = os.environ.get("NEXT_SUN", "")

for date_str, label in [(next_sat, "土"), (next_sun, "日")]:
    if not date_str:
        continue
    try:
        ids = get_race_ids_by_date(date_str)
        print(f"  [{date_str} ({label})] {len(ids)} レースID取得")
        for rid in ids:
            print(f"    {rid}")
    except Exception as e:
        print(f"  [{date_str}] 取得エラー: {e}")
PYEOF
log "Step4 完了"

log "============================================"
log "月曜パイプライン完了: $TODAY"
log "============================================"
