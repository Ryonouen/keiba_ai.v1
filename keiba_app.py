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
    st.dataframe(df, width="stretch", hide_index=True)


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

        with st.expander(label, expanded=False):
            # 買い目 + 結果を統合テーブルで表示
            bets = race["bets"]
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
                        hit     = o.get("hit", False)
                        payout  = o.get("payout", 0)
                        hit_str    = "✅" if hit else "❌"
                        payout_str = f"¥{payout:,}" if hit else "-"
                    else:
                        hit_str    = "-"
                        payout_str = "-"
                    rows.append({
                        "券種":   bet_label,
                        "組み合わせ": combo_str,
                        "投資":   f"¥{stake}",
                        "的中":   hit_str,
                        "払戻":   payout_str,
                    })
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
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
                    rows.append(
                        {
                            "馬名":   h["horse_name"],
                            "AI勝率": f"{h['ai_win_prob'] * 100:.1f}%" if h["ai_win_prob"] is not None else "-",
                            "オッズ": f"{h['win_odds']:.1f}" if h["win_odds"] is not None else "未取得",
                            "人気":   str(h["popularity"]) if h["popularity"] is not None else "未取得",
                            "脚質":   h["running_style"] or "未取得",
                        }
                    )
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


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

    # 全日付を一度だけロード
    races_by_date = {d: dl.load_races_for_date(d) for d in dates}

    # 全期間累計 KPI
    all_races: List[Dict] = []
    for day_races in races_by_date.values():
        all_races.extend(day_races)

    kpi_all = dl.calc_kpi(all_races)
    st.subheader("全期間累計")
    _render_kpi(kpi_all)

    # 券種別累計
    st.subheader("券種別累計")
    _render_bet_type_table(all_races)

    # ROI 推移グラフ
    roi_rows = []
    for d in sorted(dates):
        k = dl.calc_kpi(races_by_date[d])
        if k["total_stake"] > 0:
            roi_rows.append({"日付": d, "ROI(%)": k["roi"]})

    if roi_rows:
        st.subheader("ROI 推移")
        df_roi = pd.DataFrame(roi_rows).set_index("日付")
        st.line_chart(df_roi)

    # 選択日のレース
    st.subheader(f"{selected} のレース")
    day_races = races_by_date.get(selected, [])
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

# ──────────────────────────────────────────────────────────────
# サイドバー: Kelly 基準設定
# ──────────────────────────────────────────────────────────────
from kelly_staking import load_kelly_config, save_kelly_config

_kcfg = load_kelly_config()

with st.sidebar:
    st.header("⚙️ 設定")
    st.subheader("ケリー基準")
    kelly_enabled = st.checkbox("ケリー基準で賭け金を自動調整", value=bool(_kcfg.get("enabled")))
    kelly_bankroll = st.number_input(
        "バンクロール（円）",
        min_value=1000,
        max_value=1_000_000,
        value=int(_kcfg.get("bankroll") or 10_000),
        step=1000,
    )
    kelly_fraction = st.select_slider(
        "ケリー係数（フルケリーに対する割合）",
        options=[0.1, 0.25, 0.5, 1.0],
        value=float(_kcfg.get("fraction") or 0.25),
        format_func=lambda v: f"{int(v*100)}%",
    )
    if st.button("設定を保存"):
        save_kelly_config({
            "enabled": kelly_enabled,
            "bankroll": int(kelly_bankroll),
            "fraction": float(kelly_fraction),
        })
        st.success("保存しました。次回の --analyze 実行から適用されます。")

st.title("🏇 競馬AI パイプライン ダッシュボード")

tab_today, tab_history = st.tabs(["📅 当日", "📊 履歴"])

with tab_today:
    _tab_today()

with tab_history:
    _tab_history()
