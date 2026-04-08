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

# 印と色
_MARKS = ["◎", "○", "▲"]
_MARK_COLORS = {"◎": "#e74c3c", "○": "#e67e22", "▲": "#27ae60"}


# ──────────────────────────────────────────────────────────────
# 共通ウィジェット
# ──────────────────────────────────────────────────────────────
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

    # ── 重賞バッジ ──
    grade_title = race.get("grade_title")
    if grade_title:
        if "G1" in grade_title or "Ｇ１" in grade_title:
            grade_color = "#d4af37"   # gold
        elif "G2" in grade_title or "Ｇ２" in grade_title:
            grade_color = "#aaa9ad"   # silver
        else:
            grade_color = "#cd7f32"   # bronze (G3)
        grade_html = (
            f'<span style="background:{grade_color};color:#000;font-size:11px;'
            f'font-weight:bold;padding:2px 8px;border-radius:10px;margin-right:4px">'
            f'🏆 {grade_title}</span>'
        )
    else:
        grade_html = ""

    html = (
        f'<div style="background:#16213e;border-radius:6px;padding:10px 14px;'
        f'margin-bottom:2px;display:flex;align-items:center;gap:10px;font-family:sans-serif;">'
        f'<div style="background:#293174;color:#fff;font-size:13px;font-weight:bold;'
        f'width:40px;height:40px;border-radius:6px;display:flex;'
        f'align-items:center;justify-content:center;flex-shrink:0;">{r_num}</div>'
        f'<div style="flex:1;min-width:0;">'
        f'<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:5px;">'
        f'{grade_html}'
        f'<span style="color:#e0e0e0;font-weight:bold;font-size:13px">{venue}</span>'
        f'<span style="color:#888;font-size:12px">{start_time}発走</span>'
        f'{status_html}'
        f'</div>'
        f'<div>{marks_html}</div>'
        f'</div>'
        f'<div style="display:flex;flex-direction:column;align-items:flex-end;gap:4px;flex-shrink:0;">'
        f'{hot_html}{upset_html}'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


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


def _render_horse_table(horses: List[Dict], status: str) -> None:
    """
    AI予測順 vs 実際の着順テーブル（コンパクト表示）。
    status="result" のとき actual_rank 列を表示しハイライトする。
    """
    has_result = status == "result"
    medal = {1: "🥇", 2: "🥈", 3: "🥉"}
    actual_bg = {
        1: "rgba(231,76,60,0.22)",
        2: "rgba(52,152,219,0.16)",
        3: "rgba(149,117,205,0.13)",
    }
    ai_rank_color = {1: "#e74c3c", 2: "#e67e22", 3: "#27ae60"}

    rows_html: list[str] = []
    for ai_rank, h in enumerate(horses, 1):
        name      = h.get("horse_name", "")
        jockey    = h.get("jockey") or ""
        horse_no  = h.get("horse_no")
        pop       = h.get("popularity")
        win_prob  = h.get("ai_win_prob")
        actual    = h.get("actual_rank") if has_result else None

        row_bg = actual_bg.get(actual, "transparent") if actual else "transparent"

        # AI順: 上位3はカラーバッジ、それ以外はグレーテキスト
        ai_bg = ai_rank_color.get(ai_rank, "")
        if ai_bg:
            ai_cell = (
                f'<span style="display:inline-block;background:{ai_bg};color:#fff;'
                f'width:22px;height:22px;border-radius:50%;text-align:center;'
                f'line-height:22px;font-weight:bold;font-size:12px">{ai_rank}</span>'
            )
        else:
            ai_cell = f'<span style="color:#666;font-size:12px">{ai_rank}</span>'

        # 着順: メダル+着 or グレー数字
        if actual is not None:
            if actual <= 3:
                actual_cell = f'<span style="font-size:13px">{medal[actual]}</span><span style="font-weight:bold;font-size:12px;margin-left:2px">{actual}着</span>'
            else:
                actual_cell = f'<span style="color:#888;font-size:12px">{actual}着</span>'
        elif has_result:
            actual_cell = '<span style="color:#555;font-size:12px">?</span>'
        else:
            actual_cell = '<span style="color:#444;font-size:11px">—</span>'

        # 馬番: 小さい角丸バッジ
        no_cell = (
            f'<span style="background:#1e2a4a;color:#7fb3d3;padding:1px 5px;'
            f'border-radius:3px;font-size:12px;font-weight:bold">{horse_no}</span>'
            if horse_no is not None else '<span style="color:#444">—</span>'
        )

        prob_str = f'{win_prob * 100:.1f}%' if win_prob is not None else '—'
        pop_str  = f'{pop}人気' if pop is not None else '—'
        name_cell = (
            f'<span style="color:#e0e0e0;font-size:13px;font-weight:bold">{name}</span>'
            + (f' <span style="color:#666;font-size:11px">{jockey}</span>' if jockey else '')
        )

        rows_html.append(
            f'<tr style="background:{row_bg};border-bottom:1px solid #1e1e2e">'
            f'<td style="padding:5px 4px;text-align:center;width:46px">{ai_cell}</td>'
            f'<td style="padding:5px 4px;text-align:center;width:68px;white-space:nowrap">{actual_cell}</td>'
            f'<td style="padding:5px 4px;text-align:center;width:44px">{no_cell}</td>'
            f'<td style="padding:5px 8px">{name_cell}</td>'
            f'<td style="padding:5px 4px;text-align:center;width:54px;color:#888;font-size:11px">{pop_str}</td>'
            f'<td style="padding:5px 6px;text-align:right;width:54px;color:#7fb3d3;font-size:12px;font-weight:bold">{prob_str}</td>'
            f'</tr>'
        )

    table = (
        '<table style="width:100%;border-collapse:collapse;font-family:sans-serif;table-layout:fixed">'
        '<colgroup>'
        '<col style="width:46px"><col style="width:68px"><col style="width:44px">'
        '<col><col style="width:54px"><col style="width:54px">'
        '</colgroup>'
        '<thead><tr style="border-bottom:1px solid #3a3a5a">'
        '<th style="padding:4px;text-align:center;color:#666;font-size:10px;font-weight:normal">AI順</th>'
        '<th style="padding:4px;text-align:center;color:#666;font-size:10px;font-weight:normal">着順</th>'
        '<th style="padding:4px;text-align:center;color:#666;font-size:10px;font-weight:normal">馬番</th>'
        '<th style="padding:4px 8px;text-align:left;color:#666;font-size:10px;font-weight:normal">馬名 / 騎手</th>'
        '<th style="padding:4px;text-align:center;color:#666;font-size:10px;font-weight:normal">人気</th>'
        '<th style="padding:4px 6px;text-align:right;color:#666;font-size:10px;font-weight:normal">AI勝率</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        '</table>'
    )
    st.markdown(table, unsafe_allow_html=True)


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

            # 馬別 AI 予測 vs 実際の着順
            horses = race["horses"]
            if horses:
                st.markdown("**AI予測 vs 実際の着順**")
                _render_horse_table(horses, race["status"])


# ──────────────────────────────────────────────────────────────
# Tab 1「レース分析」— 日付→競馬場の2段サブタブ
# ──────────────────────────────────────────────────────────────
def _render_races_grouped(races_by_venue: Dict[str, List[Dict]]) -> None:
    """競馬場ごとにグループ化されたレース dict からサブタブを描画する。"""
    venue_names = list(races_by_venue.keys())
    if len(venue_names) == 1:
        _render_race_cards(races_by_venue[venue_names[0]])
    else:
        venue_tabs = st.tabs([f"🏟 {v}" for v in venue_names])
        for tab, venue in zip(venue_tabs, venue_names):
            with tab:
                _render_race_cards(races_by_venue[venue])


def _render_venues_for_date(date_str: str) -> None:
    """指定日のレースを競馬場ごとのサブタブで表示する（土日タブ用）。"""
    races_by_venue = dl.get_races_by_venue(date_str)
    if not races_by_venue:
        st.info("この日のデータがありません。")
        return
    _render_races_grouped(races_by_venue)


@st.fragment(run_every=60)
def _render_today_live() -> None:
    """今日タブ: 60秒自動更新。KPI + 券種別 + 競馬場別レース一覧。"""
    today_str = datetime.now().strftime("%Y%m%d")
    st.caption(f"最終更新: {datetime.now().strftime('%H:%M:%S')}  （60秒ごとに自動更新）")
    races_by_venue = dl.get_races_by_venue(today_str)
    if not races_by_venue:
        st.info(
            "本日のレースデータがまだありません。\n"
            "`bash weekend_pipeline.sh` を実行してください。"
        )
        return
    all_races: List[Dict] = [r for venue_races in races_by_venue.values() for r in venue_races]
    kpi = dl.calc_kpi(all_races)
    _render_kpi(kpi)
    st.subheader("券種別集計")
    _render_bet_type_table(all_races)
    st.subheader(f"レース一覧（{len(all_races)} レース）")
    _render_races_grouped(races_by_venue)


def _tab_race_analysis() -> None:
    """レース分析タブ: 今日/土/日の日付サブタブ + 競馬場サブタブ。"""
    today_str, sat_str, sun_str = dl.get_weekend_date_strs()
    today_dt = datetime.strptime(today_str, "%Y%m%d")
    sat_dt   = datetime.strptime(sat_str,   "%Y%m%d")
    sun_dt   = datetime.strptime(sun_str,   "%Y%m%d")

    today_label = f"📅 今日({today_dt.strftime('%-m/%-d')})"
    sat_label   = f"土({sat_dt.strftime('%-m/%-d')})"
    sun_label   = f"日({sun_dt.strftime('%-m/%-d')})"

    date_tabs = st.tabs([today_label, sat_label, sun_label])

    with date_tabs[0]:
        _render_today_live()

    with date_tabs[1]:
        _render_venues_for_date(sat_str)

    with date_tabs[2]:
        _render_venues_for_date(sun_str)


# ──────────────────────────────────────────────────────────────
# Tab 2「日次レポート」
# ──────────────────────────────────────────────────────────────
def _tab_daily() -> None:
    dates = dl.get_available_dates()
    if not dates:
        st.info("結果データがありません。`--evaluate` を実行してください。")
        return

    selected = st.selectbox(
        "日付を選択",
        dates,
        index=0,
        format_func=lambda d: f"{d[:4]}/{d[4:6]}/{d[6:]}",
    )

    races = dl.load_races_for_date(selected)
    if not races:
        st.info("この日のデータがありません。")
        return

    kpi = dl.calc_kpi(races)
    st.subheader(f"{selected[:4]}/{selected[4:6]}/{selected[6:]} の集計")
    _render_kpi(kpi)

    st.subheader("券種別集計")
    _render_bet_type_table(races)

    st.subheader(f"レース一覧（{len(races)} レース）")
    races_by_venue: Dict[str, List[Dict]] = {}
    for r in races:
        v = r.get("venue") or "不明"
        races_by_venue.setdefault(v, []).append(r)
    _render_races_grouped(races_by_venue)


# ──────────────────────────────────────────────────────────────
# Tab 3「月次レポート」
# ──────────────────────────────────────────────────────────────
def _tab_monthly() -> None:
    months = dl.get_available_months()
    if not months:
        st.info("集計データがありません。")
        return

    selected = st.selectbox(
        "年月を選択",
        months,
        index=0,
        format_func=lambda m: f"{m[:4]}年{m[4:]}月",
    )
    y, mo = int(selected[:4]), int(selected[4:])

    daily_rows = dl.get_daily_kpi_for_month(y, mo)
    if not daily_rows:
        st.info("この月のデータがありません。")
        return

    # 月合計 KPI
    total_stake  = sum(r["total_stake"]  for r in daily_rows)
    total_payout = sum(r["total_payout"] for r in daily_rows)
    hit_count    = sum(r["hit_count"]    for r in daily_rows)
    total_bets   = sum(r["total_bets"]   for r in daily_rows)
    roi          = round(total_payout / total_stake * 100, 1) if total_stake > 0 else 0.0

    st.subheader(f"{y}年{mo}月 合計")
    _render_kpi({"total_stake": total_stake, "total_payout": total_payout,
                 "roi": roi, "hit_count": hit_count, "total_bets": total_bets})

    # 日別 ROI 推移
    st.subheader("日別 ROI 推移")
    df_roi = pd.DataFrame([{"日付": r["date"], "ROI(%)": r["roi"]} for r in daily_rows])
    df_roi["日付"] = pd.to_datetime(df_roi["日付"], format="%Y%m%d").dt.strftime("%m/%d")
    st.line_chart(df_roi.set_index("日付"))

    # 日別明細
    st.subheader("日別明細")
    df2 = pd.DataFrame(daily_rows)
    df2.insert(0, "日付", pd.to_datetime(df2["date"], format="%Y%m%d").dt.strftime("%Y/%m/%d"))
    df2 = df2[["日付", "total_stake", "total_payout", "roi", "hit_count", "total_bets"]]
    df2.columns = ["日付", "投資額", "回収額", "ROI(%)", "的中", "買い目数"]
    st.dataframe(df2, hide_index=True)


# ──────────────────────────────────────────────────────────────
# Tab 4「年次レポート」
# ──────────────────────────────────────────────────────────────
def _tab_yearly() -> None:
    years = dl.get_available_years()
    if not years:
        st.info("集計データがありません。")
        return

    selected = st.selectbox(
        "年を選択",
        years,
        index=0,
        format_func=lambda y: f"{y}年",
    )

    monthly_rows = dl.get_monthly_kpi_for_year(int(selected))
    if not monthly_rows:
        st.info("この年のデータがありません。")
        return

    # 年合計 KPI
    total_stake  = sum(r["total_stake"]  for r in monthly_rows)
    total_payout = sum(r["total_payout"] for r in monthly_rows)
    hit_count    = sum(r["hit_count"]    for r in monthly_rows)
    total_bets   = sum(r["total_bets"]   for r in monthly_rows)
    roi          = round(total_payout / total_stake * 100, 1) if total_stake > 0 else 0.0

    st.subheader(f"{selected}年 合計")
    _render_kpi({"total_stake": total_stake, "total_payout": total_payout,
                 "roi": roi, "hit_count": hit_count, "total_bets": total_bets})

    # 月別 ROI 推移
    st.subheader("月別 ROI 推移")
    df_roi = pd.DataFrame([{"年月": r["month"], "ROI(%)": r["roi"]} for r in monthly_rows])
    df_roi["年月"] = pd.to_datetime(df_roi["年月"], format="%Y%m").dt.strftime("%Y/%m")
    st.line_chart(df_roi.set_index("年月"))

    # 月別明細
    st.subheader("月別明細")
    df2 = pd.DataFrame(monthly_rows)
    df2.insert(0, "年月", pd.to_datetime(df2["month"], format="%Y%m").dt.strftime("%Y/%m"))
    df2 = df2[["年月", "total_stake", "total_payout", "roi", "hit_count", "total_bets"]]
    df2.columns = ["年月", "投資額", "回収額", "ROI(%)", "的中", "買い目数"]
    st.dataframe(df2, hide_index=True)


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

tab_analysis, tab_daily, tab_monthly, tab_yearly = st.tabs([
    "🏇 レース分析", "📊 日次レポート", "📅 月次レポート", "📆 年次レポート"
])

with tab_analysis:
    _tab_race_analysis()

with tab_daily:
    _tab_daily()

with tab_monthly:
    _tab_monthly()

with tab_yearly:
    _tab_yearly()
