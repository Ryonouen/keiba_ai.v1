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
# ユーティリティ
# ──────────────────────────────────────────────────────────────
_GRADE_COLOR = {"A": "#f1c40f", "B": "#27ae60", "C": "#888", "D": "#555"}
_GRADE_BG    = {"A": "#3d3200", "B": "#0d2e1a", "C": "#222", "D": "#1a1a1a"}


def _confidence_grade(win_ev: "float | None") -> "tuple[str, str, str]":
    """
    単勝期待値 (win_ev) から妙味度グレード (A/B/C/D)、文字色、背景色を返す。
    A: EV>=1.2（プラス期待値・妙味あり）
    B: EV>=0.9（市場とほぼ拮抗）
    C: EV>=0.65（やや過剰人気）
    D: 以下（市場より大幅に劣位）
    """
    if win_ev is None:
        return "—", "#555", "transparent"
    if win_ev >= 1.2:
        return "A", _GRADE_COLOR["A"], _GRADE_BG["A"]
    if win_ev >= 0.9:
        return "B", _GRADE_COLOR["B"], _GRADE_BG["B"]
    if win_ev >= 0.65:
        return "C", _GRADE_COLOR["C"], _GRADE_BG["C"]
    return "D", _GRADE_COLOR["D"], _GRADE_BG["D"]


def _ability_bar_html(ability_score: "float | None") -> str:
    """
    ability_score (0〜100) をカラーバーの HTML で返す。

    ability_score は calc_base_ability_score() * 100 で生成される
    「馬のベース能力指標」。win_prob (勝率) とは独立した値。

    バー幅: ability_score を直接使用（100=フル幅）
    色: 70以上→赤, 50以上→橙, 30以上→青, それ以下→グレー
    """
    if ability_score is None:
        return '<span style="color:#555">—</span>'
    score = float(ability_score)
    bar_w = min(score, 100)
    color = ("#e74c3c" if score >= 70 else "#e67e22" if score >= 50 else
             "#3498db" if score >= 30 else "#555")
    return (
        f'<span style="display:inline-flex;align-items:center;gap:5px">'
        f'<span style="color:#ccc;font-size:12px;min-width:52px;text-align:right">{score:.1f}</span>'
        f'<span style="display:inline-block;background:#2a2a3a;border-radius:2px;'
        f'height:7px;width:50px;overflow:hidden">'
        f'<span style="display:block;background:{color};height:7px;'
        f'border-radius:2px;width:{bar_w:.0f}%"></span>'
        f'</span></span>'
    )


# ──────────────────────────────────────────────────────────────
# 共通ウィジェット
# ──────────────────────────────────────────────────────────────
def _render_race_summary(race: Dict) -> None:
    """レースカードを競合他社スタイルで描画: ヘッダー＋トップ3馬インライン行。"""
    venue      = race.get("venue") or ""
    r_num      = race.get("race_number") or ""
    start_time = race.get("start_time") or "??:??"
    status     = race.get("status", "prerace")
    race_title = race.get("race_title") or ""
    distance   = race.get("distance")
    surface    = race.get("surface") or ""
    n_runners  = race.get("n_runners")

    # ── ステータスバッジ ──
    if status == "result" and race.get("outcomes"):
        any_hit = any(o.get("hit") for o in race["outcomes"])
        if any_hit:
            status_html = (
                '<span style="background:#1d6a27;color:#6fcf97;font-size:10px;'
                'padding:2px 7px;border-radius:8px">✅ 的中</span>'
            )
        else:
            status_html = (
                '<span style="background:#4a1a1a;color:#e57373;font-size:10px;'
                'padding:2px 7px;border-radius:8px">❌ 外れ</span>'
            )
    elif status == "awaiting":
        status_html = (
            '<span style="background:#4a3f00;color:#f1c40f;font-size:10px;'
            'padding:2px 7px;border-radius:8px">⏳ 集計待ち</span>'
        )
    else:
        status_html = (
            '<span style="background:#1a3a5c;color:#7fb3d3;font-size:10px;'
            'padding:2px 7px;border-radius:8px">🕐 発走前</span>'
        )

    # ── 荒れスコア・激熱バッジ ──
    upset_label = race.get("upset_label", "")
    upset_color = race.get("upset_color", "#ff9800")
    upset_score = race.get("upset_score", "")
    upset_html  = (
        f'<span style="background:{upset_color};color:#fff;font-size:10px;'
        f'padding:2px 7px;border-radius:8px">{upset_label} {upset_score}</span>'
    )
    hot_bets = race.get("hot_bets") or []
    hot_html = (
        f'<span style="background:#c0392b;color:#fff;font-size:10px;'
        f'padding:2px 7px;border-radius:8px">🔥 {len(hot_bets)}件</span>'
        if hot_bets else ""
    )

    race_type_label = race.get("race_type_label") or ""
    race_type_color = race.get("race_type_color") or "#607d8b"
    axis_confidence = race.get("axis_confidence") or ""
    has_value_horse = bool(race.get("has_value_horse"))
    has_danger_favorite = bool(race.get("has_danger_favorite"))

    race_type_html = (
        f'<span style="background:{race_type_color};color:#fff;font-size:10px;'
        f'padding:2px 7px;border-radius:8px">{race_type_label}</span>'
        if race_type_label else ""
    )
    axis_html = (
        f'<span style="background:#1f3a5f;color:#9ad0ff;font-size:10px;'
        f'padding:2px 7px;border-radius:8px">軸信頼 {axis_confidence}</span>'
        if axis_confidence else ""
    )
    value_html = (
        '<span style="background:#145a32;color:#7dffb3;font-size:10px;'
        'padding:2px 7px;border-radius:8px">💎 妙味馬あり</span>'
        if has_value_horse else ""
    )
    danger_html = (
        '<span style="background:#5c1f1f;color:#ff9b9b;font-size:10px;'
        'padding:2px 7px;border-radius:8px">⚠ 危険人気あり</span>'
        if has_danger_favorite else ""
    )

    # ── 重賞グレードバッジ ──
    grade_title = race.get("grade_title")
    if grade_title:
        gc = ("#d4af37" if "G1" in grade_title or "Ｇ１" in grade_title
              else "#aaa9ad" if "G2" in grade_title or "Ｇ２" in grade_title
              else "#cd7f32")
        grade_badge = (
            f'<span style="background:{gc};color:#000;font-size:10px;font-weight:bold;'
            f'padding:2px 7px;border-radius:8px;margin-right:4px">🏆 {grade_title}</span>'
        )
    else:
        grade_badge = ""

    # ── レースメタ情報行 ──
    meta_parts = [start_time]
    if distance:
        surf = "芝" if surface == "芝" else "ダ" if "ダ" in surface else surface
        meta_parts.append(f"{surf}{distance}m")
    if n_runners:
        meta_parts.append(f"{n_runners}頭")
    meta_html = '<span style="color:#666;font-size:11px">' + " / ".join(meta_parts) + "</span>"

    # ── 表示するタイトル (grade_title があれば grade_title、なければ race_title) ──
    display_title = grade_title or race_title
    title_part = (
        f'<span style="color:#e0e0e0;font-weight:bold;font-size:14px">'
        f'{venue}{r_num}'
        f'{"　" + display_title if display_title else ""}'
        f'</span>'
    )

    # ── トップ3馬インライン行 ──
    medal = {1: "🥇", 2: "🥈", 3: "🥉"}
    actual_row_bg = {
        1: "rgba(231,76,60,0.15)",
        2: "rgba(52,152,219,0.10)",
        3: "rgba(149,117,205,0.08)",
    }
    ai_rank_color = {1: "#e74c3c", 2: "#e67e22", 3: "#27ae60"}

    horse_rows = []
    for ai_rank, h in enumerate(race.get("horses", [])[:3], 1):
        name       = h.get("horse_name", "")
        horse_no   = h.get("horse_no")
        win_prob   = h.get("ai_win_prob")
        actual     = h.get("actual_rank") if status == "result" else None
        eval_group = h.get("eval_group") or ""
        is_value   = bool(h.get("is_value"))
        is_danger  = bool(h.get("is_danger"))

        row_bg   = actual_row_bg.get(actual, "transparent") if actual else "transparent"
        ai_color = ai_rank_color.get(ai_rank, "#555")

        # 馬番バッジ
        no_badge = (
            f'<span style="background:#1e2a4a;color:#7fb3d3;padding:1px 5px;'
            f'border-radius:3px;font-size:11px;font-weight:bold">{horse_no}</span> '
            if horse_no is not None else ""
        )

        # 結果セル
        if actual is not None:
            if actual <= 3:
                result_html = f'<span style="font-size:12px">{medal[actual]} {actual}着</span>'
            else:
                result_html = f'<span style="color:#666;font-size:11px">{actual}着</span>'
        else:
            result_html = ""

        prob_str = f'<span style="color:#7fb3d3;font-size:11px">{win_prob*100:.1f}%</span>' if win_prob else ""

        signal_bits = []
        if eval_group:
            signal_bits.append(
                f'<span style="background:#2a2f45;color:#d6d9e6;padding:1px 5px;'
                f'border-radius:999px;font-size:10px">{eval_group}</span>'
            )
        if is_value:
            signal_bits.append('<span style="font-size:11px">💎</span>')
        if is_danger:
            signal_bits.append('<span style="font-size:11px">⚠</span>')
        signal_html = (
            f'<span style="display:inline-flex;align-items:center;gap:4px;min-width:48px;justify-content:flex-end">'
            f'{"".join(signal_bits)}</span>'
            if signal_bits else ""
        )

        horse_rows.append(
            f'<div style="display:flex;align-items:center;gap:6px;padding:4px 0;'
            f'border-top:1px solid #1e2533;background:{row_bg}">'
            f'<span style="color:{ai_color};font-size:11px;font-weight:bold;min-width:14px">{ai_rank}.</span>'
            f'{no_badge}'
            f'<span style="color:#d0d0d0;font-size:12px;flex:1;overflow:hidden;'
            f'text-overflow:ellipsis;white-space:nowrap">{name}</span>'
            f'{prob_str}'
            f'{signal_html}'
            f'<span style="min-width:50px;text-align:right">{result_html}</span>'
            f'</div>'
        )
    horse_rows_html = "".join(horse_rows)

    html = (
        f'<div style="background:#16213e;border-radius:8px;padding:10px 14px;'
        f'margin-bottom:3px;font-family:sans-serif">'
        # ヘッダー行
        f'<div style="display:flex;align-items:flex-start;gap:10px">'
        # 左: レース番号バッジ
        f'<div style="background:#293174;color:#fff;font-size:14px;font-weight:bold;'
        f'min-width:42px;height:42px;border-radius:6px;display:flex;align-items:center;'
        f'justify-content:center;flex-shrink:0">{r_num}</div>'
        # 中央: タイトル＋メタ＋馬列
        f'<div style="flex:1;min-width:0">'
        f'<div style="display:flex;align-items:center;gap:5px;flex-wrap:wrap;margin-bottom:2px">'
        f'{grade_badge}{title_part}{status_html}'
        f'</div>'
        f'<div style="margin-bottom:4px">{meta_html}</div>'
        f'{horse_rows_html}'
        f'</div>'
        # 右: バッジ群
        f'<div style="display:flex;flex-direction:column;align-items:flex-end;'
        f'gap:4px;flex-shrink:0;padding-top:2px">'
        f'{race_type_html}'
        f'{axis_html}'
        f'{value_html}'
        f'{danger_html}'
        f'{hot_html}'
        f'{upset_html}'
        f'</div>'
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


def _render_column_legend() -> None:
    """
    馬テーブルの列説明（凡例）を2行構成で表示する。
    能力勝率・AI勝率・能差の違いをコンパクトに示す。
    将来の3者比較（能力 / AI / 市場）に対応する設計を意識した配置。
    """
    st.markdown(
        '<div style="font-size:11px;color:#888;margin:2px 0 6px 0;line-height:1.8">'
        '<span><b style="color:#a0c4ff">能力勝率</b> 市場フリーの能力評価　'
        '<b style="color:#ccc">AI勝率</b> オッズ加味の最終予測　'
        '<b style="color:#f0c040">妙味度</b> 単勝EV基準 A(≥1.2)〜D(&lt;0.65)</span><br>'
        '<span><b style="color:#d6d9e6">予測群</b> AI勝率ベースのランク S(≥20%)／A(≥12%)／B(≥7%)／C　'
        '💎 乖離+3pt以上割安　⚠ 乖離-4pt以上割高　※妙味度(EV)とは独立</span><br>'
        '<span><b style="color:#2ecc71">能↑</b> 能力がAI評価を上回る（割安候補）　'
        '<b style="color:#e57373">能↓</b> AI評価が能力を上回る（要注意）</span>'
        '</div>',
        unsafe_allow_html=True,
    )


def _render_horse_table(horses: List[Dict], status: str) -> None:
    """
    AI予測順 vs 実際の着順テーブル。
    妙味度(A-D) / 能力値バー / 勝率 / 2着内率 / 3着内率 を含む全カラム表示。
    """
    has_result = status == "result"
    medal = {1: "🥇", 2: "🥈", 3: "🥉"}
    actual_bg = {
        1: "rgba(231,76,60,0.22)",
        2: "rgba(52,152,219,0.16)",
        3: "rgba(149,117,205,0.13)",
    }
    ai_rank_color = {1: "#e74c3c", 2: "#e67e22", 3: "#27ae60"}

    def _pct(v: "float | None") -> str:
        return f"{v * 100:.1f}%" if v is not None else "—"

    def _pct_cell(v: "float | None", highlight: bool = False) -> str:
        s = _pct(v)
        color = "#e74c3c" if highlight else "#ccc"
        return f'<span style="color:{color};font-size:12px">{s}</span>'

    rows_html: list[str] = []
    for ai_rank, h in enumerate(horses, 1):
        name            = h.get("horse_name", "")
        jockey          = h.get("jockey") or ""
        horse_no        = h.get("horse_no")
        pop             = h.get("popularity")
        pop_estimated   = h.get("odds_is_estimated", False)
        win_prob        = h.get("ai_win_prob")
        ability_win_prob = h.get("ability_win_prob")    # 市場フリー能力勝率
        ability_diff     = (
            round(ability_win_prob - win_prob, 4)
            if ability_win_prob is not None and win_prob is not None else None
        )
        market_win_prob = h.get("market_win_prob")
        prob_gap        = h.get("prob_gap")
        place_prob      = h.get("place_prob")
        top2_prob       = h.get("top2_prob")
        win_ev          = h.get("win_ev")
        eval_group      = h.get("eval_group") or ""
        is_value        = bool(h.get("is_value"))
        is_danger       = bool(h.get("is_danger"))
        actual          = h.get("actual_rank") if has_result else None

        row_bg = actual_bg.get(actual, "transparent") if actual else "transparent"

        # AI順バッジ
        ai_color = ai_rank_color.get(ai_rank, "")
        if ai_color:
            ai_cell = (
                f'<span style="display:inline-block;background:{ai_color};color:#fff;'
                f'width:22px;height:22px;border-radius:50%;text-align:center;'
                f'line-height:22px;font-weight:bold;font-size:12px">{ai_rank}</span>'
            )
        else:
            ai_cell = f'<span style="color:#555;font-size:12px">{ai_rank}</span>'

        # 着順
        if actual is not None:
            if actual <= 3:
                actual_cell = (
                    f'<span style="font-size:12px">{medal[actual]}</span>'
                    f'<span style="font-weight:bold;font-size:11px;margin-left:1px">{actual}着</span>'
                )
            else:
                actual_cell = f'<span style="color:#777;font-size:11px">{actual}着</span>'
        elif has_result:
            actual_cell = '<span style="color:#444;font-size:11px">?</span>'
        else:
            actual_cell = '<span style="color:#333;font-size:10px">—</span>'

        # 馬番
        no_cell = (
            f'<span style="background:#1e2a4a;color:#7fb3d3;padding:1px 5px;'
            f'border-radius:3px;font-size:12px;font-weight:bold">{horse_no}</span>'
            if horse_no is not None else '<span style="color:#333">—</span>'
        )

        # 馬名/騎手
        name_cell = (
            f'<span style="color:#e0e0e0;font-size:12px;font-weight:bold">{name}</span>'
            + (f'<br><span style="color:#666;font-size:10px">{jockey}</span>' if jockey else '')
        )

        # 人気
        if pop:
            _est_badge = '<span style="color:#e67e22;font-size:9px"> 予</span>' if pop_estimated else ''
            pop_str = f'<span style="color:#888;font-size:11px">{pop}人気{_est_badge}</span>'
        else:
            pop_str = '<span style="color:#333">—</span>'

        # 妙味度グレード
        grade, g_color, g_bg = _confidence_grade(win_ev)
        grade_cell = (
            f'<span style="background:{g_bg};color:{g_color};font-size:12px;'
            f'font-weight:bold;padding:1px 6px;border-radius:3px">{grade}</span>'
        )

        eval_bits = []
        if eval_group:
            eval_bits.append(
                f'<span style="background:#2a2f45;color:#d6d9e6;font-size:11px;'
                f'font-weight:bold;padding:1px 6px;border-radius:999px">{eval_group}</span>'
            )
        if is_value:
            eval_bits.append('<span style="font-size:11px">💎</span>')
        if is_danger:
            eval_bits.append('<span style="font-size:11px">⚠</span>')
        eval_cell = (
            f'<span style="display:inline-flex;align-items:center;gap:4px">{"".join(eval_bits)}</span>'
            if eval_bits else '<span style="color:#444;font-size:11px">—</span>'
        )

        market_cell = _pct_cell(market_win_prob, False)
        if prob_gap is None:
            gap_cell = '<span style="color:#444;font-size:11px">—</span>'
        else:
            gap_color = '#2ecc71' if prob_gap > 0 else '#e57373' if prob_gap < 0 else '#888'
            gap_prefix = '+' if prob_gap > 0 else ''
            gap_cell = f'<span style="color:{gap_color};font-size:12px">{gap_prefix}{prob_gap*100:.1f}pt</span>'

        # 能力値バー (ability_score = ベース能力 0〜100, win_prob とは独立)
        ability_cell = _ability_bar_html(h.get("ability_score"))

        # 能力勝率（market-free）
        if ability_win_prob is not None:
            ability_win_cell = f'<span style="color:#a0c4ff;font-size:12px">{ability_win_prob*100:.1f}%</span>'
        else:
            ability_win_cell = '<span style="color:#555;font-size:12px;opacity:0.45">—</span>'

        # 能↑↓（ability_win_prob - ai_win_prob）: 正=能力>AI、負=AI>能力
        if ability_diff is None:
            ability_diff_cell = '<span style="color:#555;font-size:12px;opacity:0.45">—</span>'
        else:
            diff_color = '#2ecc71' if ability_diff > 0.005 else '#e57373' if ability_diff < -0.005 else '#888'
            if ability_diff > 0.005:
                ability_diff_cell = f'<span style="color:{diff_color};font-size:12px">↑ +{ability_diff*100:.1f}pt</span>'
            elif ability_diff < -0.005:
                ability_diff_cell = f'<span style="color:{diff_color};font-size:12px">↓ {ability_diff*100:.1f}pt</span>'
            else:
                ability_diff_cell = f'<span style="color:{diff_color};font-size:12px">{ability_diff*100:.1f}pt</span>'

        # 勝率 / 2着内率 / 3着内率 (top-3ハイライト)
        is_top = actual is not None and actual <= 3
        win_cell   = _pct_cell(win_prob,   is_top)
        top2_cell  = _pct_cell(top2_prob,  is_top)
        top3_cell  = _pct_cell(place_prob, is_top)

        rows_html.append(
            f'<tr style="background:{row_bg};border-bottom:1px solid #1a1a2a">'
            f'<td style="padding:5px 3px;text-align:center">{ai_cell}</td>'
            f'<td style="padding:5px 3px;text-align:center;white-space:nowrap">{actual_cell}</td>'
            f'<td style="padding:5px 3px;text-align:center">{no_cell}</td>'
            f'<td style="padding:5px 7px">{name_cell}</td>'
            f'<td style="padding:5px 3px;text-align:center">{pop_str}</td>'
            f'<td style="padding:5px 3px;text-align:center">{eval_cell}</td>'
            f'<td style="padding:5px 4px">{ability_cell}</td>'
            f'<td style="padding:5px 4px;text-align:right">{ability_win_cell}</td>'
            f'<td style="padding:5px 4px;text-align:right">{ability_diff_cell}</td>'
            f'<td style="padding:5px 4px;text-align:right;border-left:1px solid #2a2a3a">{win_cell}</td>'
            f'<td style="padding:5px 4px;text-align:right">{market_cell}</td>'
            f'<td style="padding:5px 4px;text-align:right">{gap_cell}</td>'
            f'<td style="padding:5px 4px;text-align:center">{grade_cell}</td>'
            f'<td style="padding:5px 4px;text-align:right">{top2_cell}</td>'
            f'<td style="padding:5px 4px;text-align:right">{top3_cell}</td>'
            f'</tr>'
        )

    def _th(label: str, align: str = "center", w: str = "") -> str:
        ws = f'width:{w};' if w else ''
        return (
            f'<th style="padding:4px 4px;text-align:{align};color:#555;'
            f'font-size:10px;font-weight:normal;{ws}">{label}</th>'
        )

    table = (
        '<div style="overflow-x:auto">'
        '<table style="width:100%;min-width:976px;border-collapse:collapse;'
        'font-family:sans-serif;font-size:12px">'
        '<thead><tr style="border-bottom:1px solid #2a2a3a">'
        + _th("AI順", w="36px")
        + _th("着順", w="60px")
        + _th("馬番", w="40px")
        + _th("馬名 / 騎手", "left")
        + _th("人気", w="50px")
        + _th("予測群", w="64px")
        + _th("能力値", "left", "110px")
        + _th("能力勝率", "right", "58px")
        + _th("能↑↓", "right", "54px")
        + f'<th style="padding:4px 4px;text-align:right;color:#555;font-size:10px;font-weight:normal;width:56px;border-left:1px solid #2a2a3a">AI勝率</th>'
        + _th("市場勝率", "right", "64px")
        + _th("乖離", "right", "56px")
        + _th("妙味度", "right", "48px")
        + _th("2着内率", "right", "58px")
        + _th("3着内率", "right", "58px")
        + '</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        '</table></div>'
    )
    st.markdown(table, unsafe_allow_html=True)


def _render_race_cards(races: List[Dict]) -> None:
    def _race_no_desc_key(r: Dict) -> int:
        rnum = str(r.get("race_number") or "")
        if rnum.endswith("R"):
            rnum = rnum[:-1]
        try:
            return int(rnum)
        except Exception:
            return -1

    for race in sorted(races, key=_race_no_desc_key, reverse=True):
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
                _render_column_legend()
                _render_horse_table(horses, race["status"])

                # 各馬 傾向分析（trend_analyzer）
                _ta_horses = [h for h in horses if (h.get("feature_dict") or {}).get("trend_analyzer_result")]
                if _ta_horses:
                    with st.expander("📋 各馬傾向分析", expanded=False):
                        for h in sorted(_ta_horses, key=lambda x: float((x.get("feature_dict") or {}).get("win_prob") or 0), reverse=True):
                            _ta = (h.get("feature_dict") or {}).get("trend_analyzer_result") or {}
                            _match   = _ta.get("trend_match_items") or []
                            _risk    = _ta.get("trend_risk_items") or []
                            _summary = _ta.get("trend_summary") or ""
                            _adj     = float(_ta.get("trend_adjustment") or 1.0)
                            _horse   = h.get("horse_name") or "-"
                            if not _match and not _risk:
                                continue
                            adj_pct   = round((_adj - 1.0) * 100, 1)
                            adj_str   = f"+{adj_pct}%" if adj_pct >= 0 else f"{adj_pct}%"
                            adj_color = "#2ecc71" if adj_pct > 0 else ("#e74c3c" if adj_pct < 0 else "#aaaaaa")
                            st.markdown(
                                f'<div style="background:#1e1e2e;border-radius:6px;padding:8px 12px;margin-bottom:6px;border:1px solid #333">'
                                f'<b>{_horse}</b> '
                                f'<span style="color:{adj_color};font-weight:bold;">傾向補正 {adj_str}</span>',
                                unsafe_allow_html=True,
                            )
                            if _summary:
                                st.caption(_summary)
                            if _match:
                                st.caption("好材料: " + " ／ ".join(_match))
                            if _risk:
                                st.caption("⚠ 懸念: " + " ／ ".join(_risk))
                            st.markdown("</div>", unsafe_allow_html=True)

                score_debug_rows = race.get("score_debug_rows") or []
                top_vs_bottom_debug = race.get("top_vs_bottom_debug") or {}
                if score_debug_rows or top_vs_bottom_debug:
                    with st.expander("🔧 デバッグ情報", expanded=False):
                        if score_debug_rows:
                            st.markdown("**AIスコア中間値デバッグ**")
                            st.dataframe(pd.DataFrame(score_debug_rows), width="stretch", hide_index=True)
                        if top_vs_bottom_debug:
                            st.markdown("**AI1位 vs AI最下位 比較**")
                            compare_rows = [
                                {"区分": "AI1位", **(top_vs_bottom_debug.get("top") or {})},
                                {"区分": "AI最下位", **(top_vs_bottom_debug.get("bottom") or {})},
                                {"区分": "差分", **(top_vs_bottom_debug.get("diff") or {})},
                            ]
                            st.dataframe(pd.DataFrame(compare_rows), width="stretch", hide_index=True)


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
