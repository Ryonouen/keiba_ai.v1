import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import datetime
import re
import shutil
import requests
from bs4 import BeautifulSoup
from pathlib import Path

if "result" not in st.session_state:
    st.session_state.result = None

from race_ai_engine import (
    analyze_race,
    apply_bloodline_and_track_bias_to_result,
    recommend_bets,
    apply_simple_odds,
    refresh_result_payload,
)
from value_ai import (
    build_ev_table,
    build_ticket_ev_table,
    detect_value_horses,
    detect_danger_favorites_v3,
    classify_race_structure,
    recommend_bet_plan,
    recommend_all_bet_types,
    recommend_betmaster_plans,
    select_primary_betmaster,
    assign_marks,
    assign_roles,
    detect_rescue_candidates,
    VALUE_GAP_MIN,
    DANGER_GAP_MIN,
)
from result_store import (
    build_race_record,
    save_race_result,
    load_race_results,
    update_race_result,
    delete_race_result,
    check_bet_hit,
)
from review_ai import build_review
from review_engine import build_review_result, generate_review_tags
from analytics_ai import build_full_analytics, compare_threshold_sensitivity


def pct(x):
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return "-"


def num(x, digits=2):
    try:
        val = float(x)
        if pd.isna(val):
            return "-"
        return round(val, digits)
    except Exception:
        return "-"


def jp_style_label(x: str) -> str:
    mapping = {
        "front": "逃げ",
        "stalker": "先行",
        "closer": "差し",
        "unknown": "不明",
        "逃げ": "逃げ",
        "先行": "先行",
        "差し": "差し",
        "不明": "不明",
    }
    return mapping.get(str(x), str(x))


def difficulty_label(probs):
    if len(probs) <= 1:
        return "判定不可"
    diff = probs[0] - probs[1]
    if diff > 0.15:
        return "★ 低い（堅いレース）"
    elif diff > 0.08:
        return "★★ 普通"
    elif diff > 0.04:
        return "★★★ やや荒れる"
    return "★★★★ 波乱レース"


def roi_label(x):
    try:
        x = float(x)
    except Exception:
        return "-"
    return f"{x * 100:.1f}%"


def safe_float(value, default=0.0):
    try:
        if value in [None, "", "-"]:
            return default
        return float(value)
    except Exception:
        return default


def extract_first_float(text: str):
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", str(text).replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def get_predicted_odds_from_netkeiba(race_url: str):
    """
    netkeibaの出馬表/競馬新聞ページから予想単勝オッズを取得する。
    取得できた場合は {馬番 or 馬名: オッズ} の辞書を返す。
    """
    if not race_url:
        return {}

    def build_candidate_urls(url: str):
        urls = []
        if url:
            urls.append(url)
        if "shutuba.html" in url:
            urls.append(url.replace("shutuba.html", "newspaper.html"))
        if "newspaper.html" in url:
            urls.append(url.replace("newspaper.html", "shutuba.html"))
        seen = set()
        unique = []
        for u in urls:
            if u and u not in seen:
                unique.append(u)
                seen.add(u)
        return unique

    def looks_like_odds_value(val: float) -> bool:
        return 1.0 <= val <= 500.0

    def score_odds_candidate(val: float, raw_text: str, cls_text: str) -> float:
        score = 0.0
        text = (raw_text or "").strip()
        cls = (cls_text or "").lower()

        if "odds" in cls:
            score += 10
        if "popular" in cls or "ninki" in cls:
            score += 4
        if "." in text:
            score += 2
        if 1.0 <= val <= 30.0:
            score += 1.5
        elif 30.0 < val <= 100.0:
            score += 1.0

        # 斤量っぽい数値を除外
        if 50.0 <= val <= 60.0:
            score -= 6

        return score

    def parse_row_based(soup: BeautifulSoup):
        odds_map = {}

        for tr in soup.select("tr"):
            horse_name = None
            horse_no = None
            odds_val = None
            odds_score = -999.0

            name_node = (
                tr.select_one("td.HorseName a")
                or tr.select_one("td[class*='HorseName'] a")
                or tr.select_one("a[href*='/horse/']")
            )
            if name_node and name_node.get_text(strip=True):
                horse_name = name_node.get_text(strip=True)
            else:
                continue

            no_node = (
                tr.select_one("td.Umaban")
                or tr.select_one("td[class*='Umaban']")
                or tr.select_one("td.Horse_Num")
                or tr.select_one("td[class*='Horse_Num']")
            )
            if no_node:
                no_val = extract_first_float(no_node.get_text(" ", strip=True))
                if no_val is not None:
                    horse_no = str(int(no_val))

            for cell in tr.select("td, th, span, div"):
                text = cell.get_text(" ", strip=True)
                if not text:
                    continue
                cls_text = " ".join(cell.get("class", []))
                found = re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))
                for x in found:
                    try:
                        val = float(x)
                    except Exception:
                        continue
                    if not looks_like_odds_value(val):
                        continue
                    cand_score = score_odds_candidate(val, text, cls_text)
                    if cand_score > odds_score:
                        odds_score = cand_score
                        odds_val = val

            if odds_val is not None:
                if horse_no is not None:
                    odds_map[horse_no] = odds_val
                if horse_name:
                    odds_map[horse_name] = odds_val

        return odds_map

    def parse_table_based(url: str, html_text: str):
        odds_map = {}
        try:
            tables = pd.read_html(html_text)
        except Exception:
            return odds_map

        for tbl in tables:
            df_tbl = tbl.copy()
            df_tbl.columns = [str(c) for c in df_tbl.columns]

            horse_col = None
            no_col = None
            odds_col = None

            for c in df_tbl.columns:
                c_str = str(c)
                if horse_col is None and any(k in c_str for k in ["馬名", "Horse"]):
                    horse_col = c
                if no_col is None and any(k in c_str for k in ["馬番", "Umaban", "馬 番"]):
                    no_col = c
                if odds_col is None and any(k in c_str.lower() for k in ["odds", "単勝", "予想オッズ"]):
                    odds_col = c

            if horse_col is None or odds_col is None:
                continue

            for _, r in df_tbl.iterrows():
                horse_name = str(r.get(horse_col, "")).strip()
                if not horse_name or horse_name == "nan":
                    continue

                odds_val = extract_first_float(r.get(odds_col, ""))
                if odds_val is None or not looks_like_odds_value(float(odds_val)):
                    continue

                horse_no = None
                if no_col is not None:
                    no_val = extract_first_float(r.get(no_col, ""))
                    if no_val is not None:
                        horse_no = str(int(no_val))

                if horse_no is not None:
                    odds_map[horse_no] = float(odds_val)
                odds_map[horse_name] = float(odds_val)

        return odds_map

    candidate_urls = build_candidate_urls(race_url)

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        "Referer": "https://race.netkeiba.com/",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    best_map = {}

    for target_url in candidate_urls:
        try:
            res = requests.get(target_url, headers=headers, timeout=12)
            if res.status_code != 200:
                continue

            html_text = res.text
            soup = BeautifulSoup(html_text, "html.parser")

            table_map = parse_table_based(target_url, html_text)
            row_map = parse_row_based(soup)

            merged = {}
            merged.update(table_map)
            merged.update(row_map)

            horse_name_count = len([k for k in merged.keys() if not str(k).isdigit()])
            if horse_name_count > len([k for k in best_map.keys() if not str(k).isdigit()]):
                best_map = merged

            if horse_name_count >= 8:
                return merged

        except Exception:
            continue

    return best_map


# -----------------------------
# TODAY RACE FETCH (netkeiba)
# -----------------------------

def get_today_races():

    try:

        today = datetime.datetime.now().strftime("%Y%m%d")

        url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={today}"

        headers = {"User-Agent": "Mozilla/5.0"}

        res = requests.get(url, headers=headers, timeout=10)

        soup = BeautifulSoup(res.text, "html.parser")

        races = []

        for a in soup.select("a[href*='race_id']"):

            href = a.get("href", "")
            race_name = a.text.strip()

            if "race_id=" in href:

                race_id = href.split("race_id=")[1][:12]

                # ⭐ 重賞のみ取得
                if any(g in race_name for g in ["G1","Ｇ１","G2","Ｇ２","G3","Ｇ３"]):

                    races.append({
                        "race_id": race_id,
                        "name": race_name,
                        "url": f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
                    })

        # 重複削除
        seen = set()
        unique = []

        for r in races:
            if r["race_id"] not in seen:
                unique.append(r)
                seen.add(r["race_id"])

        return unique

    except Exception:
        return []


# -----------------------------
# PAGE
# -----------------------------

st.set_page_config(page_title="KEIBA AI", layout="wide")

# -----------------------------
# UI STYLE
# -----------------------------

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Dela+Gothic+One&family=DotGothic16&display=swap');

/*
  CSS変数を定義し、背景色・カード色・アクセントカラーなどを一元管理。
  Y2Kサイバースタイルの黒ベースとネオンカラーを採用しつつ、
  読みやすさとアクセシビリティを確保しています。
*/
:root {
  --bg-color: #0A0A23;
  --card-bg: #111827;
  --accent-green: #00FF00;   /* タブ・進捗バー・ナビ系 */
  --accent-pink: #FF00FF;    /* 後方互換用（直接使用は最小化） */
  --accent-cyan: #00D4FF;    /* メトリックカード・データ表示系 */
  --accent-gold: #FFD700;    /* セクションカード・重要判断系 */
  --text-color: #F8FAFC;
  --muted-color: #94A3B8;
}

/* 全体背景と文字色、フォント設定 */
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg-color);
  color: var(--text-color);
  font-family: 'DotGothic16', 'Inter', sans-serif;
  line-height: 1.5;
  font-size: 13px;
}

/* デフォルトの灰色余白を除去 */
[data-testid="stAppViewContainer"] > .main {
  background: var(--bg-color);
}

/* コンテナ幅と余白調整 */
.block-container {
  max-width: 1400px;
  padding-top: 1.2rem;
  padding-bottom: 1.2rem;
  padding-left: 1.2rem;
  padding-right: 1.2rem;
}

/* 全体のcolumnsに影響させず、横幅崩れだけ防ぐ */
div[data-testid="column"] {
  min-width: 0;
}

/* テキストの読みやすさを担保するため、基本カラーを統一 */
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] div,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] li,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
  color: var(--text-color);
}

/* 見出し：レトロ感あるフォントとネオンのグラデーション */
h1 {
  font-size: 30px;
  font-weight: 800;
  font-family: 'Dela Gothic One', sans-serif;
  letter-spacing: 1px;
  background: linear-gradient(90deg, var(--accent-green), var(--accent-pink));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  text-shadow: 0 0 10px rgba(0,255,0,0.6), 0 0 20px rgba(255,0,255,0.4);
}

/* タイトル・入力・ボタンなどStreamlit内部UIもフォント統一 */
button, input, textarea, select, label, span, div, p, li {
  font-family: 'DotGothic16', 'Inter', sans-serif !important;
}
/* 上部入力エリアの視認性改善 */
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
  align-self: stretch;
}

/* URL入力を大きくして主役化 */
.stTextInput input {
  min-height: 56px !important;
  font-size: 18px !important;
  padding-top: 12px !important;
  padding-bottom: 12px !important;
}

/* 軍資金入力も高さを揃える */
.stNumberInput input {
  min-height: 56px !important;
  font-size: 18px !important;
}

/* 入力ラベルを少し大きく */
.stTextInput label,
.stNumberInput label {
  font-size: 15px !important;
  font-weight: 700 !important;
}

/* AI分析ボタンを大きく */
.stButton button {
  min-height: 54px;
  font-size: 18px;
  padding: 12px 24px;
}

/* カード風パネル：ゴールド枠線 */
.card {
  background: var(--card-bg);
  border: 1px solid #374151;
  border-radius: 14px;
  padding: 16px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
  margin-bottom: 20px;
}

/* ボタン：ネオンカラーのグラデーションとホバー時の光彩 */
.stButton button {
  background: linear-gradient(90deg, var(--accent-green), var(--accent-pink));
  border: none;
  color: var(--text-color);
  font-weight: 700;
  font-size: 18px;
  border-radius: 14px;
  padding: 12px 24px;
  min-height: 54px;
  box-shadow: 0 0 12px rgba(255, 0, 255, 0.6);
  transition: box-shadow 0.2s ease;
}

.stButton button:hover {
  box-shadow: 0 0 15px rgba(0, 255, 0, 0.8), 0 0 30px rgba(255, 0, 255, 0.6);
}

/* メトリック表示：シアン枠線 */
[data-testid="stMetric"]{
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 8px;
  padding: 10px 14px;
  box-shadow: none;
}
[data-testid="stMetricLabel"] {
  color: #94a3b8 !important;
  font-weight: 500;
  font-size: 11px;
  letter-spacing: 0.04em;
  text-transform: none;
}
[data-testid="stMetricValue"] {
  color: #e2e8f0 !important;
  font-weight: 700;
  font-size: 18px !important;
  margin-top: 2px;
  line-height: 1.3;
}
[data-testid="stMetricDelta"] {
  color: #22c55e !important;
}

/* テーブル枠線と背景色 */
.stDataFrame{
  background: var(--bg-color);
  border: 1px solid var(--accent-green);
  border-radius: 14px;
}

/* タブの色：未選択時は淡いグレー、選択時は白 */
button[data-baseweb="tab"] {
  color: var(--muted-color) !important;
  font-weight: 700;
  font-family: 'DotGothic16', sans-serif;
}
button[data-baseweb="tab"][aria-selected="true"] {
  color: var(--text-color) !important;
}

/* 入力フォーム：背景とテキストカラーを統一、枠線をグリーンに */
.stTextInput input,
.stNumberInput input,
textarea {
  color: var(--text-color) !important;
  background: var(--card-bg) !important;
  border: 1px solid var(--accent-green) !important;
}

/* データフレーム内のテキストは暗い色にすることで配色バランスを維持 */
[data-testid="stDataFrame"] * {
  color: #0f172a !important;
}

/* キャプション */
[data-testid="stCaptionContainer"] {
  color: var(--muted-color) !important;
}

/* プログレスバーの色 */
.stProgress > div > div{
  background: linear-gradient(90deg, var(--accent-green), var(--accent-pink));
}

/* Expander header/body readability fix */
div[data-testid="stExpander"] {
  border: 1px solid rgba(0, 255, 0, 0.18);
  border-radius: 12px;
  background: rgba(17, 24, 39, 0.55);
  overflow: hidden;
  margin-bottom: 10px;
}

div[data-testid="stExpander"] summary {
  background: rgba(255, 255, 255, 0.04);
  color: var(--text-color) !important;
  font-size: 13px !important;
  line-height: 1.6 !important;
  padding: 10px 14px !important;
}

div[data-testid="stExpander"] summary p,
div[data-testid="stExpander"] summary span,
div[data-testid="stExpander"] summary div {
  color: var(--text-color) !important;
  font-size: 13px !important;
  line-height: 1.6 !important;
  white-space: normal !important;
  word-break: break-word !important;
}

div[data-testid="stExpanderDetails"] {
  padding-top: 6px;
}

/* Number input readability */
[data-testid="stNumberInput"] {
  margin-bottom: 4px;
}

/* NumberInputは列幅いっぱいまで使用 */
div[data-testid="column"] [data-testid="stNumberInput"] {
  width: 100%;
}

/* ── selectbox：ライトモード対策・強制ダーク固定 ── */
[data-testid="stSelectbox"] > div > div {
  background-color: var(--card-bg) !important;
  color: var(--text-color) !important;
  border: 1px solid var(--accent-cyan) !important;
  border-radius: 10px !important;
}
[data-testid="stSelectbox"] svg {
  fill: var(--text-color) !important;
}
/* ドロップダウンリスト */
[data-baseweb="select"] [role="listbox"],
[data-baseweb="popover"] [role="option"],
[data-baseweb="menu"] {
  background-color: var(--card-bg) !important;
  color: var(--text-color) !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="menu"] li:hover {
  background-color: #1a2540 !important;
}

/* ── dataframe / table：ライトモード対策・強制ダーク固定 ── */
[data-testid="stDataFrame"] iframe,
[data-testid="stDataFrame"] > div {
  background: var(--card-bg) !important;
  color: var(--text-color) !important;
}
table {
  background: var(--card-bg) !important;
  color: var(--text-color) !important;
  border-collapse: collapse;
  width: 100%;
}
thead tr th {
  background: #0d1a2e !important;
  color: var(--muted-color) !important;
  font-size: 12px;
  padding: 8px 12px;
  border-bottom: 1px solid var(--accent-green);
}
tbody tr td {
  background: var(--card-bg) !important;
  color: var(--text-color) !important;
  padding: 7px 12px;
  border-bottom: 1px solid #1e2a40;
}
tbody tr:hover td {
  background: #1a2540 !important;
}

/* ── タブ：視認性向上 ── */
/* タブリスト全体 */
[data-testid="stTabs"] [role="tablist"] {
  gap: 6px;
  border-bottom: 2px solid var(--accent-green);
  padding-bottom: 0;
}

/* 各タブボタン */
[data-testid="stTabs"] [role="tab"] {
  background: var(--card-bg);
  border: 1px solid #2a3550;
  border-bottom: none;
  border-radius: 10px 10px 0 0;
  padding: 10px 16px;
  font-size: 13px;
  font-weight: 700;
  color: var(--muted-color) !important;
  transition: background 0.15s, color 0.15s, box-shadow 0.15s;
  white-space: nowrap;
}

[data-testid="stTabs"] [role="tab"]:hover {
  background: #1a2540;
  color: var(--text-color) !important;
}

/* アクティブタブ：ネオングリーンで強調 */
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  background: linear-gradient(180deg, #0d2a1a 0%, var(--card-bg) 100%);
  border-color: var(--accent-green);
  border-bottom: 2px solid var(--bg-color);
  color: var(--accent-green) !important;
  box-shadow: 0 0 10px rgba(0,255,0,0.35), inset 0 -2px 0 var(--accent-green);
  font-size: 14px;
}

</style>
    """,
    unsafe_allow_html=True,
)

st.title("🐎 KEIBA AI ANALYZER")

# -----------------------------
# INPUT
# -----------------------------

col1, col2, col3 = st.columns([5.0, 1.6, 2.2], vertical_alignment="top")

with col1:
    race_url = st.text_input(
        "netkeiba URL（出馬表 / 競馬新聞どちらでも可）",
        value="",
        placeholder="例: https://race.netkeiba.com/race/shutuba.html?race_id=202406030811"
    )
    st.caption("※ 出馬表URL（shutuba.html）または競馬新聞URL（newspaper.html）のどちらか1つだけ入力してください。AIが不足ページを自動補完します。")

with col2:
    # bankroll = st.number_input("軍資金", min_value=100, value=1000, step=100)  # 将来復活予定
    bankroll = 100  # 1枚100円固定

with col3:
    track_condition_input = st.selectbox(
        "馬場状態",
        options=["未選択", "良", "稍重", "重", "不良"],
        index=0,
        help="当日の馬場状態を選択。過去同条件レースとの傾向比較に使用します。",
    )
    _track_condition_value = None if track_condition_input == "未選択" else track_condition_input

analyze_col1, analyze_col2, analyze_col3 = st.columns([2.2, 1.8, 4.0])
with analyze_col1:
    analyze = st.button("AI分析を実行", use_container_width=True)
with analyze_col2:
    if st.button("🗑️ キャッシュをクリア", use_container_width=True, help="同じレースを再分析しても古い結果が出る場合に使用"):
        _cache_dir = Path(".keiba_cache")
        if _cache_dir.exists():
            shutil.rmtree(_cache_dir, ignore_errors=True)
        st.session_state.result = None
        st.success("キャッシュを削除しました。再度「AI分析を実行」してください。")

# -----------------------------
# ANALYZE
# -----------------------------

# --- Run analysis when button is pressed ---
if analyze:
    if not race_url:
        st.warning("URLを入力してください")
    else:
        with st.spinner("AI分析中..."):
            try:
                st.write("レースデータ取得中...")
                result = analyze_race(race_url)
                # win_odds_scraped が取得できていれば win_odds に自動反映（手動入力不要）
                _auto_odds_count = 0
                for _f in result.get("features", []):
                    _scraped = _f.get("win_odds_scraped")
                    if _scraped is not None and _f.get("win_odds") is None:
                        _f["win_odds"] = float(_scraped)
                        _auto_odds_count += 1
                if _auto_odds_count > 0:
                    st.write(f"✅ 単勝オッズを {_auto_odds_count} 頭分自動反映しました")
                st.session_state.result = result
            except Exception as e:
                st.error("AI分析エラー")
                st.exception(e)

# --- Use stored result so Streamlit reruns don't reset the page ---
if st.session_state.result:

    result = st.session_state.result

    # 馬場状態を全 feature に注入（UI セレクトボックスから）
    for _f in result.get("features", []):
        _f["track_condition"] = _track_condition_value

    result = apply_bloodline_and_track_bias_to_result(result)

    features = result["features"]

    df = pd.DataFrame(features)

    # --- Recalculate AI metrics so ai_power_index / probabilities are populated ---
    result = refresh_result_payload(result)
    df = pd.DataFrame(result["features"])

    # -------------------------------------------------
    # ADVANCED AI FEATURES (Version10)
    # We wrap adjustments in a function so it can be reused
    # after every refresh_result_payload call.
    # -------------------------------------------------

    def apply_ai_adjustments(df):

        # --- ① Newspaper mark score ---
        mark_weights = {"◎": 1.0, "○": 0.7, "▲": 0.4, "△": 0.2}

        if "newspaper_mark" in df.columns:
            df["mark_score"] = df["newspaper_mark"].map(mark_weights).fillna(0)
        else:
            df["mark_score"] = 0

        # --- ② Last 3F performance index ---
        if "last3f" in df.columns:
            try:
                max3f = df["last3f"].max()
                min3f = df["last3f"].min()
                if max3f != min3f:
                    df["last3f_index"] = (max3f - df["last3f"]) / (max3f - min3f)
                else:
                    df["last3f_index"] = 0.5
            except:
                df["last3f_index"] = 0.5
        else:
            df["last3f_index"] = 0.5

        # --- ③ Jockey adjustment ---
        jockey_weights = {
            "ルメール": 1.15,
            "川田": 1.13,
            "戸崎": 1.10,
            "坂井": 1.08,
            "横山武": 1.07,
            "武豊": 1.06,
        }

        def jockey_adj(name):
            if not isinstance(name, str):
                return 1.0
            for k, v in jockey_weights.items():
                if k in name:
                    return v
            return 1.0

        if "jockey" in df.columns:
            df["jockey_index"] = df["jockey"].apply(jockey_adj)
        else:
            df["jockey_index"] = 1.0

        # --- ④ Gate bias ---
        if "gate" in df.columns:
            max_gate = df["gate"].max()
            df["gate_bias"] = 1 - (df["gate"] / (max_gate + 1))
        else:
            df["gate_bias"] = 0.5

        # --- Final AI adjustment factor ---
        df["ai_adjustment_factor"] = (
            (1 + df["mark_score"] * 0.1)
            * df["jockey_index"]
            * (1 + df["last3f_index"] * 0.05)
            * (1 + df["gate_bias"] * 0.03)
        )

        df["ai_adjustment_factor"] = df["ai_adjustment_factor"].fillna(1.0)

        # --- Ensure base AI power exists ---
        # If engine did not generate ai_power_index, create a base score
        if "ai_power_index" not in df.columns or df["ai_power_index"].isnull().all():

            # base probability
            if "win_prob" in df.columns:
                base_win = pd.to_numeric(df["win_prob"], errors="coerce").fillna(0)
            else:
                base_win = pd.Series([0] * len(df))

            # fallback to montecarlo probability
            if base_win.sum() == 0 and "montecarlo_win_prob" in df.columns:
                base_win = pd.to_numeric(df["montecarlo_win_prob"], errors="coerce").fillna(0)

            # last3f index
            if "last3f_index" in df.columns:
                base_last3f = df["last3f_index"].fillna(0.5)
            else:
                base_last3f = 0.5

            # jockey index
            if "jockey_index" in df.columns:
                base_jockey = df["jockey_index"].fillna(1.0)
            else:
                base_jockey = 1.0

            # Base AI score
            df["ai_power_index"] = (
                (base_win * 100)
                + (base_last3f * 40)
                + (base_jockey * 15)
            )

        # Avoid NaN
        df["ai_power_index"] = df["ai_power_index"].fillna(0)
        # Guarantee a visible baseline score
        df["ai_power_index"] = df["ai_power_index"] + 20

        # Apply AI adjustment factor
        df["ai_power_index"] = df["ai_power_index"] * df["ai_adjustment_factor"]
        # Stabilize AI power scale
        df["ai_power_index"] = df["ai_power_index"].fillna(0)
        df.loc[df["ai_power_index"] <= 0, "ai_power_index"] = 20
        df["ai_power_index"] = df["ai_power_index"].clip(0, 200)

        return df

    # Apply adjustments once after refresh
    df = apply_ai_adjustments(df)

    # -------------------------------------------------
    # ADVANCED AI MODULES
    # ① Popularity Overrating Detection (危険人気馬AI)
    # ② Pace Simulation from running styles
    # ③ Distance Extension Suitability
    # -------------------------------------------------

    # ① 人気過剰AI（危険人気馬検出）
    try:
        if "win_odds" in df.columns:
            df["implied_prob"] = 1 / df["win_odds"].replace(0, 999)
            df["danger_gap"] = df["implied_prob"] - df["win_prob"]
            danger_df = df[df["danger_gap"] > 0.05].copy()
            result["danger_favorites"] = danger_df[[
                "horse_name", "win_prob", "win_odds", "danger_gap"
            ]].to_dict("records")
        else:
            result["danger_favorites"] = []
    except Exception:
        result["danger_favorites"] = []

    # ①.5 オッズ歪みAI（VALUE検出）
    try:
        if "win_odds" in df.columns:

            # 市場確率
            df["market_prob"] = 1 / df["win_odds"].replace(0, 999)

            # AI確率とのズレ
            df["prob_gap"] = df["win_prob"] - df["market_prob"]

            # 歪み指数（AI評価 / 市場評価）
            df["odds_distortion_index"] = (
                df["win_prob"] / df["market_prob"]
            ).replace([float("inf")], 0).fillna(0)

            # VALUEラベル
            def value_flag(row):
                if row["odds_distortion_index"] >= 1.4:
                    return "SUPER_VALUE"
                elif row["odds_distortion_index"] >= 1.2:
                    return "VALUE"
                elif row["odds_distortion_index"] <= 0.7:
                    return "DANGER"
                return "FAIR"

            df["value_flag"] = df.apply(value_flag, axis=1)

            value_df = df[df["value_flag"].isin(["SUPER_VALUE", "VALUE"])]

            result["value_horses"] = value_df[[
                "horse_name",
                "win_prob",
                "win_odds",
                "odds_distortion_index",
                "value_flag"
            ]].to_dict("records")

        else:
            result["value_horses"] = []

    except Exception:
        result["value_horses"] = []

    # ② 展開シミュレーションAI（逃げ馬数からペース判定）
    try:
        pace_counts = {"逃げ": 0, "先行": 0, "差し": 0}

        for _, r in df.iterrows():
            style = r.get("running_style")
            if style == "front":
                pace_counts["逃げ"] += 1
            elif style == "stalker":
                pace_counts["先行"] += 1
            elif style == "closer":
                pace_counts["差し"] += 1

        if pace_counts["逃げ"] >= 3:
            predicted_pace = "ハイペース"
        elif pace_counts["逃げ"] == 0:
            predicted_pace = "スローペース"
        else:
            predicted_pace = "平均"

        if "race_meta" not in result:
            result["race_meta"] = {}

        result["race_meta"]["predicted_pace"] = predicted_pace
        result["pace_balance"] = pace_counts

    except Exception:
        pass

    # ③ 距離適性AI（距離延長・短縮補正）
    try:
        target_distance = result.get("race_meta", {}).get("target_distance")

        if target_distance and "distance_fit_index" in df.columns:
            if isinstance(target_distance, str):
                import re
                m = re.search(r"(\d+)", target_distance)
                if m:
                    target_distance = int(m.group(1))

            if isinstance(target_distance, (int, float)):
                distance_base = pd.to_numeric(df["distance_fit_index"], errors="coerce").fillna(0.5)
                df["distance_adjustment"] = 1 + (distance_base - 0.5) * 0.1
                df["ai_power_index"] = df["ai_power_index"] * df["distance_adjustment"]

    except Exception:
        pass

    # ④ コース適性AI（競馬場適性補正）
    try:
        target_course = result.get("race_meta", {}).get("target_course")

        # 既存AIで算出されている track_bias_index を利用
        if "track_bias_index" in df.columns:

            # コース適性補正（0.5を基準）
            course_base = pd.to_numeric(df["track_bias_index"], errors="coerce").fillna(0.5)
            df["course_adjustment"] = 1 + (course_base - 0.5) * 0.08

            # AIパワーに反映
            df["ai_power_index"] = df["ai_power_index"] * df["course_adjustment"]

            # UI用ランキング保存
            course_rank = df[["horse_name", "track_bias_index"]].copy()
            course_rank = course_rank.sort_values("track_bias_index", ascending=False)

            result["course_bias_ranking"] = course_rank.to_dict("records")

    except Exception:
        pass

    if "ai_power_index" in df.columns:
        df["ai_power_index"] = pd.to_numeric(df["ai_power_index"], errors="coerce").fillna(0).clip(0, 200)
    result["features"] = df.to_dict("records")


    if df.empty:
        st.error("出馬表データを取得できませんでした。URLまたは対象ページを確認してください。")
        st.stop()

    # 出馬表順（馬番優先、無ければ枠順）— 文字列混入に備えて数値変換
    def _sort_by(col):
        return df.assign(**{col: pd.to_numeric(df[col], errors="coerce")}).sort_values(col, na_position="last")

    if "horse_number" in df.columns and pd.to_numeric(df["horse_number"], errors="coerce").notna().any():
        df_sorted = _sort_by("horse_number")
    elif "umaban" in df.columns and pd.to_numeric(df["umaban"], errors="coerce").notna().any():
        df_sorted = _sort_by("umaban")
    elif "gate" in df.columns and pd.to_numeric(df["gate"], errors="coerce").notna().any():
        df_sorted = _sort_by("gate")
    else:
        df_sorted = df.copy()

    # -----------------------------
    # MANUAL ODDS INPUT / FETCH
    # -----------------------------

    with st.expander("🎯 単勝オッズ入力", expanded=False):
        st.caption("各馬の単勝オッズを入力してください。入力するとEV計算・危険馬判定が有効になります。")

        head1, head2 = st.columns([5.4, 1.8], gap="small")
        head1.markdown("**馬名**")
        head2.markdown("**単勝オッズ**")

        current_manual_odds = {}  # horse_name → float

        for _, row in df_sorted.iterrows():
            horse_no   = row.get("horse_number", row.get("umaban", row.get("gate", "")))
            horse_name = str(row.get("horse_name", ""))
            # 馬名をキーに使うため記号を除去して安定化
            _hkey = re.sub(r"[^\w\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]", "_", horse_name)
            widget_key = f"manual_win_odds_{_hkey}"

            has_horse_no = pd.notna(horse_no) and str(horse_no).strip() not in ["", "nan", "None"]
            if has_horse_no:
                try:
                    label = f"{int(float(horse_no))}番 {horse_name}"
                except Exception:
                    label = f"{horse_no}番 {horse_name}"
            else:
                label = horse_name

            current_val = safe_float(row.get("win_odds", 0.0), 0.0)

            row_col1, row_col2 = st.columns([5.4, 1.8], gap="small", vertical_alignment="center")
            with row_col1:
                st.markdown(
                    f"<div style='min-height: 38px; display: flex; align-items: center; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'><b>{label}</b></div>",
                    unsafe_allow_html=True,
                )
            with row_col2:
                val = st.number_input(
                    "オッズ",
                    min_value=0.0,
                    step=0.1,
                    value=float(current_val),
                    key=widget_key,
                    label_visibility="collapsed",
                )

            current_manual_odds[horse_name] = float(val)

        if st.button("入力した単勝オッズをAIに反映", key="apply_manual_win_odds_btn"):
            for horse_name, odds in current_manual_odds.items():
                df.loc[df["horse_name"] == horse_name, "win_odds"] = odds

            result["features"] = df.to_dict("records")
            result = refresh_result_payload(result)
            st.session_state.result = result
            df = pd.DataFrame(result["features"])
            df = apply_ai_adjustments(df)
            result["features"] = df.to_dict("records")
            st.session_state.result = result
            st.rerun()

    # -----------------------------
    # MANUAL RUNNING STYLE INPUT (脚質手動補正)
    # -----------------------------

    with st.expander("⚙️ 脚質手動補正（クリックで開く）", expanded=True):
        st.caption("競馬新聞から脚質を取得できない場合のみ使用してください。脚質変更を行うと展開AIと勝率が再計算されます。")

        style_mapping = {
            "自動": None,
            "逃げ": "front",
            "先行": "stalker",
            "差し": "closer",
            "追込": "closer",
        }

        manual_styles = {}  # horse_name → style str

        # --- 出馬表順 UI（枠順順）---
        st.markdown("### 🐎 出馬表順（脚質設定）")
        st.caption("出馬表と同じ順番で脚質を変更できます")

        sorted_rows = list(df_sorted.iterrows())
        half = (len(sorted_rows) + 1) // 2
        cols = st.columns(2)

        for col_idx, col in enumerate(cols):
            with col:
                for idx, (_, row) in enumerate(sorted_rows[col_idx * half:(col_idx + 1) * half]):
                    horse_no   = row.get("horse_number", row.get("umaban", row.get("gate", "")))
                    horse      = str(row.get("horse_name", f"馬{col_idx * half + idx + 1}"))
                    label      = f"{horse_no}番 {horse}" if horse_no not in ("", None) else horse
                    # 馬名ベースの安定キー
                    _hkey      = re.sub(r"[^\w\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]", "_", horse)
                    choice = st.radio(
                        label,
                        ["自動", "逃げ 🟥", "先行 🟧", "差し 🟦", "追込 🟪"],
                        horizontal=True,
                        key=f"style_{_hkey}",
                    )
                    clean_choice = (
                        choice.replace(" 🟥", "")
                        .replace(" 🟧", "")
                        .replace(" 🟦", "")
                        .replace(" 🟪", "")
                    )
                    if clean_choice != "自動":
                        manual_styles[horse] = style_mapping.get(clean_choice)

        st.markdown("---")

        # --- Auto apply running style changes ---
        changed = False

        for horse_name, style in manual_styles.items():
            mask = df["horse_name"] == horse_name
            if mask.any() and df.loc[mask, "running_style"].values[0] != style:
                df.loc[mask, "running_style"] = style
                changed = True

        if changed:

            # --- Update features ---
            result["features"] = df.to_dict("records")

            # --- Recalculate AI metrics (pace balance, win prob, EV, positioning etc.) ---
            result = refresh_result_payload(result)

            # --- Persist result so Streamlit rerun keeps updated state ---
            st.session_state.result = result

            # --- Refresh dataframe used by UI ---
            df = pd.DataFrame(result["features"])

            # Reapply AI adjustments after refresh
            df = apply_ai_adjustments(df)
            result["features"] = df.to_dict("records")

            st.info("脚質変更を検知 → 展開AI・勝率AI・AIパワーを自動更新しました")
     
    race_meta = result.get("race_meta", {})
    fav = df.iloc[0]

    # ── レース情報（タブより上に固定表示） ──────────────────────────────────
    st.metric("レース", race_meta.get("race_title", "-"))

    meta1, meta2, meta3 = st.columns(3)
    meta1.metric("コース", race_meta.get("target_course", "-"))
    meta2.metric("距離", race_meta.get("target_distance", "-"))
    meta3.metric("想定ペース", race_meta.get("predicted_pace", "-"))

    sub1, sub2, sub3 = st.columns(3)
    sub1.metric("レースタイプ", result.get("race_type", "判定不可"))
    sub2.metric("期待回収率", roi_label(result.get("expected_roi", 1.0)))
    sub3.metric("難易度", difficulty_label(df["win_prob"].sort_values(ascending=False).values))

    # ── AI本命（タブより上に固定表示） ──────────────────────────────────────
    st.markdown(
        '<div style="background:#1e293b;padding:10px 16px;border-radius:8px;margin:24px 0 10px 0;border-left:4px solid #ffffff">'
        '<div style="color:#f1f5f9;font-size:1.05rem;font-weight:700;letter-spacing:0.01em">🏆 AI本命</div>'
        '<div style="color:#64748b;font-size:0.75rem;margin-top:1px">最有力候補馬の指標</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    mark = fav.get("newspaper_mark", "")
    c1.metric("馬名", f'{fav["horse_name"]} {mark}')
    c2.metric("勝率", pct(fav.get("win_prob")))
    c3.metric("AIフェアオッズ", num(fav.get("fair_win_odds"), 2))
    c4.metric("AIパワー指数", num(fav.get("ai_power_index"), 1))

    try:
        ai_power_val = float(fav.get("ai_power_index", 0))
    except Exception:
        ai_power_val = 0
    ai_power_val = max(0, min(200, ai_power_val))
    st.caption("AIパワーゲージ")
    st.progress(ai_power_val / 200)
    st.write(f"AI POWER: {round(ai_power_val,1)} / 200")
    st.caption(
        f"脚質: {fav.get('pace_style_label', jp_style_label(fav.get('running_style','不明')))}"
        f" / モンテカルロ: {pct(fav.get('montecarlo_win_prob', 0))}"
        f" / 妙味指数: {fav.get('value_index', 0)}"
    )

    # ── AI信頼度（タブより上に固定表示） ────────────────────────────────────
    st.markdown(
        '<div style="background:#1e293b;padding:10px 16px;border-radius:8px;margin:24px 0 10px 0;border-left:4px solid #ffffff">'
        '<div style="color:#f1f5f9;font-size:1.05rem;font-weight:700;letter-spacing:0.01em">🧠 AI信頼度</div>'
        '<div style="color:#64748b;font-size:0.75rem;margin-top:1px">モデル確信度</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    confidence = result.get("ai_confidence", 0.5) * 100
    st.progress(int(confidence))
    st.caption(f"{confidence:.1f}%")

    # =====================================================================
    # 総合判断セクション（タブより上に固定表示 — 結論ファースト）
    # =====================================================================
    _sum_features   = result.get("features", [])
    _sum_ev         = result.get("ev_table") or []
    _sum_structure  = result.get("race_structure") or {}
    _sum_pace       = (result.get("race_meta") or {}).get("predicted_pace", "medium") or "medium"

    if _sum_ev and _sum_features:
        _sum_danger  = detect_danger_favorites_v3(_sum_ev, _sum_features, _sum_structure, _sum_pace)
        _sum_roles   = assign_roles(_sum_features, _sum_ev, _sum_structure, _sum_danger)
        _sum_plan    = recommend_bet_plan(_sum_features, _sum_ev, _sum_structure, bankroll, _sum_pace,
                                          horse_roles=_sum_roles)

        _danger_names = {d["horse_name"] for d in _sum_danger}
        _truly_names  = {d["horse_name"] for d in _sum_danger if d.get("is_truly_dangerous")}

        _fav_name     = str(fav.get("horse_name", ""))
        _fav_is_truly = _fav_name in _truly_names
        _fav_is_soft  = _fav_name in _danger_names and not _fav_is_truly

        st.markdown(
            '<div style="background:#1e293b;padding:16px 24px;border-radius:10px;margin:32px 0 16px 0;border-left:6px solid #ffffff">'
            '<div style="color:#f1f5f9;font-size:1.4rem;font-weight:900;letter-spacing:0.02em">🎯 総合判断</div>'
            '<div style="color:#64748b;font-size:0.75rem;margin-top:1px">AI推奨買い目・役割・危険馬</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if _sum_plan.get("skip"):
            st.error(f"**🚫 見送り推奨** — {_sum_plan.get('skip_reason', '')}")
        else:
            _bp_type    = _sum_plan.get("bet_type", "-")
            _bp_tix     = _sum_plan.get("tickets", [])
            _bp_stake   = _sum_plan.get("total_stake", 0)
            _bp_ev_type = _sum_plan.get("ev_type", "")
            _bp_badge   = "📊 EV比較型" if _bp_ev_type == "EV比較型" else "🗺️ 構造型" if _bp_ev_type == "構造型" else ""
            _bp_risk    = _sum_plan.get("risk_level", "")

            _hc1, _hc2, _hc3, _hc4 = st.columns([2, 2, 2, 2])
            _hc1.metric("券種", _bp_type)
            _hc2.metric("点数", f"{len(_bp_tix)}点")
            _hc3.metric("合計金額", f"¥{_bp_stake:,}")
            _hc4.metric("判定", f"{_bp_badge}　リスク:{_bp_risk}" if _bp_risk else _bp_badge)

            st.markdown("**買い目一覧**")
            for _i, _t in enumerate(_bp_tix, 1):
                _combo_str = "　―　".join(_t.get("combination", []))
                _amt = _t.get("stake", 0)
                st.markdown(
                    f"**{_i}.** &nbsp; {_combo_str} &nbsp;&nbsp; "
                    f"<span style='color:#aaa'>¥{_amt:,}</span>",
                    unsafe_allow_html=True,
                )

            st.caption(f"💡 {_sum_plan.get('reason', '-')}")

        if _fav_is_truly:
            st.warning(
                f"⚠️ **AI勝率1位「{_fav_name}」は危険人気馬（消し推奨）**　"
                f"→ 能力は高いがオッズが過大評価。買い目の軸には使いません。"
            )
        elif _fav_is_soft:
            st.info(
                f"ℹ️ AI勝率1位「{_fav_name}」は「相手なら残る」寄りの判定。"
                f"頭では消しですが、ヒモには残す可能性があります。"
            )

        _ROLE_ICON  = {"head": "🥇 頭", "axis": "🎯 軸", "himo": "🎴 ヒモ", "fade": "✖️ 消し"}
        _role_rows = []
        for _r in _sum_roles:
            _rname = _r["horse_name"]
            _is_danger_note = "（危険人気馬）" if _rname in _truly_names else "（過大人気注意）" if _rname in _danger_names else ""
            _role_rows.append({
                "役割":    _ROLE_ICON.get(_r["role"], _r["role"]),
                "馬名":    _rname + _is_danger_note,
                "AI勝率":  f"{_r['win_prob']*100:.1f}%",
                "3着内率": f"{_r['top3_prob']*100:.1f}%",
            })
        if _role_rows:
            st.dataframe(pd.DataFrame(_role_rows), use_container_width=True, hide_index=True)

        st.caption(
            "🥇 頭=単勝・馬連の軸　🎯 軸=馬連・3連複の軸　🎴 ヒモ=3連複・ワイドの押さえ　✖️ 消し=買い目から外す  \n"
            "詳細は各タブへ → **展開予測: 展開タブ** / **買い目根拠: 期待値AIタブ⑦** / **危険馬詳細: 期待値AIタブ⑥**"
        )

    # ── タブ（上記カードの下に配置） ────────────────────────────────────────
    top_tabs = st.tabs(["⚡ 能力スコア", "🏇 展開・ポジション", "🎯 推奨買い目", "🤖 AI詳細", "📋 回顧・検証", "📊 レビュー分析", "📈 バックテスト"])


    with top_tabs[0]:
        st.caption(
            "⚠️ **このタブは「能力スコア」順の参考表示です。オッズとの比較を考慮した推奨買い目は上の「総合判断」カード、または「推奨買い目」タブを確認してください。**"
        )

        st.subheader("能力スコアランキング（オッズ無視）")

        if "newspaper_mark" not in df.columns:
            df["newspaper_mark"] = ""

        rank = df[[
            "horse_name",
            "newspaper_mark",
            "win_prob",
            "place_prob",
            "fair_win_odds",
            "ai_power_index"
        ]].copy()

        rank = rank.rename(columns={
            "horse_name": "馬名",
            "newspaper_mark": "新聞印",
            "win_prob": "AI勝率",
            "place_prob": "複勝圏率",
            "fair_win_odds": "AI適正オッズ",
            "ai_power_index": "AIパワー",
        })

        rank["AI勝率"] = rank["AI勝率"].apply(lambda x: round(x * 100, 1))
        rank["複勝圏率"] = rank["複勝圏率"].apply(lambda x: round(x * 100, 1))

        st.dataframe(rank.sort_values("AI勝率", ascending=False), use_container_width=True, hide_index=True)

        with st.expander("🔍 取得診断（スクレイピング確認）", expanded=False):
            _diag_features = result.get("features", [])
            if not _diag_features:
                st.info("馬データがありません。")
            else:
                _diag_rows = []
                for _f in sorted(_diag_features, key=lambda x: float(x.get("win_prob") or 0), reverse=True):
                    _rec_count = int(_f.get("history_count") or 0)
                    _style     = _f.get("running_style") or "unknown"
                    _src       = _f.get("records_source") or "none"
                    _src_label = {"newspaper": "新聞", "fallback": "個別P", "none": "⛔なし"}.get(_src, _src)
                    _np_mark   = _f.get("newspaper_mark") or ""
                    _sire      = _f.get("sire_name") or ""
                    _age_raw   = _f.get("age")
                    _age_disp  = f"{int(_age_raw)}歳" if _age_raw is not None else "?"
                    _flags = []
                    if _rec_count == 0:
                        _flags.append("⛔近走なし")
                    if _style == "unknown":
                        _flags.append("⚠️脚質不明")
                    if _age_raw is None:
                        _flags.append("⚠️年齢不明")
                    if not _np_mark:
                        _flags.append("△印なし")
                    if not _sire:
                        _flags.append("△父名なし")
                    _diag_rows.append({
                        "馬名":   str(_f.get("horse_name") or ""),
                        "年齢":   _age_disp,
                        "近走数": _rec_count,
                        "取得元": _src_label,
                        "脚質":   _style,
                        "新聞印": _np_mark or "-",
                        "父名":   _sire or "-",
                        "欠損":   " ".join(_flags) if _flags else "✅",
                    })
                _total    = len(_diag_rows)
                _no_rec   = sum(1 for r in _diag_rows if r["近走数"] == 0)
                _no_style = sum(1 for r in _diag_rows if r["脚質"] == "unknown")
                _no_age   = sum(1 for r in _diag_rows if r["年齢"] == "?")
                _from_np  = sum(1 for r in _diag_rows if r["取得元"] == "新聞")
                _from_fb  = sum(1 for r in _diag_rows if r["取得元"] == "個別P")
                _m1, _m2, _m3, _m4, _m5 = st.columns(5)
                _m1.metric("取得頭数", f"{_total}頭")
                _m2.metric("新聞取得", f"{_from_np}頭", delta=f"個別P:{_from_fb}頭")
                _m3.metric("近走0件", f"{_no_rec}頭",
                           delta="要確認" if _no_rec > 0 else "OK",
                           delta_color="inverse" if _no_rec > 0 else "off")
                _m4.metric("脚質不明", f"{_no_style}頭",
                           delta="要確認" if _no_style > 3 else "OK",
                           delta_color="inverse" if _no_style > 3 else "off")
                _m5.metric("年齢不明", f"{_no_age}頭",
                           delta="要確認" if _no_age > 0 else "OK",
                           delta_color="inverse" if _no_age > 0 else "off")
                st.dataframe(pd.DataFrame(_diag_rows), use_container_width=True)
                st.caption(
                    "**取得元:** 新聞=新聞ページ取得（高速）/ 個別P=馬個別ページ取得（遅い・要サインイン）/ ⛔なし=取得失敗  \n"
                    "**⛔近走なし:** 最も精度に影響大。脚質推定・スコア計算が全てデフォルト値になります。  \n"
                    "**⚠️脚質不明:** 展開予測・ペース計算の精度が落ちます。近走データが少ない場合に発生しやすいです。  \n"
                    "**△父名なし:** 血統バイアス補正が機能しません（精度への影響は軽微）。"
                )

    with top_tabs[1]:
        # -------------------------------------------------
        # AIレースプロファイル
        # -------------------------------------------------
        st.subheader("🧠 AIレースプロファイル")

        profile = []

        # 枠順傾向
        if "gate" in df.columns:
            inner_ratio = (df["gate"] <= 4).mean()
            if inner_ratio > 0.5:
                profile.append("内枠有利")

        # 先行有利判定
        pace_balance = result.get("pace_balance", {})
        front = pace_balance.get("逃げ", 0)
        stalk = pace_balance.get("先行", 0)

        if front == 0:
            profile.append("スローペース濃厚")
        elif front >= 3:
            profile.append("ハイペース濃厚")

        if stalk >= 4:
            profile.append("先行馬多数")

        # 人気傾向
        if "win_odds" in df.columns:
            try:
                odds_series = pd.to_numeric(df["win_odds"], errors="coerce")
                odds_series = odds_series.dropna()

                if not odds_series.empty:
                    fav_odds = float(odds_series.min())

                    if fav_odds is not None:
                        if fav_odds > 4:
                            profile.append("混戦レース")
                        elif fav_odds < 2:
                            profile.append("一本被り")
            except Exception:
                pass

        if profile:
            for p in profile:
                st.markdown(f"• {p}")
        else:
            st.write("特徴的なレースプロファイルは検出されませんでした")

        st.subheader("AIレース展開総評")
        pace_balance = result.get("pace_balance", {})
        front = pace_balance.get("逃げ", 0)
        stalker = pace_balance.get("先行", 0)
        closer = pace_balance.get("差し", 0)

        if front >= 3:
            summary = "ハイペース想定。差し馬に有利な展開。"
        elif front == 0:
            summary = "スローペース想定。先行馬が有利。"
        else:
            summary = "平均ペース想定。能力勝負。"
        st.info(summary)

        st.subheader("脚質バランス（展開シミュレーション）")
        pace = result.get("pace_balance", {})
        fig = go.Figure(go.Bar(x=list(pace.keys()), y=list(pace.values())))
        fig.update_layout(
            title="脚質バランス",
            plot_bgcolor="#020617",
            paper_bgcolor="#020617",
            font=dict(color="white"),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("AIレース難易度")
        probs = df["win_prob"].sort_values(ascending=False).values
        st.info(difficulty_label(probs))

        st.markdown("---")
        st.subheader("ポジションマップ")
        pos_rows = result.get("positioning_map", [])
        if pos_rows:
            pos_df = pd.DataFrame(pos_rows)
            fig = go.Figure(
                data=go.Scatter(
                    x=pos_df["position_score"],
                    y=pos_df["win_prob"],
                    mode="markers+text",
                    text=pos_df["horse_name"],
                    textposition="top center",
                    marker=dict(
                        size=14,
                        color=pos_df["pace_simulation_index"],
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="展開指数"),
                    ),
                )
            )
            fig.update_layout(
                xaxis=dict(
                    title="想定位置取り",
                    tickmode="array",
                    tickvals=[1, 2, 3, 4],
                    ticktext=["逃げ", "先行", "差し", "不明"],
                ),
                yaxis=dict(title="勝率"),
                paper_bgcolor="#020617",
                plot_bgcolor="#020617",
                font=dict(color="white"),
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
            show_df = pos_df[["horse_name", "style", "win_prob", "pace_simulation_index"]].copy()
            show_df.columns = ["馬名", "脚質", "勝率", "展開指数"]
            show_df["勝率"] = show_df["勝率"].apply(lambda x: round(float(x) * 100, 1))
            st.dataframe(show_df, use_container_width=True, hide_index=True)
        else:
            st.write("ポジション分析データがありません")

    # =====================================================================
    # TAB 2: 推奨買い目
    # =====================================================================
    with top_tabs[2]:
        # ── 事前計算（表示なし） ─────────────────────────────────────────
        race_structure = result.get("race_structure") or {}
        pace_balance_ev = result.get("pace_balance", {})
        if not race_structure:
            race_structure = classify_race_structure(
                result.get("features", []), pace_balance_ev
            )
        horse_marks = result.get("horse_marks") or []
        ev_table_ui = result.get("ev_table") or []
        if not ev_table_ui:
            ev_table_ui = build_ev_table(result.get("features", []))
        if not horse_marks:
            danger_v2_tmp = result.get("danger_favorites_v2") or []
            horse_marks = assign_marks(result.get("features", []), ev_table_ui, danger_v2_tmp)
        race_pace_ev = result.get("race_meta", {}).get("predicted_pace", "medium") or "medium"
        danger_v2 = detect_danger_favorites_v3(
            ev_table_ui, result.get("features", []), race_structure, race_pace_ev,
        )
        horse_roles = assign_roles(result.get("features", []), ev_table_ui, race_structure, danger_v2)

        # ── ① AI馬券師の推奨買い目 ─────────────────────────────────────
        st.subheader("🏇 AI馬券師の推奨買い目")
        st.caption("1枚100円固定。AIが全券種を評価し、最適な組み合わせを提示します。")

        _bm_plans = recommend_betmaster_plans(
            result.get("features", []),
            race_structure,
            horse_roles=horse_roles,
            race_pace=race_pace_ev,
        )
        _bm_primary = select_primary_betmaster(_bm_plans, race_structure)

        # ── 主推奨ハイライト ───────────────────────────────────────────
        if _bm_primary:
            st.markdown("#### ★ 特におすすめ")
            _pri_legs = _bm_primary.get("formation_legs") or {}
            _pri_count = _bm_primary.get("ticket_count", 0)
            _pri_budget = _bm_primary.get("budget", 0)

            with st.container(border=True):
                st.markdown(f"**{_bm_primary['bet_type']}**　"
                            f"🎯 {_pri_count}点 × 100円 = **¥{_pri_budget:,}**")
                if _pri_legs:
                    for leg_label, leg_horses in _pri_legs.items():
                        st.markdown(f"&nbsp;&nbsp;**{leg_label}:** {'　'.join(leg_horses)}")
                st.caption(_bm_primary.get("reason", ""))
        else:
            st.warning("🚫 現在の予測精度では自信を持って推奨できる券種がありません。")

        # ── 全券種一覧 ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 全券種の評価")
        _risk_icon = {"最低": "⚪", "低": "🟢", "中": "🟡", "高": "🔴"}
        for _bp in _bm_plans:
            _is_primary = (_bm_primary is not None and _bp["bet_type"] == _bm_primary["bet_type"])
            _icon = _risk_icon.get(_bp.get("risk_level", "中"), "🟡")
            if _bp.get("confidence_ok") and _bp.get("tickets"):
                _cnt   = _bp["ticket_count"]
                _bdgt  = _bp["budget"]
                _label = (f"{_icon} **{_bp['bet_type']}**"
                          + ("　⭐特におすすめ" if _is_primary else "")
                          + f"　{_cnt}点 × 100円 = ¥{_bdgt:,}")
                with st.expander(_label, expanded=_is_primary):
                    _legs = _bp.get("formation_legs") or {}
                    if _legs:
                        for _leg_label, _leg_horses in _legs.items():
                            st.markdown(f"**{_leg_label}:** {'　'.join(_leg_horses)}")
                    st.caption(_bp.get("reason", ""))
                    _tix = _bp.get("tickets", [])
                    if _tix and len(_tix) <= 20:
                        st.dataframe(
                            pd.DataFrame([{
                                "組み合わせ": " - ".join(t.get("combination", [])),
                                "金額": f"¥{t.get('stake', 0):,}",
                            } for t in _tix]),
                            use_container_width=True,
                            hide_index=True,
                        )
                    elif _tix:
                        st.caption(f"（{len(_tix)}点の組み合わせ — 点数が多いため上位10点を表示）")
                        st.dataframe(
                            pd.DataFrame([{
                                "組み合わせ": " - ".join(t.get("combination", [])),
                                "金額": f"¥{t.get('stake', 0):,}",
                            } for t in _tix[:10]]),
                            use_container_width=True,
                            hide_index=True,
                        )
            else:
                _no_reason = _bp.get("no_pick_reason", "")
                st.markdown(f"{_icon} {_bp['bet_type']}　— 今回は選択なし"
                            + (f"（{_no_reason}）" if _no_reason else ""))

        # ── EV速報（オッズ入力時のみ）─────────────────────────────────
        ticket_evs_summary: list = []
        if ev_table_ui:
            ticket_evs_summary = build_ticket_ev_table(
                result.get("features", []),
                race_structure=race_structure,
                ev_table=ev_table_ui,
                forced_axis=None,
                horse_roles=horse_roles,
            )
            best_by_type: dict = {}
            for r in ticket_evs_summary:
                bt = r["bet_type"]
                if bt not in best_by_type or r["ev"] > best_by_type[bt]["ev"]:
                    best_by_type[bt] = r
            if best_by_type:
                st.markdown("---")
                st.markdown("**券種別 EV（参考 — オッズ入力時）**")
                sum_cols = st.columns(5)
                for ci, bt in enumerate(["単勝", "複勝", "馬連", "ワイド", "3連複"]):
                    row_r = best_by_type.get(bt)
                    if row_r:
                        sum_cols[ci].metric(bt, f"EV {row_r['ev']:.2f}",
                                            delta=f"hit:{row_r['ai_hit_prob']*100:.1f}%")
                    else:
                        sum_cols[ci].metric(bt, "-")

        st.markdown("---")

        # ── ② 消し候補（危険人気馬） ─────────────────────────────────────
        st.subheader("⛔ 消し候補（危険人気馬）")
        st.caption("人気になっているが AI 評価が低い馬。上の買い目から除外する根拠です。")
        if danger_v2:
            truly_list = [d for d in danger_v2 if d.get("is_truly_dangerous")]
            maybe_list = [d for d in danger_v2 if not d.get("is_truly_dangerous")]
            if truly_list:
                st.markdown("**⛔ 消し推奨（真に危険）**")
                for dh in truly_list:
                    odds_d = dh.get("win_odds")
                    st.markdown(
                        f"**⚠️ {dh['horse_name']}**  "
                        f"（AI勝率 {dh.get('ai_win_prob', 0)*100:.1f}% "
                        f"/ 市場勝率 {dh.get('market_win_prob', 0)*100:.1f}% "
                        f"/ オッズ {odds_d or '-'}倍）"
                    )
                    st.caption(f"理由: {dh.get('reason', '-')}")
            if maybe_list:
                st.markdown("**△ 相手なら残る（頭では危険）**")
                for dh in maybe_list:
                    odds_d = dh.get("win_odds")
                    st.markdown(
                        f"**△ {dh['horse_name']}**  "
                        f"（AI勝率 {dh.get('ai_win_prob', 0)*100:.1f}% "
                        f"/ オッズ {odds_d or '-'}倍）"
                    )
                    st.caption(f"理由: {dh.get('reason', '-')}")
        else:
            st.info("危険人気馬は検出されませんでした。")

        st.markdown("---")

        # ── ③ 馬の役割 ──────────────────────────────────────────────────
        st.subheader("馬の役割（頭 / 軸 / ヒモ / 消し）")
        st.caption("🥇 頭=単勝・馬連の軸　🎯 軸=馬連・3連複の軸　🎴 ヒモ=3連複・ワイドの押さえ　✖️ 消し=買い目から外す")
        _ROLE_LABEL = {"head": "頭候補", "axis": "軸候補", "himo": "ヒモ候補", "fade": "消し寄り"}
        _ROLE_ICON  = {"head": "🥇", "axis": "🎯", "himo": "🎴", "fade": "✖️"}
        for role_key, role_label in _ROLE_LABEL.items():
            group = [r for r in horse_roles if r["role"] == role_key]
            if not group:
                continue
            st.markdown(f"**{_ROLE_ICON[role_key]} {role_label}**")
            role_cols = st.columns(min(len(group), 4))
            for ci, r in enumerate(group):
                with role_cols[ci % 4]:
                    st.markdown(
                        f"**{r['horse_name']}**  \n"
                        f"勝率{r['win_prob']*100:.1f}% / 3着内{r['top3_prob']*100:.1f}%  \n"
                        f"_{r['reason']}_"
                    )

        st.markdown("---")

        # ── ④ 取りこぼし注意馬 ─────────────────────────────────────────
        st.subheader("⚡ 取りこぼし注意馬")
        st.caption("AI 上位ではないが 3 着以内の可能性が残る馬。3連複・ワイドで押さえておく候補です。")
        rescue_horses = detect_rescue_candidates(
            result.get("features", []), ev_table_ui, race_structure, race_pace_ev,
        )
        if rescue_horses:
            for rh in rescue_horses:
                st.markdown(
                    f"**⚡ {rh['horse_name']}**  "
                    f"（3着内 {rh['top3_prob']*100:.1f}% / 2着内 {rh['top2_prob']*100:.1f}%）"
                )
                st.caption(f"注意理由: {rh.get('reason', '-')}")
        else:
            st.info("取りこぼし注意馬はいません（AI上位馬で手堅い）。")

        st.markdown("---")

        # ── ⑤ 妙味候補 ──────────────────────────────────────────────────
        st.subheader("💡 妙味候補（穴・押さえ向き）")
        st.caption("AI 評価がオッズに対して優勢な馬。ヒモとして加えると期待値が出やすくなります。")
        value_v2 = result.get("value_horses_v2") or []
        if not value_v2:
            value_v2 = detect_value_horses(ev_table_ui, result.get("features", []), race_pace_ev)
        if value_v2:
            for vh in value_v2:
                vg_val = vh.get("value_gap")
                odds_v = vh.get("win_odds")
                # 年齢シグナルが強い懸念の場合は「強妙味」への昇格を禁止
                _age_cap = vh.get("age_label_cap")
                if _age_cap == "妙味":
                    tag = "💡 妙味"
                elif vg_val is not None and vg_val >= 0.06:
                    tag = "🔥 強妙味"
                else:
                    tag = "💡 妙味"
                st.markdown(
                    f"**{tag} {vh['horse_name']}**  "
                    f"（AI勝率 {vh.get('ai_win_prob', 0)*100:.1f}% "
                    f"/ 市場勝率 {vh.get('market_win_prob', 0)*100:.1f}% "
                    f"/ オッズ {odds_v or '-'}倍）"
                )
                st.caption(f"理由: {vh.get('reason', '-')}")
        else:
            st.info("妙味馬は検出されませんでした（オッズ入力後に再確認）。")

        st.markdown("---")

        # ── 以下 expanders（詳細参照用） ────────────────────────────────
        with st.expander("① レース構造詳細", expanded=False):
            rs_cols = st.columns(4)
            rs_cols[0].metric("レース構造", race_structure.get("structure_type", "-"))
            rs_cols[1].metric("荒れリスク", f"{int(race_structure.get('upset_risk', 0.5) * 100)}%")
            rs_cols[2].metric("展開有利脚質", {
                "closer": "差し", "stalker": "先行", "front": "逃げ"
            }.get(race_structure.get("favorable_style", ""), race_structure.get("favorable_style", "-")))
            rs_cols[3].metric("想定ペース", race_meta.get("predicted_pace", "-"))
            st.info(race_structure.get("description", ""))
            suitable = race_structure.get("suitable_bet_types", [])
            if suitable:
                st.caption(f"推奨券種候補: {' / '.join(suitable)}")

        # ③-B 騎手補正テーブル（expander）
        with st.expander("🏇 騎手補正テーブル", expanded=False):
            _jockey_rows = []
            for _f in sorted(result.get("features", []),
                              key=lambda x: float(x.get("win_prob") or 0.0), reverse=True):
                _jd    = float(_f.get("jockey_delta") or 0.0)
                _jconf = float(_f.get("jockey_confidence") or 0.0)
                _jname = _f.get("entry_jockey") or "-"
                _delta_str = f"{_jd:+.4f}" if _jd != 0.0 else "±0"
                _conf_str  = f"{_jconf:.0%}" if _jconf > 0 else "-"
                _reasons   = _f.get("jockey_reasons") or []
                _jockey_rows.append({
                    "馬名":      str(_f.get("horse_name") or ""),
                    "騎手":      _jname,
                    "補正値":    _delta_str,
                    "信頼度":    _conf_str,
                    "勝率":      f"{(_f.get('jockey_win_rate') or 0)*100:.1f}%",
                    "3着内率":   f"{(_f.get('jockey_top3_rate') or 0)*100:.1f}%",
                    "補正理由":  " / ".join(_reasons[:2]) if _reasons else "-",
                })
            if _jockey_rows:
                st.dataframe(pd.DataFrame(_jockey_rows), use_container_width=True)
                st.caption("※ 信頼度: 実データ量に応じた補正の確かさ。シード値の場合は55%固定。補正幅: -0.04〜+0.055。")

                # 馬別詳細理由（expander）
                st.markdown("**騎手補正 詳細理由**")
                for _f in sorted(result.get("features", []),
                                  key=lambda x: abs(float(x.get("jockey_delta") or 0.0)),
                                  reverse=True)[:8]:
                    _jd = float(_f.get("jockey_delta") or 0.0)
                    if abs(_jd) < 0.001:
                        continue
                    _reasons_all = _f.get("jockey_reasons") or []
                    _sign = "+" if _jd >= 0 else ""
                    _lbl  = f"{_f.get('horse_name','')} / {_f.get('entry_jockey','')}  補正: {_sign}{_jd:.4f}"
                    with st.expander(_lbl, expanded=False):
                        for _r in _reasons_all:
                            icon = "✅" if any(w in _r for w in ["高い", "良好", "信頼", "好成績", "激走", "高い勝率"]) else "⚠️"
                            st.markdown(f"{icon} {_r}")
                        _jdet = _f.get("jockey_details") or []
                        if _jdet:
                            _det_rows = [{
                                "条件":   d.get("label", ""),
                                "該当値": d.get("key", ""),
                                "補正値": f"{d.get('delta', 0):+.4f}",
                                "信頼度": f"{d.get('confidence', 0):.0%}",
                                "サンプル": d.get("rides", 0),
                            } for d in _jdet]
                            st.dataframe(pd.DataFrame(_det_rows), use_container_width=True)
            else:
                st.info("騎手データがありません。")

        st.markdown("---")

        # ⑨ 券種別EVランキング（⑧ で計算済みの ticket_evs_summary を再利用）
        with st.expander("⑨ 券種別EVランキング（上位5件）", expanded=False):
            if ev_table_ui:
                ticket_ev_rows = ticket_evs_summary if ticket_evs_summary else []
                if ticket_ev_rows:
                    ev_rank_display = []
                    for i, r in enumerate(ticket_ev_rows[:5], 1):
                        corr = r.get("correction", 1.0)
                        corr_str = f"×{corr:.2f}" if corr is not None and abs(corr - 1.0) >= 0.01 else "±0"
                        ev_rank_display.append({
                            "順位":       i,
                            "券種":       r.get("bet_type", "-"),
                            "組み合わせ": " / ".join(r.get("horses", [])),
                            "EV(補正後)": f"{r.get('ev', 0):.3f}",
                            "EV(補正前)": f"{r.get('ev_raw', r.get('ev', 0)):.3f}",
                            "補正":       corr_str,
                            "AI的中率":   f"{r.get('ai_hit_prob', 0)*100:.2f}%",
                        })
                    st.dataframe(pd.DataFrame(ev_rank_display), use_container_width=True)
                    st.caption("※ 補正 = レース構造×脚質適性×妙味/危険馬の乗数")
                else:
                    st.info("EVを計算できませんでした。")
            else:
                st.info("オッズを入力してください。")

        # ⑩ 単馬EV テーブル詳細
        with st.expander("⑩ 単馬期待値テーブル詳細", expanded=False):
            if ev_table_ui:
                ev_display = []
                for row in ev_table_ui:
                    ev_display.append({
                        "人気":          row.get("popularity_rank", "-"),
                        "馬名":          row.get("horse_name", ""),
                        "AI勝率":        f"{(row.get('ai_win_prob') or 0)*100:.1f}%",
                        "市場勝率":      f"{(row.get('market_win_prob') or 0)*100:.1f}%" if row.get("market_win_prob") else "-",
                        "単勝オッズ":    row.get("win_odds", "-"),
                        "value_gap":     f"{row.get('value_gap'):+.3f}" if row.get("value_gap") is not None else "-",
                        "AIスコア":      row.get("ai_score", "-"),
                    })
                ev_display.sort(key=lambda x: x["人気"] if isinstance(x["人気"], int) else 999)
                st.dataframe(pd.DataFrame(ev_display), use_container_width=True)
            else:
                st.info("オッズを入力してください。")

        # ⑩ 傾向シグナル（条件別過去実績補正）
        _features_for_signal = result.get("features") or []
        _has_signal = any(
            f.get("trend_signal_details") for f in _features_for_signal
        )
        with st.expander("⑩ 傾向シグナル（過去10年条件別補正）", expanded=_has_signal):
            if not _has_signal:
                _cs_err = result.get("condition_stats_error", "")
                if _cs_err:
                    st.warning(
                        f"⚠️ 条件別統計の取得に失敗しました。\n\n"
                        f"**原因**: {_cs_err}"
                    )
                else:
                    st.info(
                        "このレースは過去10年の同レース統計が取得できませんでした。\n\n"
                        "**傾向シグナルが表示されるケース**: 阪神大賞典・天皇賞などの重賞レースで、"
                        "同じ枠・距離・会場で毎年同一レースが開催されている場合のみ機能します。\n\n"
                        "通常の平場レース（毎週異なる番組）では統計が蓄積されないため表示されません。"
                    )
            else:
                # サマリーテーブル（全馬）
                _sig_rows = []
                for _f in _features_for_signal:
                    _sig = _f.get("trend_signal_details") or {}
                    _sig_rows.append({
                        "馬名":       _f.get("horse_name", ""),
                        "補正合計":   f"{_sig.get('total_trend_adjust', 0):+.3f}",
                        "強い懸念":   " / ".join(
                            f"{d['value']}({d['factor']})"
                            for d in _sig.get("strong_concerns", [])
                        ) or "-",
                        "弱い懸念":   " / ".join(
                            f"{d['value']}({d['factor']})"
                            for d in _sig.get("weak_concerns", [])
                        ) or "-",
                        "強い追い風": " / ".join(
                            f"{d['value']}({d['factor']})"
                            for d in _sig.get("strong_tailwinds", [])
                        ) or "-",
                        "弱い追い風": " / ".join(
                            f"{d['value']}({d['factor']})"
                            for d in _sig.get("weak_tailwinds", [])
                        ) or "-",
                    })
                st.dataframe(pd.DataFrame(_sig_rows), use_container_width=True)

                # 馬別詳細（各馬の上位3シグナル）
                st.markdown("**馬別シグナル詳細（非中立のみ）**")
                for _f in _features_for_signal:
                    _sig = _f.get("trend_signal_details") or {}
                    _details = _sig.get("details") or []
                    if not _details:
                        continue
                    _name = _f.get("horse_name", "")
                    _total = _sig.get("total_trend_adjust", 0)
                    _sign  = "+" if _total >= 0 else ""
                    with st.expander(
                        f"{_name}　補正合計: {_sign}{_total:.3f}", expanded=False
                    ):
                        _det_rows = []
                        for _d in _details[:5]:
                            _det_rows.append({
                                "条件":        _d.get("factor", ""),
                                "該当値":      _d.get("value", ""),
                                "シグナル":    _d.get("signal_jp", ""),
                                "好走率":      f"{_d.get('top3_rate', 0)*100:.1f}%",
                                "全体比":      f"{_d.get('overall_top3_rate', 0)*100:.1f}%",
                                "差分":        f"{_d.get('diff_top3', 0):+.3f}",
                                "補正値":      f"{_d.get('score_adjust', 0):+.4f}",
                                "サンプル数":  _d.get("sample_size", 0),
                                "理由":        _d.get("reason", ""),
                            })
                        st.dataframe(pd.DataFrame(_det_rows), use_container_width=True)

        # ② 印テーブル（全馬 AI評価・EV 参照用）
        with st.expander("② 印・AI評価一覧（全馬参照用）", expanded=False):
            st.caption("◎○▲△ = AI勝率順  ☆ = 妙味馬（value_gap 優位）  × = 危険人気馬")
            if horse_marks:
                marks_by_name_exp = {row["horse_name"]: row["mark"] for row in horse_marks}
                ev_by_name_exp    = {row["horse_name"]: row for row in ev_table_ui}
                mark_rows_exp = []
                for f in sorted(
                    result.get("features", []),
                    key=lambda x: float(x.get("win_prob") or 0.0),
                    reverse=True,
                ):
                    _name    = str(f.get("horse_name") or "")
                    _mark    = marks_by_name_exp.get(_name, "")
                    _ev_row  = ev_by_name_exp.get(_name, {})
                    _vg      = _ev_row.get("value_gap")
                    _ai_prob = float(f.get("win_prob") or 0.0)
                    _mk_prob = _ev_row.get("market_win_prob")
                    _w_odds  = _ev_row.get("win_odds")
                    mark_rows_exp.append({
                        "印":           _mark,
                        "馬名":         _name,
                        "AI勝率":       f"{_ai_prob * 100:.1f}%",
                        "市場期待勝率": f"{_mk_prob * 100:.1f}%" if _mk_prob else "-",
                        "単勝オッズ":   _w_odds if _w_odds else "-",
                        "value_gap":    f"{_vg:+.3f}" if _vg is not None else "-",
                    })
                st.dataframe(pd.DataFrame(mark_rows_exp), use_container_width=True)
            else:
                st.info("印を計算するにはオッズを入力してください。")

        # ── 分析結果を保存 ────────────────────────────────────────────
        st.markdown("---")
        with st.expander("💾 分析結果を保存（回顧・検証用）", expanded=False):
            st.caption("レース後に「回顧・検証」タブで着順を入力すると回顧コメントと成績集計が生成されます。")
            if st.button("分析結果を保存", key="btn_save_result"):
                try:
                    _bet_plan_for_save = recommend_bet_plan(
                        result.get("features", []),
                        ev_table_ui,
                        race_structure,
                        bankroll,
                        race_pace_ev,
                        horse_roles=horse_roles,
                    ) if ev_table_ui else {}
                    _record = build_race_record(
                        result,
                        bankroll,
                        _bet_plan_for_save,
                        race_url=race_url if race_url else "",
                        horse_roles=horse_roles,
                        rescue_horses=rescue_horses,
                        danger_horses=danger_v2,
                        value_horses=value_v2,
                        horse_marks=horse_marks,
                        ev_table=ev_table_ui,
                        race_structure=race_structure,
                    )
                    save_race_result(_record)
                    st.success(f"保存しました: {_record['race_name']} (ID: {_record['race_id']})")
                except Exception as _e:
                    st.error(f"保存エラー: {_e}")

    with top_tabs[3]:
        # -----------------------------
        # 過去10年レース傾向
        # -----------------------------
        st.subheader("📊 過去10年レース傾向")

        trend_summary = result.get("race_history_summary", "データなし")

        st.markdown(f"""
        <div class="card">
        <b>レース傾向</b><br>
        {trend_summary}
        </div>
        """, unsafe_allow_html=True)

        # -----------------------------
        # AIコメント
        # -----------------------------
        st.subheader("🤖 AIレース分析コメント")

        comment = result.get("ai_comment", "AIコメント生成に失敗しました")

        st.caption(
            f"レースタイプ: {result.get('race_type', '判定不可')} / "
            f"期待回収率: {roi_label(result.get('expected_roi', 1.0))}"
        )

        st.markdown(f'<div class="card">{comment}</div>', unsafe_allow_html=True)

        with st.expander("オッズ補正メモ", expanded=False):
            st.caption("上部の『🎯 単勝オッズ入力・取得』内の『手動オッズ入力』から必要時のみ入力してください。")

    # =====================================================================
    # TAB 4: 回顧・検証
    # =====================================================================
    with top_tabs[4]:
        _all_records = load_race_results()

        # ── A. 結果入力 ────────────────────────────────────────────────
        st.subheader("📋 結果入力")
        _pending = [r for r in _all_records if r.get("result") is None and not r.get("is_pass")]
        _done    = [r for r in _all_records if r.get("result") is not None]

        if not _all_records:
            st.info("保存済みレースがありません。「期待値AI」タブで分析後、「分析結果を保存」ボタンを押してください。")
        else:
            _record_options = {
                f"{r['race_date']} {r['race_name']} ({r['race_id']})": r["race_id"]
                for r in sorted(_all_records, key=lambda x: x.get("race_date", ""), reverse=True)
            }
            _selected_label = st.selectbox(
                "結果を入力するレースを選択",
                options=list(_record_options.keys()),
                key="review_select",
            )
            _selected_id = _record_options.get(_selected_label, "")
            _sel_record  = next((r for r in _all_records if r["race_id"] == _selected_id), None)

            if _sel_record:
                # 削除ボタン
                _del_col, _ = st.columns([1, 4])
                if _del_col.button("🗑️ このレースを削除", key="delete_record"):
                    from result_store import delete_race_result as _del_fn
                    _del_fn(_selected_id)
                    st.success("削除しました。")
                    st.rerun()

                _existing_result = _sel_record.get("result") or {}
                _horse_names = [h["horse_name"] for h in _sel_record.get("horses", [])]
                _bet_types_list = ["単勝", "複勝", "馬連", "ワイド", "ワイドBOX", "3連複", "3連複BOX", "3連単"]

                # AIの推奨通りかどうか（フォーム外で切り替え可能にする）
                _followed_ai = st.checkbox(
                    "✅ AIの推奨通りに購入した",
                    value=_existing_result.get("followed_ai_recommendation", True),
                    key="followed_ai",
                )

                with st.form(key="result_form"):
                    st.markdown(f"**{_sel_record['race_name']}** — {_sel_record['race_date']}")

                    # 着順
                    fc1, fc2, fc3 = st.columns(3)
                    _fo_default = _existing_result.get("finish_order", ["", "", ""])
                    _1st = fc1.selectbox("1着", [""] + _horse_names,
                                         index=(_horse_names.index(_fo_default[0]) + 1) if _fo_default and _fo_default[0] in _horse_names else 0,
                                         key="fin1")
                    _2nd = fc2.selectbox("2着", [""] + _horse_names,
                                         index=(_horse_names.index(_fo_default[1]) + 1) if len(_fo_default) > 1 and _fo_default[1] in _horse_names else 0,
                                         key="fin2")
                    _3rd = fc3.selectbox("3着", [""] + _horse_names,
                                         index=(_horse_names.index(_fo_default[2]) + 1) if len(_fo_default) > 2 and _fo_default[2] in _horse_names else 0,
                                         key="fin3")

                    st.markdown("---")

                    if _followed_ai:
                        # AI推奨通り：払戻金額のみ入力
                        _ret_amt = st.number_input("払戻金額（円）",
                                                    min_value=0, step=100,
                                                    value=int(_existing_result.get("return_amount") or 0),
                                                    key="ret_amt")
                        _actual_bet_type  = _sel_record.get("recommended_bet_type", "")
                        _actual_horses    = []
                        _actual_invest    = int(_sel_record.get("investment_amount") or 0)
                        _win_pay          = int(_existing_result.get("win_payout") or 0)
                    else:
                        # カスタム購入：券種・馬・金額を個別入力
                        st.markdown("**実際の購入内容**")
                        _c1, _c2 = st.columns(2)
                        _existing_abt = _existing_result.get("actual_bet_type", "3連複BOX")
                        _abt_idx = _bet_types_list.index(_existing_abt) if _existing_abt in _bet_types_list else 0
                        _actual_bet_type = _c1.selectbox("実際の券種", _bet_types_list,
                                                          index=_abt_idx, key="actual_bet_type")
                        _actual_invest   = _c2.number_input("購入金額（円）",
                                                             min_value=0, step=100,
                                                             value=int(_existing_result.get("actual_investment") or 0),
                                                             key="actual_invest")
                        _actual_horses = st.multiselect(
                            "購入した馬（組み合わせに含まれる馬を全て選択）",
                            _horse_names,
                            default=[h for h in _existing_result.get("actual_horses", []) if h in _horse_names],
                            key="actual_horses",
                        )
                        _ret_amt = st.number_input("払戻金額（円）",
                                                    min_value=0, step=100,
                                                    value=int(_existing_result.get("return_amount") or 0),
                                                    key="ret_amt")
                        _win_pay = 0

                    _memo = st.text_area("メモ（任意）",
                                         value=_existing_result.get("review_comment", ""),
                                         key="result_memo")

                    _submitted = st.form_submit_button("結果を保存")

                if _submitted:
                    _finish_order = [x for x in [_1st, _2nd, _3rd] if x]
                    _ai_invest    = _sel_record.get("investment_amount", 0)
                    _bet_type_s   = _sel_record.get("recommended_bet_type", "")
                    _tickets_s    = _sel_record.get("recommended_tickets", [])

                    # AI推奨の的中判定
                    _ai_hit = check_bet_hit(_bet_type_s, _tickets_s, _finish_order) if _finish_order else False

                    # 実際の購入の的中判定
                    if _followed_ai:
                        _actual_hit = _ai_hit
                        _actual_invest_final = _ai_invest
                    else:
                        # 選択した馬の組み合わせをticket形式に変換
                        _actual_tickets = [{"combination": _actual_horses}] if _actual_horses else []
                        _actual_hit = check_bet_hit(_actual_bet_type, _actual_tickets, _finish_order) if _finish_order and _actual_tickets else False
                        _actual_invest_final = _actual_invest

                    from datetime import datetime as _dt
                    _result_data = {
                        "finish_order":               _finish_order,
                        "win_payout":                 _win_pay if _win_pay > 0 else None,
                        "place_payouts":              {},
                        "hit":                        _actual_hit,
                        "return_amount":              _ret_amt,
                        "investment_amount":          _actual_invest_final,
                        "roi":                        round(_ret_amt / _actual_invest_final, 4) if _actual_invest_final > 0 else None,
                        "followed_ai_recommendation": _followed_ai,
                        "actual_bet_type":            _actual_bet_type,
                        "actual_horses":              _actual_horses,
                        "actual_investment":          _actual_invest_final,
                        "actual_hit":                 _actual_hit,
                        "ai_hit":                     _ai_hit,
                        "ai_investment":              _ai_invest,
                        "review_comment":             _memo,
                        "review_tags":                [],
                        "review_labels":              [],
                        "entered_at":                 _dt.now().isoformat(timespec="seconds"),
                    }
                    # 回顧タグを生成してから保存
                    _tmp_record = dict(_sel_record)
                    _tmp_record["result"] = _result_data
                    _rev_result = build_review_result(_tmp_record, "")
                    _result_data["review_tags"]    = _rev_result["review_tags"]
                    _result_data["review_summary"] = _rev_result["summary"]
                    _result_data["return_rate"]    = _rev_result["return_rate"]
                    _result_data["review_labels"]  = _rev_result["review_tags"]

                    update_race_result(_selected_id, _result_data)
                    _hit_icon = "○" if _actual_hit else "✗"
                    _ai_icon  = "○" if _ai_hit else "✗"
                    if _followed_ai:
                        st.success(f"保存しました。的中: {_hit_icon}")
                    else:
                        st.success(f"保存しました。実際: {_hit_icon} / AI推奨: {_ai_icon}")
                    st.rerun()

        st.markdown("---")

        # ── B. 回顧コメント ────────────────────────────────────────────
        st.subheader("🔍 直近レース回顧")
        _done_sorted = sorted(_done, key=lambda x: x.get("race_date", ""), reverse=True)[:10]

        if not _done_sorted:
            st.info("結果入力済みのレースがありません。")
        else:
            for _r in _done_sorted:
                _rd = _r.get("result", {})
                _hit_icon = "✅" if _rd.get("hit") else ("⏩" if _r.get("is_pass") else "❌")
                _roi_txt  = f"ROI {_rd.get('roi', 0):.2f}" if _rd.get("roi") is not None else "-"
                with st.expander(
                    f"{_hit_icon} {_r['race_date']} {_r['race_name']} — {_r.get('recommended_bet_type','-')} / {_roi_txt}",
                    expanded=False,
                ):
                    # 着順
                    fo = _rd.get("finish_order", [])
                    if fo:
                        st.markdown(f"**着順**: {' → '.join(fo[:3])}")
                    # 買い目
                    tickets = _r.get("recommended_tickets", [])
                    if tickets and not _r.get("is_pass"):
                        combos = ["・".join(t.get("combination", [])) for t in tickets]
                        st.markdown(f"**買い目** ({_r.get('recommended_bet_type','-')}): {' / '.join(combos)}")
                    # 回顧サマリー
                    summary = _rd.get("review_summary", "")
                    if not summary:
                        _rv_res = build_review_result(_r, _rd.get("actual_bets", ""))
                        summary = _rv_res.get("summary", "")
                    if summary:
                        st.info(summary)
                    # 回顧タグ詳細（日本語タグ優先、なければ旧ラベル）
                    from review_config import REVIEW_TAGS
                    _level_icon = {"good": "✅", "neutral": "📌", "bad": "⚠️"}
                    _tags = _rd.get("review_tags") or _rd.get("review_labels", [])
                    for _tag in _tags:
                        _tag_meta = REVIEW_TAGS.get(_tag, {})
                        _lvl  = _tag_meta.get("level", "neutral")
                        _icon = _level_icon.get(_lvl, "📌")
                        st.caption(f"{_icon} {_tag}")
                    # メモ
                    memo = _rd.get("review_comment", "")
                    if memo:
                        st.markdown(f"*メモ: {memo}*")

        st.markdown("---")

        # ── C. 成績サマリー ────────────────────────────────────────────
        st.subheader("📊 成績サマリー")

        if not _done:
            st.info("集計データがありません（結果入力済みのレースが0件です）。")
        else:
            try:
                _analytics = build_full_analytics(_all_records)
                _overall   = _analytics["overall"]

                # 全体指標
                _ov_cols = st.columns(4)
                _ov_cols[0].metric("総レース数",   f"{_overall['n_races']}戦")
                _ov_cols[1].metric("的中数",        f"{_overall['n_hit']}回")
                _ov_cols[2].metric("的中率",        f"{_overall['hit_rate']*100:.1f}%")
                _ov_cols[3].metric("回収率",
                                   f"{_overall['roi']*100:.1f}%" if _overall['roi'] is not None else "-",
                                   delta=f"{(_overall['roi']-1)*100:+.1f}pt" if _overall.get('roi') else None)

                _ov2_cols = st.columns(4)
                _ov2_cols[0].metric("投資総額",   f"¥{_overall['total_invest']:,}")
                _ov2_cols[1].metric("回収総額",   f"¥{_overall['total_return']:,}")
                _ov2_cols[2].metric("見送り率",   f"{_overall['pass_rate']*100:.1f}%")
                _ov2_cols[3].metric("平均点数",   f"{_overall['avg_tickets']}点")

                st.markdown("---")

                # 集計サブタブ
                _tabs_summary = st.tabs([
                    "券種別", "レース構造別", "EV種別", "グレード別", "頭数別",
                    "危険馬", "妙味馬",
                    "役割精度", "外れ方", "見送り判断", "改善提案", "感度分析",
                ])

                def _make_stats_df(rows: list, key_col: str) -> pd.DataFrame:
                    display = []
                    for row in rows:
                        roi_v = row.get("roi")
                        display.append({
                            key_col:       row.get(key_col, "-"),
                            "N":           row.get("n_races", 0),
                            "的中":        row.get("n_hit", 0),
                            "的中率":      f"{row.get('hit_rate',0)*100:.1f}%",
                            "投資":        f"¥{row.get('total_invest',0):,}",
                            "回収":        f"¥{row.get('total_return',0):,}",
                            "回収率":      f"{roi_v*100:.1f}%" if roi_v is not None else "-",
                            "平均点数":    row.get("avg_tickets", 0),
                        })
                    return pd.DataFrame(display)

                with _tabs_summary[0]:
                    _df_bt = _make_stats_df(_analytics["by_bet_type"], "bet_type")
                    if not _df_bt.empty:
                        st.dataframe(_df_bt.rename(columns={"bet_type": "券種"}), use_container_width=True)
                    else:
                        st.info("データなし")

                with _tabs_summary[1]:
                    _df_st = _make_stats_df(_analytics["by_structure"], "structure_type")
                    if not _df_st.empty:
                        st.dataframe(_df_st.rename(columns={"structure_type": "レース構造"}), use_container_width=True)
                    else:
                        st.info("データなし")

                with _tabs_summary[2]:
                    _df_ev = _make_stats_df(_analytics["by_ev_type"], "ev_type")
                    if not _df_ev.empty:
                        st.dataframe(_df_ev.rename(columns={"ev_type": "EV種別"}), use_container_width=True)
                    else:
                        st.info("データなし")

                with _tabs_summary[3]:
                    _df_gr = _make_stats_df(_analytics["by_grade"], "grade")
                    if not _df_gr.empty:
                        st.dataframe(_df_gr.rename(columns={"grade": "グレード"}), use_container_width=True)
                    else:
                        st.info("データなし")

                # ── 頭数別成績 ────────────────────────────────────────
                with _tabs_summary[4]:
                    _df_fs = _make_stats_df(_analytics.get("by_field_size", []), "field_size")
                    if not _df_fs.empty:
                        st.dataframe(_df_fs.rename(columns={"field_size": "頭数区分"}), use_container_width=True)
                        st.caption("少頭数ほど人気馬が来やすく、フルゲートほど荒れやすい傾向があります。")
                    else:
                        st.info("データなし")

                # ── 危険馬 ────────────────────────────────────────────
                with _tabs_summary[5]:
                    _dc = _analytics["danger_cutoff"]
                    _dc_cols = st.columns(3)
                    _dc_cols[0].metric("危険馬含むレース数",   _dc["n_races_with_danger"])
                    _dc_cols[1].metric("消し推奨・正解率",
                                       f"{_dc['danger_truly_correct_rate']*100:.1f}%",
                                       help=f"{_dc['truly_total']}頭中")
                    _dc_cols[2].metric("「相手なら残る」3着内率",
                                       f"{_dc['danger_soft_placed_rate']*100:.1f}%",
                                       help=f"{_dc['soft_total']}頭中")

                    # グレード別・頭数別 危険馬精度
                    def _make_danger_seg_df(seg_rows: list, key_col: str) -> pd.DataFrame:
                        rows = []
                        for row in seg_rows:
                            rows.append({
                                key_col:        row.get(key_col, "-"),
                                "判定頭数":     row.get("truly_total", 0),
                                "消し正解率":   f"{row.get('danger_correct_rate', 0)*100:.1f}%",
                                "不足データ":   "⚠️" if row.get("insufficient_data") else "",
                            })
                        return pd.DataFrame(rows)

                    with st.expander("グレード別 危険馬精度", expanded=False):
                        _dg = _make_danger_seg_df(_analytics.get("danger_by_grade", []), "race_grade")
                        st.dataframe(_dg.rename(columns={"race_grade": "グレード"}), use_container_width=True) if not _dg.empty else st.info("データなし")
                    with st.expander("頭数別 危険馬精度", expanded=False):
                        _dfs = _make_danger_seg_df(_analytics.get("danger_by_field_size", []), "field_size")
                        st.dataframe(_dfs.rename(columns={"field_size": "頭数区分"}), use_container_width=True) if not _dfs.empty else st.info("データなし")

                # ── 妙味馬 ────────────────────────────────────────────
                with _tabs_summary[6]:
                    _vh = _analytics["value_horse"]
                    _vh_cols = st.columns(3)
                    _vh_cols[0].metric("妙味馬3着内率",   f"{_vh['value_in_money_rate']*100:.1f}%",
                                       help=f"対象{_vh['value_total']}頭")
                    _vh_cols[1].metric("妙味馬買い目的中率", f"{_vh['value_bought_hit_rate']*100:.1f}%")
                    _vh_cols[2].metric("拾いすぎ率（着外）",
                                       f"{_vh.get('value_wrong_rate', 0)*100:.1f}%",
                                       help=f"着外 {_vh.get('value_wrong_count', 0)} 回（高すぎると妙味基準が甘すぎ）",
                                       delta_color="inverse")

                    st.markdown("**妙味馬を含む買い目 vs 含まない買い目**")
                    _v_cmp = pd.DataFrame([
                        {"区分": "妙味馬含む", **{
                            k: v for k, v in _vh["with_value_bet"].items()
                            if k in ("n_races","n_hit","hit_rate","roi","total_invest","total_return")
                        }},
                        {"区分": "妙味馬なし", **{
                            k: v for k, v in _vh["without_value_bet"].items()
                            if k in ("n_races","n_hit","hit_rate","roi","total_invest","total_return")
                        }},
                    ])
                    st.dataframe(_v_cmp, use_container_width=True)

                    # グレード別・頭数別 妙味馬精度
                    def _make_value_seg_df(seg_rows: list, key_col: str) -> pd.DataFrame:
                        rows = []
                        for row in seg_rows:
                            rows.append({
                                key_col:        row.get(key_col, "-"),
                                "対象頭数":     row.get("value_total", 0),
                                "3着内率":      f"{row.get('value_in_money_rate', 0)*100:.1f}%",
                                "拾いすぎ率":   f"{row.get('value_wrong_rate', 0)*100:.1f}%",
                                "不足データ":   "⚠️" if row.get("insufficient_data") else "",
                            })
                        return pd.DataFrame(rows)

                    with st.expander("グレード別 妙味馬精度", expanded=False):
                        _vg = _make_value_seg_df(_analytics.get("value_by_grade", []), "race_grade")
                        st.dataframe(_vg.rename(columns={"race_grade": "グレード"}), use_container_width=True) if not _vg.empty else st.info("データなし")
                    with st.expander("頭数別 妙味馬精度", expanded=False):
                        _vfs = _make_value_seg_df(_analytics.get("value_by_field_size", []), "field_size")
                        st.dataframe(_vfs.rename(columns={"field_size": "頭数区分"}), use_container_width=True) if not _vfs.empty else st.info("データなし")

                # ── 役割精度 ──────────────────────────────────────────
                with _tabs_summary[7]:
                    _rp = _analytics.get("role_performance", {})
                    _rp_rows = []
                    _role_label = {"head": "頭(head)", "axis": "軸(axis)",
                                   "himo": "ヒモ(himo)", "fade": "消し(fade)"}
                    for _role_key, _role_name in _role_label.items():
                        _rs = _rp.get(_role_key, {})
                        _rp_rows.append({
                            "役割":       _role_name,
                            "指定数":     _rs.get("n_labeled", 0),
                            "勝(1着)":    _rs.get("n_won", 0),
                            "勝率":       f"{_rs.get('win_rate',0)*100:.1f}%",
                            "連対(2着内)":_rs.get("n_connected", 0),
                            "連対率":     f"{_rs.get('connect_rate',0)*100:.1f}%",
                            "3着内":      _rs.get("n_placed", 0),
                            "3着内率":    f"{_rs.get('place_rate',0)*100:.1f}%",
                        })
                    st.dataframe(pd.DataFrame(_rp_rows), use_container_width=True)
                    st.caption(
                        "消し(fade) の3着内率は低いほど良い（消し精度の指標）。"
                        "頭 勝率 ≥35% / 軸 連対率 ≥40% / ヒモ 3着内率 ≥30% を目安にしてください。"
                    )

                    # 取りこぼし注意馬
                    _rsc = _analytics.get("rescue_horse", {})
                    st.markdown("---")
                    st.markdown("**取りこぼし注意馬**")
                    _rsc_cols = st.columns(3)
                    _rsc_cols[0].metric("3着内率",
                                        f"{_rsc.get('rescue_in_money_rate',0)*100:.1f}%",
                                        help=f"対象 {_rsc.get('rescue_total',0)} 頭")
                    _rsc_cols[1].metric("3着内来たのに未購入",
                                        f"{_rsc.get('rescue_ignored',0)}回",
                                        help=f"3着内 {_rsc.get('rescue_in_money',0)} 頭中")
                    _rsc_cols[2].metric("見逃し率",
                                        f"{_rsc.get('rescue_ignored_rate',0)*100:.1f}%")

                    # グレード別 役割精度
                    with st.expander("グレード別 役割精度", expanded=False):
                        _rg_rows_raw = _analytics.get("role_by_grade", [])
                        if _rg_rows_raw:
                            _rg_display = []
                            for _seg in _rg_rows_raw:
                                _grade_k = _seg.get("race_grade", "-")
                                for _rk, _rn in _role_label.items():
                                    _rd = _seg.get(_rk, {})
                                    if isinstance(_rd, dict) and _rd.get("n_labeled", 0) > 0:
                                        _rg_display.append({
                                            "グレード": _grade_k,
                                            "役割":     _rn,
                                            "指定数":   _rd.get("n_labeled", 0),
                                            "勝率":     f"{_rd.get('win_rate',0)*100:.1f}%",
                                            "3着内率":  f"{_rd.get('place_rate',0)*100:.1f}%",
                                        })
                            if _rg_display:
                                st.dataframe(pd.DataFrame(_rg_display), use_container_width=True)
                            else:
                                st.info("データなし")
                        else:
                            st.info("データなし")

                # ── 外れ方ランキング ──────────────────────────────────
                with _tabs_summary[8]:
                    _tr = _analytics.get("tag_ranking", [])
                    if not _tr:
                        st.info("review_tags が記録されたレースがありません。")
                    else:
                        _level_color = {"good": "🟢", "neutral": "🟡", "bad": "🔴"}
                        _tr_rows = []
                        for _t in _tr:
                            _tr_rows.append({
                                "":         _level_color.get(_t["level"], "⚪"),
                                "タグ":     _t["tag"],
                                "件数":     _t["count"],
                                "発生率":   f"{_t['rate']*100:.1f}%",
                                "カテゴリ": _t["category"],
                                "レベル":   _t["level"],
                            })
                        _df_tr = pd.DataFrame(_tr_rows)
                        # bad タグを上に、good タグを下に並べる
                        _level_order = {"bad": 0, "neutral": 1, "good": 2}
                        _df_tr["_sort"] = _df_tr["レベル"].map(_level_order)
                        _df_tr = _df_tr.sort_values(["_sort", "件数"], ascending=[True, False])
                        st.dataframe(
                            _df_tr.drop(columns=["レベル", "_sort"]),
                            use_container_width=True,
                        )
                        st.caption("🔴 bad タグが上位に多いほど改善余地が大きいです。")

                # ── 見送り判断 ────────────────────────────────────────
                with _tabs_summary[9]:
                    _pj = _analytics.get("pass_judgment", {})
                    _pj_cols = st.columns(4)
                    _pj_cols[0].metric("見送り件数",   _pj.get("n_pass", 0))
                    _pj_cols[1].metric("買えば的中だった",
                                       _pj.get("n_would_hit", 0),
                                       help="「見送りだが買えば的中」タグが付いたケース")
                    _pj_cols[2].metric("見送り的中率",
                                       f"{_pj.get('would_hit_rate',0)*100:.1f}%",
                                       help="高いほど見送り閾値が厳しすぎる可能性")
                    _pj_cols[3].metric("正解の見送り率",
                                       f"{_pj.get('truly_pass_rate',0)*100:.1f}%",
                                       help="高いほど見送り判断が適切")
                    st.caption(
                        "見送り的中率 ≥30% が続く場合は EV_SKIP_THRESHOLD の引き下げを検討してください。"
                    )

                # ── 改善提案 ──────────────────────────────────────────
                with _tabs_summary[10]:
                    _cmts = _analytics.get("improvement_comments", [])
                    if not _cmts:
                        st.info("改善コメントを生成できませんでした。")
                    else:
                        for _c in _cmts:
                            if _c.startswith("通算回収率") and "プラス" in _c:
                                st.success(_c)
                            elif any(kw in _c for kw in ["要見直し", "未満", "過多", "低い", "多発", "可能性", "推奨"]):
                                st.warning(_c)
                            else:
                                st.info(_c)

                # ── 感度分析 ──────────────────────────────────────────
                with _tabs_summary[11]:
                    st.caption(
                        "閾値を変えた場合の成績比較。VALUE_GAP_MIN・stable_score 等を調整する前の参考にしてください。"
                    )
                    _sens_key = st.selectbox(
                        "分析対象フィールド",
                        ["value_gap", "stable_score", "ai_win_prob"],
                        key="sens_key",
                    )
                    _sens_defaults = {
                        "value_gap":    [0.01, 0.025, 0.04, 0.06, 0.08],
                        "stable_score": [0.30, 0.35, 0.40, 0.45, 0.50],
                        "ai_win_prob":  [0.03, 0.05, 0.08, 0.10, 0.15],
                    }
                    _sens_rows = compare_threshold_sensitivity(
                        _all_records,
                        _sens_key,
                        _sens_defaults.get(_sens_key, []),
                    )
                    if _sens_rows:
                        _sens_display = []
                        for _sr in _sens_rows:
                            _roi_v = _sr.get("roi")
                            _sens_display.append({
                                "閾値":     _sr["threshold"],
                                "N":        _sr.get("n_races", 0),
                                "的中率":   f"{_sr.get('hit_rate',0)*100:.1f}%",
                                "回収率":   f"{_roi_v*100:.1f}%" if _roi_v is not None else "-",
                                "投資":     f"¥{_sr.get('total_invest',0):,}",
                                "回収":     f"¥{_sr.get('total_return',0):,}",
                            })
                        st.dataframe(pd.DataFrame(_sens_display), use_container_width=True)
                        st.caption(
                            "N が少ないほどサンプル不足。回収率だけでなく N と的中率のバランスを見て閾値を選んでください。"
                        )
                    else:
                        st.info("結果入力済みのデータが不足しています。")

            except Exception as _exc:
                st.error(f"集計エラー: {_exc}")

    # =====================================================================
    # TAB 5: レビュー分析
    # =====================================================================
    with top_tabs[5]:
        try:
            _rv_all     = load_race_results()
            _rv_done    = [r for r in _rv_all if r.get("result") is not None]

            if not _rv_done:
                st.info(
                    "結果入力済みのレースがありません。"
                    "「回顧・検証」タブで着順を入力してください。"
                )
            else:
                _rv_an      = build_full_analytics(_rv_all)
                _rv_ov      = _rv_an["overall"]

                # ── ① サマリー ────────────────────────────────────────────
                st.subheader("📊 サマリー")
                _sm1, _sm2, _sm3, _sm4, _sm5 = st.columns(5)
                _sm1.metric("総レース数", f"{_rv_ov['n_races']}戦")
                _sm2.metric("的中率",     f"{_rv_ov['hit_rate']*100:.1f}%")
                _rv_roi = _rv_ov.get("roi")
                _sm3.metric(
                    "回収率",
                    f"{_rv_roi*100:.1f}%" if _rv_roi is not None else "-",
                    delta=f"{(_rv_roi-1)*100:+.1f}pt" if _rv_roi is not None else None,
                )
                _sm4.metric("投資総額", f"¥{_rv_ov['total_invest']:,}")
                _sm5.metric("回収総額", f"¥{_rv_ov['total_return']:,}")

                st.markdown("---")

                # ── ② 直近レース一覧（最大20件）────────────────────────────
                st.subheader("📋 直近レース一覧")
                _recent = sorted(_rv_done, key=lambda x: x.get("race_date",""), reverse=True)[:20]
                _list_rows = []
                for _rr in _recent:
                    _rd = _rr.get("result") or {}
                    _hit_icon  = "✅" if _rd.get("hit") else ("⏩" if _rr.get("is_pass") else "❌")
                    _tags      = _rd.get("review_tags") or _rd.get("review_labels") or []
                    _roi_v     = _rd.get("roi")
                    _rr_rate   = _rd.get("return_rate") or _roi_v
                    _list_rows.append({
                        "日付":       _rr.get("race_date", "-"),
                        "レース名":   _rr.get("race_name", "-"),
                        "券種":       _rr.get("recommended_bet_type", "-"),
                        "投資(円)":   int(_rd.get("investment_amount") or 0),
                        "払戻(円)":   int(_rd.get("return_amount") or 0),
                        "回収率":     f"{_rr_rate*100:.1f}%" if _rr_rate is not None else "-",
                        "結果":       _hit_icon,
                        "タグ":       " / ".join(_tags) if _tags else "-",
                    })
                st.dataframe(
                    pd.DataFrame(_list_rows),
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("---")

                # ── ③ 券種別成績 ──────────────────────────────────────────
                with st.expander("🎫 券種別成績", expanded=True):
                    _bt_rows = _rv_an.get("by_bet_type", [])
                    if _bt_rows:
                        _bt_display = []
                        for _row in _bt_rows:
                            _r = _row.get("roi")
                            _bt_display.append({
                                "券種":     _row.get("bet_type", "-"),
                                "N":        _row.get("n_races", 0),
                                "的中":     _row.get("n_hit", 0),
                                "的中率":   f"{_row.get('hit_rate',0)*100:.1f}%",
                                "投資(円)": f"¥{_row.get('total_invest',0):,}",
                                "払戻(円)": f"¥{_row.get('total_return',0):,}",
                                "回収率":   f"{_r*100:.1f}%" if _r is not None else "-",
                                "平均点数": _row.get("avg_tickets", 0),
                            })
                        st.dataframe(
                            pd.DataFrame(_bt_display),
                            use_container_width=True,
                            hide_index=True,
                        )
                        # 回収率棒グラフ
                        _chart_rows = [r for r in _bt_rows if r.get("roi") is not None and r.get("n_races",0) >= 1]
                        if _chart_rows:
                            _bt_labels = [r["bet_type"] for r in _chart_rows]
                            _bt_rois   = [round((r["roi"] or 0) * 100, 1) for r in _chart_rows]
                            _bar_colors = [
                                "#2ecc71" if v >= 105 else ("#e67e22" if v >= 90 else "#e74c3c")
                                for v in _bt_rois
                            ]
                            _fig_bt = go.Figure(go.Bar(
                                x=_bt_rois,
                                y=_bt_labels,
                                orientation="h",
                                marker_color=_bar_colors,
                                text=[f"{v:.1f}%" for v in _bt_rois],
                                textposition="outside",
                            ))
                            _fig_bt.update_layout(
                                title="券種別 回収率",
                                xaxis_title="回収率 (%)",
                                xaxis=dict(range=[0, max(_bt_rois + [120])]),
                                height=max(200, len(_bt_labels) * 45),
                                margin=dict(l=0, r=60, t=40, b=20),
                                showlegend=False,
                            )
                            _fig_bt.add_vline(x=100, line_dash="dash", line_color="gray",
                                              annotation_text="損益分岐(100%)",
                                              annotation_position="top right")
                            st.plotly_chart(_fig_bt, use_container_width=True)
                    else:
                        st.info("データなし")

                # ── ④ レース構造別成績 ────────────────────────────────────
                with st.expander("🏟 レース構造別成績", expanded=False):
                    _st_rows = _rv_an.get("by_structure", [])
                    if _st_rows:
                        _st_display = []
                        for _row in _st_rows:
                            _r = _row.get("roi")
                            _st_display.append({
                                "レース構造": _row.get("structure_type", "-"),
                                "N":          _row.get("n_races", 0),
                                "的中":       _row.get("n_hit", 0),
                                "的中率":     f"{_row.get('hit_rate',0)*100:.1f}%",
                                "投資(円)":   f"¥{_row.get('total_invest',0):,}",
                                "払戻(円)":   f"¥{_row.get('total_return',0):,}",
                                "回収率":     f"{_r*100:.1f}%" if _r is not None else "-",
                            })
                        st.dataframe(
                            pd.DataFrame(_st_display),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("データなし")

                # ── ⑤ 外れ方ランキング ────────────────────────────────────
                with st.expander("❌ 外れ方ランキング", expanded=True):
                    _tr = _rv_an.get("tag_ranking", [])
                    if _tr:
                        _level_icon = {"good": "🟢", "neutral": "🟡", "bad": "🔴"}
                        _tr_display = []
                        for _t in _tr:
                            _tr_display.append({
                                "":         _level_icon.get(_t["level"], "⚪"),
                                "タグ":     _t["tag"],
                                "件数":     _t["count"],
                                "発生率":   f"{_t['rate']*100:.1f}%",
                                "カテゴリ": _t["category"],
                            })
                        _df_tr2 = pd.DataFrame(_tr_display)
                        # bad → neutral → good の順、同レベルは件数降順
                        _lv_ord = {"bad": 0, "neutral": 1, "good": 2}
                        _df_tr2["_s"] = [
                            _lv_ord.get(_t["level"], 1) for _t in _tr
                        ]
                        _df_tr2 = _df_tr2.sort_values(["_s", "件数"], ascending=[True, False])
                        st.dataframe(
                            _df_tr2.drop(columns=["_s"]),
                            use_container_width=True,
                            hide_index=True,
                        )
                        # 頻出 bad タグのみ横棒グラフ
                        _bad_only = [t for t in _tr if t["level"] == "bad" and t["count"] > 0]
                        if _bad_only:
                            _fig_tag = go.Figure(go.Bar(
                                x=[t["count"] for t in _bad_only],
                                y=[t["tag"]    for t in _bad_only],
                                orientation="h",
                                marker_color="#e74c3c",
                                text=[str(t["count"]) for t in _bad_only],
                                textposition="outside",
                            ))
                            _fig_tag.update_layout(
                                title="頻出ミスタグ（bad タグのみ）",
                                xaxis_title="件数",
                                height=max(200, len(_bad_only) * 40),
                                margin=dict(l=0, r=40, t=40, b=20),
                                showlegend=False,
                            )
                            st.plotly_chart(_fig_tag, use_container_width=True)
                    else:
                        st.info("review_tags が記録されたレースがありません。")

                # ── ⑥ 改善候補メモ ──────────────────────────────────────
                with st.expander("💡 改善候補メモ", expanded=True):
                    _cmts2 = _rv_an.get("improvement_comments", [])
                    if not _cmts2:
                        st.info("改善コメントを生成できませんでした。")
                    else:
                        for _c2 in _cmts2:
                            if _c2.startswith("通算回収率") and "プラス" in _c2:
                                st.success(_c2)
                            elif any(kw in _c2 for kw in ["要見直し", "未満", "過多", "低い", "多発", "可能性", "引き"]):
                                st.warning(_c2)
                            else:
                                st.info(_c2)

                # ── 回収率推移（直近10レース）────────────────────────────
                with st.expander("📈 回収率推移（直近10レース）", expanded=False):
                    _trend_src = sorted(
                        _rv_done,
                        key=lambda x: x.get("race_date", ""),
                    )[-10:]
                    if len(_trend_src) >= 2:
                        _cum_invest = 0
                        _cum_return = 0
                        _trend_labels = []
                        _trend_roi    = []
                        _trend_single = []  # 1レースごとの回収率
                        for _tr_r in _trend_src:
                            _tr_rd = _tr_r.get("result") or {}
                            _inv = int(_tr_rd.get("investment_amount") or 0)
                            _ret = int(_tr_rd.get("return_amount") or 0)
                            _cum_invest += _inv
                            _cum_return += _ret
                            _cum_roi = round(_cum_return / _cum_invest * 100, 1) if _cum_invest > 0 else 0
                            _sgl_roi = round(_ret / _inv * 100, 1) if _inv > 0 else 0
                            _trend_labels.append(
                                f"{_tr_r.get('race_date','?')} {_tr_r.get('race_name','')[:6]}"
                            )
                            _trend_roi.append(_cum_roi)
                            _trend_single.append(_sgl_roi)

                        _fig_roi = go.Figure()
                        _fig_roi.add_trace(go.Scatter(
                            x=_trend_labels,
                            y=_trend_roi,
                            mode="lines+markers",
                            name="累積回収率",
                            line=dict(color="#3498db", width=2),
                            marker=dict(size=7),
                        ))
                        _fig_roi.add_trace(go.Bar(
                            x=_trend_labels,
                            y=_trend_single,
                            name="単レース回収率",
                            marker_color=[
                                "#2ecc71" if v >= 100 else "#e74c3c"
                                for v in _trend_single
                            ],
                            opacity=0.4,
                            yaxis="y",
                        ))
                        _fig_roi.add_hline(
                            y=100,
                            line_dash="dash",
                            line_color="gray",
                            annotation_text="損益分岐(100%)",
                            annotation_position="top left",
                        )
                        _fig_roi.update_layout(
                            title="回収率推移（青線: 累積 / 棒: 単レース）",
                            yaxis_title="回収率 (%)",
                            xaxis_tickangle=-30,
                            height=360,
                            margin=dict(l=0, r=20, t=50, b=80),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        )
                        st.plotly_chart(_fig_roi, use_container_width=True)
                    else:
                        st.info("推移グラフには2件以上の結果入力済みレースが必要です。")

                # ── ⑩ ドリフト検出 ──────────────────────────────────────────
                st.markdown("---")
                st.subheader("🔔 パフォーマンス ドリフト検出")
                st.caption("直近10件の成績が全体平均から急落していないかを監視します。")
                try:
                    from analytics_ai import detect_performance_drift
                    _drift = detect_performance_drift(_rv_all)
                    _ds = _drift["status"]
                    if _ds == "alert":
                        st.error(f"🚨 ドリフト検出: {_drift['message']}")
                    elif _ds == "warning":
                        st.warning(f"⚠️ 要注意: {_drift['message']}")
                    elif _ds == "ok":
                        st.success(f"✅ {_drift['message']}")
                    else:
                        st.info(_drift["message"])

                    if _drift["n_overall"] >= 5:
                        _dc1, _dc2, _dc3, _dc4 = st.columns(4)
                        _dc1.metric("直近的中率", f"{_drift['recent_hitrate']*100:.1f}%",
                                    delta=f"{(_drift['recent_hitrate']-_drift['overall_hitrate'])*100:+.1f}%p")
                        _dc2.metric("全体的中率", f"{_drift['overall_hitrate']*100:.1f}%")
                        _r_roi = _drift.get("recent_roi")
                        _o_roi = _drift.get("overall_roi")
                        _dc3.metric("直近ROI",
                                    f"{_r_roi*100:.1f}%" if _r_roi is not None else "-",
                                    delta=f"{(_r_roi-_o_roi)*100:+.1f}%p" if (_r_roi and _o_roi) else None)
                        _dc4.metric("全体ROI",
                                    f"{_o_roi*100:.1f}%" if _o_roi is not None else "-")
                except Exception as _dr_exc:
                    st.warning(f"ドリフト検出エラー: {_dr_exc}")

                # ── ⑩ シグナル別的中率トラッキング ──────────────────────────
                st.markdown("---")
                st.subheader("📡 シグナル別的中率トラッキング")
                st.caption(
                    "各条件シグナルが付いた馬の実際の3着内率を集計します。"
                    "Lift がプラス = そのシグナルは実際に有効、マイナス = 逆効果の可能性。"
                )
                try:
                    from analytics_ai import summarize_signal_hit_rates
                    _shr = summarize_signal_hit_rates(_rv_all)
                    if _shr["data_horses"] < 10:
                        st.info(f"トラッキングデータが少なすぎます（{_shr['data_horses']}頭）。レース結果を入力すると精度が向上します。")
                    else:
                        _sh_c1, _sh_c2 = st.columns(2)
                        with _sh_c1:
                            st.markdown("**追い風シグナル条件別 top3率**")
                            if _shr["positive_by_factor"]:
                                st.dataframe(
                                    [{"条件": r["factor"], "頭数": r["n_horses"],
                                      "top3率": f"{r['top3_rate']*100:.1f}%",
                                      "lift": f"{r['lift']*100:+.1f}%p"}
                                     for r in _shr["positive_by_factor"]],
                                    use_container_width=True, hide_index=True,
                                )
                            else:
                                st.caption("データ不足")
                        with _sh_c2:
                            st.markdown("**懸念シグナル条件別 消し精度**")
                            if _shr["negative_by_factor"]:
                                st.dataframe(
                                    [{"条件": r["factor"], "頭数": r["n_horses"],
                                      "top3率": f"{r['top3_rate']*100:.1f}%",
                                      "消し精度lift": f"{r['lift']*100:+.1f}%p"}
                                     for r in _shr["negative_by_factor"]],
                                    use_container_width=True, hide_index=True,
                                )
                            else:
                                st.caption("データ不足")
                        st.markdown("**補正合計帯別 top3率**")
                        if _shr["by_adjust_band"]:
                            st.dataframe(
                                [{"補正帯": r["band"], "頭数": r["n_horses"],
                                  "top3率": f"{r['top3_rate']*100:.1f}%",
                                  "lift": f"{r['lift']*100:+.1f}%p"}
                                 for r in _shr["by_adjust_band"]],
                                use_container_width=True, hide_index=True,
                            )
                        _sh_m1, _sh_m2 = st.columns(2)
                        _sh_m1.metric("ベースライン3着内率", f"{_shr['base_top3_rate']*100:.1f}%")
                        _sh_m2.metric("集計頭数", f"{_shr['data_horses']:,}頭")
                except Exception as _shr_exc:
                    st.warning(f"シグナルトラッキング集計エラー: {_shr_exc}")

        except Exception as _rv_exc:
            st.error(f"レビュー分析エラー: {_rv_exc}")

else:
    # AI分析未実行時はタブだけ表示してバックテストを使えるようにする
    top_tabs = st.tabs(["⚡ 能力スコア", "🏇 展開・ポジション", "🎯 推奨買い目", "🤖 AI詳細", "📋 回顧・検証", "📊 レビュー分析", "📈 バックテスト"])
    for _i in range(6):
        with top_tabs[_i]:
            st.info("URLを入力して「AI分析を実行」を押してください")

# =====================================================================
# TAB 6: バックテスト
# =====================================================================
with top_tabs[6]:
    st.subheader("過去レース バックテスト")
    st.caption(
        "過去レースIDを入力してAIシグナルの精度と回収率を検証します。"
        "「条件統計モード」は各馬の個別ページ取得不要で高速動作します。"
    )

    # ── レース名でID自動取得 ────────────────────────────────────────
    with st.expander("レースID 自動取得（レース名 × 年数指定）", expanded=False):
        st.caption("レース名を入力すると、指定した年数分のレースIDをまとめて収集します。重賞レースの過去10年分などに最適です。")
        import datetime as _dt2
        _nm_col1, _nm_col2, _nm_col3 = st.columns(3)
        _nm_race_name = _nm_col1.text_input(
            "レース名（例: 阪神大賞典）",
            key="nm_race_name",
        )
        _nm_start_year = _nm_col2.number_input(
            "開始年", min_value=2010, max_value=_dt2.date.today().year,
            value=_dt2.date.today().year - 9, step=1, key="nm_start_year",
        )
        _nm_end_year = _nm_col3.number_input(
            "終了年", min_value=2010, max_value=_dt2.date.today().year,
            value=_dt2.date.today().year - 1, step=1, key="nm_end_year",
        )
        _nm_months = st.multiselect(
            "開催月を絞る（わかる場合は指定すると大幅に高速化）",
            options=list(range(1, 13)),
            format_func=lambda m: f"{m}月",
            key="nm_months",
            help="例: 阪神大賞典は3月 / 有馬記念は12月。未指定の場合は全月（1〜12月）を検索します。",
        )
        _nm_days_est = (
            len(_nm_months) if _nm_months else 12
        ) * 30 * max(1, int(_nm_end_year) - int(_nm_start_year) + 1)
        st.caption(
            f"推定リクエスト数: 約{_nm_days_est}件 / 所要時間: 約{max(1, _nm_days_est // 120)}〜{max(1, _nm_days_est // 60)}分"
        )
        _nm_fetch_btn = st.button("レース名で検索・取得", key="nm_fetch_ids")
        if _nm_fetch_btn:
            if not _nm_race_name.strip():
                st.warning("レース名を入力してください。")
            elif int(_nm_start_year) > int(_nm_end_year):
                st.warning("開始年 ≤ 終了年にしてください。")
            else:
                _nm_years = int(_nm_end_year) - int(_nm_start_year) + 1
                _nm_prog = st.progress(0.0, text="検索中...")
                try:
                    from dividend_scraper import fetch_race_ids_by_name

                    def _nm_cb(cur, total, ds):
                        _nm_prog.progress(cur / total, text=f"検索中: {ds} ({cur}/{total}日)")

                    _nm_result = fetch_race_ids_by_name(
                        _nm_race_name.strip(),
                        int(_nm_start_year),
                        int(_nm_end_year),
                        search_months=_nm_months or None,
                        progress_callback=_nm_cb,
                    )
                    _nm_prog.empty()
                    if _nm_result["n_races"] == 0:
                        st.warning(
                            f"「{_nm_race_name}」に一致するレースが見つかりませんでした。"
                            "レース名（部分一致）・開催月を確認してください。"
                        )
                    else:
                        _existing = st.session_state.get("_auto_fetched_ids", "")
                        _new_ids  = "\n".join(_nm_result["race_ids"])
                        st.session_state["_auto_fetched_ids"] = (
                            (_existing + "\n" + _new_ids).strip()
                        )
                        st.success(f"{_nm_result['n_races']}件のレースIDを取得しました。")
                        _yr_rows = [
                            {"年": yr, "レース数": len(ids), "レースID": ", ".join(ids)}
                            for yr, ids in sorted(_nm_result["by_year"].items())
                        ]
                        st.dataframe(_yr_rows, use_container_width=True, hide_index=True)
                except Exception as _nm_exc:
                    _nm_prog.empty()
                    st.error(f"取得エラー: {_nm_exc}")

    # ── レースID自動取得 ────────────────────────────────────────────
    with st.expander("レースID 自動取得（日付範囲指定）", expanded=False):
        st.caption("netkeiba から指定期間のレースIDを自動収集してテキストエリアに入力します。")
        from dividend_scraper import VENUE_CODE_MAP
        import datetime as _dt

        _auto_col1, _auto_col2, _auto_col3 = st.columns(3)
        _auto_start = _auto_col1.date_input(
            "開始日", value=_dt.date.today() - _dt.timedelta(days=90),
            key="auto_start_date",
        )
        _auto_end = _auto_col2.date_input(
            "終了日", value=_dt.date.today() - _dt.timedelta(days=1),
            key="auto_end_date",
        )
        _auto_venues_sel = _auto_col3.multiselect(
            "開催場所（未選択=全場）",
            options=list(VENUE_CODE_MAP.keys()),
            format_func=lambda c: f"{VENUE_CODE_MAP[c]}({c})",
            key="auto_venues",
        )

        _auto_fetch_btn = st.button("レースIDを自動取得", key="auto_fetch_ids")
        if _auto_fetch_btn:
            _s_str = _auto_start.strftime("%Y%m%d")
            _e_str = _auto_end.strftime("%Y%m%d")
            _days  = (_auto_end - _auto_start).days + 1
            if _days > 365:
                st.warning(
                    f"⚠️ {_days}日分の取得は約{_days//2//60}〜{_days//60}分かかります。"
                    "長期間の場合は1年ずつに分けて実行することをおすすめします。"
                )
            _prog_bar = st.progress(0.0, text="取得中...")
            try:
                from dividend_scraper import fetch_race_ids_by_date_range

                def _cb(cur, total, ds):
                    _prog_bar.progress(cur / total, text=f"取得中: {ds} ({cur}/{total}日)")

                _auto_result = fetch_race_ids_by_date_range(
                    _s_str, _e_str,
                    venue_codes=_auto_venues_sel or None,
                    progress_callback=_cb,
                )
                _prog_bar.empty()
                st.session_state["_auto_fetched_ids"] = "\n".join(_auto_result["race_ids"])
                st.success(
                    f"{_auto_result['n_dates']}日 / {_auto_result['n_races']}レース を取得しました。"
                    " → 下のテキストエリアに自動入力されます。"
                )
            except Exception as _af_exc:
                _prog_bar.empty()
                st.error(f"自動取得エラー: {_af_exc}")

    # ── モード選択 ──────────────────────────────────────────────────
    _bt_mode = st.radio(
        "バックテストモード",
        ["EVパイプライン（推奨買い目）", "単勝（1着予想）", "三連複（上位3頭ボックス）"],
        horizontal=True,
        help=(
            "EVパイプライン: 実際のアプリ推奨ロジック（EV+ロール）をそのまま適用してROIを計測。最も実用的。\n"
            "単勝: AI最高スコア馬を1着予想。精度検証用。\n"
            "三連複: AIシグナル上位3頭を3着内すべて当てる。精度検証用。"
        ),
    )

    # 自動取得IDをテキストエリアのセッションキーに転送
    if "_auto_fetched_ids" in st.session_state:
        st.session_state["bt_race_ids_text"] = st.session_state.pop("_auto_fetched_ids")

    _bt_col1, _bt_col2 = st.columns([2, 1])
    with _bt_col1:
        _bt_race_ids_raw = st.text_area(
            "バックテスト対象 レースID（1行1件）",
            key="bt_race_ids_text",
            height=160,
            placeholder="202501051211\n202502010411\n202503020811",
            help="netkeiba の race_id（12桁）を1行に1件ずつ入力してください。自動取得ボタンで自動入力できます。",
        )
    with _bt_col2:
        _bt_bet_amount = st.number_input("1レースあたり賭け金（円）", min_value=100, step=100, value=100)
        st.markdown("---")
        st.markdown("**レースID の調べ方**")
        st.caption(
            "netkeiba のレース結果URL\n"
            "`result.html?race_id=XXXXXXXXXXXX`\n"
            "の `XXXXXXXXXXXX` 部分（12桁）"
        )

    if _bt_mode == "EVパイプライン（推奨買い目）":
        _bt_use_cstats = st.checkbox(
            "条件統計を使う（高精度・低速）",
            value=True,
            help="ONにすると過去10年のコース別統計をrequestsで取得してシグナル補正します。OFFにすると市場オッズのみで推定（速い）。",
        )

    _bt_run = st.button("バックテスト実行", type="primary")

    # ── 既存保存データのROI集計 ─────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 保存済みレースのROI")
    _bt_saved = load_race_results()
    if _bt_saved:
        try:
            from backtest_runner import calc_roi_from_saved
            _bt_roi = calc_roi_from_saved(_bt_saved)
            _bm1, _bm2, _bm3, _bm4 = st.columns(4)
            _bm1.metric("分析済みレース", f"{_bt_roi['n_races']}件")
            _bm2.metric("総投資額", f"¥{_bt_roi['total_invest']:,}")
            _bm3.metric("総払戻額", f"¥{_bt_roi['total_return']:,}")
            _roi_val = _bt_roi.get("roi_pct")
            _roi_delta = f"{_roi_val - 100:+.1f}%" if _roi_val is not None else None
            _bm4.metric("回収率", f"{_roi_val:.1f}%" if _roi_val is not None else "—",
                        delta=_roi_delta)

            if _bt_roi.get("n_no_dividend", 0) > 0:
                st.warning(
                    f"⚠️ {_bt_roi['n_no_dividend']}件 の払戻金額が未入力です。"
                    "「回顧・検証」タブで払戻金額を入力すると正確なROIが計算されます。"
                )

            if _bt_roi["by_bet_type"]:
                _bt_type_rows = [
                    {"券種": bt["bet_type"], "件数": bt["n"],
                     "投資(円)": bt["invest"], "払戻(円)": bt["return"],
                     "回収率": f"{bt['roi']*100:.1f}%"}
                    for bt in _bt_roi["by_bet_type"]
                ]
                st.dataframe(_bt_type_rows, use_container_width=True, hide_index=True)
        except Exception as _bt_exc:
            st.error(f"ROI集計エラー: {_bt_exc}")
    else:
        st.info(
            "保存済みレースがありません。AI分析後「回顧・検証」タブで結果を保存してください。"
        )

    # ── バックテスト実行 ────────────────────────────────────────────
    if _bt_run:
        _bt_ids_raw = [x.strip() for x in (_bt_race_ids_raw or "").splitlines() if x.strip()]
        _bt_ids = [x for x in _bt_ids_raw if len(x) >= 10]

        if not _bt_ids:
            st.warning("有効なレースID（10桁以上）が入力されていません。")
        else:
            st.info(f"バックテスト開始: {len(_bt_ids)}件（Selenium不要）...")
            _bt_progress = st.progress(0)
            _bt_status   = st.empty()

            def _bt_cb(_cur, _tot, _rid):
                _bt_status.text(f"処理中 {_cur}/{_tot}: {_rid}")
                _bt_progress.progress(_cur / _tot)

            try:
                if _bt_mode == "EVパイプライン（推奨買い目）":
                    # ── EVパイプラインモード ──────────────────────────────
                    from backtest_runner import run_ev_pipeline_backtest
                    _use_cs = st.session_state.get("_bt_use_cstats_val", True)
                    _ev_result = run_ev_pipeline_backtest(
                        _bt_ids,
                        bankroll=bankroll,
                        use_condition_stats=_use_cs,
                        progress_callback=_bt_cb,
                    )
                    _bt_partial = _ev_result["race_results"]
                    _smry = _ev_result["summary"]

                    _bt_status.empty()
                    _bt_progress.empty()

                    _n_ok   = _smry["n_races"]
                    _n_err  = _smry["n_errors"]
                    _n_skip = _smry["n_skip"]
                    _n_hit  = _smry["hit_count"]
                    _t_inv  = _smry["total_invest"]
                    _t_ret  = _smry["total_return"]
                    _roi    = _smry["roi_pct"] or 0.0
                    _hrate  = round(_smry["hit_rate"] * 100, 1)

                    st.success(f"バックテスト完了: {_n_ok}件 / 見送り{_n_skip}件 / エラー{_n_err}件")

                    _bc1, _bc2, _bc3, _bc4, _bc5 = st.columns(5)
                    _bc1.metric("実行レース", f"{_n_ok}件")
                    _bc2.metric("的中数", f"{_n_hit}件")
                    _bc3.metric("的中率", f"{_hrate:.1f}%")
                    _bc4.metric("回収率", f"{_roi:.1f}%", delta=f"{_roi-100:+.1f}%")
                    _bc5.metric("見送り", f"{_n_skip}件")

                    st.markdown(f"**投資合計**: ¥{_t_inv:,}  |  **払戻合計**: ¥{_t_ret:,}")

                    if _smry.get("by_bet_type"):
                        st.markdown("##### 券種別ROI")
                        _by_type_rows = [
                            {
                                "券種": bt["bet_type"],
                                "件数": bt["n"],
                                "投資(円)": bt["invest"],
                                "払戻(円)": bt["return"],
                                "回収率": f"{bt['roi_pct']:.1f}%",
                            }
                            for bt in _smry["by_bet_type"]
                        ]
                        st.dataframe(_by_type_rows, use_container_width=True, hide_index=True)

                    # ── 年別ROI ──────────────────────────────────────────
                    _by_year: dict = {}
                    for _r in _bt_partial:
                        if _r.get("error") or _r.get("skip"):
                            continue
                        _yr = str(_r.get("race_id", ""))[:4]
                        if not _yr.isdigit():
                            continue
                        if _yr not in _by_year:
                            _by_year[_yr] = {"n": 0, "hit": 0, "invest": 0, "ret": 0}
                        _by_year[_yr]["n"]      += 1
                        _by_year[_yr]["hit"]    += 1 if _r.get("hit") else 0
                        _by_year[_yr]["invest"] += _r.get("invest", 0)
                        _by_year[_yr]["ret"]    += _r.get("return", 0)

                    if len(_by_year) > 1:
                        st.markdown("##### 年別ROI")
                        _yr_rows = []
                        for _yr in sorted(_by_year.keys()):
                            _yd = _by_year[_yr]
                            _yr_roi = (_yd["ret"] / _yd["invest"] * 100) if _yd["invest"] > 0 else 0.0
                            _yr_rows.append({
                                "年": _yr,
                                "レース数": _yd["n"],
                                "的中数": _yd["hit"],
                                "的中率": f"{_yd['hit']/_yd['n']*100:.1f}%" if _yd["n"] > 0 else "-",
                                "投資(円)": _yd["invest"],
                                "払戻(円)": _yd["ret"],
                                "回収率": f"{_yr_roi:.1f}%",
                            })
                        st.dataframe(_yr_rows, use_container_width=True, hide_index=True)

                    # レース別結果表
                    _bt_display = []
                    for _r in _bt_partial:
                        if _r.get("error"):
                            _bt_display.append({"race_id": _r["race_id"], "状態": f"エラー: {_r['error']}", "的中": "", "投資": 0, "払戻": 0})
                        elif _r.get("skip"):
                            _bt_display.append({
                                "race_id": _r.get("race_id", ""),
                                "日付":    _r.get("race_date", ""),
                                "レース名": _r.get("race_name", "")[:20],
                                "状態":    f"見送り: {_r.get('skip_reason','')[:30]}",
                                "的中": "-", "投資": 0, "払戻": 0,
                            })
                        else:
                            _bt_display.append({
                                "race_id":   _r.get("race_id", ""),
                                "日付":      _r.get("race_date", ""),
                                "レース名":  _r.get("race_name", "")[:20],
                                "券種":      _r.get("bet_type", ""),
                                "推奨馬":    _r.get("our_pick", ""),
                                "実際3着内": _r.get("actual_top3", ""),
                                "的中":      "◎" if _r.get("hit") else "×",
                                "投資(円)":  _r.get("invest", 0),
                                "払戻(円)":  _r.get("return", 0),
                                "EV型":      _r.get("ev_type", ""),
                            })
                    st.dataframe(_bt_display, use_container_width=True, hide_index=True)

                else:
                    # ── シグナルモード（単勝 / 三連複） ──────────────────
                    from dividend_scraper import scrape_race_result
                    from trend_stats import build_condition_stats, build_combo_condition_stats
                    from signal_judge import (
                        build_horse_signal_details,
                        build_horse_combo_signal_details,
                        aggregate_signal_result,
                    )
                    from race_history_ai import fetch_race_history_enriched
                    import time as _time
                    import random as _random

                    _is_trio = (_bt_mode == "三連複（上位3頭ボックス）")
                    _bt_partial: list = []

                    for _bi, _rid in enumerate(_bt_ids):
                        _bt_status.text(f"処理中 {_bi+1}/{len(_bt_ids)}: {_rid}")
                        _bt_progress.progress((_bi + 1) / len(_bt_ids))

                        try:
                            _rd = scrape_race_result(_rid)
                            if not _rd or not _rd.get("runners"):
                                _bt_partial.append({"race_id": _rid, "error": "結果取得失敗"})
                                continue

                            _runners      = _rd["runners"]
                            _dividends    = _rd.get("dividends", {})
                            _finish_order = _rd.get("finish_order", [])
                            _test_year    = int(_rid[:4])

                            _enriched = fetch_race_history_enriched(None, _rid, years=11)
                            _enriched = [
                                r for r in _enriched
                                if not str(r.get("race_id", "")).startswith(str(_test_year))
                            ]
                            if not _enriched:
                                _bt_partial.append({"race_id": _rid, "error": "過去データなし"})
                                continue

                            _cstats = build_condition_stats(_enriched)
                            _combo  = build_combo_condition_stats(_enriched)

                            _scored = []
                            for _runner in _runners:
                                _pf = {
                                    "horse_name":    _runner.get("horse_name", ""),
                                    "age":           _runner.get("age"),
                                    "gate":          _runner.get("gate"),
                                    "running_style": _runner.get("running_style"),
                                    "popularity":    _runner.get("popularity"),
                                    "win_odds":      _runner.get("win_odds"),
                                    "recent_last3f": _runner.get("last_3f"),
                                    "jockey_weight": _runner.get("jockey_weight"),
                                    "jockey":        _runner.get("jockey", ""),
                                    "model_score":   0.5,
                                }
                                _d   = build_horse_signal_details(_pf, _cstats)
                                _cd  = build_horse_combo_signal_details(_pf, _combo)
                                _sig = aggregate_signal_result(_d + _cd)
                                _scored.append({
                                    "horse_name":   _runner["horse_name"],
                                    "signal_score": _sig["total_trend_adjust"],
                                    "popularity":   _runner.get("popularity"),
                                    "win_odds":     _runner.get("win_odds"),
                                })
                            _scored.sort(key=lambda x: (
                                -x["signal_score"],
                                x["popularity"] if x["popularity"] else 99,
                            ))

                            if _is_trio:
                                _ai_top3     = {s["horse_name"] for s in _scored[:3]}
                                _actual_top3 = set(_finish_order[:3])
                                _hit         = (_ai_top3 == _actual_top3)
                                _partial     = len(_ai_top3 & _actual_top3)
                                _trio_div    = _dividends.get("3連複") or 0
                                _return_amt  = int(_trio_div * _bt_bet_amount / 100) if _hit else 0
                                _bt_partial.append({
                                    "race_id": _rid, "race_date": _rd.get("race_date", ""),
                                    "race_name": _rd.get("race_name", ""),
                                    "ai_top3": " / ".join(sorted(_ai_top3)),
                                    "actual_top3": " / ".join(sorted(_actual_top3)),
                                    "partial_match": _partial, "hit": _hit,
                                    "invest": _bt_bet_amount, "return": _return_amt,
                                    "trio_dividend": _trio_div, "n_runners": len(_runners),
                                    "error": None,
                                })
                            else:
                                _actual_win = _finish_order[0] if _finish_order else ""
                                _our_pick   = _scored[0]["horse_name"] if _scored else ""
                                _pick_score = _scored[0]["signal_score"] if _scored else 0.0
                                _pick_odds  = _scored[0]["win_odds"] if _scored else None
                                _hit        = (_our_pick == _actual_win)
                                _win_div    = _dividends.get("単勝") or 0
                                _return_amt = int(_win_div * _bt_bet_amount / 100) if _hit else 0
                                _bt_partial.append({
                                    "race_id": _rid, "race_date": _rd.get("race_date", ""),
                                    "race_name": _rd.get("race_name", ""),
                                    "our_pick": _our_pick, "pick_score": round(_pick_score, 4),
                                    "pick_odds": _pick_odds, "actual_winner": _actual_win,
                                    "hit": _hit, "invest": _bt_bet_amount, "return": _return_amt,
                                    "win_dividend": _win_div, "n_runners": len(_runners),
                                    "error": None,
                                })
                        except Exception as _be:
                            _bt_partial.append({"race_id": _rid, "error": str(_be)})

                        _time.sleep(_random.uniform(0.5, 1.0))

                    _bt_status.empty()
                    _bt_progress.empty()

                    _n_ok  = sum(1 for r in _bt_partial if not r.get("error"))
                    _n_hit = sum(1 for r in _bt_partial if r.get("hit"))
                    _t_inv = sum(r.get("invest", 0) for r in _bt_partial if not r.get("error"))
                    _t_ret = sum(r.get("return", 0) for r in _bt_partial if not r.get("error"))
                    _roi   = round(_t_ret / max(_t_inv, 1) * 100, 1)
                    _hrate = round(_n_hit / max(_n_ok, 1) * 100, 1)

                    st.success(f"バックテスト完了: {_n_ok}件")

                    if _is_trio:
                        _coverage = sum(r.get("partial_match", 0) for r in _bt_partial if not r.get("error"))
                        _cov_rate = round(_coverage / max(_n_ok * 3, 1) * 100, 1)
                        _bc1, _bc2, _bc3, _bc4, _bc5 = st.columns(5)
                        _bc1.metric("対象レース", f"{_n_ok}件")
                        _bc2.metric("完全的中", f"{_n_hit}件")
                        _bc3.metric("的中率", f"{_hrate:.1f}%")
                        _bc4.metric("平均カバー率", f"{_cov_rate:.1f}%")
                        _bc5.metric("回収率", f"{_roi:.1f}%", delta=f"{_roi-100:+.1f}%")
                    else:
                        _bc1, _bc2, _bc3, _bc4 = st.columns(4)
                        _bc1.metric("対象レース", f"{_n_ok}件")
                        _bc2.metric("的中数", f"{_n_hit}件")
                        _bc3.metric("的中率", f"{_hrate:.1f}%")
                        _bc4.metric("回収率", f"{_roi:.1f}%", delta=f"{_roi-100:+.1f}%")

                    st.markdown(f"**投資合計**: ¥{_t_inv:,}  |  **払戻合計**: ¥{_t_ret:,}")

                    _bt_display = []
                    for _r in _bt_partial:
                        if _r.get("error"):
                            _bt_display.append({"race_id": _r["race_id"], "状態": f"エラー: {_r['error']}"})
                        elif _is_trio:
                            _bt_display.append({
                                "race_id": _r.get("race_id", ""), "日付": _r.get("race_date", ""),
                                "レース名": _r.get("race_name", "")[:20],
                                "AI上位3頭": _r.get("ai_top3", ""),
                                "実際3着内": _r.get("actual_top3", ""),
                                "一致数": f"{_r.get('partial_match', 0)}/3",
                                "的中": "◎" if _r.get("hit") else "×",
                                "投資(円)": _r.get("invest", 0), "払戻(円)": _r.get("return", 0),
                                "3連複配当": _r.get("trio_dividend", 0),
                            })
                        else:
                            _bt_display.append({
                                "race_id": _r.get("race_id", ""), "日付": _r.get("race_date", ""),
                                "レース名": _r.get("race_name", "")[:20],
                                "推奨馬": _r.get("our_pick", ""), "スコア": _r.get("pick_score", 0),
                                "実際1着": _r.get("actual_winner", ""),
                                "的中": "◎" if _r.get("hit") else "×",
                                "投資(円)": _r.get("invest", 0), "払戻(円)": _r.get("return", 0),
                                "単勝配当": _r.get("win_dividend", 0),
                            })
                    st.dataframe(_bt_display, use_container_width=True, hide_index=True)

                # ── 累積ROIグラフ（全モード共通） ──
                _chart_data = [r for r in _bt_partial if not r.get("error") and not r.get("skip")]
                if len(_chart_data) >= 2:
                    try:
                        _cum_inv = 0
                        _cum_ret = 0
                        _roi_series = []
                        for _r in _chart_data:
                            _cum_inv += _r.get("invest", 0)
                            _cum_ret += _r.get("return", 0)
                            _roi_series.append(round(_cum_ret / max(_cum_inv, 1) * 100, 1))
                        _fig_bt = go.Figure()
                        _fig_bt.add_trace(go.Scatter(
                            x=list(range(1, len(_roi_series)+1)),
                            y=_roi_series,
                            mode="lines+markers",
                            name="累積回収率",
                            line=dict(color="#1f77b4"),
                        ))
                        _fig_bt.add_hline(y=100, line_dash="dash", line_color="gray",
                                          annotation_text="収支ライン(100%)")
                        _fig_bt.update_layout(
                            title="累積回収率推移",
                            xaxis_title="レース数",
                            yaxis_title="回収率 (%)",
                            height=350,
                        )
                        st.plotly_chart(_fig_bt, use_container_width=True)
                    except Exception:
                        pass

            except Exception as _bt_main_exc:
                st.error(f"バックテストエラー: {_bt_main_exc}")

    # ── 閾値最適化 ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### シグナル閾値 最適化")
    st.caption(
        "バックテスト対象と同じレースIDを使って、"
        "signal_judge の SIGNAL_THRESHOLDS を自動チューニングします。"
        "目安: 20レース以上のデータが推奨。"
    )
    _opt_run = st.button("閾値最適化を実行", help="バックテストIDが入力済みのときのみ実行できます")
    if _opt_run:
        _opt_ids_raw = [x.strip() for x in (_bt_race_ids_raw or "").splitlines() if x.strip()]
        _opt_ids = [x for x in _opt_ids_raw if len(x) >= 10]
        if not _opt_ids:
            st.warning("レースIDを入力してから実行してください。")
        elif "driver" not in st.session_state or st.session_state.driver is None:
            st.warning("ドライバを起動してから実行してください。")
        else:
            with st.spinner("閾値最適化中（数分かかる場合があります）..."):
                try:
                    from race_history_ai import fetch_race_history_enriched
                    from threshold_optimizer import (
                        optimize_thresholds,
                        format_threshold_comparison,
                    )
                    from signal_judge import SIGNAL_THRESHOLDS as _current_thresh

                    _opt_driver = st.session_state.driver
                    _all_enriched: list = []
                    for _oid in _opt_ids:
                        try:
                            _oe = fetch_race_history_enriched(_opt_driver, _oid, years=11)
                            _test_y = int(_oid[:4])
                            _oe = [r for r in _oe if not str(r.get("race_id","")).startswith(str(_test_y))]
                            _all_enriched.extend(_oe)
                        except Exception:
                            continue

                    if not _all_enriched:
                        st.warning("最適化に必要な過去データが取得できませんでした。")
                    else:
                        _opt_result = optimize_thresholds(_all_enriched)
                        st.success(_opt_result["note"])
                        st.markdown(
                            format_threshold_comparison(
                                _current_thresh,
                                _opt_result["best_thresholds"],
                            )
                        )
                        _om1, _om2, _om3 = st.columns(3)
                        _om1.metric("データサンプル数",  f"{_opt_result['data_size']:,}件")
                        _om2.metric("ベースライン3着内率", f"{_opt_result['base_top3_rate']*100:.1f}%")
                        _om3.metric(
                            "最適スコア",
                            f"{_opt_result['best_score']:+.4f}",
                            help="Positive Lift × 0.6 + Negative Lift × 0.4 の加重和",
                        )

                        if _opt_result["metrics"]:
                            _mm = _opt_result["metrics"]
                            _c1, _c2, _c3 = st.columns(3)
                            _c1.metric(
                                "Positive精度",
                                f"{_mm.get('precision_pos',0)*100:.1f}%",
                                delta=f"{_mm.get('lift_pos',0)*100:+.1f}%p",
                            )
                            _c2.metric(
                                "Top3馬のRecall",
                                f"{_mm.get('recall_top3',0)*100:.1f}%",
                            )
                            _c3.metric(
                                "Negative精度",
                                f"{_mm.get('precision_neg',0)*100:.1f}%",
                                delta=f"{_mm.get('lift_neg',0)*100:+.1f}%p (lower=better)",
                            )
                        st.info(
                            "推奨閾値を反映するには `signal_judge.py` の `SIGNAL_THRESHOLDS` を手動で更新してください。"
                        )

                        # コンボ重み最適化
                        st.markdown("##### コンボ条件 重み最適化")
                        try:
                            from threshold_optimizer import optimize_combo_weights
                            from signal_judge import COMBO_CONDITION_WEIGHTS as _cur_cw
                            _cw_result = optimize_combo_weights(_all_enriched)
                            _cw_rows = [
                                {
                                    "コンボ条件": ck,
                                    "実測リフト": f"{lift:+.3f}",
                                    "現行重み":   f"{_cur_cw.get(ck, 0.60):.2f}",
                                    "推奨重み":   f"{_cw_result['weights'].get(ck, 0.60):.2f}",
                                }
                                for ck, lift in _cw_result["lifts"].items()
                            ]
                            st.dataframe(_cw_rows, use_container_width=True, hide_index=True)
                            st.info(
                                "推奨重みを反映するには `signal_judge.py` の `COMBO_CONDITION_WEIGHTS` を手動で更新してください。"
                            )
                        except Exception as _cw_exc:
                            st.warning(f"コンボ重み最適化失敗: {_cw_exc}")
                except Exception as _opt_exc:
                    st.error(f"最適化エラー: {_opt_exc}")

    # ── 学習データ出力 & モデル再訓練 ──────────────────────────────────────
    st.markdown("---")
    st.markdown("#### ML学習データ出力 & モデル再訓練")
    st.caption(
        "バックテスト対象レースIDを使って `keiba_training_data.csv` を生成し、"
        "LightGBMモデルを訓練します。目安: 50レース以上でモデルが機能し始めます。"
    )
    _ml_col1, _ml_col2 = st.columns(2)
    _export_btn  = _ml_col1.button("学習データを出力", help="バックテストIDから CSV を生成します（追記モード）")
    _retrain_btn = _ml_col2.button("モデルを再訓練",   help="keiba_training_data.csv からLightGBMを訓練します")

    if _export_btn:
        _exp_ids_raw = [x.strip() for x in (_bt_race_ids_raw or "").splitlines() if x.strip()]
        _exp_ids     = [x for x in _exp_ids_raw if len(x) >= 10]
        if not _exp_ids:
            st.warning("レースIDを入力してから実行してください。")
        else:
            _exp_progress = st.progress(0)
            _exp_status   = st.empty()
            def _exp_cb(_c, _t, _r):
                _exp_status.text(f"処理中 {_c}/{_t}: {_r}")
                _exp_progress.progress(_c / _t)
            try:
                from backtest_runner import export_training_csv_noselenium
                _exp_result = export_training_csv_noselenium(
                    _exp_ids, append=True, progress_callback=_exp_cb,
                )
                _exp_progress.empty()
                _exp_status.empty()
                st.success(
                    f"完了: {_exp_result['n_races']}レース / "
                    f"{_exp_result['n_horses']}行 → `{_exp_result['csv_path']}`"
                )
                if _exp_result.get("n_errors", 0) > 0:
                    st.warning(f"失敗: {_exp_result['n_errors']}件")
                    for _ee in _exp_result.get("errors", [])[:5]:
                        st.text(f"  {_ee['race_id']}: {_ee['error']}")
            except Exception as _exp_exc:
                st.error(f"学習データ出力エラー: {_exp_exc}")

    if _retrain_btn:
        with st.spinner("LightGBMモデル訓練中..."):
            try:
                from race_ai_engine import train_lightgbm_model, TRAINING_CSV
                import os as _os
                if not _os.path.exists(TRAINING_CSV):
                    st.warning(f"`{TRAINING_CSV}` が存在しません。先に学習データを出力してください。")
                else:
                    import pandas as _pd
                    _df_size = len(_pd.read_csv(TRAINING_CSV))
                    ok = train_lightgbm_model()
                    if ok:
                        st.success(f"モデル訓練完了（学習行数: {_df_size:,}行）")
                    else:
                        st.error("訓練に失敗しました。LightGBM/pandasがインストールされているか確認してください。")
            except Exception as _rt_exc:
                st.error(f"再訓練エラー: {_rt_exc}")