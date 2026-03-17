import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import datetime
import re
import requests
from bs4 import BeautifulSoup

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
    assign_marks,
    assign_roles,
    detect_rescue_candidates,
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
from analytics_ai import build_full_analytics


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
  --accent-green: #00FF00;
  --accent-pink: #FF00FF;
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

/* カード風パネル：黒背景にネオンの枠線と光彩 */
.card {
  background: var(--card-bg);
  border: 1px solid var(--accent-green);
  border-radius: 14px;
  padding: 16px;
  box-shadow: 0 0 10px rgba(0, 255, 0, 0.2), 0 0 20px rgba(255, 0, 255, 0.15);
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

/* メトリック表示：カードと同系色で枠線にピンクを使用 */
[data-testid="stMetric"]{
  background: var(--card-bg);
  border: 1px solid var(--accent-pink);
  border-radius: 14px;
  padding: 14px;
  box-shadow: 0 0 8px rgba(255, 0, 255, 0.2);
}
[data-testid="stMetricLabel"] {
  color: var(--muted-color) !important;
  font-weight: 600;
  font-size: 12px;
  letter-spacing: 0.5px;
  font-family: 'DotGothic16', sans-serif;
}
[data-testid="stMetricValue"] {
  color: var(--text-color) !important;
  font-weight: 600;
  font-size: 20px !important;
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

</style>
    """,
    unsafe_allow_html=True,
)

st.title("🐎 KEIBA AI ANALYZER")

st.markdown("---")

st.markdown("""
### 🤖 KEIBA AI version9
AIがレース展開・能力・期待値・展開シミュレーション・オッズ歪みを総合分析

表示内容
- AI本命 / AI信頼度
- 穴馬AI / 危険人気馬AI
- レース展開AI / 脚質バランス
- 能力レーダー / AIヒートマップ
- 三連複 / 三連単参考
- 推奨馬券 / レース難易度
- GPT解説コメント
""")

# -----------------------------
# INPUT
# -----------------------------

col1, col2 = st.columns([6.6, 1.6], vertical_alignment="top")

with col1:
    race_url = st.text_input(
        "netkeiba URL（出馬表 / 競馬新聞どちらでも可）",
        value="",
        placeholder="例: https://race.netkeiba.com/race/shutuba.html?race_id=202406030811"
    )
    st.caption("※ 出馬表URL（shutuba.html）または競馬新聞URL（newspaper.html）のどちらか1つだけ入力してください。AIが不足ページを自動補完します。")

with col2:
    bankroll = st.number_input("軍資金", min_value=100, value=1000, step=100)

analyze_col1, analyze_col2 = st.columns([2.2, 5.8])
with analyze_col1:
    analyze = st.button("AI分析を実行", use_container_width=True)

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
                st.session_state.result = result
            except Exception as e:
                st.error("AI分析エラー")
                st.exception(e)

# --- Use stored result so Streamlit reruns don't reset the page ---
if st.session_state.result:

    result = st.session_state.result

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

    # 出馬表順（馬番優先、無ければ枠順）
    if "horse_number" in df.columns:
        df_sorted = df.sort_values("horse_number")
    elif "umaban" in df.columns:
        df_sorted = df.sort_values("umaban")
    elif "gate" in df.columns:
        df_sorted = df.sort_values("gate")
    else:
        df_sorted = df.copy()

    # -----------------------------
    # MANUAL ODDS INPUT / FETCH
    # -----------------------------

    with st.expander("🎯 単勝オッズ入力・取得", expanded=False):
        st.caption("通常は自動取得を使用し、取得失敗時のみ手動入力を使ってください。")

        if st.button("netkeibaの予想オッズを取得", key="fetch_predicted_odds_btn"):
            normalized_race_url = race_url.strip()
            fetched_odds = get_predicted_odds_from_netkeiba(normalized_race_url)

            if fetched_odds:
                applied_count = 0
                for idx, row in df.iterrows():
                    horse_no = row.get("horse_number", row.get("umaban", row.get("gate", "")))
                    horse_no_key = str(int(horse_no)) if pd.notna(horse_no) and str(horse_no) != "" else None
                    horse_name = row.get("horse_name", "")

                    matched_odds = None
                    if horse_no_key and horse_no_key in fetched_odds:
                        matched_odds = fetched_odds[horse_no_key]
                    elif horse_name and horse_name in fetched_odds:
                        matched_odds = fetched_odds[horse_name]

                    if matched_odds is not None:
                        df.loc[idx, "win_odds"] = matched_odds
                        applied_count += 1

                result["features"] = df.to_dict("records")

                if applied_count > 0:
                    st.success(f"予想オッズを {applied_count} 頭に反映しました")
                    st.session_state.result = result
                    st.rerun()
                else:
                    st.warning("予想オッズ候補は取得できましたが、馬名または馬番との照合に失敗しました。競馬新聞URL（newspaper.html）でも試してください。")
            else:
                st.warning("netkeibaから予想オッズを取得できませんでした。shutuba.html / newspaper.html の両方を自動で試しましたが取得できないため、必要な場合のみ下の手動オッズ入力を使用してください。")

        with st.expander("手動オッズ入力", expanded=False):
            st.caption("自動取得が失敗した場合のみ、各馬の単勝オッズを入力してください。")

            manual_odds_view = df_sorted.reset_index().rename(columns={"index": "source_index"})

            head1, head2 = st.columns([5.4, 1.8], gap="small")
            head1.markdown("**馬名**")
            head2.markdown("**単勝オッズ**")

            current_manual_odds = {}

            for display_idx, row in manual_odds_view.iterrows():
                source_idx = int(row["source_index"])
                horse_no = row.get("horse_number", row.get("umaban", row.get("gate", "")))
                horse_name = row.get("horse_name", f"馬{display_idx + 1}")

                has_horse_no = pd.notna(horse_no) and str(horse_no).strip() not in ["", "nan", "None"]
                if has_horse_no:
                    try:
                        label = f"{int(float(horse_no))}番 {horse_name}"
                    except Exception:
                        label = f"{horse_no}番 {horse_name}"
                else:
                    label = horse_name

                widget_key = f"manual_win_odds_input_{display_idx}_{source_idx}"
                current_val = st.session_state.get(
                    widget_key,
                    safe_float(row.get("win_odds", 0.0), 0.0),
                )

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

                current_manual_odds[source_idx] = float(val)

            if st.button("入力した単勝オッズをAIに反映", key="apply_manual_win_odds_btn"):
                for source_idx, odds in current_manual_odds.items():
                    df.loc[source_idx, "win_odds"] = odds

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

        manual_styles = {}




        def render_style_group(title, group_df):
            if group_df.empty:
                return
            # --- visual separator ---
            st.markdown("---")
            # section title
            st.markdown(f"### {title}")
            # helper caption
            st.caption("必要な馬のみ脚質を変更してください")
            cols = st.columns(2)
            for idx, (i, row) in enumerate(group_df.iterrows()):
                col = cols[idx % 2]
                with col:
                    label = row.get("horse_name", f"馬{i+1}")
                    choice = st.radio(
                        label,
                        ["自動", "逃げ 🟥", "先行 🟧", "差し 🟦", "追込 🟪"],
                        horizontal=True,
                        key=f"style_{i}",
                    )
                    clean_choice = (
                        choice.replace(" 🟥", "")
                        .replace(" 🟧", "")
                        .replace(" 🟦", "")
                        .replace(" 🟪", "")
                    )
                    if clean_choice != "自動":
                        manual_styles[i] = style_mapping.get(clean_choice)
            # bottom separator
            st.markdown("---")

        # --- 出馬表順 UI（枠順順）---
        st.markdown("### 🐎 出馬表順（脚質設定）")
        st.caption("出馬表と同じ順番で脚質を変更できます")

        cols = st.columns(2)

        for idx, (i, row) in enumerate(df_sorted.iterrows()):
            col = cols[idx % 2]
            with col:
                horse_no = row.get("horse_number", row.get("umaban", row.get("gate", "")))
                gate = horse_no
                horse = row.get("horse_name", f"馬{i+1}")
                label = f"{gate}番 {horse}" if gate != "" else horse
                choice = st.radio(
                    label,
                    ["自動", "逃げ 🟥", "先行 🟧", "差し 🟦", "追込 🟪"],
                    horizontal=True,
                    key=f"style_{i}",
                )
                clean_choice = (
                    choice.replace(" 🟥", "")
                    .replace(" 🟧", "")
                    .replace(" 🟦", "")
                    .replace(" 🟪", "")
                )
                if clean_choice != "自動":
                    manual_styles[i] = style_mapping.get(clean_choice)

        st.markdown("---")

        # --- Auto apply running style changes ---
        changed = False

        for idx, style in manual_styles.items():
            if df.loc[idx, "running_style"] != style:
                df.loc[idx, "running_style"] = style
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

    st.markdown('<div class="card">', unsafe_allow_html=True)
    # レース名（横フル）
    st.metric("レース", race_meta.get("race_title", "-"))

    # 3カラム
    meta1, meta2, meta3 = st.columns(3)

    meta1.metric("コース", race_meta.get("target_course", "-"))
    meta2.metric("距離", race_meta.get("target_distance", "-"))
    meta3.metric("想定ペース", race_meta.get("predicted_pace", "-"))

    sub1, sub2, sub3 = st.columns(3)

    sub1.metric("レースタイプ", result.get("race_type", "判定不可"))
    sub2.metric("期待回収率", roi_label(result.get("expected_roi", 1.0)))
    sub3.metric("難易度", difficulty_label(df["win_prob"].sort_values(ascending=False).values))

    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("🏆 AI本命")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    mark = fav.get("newspaper_mark", "")
    c1.metric("馬名", f'{fav["horse_name"]} {mark}')
    c2.metric("勝率", pct(fav.get("win_prob")))
    c3.metric("AIフェアオッズ", num(fav.get("fair_win_odds"), 2))
    c4.metric("AIパワー指数", num(fav.get("ai_power_index"), 1))

    # --- AI POWER GAUGE ---
    try:
        ai_power_val = float(fav.get("ai_power_index", 0))
    except Exception:
        ai_power_val = 0

    ai_power_val = max(0, min(200, ai_power_val))

    st.caption("AIパワーゲージ")

    power_ratio = ai_power_val / 200

    st.progress(power_ratio)

    st.write(f"AI POWER: {round(ai_power_val,1)} / 200")
    st.caption(
        f"脚質: {fav.get('pace_style_label', jp_style_label(fav.get('running_style','不明')))}"
        f" / モンテカルロ: {pct(fav.get('montecarlo_win_prob', 0))}"
        f" / 妙味指数: {fav.get('value_index', 0)}"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("AI信頼度（モデル確信度）")
    confidence = result.get("ai_confidence", 0.5) * 100
    st.progress(int(confidence))
    st.write(f"{confidence:.1f}%")

    top_tabs = st.tabs(["ランキング", "展開", "ポジション", "馬券", "期待値AI", "AI分析", "回顧・検証", "レビュー分析"])

    with top_tabs[0]:

        st.subheader("勝率ランキング")

        # Ensure optional columns exist to avoid KeyError
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
            "win_prob": "勝率",
            "place_prob": "複勝圏AI",
            "fair_win_odds": "AIフェア単勝",
            "ai_power_index": "AIパワー",
        })

        rank["勝率"] = rank["勝率"].apply(lambda x: round(x * 100, 1))
        rank["複勝圏AI"] = rank["複勝圏AI"].apply(lambda x: round(x * 100, 1))

        st.dataframe(rank.sort_values("勝率", ascending=False), use_container_width=True)

        st.subheader("📰 新聞AI評価")

        if "newspaper_mark" in df.columns:
            mark_df = df[["horse_name","newspaper_mark","win_prob","ai_power_index"]].copy()
            mark_df.columns = ["馬名","新聞印","勝率","AIパワー"]
            mark_df["勝率"] = mark_df["勝率"].apply(lambda x: round(x*100,1))
            st.dataframe(mark_df, use_container_width=True)

        st.subheader("期待値ランキング")
        ev = df[["horse_name", "win_ev", "place_ev", "value_index"]].copy()
        ev = ev.rename(columns={
            "horse_name": "馬名",
            "win_ev": "単勝EV",
            "place_ev": "複勝EV",
            "value_index": "妙味指数",
        })
        st.dataframe(ev.sort_values("単勝EV", ascending=False), use_container_width=True)

        st.subheader("AI穴馬候補")
        dark_horses = result.get("dark_horses", [])
        if dark_horses:
            dark_df = pd.DataFrame(dark_horses)[["horse_name", "win_prob", "win_odds", "value_index"]].copy()
            dark_df.columns = ["馬名", "勝率", "単勝オッズ", "妙味指数"]
            dark_df["勝率"] = dark_df["勝率"].apply(lambda x: round(x * 100, 1))
            st.dataframe(dark_df, use_container_width=True)
        else:
            st.write("AIが検出した穴馬はありません")

        st.subheader("AI危険人気馬")
        danger_favorites = result.get("danger_favorites", [])
        if danger_favorites:
            danger_df = pd.DataFrame(danger_favorites)[["horse_name", "win_prob", "win_odds", "danger_gap"]].copy()
            danger_df.columns = ["馬名", "勝率", "単勝オッズ", "危険度ギャップ"]
            danger_df["勝率"] = danger_df["勝率"].apply(lambda x: round(x * 100, 1))
            st.dataframe(danger_df, use_container_width=True)
        else:
            st.write("危険人気馬は検出されませんでした")

        st.subheader("🔥 オッズ歪みAI（VALUE検出）")

        value_horses = result.get("value_horses", [])

        if value_horses:
            value_df = pd.DataFrame(value_horses)[[
                "horse_name",
                "win_prob",
                "win_odds",
                "odds_distortion_index",
                "value_flag"
            ]].copy()

            value_df.columns = [
                "馬名",
                "勝率",
                "単勝オッズ",
                "歪み指数",
                "評価"
            ]

            value_df["勝率"] = value_df["勝率"].apply(lambda x: round(x * 100, 1))
            value_df["歪み指数"] = value_df["歪み指数"].apply(lambda x: round(x, 2))

            st.dataframe(
                value_df.sort_values("歪み指数", ascending=False),
                use_container_width=True
            )

        else:
            st.write("オッズ歪みは検出されませんでした")

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

    with top_tabs[2]:
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
            st.dataframe(show_df, use_container_width=True)
        else:
            st.write("ポジション分析データがありません")

    with top_tabs[3]:
        st.subheader("AI投資サマリー")
        sum1, sum2, sum3 = st.columns(3)
        sum1.metric("期待回収率", roi_label(result.get("expected_roi", 1.0)))
        sum2.metric("レースタイプ", result.get("race_type", "判定不可"))
        sum3.metric("AI信頼度", f"{result.get('ai_confidence', 0.5) * 100:.1f}%")

        st.subheader("AI三連複候補")
        if len(df) >= 3:
            trio = df.sort_values("win_prob", ascending=False).head(5)[["horse_name", "win_prob", "win_odds"]].copy()
            trio.columns = ["馬名", "勝率", "単勝オッズ"]
            trio["勝率"] = trio["勝率"].apply(lambda x: round(x * 100, 1))
            st.dataframe(trio, use_container_width=True)
        else:
            st.write("データ不足")

        st.subheader("AI三連単参考")
        if len(df) >= 3:
            top = df.sort_values("win_prob", ascending=False).head(3)[["horse_name", "win_prob", "win_odds"]].copy()
            top.columns = ["馬名", "勝率", "単勝オッズ"]
            top["勝率"] = top["勝率"].apply(lambda x: round(x * 100, 1))
            st.dataframe(top, use_container_width=True)
        else:
            st.write("データ不足")

        # -------------------------------------------------
        # AIフォーメーション馬券（engine側 generate_ai_bets）
        # -------------------------------------------------
        st.subheader("🤖 AIフォーメーション候補")

        ai_bets = result.get("ai_bets", [])

        if ai_bets:

            ai_bets_df = pd.DataFrame(ai_bets)

            # UI用に列名を日本語化
            ai_bets_df = ai_bets_df.rename(columns={
                "type": "券種",
                "horse": "馬名",
                "horses": "組み合わせ",
                "axis": "軸",
                "others": "相手"
            })

            st.dataframe(ai_bets_df, use_container_width=True)

        else:

            st.info("AIフォーメーション候補はまだ生成されていません")

        # -------------------------------------------------
        # AI資金配分エンジン
        # -------------------------------------------------

        st.subheader("💰 AI資金配分エンジン")

        try:
            bankroll_local = bankroll

            alloc_df = df.copy()

            if "win_odds" in alloc_df.columns:
                alloc_df["kelly_fraction"] = (
                    (alloc_df["win_prob"] * alloc_df["win_odds"] - 1)
                    / (alloc_df["win_odds"] - 1)
                )

                alloc_df["kelly_fraction"] = alloc_df["kelly_fraction"].clip(lower=0)

                alloc_df = alloc_df.sort_values("kelly_fraction", ascending=False)

                alloc_df = alloc_df.head(5)

                alloc_df["stake"] = (alloc_df["kelly_fraction"] * bankroll_local).round(-2)

                show_alloc = alloc_df[[
                    "horse_name",
                    "win_prob",
                    "win_odds",
                    "kelly_fraction",
                    "stake"
                ]].copy()

                show_alloc.columns = [
                    "馬名",
                    "勝率",
                    "単勝オッズ",
                    "ケリー指数",
                    "推奨投資額"
                ]

                show_alloc["勝率"] = show_alloc["勝率"].apply(lambda x: round(x*100,1))
                show_alloc["ケリー指数"] = show_alloc["ケリー指数"].apply(lambda x: round(x,2))

                st.dataframe(show_alloc, use_container_width=True)

            else:
                st.info("オッズが無いため資金配分を計算できません")

        except Exception:
            st.info("資金配分計算に失敗しました")

    # =====================================================================
    # TAB 4: 期待値AI（馬券期待値・妙味馬・危険人気馬・推奨買い目）
    # =====================================================================
    with top_tabs[4]:
        # ① レース構造分析
        st.subheader("① レース構造分析")
        race_structure = result.get("race_structure") or {}
        pace_balance_ev = result.get("pace_balance", {})
        if not race_structure:
            race_structure = classify_race_structure(
                result.get("features", []), pace_balance_ev
            )

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

        st.markdown("---")

        # ② 印
        st.subheader("② 印")
        horse_marks = result.get("horse_marks") or []
        ev_table_ui = result.get("ev_table") or []
        if not ev_table_ui:
            ev_table_ui = build_ev_table(result.get("features", []))
        if not horse_marks:
            danger_v2_tmp = result.get("danger_favorites_v2") or []
            horse_marks = assign_marks(result.get("features", []), ev_table_ui, danger_v2_tmp)

        race_pace_ev = result.get("race_meta", {}).get("predicted_pace", "medium") or "medium"

        if horse_marks:
            marks_by_name = {row["horse_name"]: row["mark"] for row in horse_marks}
            ev_by_name    = {row["horse_name"]: row for row in ev_table_ui}

            mark_rows = []
            for f in sorted(
                result.get("features", []),
                key=lambda x: float(x.get("win_prob") or 0.0),
                reverse=True,
            ):
                name     = str(f.get("horse_name") or "")
                mark     = marks_by_name.get(name, "")
                ev_row   = ev_by_name.get(name, {})
                vg       = ev_row.get("value_gap")
                ai_prob  = float(f.get("win_prob") or 0.0)
                mk_prob  = ev_row.get("market_win_prob")
                win_odds = ev_row.get("win_odds")
                mark_rows.append({
                    "印":           mark,
                    "馬名":         name,
                    "AI勝率":       f"{ai_prob * 100:.1f}%",
                    "市場期待勝率": f"{mk_prob * 100:.1f}%" if mk_prob else "-",
                    "単勝オッズ":   win_odds if win_odds else "-",
                    "value_gap":    f"{vg:+.3f}" if vg is not None else "-",
                })

            st.dataframe(
                pd.DataFrame(mark_rows),
                use_container_width=True,
            )
        else:
            st.info("印を計算するにはオッズを入力してください。")

        st.markdown("---")

        # ③ 馬役割分類（頭/軸/ヒモ/消し）
        st.subheader("③ 馬役割分類")
        horse_roles = assign_roles(result.get("features", []), ev_table_ui, race_structure)

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
                    win_pct  = f"{r['win_prob']*100:.1f}%"
                    top3_pct = f"{r['top3_prob']*100:.1f}%"
                    st.markdown(
                        f"**{r['horse_name']}**  \n"
                        f"勝率{win_pct} / 3着内{top3_pct}  \n"
                        f"_{r['reason']}_"
                    )

        st.markdown("---")

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

        # ④ 取りこぼし注意馬
        st.subheader("④ 取りこぼし注意馬")
        rescue_horses = detect_rescue_candidates(
            result.get("features", []),
            ev_table_ui,
            race_structure,
            race_pace_ev,
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

        # ⑤ 妙味馬
        st.subheader("⑤ 妙味馬")
        value_v2 = result.get("value_horses_v2") or []
        if not value_v2:
            value_v2 = detect_value_horses(ev_table_ui, result.get("features", []), race_pace_ev)

        if value_v2:
            for vh in value_v2:
                vg_val = vh.get("value_gap")
                odds_v = vh.get("win_odds")
                tag = "🔥 強妙味" if vg_val is not None and vg_val >= 0.06 else "💡 妙味"
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

        # ⑥ 危険な人気馬（v3: 真危険 vs 相手残り の区別）
        st.subheader("⑥ 危険な人気馬")
        danger_v2 = detect_danger_favorites_v3(
            ev_table_ui,
            result.get("features", []),
            race_structure,
            race_pace_ev,
        )

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
                st.markdown("**⚠️ 相手なら残る（頭では危険）**")
                for dh in maybe_list:
                    odds_d = dh.get("win_odds")
                    st.markdown(
                        f"**△ {dh['horse_name']}**  "
                        f"（AI勝率 {dh.get('ai_win_prob', 0)*100:.1f}% "
                        f"/ 市場勝率 {dh.get('market_win_prob', 0)*100:.1f}% "
                        f"/ オッズ {odds_d or '-'}倍）"
                    )
                    st.caption(f"理由: {dh.get('reason', '-')}")
        else:
            st.info("危険人気馬は検出されませんでした。")

        st.markdown("---")

        # ⑦ 推奨買い目（1案に絞る）
        st.subheader("⑦ 推奨買い目")
        if not ev_table_ui:
            st.info("オッズを入力後に推奨買い目が表示されます。")
        else:
            # 券種別比較サマリー（plan 取得前に build_ticket_ev_table を呼ぶ）
            ticket_evs_summary = build_ticket_ev_table(
                result.get("features", []),
                top_n=6,
                race_structure=race_structure,
                ev_table=ev_table_ui,
            )
            best_by_type: dict = {}
            for r in ticket_evs_summary:
                bt = r["bet_type"]
                if bt not in best_by_type or r["ev"] > best_by_type[bt]["ev"]:
                    best_by_type[bt] = r

            st.markdown("**券種別 EV サマリー**")
            sum_cols = st.columns(5)
            for ci, bt in enumerate(["単勝", "複勝", "馬連", "ワイド", "3連複"]):
                row_r = best_by_type.get(bt)
                if row_r:
                    hit_pct = f"hit:{row_r['ai_hit_prob']*100:.1f}%"
                    sum_cols[ci].metric(bt, f"EV {row_r['ev']:.2f}", delta=hit_pct)
                else:
                    sum_cols[ci].metric(bt, "-")

            st.markdown("---")

            bet_plan = recommend_bet_plan(
                result.get("features", []),
                ev_table_ui,
                race_structure,
                bankroll,
                race_pace_ev,
            )

            if bet_plan.get("skip"):
                st.warning(f"🚫 見送り推奨")
                skip_reason = bet_plan.get("skip_reason", "")
                if skip_reason:
                    st.caption(skip_reason)
                # 見送り時は全券種EV一覧を参考表示
                if ticket_evs_summary:
                    with st.expander("参考: 全券種EV（閾値未達）", expanded=False):
                        st.caption("※ いずれも推奨閾値（単勝/複勝≥1.02、複合≥0.95）に届きませんでした。")
                        skip_rows = [{
                            "券種": r["bet_type"],
                            "馬": " / ".join(r["horses"]),
                            "EV(補正後)": f"{r['ev']:.3f}",
                        } for r in ticket_evs_summary[:8]]
                        st.dataframe(pd.DataFrame(skip_rows), use_container_width=True)
            else:
                bp_cols = st.columns(3)
                bp_cols[0].metric("券種", bet_plan.get("bet_type", "-"))
                bp_cols[1].metric("点数", f"{bet_plan.get('ticket_count', 0)}点")
                bp_cols[2].metric("合計金額", f"¥{bet_plan.get('total_stake', 0):,}")

                risk_col, ev_col = st.columns(2)
                risk_col.metric("リスクレベル", bet_plan.get("risk_level", "-"))

                ev_type = bet_plan.get("ev_type", "-")
                ev_type_label = (
                    "📊 EV比較型" if ev_type == "EV比較型"
                    else "🗺️ 構造型" if ev_type == "構造型"
                    else ev_type
                )
                ev_col.metric("選定方式", ev_type_label)

                st.caption(f"根拠: {bet_plan.get('reason', '-')}")

                # 危険人気馬を除外した場合の注釈
                if ev_type == "EV比較型" and danger_v2:
                    excluded = [d["horse_name"] for d in danger_v2]
                    st.caption(f"⚠️ 危険人気馬（{' / '.join(excluded)}）を含む組み合わせは減点されています。")

                tickets = bet_plan.get("tickets", [])
                if tickets:
                    st.markdown("**買い目詳細**")
                    ticket_rows = [
                        {
                            "組み合わせ": "・".join(t.get("combination", [])),
                            "金額": f"¥{t.get('stake', 0):,}",
                        }
                        for t in tickets
                    ]
                    st.dataframe(pd.DataFrame(ticket_rows), use_container_width=True)

                # 推奨理由詳細
                sel_detail = bet_plan.get("selection_detail", {})
                if sel_detail:
                    with st.expander("推奨理由の詳細", expanded=False):
                        if sel_detail.get("why_bet_type"):
                            st.markdown(f"**券種選択:** {sel_detail['why_bet_type']}")
                        if sel_detail.get("why_combo"):
                            st.markdown(f"**組み合わせ:** {sel_detail['why_combo']}")
                        if sel_detail.get("why_not_other"):
                            st.markdown(f"**他券種比較:** {sel_detail['why_not_other']}")

        st.markdown("---")

        # ⑧ 券種別EVランキング
        with st.expander("⑧ 券種別EVランキング（上位5件）", expanded=False):
            if ev_table_ui:
                ticket_ev_rows = build_ticket_ev_table(
                    result.get("features", []),
                    top_n=6,
                    race_structure=race_structure,
                    ev_table=ev_table_ui,
                )
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

        # ⑨ 単馬EV テーブル詳細
        with st.expander("⑨ 単馬期待値テーブル詳細", expanded=False):
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
                st.info("過去10年の全走者データが取得できた場合に表示されます。")
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

    with top_tabs[5]:
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
    # TAB 6: 回顧・検証
    # =====================================================================
    with top_tabs[6]:
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
                _existing_result = _sel_record.get("result") or {}
                _horse_names = [h["horse_name"] for h in _sel_record.get("horses", [])]

                with st.form(key="result_form"):
                    st.markdown(f"**{_sel_record['race_name']}** — {_sel_record['race_date']}")
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

                    pc1, pc2 = st.columns(2)
                    _win_pay   = pc1.number_input("単勝払戻（円、0=該当なし）",
                                                   min_value=0, step=10,
                                                   value=int(_existing_result.get("win_payout") or 0),
                                                   key="win_pay")
                    _ret_amt   = pc2.number_input("実際の払戻金額（円）",
                                                   min_value=0, step=100,
                                                   value=int(_existing_result.get("return_amount") or 0),
                                                   key="ret_amt")

                    _actual_bets = st.text_area(
                        "実際の購入内容（任意）",
                        value=_existing_result.get("actual_bets", ""),
                        placeholder="例: 馬連 イクイノックス-ジャスティンパレス 500円",
                        key="actual_bets",
                    )

                    _memo = st.text_area("メモ（任意）",
                                         value=_existing_result.get("review_comment", ""),
                                         key="result_memo")

                    _submitted = st.form_submit_button("結果を保存")

                if _submitted:
                    _finish_order = [x for x in [_1st, _2nd, _3rd] if x]
                    _invest       = _sel_record.get("investment_amount", 0)
                    _bet_type_s   = _sel_record.get("recommended_bet_type", "")
                    _tickets_s    = _sel_record.get("recommended_tickets", [])
                    _hit          = check_bet_hit(_bet_type_s, _tickets_s, _finish_order) if _finish_order else False

                    from datetime import datetime as _dt
                    _result_data = {
                        "finish_order":      _finish_order,
                        "win_payout":        _win_pay if _win_pay > 0 else None,
                        "place_payouts":     {},
                        "hit":               _hit,
                        "return_amount":     _ret_amt,
                        "investment_amount": _invest,
                        "roi":               round(_ret_amt / _invest, 4) if _invest > 0 else None,
                        "actual_bets":       _actual_bets,
                        "review_comment":    _memo,
                        "review_tags":       [],
                        "review_labels":     [],
                        "entered_at":        _dt.now().isoformat(timespec="seconds"),
                    }
                    # 回顧タグを生成してから保存
                    _tmp_record = dict(_sel_record)
                    _tmp_record["result"] = _result_data
                    _rev_result = build_review_result(_tmp_record, _actual_bets)
                    _result_data["review_tags"]    = _rev_result["review_tags"]
                    _result_data["review_summary"] = _rev_result["summary"]
                    _result_data["return_rate"]    = _rev_result["return_rate"]
                    # 旧フィールド互換
                    _result_data["review_labels"]  = _rev_result["review_tags"]

                    update_race_result(_selected_id, _result_data)
                    st.success(f"結果を保存しました。的中: {'○' if _hit else '✗'}")
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
                    "券種別", "レース構造別", "EV種別", "グレード別",
                    "危険馬", "妙味馬",
                    "役割精度", "外れ方", "見送り判断", "改善提案",
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

                with _tabs_summary[4]:
                    _dc = _analytics["danger_cutoff"]
                    _dc_cols = st.columns(3)
                    _dc_cols[0].metric("危険馬含むレース数",   _dc["n_races_with_danger"])
                    _dc_cols[1].metric("消し推奨・正解率",
                                       f"{_dc['danger_truly_correct_rate']*100:.1f}%",
                                       help=f"{_dc['truly_total']}頭中")
                    _dc_cols[2].metric("「相手なら残る」3着内率",
                                       f"{_dc['danger_soft_placed_rate']*100:.1f}%",
                                       help=f"{_dc['soft_total']}頭中")

                with _tabs_summary[5]:
                    _vh = _analytics["value_horse"]
                    _vh_cols = st.columns(3)
                    _vh_cols[0].metric("妙味馬3着内率",   f"{_vh['value_in_money_rate']*100:.1f}%",
                                       help=f"対象{_vh['value_total']}頭")
                    _vh_cols[1].metric("妙味馬買い目的中率", f"{_vh['value_bought_hit_rate']*100:.1f}%")

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

                # ── 役割精度 ──────────────────────────────────────────
                with _tabs_summary[6]:
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

                # ── 外れ方ランキング ──────────────────────────────────
                with _tabs_summary[7]:
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
                with _tabs_summary[8]:
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
                with _tabs_summary[9]:
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

            except Exception as _exc:
                st.error(f"集計エラー: {_exc}")

    # =====================================================================
    # TAB 7: レビュー分析
    # =====================================================================
    with top_tabs[7]:
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

        except Exception as _rv_exc:
            st.error(f"レビュー分析エラー: {_rv_exc}")