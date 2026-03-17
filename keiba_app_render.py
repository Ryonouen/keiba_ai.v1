import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import datetime
import re
import requests
from bs4 import BeautifulSoup

if "result" not in st.session_state:
    st.session_state.result = None

from race_ai_engine_render import (
    analyze_race,
    apply_bloodline_and_track_bias_to_result,
    recommend_bets,
    apply_simple_odds,
    refresh_result_payload,
)


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
/* 手動オッズ入力の横幅を詰める */
div[data-testid="column"] {
  gap: 0.2rem;
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
  font-weight: 600;
  font-size: 13px;
  border-radius: 12px;
  padding: 8px 18px;
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
  margin-bottom: 6px;
}

</style>
    """
    + """
<style>
/* 手動オッズ入力の1行をコンパクトに */
div[data-testid="stHorizontalBlock"] {
  align-items: center;
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

# -----------------------------
# TODAY RACE LIST
# -----------------------------

st.subheader("🐎 今日のレース（自動取得）")

if "today_races" not in st.session_state:
    st.session_state.today_races = []

if st.button("今日のレースを取得"):
    st.session_state.today_races = get_today_races()

if st.session_state.today_races:

    cols = st.columns(3)

    for i, race in enumerate(st.session_state.today_races[:12]):
        col = cols[i % 3]
        with col:
            if st.button(race["name"], key=f"racebtn_{race['race_id']}"):
                st.session_state["auto_race_url"] = race["url"]

col1, col2, col3 = st.columns([4, 1, 1])

with col1:
    race_url = st.text_input(
        "netkeiba URL（出馬表 / 競馬新聞どちらでも可）",
        value=st.session_state.get("auto_race_url", ""),
        placeholder="例: https://race.netkeiba.com/race/shutuba.html?race_id=202406030811"
    )
    st.caption("※ 出馬表URL（shutuba.html）または競馬新聞URL（newspaper.html）のどちらか1つだけ入力してください。AIが不足ページを自動補完します。")

with col2:
    bankroll = st.number_input("軍資金", value=1000, step=100)

with col3:
    history_note = st.caption("version9 UI")

analyze = st.button("AI分析を実行")

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

            head1, head2, head3 = st.columns([4.2, 1.2, 1.6], gap="small")
            head1.markdown("**馬名**")
            head2.markdown("")
            head3.markdown("**単勝オッズ**")

            current_manual_odds = {}

            for row_idx, row in df_sorted.iterrows():
                horse_no = row.get("horse_number", row.get("umaban", row.get("gate", "")))
                horse_name = row.get("horse_name", f"馬{row_idx+1}")
                label = f"{horse_no}番 {horse_name}" if str(horse_no) not in ["", "nan", "None"] else horse_name

                current_val = st.session_state.get(
                    f"manual_win_odds_input_{row_idx}",
                    safe_float(row.get("win_odds", 0.0), 0.0),
                )

                row_col1, row_col2, row_col3 = st.columns([4.2, 1.2, 1.6], gap="small")
                with row_col1:
                    st.markdown(
                        f"<div style='padding-top: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'><b>{label}</b></div>",
                        unsafe_allow_html=True,
                    )
                with row_col2:
                    st.markdown(
                        "<div style='padding-top: 8px; text-align: center; color: #94A3B8;'>：</div>",
                        unsafe_allow_html=True,
                    )
                with row_col3:
                    val = st.number_input(
                        "オッズ",
                        min_value=0.0,
                        step=0.1,
                        value=float(current_val),
                        key=f"manual_win_odds_input_{row_idx}",
                        label_visibility="collapsed",
                    )

                if val > 0:
                    current_manual_odds[row_idx] = float(val)

            if st.button("入力した単勝オッズをAIに反映", key="apply_manual_win_odds_btn"):
                for row_idx, odds in current_manual_odds.items():
                    df.loc[row_idx, "win_odds"] = odds

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

    top_tabs = st.tabs(["ランキング", "展開", "ポジション", "馬券", "AI分析"])

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

        st.dataframe(rank.sort_values("勝率", ascending=False), width="stretch")

        st.subheader("📰 新聞AI評価")

        if "newspaper_mark" in df.columns:
            mark_df = df[["horse_name","newspaper_mark","win_prob","ai_power_index"]].copy()
            mark_df.columns = ["馬名","新聞印","勝率","AIパワー"]
            mark_df["勝率"] = mark_df["勝率"].apply(lambda x: round(x*100,1))
            st.dataframe(mark_df, width="stretch")

        st.subheader("期待値ランキング")
        ev = df[["horse_name", "win_ev", "place_ev", "value_index"]].copy()
        ev = ev.rename(columns={
            "horse_name": "馬名",
            "win_ev": "単勝EV",
            "place_ev": "複勝EV",
            "value_index": "妙味指数",
        })
        st.dataframe(ev.sort_values("単勝EV", ascending=False), width="stretch")

        st.subheader("AI穴馬候補")
        dark_horses = result.get("dark_horses", [])
        if dark_horses:
            dark_df = pd.DataFrame(dark_horses)[["horse_name", "win_prob", "win_odds", "value_index"]].copy()
            dark_df.columns = ["馬名", "勝率", "単勝オッズ", "妙味指数"]
            dark_df["勝率"] = dark_df["勝率"].apply(lambda x: round(x * 100, 1))
            st.dataframe(dark_df, width="stretch")
        else:
            st.write("AIが検出した穴馬はありません")

        st.subheader("AI危険人気馬")
        danger_favorites = result.get("danger_favorites", [])
        if danger_favorites:
            danger_df = pd.DataFrame(danger_favorites)[["horse_name", "win_prob", "win_odds", "danger_gap"]].copy()
            danger_df.columns = ["馬名", "勝率", "単勝オッズ", "危険度ギャップ"]
            danger_df["勝率"] = danger_df["勝率"].apply(lambda x: round(x * 100, 1))
            st.dataframe(danger_df, width="stretch")
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
                width="stretch"
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
        st.plotly_chart(fig, width="stretch")

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
            st.plotly_chart(fig, width="stretch")
            show_df = pos_df[["horse_name", "style", "win_prob", "pace_simulation_index"]].copy()
            show_df.columns = ["馬名", "脚質", "勝率", "展開指数"]
            show_df["勝率"] = show_df["勝率"].apply(lambda x: round(float(x) * 100, 1))
            st.dataframe(show_df, width="stretch")
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
            st.dataframe(trio, width="stretch")
        else:
            st.write("データ不足")

        st.subheader("AI三連単参考")
        if len(df) >= 3:
            top = df.sort_values("win_prob", ascending=False).head(3)[["horse_name", "win_prob", "win_odds"]].copy()
            top.columns = ["馬名", "勝率", "単勝オッズ"]
            top["勝率"] = top["勝率"].apply(lambda x: round(x * 100, 1))
            st.dataframe(top, width="stretch")
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

            st.dataframe(ai_bets_df, width="stretch")

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

                st.dataframe(show_alloc, width="stretch")

            else:
                st.info("オッズが無いため資金配分を計算できません")

        except Exception:
            st.info("資金配分計算に失敗しました")

    with top_tabs[4]:
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
