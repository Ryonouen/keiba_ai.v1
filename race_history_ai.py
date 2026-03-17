from __future__ import annotations

from selenium.webdriver.common.by import By
from typing import List, Dict, Any, Optional, Tuple
from statistics import mean
from openai import OpenAI
import json
import os
import re
import time
import random
from trend_stats import bucket_gate

# -------------------------------------------------
# Utilities
# -------------------------------------------------

def random_sleep(a: float = 1.0, b: float = 2.0):
    time.sleep(random.uniform(a, b))


def safe_text(elem):
    try:
        return elem.text.strip()
    except Exception:
        return ""


def parse_int(text: str) -> Optional[int]:
    text = (text or "").strip()
    return int(text) if text.isdigit() else None


# -------------------------------------------------
# Helper functions
# -------------------------------------------------

def parse_float(text: str) -> Optional[float]:
    text = (text or "").strip().replace(",", "")
    try:
        return float(text)
    except Exception:
        return None


def normalize_style_label(style: str) -> str:
    mapping = {
        "front": "逃げ",
        "stalker": "先行",
        "closer": "差し",
        "unknown": "不明",
    }
    return mapping.get(style, str(style))


def bucket_popularity(popularity: Optional[int]) -> str:
    if popularity is None:
        return "不明"
    if popularity == 1:
        return "1番人気"
    if 2 <= popularity <= 3:
        return "2〜3番人気"
    if 4 <= popularity <= 6:
        return "4〜6番人気"
    if 7 <= popularity <= 9:
        return "7〜9番人気"
    return "10番人気以下"


def summarize_counts(d: Dict[str, int], top_n: int = 2) -> str:
    if not d:
        return "データなし"
    ordered = sorted(d.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return "・".join([f"{k}({v}回)" for k, v in ordered])


def infer_style(passing: str) -> str:
    if not passing:
        return "unknown"
    try:
        pos = int(passing.split("-")[0])
    except Exception:
        return "unknown"

    if pos <= 3:
        return "front"
    elif pos <= 7:
        return "stalker"
    else:
        return "closer"


# -------------------------------------------------
# Scrape single race
# -------------------------------------------------

def fetch_single_race(driver, race_id: str) -> Optional[Dict[str, Any]]:

    url = f"https://db.netkeiba.com/race/{race_id}/"

    try:
        driver.get(url)
        random_sleep()
    except Exception:
        return None

    rows = driver.find_elements(By.CSS_SELECTOR, ".race_table_01 tbody tr")

    if not rows:
        return None

    race_title = ""
    try:
        race_title = safe_text(driver.find_element(By.CSS_SELECTOR, "title"))
    except Exception:
        race_title = ""

    for row in rows:

        cols = row.find_elements(By.TAG_NAME, "td")

        if len(cols) < 14:
            continue

        rank = parse_int(safe_text(cols[0]))

        if rank != 1:
            continue

        gate = parse_int(safe_text(cols[1]))
        horse = safe_text(cols[3])
        sex_age = safe_text(cols[4])
        passing = safe_text(cols[10])
        odds = parse_float(safe_text(cols[12]))
        popularity = parse_int(safe_text(cols[13]))

        age_match = re.search(r"(\d+)", sex_age)
        age = int(age_match.group(1)) if age_match else None

        style = infer_style(passing)

        return {
            "race_id": race_id,
            "race_title": race_title,
            "horse_name": horse,
            "gate": gate,
            "age": age,
            "popularity": popularity,
            "odds": odds,
            "running_style": style,
            "prep_race": "",
            "rank": 1,
        }

    return None


# -------------------------------------------------
# Fetch past races
# -------------------------------------------------

def fetch_race_history(driver, race_id: str, years: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch past race winners for the same race ID across previous years.
    Example race_id: 202405030811
    """

    try:
        current_year = int(str(race_id)[:4])
    except Exception:
        return []

    history: List[Dict[str, Any]] = []

    for year in range(current_year, current_year - years, -1):

        try:
            new_race_id = f"{year}{str(race_id)[4:]}"
        except Exception:
            continue

        data = fetch_single_race(driver, new_race_id)

        if data:
            history.append(data)

        # polite delay to avoid blocking
        random_sleep(0.8, 1.6)

    return history


# -------------------------------------------------
# Trend analysis
# -------------------------------------------------

def analyze_race_trend(history: List[Dict[str, Any]]) -> Dict[str, Any]:

    gate: Dict[str, int] = {}
    age: Dict[str, int] = {}
    style: Dict[str, int] = {}
    popularity: Dict[str, int] = {}
    prep_race: Dict[str, int] = {}

    for h in history:

        gate_key = bucket_gate(h.get("gate")) or "不明"
        gate[gate_key] = gate.get(gate_key, 0) + 1

        a = h.get("age")
        age_key = f"{a}歳" if a else "不明"
        age[age_key] = age.get(age_key, 0) + 1

        s = normalize_style_label(h.get("running_style", "unknown"))
        style[s] = style.get(s, 0) + 1

        p = bucket_popularity(h.get("popularity"))
        popularity[p] = popularity.get(p, 0) + 1

        prep = h.get("prep_race") or "不明"
        prep_race[prep] = prep_race.get(prep, 0) + 1

    return {
        "gate": gate,
        "age": age,
        "style": style,
        "popularity": popularity,
        "prep_race": prep_race,
    }


# -------------------------------------------------
# Build simple summary
# -------------------------------------------------

def build_race_summary(history: List[Dict[str, Any]]) -> str:

    if not history:
        return "過去10年傾向データなし"

    trend = analyze_race_trend(history)

    gate_text = summarize_counts(trend.get("gate", {}), top_n=1)
    style_text = summarize_counts(trend.get("style", {}), top_n=2)
    pop_text = summarize_counts(trend.get("popularity", {}), top_n=2)
    age_text = summarize_counts(trend.get("age", {}), top_n=2)

    return (
        f"枠順傾向: {gate_text} / "
        f"脚質傾向: {style_text} / "
        f"人気傾向: {pop_text} / "
        f"年齢傾向: {age_text}"
    )

# -------------------------------------------------
# Backward compatibility function
# -------------------------------------------------


def build_race_trend_summary(history):
    """
    Compatibility wrapper used by other modules.
    Returns the same output as build_race_summary.
    """
    return build_race_summary(history)


# -------------------------------------------------
# Winner condition AI (simple rule extraction)
# -------------------------------------------------

def build_winner_condition_ai(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract simple winning patterns from past race winners.
    Returns a dict containing conditions and counts.
    """

    if not history:
        return {
            "conditions": [],
            "counts": {}
        }

    trend = analyze_race_trend(history)
    conditions: List[str] = []

    gate = trend.get("gate", {})
    style = trend.get("style", {})
    popularity = trend.get("popularity", {})
    age = trend.get("age", {})

    if gate.get("内枠(1〜3)", 0) >= max(3, len(history) // 3):
        conditions.append("内枠有利")
    elif gate.get("外枠(7〜)", 0) >= max(3, len(history) // 3):
        conditions.append("外枠優勢")

    if (style.get("逃げ", 0) + style.get("先行", 0)) >= max(4, len(history) // 2):
        conditions.append("逃げ先行有利")
    elif style.get("差し", 0) >= max(4, len(history) // 2):
        conditions.append("差し有利")

    if popularity.get("1番人気", 0) >= max(3, len(history) // 3):
        conditions.append("本命寄り")
    elif (popularity.get("4〜6番人気", 0) + popularity.get("7〜9番人気", 0)) >= max(4, len(history) // 2):
        conditions.append("中穴傾向")

    top_age = max(age.items(), key=lambda x: x[1])[0] if age else None
    if top_age and top_age != "不明":
        conditions.append(f"{top_age}中心")

    return {
        "conditions": conditions,
        "counts": {
            "gate": gate,
            "style": style,
            "popularity": popularity,
            "age": age,
        }
    }


# -------------------------------------------------
# ChatGPT analysis
# -------------------------------------------------

def analyze_with_chatgpt(history, race_name):

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return {
            "trend_comment": "APIキー未設定",
            "winner_conditions": []
        }

    client = OpenAI(api_key=api_key)

    prompt = {
        "race": race_name,
        "history": history,
        "instruction": (
            "過去10年の同レース勝ち馬データを分析し、"
            "脚質・枠順・人気・年齢の傾向を3行程度で要約し、"
            "勝ち馬条件を箇条書きで抽出してください。"
        )
    }

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたは中央競馬の重賞傾向分析AIです。簡潔かつロジカルに答えてください。"},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}
            ]
        )

        content = response.choices[0].message.content

        return {
            "trend_comment": content,
            "winner_conditions": []
        }

    except Exception as e:
        return {
            "trend_comment": f"ChatGPT分析は現在利用できません ({type(e).__name__})",
            "winner_conditions": []
        }
    


def extract_race_id_from_url(url: str) -> str:
    m = re.search(r"race_id=(\d+)", url)
    return m.group(1) if m else ""

def build_past_race_ids(current_race_id: str, years: int = 10) -> List[str]:
    if len(current_race_id) < 12:
        return []

    year = int(current_race_id[:4])
    race_code = current_race_id[4:]

    ids: List[str] = []
    for i in range(years):
        y = year - i
        ids.append(f"{y}{race_code}")

    return ids

def fetch_past_10y_results(driver: Any, current_race_id: str) -> List[Dict[str, Any]]:
    past_ids = build_past_race_ids(current_race_id, years=10)
    results: List[Dict[str, Any]] = []

    for rid in past_ids:
        url = f"https://race.netkeiba.com/race/result.html?race_id={rid}"

        try:
            driver.get(url)

            rows = driver.find_elements("css selector", "table.RaceTable01 tbody tr")
            for row in rows[:3]:
                cols = row.find_elements("tag name", "td")
                if len(cols) < 13:
                    continue

                rank_text = cols[0].text.strip()
                horse_name = cols[3].text.strip()
                odds_text = cols[12].text.strip()

                rank = int(rank_text) if rank_text.isdigit() else None
                try:
                    odds = float(odds_text.replace(",", ""))
                except Exception:
                    odds = None

                results.append({
                    "race_id": rid,
                    "year": rid[:4],
                    "rank": rank,
                    "horse_name": horse_name,
                    "odds": odds,
                })

        except Exception:
            continue

    return results

def analyze_10y_race_trend(past_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    winners = [r for r in past_results if r.get("rank") == 1]

    if not winners:
        return {
            "sample_size": 0,
            "avg_win_odds": None,
            "favorite_ratio": 0.0,
            "mid_ratio": 0.0,
            "long_ratio": 0.0,
            "style": {},
            "popularity": {},
            "gate": {},
            "age": {},
            "style_summary": "",
            "popularity_summary": "",
        }

    odds_list = [r["odds"] for r in winners if isinstance(r.get("odds"), (int, float))]

    if odds_list:
        avg_win_odds = round(mean(odds_list), 2)
        favorite_ratio = round(sum(1 for x in odds_list if x <= 3.0) / len(odds_list), 3)
        mid_ratio = round(sum(1 for x in odds_list if 3.0 < x <= 10.0) / len(odds_list), 3)
        long_ratio = round(sum(1 for x in odds_list if x > 10.0) / len(odds_list), 3)
    else:
        avg_win_odds = None
        favorite_ratio = 0.0
        mid_ratio = 0.0
        long_ratio = 0.0

    trend = analyze_race_trend(winners)
    style_counts = trend.get("style", {})
    popularity_counts = trend.get("popularity", {})

    style_summary = summarize_counts(style_counts, top_n=2)
    popularity_summary = summarize_counts(popularity_counts, top_n=2)

    return {
        "sample_size": len(winners),
        "avg_win_odds": avg_win_odds,
        "favorite_ratio": favorite_ratio,
        "mid_ratio": mid_ratio,
        "long_ratio": long_ratio,
        "style": style_counts,
        "popularity": popularity_counts,
        "gate": trend.get("gate", {}),
        "age": trend.get("age", {}),
        "style_summary": style_summary,
        "popularity_summary": popularity_summary,
    }

def match_current_runners_with_10y_trend(
    features: List[Dict[str, Any]],
    trend: Dict[str, Any]
) -> List[str]:
    scored: List[Tuple[str, int]] = []

    avg_win_odds = trend.get("avg_win_odds")
    style_summary = str(trend.get("style_summary", ""))
    popularity_summary = str(trend.get("popularity_summary", ""))

    for f in features:
        score = 0
        horse_name = str(f.get("horse_name", ""))
        odds = f.get("win_odds")
        style = f.get("running_style")
        pop = f.get("popularity")

        if isinstance(avg_win_odds, (int, float)) and isinstance(odds, (int, float)):
            if abs(odds - avg_win_odds) <= max(2.0, avg_win_odds * 0.5):
                score += 2

        if "逃げ" in style_summary and style == "front":
            score += 2
        elif "先行" in style_summary and style == "stalker":
            score += 2
        elif "差し" in style_summary and style == "closer":
            score += 2

        if "1番人気" in popularity_summary and pop == 1:
            score += 1
        elif "4〜6番人気" in popularity_summary and isinstance(pop, int) and 4 <= pop <= 6:
            score += 1
        elif "7〜9番人気" in popularity_summary and isinstance(pop, int) and 7 <= pop <= 9:
            score += 1

        if score >= 2:
            scored.append((horse_name, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in scored[:5]]


# -------------------------------------------------
# Enriched single race (all runners)
# -------------------------------------------------

def fetch_single_race_enriched(driver: Any, race_id: str) -> Optional[Dict[str, Any]]:
    """
    指定レースの全走者を返す（勝ち馬のみではなく全馬）。

    Returns
    -------
    {
        "race_id":   str,
        "n_runners": int,
        "runners":   List[Dict]  # rank, gate, horse_name, age, popularity, odds, running_style
    }
    None なら取得失敗
    """
    url = f"https://db.netkeiba.com/race/{race_id}/"
    try:
        driver.get(url)
        random_sleep()
    except Exception:
        return None

    rows = driver.find_elements(By.CSS_SELECTOR, ".race_table_01 tbody tr")
    if not rows:
        return None

    runners = []
    for row in rows:
        cols = row.find_elements(By.TAG_NAME, "td")
        if len(cols) < 14:
            continue

        rank = parse_int(safe_text(cols[0]))
        if rank is None:
            continue

        gate       = parse_int(safe_text(cols[1]))
        horse      = safe_text(cols[3])
        sex_age    = safe_text(cols[4])
        passing    = safe_text(cols[10])
        odds       = parse_float(safe_text(cols[12]))
        popularity = parse_int(safe_text(cols[13]))

        age_match = re.search(r"(\d+)", sex_age)
        age = int(age_match.group(1)) if age_match else None
        style = infer_style(passing)

        runners.append({
            "rank":          rank,
            "gate":          gate,
            "horse_name":    horse,
            "age":           age,
            "popularity":    popularity,
            "odds":          odds,
            "running_style": style,
        })

    if not runners:
        return None

    return {
        "race_id":   race_id,
        "n_runners": len(runners),
        "runners":   runners,
    }


def fetch_race_history_enriched(
    driver: Any,
    race_id: str,
    years: int = 10,
) -> List[Dict[str, Any]]:
    """
    過去N年の同レースを全走者付きで返す。
    """
    try:
        current_year = int(str(race_id)[:4])
    except Exception:
        return []

    history: List[Dict[str, Any]] = []
    for year in range(current_year, current_year - years, -1):
        try:
            new_race_id = f"{year}{str(race_id)[4:]}"
        except Exception:
            continue

        data = fetch_single_race_enriched(driver, new_race_id)
        if data:
            history.append(data)
        random_sleep(0.8, 1.6)

    return history


    return result