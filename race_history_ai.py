from __future__ import annotations

from selenium.webdriver.common.by import By
from typing import List, Dict, Any, Optional, Tuple
from statistics import mean
from openai import OpenAI
from openai_cost_guard import safe_chat_create
import hashlib
import json
import os
import re
import time
import random
from pathlib import Path
from trend_stats import bucket_gate

# -------------------------------------------------
# 簡易キャッシュ（race_history_ai 専用）
# fetch_single_race_enriched の結果をキャッシュして
# 毎回10ページ再取得しなくて済むようにする
# -------------------------------------------------
_ENRICHED_CACHE_DIR = Path(".keiba_cache")
_ENRICHED_CACHE_TTL = 60 * 60 * 24 * 7  # 7日（レース結果は変わらないので長め）
_ENRICHED_CACHE_VERSION = "v4"           # track_condition（馬場状態）を追加した際に更新


def _enriched_cache_path(race_id: str) -> Path:
    key = hashlib.md5(f"enriched_{_ENRICHED_CACHE_VERSION}_{race_id}".encode()).hexdigest()
    return _ENRICHED_CACHE_DIR / f"enriched_{key}.json"


def _load_enriched_cache(race_id: str) -> Optional[Dict[str, Any]]:
    try:
        path = _enriched_cache_path(race_id)
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > _ENRICHED_CACHE_TTL:
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_enriched_cache(race_id: str, data: Dict[str, Any]) -> None:
    try:
        _ENRICHED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_enriched_cache_path(race_id), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        pass

# -------------------------------------------------
# Race name matching — 別レース混入防止
# -------------------------------------------------

def _extract_core_race_name(title: str) -> str:
    """
    ページタイトル文字列からレース名のコア部分を抽出する。

    目的: 年号・回次・グレード括弧・区切り文字以降のノイズを除去し、
          「スプリングステークス」のようなレース名だけを残す。

    例:
        "スプリングステークス｜2024年中山11Rの競馬予想..." → "スプリングステークス"
        "スプリングステークス(G2) | netkeiba.com"         → "スプリングステークス"
        "2024年 第68回スプリングステークス"               → "スプリングステークス"
    """
    if not title:
        return ""
    # 区切り文字で最初のトークンだけを取る（レース名が先頭にある前提）
    for sep in ("｜", " | ", " - ", "　"):
        if sep in title:
            title = title.split(sep)[0].strip()
            break
    title = re.sub(r'\d{4}年?', '', title)              # 年号除去
    title = re.sub(r'第\d+回', '', title)                # 第XX回除去
    title = re.sub(r'[\(（][^)\)）]*[\)）]', '', title)  # グレード括弧除去
    title = title.replace('　', '').strip()
    return title


def race_names_match(scraped_title: str, target_name: str) -> bool:
    """
    スクレイプ済みページタイトルが対象レース名と同一レースを指すか判定する。

    目的: スプリングSのURLで過去年を辿った際、アネモネSなど別レースが
          同じ race_id スロットに割り当てられているケースを検出・除外する。

    判定基準:
        1. 正規化後にどちらかがもう一方を含む → 同一（略称 vs フル表記に対応）
        2. 先頭5文字が一致 → 同一（表記揺れに対応）
        3. 判定不能（空文字・取得失敗）→ True を返す（安全側で通す）

    例:
        "スプリングステークス" vs "アネモネステークス" → False  (先頭5文字: スプリン vs アネモネ)
        "スプリングS"          vs "スプリングステークス" → True  (前者が後者の prefix 含む)
    """
    if not scraped_title or not target_name:
        return True  # 判定不能 → 通す
    n1 = _extract_core_race_name(scraped_title)
    n2 = _extract_core_race_name(target_name)
    if not n1 or not n2:
        return True  # 正規化後も空 → 通す
    if n1 in n2 or n2 in n1:
        return True
    # 先頭5文字が一致（表記揺れ許容）
    if len(n1) >= 5 and len(n2) >= 5 and n1[:5] == n2[:5]:
        return True
    return False


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

def fetch_race_history(
    driver,
    race_id: str,
    years: int = 10,
    expected_race_name: str = "",
) -> List[Dict[str, Any]]:
    """
    Fetch past race winners for the same race ID across previous years.
    Example race_id: 202405030811

    expected_race_name: 分析対象レースの正式名称。指定すると別レースが
                        同一 race_id スロットに入っているページを自動スキップする。
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
            # レース名照合: 別レースが同スロットに入っているページを除外
            if expected_race_name and not race_names_match(
                data.get("race_title", ""), expected_race_name
            ):
                continue
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
        response = safe_chat_create(
            client,
            model="gpt-5.4-mini",
            messages=[
                {"role": "system", "content": "あなたは中央競馬の重賞傾向分析AIです。簡潔かつロジカルに答えてください。"},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
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

def fetch_past_10y_results(
    driver: Any,
    current_race_id: str,
    expected_race_name: str = "",
) -> List[Dict[str, Any]]:
    past_ids = build_past_race_ids(current_race_id, years=10)
    results: List[Dict[str, Any]] = []

    for rid in past_ids:
        url = f"https://race.netkeiba.com/race/result.html?race_id={rid}"

        try:
            driver.get(url)

            # レース名照合: 別レースが同スロットに入っているページを除外
            if expected_race_name:
                try:
                    page_title = driver.title or ""
                except Exception:
                    page_title = ""
                if page_title and not race_names_match(page_title, expected_race_name):
                    continue  # 例: スプリングSのはずがアネモネSだった場合

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


def fetch_past_10y_results_requests(
    current_race_id: str,
    expected_race_name: str = "",
) -> List[Dict[str, Any]]:
    """
    NetkeibaSession (curl_cffi Chrome偽装) 版の fetch_past_10y_results。
    Selenium なしで result.html から過去10年の上位3着を取得する。
    """
    from bs4 import BeautifulSoup as _BS
    from netkeiba_session import NetkeibaSession
    _session = NetkeibaSession()

    past_ids = build_past_race_ids(current_race_id, years=10)
    results: List[Dict[str, Any]] = []

    for rid in past_ids:
        url = f"https://race.netkeiba.com/race/result.html?race_id={rid}"
        try:
            html = _session.fetch_html(url)
            if not html:
                continue

            soup = _BS(html, "html.parser")

            # レース名照合
            if expected_race_name:
                title_tag = soup.select_one("title")
                page_title = title_tag.get_text(strip=True) if title_tag else ""
                if page_title and not race_names_match(page_title, expected_race_name):
                    continue

            rows = soup.select("table.RaceTable01 tbody tr")
            for row in rows[:3]:
                cols = row.find_all("td")
                if len(cols) < 13:
                    continue
                rank_text = cols[0].get_text(strip=True)
                horse_name = cols[3].get_text(strip=True)
                odds_text = cols[12].get_text(strip=True)
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
        "race_title": str,  # ページタイトル（レース混入チェックに使用）
        "n_runners": int,
        "runners":   List[Dict]  # rank, gate, horse_name, age, popularity, odds, running_style
    }
    None なら取得失敗
    """
    # キャッシュヒット確認（過去レース結果は変わらないので長期キャッシュOK）
    _cached = _load_enriched_cache(race_id)
    if _cached:
        return _cached

    url = f"https://db.netkeiba.com/race/{race_id}/"
    try:
        driver.get(url)
        random_sleep()
    except Exception:
        return None

    # ページタイトルを取得（レース名照合用）
    try:
        page_race_title = driver.title or ""
    except Exception:
        page_race_title = ""

    # 馬場状態を取得（レース情報ヘッダから）
    track_condition: Optional[str] = None
    try:
        race_data_text = ""
        for sel in (".RaceData01", ".race_data", "p.RaceData01"):
            elems = driver.find_elements(By.CSS_SELECTOR, sel)
            if elems:
                race_data_text = elems[0].text
                break
        m = re.search(r"[／/]\s*(良|稍重|重|不良)", race_data_text)
        if m:
            track_condition = m.group(1)
    except Exception:
        track_condition = None

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

        gate           = parse_int(safe_text(cols[1]))
        horse          = safe_text(cols[3])
        sex_age        = safe_text(cols[4])
        jockey_weight  = parse_float(safe_text(cols[5]))   # 斤量
        jockey         = safe_text(cols[6])                # 騎手名
        passing        = safe_text(cols[10])
        last_3f        = parse_float(safe_text(cols[11]))  # 上がり3F
        odds           = parse_float(safe_text(cols[12]))
        popularity     = parse_int(safe_text(cols[13]))

        age_match = re.search(r"(\d+)", sex_age)
        age = int(age_match.group(1)) if age_match else None
        style = infer_style(passing)

        runners.append({
            "rank":            rank,
            "gate":            gate,
            "horse_name":      horse,
            "age":             age,
            "popularity":      popularity,
            "odds":            odds,
            "running_style":   style,
            "last_3f":         last_3f,          # 上がり3F (秒)
            "jockey_weight":   jockey_weight,     # 斤量 (kg)
            "jockey":          jockey,            # 騎手名
            "track_condition": track_condition,   # 馬場状態
        })

    if not runners:
        return None

    result = {
        "race_id":    race_id,
        "race_title": page_race_title,   # レース名照合用
        "n_runners":  len(runners),
        "runners":    runners,
    }
    _save_enriched_cache(race_id, result)
    return result


def fetch_single_race_enriched_noselenium(race_id: str) -> Optional[Dict[str, Any]]:
    """
    Selenium不要版の fetch_single_race_enriched。
    dividend_scraper.scrape_race_result() を使ってrequestsで取得する。
    キャッシュも共有する（fetch_single_race_enriched と同じキーを使用）。
    """
    # キャッシュヒット確認
    _cached = _load_enriched_cache(race_id)
    if _cached:
        return _cached

    from dividend_scraper import scrape_race_result

    data = scrape_race_result(race_id)
    if not data or not data.get("runners"):
        return None

    # fetch_single_race_enriched 互換フォーマットに変換
    runners = []
    for r in data["runners"]:
        runners.append({
            "rank":            r.get("rank"),
            "gate":            r.get("gate"),
            "horse_name":      r.get("horse_name", ""),
            "age":             r.get("age"),
            "popularity":      r.get("popularity"),
            "odds":            r.get("win_odds"),       # win_odds → odds
            "running_style":   r.get("running_style", "unknown"),
            "last_3f":         r.get("last_3f"),
            "jockey_weight":   r.get("jockey_weight"),
            "jockey":          r.get("jockey", ""),
            "track_condition": data.get("track_condition"),
        })

    result = {
        "race_id":    race_id,
        "race_title": data.get("race_name", ""),
        "n_runners":  len(runners),
        "runners":    runners,
    }
    _save_enriched_cache(race_id, result)
    return result


def fetch_race_history_enriched(
    driver: Any,
    race_id: str,
    years: int = 10,
    expected_race_name: str = "",
) -> List[Dict[str, Any]]:
    """
    過去N年の同レースを全走者付きで返す。

    driver=None の場合は Selenium を使わず requests ベースで取得する。
    expected_race_name: 分析対象レースの正式名称。指定すると別レースが
                        同一 race_id スロットに入っているページを自動スキップする。
    """
    try:
        current_year = int(str(race_id)[:4])
    except Exception:
        return []

    use_selenium = driver is not None

    history: List[Dict[str, Any]] = []
    for year in range(current_year, current_year - years, -1):
        try:
            new_race_id = f"{year}{str(race_id)[4:]}"
        except Exception:
            continue

        if use_selenium:
            data = fetch_single_race_enriched(driver, new_race_id)
        else:
            data = fetch_single_race_enriched_noselenium(new_race_id)

        if data:
            # レース名照合: 別レースが同スロットに入っているページを除外
            if expected_race_name and not race_names_match(
                data.get("race_title", ""), expected_race_name
            ):
                continue
            history.append(data)
        random_sleep(0.8, 1.6)

    return history