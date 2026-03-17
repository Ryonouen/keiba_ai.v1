from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from openai import OpenAI
from bet_generator import generate_ai_bets
from course_bias import get_course_bias
import json
import hashlib
import math
import os
import random
import re
import time

from race_history_ai import (
    fetch_race_history as history_fetch_race_history,
    analyze_race_trend as history_analyze_race_trend,
    build_race_trend_summary as history_build_race_trend_summary,
    build_winner_condition_ai as history_build_winner_condition_ai,
    extract_race_id_from_url,
    fetch_past_10y_results,
    analyze_10y_race_trend,
    match_current_runners_with_10y_trend,
)

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException, NoSuchWindowException, WebDriverException, TimeoutException

try:
    import pandas as pd  # type: ignore
    import lightgbm as lgb  # type: ignore
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

try:
    import undetected_chromedriver as uc  # type: ignore
    UC_AVAILABLE = True
except Exception:
    UC_AVAILABLE = False


# =========================================================
# Configuration
# =========================================================

HISTORY_LIMIT_DEFAULT: int = 5
MODEL_FILE: str = "keiba_lgbm_model.txt"
TRAINING_CSV: str = "keiba_training_data.csv"
COOKIE_FILE: str = "netkeiba_cookies.json"
USE_UNDETECTED_CHROMEDRIVER: bool = False
CACHE_DIR: str = ".keiba_cache"
CACHE_TTL_SECONDS: int = 60 * 60 * 6
def random_sleep(min_sec: float = 0.8, max_sec: float = 1.8) -> None:
    time.sleep(random.uniform(min_sec, max_sec))


def save_cookies(driver: webdriver.Chrome, filepath: str = COOKIE_FILE) -> None:
    try:
        cookies = driver.get_cookies() if driver is not None else []
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(cookies, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_cookies(driver: webdriver.Chrome, filepath: str = COOKIE_FILE) -> bool:
    if not Path(filepath).exists():
        return False

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            cookies = json.load(f)

        for cookie in cookies:
            c = dict(cookie)
            c.pop("sameSite", None)
            c.pop("expiry", None) if c.get("expiry") is None else None
            try:
                driver.add_cookie(c)
            except Exception:
                continue
        return True
    except Exception:
        return False


def emulate_human_behavior(driver: webdriver.Chrome) -> None:
    if driver is None:
        return
    try:
        random_sleep(0.6, 1.2)
        scroll_y = random.randint(200, 900)
        driver.execute_script(f"window.scrollTo({{top: {scroll_y}, behavior: 'smooth'}});")
        random_sleep(0.8, 1.6)
        driver.execute_script("window.scrollTo({top: 0, behavior: 'smooth'});")
        random_sleep(0.6, 1.2)

        try:
            body = driver.find_element(By.TAG_NAME, "body")
            ActionChains(driver).move_to_element_with_offset(body, random.randint(50, 300), random.randint(50, 250)).perform()
            random_sleep(0.2, 0.6)
        except Exception:
            pass
    except Exception:
        pass


def build_webdriver(headless: bool = True) -> webdriver.Chrome:
    user_agent = os.getenv(
        "NETKEIBA_USER_AGENT",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    )

    chrome_bin = os.getenv("CHROME_BIN", "/usr/bin/chromium")
    chromedriver_path = os.getenv("CHROMEDRIVER_PATH", "/usr/bin/chromedriver")

    # Render / Docker 環境では GUI がないため headless を強制
    if os.getenv("RENDER") or os.getenv("RENDER_SERVICE_ID") or not os.getenv("DISPLAY"):
        headless = True

    options = Options()
    options.binary_location = chrome_bin

    options.add_argument("--window-size=1400,900")
    options.add_argument(f"--user-agent={user_agent}")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-setuid-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-background-networking")
    options.add_argument("--disable-background-timer-throttling")
    options.add_argument("--disable-backgrounding-occluded-windows")
    options.add_argument("--disable-breakpad")
    options.add_argument("--disable-component-update")
    options.add_argument("--disable-default-apps")
    options.add_argument("--disable-ipc-flooding-protection")
    options.add_argument("--disable-renderer-backgrounding")
    options.add_argument("--metrics-recording-only")
    options.add_argument("--mute-audio")
    options.add_argument("--no-first-run")
    options.add_argument("--password-store=basic")
    options.add_argument("--use-mock-keychain")
    options.add_argument("--remote-debugging-port=9222")
    options.add_argument("--lang=ja-JP")
    options.page_load_strategy = "eager"

    if headless:
        options.add_argument("--headless=new")

    service = Service(
        executable_path=chromedriver_path,
        log_output="/tmp/chromedriver.log",
    )

    driver = webdriver.Chrome(service=service, options=options)
    driver.implicitly_wait(5)
    driver.set_page_load_timeout(25)

    try:
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        driver.execute_cdp_cmd(
            "Network.setUserAgentOverride",
            {"userAgent": user_agent}
        )
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"}
        )
    except Exception:
        pass

    return driver

def restart_driver(old_driver: Optional[webdriver.Chrome], headless: bool = True) -> webdriver.Chrome:
    try:
        if old_driver is not None:
            old_driver.quit()
    except Exception:
        pass
    return build_webdriver(headless=headless)


def safe_page_source(driver: Optional[webdriver.Chrome]) -> str:
    if driver is None:
        return ""
    try:
        return driver.page_source or ""
    except Exception:
        return ""


def safe_driver_title(driver: Optional[webdriver.Chrome]) -> str:
    if driver is None:
        return ""
    try:
        return driver.title or ""
    except Exception:
        return ""


def safe_get(driver: webdriver.Chrome, url: str, headless: bool = True, retries: int = 2) -> webdriver.Chrome:
    last_error: Optional[Exception] = None

    for attempt in range(retries + 1):
        try:
            driver.set_page_load_timeout(25)
            driver.get(url)

            try:
                WebDriverWait(driver, 20).until(
                    lambda d: d.execute_script("return document.readyState") in ("interactive", "complete")
                )
            except Exception:
                pass

            return driver

        except TimeoutException as e:
            last_error = e
            print(f"ブラウザ遷移タイムアウト: {url}")

            try:
                driver.execute_script("window.stop();")
                page_src = safe_page_source(driver)
                if page_src and len(page_src) > 1000:
                    return driver
            except Exception:
                pass

            if attempt >= retries:
                raise e

            driver = restart_driver(driver, headless=headless)
            random_sleep(2.0, 3.5)

        except (NoSuchWindowException, WebDriverException) as e:
            last_error = e
            print(f"ブラウザ遷移失敗: {type(e).__name__}")
            if attempt >= retries:
                raise e
            driver = restart_driver(driver, headless=headless)
            random_sleep(2.0, 3.5)

        except Exception as e:
            last_error = e
            print(f"ブラウザ遷移エラー: {type(e).__name__}")
            if attempt >= retries:
                raise e
            random_sleep(2.0, 3.5)

    if last_error is not None:
        raise last_error
    return driver


def warmup_netkeiba_session(driver: webdriver.Chrome, headless: bool = True) -> webdriver.Chrome:
    driver = safe_get(driver, "https://race.netkeiba.com/", headless=headless, retries=1)
    random_sleep(4.5, 7.0)
    emulate_human_behavior(driver)

    if load_cookies(driver, COOKIE_FILE):
        driver = safe_get(driver, "https://race.netkeiba.com/", headless=headless, retries=1)
        random_sleep(4.5, 7.0)
        emulate_human_behavior(driver)

    return driver


# =========================================================
# Caching helpers
# =========================================================

def ensure_cache_dir() -> Path:
    path = Path(CACHE_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_key(*parts: Any) -> str:
    raw = "||".join(str(p) for p in parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def load_json_cache(name: str, ttl_seconds: int = CACHE_TTL_SECONDS) -> Optional[Dict[str, Any]]:
    try:
        path = ensure_cache_dir() / f"{name}.json"
        if not path.exists():
            return None
        if time.time() - path.stat().st_mtime > ttl_seconds:
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def save_json_cache(name: str, data: Dict[str, Any]) -> None:
    try:
        path = ensure_cache_dir() / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def is_blocked_page(page_src: str, title: str = "") -> bool:
    text = page_src or ""
    page_title = title or ""
    if not text:
        return True
    if "HTTP ERROR 400" in text:
        return True
    if "このページは動作していません" in text:
        return True
    if page_title.strip() == "race.netkeiba.com":
        return True
    return False

JOCKEY_BASE_INDEX: Dict[str, float] = {
    "ルメール": 1.15,
    "川田": 1.13,
    "戸崎": 1.09,
    "坂井": 1.08,
    "横山武": 1.08,
    "武豊": 1.06,
    "松山": 1.05,
    "西村淳": 1.04,
    "鮫島駿": 1.03,
    "丹内": 1.03,
    "岩田望": 1.04,
    "横山和": 1.03,
    "田辺": 1.02,
    "団野": 1.02,
    "幸": 1.01,
    "菅原明": 1.03,
    "Mデムーロ": 1.02,
    "Cデムーロ": 1.08,
}

DRAW_BIAS: Dict[Tuple[str, str], Dict[str, float]] = {
    ("中山", "lt_1200"): {"inner": 1.08, "middle": 1.00, "outer": 0.94},
    ("中山", "1200_1600"): {"inner": 1.06, "middle": 1.00, "outer": 0.96},
    ("中山", "1600_2000"): {"inner": 1.05, "middle": 1.00, "outer": 0.97},
    ("東京", "1200_1600"): {"inner": 0.99, "middle": 1.00, "outer": 1.01},
    ("東京", "1600_2000"): {"inner": 0.99, "middle": 1.00, "outer": 1.01},
    ("阪神", "1200_1600"): {"inner": 1.03, "middle": 1.00, "outer": 0.98},
    ("阪神", "1600_2000"): {"inner": 1.02, "middle": 1.00, "outer": 0.99},
    ("京都", "1200_1600"): {"inner": 1.03, "middle": 1.00, "outer": 0.98},
    ("中京", "1200_1600"): {"inner": 1.01, "middle": 1.00, "outer": 0.99},
    ("小倉", "lt_1200"): {"inner": 1.07, "middle": 1.00, "outer": 0.95},
    ("福島", "lt_1200"): {"inner": 1.06, "middle": 1.00, "outer": 0.96},
    ("新潟", "1200_1600"): {"inner": 0.99, "middle": 1.00, "outer": 1.01},
    ("札幌", "1200_1600"): {"inner": 1.04, "middle": 1.00, "outer": 0.97},
    ("函館", "1200_1600"): {"inner": 1.04, "middle": 1.00, "outer": 0.97},
}


# =========================================================
# Dataclasses
# =========================================================

@dataclass
class RaceMeta:
    race_title: str
    race_info_text: str
    target_surface: Optional[str]
    target_distance: Optional[int]
    target_course: str
    target_ground: str
    predicted_pace: str = ""


# =========================================================
# Utilities
# =========================================================

def safe_text(elem: Any) -> str:
    try:
        return elem.text.strip()
    except Exception:
        return ""


def safe_attr(elem: Any, attr_name: str) -> str:
    try:
        return elem.get_attribute(attr_name) or ""
    except Exception:
        return ""


def safe_find_text(parent: Any, by: str, selector: str) -> str:
    try:
        return parent.find_element(by, selector).text.strip()
    except NoSuchElementException:
        return ""
    except Exception:
        return ""


def parse_rank(rank_text: str) -> Optional[int]:
    rank_text = rank_text.strip()
    return int(rank_text) if rank_text.isdigit() else None


def parse_time_to_seconds(time_text: str) -> Optional[float]:
    time_text = time_text.strip()
    if not time_text or ":" not in time_text:
        return None
    try:
        minute_str, sec_str = time_text.split(":")
        return int(minute_str) * 60 + float(sec_str)
    except Exception:
        return None


def parse_float(text: str) -> Optional[float]:
    text = text.strip().replace(",", "")
    if not text or text in {"--", "---", "取消", "除外"}:
        return None
    try:
        return float(text)
    except Exception:
        return None


def parse_distance(distance_text: str) -> Tuple[Optional[str], Optional[int]]:
    distance_text = distance_text.strip()
    m = re.search(r"(芝|ダ|障)\s*(\d+)", distance_text)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def distance_band(distance: Optional[int]) -> str:
    if distance is None:
        return "unknown"
    if distance < 1200:
        return "lt_1200"
    elif 1200 <= distance < 1600:
        return "1200_1600"
    elif 1600 <= distance < 2000:
        return "1600_2000"
    else:
        return "ge_2000"


def parse_course_name(text: str) -> str:
    courses = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]
    for c in courses:
        if c in text:
            return c
    return "unknown"


def parse_course_name_from_title(title: str) -> str:
    return parse_course_name(title)


def parse_first_corner_position(passing_text: str) -> Optional[int]:
    passing_text = passing_text.strip()
    if not passing_text:
        return None
    first = passing_text.split("-")[0]
    return int(first) if first.isdigit() else None


def parse_last3f(text: str) -> Optional[float]:
    return parse_float(text)


def safe_float_mean(values: List[float]) -> Optional[float]:
    return mean(values) if values else None


def safe_int_mean(values: List[int]) -> Optional[float]:
    return mean(values) if values else None


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def parse_odds_range_text(text: str) -> Optional[float]:
    text = text.strip()
    if not text:
        return None
    m = re.findall(r"\d+\.\d+|\d+", text)
    if not m:
        return None
    nums = [float(x) for x in m]
    return round(sum(nums) / len(nums), 3)


def parse_comma_odds_input(raw: str) -> List[float]:
    raw = raw.strip()
    if not raw:
        return []
    raw = raw.replace("，", ",")
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return [float(x) for x in items]


# =========================================================
# Basic stats
# =========================================================

def calc_win_rate(ranks: List[int]) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r == 1) / len(ranks)


def calc_place_rate(ranks: List[int]) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r <= 2) / len(ranks)


def calc_show_rate(ranks: List[int]) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r <= 3) / len(ranks)


def normalize_rank_index(avg_rank: Optional[float]) -> float:
    if avg_rank is None:
        return 0.0
    return max(0.0, 1.0 - ((avg_rank - 1.0) / 18.0))


# =========================================================
# Style / pace AI
# =========================================================

def infer_running_style(passing_positions: List[int]) -> str:
    if not passing_positions:
        return "unknown"
    avg_pos = mean(passing_positions)
    if avg_pos <= 3:
        return "front"
    elif avg_pos <= 7:
        return "stalker"
    else:
        return "closer"


def calc_pace_pressure(entry_styles: List[str]) -> str:
    front = entry_styles.count("front")
    stalker = entry_styles.count("stalker")
    pressure = front + stalker * 0.45
    if pressure >= 5:
        return "very_fast"
    elif pressure >= 3:
        return "fast"
    elif pressure <= 1:
        return "slow"
    return "medium"


def calc_pace_advantage(style: str, pace: str) -> float:

    table = {
        "very_fast": {"closer": 1.15, "stalker": 1.05, "front": 0.85, "unknown": 1.0},
        "fast": {"closer": 1.10, "stalker": 1.04, "front": 0.90, "unknown": 1.0},
        "medium": {"closer": 1.00, "stalker": 1.04, "front": 1.00, "unknown": 1.0},
        "slow": {"closer": 0.90, "stalker": 1.03, "front": 1.10, "unknown": 1.0},
    }
    return table.get(pace, {}).get(style, 1.0)

def style_to_japanese(style: str) -> str:
    mapping = {
        "front": "逃げ",
        "stalker": "先行",
        "closer": "差し",
        "unknown": "不明",
    }
    return mapping.get(style, "不明")


def pace_to_japanese(pace: str) -> str:
    mapping = {
        "very_fast": "かなり速い流れ",
        "fast": "速めの流れ",
        "medium": "平均ペース",
        "slow": "スローペース",
        "unknown": "不明",
    }
    return mapping.get(pace, "不明")


def build_pace_balance(features: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"逃げ": 0, "先行": 0, "差し": 0, "不明": 0}
    for f in features:
        label = style_to_japanese(f.get("running_style", "unknown"))
        counts[label] = counts.get(label, 0) + 1
    return counts


def calc_ai_power_index(feature: Dict[str, Any]) -> float:
    model_score = float(feature.get("model_score") or 0.0)
    mc_prob = float(feature.get("montecarlo_win_prob") or 0.0)
    pace_index = float(feature.get("pace_simulation_index") or 0.0)

    ai_power = model_score * 100 + mc_prob * 50 + pace_index * 30

    # 最低値補正（完全0を防ぐ）
    if ai_power == 0:
        ai_power = model_score * 100

    return round(ai_power, 2)


def build_radar_payload(feature: Dict[str, Any]) -> Dict[str, Any]:
    labels = ["近走", "上昇度", "安定感", "距離適性", "展開", "モンテカルロ"]
    values = [
        round(feature.get("recent_form_index", 0.0), 4),
        round(feature.get("trend_index", 0.0), 4),
        round(feature.get("consistency_index", 0.0), 4),
        round(feature.get("distance_fit_index", 0.0), 4),
        round(feature.get("pace_simulation_index", 0.0), 4),
        round(feature.get("montecarlo_win_prob", 0.0), 4),
    ]
    return {"labels": labels, "values": values}

def calc_style_index(records: List[Dict[str, Any]], predicted_pace: str) -> Dict[str, Any]:
    passing_positions = [r["first_corner_pos"] for r in records if r["first_corner_pos"] is not None]
    style = infer_running_style(passing_positions)

    style_bias_map: Dict[str, Dict[str, float]] = {
        "slow": {"front": 1.00, "stalker": 0.80, "closer": 0.55, "unknown": 0.50},
        "medium": {"front": 0.85, "stalker": 0.90, "closer": 0.80, "unknown": 0.50},
        "fast": {"front": 0.55, "stalker": 0.85, "closer": 1.00, "unknown": 0.50},
        "very_fast": {"front": 0.45, "stalker": 0.80, "closer": 1.05, "unknown": 0.50},
    }

    style_bias = style_bias_map.get(predicted_pace, {}).get(style, 0.5)
    ranks = [r["rank"] for r in records if r["rank"] is not None]
    avg_rank = safe_int_mean(ranks)
    style_perf = normalize_rank_index(avg_rank) if avg_rank is not None else 0.0
    style_index = 0.6 * style_bias + 0.4 * style_perf

    return {
        "style": style,
        "sample_size": len(passing_positions),
        "pace_bias_index": round(style_bias, 4),
        "style_perf_index": round(style_perf, 4),
        "style_index": round(style_index, 4),
    }


# =========================================================
# Suitability AI
# =========================================================

def calc_distance_course_index(
    records: List[Dict[str, Any]],
    target_surface: Optional[str],
    target_distance: Optional[int],
    target_course: str,
) -> Dict[str, Any]:
    if target_distance is None or target_surface is None:
        return {
            "sample_size": 0,
            "win_rate": 0.0,
            "place_rate": 0.0,
            "show_rate": 0.0,
            "avg_rank": None,
            "avg_time_sec": None,
            "index": 0.0,
        }

    target_band = distance_band(target_distance)

    filtered = [
        r
        for r in records
        if r["surface"] == target_surface
        and r["course_name"] == target_course
        and distance_band(r["distance"]) == target_band
        and r["rank"] is not None
    ]

    ranks = [r["rank"] for r in filtered if r["rank"] is not None]
    times = [r["time_sec"] for r in filtered if r["time_sec"] is not None]

    win_rate = calc_win_rate(ranks)
    place_rate = calc_place_rate(ranks)
    show_rate = calc_show_rate(ranks)
    avg_rank = safe_int_mean(ranks)
    avg_time_sec = safe_float_mean(times)

    index_value = (
        0.40 * win_rate
        + 0.30 * place_rate
        + 0.20 * show_rate
        + 0.10 * normalize_rank_index(avg_rank)
    )

    return {
        "sample_size": len(filtered),
        "win_rate": round(win_rate, 4),
        "place_rate": round(place_rate, 4),
        "show_rate": round(show_rate, 4),
        "avg_rank": round(avg_rank, 3) if avg_rank is not None else None,
        "avg_time_sec": round(avg_time_sec, 3) if avg_time_sec is not None else None,
        "index": round(index_value, 4),
    }


def calc_distance_band_stats(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r["rank"] is None:
            continue
        grouped[distance_band(r["distance"])].append(r)

    result: Dict[str, Dict[str, Any]] = {}
    for band, rows in grouped.items():
        ranks = [r["rank"] for r in rows if r["rank"] is not None]
        times = [r["time_sec"] for r in rows if r["time_sec"] is not None]
        result[band] = {
            "sample_size": len(rows),
            "win_rate": round(calc_win_rate(ranks), 4),
            "place_rate": round(calc_place_rate(ranks), 4),
            "show_rate": round(calc_show_rate(ranks), 4),
            "avg_rank": round(safe_int_mean(ranks), 3) if ranks else None,
            "avg_time_sec": round(safe_float_mean(times), 3) if times else None,
        }
    return result


def calc_course_stats(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r["rank"] is None:
            continue
        grouped[r["course_name"]].append(r)

    result: Dict[str, Dict[str, Any]] = {}
    for course_name, rows in grouped.items():
        ranks = [r["rank"] for r in rows if r["rank"] is not None]
        times = [r["time_sec"] for r in rows if r["time_sec"] is not None]
        result[course_name] = {
            "sample_size": len(rows),
            "win_rate": round(calc_win_rate(ranks), 4),
            "place_rate": round(calc_place_rate(ranks), 4),
            "show_rate": round(calc_show_rate(ranks), 4),
            "avg_rank": round(safe_int_mean(ranks), 3) if ranks else None,
            "avg_time_sec": round(safe_float_mean(times), 3) if times else None,
        }
    return result


# =========================================================
# Recent form / class / ground
# =========================================================

def parse_race_class_index(race_name: str) -> float:
    text = race_name.strip()
    if "G1" in text:
        return 1.00
    if "G2" in text:
        return 0.90
    if "G3" in text:
        return 0.80
    if "L" in text or "リステッド" in text:
        return 0.70
    if "オープン" in text or "OP" in text:
        return 0.65
    if "3勝" in text:
        return 0.55
    if "2勝" in text:
        return 0.45
    if "1勝" in text:
        return 0.35
    if "新馬" in text or "未勝利" in text:
        return 0.25
    return 0.30


def calc_recent_form_index(records: List[Dict[str, Any]], top_n: int = 5) -> float:
    recent = [r for r in records[:top_n] if r["rank"] is not None]
    if not recent:
        return 0.0

    rank_component = normalize_rank_index(safe_int_mean([r["rank"] for r in recent]))
    weighted_class_sum = 0.0
    weight_sum = 0.0

    for i, r in enumerate(recent):
        w = top_n - i
        weighted_class_sum += r["class_index"] * w
        weight_sum += w

    class_component = weighted_class_sum / weight_sum if weight_sum else 0.0
    return round(0.65 * rank_component + 0.35 * class_component, 4)


def calc_ground_match_index(records: List[Dict[str, Any]], target_ground: str) -> float:
    if not target_ground:
        return 0.0
    filtered = [r for r in records if r["ground"] == target_ground and r["rank"] is not None]
    if not filtered:
        return 0.0
    avg_rank = safe_int_mean([r["rank"] for r in filtered])
    return round(normalize_rank_index(avg_rank), 4)


# =========================================================
# Enhanced indices
# =========================================================

def calc_trend_index(records: List[Dict[str, Any]]) -> float:
    ranks = [r["rank"] for r in records if r["rank"] is not None]
    if len(ranks) < 3:
        return 0.5

    recent_part = ranks[:2]
    old_part = ranks[-2:] if len(ranks) >= 4 else ranks[2:]
    recent_avg = safe_int_mean(recent_part)
    old_avg = safe_int_mean(old_part)

    if recent_avg is None or old_avg is None:
        return 0.5

    improvement = old_avg - recent_avg
    return round(float(sigmoid(improvement / 2.0)), 4)


def calc_consistency_index(records: List[Dict[str, Any]]) -> float:
    ranks = [r["rank"] for r in records if r["rank"] is not None]
    if len(ranks) <= 1:
        return 0.5
    std = pstdev(ranks)
    score = 1.0 - min(std / 10.0, 0.8)
    return round(max(0.1, score), 4)


def calc_distance_fit_index(records: List[Dict[str, Any]], target_distance: Optional[int]) -> float:

    if target_distance is None:
        return 0.5

    distances = [r["distance"] for r in records if r.get("distance")]

    if not distances:
        return 0.5

    avg_dist = mean(distances)
    std_dist = pstdev(distances) if len(distances) > 1 else 0

    gap = abs(avg_dist - target_distance)

    # 距離安定性
    stability = 1.0 - min(std_dist / 900.0, 0.7)

    # 距離適合
    base = 1.0 - min(gap / 900.0, 0.8)

    score = base * 0.7 + stability * 0.3

    return round(max(0.1, score), 4)


# =========================================================
# Jockey / gate / last3f / lap
# =========================================================

def calc_jockey_index(jockey: str) -> float:
    return JOCKEY_BASE_INDEX.get(jockey, 1.0)


def gate_bucket(gate: Optional[int]) -> str:
    if gate is None:
        return "middle"
    if gate <= 3:
        return "inner"
    if gate >= 7:
        return "outer"
    return "middle"


def calc_gate_index(course: str, distance: Optional[int], gate: Optional[int]) -> float:
    band = distance_band(distance)
    bias = DRAW_BIAS.get((course, band))
    if not bias:
        return 1.0
    return bias.get(gate_bucket(gate), 1.0)


def calc_last3f_index(records: List[Dict[str, Any]]) -> float:
    values = [r["last3f"] for r in records if r.get("last3f") is not None]
    if not values:
        return 0.50
    avg = mean(values)
    if avg <= 33.0:
        return 1.10
    elif avg <= 34.0:
        return 1.05
    elif avg <= 35.0:
        return 1.00
    else:
        return 0.95


def calc_lap_suitability_index(records: List[Dict[str, Any]], predicted_pace: str) -> float:
    deltas: List[int] = []
    for r in records:
        if r["rank"] is None or r["first_corner_pos"] is None:
            continue
        deltas.append(r["first_corner_pos"] - r["rank"])

    if not deltas:
        return 0.50

    avg_delta = mean(deltas)

    if predicted_pace in ("fast", "very_fast"):
        base = sigmoid(avg_delta / 2.5)
    elif predicted_pace == "slow":
        base = sigmoid((-avg_delta + 2.0) / 2.5)
    else:
        base = sigmoid((0.3 * avg_delta) / 2.5)

    return round(float(base), 4)


# =========================================================
# Market distortion AI
# =========================================================

def calc_market_edge(ai_prob: float, market_odds: Optional[float]) -> Optional[float]:
    if market_odds is None or market_odds <= 0:
        return None
    return round(ai_prob - (1.0 / market_odds), 4)


def classify_value_label(edge: Optional[float], ev: Optional[float]) -> str:
    if edge is None or ev is None:
        return "no_market"
    if ev >= 1.10 and edge >= 0.02:
        return "strong_value"
    if ev >= 1.03 and edge >= 0.005:
        return "value"
    if ev < 0.97 and edge <= -0.02:
        return "overbet"
    return "fair"


# =========================================================
# Model score / probability / EV
# =========================================================

def calc_model_score(feature_dict: Dict[str, Any]) -> float:
    dc = feature_dict["distance_course_suitability_index"]
    style = feature_dict["style_suitability_index"]
    recent = feature_dict["recent_form_index"]
    race_level = feature_dict.get("race_level_index", 0.5)
    race_level_boost = race_level * 0.05
    ground = feature_dict["ground_match_index"]
    pace_bias = feature_dict["pace_bias_index"]
    jockey = feature_dict["jockey_index"]
    gate = feature_dict["gate_index"]
    last3f = feature_dict["last3f_index"]
    pace_adv = feature_dict["pace_advantage"]
    lap = feature_dict["lap_suitability_index"]
    trend = feature_dict["trend_index"]
    consistency = feature_dict["consistency_index"]
    distance_fit = feature_dict["distance_fit_index"]

    sample_adj = min(1.0, feature_dict.get("distance_course_sample_size",0) / 3.0)

    model_score = (
        0.18 * dc
        + 0.13 * style
        + 0.15 * recent
        + 0.05 * ground
        + 0.07 * pace_bias
        + 0.07 * last3f
        + 0.07 * lap
        + 0.06 * jockey
        + 0.04 * gate
        + 0.04 * pace_adv
        + 0.06 * trend
        + 0.04 * consistency
        + 0.04 * distance_fit
        + race_level_boost
    ) * (0.75 + 0.25 * sample_adj)

    return round(model_score, 6)


def softmax(scores: List[float], temperature: float = 0.25) -> List[float]:
    if not scores:
        return []
    scaled = [s / max(temperature, 1e-6) for s in scores]
    max_score = max(scaled)
    exps = [math.exp(s - max_score) for s in scaled]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(scores)] * len(scores)
    return [e / total for e in exps]


def estimate_place_prob(win_prob: float) -> float:
    place_prob = 0.12 + 1.75 * win_prob
    return round(min(0.85, max(0.05, place_prob)), 4)


def fair_odds(prob: float) -> Optional[float]:
    if prob <= 0:
        return None
    return round(1.0 / prob, 2)


def calc_expected_value(prob: float, odds: Optional[float]) -> Optional[float]:
    if odds is None:
        return None
    return round(prob * odds, 4)


def kelly_fraction(prob: float, odds: Optional[float]) -> float:
    if odds is None or odds <= 1.0:
        return 0.0
    b = odds - 1.0
    q = 1.0 - prob
    edge = (b * prob - q) / b
    return max(0.0, edge)


# =========================================================
# LightGBM
# =========================================================

ML_FEATURE_COLUMNS: List[str] = [
    "distance_course_suitability_index",
    "style_suitability_index",
    "recent_form_index",
    "ground_match_index",
    "pace_bias_index",
    "jockey_index",
    "gate_index",
    "last3f_index",
    "pace_advantage",
    "lap_suitability_index",
    "trend_index",
    "consistency_index",
    "distance_fit_index",
]


def train_lightgbm_model(csv_path: str = TRAINING_CSV, model_file: str = MODEL_FILE) -> bool:
    if not LIGHTGBM_AVAILABLE:
        print("LightGBM/pandas が未インストールのため学習をスキップします。")
        return False

    if not Path(csv_path).exists():
        print(f"学習CSVが存在しないため学習をスキップします: {csv_path}")
        return False

    df = pd.read_csv(csv_path)
    required = set(ML_FEATURE_COLUMNS + ["target_win"])
    missing = required - set(df.columns)

    if missing:
        print(f"学習CSVに必要列が不足しています: {missing}")
        return False

    X = df[ML_FEATURE_COLUMNS]
    y = df["target_win"]

    train_data = lgb.Dataset(X, label=y)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "seed": 42,
    }

    model = lgb.train(params, train_data, num_boost_round=200)
    model.save_model(model_file)
    print(f"LightGBMモデルを保存しました: {model_file}")
    return True


def predict_win_probability_with_model(
    features: List[Dict[str, Any]],
    model_file: str = MODEL_FILE
) -> Optional[List[float]]:
    if not LIGHTGBM_AVAILABLE:
        return None
    if not Path(model_file).exists():
        return None

    try:
        model = lgb.Booster(model_file=model_file)
        X = pd.DataFrame([{col: f.get(col, 0.0) for col in ML_FEATURE_COLUMNS} for f in features])
        preds = model.predict(X)
        preds = [float(p) for p in preds]
        total = sum(preds)
        if total <= 0:
            return None
        return [p / total for p in preds]
    except Exception as e:
        print(f"LightGBM予測に失敗したためルールベースにフォールバックします: {e}")
        return None

# =========================================================
# Race Pace Simulation AI
# =========================================================

def simulate_race_position(style: str) -> float:
    """
    脚質ごとの位置取りランダム化
    """
    if style == "front":
        return random.gauss(2.5, 1.2)
    elif style == "stalker":
        return random.gauss(5.5, 1.8)
    elif style == "closer":
        return random.gauss(9.0, 2.5)
    return random.gauss(7.0, 2.0)


def race_pace_simulation(features: List[Dict[str, Any]], simulations: int = 2000) -> Dict[str, float]:

    score_counter = {f["horse_name"]: 0.0 for f in features}

    for _ in range(simulations):

        positions = []

        for f in features:

            pos = simulate_race_position(f["running_style"])

            # 逃げ干渉AI
            front_count = sum(1 for x in features if x.get("running_style") == "front")

            if f.get("running_style") == "front" and front_count >= 2:
                pos += random.uniform(0.7, 1.6) * (front_count - 1)

            pace_adv = f.get("pace_advantage", 1.0)
            score = f["model_score"] * pace_adv - pos * 0.01

            positions.append((f["horse_name"], score))

        positions.sort(key=lambda x: x[1], reverse=True)

        for rank, (horse, score) in enumerate(positions):

            if rank == 0:
                score_counter[horse] += 1.0
            elif rank == 1:
                score_counter[horse] += 0.6
            elif rank == 2:
                score_counter[horse] += 0.3

    results = {}

    for horse, s in score_counter.items():
        results[horse] = s / simulations

    return results

def calc_pace_collapse_risk(features: List[Dict[str, Any]]) -> float:

    front_count = sum(1 for f in features if f.get("running_style") == "front")
    stalker_count = sum(1 for f in features if f.get("running_style") == "stalker")

    pace_pressure = front_count + stalker_count * 0.6

    if pace_pressure >= 6:
        return 1.15
    elif pace_pressure >= 4:
        return 1.08
    elif pace_pressure <= 1:
        return 0.92

    return 1.0

# =========================================================
# MonteCarlo Simulation AI
# =========================================================

def monte_carlo_simulation(features: List[Dict[str, Any]], simulations: int = 5000) -> Dict[str, float]:

    if not features:
        return {}

    win_counter = {f["horse_name"]: 0 for f in features}

    for _ in range(simulations):
        sampled = []

        for f in features:
            noise = random.gauss(0, 0.12)
            score = f["model_score"] + noise
            sampled.append((f["horse_name"], score))

        sampled.sort(key=lambda x: x[1], reverse=True)
        winner = sampled[0][0]
        win_counter[winner] += 1

    results: Dict[str, float] = {}
    for horse, wins in win_counter.items():
        results[horse] = wins / simulations

    return results
def apply_montecarlo_to_features(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

    try:
        mc_probs = monte_carlo_simulation(features)

        for f in features:
            f["montecarlo_win_prob"] = round(
                mc_probs.get(f["horse_name"], 0),
                4
            )

    except Exception as e:
        print("MonteCarlo計算エラー:", e)

    return features

# =========================================================
# Odds Value Detection AI
# =========================================================

def detect_value_bets(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for f in features:
        ai_prob = f.get("win_prob", 0)
        odds = f.get("win_odds", None)
        if odds is None or odds == 0:
            f["value_index"] = 0
            continue
        market_prob = 1 / odds
        value = ai_prob - market_prob
        f["value_index"] = round(value, 4)
    return features

def calc_odds_distortion(feature):

    prob = feature.get("win_prob",0)
    odds = feature.get("win_odds")

    if not odds:
        return 0

    market_prob = 1/odds

    if market_prob == 0:
        return 0

    distortion = prob / market_prob

    return round(distortion,3)

def calc_expected_value_score(feature: Dict[str, Any]) -> float:
    win_ev = feature.get("win_ev")
    place_ev = feature.get("place_ev")
    value_index = float(feature.get("value_index", 0.0) or 0.0)
    trend = float(feature.get("trend_index", 0.5) or 0.5)
    consistency = float(feature.get("consistency_index", 0.5) or 0.5)
    win_prob = float(feature.get("win_prob", 0.0) or 0.0)

    win_ev_score = max(0.0, float(win_ev) - 1.0) if isinstance(win_ev, (int, float)) else 0.0
    place_ev_score = max(0.0, float(place_ev) - 1.0) if isinstance(place_ev, (int, float)) else 0.0

    score = (
        win_ev_score * 1.3
        + place_ev_score * 0.8
        + value_index * 1.8
        + max(0.0, trend - 0.5) * 0.6
        + max(0.0, consistency - 0.5) * 0.5
        + win_prob * 0.25
    )
    return round(score, 4)


def classify_bet_suitability(feature: Dict[str, Any]) -> str:
    win_ev = feature.get("win_ev")
    place_ev = feature.get("place_ev")
    win_prob = float(feature.get("win_prob", 0.0) or 0.0)
    odds = feature.get("win_odds")

    if isinstance(win_ev, (int, float)) and win_ev >= 1.12 and win_prob >= 0.12:
        return "win"
    if isinstance(place_ev, (int, float)) and place_ev >= 1.05 and win_prob >= 0.08:
        return "place"
    if isinstance(odds, (int, float)) and odds >= 10 and win_prob >= 0.06:
        return "wide_hole"
    return "pass"


def build_value_summary(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not features:
        return {
            "best_win_value": None,
            "best_place_value": None,
            "best_expected_value": None,
            "danger_popular": None,
        }

    best_win = None
    best_place = None
    best_total = None
    danger = None

    win_candidates = [
        f for f in features
        if isinstance(f.get("win_ev"), (int, float)) and f.get("win_odds") is not None
    ]
    if win_candidates:
        best_win = max(
            win_candidates,
            key=lambda x: (float(x.get("win_ev", 0.0)), float(x.get("expected_value_score", 0.0)))
        )

    place_candidates = [
        f for f in features
        if isinstance(f.get("place_ev"), (int, float)) and f.get("place_odds") is not None
    ]
    if place_candidates:
        best_place = max(
            place_candidates,
            key=lambda x: (float(x.get("place_ev", 0.0)), float(x.get("expected_value_score", 0.0)))
        )

    best_total = max(features, key=lambda x: float(x.get("expected_value_score", 0.0)))

    danger_candidates = [f for f in features if f.get("is_danger_favorite")]
    if danger_candidates:
        danger = max(danger_candidates, key=lambda x: float(x.get("danger_favorite_score", 0.0)))

    return {
        "best_win_value": best_win,
        "best_place_value": best_place,
        "best_expected_value": best_total,
        "danger_popular": danger,
    }

# =========================================================
# AI Confidence
# =========================================================

def calc_ai_confidence(features: List[Dict[str, Any]]) -> float:
    probs = [f["win_prob"] for f in features if f.get("win_prob") is not None]
    if len(probs) < 2:
        return 0.5

    top = max(probs)
    second = sorted(probs, reverse=True)[1]
    diff = top - second
    variance = pstdev(probs) if len(probs) >= 2 else 0.0

    confidence = min(1.0, diff * 2.2 + max(0.0, 0.28 - variance))
    return round(max(0.05, confidence), 3)

# =========================================================
# Dark Horse AI

def detect_dark_horses(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dark: List[Dict[str, Any]] = []

    for f in features:
        odds = f.get("win_odds")
        value_index = f.get("value_index", 0.0)

        if odds is None:
            continue

        if odds >= 8 and f.get("win_prob", 0.0) >= 0.05 and value_index >= 0:
            dark.append(f)

    dark.sort(
        key=lambda x: (x.get("value_index", 0.0), x.get("win_prob", 0.0)),
        reverse=True
    )

    return dark[:3]
# =========================================================

# =========================================================
# Danger Favorite AI
# =========================================================

# =========================================================
# Version10 Race Summary Helpers
# =========================================================

def classify_race_type(features: List[Dict[str, Any]]) -> str:
    if not features or len(features) < 2:
        return "判定不可"

    probs = sorted([float(f.get("win_prob", 0.0)) for f in features], reverse=True)
    top = probs[0]
    second = probs[1]
    gap = top - second

    if gap >= 0.15:
        return "本命信頼型"
    if gap >= 0.08:
        return "標準型"
    if gap >= 0.04:
        return "混戦型"
    return "波乱警戒型"


def calc_expected_roi(features: List[Dict[str, Any]]) -> float:
    candidates: List[float] = []
    for f in features:
        win_ev = f.get("win_ev")
        place_ev = f.get("place_ev")
        if isinstance(win_ev, (int, float)):
            candidates.append(float(win_ev))
        if isinstance(place_ev, (int, float)):
            candidates.append(float(place_ev))

    if not candidates:
        return 1.0

    return round(max(candidates), 4)


def build_positioning_map(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for idx, f in enumerate(features, start=1):
        style = f.get("running_style", "unknown")

        if style == "front":
            pos_score = 1
        elif style == "stalker":
            pos_score = 2
        elif style == "closer":
            pos_score = 3
        else:
            pos_score = 4

        rows.append({
            "rank": idx,
            "horse_name": f.get("horse_name", ""),
            "style": style_to_japanese(style),
            "position_score": pos_score,
            "win_prob": f.get("win_prob", 0.0),
            "pace_simulation_index": f.get("pace_simulation_index", 0.0),
        })

    return rows


def detect_danger_favorites(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

    dangers = []

    for f in features:

        odds = f.get("win_odds")
        prob = f.get("win_prob", 0)

        trend = f.get("trend_index", 0.5)
        consistency = f.get("consistency_index", 0.5)

        if odds is None:
            continue

        market_prob = 1 / odds if odds > 0 else 0
        gap = market_prob - prob

        risk_score = (
            gap * 2.0
            + (0.5 - trend) * 0.9
            + (0.5 - consistency) * 0.7
        )

        f["danger_favorite_score"] = round(risk_score, 4)
        f["is_danger_favorite"] = bool(odds <= 5 and risk_score > 0.05)

        if f["is_danger_favorite"]:
            dangers.append({
                "horse_name": f.get("horse_name"),
                "win_prob": round(float(prob), 4),
                "win_odds": odds,
                "danger_gap": round(gap, 4),
                "risk_score": round(risk_score, 4)
            })

    dangers.sort(key=lambda x: x["risk_score"], reverse=True)

    return dangers[:3]

# =========================================================
# Result payload refresh
# =========================================================

def refresh_feature_outputs(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    features = apply_montecarlo_to_features(features)
    features = detect_value_bets(features)

    for f in features:
        f["pace_style_label"] = style_to_japanese(f.get("running_style", "unknown"))
        f["newspaper_style_label"] = style_to_japanese(f.get("newspaper_entry_style", "unknown"))
        f["ai_power_index"] = calc_ai_power_index(f)
        f["radar_payload"] = build_radar_payload(f)
        f["expected_value_score"] = calc_expected_value_score(f)
        f["bet_suitability"] = classify_bet_suitability(f)

    features.sort(key=lambda x: (x.get("win_prob", 0.0), x.get("ai_power_index", 0.0)), reverse=True)
    return features


def refresh_result_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    features = result.get("features", [])
    if not features:
        result["ai_confidence"] = 0.5
        result["dark_horses"] = []
        result["danger_favorites"] = []
        result["pace_balance"] = {"逃げ": 0, "先行": 0, "差し": 0, "不明": 0}
        result["top_pick"] = None
        result["race_type"] = "判定不可"
        result["expected_roi"] = 1.0
        result["positioning_map"] = []
        result["ai_comment"] = result.get("ai_comment") or "分析データ不足"
        result["value_summary"] = {
            "best_win_value": None,
            "best_place_value": None,
            "best_expected_value": None,
            "danger_popular": None,
        }
        result["race_trend_10y"] = result.get("race_trend_10y") or {}
        result["past_10y_results"] = result.get("past_10y_results") or []
        result["trend_match_horses"] = result.get("trend_match_horses") or []
        result["race_history_summary"] = result.get("race_history_summary") or "過去10年傾向データなし"
        result["winner_conditions"] = result.get("winner_conditions") or []
        result["condition_match_horses"] = result.get("condition_match_horses") or []
        result["winner_pattern_ai"] = result.get("winner_pattern_ai") or {}
        return result

    features = refresh_feature_outputs(features)
    result["features"] = features
    result["ai_confidence"] = calc_ai_confidence(features)
    result["dark_horses"] = detect_dark_horses(features)
    result["danger_favorites"] = detect_danger_favorites(features)
    result["pace_balance"] = build_pace_balance(features)
    result["top_pick"] = features[0] if features else None
    result["race_type"] = classify_race_type(features)
    result["expected_roi"] = calc_expected_roi(features)
    result["positioning_map"] = build_positioning_map(features)
    result["value_summary"] = build_value_summary(features)
    result["race_trend_10y"] = result.get("race_trend_10y") or {}
    result["past_10y_results"] = result.get("past_10y_results") or []
    result["trend_match_horses"] = result.get("trend_match_horses") or []
    result["race_history_summary"] = result.get("race_history_summary") or "過去10年傾向データなし"
    result["winner_conditions"] = result.get("winner_conditions") or []
    result["condition_match_horses"] = result.get("condition_match_horses") or []
    result["winner_pattern_ai"] = result.get("winner_pattern_ai") or {}
    result["ai_comment"] = generate_ai_comment(result)
    return result

# =========================================================
# GPT Commentary AI
# =========================================================

def generate_ai_comment(result: Dict[str, Any]) -> str:
    features = result.get("features", [])
    if not features:
        return "分析データ不足"

    top = features[0]
    top2 = features[1] if len(features) > 1 else None
    pace = result.get("race_meta", {}).get("predicted_pace", "unknown")
    dark_horses = result.get("dark_horses", []) or []
    danger_favorites = result.get("danger_favorites", []) or []
    race_trend_10y = result.get("race_trend_10y", {}) or {}
    race_history_summary = result.get("race_history_summary", "") or ""
    winner_conditions = result.get("winner_conditions", []) or []
    condition_match_horses = result.get("condition_match_horses", []) or []

    avg_win_odds = race_trend_10y.get("avg_win_odds")
    favorite_ratio = race_trend_10y.get("favorite_ratio")
    style_trend = race_trend_10y.get("style_summary")
    popularity_trend = race_trend_10y.get("popularity_summary")

    if style_trend:
        trend_line_1 = f"過去10年では{style_trend}。"
    else:
        trend_line_1 = race_history_summary if race_history_summary else "過去10年の傾向は取得済みだが、要約は限定的。"

    if isinstance(avg_win_odds, (int, float)):
        trend_line_2 = f"平均勝ち馬オッズは{round(float(avg_win_odds), 1)}倍"
    else:
        trend_line_2 = "平均勝ち馬オッズは不明"

    if isinstance(favorite_ratio, (int, float)):
        trend_line_2 += f"、1番人気寄与率は{round(float(favorite_ratio) * 100, 1)}%"

    if popularity_trend:
        trend_line_2 += f"。人気傾向は{popularity_trend}。"
    else:
        trend_line_2 += "。"

    pace_label = pace_to_japanese(pace)
    front_count = result.get("pace_balance", {}).get("逃げ", 0)
    stalker_count = result.get("pace_balance", {}).get("先行", 0)

    if pace == "slow":
        pace_line_2 = "逃げ馬が少なく、先行勢が有利になりやすい展開。"
    elif pace in ("fast", "very_fast"):
        pace_line_2 = "前に行く馬が多く、差し馬の台頭に警戒。"
    else:
        pace_line_2 = "極端な隊列になりにくく、総合力勝負になりやすい。"

    pace_line_1 = f"展開は{pace_label}想定。逃げ{front_count}頭・先行{stalker_count}頭構成。"

    top_reason_parts = []
    if float(top.get("distance_fit_index", 0.0) or 0.0) >= 0.7:
        top_reason_parts.append("距離適性が高い")
    if float(top.get("recent_form_index", 0.0) or 0.0) >= 0.65:
        top_reason_parts.append("近走指数が上位")
    if float(top.get("trend_index", 0.0) or 0.0) >= 0.6:
        top_reason_parts.append("上昇度が高い")
    if float(top.get("consistency_index", 0.0) or 0.0) >= 0.6:
        top_reason_parts.append("安定感がある")
    if float(top.get("pace_advantage", 1.0) or 1.0) > 1.02:
        top_reason_parts.append("展開恩恵が見込める")
    if not top_reason_parts:
        top_reason_parts.append("総合指数が最上位")

    ability_block = (
        f"1位 {top['horse_name']}\n"
        f"AIパワー{top.get('ai_power_index', 0)} / 勝率{round(float(top.get('win_prob', 0.0) or 0.0) * 100, 1)}%\n"
        f"{('・'.join(top_reason_parts[:3]))}。"
    )

    if top2:
        top2_reason_parts = []
        if float(top2.get("distance_fit_index", 0.0) or 0.0) >= 0.7:
            top2_reason_parts.append("距離適性が高い")
        if float(top2.get("recent_form_index", 0.0) or 0.0) >= 0.65:
            top2_reason_parts.append("近走指数が上位")
        if float(top2.get("consistency_index", 0.0) or 0.0) >= 0.6:
            top2_reason_parts.append("安定指数が高い")
        if float(top2.get("pace_advantage", 1.0) or 1.0) > 1.02:
            top2_reason_parts.append("展開利がある")
        if not top2_reason_parts:
            top2_reason_parts.append("崩れにくい総合型")

        ability_block += (
            f"\n\n2位 {top2['horse_name']}\n"
            f"AIパワー{top2.get('ai_power_index', 0)} / 勝率{round(float(top2.get('win_prob', 0.0) or 0.0) * 100, 1)}%\n"
            f"{('・'.join(top2_reason_parts[:3]))}。"
        )

    if dark_horses:
        dark = dark_horses[0]
        dark_name = dark.get("horse_name", "不明")
        dark_line = f"{dark_name}\nAI指数は中位以上で、オッズに対する妙味あり。"
    else:
        dark_line = "該当馬なし"

    if danger_favorites:
        danger = danger_favorites[0]
        danger_name = danger.get("horse_name", "不明")
        danger_line = f"{danger_name}\n人気に対して期待値が低く、過剰人気の可能性。"
    else:
        danger_line = "該当馬なし"

    cond_line = ""
    if winner_conditions:
        cond_line = "\n\n■過去10年勝ち馬パターン\n"
        for c in winner_conditions[:5]:
            cond_line += f"・{c}\n"

        cond_line += "\n■今年の該当馬\n"

        match_counts: Dict[str, int] = {}
        if condition_match_horses:
            for raw_name in condition_match_horses:
                name = str(raw_name).strip()
                if not name:
                    continue
                match_counts[name] = match_counts.get(name, 0) + 1

        if match_counts:
            sorted_matches = sorted(match_counts.items(), key=lambda x: x[1], reverse=True)
            for h, count in sorted_matches[:5]:
                cond_line += f"・{h}（{count}条件一致）\n"
        else:
            cond_line += "該当馬なし\n"

    return (
        "■レース傾向\n"
        f"{trend_line_1}\n"
        f"{trend_line_2}\n\n"
        "■展開予測\n"
        f"{pace_line_1}\n"
        f"{pace_line_2}\n\n"
        "■能力上位\n"
        f"{ability_block}\n\n"
        "■穴候補\n"
        f"{dark_line}\n\n"
        "■危険人気馬\n"
        f"{danger_line}"
        f"{cond_line}"
    )

# =========================================================
# ChatGPT Winner Pattern Extraction AI
# =========================================================

WINNER_PATTERN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "trend_summary": {"type": "string"},
        "winner_conditions": {
            "type": "array",
            "items": {"type": "string"}
        },
        "matching_horses": {
            "type": "array",
            "items": {"type": "string"}
        },
        "confidence": {"type": "number"},
        "notes": {"type": "string"}
    },
    "required": [
        "trend_summary",
        "winner_conditions",
        "matching_horses",
        "confidence",
        "notes"
    ],
    "additionalProperties": False,
}


def analyze_winner_patterns_with_chatgpt(
    race_title: str,
    past_results: List[Dict[str, Any]],
    features: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    ChatGPTを使って
    1. 過去データから勝ち馬パターンを抽出
    2. 今回の出走馬の中で該当馬を返す

    注意:
    - 入力で与えた情報以外を事実として作らせない
    - 不明なものは不明として返す
    """

    if not past_results:
        return {
            "trend_summary": "過去データ不足",
            "winner_conditions": ["条件データ不足"],
            "matching_horses": [],
            "confidence": 0.0,
            "notes": "past_results が空のため ChatGPT 分析をスキップしました",
        }

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {
            "trend_summary": "OPENAI_API_KEY 未設定のためChatGPT分析をスキップ",
            "winner_conditions": ["APIキー未設定"],
            "matching_horses": [],
            "confidence": 0.0,
            "notes": "環境変数 OPENAI_API_KEY を設定してください",
        }

    try:
        client = OpenAI(api_key=api_key)

        condensed_features: List[Dict[str, Any]] = []
        for f in features:
            condensed_features.append({
                "horse_name": f.get("horse_name"),
                "running_style": style_to_japanese(f.get("running_style", "unknown")),
                "gate": f.get("gate"),
                "win_prob": f.get("win_prob"),
                "win_odds": f.get("win_odds"),
                "win_ev": f.get("win_ev"),
                "place_ev": f.get("place_ev"),
                "recent_form_index": f.get("recent_form_index"),
                "distance_fit_index": f.get("distance_fit_index"),
                "race_level_index": f.get("race_level_index"),
                "value_flag": f.get("value_flag"),
                "newspaper_mark": f.get("newspaper_mark"),
            })

        payload = {
            "race_title": race_title,
            "past_results": past_results,
            "current_runners": condensed_features,
        }

        developer_prompt = """
あなたは中央競馬の重賞を期待値ベースで分析するAIです。

絶対ルール:
- 入力で与えられた情報以外を事実として断定しない
- 架空の過去データ、枠順、脚質、人気、オッズ、騎手を作らない
- 根拠が弱い場合はその旨を notes に書く
- winner_conditions は箇条書き向けの短文で返す
- matching_horses は current_runners に存在する horse_name だけを返す
- 該当馬がいない場合は matching_horses を空配列にする
- JSONのみを返す
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "developer", "content": developer_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "winner_pattern_analysis",
                    "schema": WINNER_PATTERN_SCHEMA,
                },
            },
        )

        content = response.choices[0].message.content
        parsed = json.loads(content)

        # 念のため、存在しない馬名を除外
        valid_names = {str(f.get("horse_name")) for f in features if f.get("horse_name")}
        parsed["matching_horses"] = [
            name for name in parsed.get("matching_horses", []) if name in valid_names
        ]
        return parsed

    except Exception as e:
        return {
            "trend_summary": f"ChatGPT勝ち馬パターン分析に失敗: {type(e).__name__}",
            "winner_conditions": ["条件抽出失敗"],
            "matching_horses": [],
            "confidence": 0.0,
            "notes": str(e),
        }

# =========================================================
# Scraping helpers
# =========================================================

def extract_race_meta(driver: webdriver.Chrome) -> RaceMeta:
    title = safe_driver_title(driver)
    race_info_text = ""

    race_info_elements = driver.find_elements(By.CSS_SELECTOR, ".RaceData01, .RaceData02")
    if race_info_elements:
        race_info_text = " ".join([e.text for e in race_info_elements if e.text])

    target_surface, target_distance = parse_distance(race_info_text)

    target_course = parse_course_name(race_info_text)
    if target_course == "unknown":
        target_course = parse_course_name_from_title(title)

    target_ground = ""
    for g in ["良", "稍重", "重", "不良"]:
        if g in race_info_text:
            target_ground = g
            break

    return RaceMeta(
        race_title=title,
        race_info_text=race_info_text,
        target_surface=target_surface,
        target_distance=target_distance,
        target_course=target_course,
        target_ground=target_ground,
    )


def fetch_horses(driver: webdriver.Chrome) -> List[Dict[str, Any]]:
    horses: List[Dict[str, Any]] = []

    if driver is None:
        return horses

    try:
        rows = driver.find_elements(By.CSS_SELECTOR, "table.Shutuba_Table tbody tr")
    except Exception:
        return horses

    for row in rows:
        try:
            horse_link_elem = row.find_element(By.CSS_SELECTOR, ".HorseName a")
            horse_name = safe_text(horse_link_elem)
            horse_link = safe_attr(horse_link_elem, "href")

            gate_text = safe_find_text(row, By.CSS_SELECTOR, ".Waku")
            number_text = safe_find_text(row, By.CSS_SELECTOR, ".Umaban")
            jockey_text = safe_find_text(row, By.CSS_SELECTOR, ".Jockey")
            win_odds_text = safe_find_text(row, By.CSS_SELECTOR, ".Odds")
            place_odds_text = safe_find_text(row, By.CSS_SELECTOR, ".Place_Odds")

            if not gate_text:
                all_tds = row.find_elements(By.TAG_NAME, "td")
                td_texts = [safe_text(td) for td in all_tds]
                digit_cells = [t for t in td_texts if t.isdigit()]
                if len(digit_cells) >= 2:
                    gate_text = digit_cells[0]
                    number_text = digit_cells[1]

            gate = int(gate_text) if gate_text.isdigit() else None
            number = int(number_text) if number_text.isdigit() else None
            win_odds = parse_float(win_odds_text)
            place_odds = parse_odds_range_text(place_odds_text)

            horses.append({
                "name": horse_name,
                "link": horse_link,
                "gate": gate,
                "number": number,
                "jockey": jockey_text,
                "win_odds_scraped": win_odds,
                "place_odds_scraped": place_odds,
            })
        except Exception:
            continue

    return horses



# 新聞ページ解析用ヘルパー
def style_char_to_running_style(style_char: str) -> str:
    mapping = {
        "逃": "front",
        "先": "stalker",
        "差": "closer",
        "追": "closer",
        "自": "stalker",
    }
    return mapping.get(style_char, "unknown")


def style_char_to_first_corner_pos(style_char: str) -> Optional[int]:
    mapping = {
        "逃": 2,
        "先": 5,
        "差": 9,
        "追": 12,
        "自": 7,
    }
    return mapping.get(style_char)


def parse_newspaper_style_char(row_text: str) -> str:
    text = row_text or ""

    # 最優先: 「逃/先/差/追/自」の直後に「中◯週」が続く新聞表記
    m = re.search(r"([逃先差追自])\s*中\d+週", text)
    if m:
        return m.group(1)

    # 次点: オッズや人気の手前に単独で置かれることが多い
    m = re.search(r"([逃先差追自])\s*(?:\d+\.\d|\(|\d+人気)", text)
    if m:
        return m.group(1)

    # 行ごとに単独文字を拾う
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        if line in {"逃", "先", "差", "追", "自"}:
            return line

        if line.endswith(("逃", "先", "差", "追", "自")) and len(line) <= 3:
            return line[-1]

        # 例: 「◀◀◀◀ 自」「◀ 先」
        m = re.search(r"([逃先差追自])$", line)
        if m and ("◀" in line or "←" in line or "■" in line or "□" in line):
            return m.group(1)

    # 最後の保険
    candidates = re.findall(r"[逃先差追自]", text)
    if candidates:
        return candidates[-1]

    return ""

def parse_newspaper_mark(row_text: str) -> str:
    """
    新聞印（◎◯▲△☆など）を抽出
    """
    if not row_text:
        return ""

    marks = re.findall(r"[◎○◯▲△☆★×]", row_text)

    if not marks:
        return ""

    return marks[0]


def newspaper_mark_to_index(mark: str) -> float:
    """
    新聞印をAI指数化
    """

    table = {
        "◎": 1.12,
        "○": 1.08,
        "◯": 1.08,
        "▲": 1.05,
        "△": 1.02,
        "☆": 1.03,
        "★": 1.04,
        "×": 0.95,
    }

    return table.get(mark, 1.0)

def parse_newspaper_past_record_text(text: str, style_char: str = "") -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    joined = " ".join(lines)

    rank: Optional[int] = None
    m = re.search(r"(?<!\d)(\d{1,2})\s*\d{1,2}頭", joined)
    if m:
        rank = int(m.group(1))
    else:
        m = re.search(r"(\d+)着", joined)
        if m:
            rank = int(m.group(1))

    surface: Optional[str] = None
    distance: Optional[int] = None
    m = re.search(r"(芝|ダ|障)\s*(\d{3,4})", joined)
    if m:
        surface = m.group(1)
        distance = int(m.group(2))

    time_sec: Optional[float] = None
    m = re.search(r"(\d+:\d{2}\.\d)", joined)
    if m:
        time_sec = parse_time_to_seconds(m.group(1))

    ground = ""
    for g in ["良", "稍重", "重", "不良"]:
        if g in joined:
            ground = g
            break

    passing_text = ""
    first_corner_pos: Optional[int] = None
    m = re.search(r"(\d+(?:-\d+)+)", joined)
    if m:
        passing_text = m.group(1)
        first_corner_pos = parse_first_corner_position(passing_text)
    else:
        first_corner_pos = style_char_to_first_corner_pos(style_char)

    last3f: Optional[float] = None
    m = re.search(r"後\s*(\d{2}\.\d)", joined)
    if m:
        last3f = parse_float(m.group(1))
    else:
        floats = re.findall(r"\d{2}\.\d", joined)
        if floats:
            last3f = parse_float(floats[-1])

    course_name = parse_course_name(joined)

    race_name = ""
    # 2行目付近にレース名が来ることが多いが、馬齢や人気だけの行は除外
    for line in lines[1:3]:
        if not re.search(r"\d+頭|\d+人気|馬齢|右|左|芝|ダ|障", line):
            race_name = line
            break
    if not race_name and len(lines) >= 2:
        race_name = lines[1]
    elif not race_name and lines:
        race_name = lines[0]

    class_index = parse_race_class_index(race_name)

    if all(v in (None, "", 0.5) for v in [rank, distance, time_sec, last3f]) and not passing_text and not race_name:
        return None

    return {
        "date": "",
        "course_text": joined,
        "course_name": course_name,
        "weather": "",
        "ground": ground,
        "race_name": race_name,
        "class_index": class_index,
        "rank_text": str(rank) if rank is not None else "",
        "rank": rank,
        "jockey": "",
        "time_text": "",
        "time_sec": time_sec,
        "last3f": last3f,
        "passing_text": passing_text,
        "first_corner_pos": first_corner_pos,
        "distance_text": f"{surface or ''}{distance or ''}",
        "surface": surface,
        "distance": distance,
    }


def fetch_newspaper_records(driver: webdriver.Chrome) -> Dict[str, Dict[str, Any]]:
    """
    競馬新聞ページから各馬の近走データをまとめて取得する。
    horse_name -> {records, sire_name, style_char} の辞書を返す
    """
    records_map: Dict[str, Dict[str, Any]] = {}

    try:
        rows = driver.find_elements(By.CSS_SELECTOR, "table.Newspaper_Table tbody tr")
    except Exception:
        return records_map

    for row in rows:
        try:
            horse_name = safe_find_text(row, By.CSS_SELECTOR, ".HorseName a")
            if not horse_name:
                horse_name = safe_find_text(row, By.CSS_SELECTOR, ".HorseName")
            if not horse_name:
                continue

            row_text = safe_text(row)
            row_lines = [line.strip() for line in row_text.splitlines() if line.strip()]

            sire_name = ""
            if row_lines:
                # 1行目は父名であることが多いが、馬名や脚質記号の誤取得を避ける
                if row_lines[0] != horse_name and row_lines[0] not in {"逃", "先", "差", "追", "自"}:
                    sire_name = row_lines[0]
            style_char = parse_newspaper_style_char(row_text)
            newspaper_mark = parse_newspaper_mark(row_text)

            cols = row.find_elements(By.TAG_NAME, "td")
            past_records: List[Dict[str, Any]] = []

            for i in range(5):
                idx = 6 + i
                if idx >= len(cols):
                    break

                cell_text = safe_text(cols[idx])
                parsed = parse_newspaper_past_record_text(cell_text, style_char=style_char)
                if parsed:
                    past_records.append(parsed)

            records_map[horse_name] = {
                "records": past_records,
                "sire_name": sire_name,
                "style_char": style_char,
                "newspaper_mark": newspaper_mark,
            }

        except Exception:
            continue

    return records_map


def fetch_horse_records(
    driver: webdriver.Chrome,
    horse_url: str,
    history_limit: int = HISTORY_LIMIT_DEFAULT,
    headless: bool = False,
) -> Tuple[List[Dict[str, Any]], str]:
    records: List[Dict[str, Any]] = []

    if driver is None:
        return records, ""

    cache_name = cache_key("horse_records", horse_url, history_limit)
    cached = load_json_cache(cache_name)
    if cached:
        cached_records = cached.get("records", [])
        cached_sire = str(cached.get("sire_name", ""))
        if isinstance(cached_records, list):
            return cached_records, cached_sire

    try:
        random_sleep(2.4, 4.2)
        driver = safe_get(driver, horse_url, headless=headless, retries=1)
        emulate_human_behavior(driver)

        page_src = safe_page_source(driver)
        page_title = safe_driver_title(driver)
        if is_blocked_page(page_src, page_title):
            print("馬ページ取得ブロック")
            return records, ""

        sire_name = parse_sire_name(driver)

        try:
            rows = driver.find_elements(By.CSS_SELECTOR, ".db_h_race_results tbody tr")
        except Exception:
            return records, sire_name

        for r in rows[:history_limit]:
            try:
                cols = r.find_elements(By.TAG_NAME, "td")
            except Exception:
                continue

            if len(cols) < 21:
                continue

            date = safe_text(cols[0])
            course_text = safe_text(cols[1])
            weather = safe_text(cols[2]) if len(cols) > 2 else ""
            ground = safe_text(cols[3]) if len(cols) > 3 else ""
            race_name = safe_text(cols[4]) if len(cols) > 4 else ""
            rank_text = safe_text(cols[11])
            jockey = safe_text(cols[12]) if len(cols) > 12 else ""
            distance_text = safe_text(cols[14]) if len(cols) > 14 else ""
            time_text = safe_text(cols[17]) if len(cols) > 17 else ""
            last3f_text = safe_text(cols[18]) if len(cols) > 18 else ""
            passing_text = safe_text(cols[20]) if len(cols) > 20 else ""

            surface, distance = parse_distance(distance_text)
            rank = parse_rank(rank_text)
            time_sec = parse_time_to_seconds(time_text)
            first_corner_pos = parse_first_corner_position(passing_text)
            course_name = parse_course_name(course_text)
            class_index = parse_race_class_index(race_name)
            last3f = parse_last3f(last3f_text)

            records.append({
                "date": date,
                "course_text": course_text,
                "course_name": course_name,
                "weather": weather,
                "ground": ground,
                "race_name": race_name,
                "class_index": class_index,
                "rank_text": rank_text,
                "rank": rank,
                "jockey": jockey,
                "time_text": time_text,
                "time_sec": time_sec,
                "last3f": last3f,
                "passing_text": passing_text,
                "first_corner_pos": first_corner_pos,
                "distance_text": distance_text,
                "surface": surface,
                "distance": distance,
            })

        save_json_cache(cache_name, {
            "records": records,
            "sire_name": sire_name,
        })
        return records, sire_name
    except Exception as e:
        print(f"馬ページ取得中エラー: {type(e).__name__}")
        return records, ""


# =========================================================
# Feature construction
# =========================================================

def build_feature_dict(
    horse_name: str,
    records: List[Dict[str, Any]],
    race_meta: RaceMeta,
    predicted_pace: str,
    gate: Optional[int] = None,
    jockey_from_entry: str = "",
    scraped_win_odds: Optional[float] = None,
    scraped_place_odds: Optional[float] = None,
) -> Dict[str, Any]:
    dc_index = calc_distance_course_index(
        records=records,
        target_surface=race_meta.target_surface,
        target_distance=race_meta.target_distance,
        target_course=race_meta.target_course,
    )
    style_index = calc_style_index(records, predicted_pace=predicted_pace)
    distance_band_stats = calc_distance_band_stats(records)
    course_stats = calc_course_stats(records)

    ranks = [r["rank"] for r in records if r["rank"] is not None]
    avg_rank_last_n = safe_int_mean(ranks)

    recent_form_index = calc_recent_form_index(records, top_n=5)
    ground_match_index = calc_ground_match_index(records, race_meta.target_ground)

    latest_jockey = jockey_from_entry or (records[0]["jockey"] if records else "")
    jockey_index = calc_jockey_index(latest_jockey)
    gate_index = calc_gate_index(race_meta.target_course, race_meta.target_distance, gate)
    last3f_index = calc_last3f_index(records)
    pace_advantage = calc_pace_advantage(style_index["style"], predicted_pace)
    lap_suitability_index = calc_lap_suitability_index(records, predicted_pace)
    trend_index = calc_trend_index(records)
    consistency_index = calc_consistency_index(records)
    distance_fit_index = calc_distance_fit_index(records, race_meta.target_distance)
    race_level_index = calc_race_level_index(records)

    return {
        "horse_name": horse_name,
        "target_surface": race_meta.target_surface,
        "target_distance": race_meta.target_distance,
        "target_course": race_meta.target_course,
        "target_ground": race_meta.target_ground,
        "predicted_pace": predicted_pace,
        "gate": gate,
        "entry_jockey": latest_jockey,
        "history_count": len(records),
        "avg_rank_last_n": round(avg_rank_last_n, 4) if avg_rank_last_n is not None else None,

        "distance_course_suitability_index": dc_index["index"],
        "distance_course_sample_size": dc_index["sample_size"],
        "distance_course_win_rate": dc_index["win_rate"],
        "distance_course_place_rate": dc_index["place_rate"],
        "distance_course_show_rate": dc_index["show_rate"],
        "distance_course_avg_rank": dc_index["avg_rank"],
        "distance_course_avg_time_sec": dc_index["avg_time_sec"],

        "running_style": style_index["style"],
        "style_sample_size": style_index["sample_size"],
        "pace_bias_index": style_index["pace_bias_index"],
        "style_perf_index": style_index["style_perf_index"],
        "style_suitability_index": style_index["style_index"],

        "recent_form_index": recent_form_index,
        "ground_match_index": ground_match_index,

        "jockey_index": jockey_index,
        "gate_index": gate_index,
        "last3f_index": last3f_index,
        "pace_advantage": pace_advantage,
        "lap_suitability_index": lap_suitability_index,

        "trend_index": trend_index,
        "consistency_index": consistency_index,
        "distance_fit_index": distance_fit_index,
        "race_level_index": race_level_index,

        "distance_band_stats": distance_band_stats,
        "course_stats": course_stats,
        "past_races": records,

        "win_odds_scraped": scraped_win_odds,
        "place_odds_scraped": scraped_place_odds,
    }

# =========================================================
# Past Race History AI (過去レース傾向AI)
# =========================================================

def fetch_race_history(driver: webdriver.Chrome, race_id: str) -> List[Dict[str, Any]]:
    """
    過去同レースの結果ページから簡易データを取得する（最大10年想定）
    """
    results: List[Dict[str, Any]] = []

    try:
        history_url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
        driver = safe_get(driver, history_url, headless=False, retries=1)
        random_sleep(1.5, 2.5)

        rows = driver.find_elements(By.CSS_SELECTOR, "table.RaceTable01 tbody tr")

        for row in rows:
            try:
                cols = row.find_elements(By.TAG_NAME, "td")
                if len(cols) < 15:
                    continue

                rank = parse_rank(safe_text(cols[0]))
                horse = safe_text(cols[3])
                jockey = safe_text(cols[6])
                odds = parse_float(safe_text(cols[12]))

                results.append({
                    "rank": rank,
                    "horse_name": horse,
                    "jockey": jockey,
                    "odds": odds,
                })

            except Exception:
                continue

    except Exception:
        return []

    return results


def analyze_race_trend(past_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    過去結果から人気傾向などを簡易分析
    """
    if not past_results:
        return {}

    favorite_wins = 0
    total = 0

    for r in past_results:
        odds = r.get("odds")
        if odds is None:
            continue

        total += 1
        if odds <= 3:
            favorite_wins += 1

    fav_rate = favorite_wins / total if total else 0

    return {
        "favorite_win_rate": round(fav_rate, 3),
        "sample_size": total
    }


def build_race_trend_summary(past_results: List[Dict[str, Any]]) -> str:
    """
    レース傾向テキスト生成
    """
    if not past_results:
        return "過去データなし"

    trends = analyze_race_trend(past_results)

    fav_rate = trends.get("favorite_win_rate", 0)

    if fav_rate >= 0.6:
        return "過去傾向: 本命寄りレース"
    elif fav_rate >= 0.35:
        return "過去傾向: 標準レース"
    else:
        return "過去傾向: 波乱傾向"


def build_winner_condition_ai(past_results: List[Dict[str, Any]]) -> str:
    """
    勝ち馬の共通条件を簡易生成
    """
    if not past_results:
        return "条件データなし"

    low_odds = [r for r in past_results if r.get("odds") and r["odds"] <= 5]

    if len(low_odds) >= len(past_results) * 0.6:
        return "勝ち馬条件: 5倍以内の人気馬が中心"

    mid_odds = [r for r in past_results if r.get("odds") and 5 < r["odds"] <= 15]

    if len(mid_odds) >= len(past_results) * 0.4:
        return "勝ち馬条件: 中穴ゾーン（5〜15倍）"

    return "勝ち馬条件: 人気薄の激走あり"

# =========================================================
# Main analysis
# =========================================================

def analyze_race(
    race_url: str,
    history_limit: int = HISTORY_LIMIT_DEFAULT,
    headless: bool = True,
) -> Dict[str, Any]:
    input_url = race_url.strip()
    if "newspaper.html" in input_url:
        shutuba_url = input_url.replace("newspaper.html", "shutuba.html")
        newspaper_url = input_url
    elif "shutuba.html" in input_url:
        shutuba_url = input_url
        newspaper_url = input_url.replace("shutuba.html", "newspaper.html")
    else:
        shutuba_url = input_url
        newspaper_url = input_url

    race_cache_name = cache_key("race_analysis", shutuba_url, history_limit)
    cached_result = load_json_cache(race_cache_name)
    if cached_result and cached_result.get("race_meta") and cached_result.get("features"):
        return refresh_result_payload(cached_result)

    driver = build_webdriver(headless=headless)
    try:
        driver = warmup_netkeiba_session(driver, headless=headless)

        driver = safe_get(driver, shutuba_url, headless=headless, retries=1)
        random_sleep(5.0, 7.5)
        emulate_human_behavior(driver)

        page_src = safe_page_source(driver)
        current_title = safe_driver_title(driver)

        if is_blocked_page(page_src, current_title):
            return {
                "race_meta": {
                    "race_title": "データ取得失敗",
                    "race_info_text": "netkeiba側でアクセスがブロックされました",
                    "target_surface": None,
                    "target_distance": None,
                    "target_course": "unknown",
                    "target_ground": "",
                    "predicted_pace": "unknown",
                },
                "features": [],
                "error": "HTTP400_BLOCKED",
            }

        save_cookies(driver, COOKIE_FILE)

        tables: List[Any] = []
        for _ in range(20):
            try:
                tables = driver.find_elements(By.CSS_SELECTOR, "table.Shutuba_Table tbody tr")
            except Exception:
                tables = []

            if tables and len(tables) > 3:
                break

            time.sleep(1)

        time.sleep(1.5)

        race_meta = extract_race_meta(driver)
        horses = fetch_horses(driver)
        newspaper_records: Dict[str, Dict[str, Any]] = {}

        try:
            driver = safe_get(driver, newspaper_url, headless=headless, retries=1)
            random_sleep(2.0, 3.0)
            newspaper_records = fetch_newspaper_records(driver)
        except Exception:
            newspaper_records = {}

        if not horses:
            return {
                "race_meta": {
                    **race_meta.__dict__,
                    "race_title": race_meta.race_title or "データ取得失敗",
                    "race_info_text": race_meta.race_info_text or "出馬表を取得できませんでした",
                },
                "features": [],
                "error": "NO_HORSES_FOUND",
            }

        horse_results: List[Dict[str, Any]] = []
        entry_styles: List[str] = []

        for horse in horses:
            horse_url = str(horse.get("link") or "")
            if not horse_url:
                continue

            # 新聞データがあればそれを優先使用（高速化 + 脚質取得の正確化）
            newspaper_entry = newspaper_records.get(horse["name"], {})
            records = newspaper_entry.get("records", []) if isinstance(newspaper_entry, dict) else []
            sire_name = str(newspaper_entry.get("sire_name", "")) if isinstance(newspaper_entry, dict) else ""
            style_char = str(newspaper_entry.get("style_char", "")) if isinstance(newspaper_entry, dict) else ""
            newspaper_mark = str(newspaper_entry.get("newspaper_mark", "")) if isinstance(newspaper_entry, dict) else ""

            if not records:
                records, sire_name = fetch_horse_records(
                    driver,
                    horse_url,
                    history_limit=history_limit,
                    headless=headless,
                )

            if style_char:
                proxy_pos = style_char_to_first_corner_pos(style_char)
                if proxy_pos is not None:
                    for rec in records:
                        if rec.get("first_corner_pos") is None:
                            rec["first_corner_pos"] = proxy_pos
                    for rec in records:
                        if rec.get("passing_text") in (None, "") and rec.get("first_corner_pos") is not None:
                            rec["passing_text"] = str(rec["first_corner_pos"])

            passing_positions = [r["first_corner_pos"] for r in records if r["first_corner_pos"] is not None]
            if style_char:
                style = style_char_to_running_style(style_char)
            else:
                style = infer_running_style(passing_positions)

            horse["style_char_scraped"] = style_char
            entry_styles.append(style)

            horse_results.append({
                "name": horse["name"],
                "link": horse.get("link", ""),
                "records": records,
                "entry_style":style,
                "gate": horse.get("gate"),
                "jockey": horse.get("jockey", ""),
                "win_odds_scraped": horse.get("win_odds_scraped"),
                "place_odds_scraped": horse.get("place_odds_scraped"),
                "sire_name": sire_name,
                "style_char_scraped": horse.get("style_char_scraped", ""),
                "newspaper_mark": newspaper_mark,
            })

        predicted_pace = calc_pace_pressure(entry_styles)
        race_meta.predicted_pace = predicted_pace

        # =========================================================
        # コース傾向バイアス取得
        # =========================================================
        try:
            course_bias = get_course_bias(
                race_meta.target_course,
                race_meta.target_surface,
                race_meta.target_distance,
            )
        except Exception:
            course_bias = {}

        features: List[Dict[str, Any]] = []
        for hr in horse_results:
            feature_dict = build_feature_dict(
               horse_name=str(hr["name"]),
               records=hr["records"],
               race_meta=race_meta,
               predicted_pace=predicted_pace,
               gate=hr.get("gate"),
               jockey_from_entry=hr.get("jockey", ""),
               scraped_win_odds=hr.get("win_odds_scraped"),
               scraped_place_odds=hr.get("place_odds_scraped"),
            )
            # 新聞脚質があれば上書き
            if hr.get("entry_style"):
                feature_dict["running_style"] = hr.get("entry_style")
            # 新聞印AI
            feature_dict["newspaper_mark"] = hr.get("newspaper_mark", "")
            feature_dict["newspaper_mark_index"] = newspaper_mark_to_index(hr.get("newspaper_mark", ""))
            feature_dict["newspaper_entry_style"] = hr.get("entry_style", "unknown")
            features.append(feature_dict)

        for f in features:
            base_score = calc_model_score(f)

            # コース脚質バイアス補正
            style = f.get("running_style")
            bias = course_bias.get(style) if isinstance(course_bias, dict) else None

            if bias is not None:
                base_score = base_score * (1 + (bias - 0.25))

            f["course_bias"] = bias
            f["model_score"] = round(base_score, 6)
        # =========================================================
        # 過去10年レース傾向AI（完全版）
        # =========================================================
        current_race_id = extract_race_id_from_url(shutuba_url)
        past_10y_results: List[Dict[str, Any]] = []
        race_trend_10y: Dict[str, Any] = {}
        trend_match_horses: List[str] = []
        race_history_summary = "過去10年傾向データなし"
        winner_conditions: List[str] = []
        condition_match_horses: List[str] = []
        winner_pattern_ai: Dict[str, Any] = {}

        try:
            if current_race_id:
                past_10y_results = fetch_past_10y_results(driver, current_race_id)
                trend_base = analyze_10y_race_trend(past_10y_results) or {}
                trend_match_horses = match_current_runners_with_10y_trend(features, trend_base) or []

                aux_trend = history_analyze_race_trend(past_10y_results) or {}
                race_history_summary = history_build_race_trend_summary(past_10y_results)

                race_trend_10y = {**trend_base, **aux_trend}

                style_counts = race_trend_10y.get("style", {}) if isinstance(race_trend_10y.get("style", {}), dict) else {}
                pop_counts = race_trend_10y.get("popularity", {}) if isinstance(race_trend_10y.get("popularity", {}), dict) else {}

                if style_counts:
                    top_style = max(style_counts.items(), key=lambda x: x[1])[0]
                    race_trend_10y["style_summary"] = f"{top_style}優勢"
                else:
                    race_trend_10y["style_summary"] = "差し〜先行が優勢"

                if pop_counts:
                    top_pop = max(pop_counts.items(), key=lambda x: x[1])[0]
                    race_trend_10y["popularity_summary"] = f"{top_pop}中心"
                else:
                    fav_ratio = race_trend_10y.get("favorite_ratio")
                    if isinstance(fav_ratio, (int, float)) and fav_ratio >= 0.55:
                        race_trend_10y["popularity_summary"] = "本命寄り"
                    elif isinstance(fav_ratio, (int, float)) and fav_ratio <= 0.30:
                        race_trend_10y["popularity_summary"] = "中穴傾向"
                    else:
                        race_trend_10y["popularity_summary"] = "標準型"

                deterministic_pattern = history_build_winner_condition_ai(past_10y_results) or {}
                winner_conditions = deterministic_pattern.get("conditions", []) or []

                condition_match_horses = []
                condition_scores = deterministic_pattern.get("condition_scores", {}) or {}
                match_score_rows: List[Tuple[str, int]] = []

                for f in features:
                    score = 0
                    name = str(f.get("horse_name") or "")
                    style = str(f.get("running_style") or "")
                    odds = f.get("win_odds")
                    pop = f.get("popularity")
                    distance_fit = float(f.get("distance_fit_index", 0.0) or 0.0)
                    recent_form = float(f.get("recent_form_index", 0.0) or 0.0)

                    corner_rule = str(condition_scores.get("corner") or "")
                    if corner_rule == "4角5番手以内" and style in {"front", "stalker"}:
                        score += 1
                    elif corner_rule == "4角6〜9番手" and style in {"stalker", "closer"}:
                        score += 1
                    elif corner_rule == "4角10番手以降" and style == "closer":
                        score += 1

                    dist_rule = str(condition_scores.get("distance") or "")
                    if dist_rule == "前走1800m以上" and distance_fit >= 0.65:
                        score += 1
                    elif dist_rule == "前走1600m以下" and distance_fit < 0.65:
                        score += 1

                    prev_rank_rule = str(condition_scores.get("prev_rank") or "")
                    if prev_rank_rule == "前走3着以内" and recent_form >= 0.62:
                        score += 1
                    elif prev_rank_rule == "前走4着以下" and recent_form < 0.62:
                        score += 1

                    pop_rule = str(condition_scores.get("popularity") or "")
                    if pop_rule == "1〜4番人気" and isinstance(pop, int) and 1 <= pop <= 4:
                        score += 1
                    elif pop_rule == "5〜9番人気" and isinstance(pop, int) and 5 <= pop <= 9:
                        score += 1
                    elif pop_rule == "10番人気以下" and isinstance(pop, int) and pop >= 10:
                        score += 1
                    elif pop_rule == "1〜4番人気" and not isinstance(pop, int) and isinstance(odds, (int, float)) and odds <= 10:
                        score += 1

                    if score >= 2:
                        match_score_rows.append((name, score))

                match_score_rows.sort(key=lambda x: x[1], reverse=True)

                winner_pattern_ai = {
                    "trend_summary": race_history_summary,
                    "winner_conditions": winner_conditions,
                    "matching_horses": [name for name, _ in match_score_rows[:5]],
                    "confidence": 0.8 if winner_conditions else 0.4,
                    "notes": "deterministic_rule_based_pattern"
                }

                condition_match_horses = []
                for name, score in match_score_rows[:5]:
                    for _ in range(score):
                        condition_match_horses.append(name)

        except Exception:
            past_10y_results = []
            race_trend_10y = {}
            trend_match_horses = []
            race_history_summary = "過去10年傾向データなし"
            winner_conditions = []
            condition_match_horses = []
            winner_pattern_ai = {}

        ml_probs = predict_win_probability_with_model(features, MODEL_FILE)

        if ml_probs is not None:
            probs = ml_probs
            for f in features:
                f["model_type"] = "lightgbm"
        else:
            probs = softmax([f["model_score"] for f in features], temperature=0.25)
            for f in features:
                f["model_type"] = "rule_based"

        for f, p in zip(features, probs):
            f["win_prob"] = round(p, 4)

        pace_probs = race_pace_simulation(features)
        collapse_factor = calc_pace_collapse_risk(features)

        for f in features:
            pace_val = pace_probs.get(f["horse_name"], 0)

            # 展開崩壊AI
            if f.get("running_style") == "closer":
                pace_val *= collapse_factor

            f["pace_simulation_index"] = round(pace_val, 4)
            f["place_prob"] = estimate_place_prob(f["win_prob"])

        for f in features:
            f["fair_win_odds"] = fair_odds(f["win_prob"])
            f["fair_place_odds"] = fair_odds(f["place_prob"])

        for f in features:
            f["win_odds"] = f["win_odds_scraped"]
            f["place_odds"] = f["place_odds_scraped"]
            f["odds_distortion_index"] = calc_odds_distortion(f)
            if f["odds_distortion_index"] >= 1.4:
                f["value_flag"] = "SUPER_VALUE"
            elif f["odds_distortion_index"] >= 1.15:
                f["value_flag"] = "VALUE"
            else:
                f["value_flag"] = "NORMAL"
            f["win_ev"] = calc_expected_value(f["win_prob"], f["win_odds"])
            f["place_ev"] = calc_expected_value(f["place_prob"], f["place_odds"])
            f["win_market_edge"] = calc_market_edge(f["win_prob"], f["win_odds"])
            f["place_market_edge"] = calc_market_edge(f["place_prob"], f["place_odds"])
            f["win_value_label"] = classify_value_label(f["win_market_edge"], f["win_ev"])
            f["place_value_label"] = classify_value_label(f["place_market_edge"], f["place_ev"])
            f["expected_value_score"] = calc_expected_value_score(f)
            f["bet_suitability"] = classify_bet_suitability(f)
            f["danger_favorite_score"] = 0.0
            f["is_danger_favorite"] = False
                    # =========================================================
        # 過去10年レース傾向AI
        # =========================================================
        current_race_id = extract_race_id_from_url(shutuba_url)

        past_10y_results = []
        race_trend_10y = {}
        trend_match_horses = []

        try:
            if current_race_id:
                past_10y_results = fetch_past_10y_results(driver, current_race_id)
                race_trend_10y = analyze_10y_race_trend(past_10y_results)
                trend_match_horses = match_current_runners_with_10y_trend(features, race_trend_10y)
                past_10y_results = fetch_past_10y_results(race_id)
                race_trend_10y = analyze_10y_race_trend(past_10y_results)       
                trend_match_horses = match_current_runners_with_10y_trend(features, race_trend_10y)

# ==============================
# 勝ち馬パターンAI（ChatGPT分析）
# ==============================
                winner_pattern_ai = analyze_winner_patterns_with_chatgpt(
                    race_meta.race_title,
                    past_10y_results,
                    features
                )

                result["winner_pattern_ai"] = winner_pattern_ai
                result["winner_conditions"] = winner_pattern_ai.get("winner_conditions", [])
                result["condition_match_horses"] = winner_pattern_ai.get("matching_horses", [])

        except Exception:
            past_10y_results = []
            race_trend_10y = {}
            trend_match_horses = []
        

        ai_bets = generate_ai_bets(features)
        result = {
            "race_meta": race_meta.__dict__,
            "features": features,
            "ai_bets": ai_bets,
            "value_summary": build_value_summary(features),
            "race_trend_10y": race_trend_10y,
            "past_10y_results": past_10y_results,
            "trend_match_horses": trend_match_horses,
            "race_history_summary": race_history_summary,
            "winner_conditions": winner_conditions,
            "condition_match_horses": condition_match_horses,
            "winner_pattern_ai": winner_pattern_ai,
        }
                # =========================================================
        # 過去10年レース傾向AI
        # =========================================================
        try:
            race_id_match = re.search(r"race_id=(\d+)", shutuba_url)
            if race_id_match:
                race_id = race_id_match.group(1)

                past_results = history_fetch_race_history(driver, race_id)
                race_trend = history_analyze_race_trend(past_results)
                race_summary = history_build_race_trend_summary(past_results)
                winner_conditions = history_build_winner_condition_ai(past_results)
                gpt_pattern_result = analyze_winner_patterns_with_chatgpt(
                    race_title=race_meta.race_title,
                    past_results=past_results,
                    features=features,
                )

                result["past_results"] = past_results
                result["race_trends"] = race_trend if isinstance(race_trend, dict) else {}
                result["race_history_summary"] = gpt_pattern_result.get("trend_summary") or race_summary
                result["winner_conditions"] = gpt_pattern_result.get("winner_conditions") or (
                    winner_conditions if isinstance(winner_conditions, list) else [winner_conditions]
                )
                result["condition_match_horses"] = gpt_pattern_result.get("matching_horses", [])
                result["winner_pattern_confidence"] = gpt_pattern_result.get("confidence", 0.0)
                result["winner_pattern_notes"] = gpt_pattern_result.get("notes", "")
            else:
                result["past_results"] = []
                result["race_trends"] = {}
                result["race_history_summary"] = "race_idを取得できませんでした"
                result["winner_conditions"] = ["条件データなし"]
                result["condition_match_horses"] = []
                result["winner_pattern_confidence"] = 0.0
                result["winner_pattern_notes"] = "race_idなし"

        except Exception as e:
            result["past_results"] = []
            result["race_trends"] = {}
            result["race_history_summary"] = f"過去10年傾向の取得に失敗: {e}"
            result["winner_conditions"] = ["条件データなし"]
            result["condition_match_horses"] = []
            result["winner_pattern_confidence"] = 0.0
            result["winner_pattern_notes"] = str(e)
        # 最終AI指標・期待値・危険人気馬などを再計算
        result = refresh_result_payload(result)

        result = refresh_result_payload(result)
        save_json_cache(race_cache_name, result)
        return result
    
    finally:
        try:
            driver.quit()
        except Exception:
            pass

# =========================================================
# Odds apply
# =========================================================

def apply_simple_odds(
    features: List[Dict[str, Any]],
    win_odds_list: Optional[List[float]] = None,
    place_odds_list: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    win_odds_list = win_odds_list or []
    place_odds_list = place_odds_list or []

    for i, feature in enumerate(features):
        if i < len(win_odds_list):
            feature["win_odds"] = win_odds_list[i]
        if i < len(place_odds_list):
            feature["place_odds"] = place_odds_list[i]

        feature["win_ev"] = calc_expected_value(feature["win_prob"], feature.get("win_odds"))
        feature["place_ev"] = calc_expected_value(feature["place_prob"], feature.get("place_odds"))

        feature["win_market_edge"] = calc_market_edge(feature["win_prob"], feature.get("win_odds"))
        feature["place_market_edge"] = calc_market_edge(feature["place_prob"], feature.get("place_odds"))

        feature["win_value_label"] = classify_value_label(feature["win_market_edge"], feature["win_ev"])
        feature["place_value_label"] = classify_value_label(feature["place_market_edge"], feature["place_ev"])

    features = detect_value_bets(features)
    return features


# =========================================================
# Betting AI
# =========================================================
def generate_ai_bets(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

    """
    AIフォーメーション馬券生成
    """

    if not features:
        return []

    # 勝率順
    sorted_horses = sorted(
        features,
        key=lambda x: x.get("win_prob", 0),
        reverse=True
    )

    bets = []

    # 本命
    if len(sorted_horses) >= 1:
        bets.append({
            "type": "単勝",
            "horse": sorted_horses[0]["horse_name"]
        })

    # ワイド
    if len(sorted_horses) >= 3:
        bets.append({
            "type": "ワイドBOX",
            "horses": [
                sorted_horses[0]["horse_name"],
                sorted_horses[1]["horse_name"],
                sorted_horses[2]["horse_name"]
            ]
        })

    # 馬連
    if len(sorted_horses) >= 2:
        bets.append({
            "type": "馬連",
            "horses": [
                sorted_horses[0]["horse_name"],
                sorted_horses[1]["horse_name"]
            ]
        })

    # 三連複フォーメーション
    if len(sorted_horses) >= 5:

        bets.append({
            "type": "三連複フォーメーション",
            "axis": sorted_horses[0]["horse_name"],
            "others": [
                sorted_horses[1]["horse_name"],
                sorted_horses[2]["horse_name"],
                sorted_horses[3]["horse_name"],
                sorted_horses[4]["horse_name"]
            ]
        })

    return bets


def recommend_bets(features: List[Dict[str, Any]], bankroll: int) -> List[Dict[str, Any]]:
    recommendations: List[Dict[str, Any]] = []

    for f in features:
        win_ev = f.get("win_ev")
        place_ev = f.get("place_ev")

        if win_ev is not None and win_ev >= 1.05 and f.get("win_odds") is not None:
            raw_kelly = kelly_fraction(f["win_prob"], f["win_odds"])
            stake_ratio = min(0.05, raw_kelly * 0.5)
            stake = int(bankroll * stake_ratio // 100 * 100)
            if stake >= 100:
                recommendations.append({
                    "type": "単勝",
                    "horse_name": f["horse_name"],
                    "odds": f["win_odds"],
                    "prob": f["win_prob"],
                    "ev": win_ev,
                    "market_edge": f.get("win_market_edge"),
                    "value_label": f.get("win_value_label"),
                    "stake": stake,
                })

        if place_ev is not None and place_ev >= 1.03 and f.get("place_odds") is not None:
            raw_kelly = kelly_fraction(f["place_prob"], f["place_odds"])
            stake_ratio = min(0.07, raw_kelly * 0.5)
            stake = int(bankroll * stake_ratio // 100 * 100)
            if stake >= 100:
                recommendations.append({
                    "type": "複勝",
                    "horse_name": f["horse_name"],
                    "odds": f["place_odds"],
                    "prob": f["place_prob"],
                    "ev": place_ev,
                    "market_edge": f.get("place_market_edge"),
                    "value_label": f.get("place_value_label"),
                    "stake": stake,
                })

    recommendations.sort(key=lambda x: x["ev"], reverse=True)
    return recommendations


# =========================================================
# Print
# =========================================================

def print_analysis(result: Dict[str, Any]) -> None:
    print("\n===== レース情報 =====")
    for k, v in result["race_meta"].items():
        print(f"{k}: {v}")

    print("\n===== 勝率AI =====")
    for i, feature in enumerate(result.get("features", []), start=1):
        print("\n------------------------")
        print(f"{i}位")
        print("horse_name:", feature["horse_name"])
        print("model_type:", feature["model_type"])
        print("model_score:", feature["model_score"])
        print("win_prob:", feature["win_prob"])
        print("place_prob:", feature["place_prob"])
        print("fair_win_odds:", feature["fair_win_odds"])
        print("fair_place_odds:", feature["fair_place_odds"])

        print("distance_course_suitability_index:", feature["distance_course_suitability_index"])
        print("recent_form_index:", feature["recent_form_index"])
        print("ground_match_index:", feature["ground_match_index"])

        print("running_style:", feature["running_style"])
        print("newspaper_entry_style:", feature.get("newspaper_entry_style"))
        print("newspaper_mark:", feature.get("newspaper_mark"))
        print("newspaper_mark_index:", feature.get("newspaper_mark_index"))
        print("pace_bias_index:", feature["pace_bias_index"])
        print("style_suitability_index:", feature["style_suitability_index"])
        print("pace_advantage:", feature["pace_advantage"])

        print("jockey_index:", feature["jockey_index"])
        print("gate:", feature["gate"])
        print("gate_index:", feature["gate_index"])
        print("last3f_index:", feature["last3f_index"])
        print("lap_suitability_index:", feature["lap_suitability_index"])

        print("trend_index:", feature["trend_index"])
        print("consistency_index:", feature["consistency_index"])
        print("distance_fit_index:", feature["distance_fit_index"])
        print("race_level_index:", feature["race_level_index"])
        print("pace_style_label:", feature.get("pace_style_label"))
        print("pace_simulation_index:", feature.get("pace_simulation_index"))
        print("ai_power_index:", feature.get("ai_power_index"))
        print("value_index:", feature.get("value_index"))
        print("montecarlo_win_prob:", feature.get("montecarlo_win_prob"))
        print("sire_name:", feature.get("sire_name"))
        print("bloodline_index:", feature.get("bloodline_index"))
        print("track_bias_index:", feature.get("track_bias_index")) 

        print("win_odds:", feature.get("win_odds"))
        print("win_ev:", feature.get("win_ev"))
        print("win_market_edge:", feature.get("win_market_edge"))
        print("win_value_label:", feature.get("win_value_label"))

        print("place_odds:", feature.get("place_odds"))
        print("place_ev:", feature.get("place_ev"))
        print("place_market_edge:", feature.get("place_market_edge"))
        print("place_value_label:", feature.get("place_value_label"))


def print_recommendations(recommendations: List[Dict[str, Any]]) -> None:
    print("\n===== 馬券AI（推奨） =====")
    if not recommendations:
        print("期待値条件を満たす推奨馬券はありません。")
        return

    for rec in recommendations:
        print(
            f"{rec['type']} | {rec['horse_name']} | odds={rec['odds']} | "
            f"prob={rec['prob']} | ev={rec['ev']} | edge={rec['market_edge']} | "
            f"label={rec['value_label']} | 推奨金額={rec['stake']}円"
        )

def calc_race_level_index_legacy(records):
    """
    過去レースのクラスレベルから簡易レースレベル指数を算出
    """
    if not records:
        return 0.5

    class_values = [r.get("class_index", 0.3) for r in records if r.get("class_index") is not None]

    if not class_values:
        return 0.5

    avg_class = mean(class_values)

    # 0.25〜1.0程度の値を0〜1スケールへ軽く正規化
    normalized = (avg_class - 0.25) / (1.0 - 0.25)
    normalized = max(0.0, min(1.0, normalized))

    return round(normalized, 4)


BLOODLINE_SURFACE_DISTANCE_BIAS: Dict[Tuple[str, str], Dict[str, float]] = {
    ("芝", "1200_1600"): {
        "ディープインパクト": 1.05,
        "ロードカナロア": 1.06,
        "ダイワメジャー": 1.05,
        "ハーツクライ": 1.03,
        "モーリス": 1.05,
        "エピファネイア": 1.03,
        "キズナ": 1.03,
    },
    ("芝", "1600_2000"): {
        "ディープインパクト": 1.08,
        "ハーツクライ": 1.06,
        "エピファネイア": 1.05,
        "キズナ": 1.05,
        "モーリス": 1.03,
        "ドゥラメンテ": 1.06,
        "キングカメハメハ": 1.04,
    },
    ("芝", "ge_2000"): {
        "ディープインパクト": 1.07,
        "ハーツクライ": 1.08,
        "キズナ": 1.05,
        "ドゥラメンテ": 1.05,
        "ステイゴールド": 1.07,
        "オルフェーヴル": 1.06,
    },
    ("ダ", "1200_1600"): {
        "ヘニーヒューズ": 1.08,
        "ロードカナロア": 1.03,
        "ドレフォン": 1.07,
        "シニスターミニスター": 1.09,
        "パイロ": 1.06,
        "ホッコータルマエ": 1.05,
    },
    ("ダ", "1600_2000"): {
        "ドレフォン": 1.05,
        "ホッコータルマエ": 1.08,
        "シニスターミニスター": 1.05,
        "ルーラーシップ": 1.03,
        "キズナ": 1.02,
        "エスポワールシチー": 1.06,
    },
}

TRACK_BIAS_TABLE: Dict[Tuple[str, str, str], Dict[str, float]] = {
    ("中山", "芝", "良"): {
        "front": 1.05,
        "stalker": 1.03,
        "closer": 0.95,
        "inner_gate": 1.04,
        "outer_gate": 0.97,
    },
    ("中山", "芝", "稍重"): {
        "front": 1.06,
        "stalker": 1.03,
        "closer": 0.93,
        "inner_gate": 1.03,
        "outer_gate": 0.98,
    },
    ("東京", "芝", "良"): {
        "front": 0.98,
        "stalker": 1.01,
        "closer": 1.04,
        "inner_gate": 1.00,
        "outer_gate": 1.00,
    },
    ("阪神", "芝", "良"): {
        "front": 1.02,
        "stalker": 1.03,
        "closer": 0.99,
        "inner_gate": 1.02,
        "outer_gate": 0.99,
    },
    ("京都", "芝", "良"): {
        "front": 1.03,
        "stalker": 1.02,
        "closer": 0.98,
        "inner_gate": 1.02,
        "outer_gate": 0.99,
    },
    ("中京", "芝", "良"): {
        "front": 0.99,
        "stalker": 1.02,
        "closer": 1.01,
        "inner_gate": 1.00,
        "outer_gate": 1.00,
    },
    ("小倉", "芝", "良"): {
        "front": 1.06,
        "stalker": 1.03,
        "closer": 0.93,
        "inner_gate": 1.04,
        "outer_gate": 0.96,
    },
    ("中山", "ダ", "良"): {
        "front": 1.07,
        "stalker": 1.03,
        "closer": 0.92,
        "inner_gate": 1.03,
        "outer_gate": 0.98,
    },
    ("東京", "ダ", "良"): {
        "front": 1.01,
        "stalker": 1.03,
        "closer": 0.98,
        "inner_gate": 0.99,
        "outer_gate": 1.01,
    },
}

def parse_sire_name(driver: webdriver.Chrome) -> str:
    """
    netkeiba馬ページから父名をざっくり取得する
    """
    try:
        text = safe_page_source(driver)
        patterns = [
            r"父[:：]\s*([^<\n]+)",
            r"Father[:：]?\s*([^<\n]+)",
        ]
        for pattern in patterns:
            m = re.search(pattern, text)
            if m:
                return m.group(1).strip()
    except Exception:
        pass
    return ""


def calc_bloodline_index(
    sire_name: str,
    target_surface: Optional[str],
    target_distance: Optional[int],
) -> float:
    if not sire_name or not target_surface or target_distance is None:
        return 1.0

    band = distance_band(target_distance)
    table = BLOODLINE_SURFACE_DISTANCE_BIAS.get((target_surface, band), {})
    return round(table.get(sire_name, 1.0), 4)


def calc_track_bias_index(
    course: str,
    surface: Optional[str],
    ground: str,
    running_style: str,
    gate: Optional[int],
) -> float:
    if not surface:
        return 1.0

    bias = TRACK_BIAS_TABLE.get((course, surface, ground))
    if not bias:
        return 1.0

    style_factor = bias.get(running_style, 1.0)

    gate_factor = 1.0
    if gate is not None:
        if gate <= 3:
            gate_factor = bias.get("inner_gate", 1.0)
        elif gate >= 7:
            gate_factor = bias.get("outer_gate", 1.0)

    return round((style_factor * gate_factor) ** 0.5, 4)


def fetch_horse_sire_name(driver: webdriver.Chrome, horse_url: str) -> str:
    """
    馬ページから父名を取得
    """
    try:
        driver = safe_get(driver, horse_url, headless=False, retries=1)
        random_sleep(0.8, 1.4)
        if "HTTP ERROR 400" in safe_page_source(driver):
            return ""
        return parse_sire_name(driver)
    except Exception:
        return ""


def calc_model_score_v2(feature_dict: Dict[str, Any]) -> float:
    """
    血統AI・馬場バイアスAI込みの拡張版スコア
    """
    dc = feature_dict["distance_course_suitability_index"]
    style = feature_dict["style_suitability_index"]
    recent = feature_dict["recent_form_index"]
    ground = feature_dict["ground_match_index"]
    pace_bias = feature_dict["pace_bias_index"]
    pace_sim = feature_dict.get("pace_simulation_index", 0.5)
    jockey = feature_dict["jockey_index"]
    gate = feature_dict["gate_index"]
    last3f = feature_dict["last3f_index"]
    pace_adv = feature_dict["pace_advantage"]
    lap = feature_dict["lap_suitability_index"]
    trend = feature_dict["trend_index"]
    consistency = feature_dict["consistency_index"]
    distance_fit = feature_dict["distance_fit_index"]
    bloodline = feature_dict.get("bloodline_index", 1.0)
    track_bias = feature_dict.get("track_bias_index", 1.0)
    race_level = feature_dict.get("race_level_index", 0.5)
    newspaper = feature_dict.get("newspaper_mark_index", 1.0)

    sample_adj = min(1.0, feature_dict.get("distance_course_sample_size", 0) / 3.0)

    model_score = (
        0.16 * dc
        + 0.12 * style
        + 0.15 * recent
        + 0.05 * ground
        + 0.06 * pace_bias
        + 0.06 * last3f
        + 0.06 * lap
        + 0.06 * jockey
        + 0.04 * gate
        + 0.04 * pace_adv
        + 0.06 * trend
        + 0.04 * consistency
        + 0.04 * distance_fit
        + 0.03 * bloodline
        + 0.03 * track_bias
        + 0.05 * pace_sim
        + 0.06 * race_level
    ) * (0.75 + 0.25 * sample_adj)

    return round(model_score * newspaper, 6)

def build_feature_dict_v2(
    horse_name: str,
    records: List[Dict[str, Any]],
    race_meta: RaceMeta,
    predicted_pace: str,
    entry_style: str = "unknown",
    gate: Optional[int] = None,
    jockey_from_entry: str = "",
    scraped_win_odds: Optional[float] = None,
    scraped_place_odds: Optional[float] = None,
    sire_name: str = "",
    manual_styles: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    # UIなどから手動入力された脚質があれば優先する
    if manual_styles and horse_name in manual_styles:
        entry_style = manual_styles[horse_name]
    feature = build_feature_dict(
        horse_name=horse_name,
        records=records,
        race_meta=race_meta,
        predicted_pace=predicted_pace,
        gate=gate,
        jockey_from_entry=jockey_from_entry,
        scraped_win_odds=scraped_win_odds,
        scraped_place_odds=scraped_place_odds,
    )

    # 新聞脚質を優先
    if entry_style and entry_style != "unknown":
        feature["running_style"] = entry_style

        style_bias_map: Dict[str, Dict[str, float]] = {
            "slow": {"front": 1.00, "stalker": 0.80, "closer": 0.55, "unknown": 0.50},
            "medium": {"front": 0.85, "stalker": 0.90, "closer": 0.80, "unknown": 0.50},
            "fast": {"front": 0.55, "stalker": 0.85, "closer": 1.00, "unknown": 0.50},
            "very_fast": {"front": 0.45, "stalker": 0.80, "closer": 1.05, "unknown": 0.50},
        }

        style_bias = style_bias_map.get(predicted_pace, {}).get(entry_style, 0.5)
        ranks = [r["rank"] for r in records if r.get("rank") is not None]
        avg_rank = safe_int_mean(ranks)
        style_perf = normalize_rank_index(avg_rank) if avg_rank is not None else 0.0
        style_index = 0.6 * style_bias + 0.4 * style_perf

        feature["style_sample_size"] = max(1, feature.get("style_sample_size", 0))
        feature["pace_bias_index"] = round(style_bias, 4)
        feature["style_perf_index"] = round(style_perf, 4)
        feature["style_suitability_index"] = round(style_index, 4)
        feature["pace_advantage"] = calc_pace_advantage(entry_style, predicted_pace)
    

    feature["sire_name"] = sire_name
    feature["bloodline_index"] = calc_bloodline_index(
        sire_name=sire_name,
        target_surface=race_meta.target_surface,
        target_distance=race_meta.target_distance,
    )
    feature["newspaper_entry_style"] = entry_style
    feature["race_level_index"] = calc_race_level_index(records)
    feature["track_bias_index"] = calc_track_bias_index(
        course=race_meta.target_course,
        surface=race_meta.target_surface,
        ground=race_meta.target_ground,
        running_style=feature.get("running_style", "unknown"),
        gate=gate,
    )

    return feature


def apply_bloodline_and_track_bias_to_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    既存resultに後付けで血統・馬場バイアス列を安全に付与
    """
    race_meta_dict = result.get("race_meta", {})
    target_surface = race_meta_dict.get("target_surface")
    target_distance = race_meta_dict.get("target_distance")
    target_course = race_meta_dict.get("target_course", "")
    target_ground = race_meta_dict.get("target_ground", "")

    for feature in result.get("features", []):
        sire_name = feature.get("sire_name", "")
        if "bloodline_index" not in feature:
            feature["bloodline_index"] = calc_bloodline_index(
                sire_name=sire_name,
                target_surface=target_surface,
                target_distance=target_distance,
            )
        if "track_bias_index" not in feature:
            feature["track_bias_index"] = calc_track_bias_index(
                course=target_course,
                surface=target_surface,
                ground=target_ground,
                running_style=feature.get("running_style", "unknown"),
                gate=feature.get("gate"),
            )

        feature["model_score"] = calc_model_score_v2(feature)

    probs = softmax([f["model_score"] for f in result.get("features", [])], temperature=0.25)
    for f, p in zip(result.get("features", []), probs):
        f["win_prob"] = round(p, 4)
        f["place_prob"] = estimate_place_prob(f["win_prob"])
        f["fair_win_odds"] = fair_odds(f["win_prob"])
        f["fair_place_odds"] = fair_odds(f["place_prob"])

        f["win_ev"] = calc_expected_value(f["win_prob"], f.get("win_odds"))
        f["place_ev"] = calc_expected_value(f["place_prob"], f.get("place_odds"))
        f["win_market_edge"] = calc_market_edge(f["win_prob"], f.get("win_odds"))
        f["place_market_edge"] = calc_market_edge(f["place_prob"], f.get("place_odds"))
        f["win_value_label"] = classify_value_label(f["win_market_edge"], f["win_ev"])
        f["place_value_label"] = classify_value_label(f["place_market_edge"], f["place_ev"])
        f["pace_style_label"] = style_to_japanese(f.get("running_style", "unknown"))
        f["ai_power_index"] = calc_ai_power_index(f)
        f["radar_payload"] = build_radar_payload(f)
    return refresh_result_payload(result)

# =========================================================
# Race Level AI
# =========================================================

def calc_race_level_index(records: List[Dict[str, Any]]) -> float:
    """
    過去レースのレベルを簡易評価する
    G1 > G2 > G3 > OP > 条件戦
    """

    if not records:
        return 0.5

    level_score = 0
    count = 0

    for r in records:

        race_name = str(r.get("race_name", ""))

        if "G1" in race_name:
            level = 1.0
        elif "G2" in race_name:
            level = 0.9
        elif "G3" in race_name:
            level = 0.8
        elif "オープン" in race_name or "OP" in race_name:
            level = 0.7
        elif "3勝" in race_name:
            level = 0.6
        elif "2勝" in race_name:
            level = 0.5
        elif "1勝" in race_name:
            level = 0.4
        else:
            level = 0.5

        level_score += level
        count += 1

    return round(level_score / count, 4)

# =========================================================
# Entry
# =========================================================

if __name__ == "__main__":
    print("1: 通常分析")
    print("2: LightGBMモデル学習（keiba_training_data.csv が必要）")
    mode = input("モードを選んでください (1 or 2): ").strip()

    if mode == "2":
        ok = train_lightgbm_model(TRAINING_CSV, MODEL_FILE)
        print("学習完了" if ok else "学習失敗またはスキップ")
        raise SystemExit

    print("\n==============================")
    print(" netkeiba レースURL入力")
    print("==============================")
    print("① 出馬表ページURL または 競馬新聞URL を貼ってください")
    print("例1:")
    print("https://race.netkeiba.com/race/shutuba.html?race_id=202406030811")
    print("例2:")
    print("https://race.netkeiba.com/race/newspaper.html?race_id=202406030811")
    print("")
    print("※ どちらか1つでOKです")
    print("※ AIが 出馬表(shutuba) / 新聞(newspaper) を自動補完します")
    print("")

    race_url = input("netkeiba URL: ").strip()

    if not race_url.startswith("http") or ("shutuba" not in race_url and "newspaper" not in race_url):
        print("\n[エラー]")
        print("出馬表(shutuba.html) または 競馬新聞(newspaper.html) のURLを入力してください")
        raise SystemExit

    result = analyze_race(race_url, history_limit=HISTORY_LIMIT_DEFAULT, headless=False)
    if result.get("features"):
        result = apply_bloodline_and_track_bias_to_result(result)
    else:
        result = refresh_result_payload(result)
    print_analysis(result)

    if result.get("error") == "HTTP400_BLOCKED":
        print("\n[ERROR] netkeiba側でアクセスがブロックされました。")
        print("[INFO] Version15 ではキャッシュを追加し、race_url の再取得回数をさらに減らしています。")
        print("[INFO] 少し時間を空けて再実行するか、通常のChromeで先にnetkeibaを開いてから試してください。")
        raise SystemExit

    if result.get("error") == "NO_HORSES_FOUND":
        print("\n[ERROR] 出馬表テーブルを取得できませんでした。")
        raise SystemExit

    print("\n===== オッズ入力（任意・馬名不要） =====")
    print("月曜などでオッズ未確定なら空EnterでOKです。")
    print("出馬表順にカンマ区切りで入力してください。")
    print("例: 3.2,5.8,12.4,7.1")

    win_odds_raw = input("\n単勝オッズ: ").strip()
    place_odds_raw = input("複勝オッズ: ").strip()
    bankroll_raw = input("軍資金（円、例 3000）: ").strip()

    win_odds_list: List[float] = []
    place_odds_list: List[float] = []

    try:
        win_odds_list = parse_comma_odds_input(win_odds_raw)
    except Exception:
        if win_odds_raw:
            print("単勝オッズの形式が不正です。手入力単勝はスキップします。")

    try:
        place_odds_list = parse_comma_odds_input(place_odds_raw)
    except Exception:
        if place_odds_raw:
            print("複勝オッズの形式が不正です。手入力複勝はスキップします。")

    result["features"] = apply_simple_odds(
        result["features"],
        win_odds_list=win_odds_list,
        place_odds_list=place_odds_list,
    )
    result = refresh_result_payload(result)

    bankroll = 3000
    if bankroll_raw.isdigit():
        bankroll = int(bankroll_raw)

    print_analysis(result)
    print("\n===== AI信頼度 =====")
    print(result.get("ai_confidence"))
    print("\n===== 穴馬AI =====")
    for item in result.get("dark_horses", []):
        print(item.get("horse_name"), item.get("win_prob"), item.get("value_index"))
    print("\n===== 危険人気馬AI =====")
    for item in result.get("danger_favorites", []):
        print(item.get("horse_name"), item.get("danger_gap"))
    print("\n===== GPT解説 =====")
    print(result.get("ai_comment", ""))
    recommendations = recommend_bets(result.get("features", []), bankroll=bankroll)
    print_recommendations(recommendations)
    
