"""
dividend_scraper.py
netkeiba 過去レース結果スクレイパー（Selenium 不要）

責務:
- requests + BeautifulSoup で netkeiba 結果ページを取得
- 着順・払戻金額・全走者データを返す
- Selenium ドライバ不要なのでバックグラウンドで実行可能
"""
from __future__ import annotations

import random
import re
import time
from datetime import date, timedelta
from typing import Any, Dict, Iterator, List, Optional

import requests
from bs4 import BeautifulSoup

from race_history_ai import infer_style


# =========================================================
# 定数
# =========================================================

# ブラウザに偽装したヘッダー（botと判定されにくくする）
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,"
        "application/signed-exchange;v=b3;q=0.7"
    ),
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}
_RESULT_URL = "https://race.netkeiba.com/race/result.html?race_id={race_id}"

# セッション（Cookie を自動保持し、コネクションを再利用する）
_session = requests.Session()
_session.headers.update(_HEADERS)


# =========================================================
# 内部ユーティリティ
# =========================================================

def _fetch_html(url: str, retries: int = 3) -> Optional[str]:
    """汎用 HTML 取得（レース一覧など正常な400が起こりうるページ用）"""
    for i in range(retries):
        try:
            r = _session.get(url, timeout=15)
            r.encoding = "euc-jp"
            if r.status_code == 400 and len(r.content) == 0:
                return None  # 非開催日など正常な空レスポンス
            return r.text
        except Exception:
            if i < retries - 1:
                time.sleep(random.uniform(1.5, 3.0))
    return None


_consecutive_blocks = 0
_BLOCK_THRESHOLD    = 3
# 指数バックオフ: 1回目30秒 → 2回目2分 → 3回目10分 → 4回目以降20分
_BACKOFF_SCHEDULE   = [30, 120, 600, 1200]

# 短い HTML でも「存在する」と判断するための最小バイト数
# netkeiba の「レースなし」ページは ~400 bytes、正常ページは数十KB
_MIN_VALID_CONTENT_BYTES = 2000


def _fetch_result_html(url: str, retries: int = 3) -> Optional[str]:
    """レース結果・馬情報ページ用（IP制限を指数バックオフで自動回復）

    400 empty → IPブロック or レース不存在の2パターンがある。
    区別方法:
      - 連続して起きている → IPブロックの可能性が高い → バックオフ
      - 初回から発生 & 中身が小さい → レース不存在 → 即 None 返し
    """
    global _consecutive_blocks
    for i in range(retries):
        try:
            r = _session.get(url, timeout=15)
            r.encoding = "euc-jp"
            is_empty_response = (
                r.status_code in (400, 404)
                or (r.status_code == 200 and len(r.content) < _MIN_VALID_CONTENT_BYTES)
            )
            if is_empty_response:
                _consecutive_blocks += 1
                # 連続ブロックがしきい値未満 & 初回失敗 → レース不存在とみなし即 None
                if _consecutive_blocks < _BLOCK_THRESHOLD and i == 0:
                    _consecutive_blocks = max(0, _consecutive_blocks - 1)
                    return None
                if _consecutive_blocks >= _BLOCK_THRESHOLD:
                    wait = _BACKOFF_SCHEDULE[
                        min(_consecutive_blocks - _BLOCK_THRESHOLD,
                            len(_BACKOFF_SCHEDULE) - 1)
                    ]
                    print(
                        f"\n  [警告] IP制限を検出 ({_consecutive_blocks}回連続)。"
                        f"{wait}秒待機後に再開します...",
                        flush=True,
                    )
                    time.sleep(wait)
                    _consecutive_blocks = 0
                elif i < retries - 1:
                    time.sleep(random.uniform(3.0, 6.0))
                continue
            _consecutive_blocks = 0
            return r.text
        except Exception:
            if i < retries - 1:
                time.sleep(random.uniform(1.5, 3.0))
    return None


def _parse_int(text: str) -> Optional[int]:
    t = (text or "").strip()
    return int(t) if t.isdigit() else None


def _parse_float(text: str) -> Optional[float]:
    try:
        return float((text or "").replace(",", "").strip())
    except Exception:
        return None


def _parse_payout_amounts(text: str) -> List[int]:
    """'190円700円390円' → [190, 700, 390]"""
    return [int(a.replace(",", "")) for a in re.findall(r"([\d,]+)円", text)]


def _race_id_to_date(race_id: str, soup: BeautifulSoup) -> str:
    """ページタイトルまたは race_id から開催日を推定。"""
    # ページ内の日付テキストを探す
    for sel in (".RaceData01", ".RaceList_DataTitle", "title"):
        elem = soup.select_one(sel)
        if elem:
            text = elem.get_text()
            m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
            if m:
                return f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"
    # fallback: race_id の先頭4桁が年
    year = race_id[:4] if len(race_id) >= 4 else "2024"
    return f"{year}-01-01"


# =========================================================
# メイン関数
# =========================================================

def scrape_race_result(race_id: str) -> Optional[Dict[str, Any]]:
    """
    指定 race_id の過去レース結果を取得する。

    Parameters
    ----------
    race_id : "202406050812" 形式の12桁レースID

    Returns
    -------
    {
        "race_id":      str,
        "race_date":    str,          # "YYYY-MM-DD"
        "race_name":    str,
        "runners": [
            {
                "rank":         int,
                "gate":         int,
                "horse_no":     int,
                "horse_name":   str,
                "horse_url":    str,  # https://db.netkeiba.com/horse/XXXXXXXX
                "age":          int,
                "sex":          str,
                "jockey_weight": float,
                "jockey":       str,
                "popularity":   int,
                "win_odds":     float,
                "last_3f":      float | None,
                "passing":      str,
                "running_style": str,  # "front"/"stalker"/"closer"/"unknown"
            },
            ...
        ],
        "dividends": {
            "単勝":  int,
            "複勝":  List[int],
            "馬連":  int | None,
            "ワイド": List[int],
            "馬単":  int | None,
            "3連複": int | None,
            "3連単": int | None,
        },
        "finish_order": List[str],   # 馬名 1着→2着→3着...
    }
    None なら取得失敗
    """
    url = _RESULT_URL.format(race_id=race_id)
    html = _fetch_result_html(url)
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")
    race_date = _race_id_to_date(race_id, soup)

    # レース名
    race_name = ""
    for sel in (".RaceName", ".RaceMainName", "h1"):
        elem = soup.select_one(sel)
        if elem:
            race_name = elem.get_text(strip=True)
            break

    # 馬場状態 / 距離 / 馬場種別
    track_condition: Optional[str] = None
    surface: Optional[str] = None      # "芝" or "ダート"
    distance: Optional[int] = None     # metres e.g. 1600
    _rd01 = soup.select_one(".RaceData01")
    if _rd01:
        _rd01_text = _rd01.get_text()
        _tc_m = re.search(r"[／/]\s*(良|稍重|重|不良)", _rd01_text)
        if _tc_m:
            track_condition = _tc_m.group(1)
        # 馬場種別（芝 / ダート）
        if "芝" in _rd01_text:
            surface = "芝"
        elif re.search(r"ダ[ー・]?[トレ]?", _rd01_text):
            surface = "ダート"
        # 距離
        _dist_m = re.search(r"(\d{3,4})m", _rd01_text)
        if _dist_m:
            distance = int(_dist_m.group(1))

    # ── 走者データ ──────────────────────────────────────────
    runners: List[Dict[str, Any]] = []
    finish_order: List[str] = []

    table = soup.find("table", class_="RaceTable01")
    if not table:
        return None

    for row in table.find_all("tr"):
        tds = row.find_all("td")
        if len(tds) < 11:
            continue

        rank          = _parse_int(tds[0].get_text(strip=True))
        gate          = _parse_int(tds[1].get_text(strip=True))
        horse_no      = _parse_int(tds[2].get_text(strip=True))
        horse_cell    = tds[3]
        horse_name    = horse_cell.get_text(strip=True)
        horse_link    = horse_cell.find("a")
        horse_url     = horse_link.get("href", "") if horse_link else ""
        sex_age_text  = tds[4].get_text(strip=True)
        jockey_weight = _parse_float(tds[5].get_text(strip=True))
        jockey        = tds[6].get_text(strip=True)
        popularity    = _parse_int(tds[9].get_text(strip=True))
        win_odds      = _parse_float(tds[10].get_text(strip=True))
        last_3f       = _parse_float(tds[11].get_text(strip=True)) if len(tds) > 11 else None
        passing       = tds[12].get_text(strip=True) if len(tds) > 12 else ""

        age_m = re.search(r"(\d+)", sex_age_text)
        age   = int(age_m.group(1)) if age_m else None
        sex   = sex_age_text[0] if sex_age_text else ""

        if rank is None or not horse_name:
            continue

        runners.append({
            "rank":          rank,
            "gate":          gate,
            "horse_no":      horse_no,
            "horse_name":    horse_name,
            "horse_url":     horse_url,
            "age":           age,
            "sex":           sex,
            "jockey_weight": jockey_weight,
            "jockey":        jockey,
            "popularity":    popularity,
            "win_odds":      win_odds,
            "last_3f":       last_3f,
            "passing":       passing,
            "running_style": infer_style(passing),
        })
        finish_order.append(horse_name)

    if not runners:
        return None

    # 着順順にソート（念のため）
    runners.sort(key=lambda r: r["rank"] if r["rank"] is not None else 99)
    finish_order = [r["horse_name"] for r in runners]

    # ── 払戻データ ──────────────────────────────────────────
    BET_TYPE_MAP = {
        "単勝":  "単勝",
        "複勝":  "複勝",
        "枠連":  "枠連",
        "馬連":  "馬連",
        "ワイド": "ワイド",
        "馬単":  "馬単",
        "3連複": "3連複",
        "3連単": "3連単",
    }
    dividends: Dict[str, Any] = {}

    for pay_table in soup.find_all("table", class_="Payout_Detail_Table"):
        for row in pay_table.find_all("tr"):
            th = row.find("th")
            if not th:
                continue
            bet_type = th.get_text(strip=True)
            if bet_type not in BET_TYPE_MAP:
                continue
            tds_p = row.find_all("td")
            if len(tds_p) < 2:
                continue
            payout_text = tds_p[1].get_text(strip=True)
            amounts = _parse_payout_amounts(payout_text)

            if bet_type in ("単勝", "枠連", "馬連", "馬単", "3連複", "3連単"):
                dividends[bet_type] = amounts[0] if amounts else None
            else:  # 複勝・ワイド は複数
                dividends[bet_type] = amounts

    return {
        "race_id":         race_id,
        "race_date":       race_date,
        "race_name":       race_name,
        "track_condition": track_condition,
        "surface":         surface,
        "distance":        distance,
        "runners":         runners,
        "dividends":       dividends,
        "finish_order":    finish_order,
    }


def get_win_dividend(race_id: str) -> Optional[int]:
    """単勝配当だけを返す簡易版。"""
    data = scrape_race_result(race_id)
    if not data:
        return None
    return data.get("dividends", {}).get("単勝")


# =========================================================
# 日付範囲からレースID一括取得
# =========================================================

_RACE_LIST_URL = "https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date}"

# 開催場所コード（netkeiba の venue_id → 表示名）
VENUE_CODE_MAP: Dict[str, str] = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}


def fetch_race_ids_by_date(
    date_str: str,
    venue_codes: Optional[List[str]] = None,
) -> List[str]:
    """
    指定日（YYYYMMDD）の全レースIDを返す。

    Parameters
    ----------
    date_str     : "20240601" 形式
    venue_codes  : 絞り込む開催場所コードリスト（例: ["05","09"]）。None=全場

    Returns
    -------
    race_id のリスト（12桁）
    """
    url  = _RACE_LIST_URL.format(date=date_str)
    html = _fetch_html(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    race_ids: List[str] = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = re.search(r"/race/(\d{12})/", href)
        if not m:
            m = re.search(r"race_id=(\d{12})", href)
        if not m:
            continue
        rid = m.group(1)
        if venue_codes and rid[4:6] not in venue_codes:
            continue
        if rid not in race_ids:
            race_ids.append(rid)

    # フォールバック:
    # race_list_sub のHTMLによっては race_id が a[href] ではなく
    # script 内にのみ含まれる場合があるため、HTML全文からも抽出する。
    for rid in re.findall(r"(?:/race/|race_id=)(\d{12})", html):
        if venue_codes and rid[4:6] not in venue_codes:
            continue
        if rid not in race_ids:
            race_ids.append(rid)

    return race_ids


def fetch_race_ids_by_name(
    race_name: str,
    start_year: int,
    end_year: int,
    search_months: Optional[List[int]] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    レース名（部分一致）で過去N年分のレースIDを収集する。

    日付ベースのレース一覧ページからリンクテキストでフィルタする方式。
    search_months を指定すると対象月に絞るため大幅に高速化できる。
    例: 阪神大賞典は3月開催なので search_months=[3] にすると
        31日 × 10年 = 310リクエスト ≈ 3分で完了。

    Parameters
    ----------
    race_name     : 検索するレース名（例: "阪神大賞典"）部分一致
    start_year    : 検索開始年（例: 2015）
    end_year      : 検索終了年（例: 2024）
    search_months : 対象月リスト（例: [3] or [5,6]）。None=全月
    progress_callback: fn(current, total, date_str) — 進捗通知用

    Returns
    -------
    {
        "race_ids":    List[str],
        "n_races":     int,
        "by_year":     {year_str: List[str]},
    }
    """
    months = search_months or list(range(1, 13))

    # 対象日付リストを構築
    target_dates: List[str] = []
    for year in range(start_year, end_year + 1):
        for month in months:
            import calendar
            _, days_in_month = calendar.monthrange(year, month)
            for day in range(1, days_in_month + 1):
                target_dates.append(f"{year:04d}{month:02d}{day:02d}")

    all_ids: List[str] = []
    by_year: Dict[str, List[str]] = {}
    total = len(target_dates)

    for i, ds in enumerate(target_dates):
        url  = _RACE_LIST_URL.format(date=ds)
        html = _fetch_html(url)
        if html:
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                # リンクテキストにレース名が含まれるものを抽出
                link_text = a.get_text(strip=True)
                if race_name not in link_text:
                    continue
                href = a["href"]
                m = re.search(r"/race/(\d{12})/", href)
                if not m:
                    m = re.search(r"race_id=(\d{12})", href)
                if not m:
                    continue
                rid = m.group(1)
                year_str = ds[:4]
                if rid not in all_ids:
                    all_ids.append(rid)
                    by_year.setdefault(year_str, [])
                    if rid not in by_year[year_str]:
                        by_year[year_str].append(rid)

        if progress_callback:
            progress_callback(i + 1, total, ds)
        time.sleep(random.uniform(1.0, 2.0))

    return {
        "race_ids": all_ids,
        "n_races":  len(all_ids),
        "by_year":  by_year,
    }


def fetch_race_ids_by_date_range(
    start_date: str,
    end_date:   str,
    venue_codes: Optional[List[str]] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    日付範囲のレースIDを一括取得する。

    Parameters
    ----------
    start_date       : "YYYYMMDD" 形式
    end_date         : "YYYYMMDD" 形式
    venue_codes      : 絞り込み開催場所コードリスト（None=全場）
    progress_callback: fn(current, total, date_str) — 進捗通知用

    Returns
    -------
    {
        "race_ids":   List[str],  # 重複なし
        "n_dates":    int,        # 処理日数
        "n_races":    int,        # 取得レース数
        "date_counts": {date_str: int},  # 日付ごとのレース数
    }
    """
    s = date(int(start_date[:4]), int(start_date[4:6]), int(start_date[6:8]))
    e = date(int(end_date[:4]),   int(end_date[4:6]),   int(end_date[6:8]))

    all_ids:     List[str]       = []
    date_counts: Dict[str, int]  = {}
    current      = s
    total_days   = (e - s).days + 1
    day_idx      = 0

    while current <= e:
        ds = current.strftime("%Y%m%d")
        ids = fetch_race_ids_by_date(ds, venue_codes)
        if ids:
            date_counts[ds] = len(ids)
            all_ids.extend(x for x in ids if x not in all_ids)
        day_idx += 1
        if progress_callback:
            progress_callback(day_idx, total_days, ds)
        current += timedelta(days=1)
        time.sleep(random.uniform(1.0, 2.0))

    return {
        "race_ids":    all_ids,
        "n_dates":     len(date_counts),
        "n_races":     len(all_ids),
        "date_counts": date_counts,
    }
