"""
jockey_ai.py
騎手補正モジュール — 実データ前提設計

責務:
- 騎手実績データの読み込みとプロファイル構築
- ベイズ平滑化によるサンプルサイズを加味したスコア計算
- 条件別（コース/距離/脚質/人気帯/枠）補正値の算出
- 各馬へのジョッキー補正値・信頼度・理由の返却
- UI表示用の日本語理由テキスト生成

設計方針:
- 感覚補正ではなく必ず実績データに基づく
- サンプル数が少ない条件はベイズ平滑化で全体平均へ寄せる
- シード値（公開統計）は信頼度 SEED_CONFIDENCE で使用
- 実CSV/DBデータが入れば自動的にそちらが優先される
- delta は小さめ（既存 jockey_index との二重カウントを避ける）
"""
from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional, Tuple

from trend_stats import (
    bucket_gate,
    bucket_popularity_odds as bucket_pop_from_odds,
    bucket_distance,
)

# =========================================================
# 定数（全て後から調整可能）
# =========================================================

# ── JRA重賞グローバル基準値（ベイズ平滑化のプリオール）────────────────
GLOBAL_WIN_RATE:   float = 0.085
GLOBAL_TOP3_RATE:  float = 0.280
GLOBAL_FAV_WIN:    float = 0.340   # 1〜3番人気時の全体勝率
GLOBAL_FAV_TOP3:   float = 0.640
GLOBAL_LONG_TOP3:  float = 0.020   # 10番人気以下の全体3着内率

# ── ベイズ平滑化プリオール強度（等価騎乗回数）────────────────────────
PRIOR_STRENGTH_OVERALL:   int = 30   # 全体勝率の平滑化強度
PRIOR_STRENGTH_CONDITION: int = 20   # 条件別の平滑化強度
PRIOR_STRENGTH_COMBO:     int = 10   # 馬別コンビの平滑化強度

# ── シードデータの信頼度（公開統計から得た値で自前DBではない）────────
SEED_CONFIDENCE:    float = 0.55
SEED_RIDES_EQUIV:   int   = 150     # シード勝率を等価騎乗数に変換するときの基準

# ── 各補正コンポーネントの上限 ──────────────────────────────────────
DELTA_STYLE_MAX:    float = 0.020
DELTA_POP_MAX:      float = 0.020
DELTA_TRACK_MAX:    float = 0.015
DELTA_DISTANCE_MAX: float = 0.012
DELTA_COMBO_MAX:    float = 0.010

# ── 最終合計補正の上下限 ─────────────────────────────────────────────
JOCKEY_DELTA_MAX:   float = +0.055
JOCKEY_DELTA_MIN:   float = -0.040

# ── 補正感度係数（delta = diff × sensitivity） ──────────────────────
SENSITIVITY_STYLE:    float = 0.18
SENSITIVITY_POP:      float = 0.16
SENSITIVITY_TRACK:    float = 0.14
SENSITIVITY_DISTANCE: float = 0.12

# ── 理由テキストの閾値（この差分以上で言及する）───────────────────────
REASON_WIN_DIFF_THRESH: float = 0.025    # 勝率差がこれ以上で言及
REASON_SAMPLE_WARN:     int   = 8        # サンプルがこれ未満で「不足」警告

# ── スタイル補正の基準値（これより高い/低いで補正が入る）────────────
JOCKEY_STYLE_BASE: float = 0.80

# ── 人気馬/人気薄の定義 ─────────────────────────────────────────────
FAVORITE_ODDS_MAX:  float = 5.0
LONGSHOT_ODDS_MIN:  float = 20.0


# =========================================================
# 生データスキーマ（CSV/APIから読み込む際の標準フォーマット）
# =========================================================
# 各レコードは以下のキーを持つ dict:
#
#   jockey_name:      str   — 騎手名
#   condition_type:   str   — "overall" | "by_style" | "by_pop_bucket"
#                             | "by_track" | "by_distance" | "by_surface"
#                             | "by_gate" | "by_horse"
#   condition_value:  str   — 条件値 (例: "中京", "差し", "1番人気")
#                             overall の場合は空文字
#   rides:            int   — 騎乗回数
#   wins:             int   — 1着回数
#   top2:             int   — 2着以内回数
#   top3:             int   — 3着以内回数
# =========================================================


# =========================================================
# シードデータ（公開統計からの重賞成績概算）
# 実CSV/DBが入った場合はこのデータは使われない
# =========================================================

_JOCKEY_SEED: Dict[str, Dict[str, float]] = {
    "ルメール": {
        "win_rate": 0.220, "top3_rate": 0.510,
        "fav_win_rate": 0.460, "fav_top3_rate": 0.740,
        "longshot_rate": 0.015,
        "style_front": 0.75, "style_stalker": 0.90, "style_closer": 0.88,
    },
    "川田": {
        "win_rate": 0.195, "top3_rate": 0.480,
        "fav_win_rate": 0.430, "fav_top3_rate": 0.720,
        "longshot_rate": 0.018,
        "style_front": 0.82, "style_stalker": 0.90, "style_closer": 0.80,
    },
    "戸崎": {
        "win_rate": 0.155, "top3_rate": 0.405,
        "fav_win_rate": 0.360, "fav_top3_rate": 0.640,
        "longshot_rate": 0.025,
        "style_front": 0.80, "style_stalker": 0.86, "style_closer": 0.81,
    },
    "坂井": {
        "win_rate": 0.175, "top3_rate": 0.435,
        "fav_win_rate": 0.390, "fav_top3_rate": 0.670,
        "longshot_rate": 0.022,
        "style_front": 0.74, "style_stalker": 0.82, "style_closer": 0.88,
    },
    "横山武": {
        "win_rate": 0.160, "top3_rate": 0.415,
        "fav_win_rate": 0.330, "fav_top3_rate": 0.650,
        "longshot_rate": 0.033,
        "style_front": 0.78, "style_stalker": 0.83, "style_closer": 0.85,
    },
    "武豊": {
        "win_rate": 0.145, "top3_rate": 0.390,
        "fav_win_rate": 0.340, "fav_top3_rate": 0.630,
        "longshot_rate": 0.018,
        "style_front": 0.88, "style_stalker": 0.88, "style_closer": 0.70,
    },
    "松山": {
        "win_rate": 0.140, "top3_rate": 0.382,
        "fav_win_rate": 0.330, "fav_top3_rate": 0.618,
        "longshot_rate": 0.028,
        "style_front": 0.79, "style_stalker": 0.84, "style_closer": 0.80,
    },
    "西村淳": {
        "win_rate": 0.130, "top3_rate": 0.365,
        "fav_win_rate": 0.310, "fav_top3_rate": 0.590,
        "longshot_rate": 0.028,
        "style_front": 0.80, "style_stalker": 0.82, "style_closer": 0.82,
    },
    "岩田望": {
        "win_rate": 0.140, "top3_rate": 0.378,
        "fav_win_rate": 0.330, "fav_top3_rate": 0.620,
        "longshot_rate": 0.030,
        "style_front": 0.76, "style_stalker": 0.82, "style_closer": 0.86,
    },
    "菅原明": {
        "win_rate": 0.125, "top3_rate": 0.360,
        "fav_win_rate": 0.305, "fav_top3_rate": 0.580,
        "longshot_rate": 0.027,
        "style_front": 0.79, "style_stalker": 0.82, "style_closer": 0.82,
    },
    "鮫島駿": {
        "win_rate": 0.115, "top3_rate": 0.350,
        "fav_win_rate": 0.290, "fav_top3_rate": 0.570,
        "longshot_rate": 0.030,
        "style_front": 0.80, "style_stalker": 0.81, "style_closer": 0.82,
    },
    "横山和": {
        "win_rate": 0.105, "top3_rate": 0.335,
        "fav_win_rate": 0.285, "fav_top3_rate": 0.560,
        "longshot_rate": 0.028,
        "style_front": 0.80, "style_stalker": 0.81, "style_closer": 0.80,
    },
    "田辺": {
        "win_rate": 0.120, "top3_rate": 0.355,
        "fav_win_rate": 0.305, "fav_top3_rate": 0.580,
        "longshot_rate": 0.027,
        "style_front": 0.82, "style_stalker": 0.83, "style_closer": 0.79,
    },
    "団野": {
        "win_rate": 0.105, "top3_rate": 0.330,
        "fav_win_rate": 0.278, "fav_top3_rate": 0.555,
        "longshot_rate": 0.032,
        "style_front": 0.79, "style_stalker": 0.81, "style_closer": 0.82,
    },
    "幸": {
        "win_rate": 0.100, "top3_rate": 0.325,
        "fav_win_rate": 0.272, "fav_top3_rate": 0.548,
        "longshot_rate": 0.030,
        "style_front": 0.83, "style_stalker": 0.82, "style_closer": 0.78,
    },
    "丹内": {
        "win_rate": 0.100, "top3_rate": 0.320,
        "fav_win_rate": 0.270, "fav_top3_rate": 0.545,
        "longshot_rate": 0.025,
        "style_front": 0.80, "style_stalker": 0.81, "style_closer": 0.80,
    },
    "Mデムーロ": {
        "win_rate": 0.138, "top3_rate": 0.375,
        "fav_win_rate": 0.328, "fav_top3_rate": 0.618,
        "longshot_rate": 0.022,
        "style_front": 0.82, "style_stalker": 0.85, "style_closer": 0.76,
    },
    "Cデムーロ": {
        "win_rate": 0.172, "top3_rate": 0.435,
        "fav_win_rate": 0.388, "fav_top3_rate": 0.668,
        "longshot_rate": 0.020,
        "style_front": 0.75, "style_stalker": 0.82, "style_closer": 0.88,
    },
    "石川裕": {
        "win_rate": 0.090, "top3_rate": 0.305,
        "fav_win_rate": 0.258, "fav_top3_rate": 0.535,
        "longshot_rate": 0.028,
        "style_front": 0.80, "style_stalker": 0.80, "style_closer": 0.80,
    },
    "柴田大": {
        "win_rate": 0.090, "top3_rate": 0.300,
        "fav_win_rate": 0.255, "fav_top3_rate": 0.530,
        "longshot_rate": 0.027,
        "style_front": 0.80, "style_stalker": 0.80, "style_closer": 0.80,
    },
}

# 表記揺れ正規化マップ
_JOCKEY_ALIAS: Dict[str, str] = {
    "西村淳也": "西村淳",
    "岩田望来": "岩田望",
    "鮫島克駿": "鮫島駿",
    "横山武史": "横山武",
    "横山和生": "横山和",
    "菅原明良": "菅原明",
    "坂井瑠星": "坂井",
    "川田将雅": "川田",
    "松山弘平": "松山",
    "団野大成": "団野",
    "田辺裕信": "田辺",
    "幸英明": "幸",
    "武豊": "武豊",
    "ルメール": "ルメール",
    "C.デムーロ": "Cデムーロ",
    "M.デムーロ": "Mデムーロ",
    "石川裕紀人": "石川裕",
    "柴田大知": "柴田大",
    "丹内祐次": "丹内",
}

# プロファイルキャッシュ（プロセス内で1回だけビルドされる）
_PROFILE_CACHE: Dict[str, Dict[str, Any]] = {}
_RAW_RECORDS_CACHE: Optional[List[Dict[str, Any]]] = None


# =========================================================
# ベイズ平滑化ユーティリティ
# =========================================================

def smooth_rate(
    wins: float,
    rides: float,
    prior_strength: int,
    global_rate: float,
) -> float:
    """
    ベイズ平滑化後の勝率を返す。

    smoothed = (wins + prior * global_rate) / (rides + prior)

    rides=0 でも安全に global_rate を返す。
    """
    denom = rides + prior_strength
    if denom <= 0:
        return global_rate
    return (wins + prior_strength * global_rate) / denom


def calc_confidence(rides: float, prior_strength: int) -> float:
    """
    サンプルサイズに基づく信頼度を返す（0〜1）。

    rides → 0 のとき confidence → 0 (全て prior に依存)
    rides → ∞ のとき confidence → 1 (実データに完全依存)
    """
    return rides / (rides + prior_strength)


# bucket_gate / bucket_pop_from_odds / bucket_distance は trend_stats からインポート済み


# =========================================================
# 名前正規化
# =========================================================

def normalize_jockey_name(name: str) -> str:
    """表記揺れを正規化する。"""
    if not name:
        return ""
    if name in _JOCKEY_SEED:
        return name
    if name in _JOCKEY_ALIAS:
        return _JOCKEY_ALIAS[name]
    for alias, canonical in _JOCKEY_ALIAS.items():
        if name.startswith(alias[:2]) and len(name) >= 2:
            return canonical
    for canonical in _JOCKEY_SEED:
        if name.startswith(canonical[:2]) and len(canonical) >= 2:
            return canonical
    return name


# =========================================================
# データ読み込み
# =========================================================

def load_jockey_stats(
    filepath: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    騎手実績データをCSVから読み込む。

    Parameters
    ----------
    filepath : CSVファイルパス（None ならシードデータを使用）

    Returns
    -------
    List of records, each containing:
        jockey_name, condition_type, condition_value,
        rides, wins, top2, top3

    CSVが存在しない場合はシードデータから生成したレコードを返す。
    """
    if filepath and os.path.exists(filepath):
        records: List[Dict[str, Any]] = []
        try:
            with open(filepath, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append({
                        "jockey_name":     str(row.get("jockey_name", "")),
                        "condition_type":  str(row.get("condition_type", "overall")),
                        "condition_value": str(row.get("condition_value", "")),
                        "rides": int(row.get("rides", 0) or 0),
                        "wins":  int(row.get("wins",  0) or 0),
                        "top2":  int(row.get("top2",  0) or 0),
                        "top3":  int(row.get("top3",  0) or 0),
                    })
        except Exception:
            records = []
        if records:
            return records

    # シードデータから合成レコードを生成
    return _build_seed_records()


def _build_seed_records() -> List[Dict[str, Any]]:
    """
    `_JOCKEY_SEED` から標準レコード形式に変換する。
    シード値は "等価騎乗数" SEED_RIDES_EQUIV 回分として扱う。
    """
    records: List[Dict[str, Any]] = []
    for name, s in _JOCKEY_SEED.items():
        r = SEED_RIDES_EQUIV
        # overall
        records.append({
            "jockey_name": name, "condition_type": "overall",
            "condition_value": "",
            "rides": r,
            "wins":  int(s["win_rate"]  * r),
            "top2":  int(s["top3_rate"] * r * 0.65),
            "top3":  int(s["top3_rate"] * r),
        })
        # by_pop_bucket — 1〜3番人気
        r_fav = int(r * 0.22)   # 全騎乗の約22%が上位人気
        records.append({
            "jockey_name": name, "condition_type": "by_pop_bucket",
            "condition_value": "1番人気",
            "rides": r_fav,
            "wins":  int(s["fav_win_rate"]  * r_fav),
            "top2":  int(s["fav_top3_rate"] * r_fav * 0.70),
            "top3":  int(s["fav_top3_rate"] * r_fav),
        })
        r_2_3 = int(r * 0.20)
        fav_2_3_win = s["fav_win_rate"] * 0.70
        records.append({
            "jockey_name": name, "condition_type": "by_pop_bucket",
            "condition_value": "2〜3番人気",
            "rides": r_2_3,
            "wins":  int(fav_2_3_win       * r_2_3),
            "top2":  int(fav_2_3_win * 2.2 * r_2_3),
            "top3":  int(s["fav_top3_rate"] * 0.75 * r_2_3),
        })
        # by_pop_bucket — 10番人気以下
        r_ls = int(r * 0.18)
        records.append({
            "jockey_name": name, "condition_type": "by_pop_bucket",
            "condition_value": "10番人気以下",
            "rides": r_ls,
            "wins":  int(s["longshot_rate"] * 0.35 * r_ls),
            "top2":  int(s["longshot_rate"] * 0.65 * r_ls),
            "top3":  int(s["longshot_rate"] * r_ls),
        })
        # by_style
        for style_key, jp_key in [("style_front", "逃げ"), ("style_stalker", "先行"), ("style_closer", "差し")]:
            style_score = s.get(style_key, JOCKEY_STYLE_BASE)
            r_s = int(r * 0.30)
            adj = style_score / JOCKEY_STYLE_BASE
            records.append({
                "jockey_name": name, "condition_type": "by_style",
                "condition_value": jp_key,
                "rides": r_s,
                "wins":  int(s["win_rate"]  * adj * r_s),
                "top2":  int(s["top3_rate"] * adj * 0.60 * r_s),
                "top3":  int(s["top3_rate"] * adj * 0.90 * r_s),
            })
    return records


# =========================================================
# プロファイル構築
# =========================================================

def build_jockey_profile(
    records: List[Dict[str, Any]],
    jockey_name: str,
    is_seed: bool = False,
) -> Dict[str, Any]:
    """
    1騎手分のプロファイルを構築する。

    Returns
    -------
    {
        "jockey_name": str,
        "data_source": "seed" | "real",
        "overall":       {"rides": int, "wins": int, "top3": int,
                          "smoothed_win_rate": float, "smoothed_top3_rate": float,
                          "confidence": float},
        "by_style":      {style_jp: {...}},
        "by_pop_bucket": {pop_key:  {...}},
        "by_track":      {track:    {...}},
        "by_distance":   {dist_key: {...}},
        "by_surface":    {surface:  {...}},
        "by_gate":       {gate_key: {...}},
        "by_horse":      {horse:    {...}},
    }
    """
    my_records = [r for r in records if normalize_jockey_name(r["jockey_name"]) == jockey_name]

    profile: Dict[str, Any] = {
        "jockey_name": jockey_name,
        "data_source": "seed" if is_seed else "real",
        "overall": None,
        "by_style":      {},
        "by_pop_bucket": {},
        "by_track":      {},
        "by_distance":   {},
        "by_surface":    {},
        "by_gate":       {},
        "by_horse":      {},
    }

    for rec in my_records:
        ctype = rec["condition_type"]
        cval  = rec.get("condition_value", "") or ""
        rides = int(rec.get("rides", 0))
        wins  = int(rec.get("wins",  0))
        top3  = int(rec.get("top3",  0))

        if ctype == "overall":
            prior    = PRIOR_STRENGTH_OVERALL
            swr  = smooth_rate(wins, rides, prior, GLOBAL_WIN_RATE)
            st3r = smooth_rate(top3, rides, prior, GLOBAL_TOP3_RATE)
            conf = SEED_CONFIDENCE if is_seed else calc_confidence(rides, prior)
            profile["overall"] = {
                "rides": rides, "wins": wins, "top3": top3,
                "smoothed_win_rate":  round(swr, 4),
                "smoothed_top3_rate": round(st3r, 4),
                "confidence": round(conf, 3),
            }
        else:
            dest = profile.get(ctype)
            if dest is None:
                continue
            prior = PRIOR_STRENGTH_COMBO if ctype == "by_horse" else PRIOR_STRENGTH_CONDITION
            ovr_wr  = GLOBAL_WIN_RATE
            ovr_t3  = GLOBAL_TOP3_RATE
            if profile["overall"]:
                ovr_wr = profile["overall"]["smoothed_win_rate"]
                ovr_t3 = profile["overall"]["smoothed_top3_rate"]
            swr  = smooth_rate(wins, rides, prior, ovr_wr)
            st3r = smooth_rate(top3, rides, prior, ovr_t3)
            conf = SEED_CONFIDENCE if is_seed else calc_confidence(rides, prior)
            dest[cval] = {
                "rides": rides, "wins": wins, "top3": top3,
                "smoothed_win_rate":  round(swr, 4),
                "smoothed_top3_rate": round(st3r, 4),
                "confidence": round(conf, 3),
            }

    # overall が取れなかった場合はグローバル平均でフォールバック
    if profile["overall"] is None:
        profile["overall"] = {
            "rides": 0, "wins": 0, "top3": 0,
            "smoothed_win_rate":  GLOBAL_WIN_RATE,
            "smoothed_top3_rate": GLOBAL_TOP3_RATE,
            "confidence": 0.0,
        }

    return profile


def build_all_profiles(
    records: List[Dict[str, Any]],
    is_seed: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """全騎手のプロファイルを構築してキャッシュ用辞書を返す。"""
    names = {normalize_jockey_name(r["jockey_name"]) for r in records if r["jockey_name"]}
    return {name: build_jockey_profile(records, name, is_seed=is_seed) for name in names if name}


# =========================================================
# プロファイルキャッシュ管理
# =========================================================

def _ensure_profiles() -> None:
    """プロファイルキャッシュが未構築ならビルドする。"""
    global _PROFILE_CACHE, _RAW_RECORDS_CACHE
    if _PROFILE_CACHE:
        return

    # 実CSVが存在すれば優先使用
    default_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jockey_stats.csv")
    records     = load_jockey_stats(default_csv)
    is_seed     = not (os.path.exists(default_csv))
    _RAW_RECORDS_CACHE = records
    _PROFILE_CACHE.update(build_all_profiles(records, is_seed=is_seed))


def get_jockey_profile(jockey_name: str, jockey_index: float = 1.0) -> Dict[str, Any]:
    """
    騎手名からプロファイルを取得する（キャッシュ参照）。
    未知騎手は jockey_index から近似プロファイルを生成する。
    """
    _ensure_profiles()
    canonical = normalize_jockey_name(jockey_name)
    if canonical in _PROFILE_CACHE:
        return _PROFILE_CACHE[canonical]

    # 未知騎手: jockey_index から近似プロファイルを生成
    idx  = max(1.0, min(1.20, float(jockey_index)))
    d    = idx - 1.0
    win  = GLOBAL_WIN_RATE  + d * 0.70
    top3 = GLOBAL_TOP3_RATE + d * 1.00
    return {
        "jockey_name": jockey_name,
        "data_source": "estimated",
        "overall": {
            "rides": 0, "wins": 0, "top3": 0,
            "smoothed_win_rate":  round(win,  4),
            "smoothed_top3_rate": round(top3, 4),
            "confidence": 0.20,
        },
        "by_style":      {},
        "by_pop_bucket": {},
        "by_track":      {},
        "by_distance":   {},
        "by_surface":    {},
        "by_gate":       {},
        "by_horse":      {},
    }


# =========================================================
# 補正コンポーネント計算
# =========================================================

def _cond_delta(
    cond_dict: Dict[str, Any],
    cond_key: Optional[str],
    overall_wr: float,
    sensitivity: float,
    delta_max: float,
) -> Tuple[float, float, int]:
    """
    1条件のデルタ・信頼度・サンプル数を返す。

    Returns
    -------
    (delta, confidence, rides)
    """
    if not cond_key or cond_key not in cond_dict:
        return 0.0, 0.0, 0

    c    = cond_dict[cond_key]
    diff = c["smoothed_win_rate"] - overall_wr
    conf = c["confidence"]
    raw  = diff * sensitivity * conf
    return max(-delta_max, min(delta_max, raw)), conf, c["rides"]


# =========================================================
# 全コンポーネントのデルタ + 理由生成
# =========================================================

def calc_jockey_delta(
    feature: Dict[str, Any],
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    1頭分のジョッキー補正値・信頼度・理由テキストを返す。

    Returns
    -------
    {
        "jockey_name":       str,
        "jockey_score":      float,  # 総合スコア 0-1
        "jockey_delta":      float,  # model_score への加減算値
        "jockey_confidence": float,  # 補正全体の信頼度
        "jockey_reasons":    List[str],  # UI表示用日本語テキスト
        "jockey_summary":    str,    # 1行サマリー
        "jockey_win_rate":   float,  # 全体勝率（表示用）
        "jockey_top3_rate":  float,  # 全体3着内率（表示用）
        "jockey_reason_codes": List[str],  # 後方互換用コード
        "jockey_details":    List[Dict],  # 詳細デバッグ情報
    }
    """
    jockey_name  = str(feature.get("entry_jockey") or "")
    jockey_index = float(feature.get("jockey_index") or 1.0)
    running_style = feature.get("running_style") or ""
    win_odds      = feature.get("win_odds")
    target_course = str(feature.get("target_course") or "")
    target_dist   = feature.get("target_distance")
    target_surface = str(feature.get("target_surface") or "")
    gate          = feature.get("gate")
    horse_name    = str(feature.get("horse_name") or "")

    ovr = profile["overall"]
    ovr_wr  = ovr["smoothed_win_rate"]
    ovr_t3  = ovr["smoothed_top3_rate"]
    ovr_conf = ovr["confidence"]

    _details: List[Dict[str, Any]] = []
    total_delta = 0.0
    total_conf  = 0.0
    n_components = 0

    def _add(label: str, key_used: Optional[str], cond_dict: Dict, sens: float, dmax: float) -> None:
        nonlocal total_delta, total_conf, n_components
        d, c, rides = _cond_delta(cond_dict, key_used, ovr_wr, sens, dmax)
        if abs(d) < 1e-6:
            return
        total_delta  += d
        total_conf   += c
        n_components += 1
        _details.append({
            "label":      label,
            "key":        key_used,
            "delta":      d,
            "confidence": c,
            "rides":      rides,
        })

    # ── 脚質 ──────────────────────────────────────────────────────────────
    style_jp = {"front": "逃げ", "stalker": "先行", "closer": "差し"}.get(running_style)
    _add("脚質", style_jp, profile["by_style"], SENSITIVITY_STYLE, DELTA_STYLE_MAX)

    # ── 人気帯 ────────────────────────────────────────────────────────────
    pop_key = bucket_pop_from_odds(win_odds)
    _add("人気帯", pop_key, profile["by_pop_bucket"], SENSITIVITY_POP, DELTA_POP_MAX)

    # ── コース ────────────────────────────────────────────────────────────
    _add("コース", target_course or None, profile["by_track"], SENSITIVITY_TRACK, DELTA_TRACK_MAX)

    # ── 距離 ──────────────────────────────────────────────────────────────
    dist_key = bucket_distance(int(target_dist) if target_dist else None)
    _add("距離", dist_key, profile["by_distance"], SENSITIVITY_DISTANCE, DELTA_DISTANCE_MAX)

    # ── 馬コンビ ──────────────────────────────────────────────────────────
    _add("馬コンビ", horse_name, profile["by_horse"], SENSITIVITY_DISTANCE, DELTA_COMBO_MAX)

    # ── 合計 ──────────────────────────────────────────────────────────────
    jockey_delta = max(JOCKEY_DELTA_MIN, min(JOCKEY_DELTA_MAX, total_delta))
    jockey_confidence = round(
        (total_conf / n_components * 0.7 + ovr_conf * 0.3)
        if n_components > 0
        else ovr_conf * 0.3,
        3,
    )

    # ── 理由テキスト生成 ──────────────────────────────────────────────────
    reasons, reason_codes = _build_reasons(
        jockey_name, profile, _details,
        ovr_wr, ovr_t3, ovr_conf,
        running_style, win_odds,
    )

    # 総合スコア 0-1（0.5 = 補正なし）
    jockey_score = round(min(1.0, max(0.0, 0.5 + total_delta * 7.0)), 4)
    jockey_summary = "・".join(reasons[:2]) if reasons else ""

    return {
        "jockey_name":       jockey_name,
        "jockey_score":      jockey_score,
        "jockey_delta":      round(jockey_delta, 5),
        "jockey_confidence": jockey_confidence,
        "jockey_reasons":    reasons,
        "jockey_summary":    jockey_summary,
        "jockey_win_rate":   ovr_wr,
        "jockey_top3_rate":  ovr_t3,
        "jockey_reason_codes": reason_codes,
        "jockey_details":    _details,
        # 後方互換用追加フィールド
        "jockey_course_win_rate":          ovr_wr,
        "jockey_course_top3_rate":         ovr_t3,
        "jockey_distance_win_rate":        ovr_wr,
        "jockey_distance_top3_rate":       ovr_t3,
        "jockey_style_fit_score":          round(min(1.0, ovr_wr / max(GLOBAL_WIN_RATE, 0.001)), 4),
        "jockey_favorite_trust_score":     _fav_trust_score(profile),
        "jockey_longshot_upside_score":    _longshot_score(profile),
        "jockey_pop_fav_rate":             _pop_rate(profile, "1番人気", "smoothed_win_rate"),
        "jockey_pop_longshot_rate":        _pop_rate(profile, "10番人気以下", "smoothed_top3_rate"),
        "jockey_running_style_fit":        _style_fit(profile, running_style),
        "jockey_gate_fit":                 _gate_fit(profile, bucket_gate(int(gate) if gate else None)),
        "jockey_horse_combo_rate":         _combo_rate(profile, horse_name),
    }


# ── 後方互換スコア計算ヘルパー ────────────────────────────────────────

def _fav_trust_score(profile: Dict) -> float:
    fav = profile["by_pop_bucket"].get("1番人気")
    if fav:
        return round(min(1.0, fav["smoothed_win_rate"] / max(GLOBAL_FAV_WIN, 0.001)), 4)
    return round(min(1.0, profile["overall"]["smoothed_win_rate"] / max(GLOBAL_FAV_WIN, 0.001)), 4)


def _longshot_score(profile: Dict) -> float:
    ls = profile["by_pop_bucket"].get("10番人気以下")
    if ls:
        return round(min(1.0, ls["smoothed_top3_rate"] / max(GLOBAL_LONG_TOP3 * 2, 0.001)), 4)
    return 0.5


def _pop_rate(profile: Dict, bucket: str, key: str) -> float:
    c = profile["by_pop_bucket"].get(bucket)
    return c[key] if c else profile["overall"]["smoothed_win_rate"]


def _style_fit(profile: Dict, running_style: str) -> float:
    jp = {"front": "逃げ", "stalker": "先行", "closer": "差し"}.get(running_style)
    c  = profile["by_style"].get(jp) if jp else None
    if c:
        return round(c["smoothed_win_rate"] / max(profile["overall"]["smoothed_win_rate"], 0.001), 4)
    return 1.0


def _gate_fit(profile: Dict, gate_key: Optional[str]) -> float:
    c = profile["by_gate"].get(gate_key) if gate_key else None
    if c:
        return round(c["smoothed_win_rate"] / max(profile["overall"]["smoothed_win_rate"], 0.001), 4)
    return 1.0


def _combo_rate(profile: Dict, horse_name: str) -> float:
    c = profile["by_horse"].get(horse_name)
    if c and c["rides"] > 0:
        return c["smoothed_win_rate"]
    return profile["overall"]["smoothed_win_rate"]


# ── 理由テキスト生成 ─────────────────────────────────────────────────

def _build_reasons(
    jockey_name: str,
    profile: Dict,
    details: List[Dict],
    ovr_wr: float, ovr_t3: float, ovr_conf: float,
    running_style: str,
    win_odds: Any,
) -> Tuple[List[str], List[str]]:
    reasons: List[str] = []
    codes:   List[str] = []
    data_src = profile["data_source"]

    for d in details:
        label    = d["label"]
        key      = d["key"] or ""
        delta    = d["delta"]
        conf     = d["confidence"]
        rides    = d["rides"]
        sign     = "高い" if delta > 0 else "低い"
        sign_jp  = "プラス補正" if delta > 0 else "マイナス補正"
        abs_d    = abs(delta)

        if abs_d < 0.003:
            continue

        rides_note = f"（{rides}戦、信頼度{conf:.0%}）" if rides > 0 else ""
        if label == "脚質":
            if delta > 0:
                reasons.append(f"{key}馬との相性良好{rides_note}")
                codes.append("STYLE_FIT")
            else:
                reasons.append(f"{key}馬との相性やや不安{rides_note}")
                codes.append("STYLE_MISMATCH")
        elif label == "人気帯":
            if key in ("1番人気", "2〜3番人気"):
                if delta > 0:
                    reasons.append(f"人気馬騎乗時の信頼度高{rides_note}")
                    codes.append("FAVORITE_TRUST")
                else:
                    reasons.append(f"人気馬騎乗時の信頼度やや低{rides_note}")
                    codes.append("FAVORITE_LOW_TRUST")
            else:
                if delta > 0:
                    reasons.append(f"人気薄での激走率あり{rides_note}")
                    codes.append("LONGSHOT_UPSIDE")
        elif label == "コース":
            reasons.append(f"{key}コースで{sign}勝率{rides_note}")
            codes.append("TRACK_FIT" if delta > 0 else "TRACK_WEAK")
        elif label == "距離":
            reasons.append(f"{key}で{sign}成績{rides_note}")
            codes.append("DISTANCE_FIT" if delta > 0 else "DISTANCE_WEAK")
        elif label == "馬コンビ":
            if delta > 0:
                reasons.append(f"当該馬とのコンビで好成績{rides_note}")
                codes.append("COMBO_GOOD")

    # サンプル不足の警告
    low_conf_components = [d for d in details if d["rides"] < REASON_SAMPLE_WARN and d["rides"] > 0]
    if low_conf_components or ovr_conf < 0.30:
        reasons.append("サンプル不足のため補正は弱め")
        codes.append("LOW_SAMPLE")

    # データソース注記
    if data_src == "seed" and not any("サンプル" in r for r in reasons):
        reasons.append("シード値使用（実DB未取得）")
    elif data_src == "estimated":
        reasons.append(f"騎手IDXから推定（{jockey_name}の実績データ未登録）")
        codes.append("ESTIMATED")

    # jockey_index 由来の一般評価
    if not reasons:
        if ovr_wr >= GLOBAL_WIN_RATE * 1.30:
            reasons.append("重賞での実績が高い騎手")
            codes.append("TOP_JOCKEY")
        elif ovr_wr >= GLOBAL_WIN_RATE * 1.10:
            reasons.append("実力騎手")
            codes.append("GOOD_JOCKEY")

    return reasons, codes


# =========================================================
# 後方互換: calc_jockey_score
# =========================================================

def calc_jockey_score(f: Dict[str, Any]) -> Dict[str, Any]:
    """
    1頭分のジョッキー補正スコアを計算する（後方互換インターフェース）。
    calc_jockey_delta のラッパー。
    """
    jockey_name  = str(f.get("entry_jockey") or "")
    jockey_index = float(f.get("jockey_index") or 1.0)
    profile      = get_jockey_profile(jockey_name, jockey_index)
    return calc_jockey_delta(f, profile)


# =========================================================
# 全馬への適用
# =========================================================

def apply_jockey_adjustments(
    features: List[Dict[str, Any]],
    raw_records: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    全馬に騎手補正を適用して model_score を更新する。

    Parameters
    ----------
    features    : race_ai_engine の features リスト
    raw_records : 実データレコードリスト（None なら内部キャッシュを使用）

    各 feature に追加されるフィールド:
    - jockey_delta, jockey_score, jockey_confidence, jockey_reasons,
      jockey_summary, jockey_win_rate, jockey_top3_rate,
      jockey_reason_codes, jockey_details, ... (後方互換フィールド)
    """
    global _PROFILE_CACHE

    if raw_records:
        # 外部からデータが渡された場合はキャッシュを更新
        _PROFILE_CACHE.clear()
        _PROFILE_CACHE.update(build_all_profiles(raw_records, is_seed=False))

    for f in features:
        jockey_name  = str(f.get("entry_jockey") or "")
        jockey_index = float(f.get("jockey_index") or 1.0)
        profile      = get_jockey_profile(jockey_name, jockey_index)
        js           = calc_jockey_delta(f, profile)
        f.update(js)
        f["model_score"] = round(float(f.get("model_score") or 0.0) + js["jockey_delta"], 6)

    return features
