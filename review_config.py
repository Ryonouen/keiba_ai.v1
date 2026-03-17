"""
review_config.py
レビュータグ定義・閾値設定の一元管理

責務:
- 日本語タグ定義（後から追加・変更が1ファイルで完結）
- タグと英語コードの対応
- カテゴリ分類（回収率分析の軸として使用）
- 判定閾値の定数化

このファイルを変更することで、review_engine.py のロジックを変えずに
タグ名・閾値・カテゴリを調整できる。
"""
from __future__ import annotations

from typing import Any, Dict, List

# =========================================================
# レビュータグ定義
# =========================================================
# 各タグ:
#   code:     英語コード（回顧ラベルとの対応、集計キーに使用）
#   level:    good / neutral / bad（UIでの色分け・評点に使用）
#   category: タグのカテゴリ（集計軸）
#   desc:     詳細説明（tooltip / ログ用）
# =========================================================

REVIEW_TAGS: Dict[str, Dict[str, Any]] = {

    # ── 頭候補 ───────────────────────────────────────────────────────────
    "頭候補的中":             {
        "code": "HEAD_HIT",
        "level": "good",
        "category": "役割判定",
        "desc": "頭候補に指定した馬が1着に来た",
    },
    "頭候補凡走":             {
        "code": "HEAD_MISS",
        "level": "bad",
        "category": "役割判定",
        "desc": "頭候補に指定した馬が着外に終わった",
    },
    "頭候補は2〜3着":         {
        "code": "HEAD_PLACED",
        "level": "neutral",
        "category": "役割判定",
        "desc": "頭候補が1着は外したが馬券内に来た",
    },

    # ── 軸・ヒモ ─────────────────────────────────────────────────────────
    "軸は来たがヒモ抜け":     {
        "code": "AXIS_HIMO_MISS",
        "level": "bad",
        "category": "役割判定",
        "desc": "軸候補は馬券内に来たが、ヒモ候補の選定が不足していた",
    },
    "軸も来ず":               {
        "code": "AXIS_MISS",
        "level": "bad",
        "category": "役割判定",
        "desc": "軸候補に指定した馬が全員着外に終わった",
    },
    "軸的中":                 {
        "code": "AXIS_HIT",
        "level": "good",
        "category": "役割判定",
        "desc": "軸候補が馬券内に来た",
    },

    # ── 危険人気馬 ────────────────────────────────────────────────────────
    "危険人気馬を切って正解":  {
        "code": "DANGER_CUT_OK",
        "level": "good",
        "category": "危険馬判定",
        "desc": "消し推奨にした危険人気馬が着外に終わった（判断が正しかった）",
    },
    "危険人気馬を切って失敗":  {
        "code": "DANGER_CUT_NG",
        "level": "bad",
        "category": "危険馬判定",
        "desc": "消し推奨にした危険人気馬が馬券内に来た（切り判断がミス）",
    },
    "相手残り危険馬が3着内":  {
        "code": "DANGER_SOFT_PLACED",
        "level": "neutral",
        "category": "危険馬判定",
        "desc": "「相手なら残る」と判定した危険馬が3着内に来た",
    },

    # ── 妙味馬 ───────────────────────────────────────────────────────────
    "妙味馬を活かした":       {
        "code": "VALUE_HIT",
        "level": "good",
        "category": "妙味馬",
        "desc": "妙味馬が馬券内に来て、買い目にも含めていた",
    },
    "妙味馬は来たが買っていない": {
        "code": "VALUE_MISS",
        "level": "bad",
        "category": "妙味馬",
        "desc": "妙味馬が馬券内に来たが、買い目に含めていなかった",
    },
    "妙味馬は着外":           {
        "code": "VALUE_WRONG",
        "level": "neutral",
        "category": "妙味馬",
        "desc": "妙味馬として抽出したが着外に終わった",
    },

    # ── 取りこぼし注意馬 ──────────────────────────────────────────────────
    "取りこぼし注意馬を活かした": {
        "code": "RESCUE_HIT",
        "level": "good",
        "category": "取りこぼし",
        "desc": "取りこぼし注意馬が馬券内に来て、買い目にも含めていた",
    },
    "取りこぼし注意馬を拾えなかった": {
        "code": "RESCUE_MISS",
        "level": "bad",
        "category": "取りこぼし",
        "desc": "取りこぼし注意馬が馬券内に来たが、買い目に含めていなかった",
    },

    # ── 券種 ─────────────────────────────────────────────────────────────
    "券種選択適切":           {
        "code": "BET_TYPE_OK",
        "level": "good",
        "category": "券種",
        "desc": "推奨券種がレース構造に合っていた",
    },
    "券種選択ミス":           {
        "code": "BET_TYPE_MISS",
        "level": "neutral",
        "category": "券種",
        "desc": "実際の結果から見ると別の券種が有効だった可能性がある",
    },

    # ── 展開 ─────────────────────────────────────────────────────────────
    "展開判断適切":           {
        "code": "PACE_OK",
        "level": "good",
        "category": "展開予測",
        "desc": "有利脚質の予測と実際の上位馬の脚質が一致した",
    },
    "展開判断ミス":           {
        "code": "PACE_NG",
        "level": "neutral",
        "category": "展開予測",
        "desc": "有利脚質の予測と実際の展開が乖離した可能性がある",
    },

    # ── 全体結果 ─────────────────────────────────────────────────────────
    "的中":                   {
        "code": "HIT",
        "level": "good",
        "category": "結果",
        "desc": "推奨買い目が的中した",
    },
    "不的中":                 {
        "code": "MISS",
        "level": "bad",
        "category": "結果",
        "desc": "推奨買い目が不的中だった",
    },
    "見送り":                 {
        "code": "PASS",
        "level": "neutral",
        "category": "結果",
        "desc": "EVが閾値未満のため見送りを選択した",
    },
    "見送りだが買えば的中":   {
        "code": "PASS_WOULD_HIT",
        "level": "bad",
        "category": "結果",
        "desc": "見送ったが、推奨買い目を実行していれば的中していた",
    },
}

# =========================================================
# 逆引きマップ（コード → 日本語タグ名）
# =========================================================
CODE_TO_TAG: Dict[str, str] = {v["code"]: k for k, v in REVIEW_TAGS.items()}

# =========================================================
# 判定閾値（変更はここだけで済む）
# =========================================================

# 頭候補が「凡走」と見なす着順（これより下なら凡走）
HEAD_FAIL_POSITION: int  = 3    # 3着内に入らなければ凡走

# 軸候補の「ヒモ抜け」判定: 軸が来ているのにヒモが来ていない場合
AXIS_SUCCESS_POSITION: int = 3  # 3着内に軸が1頭でも入れば「軸は来た」

# 危険馬の「切って失敗」判定
DANGER_FAIL_POSITION: int = 3   # 3着内に来たら失敗

# 妙味馬・取りこぼし注意馬の「来た」判定
VALUE_SUCCESS_POSITION: int = 3

# 展開判断: top N 着の馬の脚質がマッチすれば「適切」
PACE_CHECK_TOP_N: int = 2

# 券種の構造適合マップ（structure_type → 適切な券種リスト）
BET_TYPE_STRUCTURE_MAP: Dict[str, List[str]] = {
    "本命信頼型":    ["単勝", "複勝", "馬連"],
    "先行有利型":    ["単勝", "複勝", "馬連"],
    "1強相手混戦型": ["馬連", "ワイド", "3連複"],
    "混戦型":        ["ワイド", "3連複"],
    "波乱型":        ["ワイド", "3連複"],
    "差し届く型":    ["3連複", "ワイド"],
    "標準型":        ["馬連", "ワイド", "3連複"],
}

# 騎手の脚質ラベルマップ（running_style → 日本語）
STYLE_JP: Dict[str, str] = {
    "front":   "逃げ",
    "stalker": "先行",
    "closer":  "差し",
}

# =========================================================
# ユーティリティ
# =========================================================

def tag_level(tag_name: str) -> str:
    """日本語タグ名からレベルを返す。"""
    return REVIEW_TAGS.get(tag_name, {}).get("level", "neutral")


def tag_category(tag_name: str) -> str:
    """日本語タグ名からカテゴリを返す。"""
    return REVIEW_TAGS.get(tag_name, {}).get("category", "その他")


def tags_by_category(tags: List[str]) -> Dict[str, List[str]]:
    """タグリストをカテゴリ別に分類して返す。"""
    result: Dict[str, List[str]] = {}
    for t in tags:
        cat = tag_category(t)
        result.setdefault(cat, []).append(t)
    return result


def tags_by_level(tags: List[str]) -> Dict[str, List[str]]:
    """タグリストをレベル別（good/neutral/bad）に分類して返す。"""
    result: Dict[str, List[str]] = {"good": [], "neutral": [], "bad": []}
    for t in tags:
        lvl = tag_level(t)
        result.setdefault(lvl, []).append(t)
    return result
