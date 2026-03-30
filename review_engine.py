"""
review_engine.py
レースレビュー生成エンジン

責務:
- actual_bets（実際の購入内容）を考慮したタグ生成
- review_config.py の定義に基づく日本語タグ出力
- 後フェーズの集計・分析に使いやすいデータ構造を返す

フェーズ4の review_ai.py との違い:
- 日本語タグ（review_config.REVIEW_TAGS）を出力する
- actual_bets（ユーザーが実際に買った内容）も判定に使う
- タグ生成ロジックを review_config と分離してテスト可能にしている
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from review_config import (
    REVIEW_TAGS,
    CODE_TO_TAG,
    BET_TYPE_STRUCTURE_MAP,
    HEAD_FAIL_POSITION,
    AXIS_SUCCESS_POSITION,
    DANGER_FAIL_POSITION,
    VALUE_SUCCESS_POSITION,
    PACE_CHECK_TOP_N,
    STYLE_JP,
    tag_level,
    tags_by_category,
    tags_by_level,
)


# =========================================================
# 内部ヘルパー
# =========================================================

def _top3_set(finish_order: List[str]) -> Set[str]:
    return set(finish_order[:3])


def _top2_set(finish_order: List[str]) -> Set[str]:
    return set(finish_order[:2])


def _bought_names(tickets: List[Dict[str, Any]]) -> Set[str]:
    """推奨買い目に含まれる馬名セットを返す。"""
    names: Set[str] = set()
    for t in tickets:
        names.update(t.get("combination", []))
    return names


def _actual_bought_names(actual_bets_text: str) -> Set[str]:
    """
    actual_bets テキストから馬名を抽出する。
    形式例: "馬連 イクイノックス-ジャスティンパレス 500円"
    馬名はスペース・ハイフン・スラッシュで区切られていることが多い。
    券種名・数字・金額表記・一般ノイズを除去してから返す。
    """
    if not actual_bets_text:
        return set()
    import re
    # 除外ワード: 券種名・単位・その他ノイズ
    _NOISE = frozenset({
        "単勝", "複勝", "馬連", "馬単", "ワイド", "3連複", "3連単",
        "枠連", "枠単", "円", "票", "点", "番", "着", "倍",
    })
    tokens = re.split(r"[\s\-/・,、　→×]+", actual_bets_text)
    result: Set[str] = set()
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if t.isdigit():
            continue
        # 末尾の円・倍などを除去した数字も除外
        if re.fullmatch(r"[\d,]+[円倍票点]?", t):
            continue
        if t in _NOISE:
            continue
        result.add(t)
    return result


def _check_bet_hit_tickets(
    bet_type: str,
    tickets: List[Dict[str, Any]],
    finish_order: List[str],
) -> bool:
    """推奨買い目が的中したか（result_store.check_bet_hit と同一ロジック）。"""
    if not finish_order:
        return False
    top1 = finish_order[0] if finish_order else ""
    top3 = set(finish_order[:3])
    top2 = set(finish_order[:2])
    for ticket in tickets:
        combo = set(ticket.get("combination", []))
        if not combo:
            continue
        if bet_type == "単勝"                        and top1 in combo:            return True
        if bet_type == "複勝"                        and combo & top3:             return True
        if bet_type == "馬連"                        and combo <= top2:            return True
        if bet_type in ("ワイド", "ワイドBOX")       and len(combo & top3) >= 2:  return True
        if bet_type in ("3連複", "3連複BOX")         and combo <= top3:           return True
    return False


# =========================================================
# 各判定ロジック（1タグ = 1関数 で後から追加しやすくする）
# =========================================================

def _judge_head(record: Dict, finish_order: List[str], bought: Set[str]) -> List[str]:
    tags: List[str] = []
    heads = [h["horse_name"] for h in record.get("horses", []) if h.get("role") == "head"]
    if not heads:
        return tags
    top3 = _top3_set(finish_order)
    for h in heads:
        if h == (finish_order[0] if finish_order else ""):
            tags.append("頭候補的中")
        elif h in top3:
            tags.append("頭候補は2〜3着")
        else:
            tags.append("頭候補凡走")
    return tags


def _judge_axis_himo(record: Dict, finish_order: List[str], bought: Set[str]) -> List[str]:
    tags: List[str] = []
    axes  = [h["horse_name"] for h in record.get("horses", []) if h.get("role") == "axis"]
    himos = [h["horse_name"] for h in record.get("horses", []) if h.get("role") == "himo"]
    if not axes:
        return tags
    top3 = _top3_set(finish_order)
    axis_in_money = any(a in top3 for a in axes)
    himo_in_money = any(h in top3 for h in himos)
    himo_covered  = himo_in_money and any(h in bought for h in himos if h in top3)
    if axis_in_money:
        tags.append("軸的中")
        if himo_in_money and not himo_covered:
            tags.append("軸は来たがヒモ抜け")
    else:
        tags.append("軸も来ず")
    return tags


def _judge_danger(record: Dict, finish_order: List[str]) -> List[str]:
    tags: List[str] = []
    top3 = _top3_set(finish_order)
    for h in record.get("horses", []):
        if not h.get("is_danger_favorite"):
            continue
        name = h["horse_name"]
        is_truly = h.get("is_truly_dangerous", True)
        if is_truly:
            if name not in top3:
                tags.append("危険人気馬を切って正解")
            else:
                tags.append("危険人気馬を切って失敗")
        else:
            if name in top3:
                tags.append("相手残り危険馬が3着内")
    return tags


def _judge_value(record: Dict, finish_order: List[str], bought: Set[str]) -> List[str]:
    tags: List[str] = []
    top3 = _top3_set(finish_order)
    for h in record.get("horses", []):
        if not h.get("is_value_horse"):
            continue
        name = h["horse_name"]
        if name in top3:
            if name in bought:
                tags.append("妙味馬を活かした")
            else:
                tags.append("妙味馬は来たが買っていない")
        else:
            tags.append("妙味馬は着外")
    return tags


def _judge_rescue(record: Dict, finish_order: List[str], bought: Set[str]) -> List[str]:
    tags: List[str] = []
    top3 = _top3_set(finish_order)
    for h in record.get("horses", []):
        if not h.get("is_rescue_candidate"):
            continue
        name = h["horse_name"]
        if name in top3:
            if name in bought:
                tags.append("取りこぼし注意馬を活かした")
            else:
                tags.append("取りこぼし注意馬を拾えなかった")
    return tags


def _judge_bet_type(record: Dict) -> List[str]:
    bet_type   = record.get("recommended_bet_type", "")
    structure  = record.get("structure_type", "")
    if not bet_type or not structure:
        return []
    valid_types = BET_TYPE_STRUCTURE_MAP.get(structure, [])
    if bet_type in valid_types:
        return ["券種選択適切"]
    return ["券種選択ミス"]


def _judge_pace(record: Dict, finish_order: List[str]) -> List[str]:
    favorable  = record.get("favorable_style", "")
    if not favorable or not finish_order:
        return []
    top_n = finish_order[:PACE_CHECK_TOP_N]
    horses_by_name = {h["horse_name"]: h for h in record.get("horses", [])}
    # running_style は feature ではなく role_dict に保存されていないため、
    # jockey_summary や reason の文字列でフォールバック判定する
    # 完全な判定は running_style が horses に含まれる場合のみ
    style_jp = STYLE_JP.get(favorable, "")
    match = 0
    for name in top_n:
        h = horses_by_name.get(name, {})
        reason = h.get("reason", "") or ""
        if style_jp and style_jp in reason:
            match += 1
    # 上位2頭のうち1頭でも一致 → 適切
    if match >= 1:
        return ["展開判断適切"]
    return ["展開判断ミス"]


def _judge_result(
    record: Dict,
    finish_order: List[str],
    is_pass: bool,
    hit: Optional[bool],
    tickets: List[Dict[str, Any]],
    bet_type: str,
) -> List[str]:
    tags: List[str] = []
    if is_pass:
        if finish_order and tickets:
            would_hit = _check_bet_hit_tickets(bet_type, tickets, finish_order)
            tags.append("見送りだが買えば的中" if would_hit else "見送り")
        else:
            tags.append("見送り")
        return tags
    if hit is True:
        tags.append("的中")
    elif hit is False:
        tags.append("不的中")
    return tags


# =========================================================
# メイン関数
# =========================================================

def generate_review_tags(
    record: Dict[str, Any],
    actual_bets_text: str = "",
) -> List[str]:
    """
    レコードから日本語レビュータグリストを生成する。

    Parameters
    ----------
    record           : result_store.build_race_record で構築したレコード
                       (result フィールドに finish_order が入力済みであること)
    actual_bets_text : ユーザーが実際に購入した内容のテキスト（任意）
                       「妙味馬は来たが買っていない」などの判定に使用

    Returns
    -------
    tags : List[str]  日本語タグ名のリスト（重複除去済み）
    """
    result_data  = record.get("result") or {}
    is_pass      = record.get("is_pass", False)
    finish_order = result_data.get("finish_order") or []
    hit          = result_data.get("hit")
    tickets      = record.get("recommended_tickets") or []
    bet_type     = record.get("recommended_bet_type") or ""

    # 買い目に含まれる馬名（推奨 + 実際）
    rec_bought    = _bought_names(tickets)
    actual_bought = _actual_bought_names(actual_bets_text)
    # 実際の購入がある場合はそちらを優先、なければ推奨買い目で代替
    effective_bought = actual_bought if actual_bought else rec_bought

    if not finish_order and not is_pass:
        return []

    tags: List[str] = []
    tags += _judge_result(record, finish_order, is_pass, hit, tickets, bet_type)
    tags += _judge_head(record, finish_order, effective_bought)
    tags += _judge_axis_himo(record, finish_order, effective_bought)
    tags += _judge_danger(record, finish_order)
    tags += _judge_value(record, finish_order, effective_bought)
    tags += _judge_rescue(record, finish_order, effective_bought)
    tags += _judge_bet_type(record)
    tags += _judge_pace(record, finish_order)

    # 重複除去（順序保持）
    seen: Set[str] = set()
    unique: List[str] = []
    for t in tags:
        if t not in seen and t in REVIEW_TAGS:
            seen.add(t)
            unique.append(t)
    return unique


def build_review_result(
    record: Dict[str, Any],
    actual_bets_text: str = "",
) -> Dict[str, Any]:
    """
    レビュー結果を構造化して返す。

    Returns
    -------
    {
        "review_tags":         List[str],      # 日本語タグ
        "tags_by_category":    Dict[str, List[str]],
        "tags_by_level":       Dict[str, List[str]],
        "good_count":          int,
        "bad_count":           int,
        "summary":             str,            # 1〜2文のサマリー
        "return_rate":         float | None,   # 回収率（roi の別名）
        "hit":                 bool | None,
        "invested":            int,
        "payout":              int,
    }
    """
    result_data = record.get("result") or {}
    tags        = generate_review_tags(record, actual_bets_text)
    by_cat      = tags_by_category(tags)
    by_lvl      = tags_by_level(tags)

    good_count  = len(by_lvl.get("good", []))
    bad_count   = len(by_lvl.get("bad", []))

    # サマリー生成
    summary     = _build_summary(tags, by_lvl, record.get("is_pass", False))

    invested    = int(result_data.get("investment_amount") or 0)
    payout      = int(result_data.get("return_amount") or 0)
    roi         = result_data.get("roi")
    return_rate = round(roi, 4) if roi is not None else (
        round(payout / invested, 4) if invested > 0 else None
    )

    return {
        "review_tags":      tags,
        "tags_by_category": by_cat,
        "tags_by_level":    by_lvl,
        "good_count":       good_count,
        "bad_count":        bad_count,
        "summary":          summary,
        "return_rate":      return_rate,
        "hit":              result_data.get("hit"),
        "invested":         invested,
        "payout":           payout,
    }


def _build_summary(
    tags: List[str],
    by_lvl: Dict[str, List[str]],
    is_pass: bool,
) -> str:
    """タグリストから1〜2文のサマリーを生成する。"""
    if is_pass:
        if "見送りだが買えば的中" in tags:
            return "見送りを選んだが、推奨買い目を買っていれば的中していた。閾値の見直しを検討したい。"
        return "EVが閾値未満で見送りを選択。波乱ないし難解なレースだった。"

    bads  = by_lvl.get("bad", [])
    goods = by_lvl.get("good", [])

    parts = []
    if "的中" in tags:
        parts.append("的中。")
    elif "不的中" in tags:
        parts.append("不的中。")

    if bads:
        parts.append("改善点: " + " / ".join(bads[:2]) + "。")
    if goods:
        parts.append("良かった点: " + " / ".join(goods[:2]) + "。")

    return "".join(parts) if parts else "レース結果を記録しました。"
