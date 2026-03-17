"""
review_ai.py
レース後回顧モジュール

責務:
- 保存レコード + 実着順から回顧ラベル・テキストを生成
- ルールベースで判定（後からLLMに差し替えやすい構造）
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

# =========================================================
# 回顧ラベル定義（machine-readable）
# =========================================================

REVIEW_LABEL_TEXT: Dict[str, str] = {
    # 頭候補
    "HEAD_WON":             "頭候補が1着：単勝・馬連軸として正解",
    "HEAD_PLACED":          "頭候補が2〜3着：頭は外れたが馬券内",
    "HEAD_MISSED":          "頭候補が着外：頭の読みを外した",
    # 軸候補
    "AXIS_IN_MONEY":        "軸候補が馬券内：軸の読みは正解",
    "AXIS_MISSED":          "軸候補が全員着外：軸を外した",
    # ヒモ候補
    "HIMO_COVERED":         "ヒモ候補を買い目に含められた",
    "HIMO_MISSED":          "ヒモ候補が着内だが買い目になかった：ヒモ選定不足",
    # 危険人気馬
    "DANGER_TRULY_CORRECT": "消し推奨の危険人気馬が着外：危険馬判定は正解",
    "DANGER_TRULY_WRONG":   "消し推奨の危険人気馬が馬券内：切り判断がミス",
    "DANGER_SOFT_CORRECT":  "「相手なら残る」危険馬が着外",
    "DANGER_SOFT_PLACED":   "「相手なら残る」危険馬が3着内：連から除外が妥当だった",
    "DANGER_SOFT_WRONG":    "「相手なら残る」危険馬が1〜2着：頭として読めなかった",
    # 妙味馬
    "VALUE_CAPTURED":       "妙味馬が馬券内で買い目にも含めた：価値判断◎",
    "VALUE_IN_MONEY":       "妙味馬が馬券内だが買い目になかった：見逃した",
    "VALUE_MISSED":         "妙味馬が着外：value判断がズレていた",
    # 取りこぼし注意馬
    "RESCUE_IN_MONEY":      "取りこぼし注意馬が馬券内：警告を活かせた",
    "RESCUE_IGNORED":       "取りこぼし注意馬が馬券内なのに買い目外：取りこぼした",
    "RESCUE_CORRECT":       "取りこぼし注意馬が着外：ノイズではなかった",
    # レース構造
    "STRUCTURE_CORRECT":    "展開予想（有利脚質）と実際の上位馬の脚質が一致",
    "STRUCTURE_MISMATCH":   "展開予想と実際の展開が乖離した可能性",
    # 券種
    "BET_TYPE_VALID":       "推奨券種はレース構造に合っていた",
    "BET_TYPE_SUBOPTIMAL":  "実際の結果から見ると別の券種が有効だった可能性",
    # 全体
    "HIT":                  "的中",
    "MISS":                 "不的中",
    "PASS":                 "見送り（正解かどうかは結果次第）",
    "PASS_WOULD_HIT":       "見送りを選んだが買っていれば的中だった",
    "PASS_CORRECT":         "見送りを選んだ：買い目なしで正解（レース荒れ/難解）",
}

REVIEW_LEVEL: Dict[str, str] = {
    "HEAD_WON":             "good",
    "HEAD_PLACED":          "neutral",
    "HEAD_MISSED":          "bad",
    "AXIS_IN_MONEY":        "good",
    "AXIS_MISSED":          "bad",
    "HIMO_COVERED":         "good",
    "HIMO_MISSED":          "bad",
    "DANGER_TRULY_CORRECT": "good",
    "DANGER_TRULY_WRONG":   "bad",
    "DANGER_SOFT_CORRECT":  "neutral",
    "DANGER_SOFT_PLACED":   "good",
    "DANGER_SOFT_WRONG":    "bad",
    "VALUE_CAPTURED":       "good",
    "VALUE_IN_MONEY":       "bad",
    "VALUE_MISSED":         "neutral",
    "RESCUE_IN_MONEY":      "good",
    "RESCUE_IGNORED":       "bad",
    "RESCUE_CORRECT":       "neutral",
    "STRUCTURE_CORRECT":    "good",
    "STRUCTURE_MISMATCH":   "bad",
    "BET_TYPE_VALID":       "good",
    "BET_TYPE_SUBOPTIMAL":  "neutral",
    "HIT":                  "good",
    "MISS":                 "bad",
    "PASS":                 "neutral",
    "PASS_WOULD_HIT":       "bad",
    "PASS_CORRECT":         "good",
}


# =========================================================
# 内部ヘルパー
# =========================================================

def _get_horse_roles(record: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    """head / axis / himo 馬名リストを返す。"""
    heads, axes, himos = [], [], []
    for h in record.get("horses", []):
        role = h.get("role", "")
        name = h.get("horse_name", "")
        if role == "head":
            heads.append(name)
        elif role == "axis":
            axes.append(name)
        elif role == "himo":
            himos.append(name)
    return heads, axes, himos


def _get_danger_names(record: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """(truly_dangerous_names, soft_dangerous_names) を返す。"""
    # result_store には is_danger_favorite しか保存していないため、
    # レコード保存時に is_truly_dangerous も horses に保持するよう拡張予定。
    # 現状はフォールバックとして is_danger_favorite をそのまま使う。
    truly, soft = [], []
    for h in record.get("horses", []):
        if h.get("is_danger_favorite"):
            # is_truly_dangerous があれば使う（build_race_record 拡張後）
            if h.get("is_truly_dangerous", True):
                truly.append(h["horse_name"])
            else:
                soft.append(h["horse_name"])
    return truly, soft


def _combo_in_tickets(
    tickets: List[Dict[str, Any]], finish_order: List[str], bet_type: str
) -> bool:
    """買い目が着順と照合して的中か否かを返す（result_store.check_bet_hit と同一ロジック）。"""
    from result_store import check_bet_hit
    return check_bet_hit(bet_type, tickets, finish_order)


# =========================================================
# 回顧アイテム生成ヘルパー
# =========================================================

def _item(label: str, detail: str = "") -> Dict[str, Any]:
    text = REVIEW_LABEL_TEXT.get(label, label)
    if detail:
        text = f"{text}（{detail}）"
    return {
        "label": label,
        "level": REVIEW_LEVEL.get(label, "neutral"),
        "text":  text,
    }


# =========================================================
# 各観点の判定ロジック
# =========================================================

def _review_head(
    items: List, heads: List[str], finish_order: List[str]
) -> None:
    if not heads:
        return
    top3 = finish_order[:3]
    for head in heads:
        if head == finish_order[0]:
            items.append(_item("HEAD_WON", head))
        elif head in top3:
            items.append(_item("HEAD_PLACED", head))
        else:
            items.append(_item("HEAD_MISSED", head))


def _review_axis(
    items: List, axes: List[str], finish_order: List[str]
) -> None:
    if not axes:
        return
    top3 = set(finish_order[:3])
    in_money = [a for a in axes if a in top3]
    if in_money:
        items.append(_item("AXIS_IN_MONEY", "・".join(in_money)))
    else:
        items.append(_item("AXIS_MISSED", "・".join(axes)))


def _review_himo(
    items: List,
    himos: List[str],
    finish_order: List[str],
    bet_tickets: List[Dict[str, Any]],
) -> None:
    if not himos:
        return
    top3 = set(finish_order[:3])
    # 馬券内だったヒモ候補
    bought_names: set = set()
    for t in bet_tickets:
        bought_names.update(t.get("combination", []))

    for himo in himos:
        if himo in top3:
            if himo in bought_names:
                items.append(_item("HIMO_COVERED", himo))
            else:
                items.append(_item("HIMO_MISSED", himo))


def _review_danger(
    items: List,
    truly: List[str],
    soft: List[str],
    finish_order: List[str],
) -> None:
    top1 = finish_order[0] if finish_order else ""
    top3 = set(finish_order[:3])
    top2 = set(finish_order[:2])

    for name in truly:
        if name not in top3:
            items.append(_item("DANGER_TRULY_CORRECT", name))
        else:
            items.append(_item("DANGER_TRULY_WRONG", name))

    for name in soft:
        if name not in top3:
            items.append(_item("DANGER_SOFT_CORRECT", name))
        elif name == top1 or name in top2:
            items.append(_item("DANGER_SOFT_WRONG", name))
        else:
            items.append(_item("DANGER_SOFT_PLACED", name))


def _review_value(
    items: List,
    record: Dict[str, Any],
    finish_order: List[str],
    bet_tickets: List[Dict[str, Any]],
) -> None:
    top3 = set(finish_order[:3])
    bought: set = set()
    for t in bet_tickets:
        bought.update(t.get("combination", []))

    for h in record.get("horses", []):
        if not h.get("is_value_horse"):
            continue
        name = h["horse_name"]
        if name in top3:
            if name in bought:
                items.append(_item("VALUE_CAPTURED", name))
            else:
                items.append(_item("VALUE_IN_MONEY", name))
        else:
            items.append(_item("VALUE_MISSED", name))


def _review_rescue(
    items: List,
    record: Dict[str, Any],
    finish_order: List[str],
    bet_tickets: List[Dict[str, Any]],
) -> None:
    top3 = set(finish_order[:3])
    bought: set = set()
    for t in bet_tickets:
        bought.update(t.get("combination", []))

    for h in record.get("horses", []):
        if not h.get("is_rescue_candidate"):
            continue
        name = h["horse_name"]
        if name in top3:
            if name in bought:
                items.append(_item("RESCUE_IN_MONEY", name))
            else:
                items.append(_item("RESCUE_IGNORED", name))
        else:
            items.append(_item("RESCUE_CORRECT", name))


def _review_structure(
    items: List,
    record: Dict[str, Any],
    finish_order: List[str],
    features_by_name: Dict[str, Dict[str, Any]],
) -> None:
    favorable = record.get("favorable_style", "")
    if not favorable or not finish_order:
        return
    top3 = finish_order[:3]

    match_count = 0
    for name in top3:
        h = next((x for x in record.get("horses", []) if x["horse_name"] == name), {})
        # 脚質は features から取れないため horses に保存した role で代替
        # 実際には running_style を horses に保存するのが望ましい
        role = h.get("role", "")
        # 頭候補・軸候補が先行系の場合に先行有利と判定するのは難しいため、
        # 現状は 1着馬の役割で大まかに判定する
    # シンプル判定: top3 の馬が期待された role かどうか
    top1_horse = next((x for x in record.get("horses", []) if x["horse_name"] == top3[0]), {})
    top1_role = top1_horse.get("role", "")
    if top1_role in ("head", "axis"):
        items.append(_item("STRUCTURE_CORRECT"))
    else:
        items.append(_item("STRUCTURE_MISMATCH"))


def _review_bet_type(
    items: List,
    record: Dict[str, Any],
    finish_order: List[str],
) -> None:
    bet_type = record.get("recommended_bet_type", "")
    structure = record.get("structure_type", "")
    if not bet_type:
        return

    # 波乱型 → ワイドが有効
    if structure in ("波乱型", "混戦型") and bet_type in ("ワイド", "3連複"):
        items.append(_item("BET_TYPE_VALID", f"{structure}→{bet_type}"))
    elif structure in ("本命信頼型", "先行有利型") and bet_type in ("単勝", "複勝", "馬連"):
        items.append(_item("BET_TYPE_VALID", f"{structure}→{bet_type}"))
    else:
        items.append(_item("BET_TYPE_SUBOPTIMAL", f"{structure}→{bet_type}"))


# =========================================================
# サマリー文生成
# =========================================================

def _build_summary(items: List[Dict[str, Any]], is_pass: bool, hit: bool) -> str:
    """回顧アイテムリストから1〜3文の要約を生成する。"""
    goods = [i for i in items if i["level"] == "good"]
    bads  = [i for i in items if i["level"] == "bad"]

    if is_pass:
        hit_items = [i for i in items if "WOULD_HIT" in i["label"]]
        if hit_items:
            return "見送りを選んだが、買っていれば的中だった。次回は慎重に閾値を見直したい。"
        return "見送り判断。レース構造が読みにくく、回避は妥当だった可能性が高い。"

    parts = []
    if hit:
        parts.append("的中。")
    else:
        parts.append("不的中。")

    bad_texts = [i["text"] for i in bads[:2]]
    if bad_texts:
        parts.append("課題: " + " / ".join(bad_texts) + "。")

    good_texts = [i["text"] for i in goods[:2]]
    if good_texts:
        parts.append("良かった点: " + " / ".join(good_texts) + "。")

    return "".join(parts)


# =========================================================
# メイン関数
# =========================================================

def build_review(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    保存済みレコード（result フィールド入力済み）から回顧を生成する。

    Returns
    -------
    {
        "labels":  List[str],               # 機械可読ラベルコード一覧
        "items":   List[{label, level, text}],  # 判定ごとの詳細
        "summary": str,                     # 1〜3文の要約テキスト
        "hit":     bool | None,
        "roi":     float | None,
    }
    """
    result_data = record.get("result") or {}
    is_pass     = record.get("is_pass", False)

    finish_order  = result_data.get("finish_order", [])
    hit           = result_data.get("hit")
    roi           = result_data.get("roi")
    bet_type      = record.get("recommended_bet_type", "")
    tickets       = record.get("recommended_tickets", [])

    items: List[Dict[str, Any]] = []

    # 見送り
    if is_pass:
        # 実着順が入力されている場合に「買っていれば的中だったか」を判定
        if finish_order and tickets:
            would_hit = _combo_in_tickets(tickets, finish_order, bet_type)
            items.append(_item("PASS_WOULD_HIT" if would_hit else "PASS_CORRECT"))
        else:
            items.append(_item("PASS"))
        labels  = [i["label"] for i in items]
        summary = _build_summary(items, is_pass=True, hit=False)
        return {"labels": labels, "items": items, "summary": summary, "hit": None, "roi": None}

    if not finish_order:
        return {"labels": [], "items": [], "summary": "着順未入力のため回顧を生成できません。", "hit": hit, "roi": roi}

    # 各観点を評価
    heads, axes, himos = _get_horse_roles(record)
    truly_danger, soft_danger = _get_danger_names(record)

    _review_head(items, heads, finish_order)
    _review_axis(items, axes, finish_order)
    _review_himo(items, himos, finish_order, tickets)
    _review_danger(items, truly_danger, soft_danger, finish_order)
    _review_value(items, record, finish_order, tickets)
    _review_rescue(items, record, finish_order, tickets)
    _review_structure(items, record, finish_order, {})
    _review_bet_type(items, record, finish_order)

    # 的中ラベル
    if hit is True:
        items.append(_item("HIT"))
    elif hit is False:
        items.append(_item("MISS"))

    labels  = [i["label"] for i in items]
    summary = _build_summary(items, is_pass=False, hit=bool(hit))

    return {
        "labels":  labels,
        "items":   items,
        "summary": summary,
        "hit":     hit,
        "roi":     roi,
    }
