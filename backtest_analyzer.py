"""
backtest_analyzer.py
レビュー集計・改善分析モジュール

責務:
- review_tags の頻度集計（外れ方ランキング）
- 役割判定精度集計（head / axis / himo / fade）
- 見送り判断の妥当性集計
- 集計結果から改善候補コメントを生成

フェーズ6以降の自動閾値最適化に向け、
閾値は全て COMMENT_THRESHOLDS に集約している。
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from review_config import REVIEW_TAGS


# =========================================================
# 閾値・定数
# =========================================================

# 改善コメント生成の判定閾値（後から1箇所で調整可能）
COMMENT_THRESHOLDS: Dict[str, float] = {
    "head_win_rate_low":       0.35,   # head 勝率がこれ未満で警告
    "axis_connect_rate_low":   0.40,   # axis 連対率がこれ未満で警告
    "himo_top3_rate_low":      0.30,   # himo 3着内率がこれ未満で警告
    "danger_correct_low":      0.55,   # 消し推奨正解率がこれ未満で警告
    "value_in_money_low":      0.28,   # 妙味馬3着内率がこれ未満で警告
    "pass_would_hit_high":     0.30,   # 見送りで的中しそうが多い → 閾値が高すぎ
    "roi_good":                1.05,   # この回収率以上は継続推奨
    "roi_bad":                 0.85,   # この回収率以下は要見直し
    "avg_tickets_high":        4.5,    # 平均点数がこれ超で「点数過多」警告
    "rescue_in_money_low":     0.28,   # 取りこぼし注意馬3着内率がこれ未満で警告
    "value_wrong_rate_high":   0.70,   # 妙味馬の着外率がこれ超で「拾いすぎ」警告
    "min_samples":             3,      # この件数以上ないとコメントを出さない
}

# 集計対象タグ（review_config.REVIEW_TAGS にあるものは全て集計するが
# このリストに含まれるものは外れ方ランキングの「着目タグ」として強調する）
TRACKED_TAGS: List[str] = [
    "不的中",
    "頭候補凡走",
    "軸も来ず",
    "軸は来たがヒモ抜け",
    "危険人気馬を切って失敗",
    "妙味馬は来たが買っていない",
    "取りこぼし注意馬を拾えなかった",
    "券種選択ミス",
    "展開判断ミス",
    "見送りだが買えば的中",
    "的中",
    "頭候補的中",
    "軸的中",
    "危険人気馬を切って正解",
    "妙味馬を活かした",
    "取りこぼし注意馬を活かした",
    "見送り",
]


# =========================================================
# 内部ヘルパー
# =========================================================

def _has_result(record: Dict[str, Any]) -> bool:
    r = record.get("result")
    return isinstance(r, dict) and bool(r.get("finish_order"))


def _top2_set(finish_order: List[str]) -> set:
    return set(finish_order[:2])


def _top3_set(finish_order: List[str]) -> set:
    return set(finish_order[:3])


def _record_tags(record: Dict[str, Any]) -> List[str]:
    """レコードの result から review_tags を返す（旧フィールド互換）。"""
    result_data = record.get("result") or {}
    tags = result_data.get("review_tags") or result_data.get("review_labels") or []
    return [t for t in tags if t in REVIEW_TAGS]


# =========================================================
# 役割判定精度
# =========================================================

def summarize_role_performance(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    head / axis / himo / fade の役割判定精度を集計する。

    Returns
    -------
    {
        "head":  {n_labeled, n_won, win_rate, n_connected, connect_rate, n_placed, place_rate},
        "axis":  同上,
        "himo":  同上,
        "fade":  同上,   # place_rate が低いほど消し精度が高い
    }
    """
    completed = [r for r in records if _has_result(r)]

    counts: Dict[str, Dict[str, int]] = {
        role: {"n_labeled": 0, "n_won": 0, "n_connected": 0, "n_placed": 0}
        for role in ("head", "axis", "himo", "fade")
    }

    for r in completed:
        finish_order = r["result"].get("finish_order", [])
        top1 = finish_order[0] if finish_order else ""
        top2 = _top2_set(finish_order)
        top3 = _top3_set(finish_order)

        for h in r.get("horses", []):
            role = h.get("role", "")
            if role not in counts:
                continue
            name = h["horse_name"]
            counts[role]["n_labeled"] += 1
            if name == top1:
                counts[role]["n_won"] += 1
            if name in top2:
                counts[role]["n_connected"] += 1
            if name in top3:
                counts[role]["n_placed"] += 1

    def _rates(d: Dict[str, int]) -> Dict[str, Any]:
        n = max(d["n_labeled"], 1)
        return {
            "n_labeled":    d["n_labeled"],
            "n_won":        d["n_won"],
            "n_connected":  d["n_connected"],
            "n_placed":     d["n_placed"],
            "win_rate":     round(d["n_won"]       / n, 4),
            "connect_rate": round(d["n_connected"] / n, 4),
            "place_rate":   round(d["n_placed"]    / n, 4),
        }

    return {role: _rates(d) for role, d in counts.items()}


# =========================================================
# 見送り判断の妥当性
# =========================================================

def summarize_pass_judgment(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    見送り（is_pass=True）レースのうち、
    実際に買えば的中していた割合を集計する。

    Returns
    -------
    {
        "n_pass":          int,
        "n_would_hit":     int,   # 見送りだが買えば的中だったケース
        "n_truly_pass":    int,   # 見送りかつ外れ（正解の見送り）
        "n_no_result":     int,   # 結果未入力の見送り
        "would_hit_rate":  float,
        "truly_pass_rate": float,
    }
    """
    pass_records = [r for r in records if r.get("is_pass", False)]
    n_pass       = len(pass_records)
    n_would_hit  = 0
    n_truly_pass = 0
    n_no_result  = 0

    for r in pass_records:
        if not _has_result(r) and not r.get("result"):
            n_no_result += 1
            continue
        tags = _record_tags(r)
        if "見送りだが買えば的中" in tags:
            n_would_hit += 1
        elif "見送り" in tags:
            n_truly_pass += 1

    return {
        "n_pass":          n_pass,
        "n_would_hit":     n_would_hit,
        "n_truly_pass":    n_truly_pass,
        "n_no_result":     n_no_result,
        "would_hit_rate":  round(n_would_hit  / max(n_pass, 1), 4),
        "truly_pass_rate": round(n_truly_pass / max(n_pass, 1), 4),
    }


# =========================================================
# 外れ方ランキング（review_tags 集計）
# =========================================================

def aggregate_review_tags(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    全レコードの review_tags を集計し、出現頻度ランキングを返す。
    結果未入力のレコードは除外する。

    Returns
    -------
    List of {tag, count, rate, level, category}
    sorted by count desc, then tag asc
    """
    completed = [r for r in records if _has_result(r)]
    n_races = max(len(completed), 1)
    counter: Counter = Counter()

    for r in completed:
        tags = _record_tags(r)
        counter.update(tags)

    rows = []
    for tag, count in counter.most_common():
        meta = REVIEW_TAGS.get(tag, {})
        rows.append({
            "tag":      tag,
            "count":    count,
            "rate":     round(count / n_races, 4),
            "level":    meta.get("level", "neutral"),
            "category": meta.get("category", "その他"),
        })
    return rows


# =========================================================
# 改善候補コメント生成
# =========================================================

def generate_improvement_comments(analytics: Dict[str, Any]) -> List[str]:
    """
    build_full_analytics() の戻り値を受け取り、
    改善候補コメントのリストを返す。

    Parameters
    ----------
    analytics : analytics_ai.build_full_analytics() の戻り値

    Returns
    -------
    List[str]
    """
    comments: List[str] = []
    thr = COMMENT_THRESHOLDS
    min_n = int(thr["min_samples"])

    overall  = analytics.get("overall", {})
    by_bt    = analytics.get("by_bet_type", [])
    role_p   = analytics.get("role_performance", {})
    pass_j   = analytics.get("pass_judgment", {})
    danger   = analytics.get("danger_cutoff", {})
    value    = analytics.get("value_horse", {})
    rescue   = analytics.get("rescue_horse", {})
    tag_rank = analytics.get("tag_ranking", [])

    n_races = overall.get("n_races", 0)
    if n_races < min_n:
        comments.append(
            f"データが {n_races} 件と少ないです（{min_n} 件以上で精度が上がります）。"
        )
        return comments

    # ── 全体回収率 ────────────────────────────────────────────────────
    roi = overall.get("roi")
    if roi is not None:
        if roi >= thr["roi_good"]:
            comments.append(
                f"通算回収率 {roi*100:.1f}% — プラス収支。現状のアプローチを継続推奨。"
            )
        elif roi <= thr["roi_bad"]:
            comments.append(
                f"通算回収率 {roi*100:.1f}% — 要見直し水準。下記の改善ポイントを優先してください。"
            )

    # ── 平均点数 ──────────────────────────────────────────────────────
    avg_tix = overall.get("avg_tickets", 0)
    if avg_tix > thr["avg_tickets_high"]:
        comments.append(
            f"平均点数 {avg_tix:.1f}点 — 点数過多の傾向。"
            f"買い目を絞ることで1点あたりの期待値が向上します。"
        )

    # ── 券種別：高い・低い回収率 ─────────────────────────────────────
    eligible_bt = [row for row in by_bt if (row.get("roi") is not None) and row.get("n_races", 0) >= min_n]
    if eligible_bt:
        best_bt  = max(eligible_bt, key=lambda r: r["roi"])
        worst_bt = min(eligible_bt, key=lambda r: r["roi"])
        if best_bt["roi"] >= thr["roi_good"]:
            comments.append(
                f"【券種】{best_bt['bet_type']} の回収率 {best_bt['roi']*100:.1f}% が最高。"
                f"このレース構造での優先券種として継続推奨。"
            )
        if worst_bt["roi"] <= thr["roi_bad"] and worst_bt["bet_type"] != best_bt.get("bet_type"):
            comments.append(
                f"【券種】{worst_bt['bet_type']} の回収率 {worst_bt['roi']*100:.1f}% が最低。"
                f"この券種での購入頻度を下げることを検討してください。"
            )

    # ── head 候補勝率 ─────────────────────────────────────────────────
    head_s = role_p.get("head", {})
    if head_s.get("n_labeled", 0) >= min_n:
        wr = head_s["win_rate"]
        if wr < thr["head_win_rate_low"]:
            comments.append(
                f"【役割】頭候補の勝率 {wr*100:.1f}%（目標 {thr['head_win_rate_low']*100:.0f}% 以上）。"
                f"WIN_PROB_HEAD_MIN / AXIS_SCORE_HEAD_MIN の引き上げを検討してください。"
            )
        else:
            comments.append(f"【役割】頭候補の勝率 {wr*100:.1f}% — 良好。")

    # ── axis 連対率 ───────────────────────────────────────────────────
    axis_s = role_p.get("axis", {})
    if axis_s.get("n_labeled", 0) >= min_n:
        cr = axis_s["connect_rate"]
        if cr < thr["axis_connect_rate_low"]:
            comments.append(
                f"【役割】軸候補の連対率 {cr*100:.1f}%（目標 {thr['axis_connect_rate_low']*100:.0f}% 以上）。"
                f"stable_score / axis_score の閾値を上げて軸の選別を厳しくすることを推奨。"
            )

    # ── himo 3着内率 ─────────────────────────────────────────────────
    himo_s = role_p.get("himo", {})
    if himo_s.get("n_labeled", 0) >= min_n:
        pr = himo_s["place_rate"]
        if pr < thr["himo_top3_rate_low"]:
            comments.append(
                f"【役割】ヒモ候補の3着内率 {pr*100:.1f}%（目標 {thr['himo_top3_rate_low']*100:.0f}% 以上）。"
                f"top3_prob / TOP3_HIMO_MIN の引き上げによる絞り込みを推奨。"
            )

    # ── 危険馬判定精度 ────────────────────────────────────────────────
    d_total   = danger.get("truly_total", 0)
    d_correct = danger.get("danger_truly_correct_rate", 1.0)
    if d_total >= min_n:
        if d_correct < thr["danger_correct_low"]:
            comments.append(
                f"【危険馬】消し推奨の正解率 {d_correct*100:.1f}%（目標 {thr['danger_correct_low']*100:.0f}% 以上）。"
                f"危険馬判定が強すぎる可能性。DANGER_V3_AXIS_MAX / DANGER_V3_TOP3_MAX を緩める方向で調整を。"
            )
        elif d_correct >= 0.70:
            comments.append(f"【危険馬】消し推奨正解率 {d_correct*100:.1f}% — 高精度。")

    # ── 妙味馬3着内率 / 拾いすぎ率 ──────────────────────────────────────────
    v_total      = value.get("value_total", 0)
    v_rate       = value.get("value_in_money_rate", 0)
    v_wrong_rate = value.get("value_wrong_rate", 0)
    v_wrong_n    = value.get("value_wrong_count", 0)
    if v_total >= min_n:
        if v_wrong_rate > thr["value_wrong_rate_high"]:
            comments.append(
                f"【妙味馬・拾いすぎ】着外 {v_wrong_n}回（着外率 {v_wrong_rate*100:.1f}%）。"
                f"VALUE_MIN_WIN_PROB / VALUE_MIN_STABLE_SCORE の引き上げを推奨。"
            )
        elif v_rate < thr["value_in_money_low"]:
            comments.append(
                f"【妙味馬】3着内率 {v_rate*100:.1f}%（目標 {thr['value_in_money_low']*100:.0f}% 以上）。"
                f"VALUE_GAP_MIN を引き上げて、より厳選した妙味馬のみを対象にすることを推奨。"
            )

    # ── 取りこぼし注意馬3着内率 ──────────────────────────────────────
    rsc_total = rescue.get("rescue_total", 0)
    rsc_rate  = rescue.get("rescue_in_money_rate", 0)
    if rsc_total >= min_n and rsc_rate < thr["rescue_in_money_low"]:
        comments.append(
            f"【取りこぼし】3着内率 {rsc_rate*100:.1f}%（目標 {thr['rescue_in_money_low']*100:.0f}% 以上）。"
            f"TOP3_RESCUE_MIN / UPSIDE_SCORE_HIGH の閾値見直しを推奨。"
        )

    # ── 見送り判断 ────────────────────────────────────────────────────
    n_pass = pass_j.get("n_pass", 0)
    if n_pass >= min_n:
        whr = pass_j.get("would_hit_rate", 0)
        if whr > thr["pass_would_hit_high"]:
            comments.append(
                f"【見送り】見送りで的中しそうだった割合 {whr*100:.1f}% — 見送り閾値が高すぎる可能性。"
                f"EV_SKIP_THRESHOLD / EV_COMPOUND_SKIP を少し引き下げることを検討してください。"
            )

    # ── 頻出ミスタグ（上位3件） ──────────────────────────────────────
    bad_tags = [t for t in tag_rank if t["level"] == "bad"][:3]
    if bad_tags:
        tag_str = " / ".join(f"{t['tag']}({t['count']}回)" for t in bad_tags)
        comments.append(
            f"【頻出ミス】{tag_str} が多発しています。"
            f"対応する判定ロジックの精度向上を優先してください。"
        )

    if not comments:
        comments.append(
            "現状のデータでは特定の改善ポイントが見つかりませんでした。"
            "レース数を増やすとより精度の高い示唆が出ます。"
        )
    return comments
