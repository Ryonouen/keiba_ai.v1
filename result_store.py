"""
result_store.py
レース結果保存・読み込みモジュール

責務:
- RaceRecord の JSON 読み書き（SQLite 移行を見越した構造）
- レース記録の構築（build_race_record）
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

# =========================================================
# 保存先パス
# =========================================================

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE: str = os.path.join(_HERE, "race_results.json")


# =========================================================
# CRUD
# =========================================================

def load_race_results() -> List[Dict[str, Any]]:
    """保存済みレース一覧を返す（なければ空リスト）。"""
    if not os.path.exists(RESULTS_FILE):
        return []
    try:
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_all(records: List[Dict[str, Any]]) -> None:
    """全レコードをアトミックに書き込む。"""
    dir_ = os.path.dirname(RESULTS_FILE) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, RESULTS_FILE)
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise


def save_race_result(record: Dict[str, Any]) -> None:
    """レース記録を保存（同一 race_id があれば上書き）。"""
    records = load_race_results()
    race_id = record.get("race_id") or ""
    idx = next((i for i, r in enumerate(records) if r.get("race_id") == race_id), None)
    if idx is not None:
        records[idx] = record
    else:
        records.append(record)
    _save_all(records)


def update_race_result(race_id: str, result_data: Dict[str, Any]) -> bool:
    """指定 race_id の result フィールドを更新する。"""
    records = load_race_results()
    for r in records:
        if r.get("race_id") == race_id:
            r["result"] = result_data
            # 各馬の finish_position も同期
            finish_order = result_data.get("finish_order", [])
            for h in r.get("horses", []):
                name = h.get("horse_name", "")
                try:
                    h["finish_position"] = finish_order.index(name) + 1
                except ValueError:
                    h["finish_position"] = None
            _save_all(records)
            return True
    return False


def delete_race_result(race_id: str) -> bool:
    """指定 race_id の記録を削除する。"""
    records = load_race_results()
    new_records = [r for r in records if r.get("race_id") != race_id]
    if len(new_records) == len(records):
        return False
    _save_all(new_records)
    return True


def get_race_result(race_id: str) -> Optional[Dict[str, Any]]:
    """指定 race_id の記録を返す（なければ None）。"""
    for r in load_race_results():
        if r.get("race_id") == race_id:
            return r
    return None


# =========================================================
# レコード構築
# =========================================================

def _extract_race_id(race_url: str) -> str:
    """URL から race_id を抽出。なければ UUID を生成。"""
    if race_url:
        m = re.search(r"race_id=(\d{10,12})", race_url)
        if m:
            return m.group(1)
    return str(uuid.uuid4())[:12]


def _extract_grade(race_name: str) -> str:
    for g in ["G1", "G2", "G3", "Ｇ１", "Ｇ２", "Ｇ３"]:
        if g in race_name:
            return g.replace("Ｇ", "G").replace("１", "1").replace("２", "2").replace("３", "3")
    return "重賞"


def build_race_record(
    result: Dict[str, Any],
    bankroll: int,
    bet_plan: Optional[Dict[str, Any]] = None,
    *,
    race_url: str = "",
    horse_roles: Optional[List[Dict[str, Any]]] = None,
    rescue_horses: Optional[List[Dict[str, Any]]] = None,
    danger_horses: Optional[List[Dict[str, Any]]] = None,
    value_horses: Optional[List[Dict[str, Any]]] = None,
    horse_marks: Optional[List[Dict[str, Any]]] = None,
    ev_table: Optional[List[Dict[str, Any]]] = None,
    race_structure: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    分析結果から保存用 RaceRecord を構築する。

    Parameters
    ----------
    result      : analyze_race() の戻り値（st.session_state.result）
    bankroll    : 軍資金（円）
    bet_plan    : recommend_bet_plan() の戻り値
    race_url    : 入力 URL（race_id 抽出に使用）
    その他      : 期待値AIタブで計算済みのデータを渡す（再計算を避けるため）
    """
    race_meta     = result.get("race_meta", {})
    features      = result.get("features", [])
    ev_table      = ev_table or result.get("ev_table") or []
    race_structure = race_structure or result.get("race_structure") or {}
    horse_marks   = horse_marks or result.get("horse_marks") or []
    value_horses  = value_horses or result.get("value_horses_v2") or []
    danger_horses = danger_horses or []
    horse_roles   = horse_roles or []
    rescue_horses = rescue_horses or []
    bet_plan      = bet_plan or {}

    race_id   = _extract_race_id(race_url)
    race_name = race_meta.get("race_title", "不明")
    race_date = datetime.now().strftime("%Y-%m-%d")

    # インデックス構築
    marks_by_name  = {r["horse_name"]: r["mark"]  for r in horse_marks}
    roles_by_name  = {r["horse_name"]: r           for r in horse_roles}
    ev_by_name     = {r["horse_name"]: r           for r in ev_table}
    rescue_names   = {r["horse_name"] for r in rescue_horses}
    danger_names   = {r["horse_name"] for r in danger_horses}
    value_names    = {r["horse_name"] for r in value_horses}

    # 馬ごとのデータ
    horse_records: List[Dict[str, Any]] = []
    for f in features:
        name   = str(f.get("horse_name") or "")
        ev_row = ev_by_name.get(name, {})
        role_d = roles_by_name.get(name, {})
        horse_records.append({
            "horse_name":         name,
            "mark":               marks_by_name.get(name, ""),
            "role":               role_d.get("role", ""),
            "ai_win_prob":        round(float(f.get("win_prob") or 0.0), 4),
            "top2_prob":          round(float(role_d.get("top2_prob") or 0.0), 4),
            "top3_prob":          round(float(role_d.get("top3_prob") or 0.0), 4),
            "stable_score":       round(float(role_d.get("stable_score") or 0.0), 4),
            "upside_score":       round(float(role_d.get("upside_score") or 0.0), 4),
            "axis_score":         round(float(role_d.get("axis_score") or 0.0), 4),
            "value_gap":          ev_row.get("value_gap"),
            "win_odds":           ev_row.get("win_odds"),
            "is_value_horse":     name in value_names,
            "is_danger_favorite": name in danger_names,
            "is_rescue_candidate": name in rescue_names,
            "finish_position":    None,
        })

    investment_amount = bet_plan.get("total_stake", 0) if not bet_plan.get("skip") else 0

    # 妙味馬・危険馬・役割・取りこぼし注意馬を集計用に独立したリストで保持
    value_horses_list  = [h["horse_name"] for h in horse_records if h.get("is_value_horse")]
    danger_horses_list = [h["horse_name"] for h in horse_records if h.get("is_danger_favorite")]
    rescue_horses_list = [h["horse_name"] for h in horse_records if h.get("is_rescue_candidate")]
    horse_roles_list   = [
        {"horse_name": h["horse_name"], "role": h.get("role", "")}
        for h in horse_records
    ]

    record: Dict[str, Any] = {
        # レース単位
        "race_id":               race_id,
        "race_name":             race_name,
        "race_date":             race_date,
        "race_course":           race_meta.get("target_course", ""),
        "race_grade":            _extract_grade(race_name),
        "race_info_text":        race_meta.get("race_info_text", ""),
        "predicted_pace":        race_meta.get("predicted_pace", "medium"),
        "race_structure": {
            "structure_type":  race_structure.get("structure_type", ""),
            "favorable_style": race_structure.get("favorable_style", ""),
            "upset_risk":      race_structure.get("upset_risk", 0.5),
        },
        # フラット版（既存との後方互換）
        "structure_type":        race_structure.get("structure_type", ""),
        "favorable_style":       race_structure.get("favorable_style", ""),
        "upset_risk":            race_structure.get("upset_risk", 0.5),
        "bankroll":              bankroll,
        "recommended_bet_plan": {
            "bet_type":  bet_plan.get("bet_type", ""),
            "horses":    bet_plan.get("horses", []),
            "tickets":   bet_plan.get("tickets", []),
            "reason":    bet_plan.get("reason", ""),
            "ev_type":   bet_plan.get("ev_type", ""),
            "risk_level": bet_plan.get("risk_level", ""),
            "is_pass":   bool(bet_plan.get("skip", False)),
        },
        # フラット版（既存との後方互換）
        "recommended_bet_type":  bet_plan.get("bet_type", ""),
        "recommended_horses":    bet_plan.get("horses", []),
        "recommended_tickets":   bet_plan.get("tickets", []),
        "recommended_reason":    bet_plan.get("reason", ""),
        "ev_type":               bet_plan.get("ev_type", ""),
        "risk_level":            bet_plan.get("risk_level", ""),
        "is_pass":               bool(bet_plan.get("skip", False)),
        "investment_amount":     investment_amount,
        # 集計用サマリーリスト
        "value_horses":          value_horses_list,
        "danger_favorites":      danger_horses_list,
        "horse_roles":           horse_roles_list,
        "rescue_candidates":     rescue_horses_list,
        # 馬単位（詳細）
        "horses":                horse_records,
        # 結果（後入力）
        "result":                None,
        "saved_at":              datetime.now().isoformat(timespec="seconds"),
    }
    return record


# =========================================================
# ヒット判定ユーティリティ（レビュー・UIから共用）
# =========================================================

def check_bet_hit(
    bet_type: str,
    tickets: List[Dict[str, Any]],
    finish_order: List[str],
) -> bool:
    """推奨買い目が的中したか判定する。"""
    if not finish_order:
        return False
    top1 = finish_order[0] if len(finish_order) > 0 else ""
    top2 = set(finish_order[:2])
    top3 = set(finish_order[:3])

    for ticket in tickets:
        combo = set(ticket.get("combination", []))
        if not combo:
            continue
        if bet_type == "単勝" and top1 in combo:
            return True
        if bet_type == "複勝" and combo & top3:
            return True
        if bet_type == "馬連" and combo <= top2:
            return True
        if bet_type == "ワイド" and len(combo & top3) >= 2:
            return True
        if bet_type == "3連複" and combo <= top3:
            return True
    return False
