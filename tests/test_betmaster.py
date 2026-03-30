"""tests/test_betmaster.py — AI馬券師推奨機能のユニットテスト"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from value_ai import recommend_betmaster_plans, select_primary_betmaster

# テスト用フィクスチャ: 18頭立て想定
FEATURES_18 = [
    {"horse_name": f"馬{i}", "win_prob": max(0.01, 0.25 - i * 0.013),
     "consistency_index": max(0.01, 0.7 - i * 0.03), "trend_index": max(0.01, 0.65 - i * 0.02),
     "popularity": i + 1}
    for i in range(18)
]

HORSE_ROLES_18 = [
    {"horse_name": "馬0", "role": "head"},
    {"horse_name": "馬1", "role": "axis"},
    {"horse_name": "馬2", "role": "axis"},
    {"horse_name": "馬3", "role": "himo"},
    {"horse_name": "馬4", "role": "himo"},
    {"horse_name": "馬5", "role": "himo"},
    {"horse_name": "馬6", "role": "fade"},
] + [{"horse_name": f"馬{i}", "role": "himo"} for i in range(7, 18)]

RACE_STRUCTURE = {"structure_type": "標準型", "favorable_style": "front"}


def test_recommend_betmaster_plans_returns_nine_types():
    plans = recommend_betmaster_plans(FEATURES_18, RACE_STRUCTURE, HORSE_ROLES_18)
    bet_types = [p["bet_type"] for p in plans]
    assert "単勝" in bet_types
    assert "複勝" in bet_types
    assert "ワイド" in bet_types
    assert "馬連（流し）" in bet_types
    assert "馬単フォーメーション" in bet_types
    assert "三連複フォーメーション（AI絞り）" in bet_types
    assert "三連複フォーメーション（全頭）" in bet_types
    assert "三連単フォーメーション（AI絞り）" in bet_types
    assert "三連単フォーメーション（全頭）" in bet_types


def test_all_tickets_are_100_yen():
    plans = recommend_betmaster_plans(FEATURES_18, RACE_STRUCTURE, HORSE_ROLES_18)
    for plan in plans:
        if plan["confidence_ok"]:
            for t in plan["tickets"]:
                assert t["stake"] == 100, f"{plan['bet_type']} has stake {t['stake']}"


def test_budget_equals_ticket_count_times_100():
    plans = recommend_betmaster_plans(FEATURES_18, RACE_STRUCTURE, HORSE_ROLES_18)
    for plan in plans:
        if plan["confidence_ok"]:
            assert plan["budget"] == plan["ticket_count"] * 100


def test_no_fade_horse_in_tickets():
    plans = recommend_betmaster_plans(FEATURES_18, RACE_STRUCTURE, HORSE_ROLES_18)
    for plan in plans:
        if plan["confidence_ok"]:
            for t in plan["tickets"]:
                assert "馬6" not in t["combination"], \
                    f"fade馬が {plan['bet_type']} に含まれている: {t['combination']}"


def test_formation_legs_present_for_multi_leg_bets():
    plans = recommend_betmaster_plans(FEATURES_18, RACE_STRUCTURE, HORSE_ROLES_18)
    for plan in plans:
        if "フォーメーション" in plan["bet_type"] or "流し" in plan["bet_type"]:
            assert plan["formation_legs"] is not None, \
                f"{plan['bet_type']} に formation_legs がない"


def test_select_primary_betmaster_returns_one():
    plans = recommend_betmaster_plans(FEATURES_18, RACE_STRUCTURE, HORSE_ROLES_18)
    primary = select_primary_betmaster(plans, RACE_STRUCTURE)
    assert primary is not None
    assert primary["confidence_ok"] is True


def test_fallback_no_roles():
    """horse_roles なしでも動作する"""
    plans = recommend_betmaster_plans(FEATURES_18, RACE_STRUCTURE, horse_roles=None)
    assert len(plans) == 9
    for plan in plans:
        assert "bet_type" in plan
        assert "confidence_ok" in plan
