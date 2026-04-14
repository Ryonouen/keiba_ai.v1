import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from trend_analyzer import (
    shrink_by_sample,
    score_prev_class,
    score_prev_rank,
    score_running_style,
    score_gate,
    score_body_weight,
    score_distance_change,
    score_sex,
    score_age,
    score_jockey,
    analyze_horse_trend,
    apply_trend_analyzer_bias,
)


def _f(**kw):
    """最小 feature dict を生成するヘルパー。"""
    base = {
        "prev_race_class_index": 0.0,
        "prev_race_name":        "",
        "prev_rank":             None,
        "past_races":            [],
        "target_distance":       None,
        "target_course":         None,
        "running_style":         None,
        "win_odds":              None,
        "gate":                  None,
    }
    base.update(kw)
    return base


# ── shrink_by_sample ─────────────────────────────────────────────────────

def test_shrink_full_sample():
    assert shrink_by_sample(0.04, 30) == 0.04

def test_shrink_medium_sample():
    assert abs(shrink_by_sample(0.04, 20) - 0.028) < 1e-9

def test_shrink_small_sample():
    assert abs(shrink_by_sample(0.04, 10) - 0.016) < 1e-9

def test_shrink_tiny_sample():
    assert abs(shrink_by_sample(0.04, 4) - 0.008) < 1e-9


# ── score_prev_class ─────────────────────────────────────────────────────

def test_prev_class_g1_gives_positive():
    edge, match, risk = score_prev_class(_f(prev_race_class_index=0.97))
    assert edge > 0
    assert match is not None
    assert risk is None

def test_prev_class_g23_gives_small_positive():
    edge_g1, _, _ = score_prev_class(_f(prev_race_class_index=0.97))
    edge_g2, _, _ = score_prev_class(_f(prev_race_class_index=0.80))
    assert 0 < edge_g2 < edge_g1

def test_prev_class_open_gives_zero():
    edge, match, risk = score_prev_class(_f(prev_race_class_index=0.65))
    assert edge == 0.0
    assert match is None
    assert risk is None

def test_prev_class_boundary_below_60_gives_negative():
    """0.60未満（条件戦ゾーン境界直下）は減点。"""
    edge, match, risk = score_prev_class(_f(prev_race_class_index=0.59))
    assert edge < 0
    assert risk is not None

def test_prev_class_conditions_gives_negative():
    edge, match, risk = score_prev_class(_f(prev_race_class_index=0.50))
    assert edge < 0
    assert risk is not None
    assert match is None

def test_prev_class_zero_gives_zero():
    edge, match, risk = score_prev_class(_f(prev_race_class_index=0.0))
    assert edge == 0.0


# ── score_prev_rank ─────────────────────────────────────────────────────

def test_prev_rank_win_positive():
    edge, match, risk = score_prev_rank(_f(prev_rank=1, prev_race_class_index=0.80))
    assert edge > 0
    assert match is not None

def test_prev_rank_top3_smaller_than_win():
    e1, _, _ = score_prev_rank(_f(prev_rank=1))
    e3, _, _ = score_prev_rank(_f(prev_rank=3))
    assert 0 < e3 < e1

def test_prev_rank_4th_neutral():
    edge, match, risk = score_prev_rank(_f(prev_rank=4))
    assert edge == 0.0
    assert match is None
    assert risk is None

def test_prev_rank_5th_neutral():
    edge, match, risk = score_prev_rank(_f(prev_rank=5))
    assert edge == 0.0
    assert match is None
    assert risk is None

def test_prev_rank_bad_non_g1_negative():
    edge, match, risk = score_prev_rank(_f(
        prev_rank=8, prev_race_class_index=0.70,
        past_races=[{"distance": 2000, "course_name": "中山"}],
        target_distance=2000,
    ))
    assert edge < 0
    assert risk is not None

def test_prev_rank_bad_g1_large_dist_no_penalty():
    edge, match, risk = score_prev_rank(_f(
        prev_rank=8, prev_race_class_index=0.97,
        past_races=[{"distance": 2500, "course_name": "中山"}],
        target_distance=2000,
    ))
    assert edge >= 0, f"G1大敗+距離差 → ペナルティ禁止。edge={edge}"

def test_prev_rank_none_returns_zero():
    edge, _, _ = score_prev_rank(_f(prev_rank=None))
    assert edge == 0.0


# ── score_running_style ─────────────────────────────────────────────────

def test_style_match_returns_match_item():
    trend = {"style": {"逃げ": 8, "先行": 2}}
    _, match, risk = score_running_style(_f(running_style="front"), trend)
    assert match is not None
    assert risk is None

def test_style_mismatch_returns_risk_item():
    trend = {"style": {"逃げ": 1, "先行": 1, "差し": 8}}
    _, match, risk = score_running_style(_f(running_style="front"), trend)
    assert risk is not None
    assert match is None

def test_style_edge_always_zero():
    trend = {"style": {"逃げ": 10}}
    edge, _, _ = score_running_style(_f(running_style="front"), trend)
    assert edge == 0.0


# ── score_gate ────────────────────────────────────────────────────────────

def test_gate_inner_match():
    # Use bucket keys as produced by trend_stats.bucket_gate
    trend = {"gate": {"内枠(1〜3)": 15}}
    edge, match, risk = score_gate(_f(gate=2), trend)
    assert edge == 0.0
    assert match is not None

def test_gate_no_trend_returns_none():
    edge, match, risk = score_gate(_f(gate=2), {})
    assert edge == 0.0 and match is None and risk is None


# ── analyze_horse_trend ──────────────────────────────────────────────────

def test_analyze_returns_all_keys():
    result = analyze_horse_trend(_f())
    for key in ("trend_score", "trend_confidence", "trend_match_items",
                "trend_risk_items", "trend_adjustment", "trend_summary"):
        assert key in result, f"missing key: {key}"

def test_analyze_g1_winner_above_1():
    f = _f(
        prev_race_class_index=0.97,
        prev_rank=1,
        past_races=[{"distance": 2000, "course_name": "中山"}],
        target_distance=2000,
    )
    result = analyze_horse_trend(f)
    assert result["trend_adjustment"] > 1.0

def test_analyze_bad_conditions_horse_below_1():
    f = _f(
        prev_race_class_index=0.45,
        prev_rank=8,
        past_races=[{"distance": 2000, "course_name": "中山"}],
        target_distance=2000,
    )
    result = analyze_horse_trend(f)
    assert result["trend_adjustment"] < 1.0

def test_analyze_adjustment_within_typical_range():
    # Note: TREND_ADJ_MAX=1.12 / TREND_ADJ_MIN=0.90 clip is not reachable
    # with Phase 1 max edges (0.04+0.030=0.070 → max 1.07), but we verify
    # the output stays within the documented range.
    f = _f(
        prev_race_class_index=0.99,
        prev_rank=1,
    )
    result = analyze_horse_trend(f)
    assert 0.90 <= result["trend_adjustment"] <= 1.12

def test_analyze_match_items_populated():
    f = _f(prev_race_class_index=0.97, prev_rank=1)
    result = analyze_horse_trend(f)
    assert len(result["trend_match_items"]) >= 1

def test_analyze_risk_items_populated():
    f = _f(prev_race_class_index=0.40, prev_rank=8,
           past_races=[{"distance": 2000, "course_name": "中山"}],
           target_distance=2000)
    result = analyze_horse_trend(f)
    assert len(result["trend_risk_items"]) >= 1


# ── apply_trend_analyzer_bias ────────────────────────────────────────────

def test_bias_normalizes_to_one():
    features = [
        {"trend_analyzer_result": {"trend_adjustment": 1.05}},
        {"trend_analyzer_result": {"trend_adjustment": 0.95}},
        {"trend_analyzer_result": {"trend_adjustment": 1.00}},
    ]
    probs = [0.4, 0.3, 0.3]
    result = apply_trend_analyzer_bias(probs, features)
    assert abs(sum(result) - 1.0) < 1e-9

def test_bias_boosts_positive_adj():
    features = [
        {"trend_analyzer_result": {"trend_adjustment": 1.10}},
        {"trend_analyzer_result": {"trend_adjustment": 0.90}},
    ]
    probs = [0.5, 0.5]
    result = apply_trend_analyzer_bias(probs, features, weight=1.0)
    assert result[0] > result[1]

def test_bias_mismatched_length_returns_original():
    probs = [0.5, 0.5]
    features = [{"trend_analyzer_result": {"trend_adjustment": 1.05}}]
    result = apply_trend_analyzer_bias(probs, features)
    assert result == probs

def test_bias_missing_result_uses_neutral():
    features = [{}, {}]
    probs = [0.6, 0.4]
    result = apply_trend_analyzer_bias(probs, features)
    assert abs(sum(result) - 1.0) < 1e-9
    assert abs(result[0] - 0.6) < 1e-6  # no adjustment = unchanged ratios


# ── Phase 2 scoring functions ─────────────────────────────────────────────

def test_body_weight_large_gain_risk():
    f = _f(past_races=[{"body_weight_change": 16}])
    _, match, risk = score_body_weight(f)
    assert risk is not None
    assert match is None


def test_body_weight_small_gain_match():
    f = _f(past_races=[{"body_weight_change": 4}])
    _, match, risk = score_body_weight(f)
    assert match is not None


def test_body_weight_no_data_returns_none():
    f = _f(past_races=[{}])
    edge, match, risk = score_body_weight(f)
    assert edge == 0.0 and match is None and risk is None


def test_distance_large_extension_risk():
    f = _f(past_races=[{"distance": 1600}], target_distance=2200)
    _, match, risk = score_distance_change(f)
    assert risk is not None


def test_distance_same_match():
    f = _f(past_races=[{"distance": 2000}], target_distance=2000)
    _, match, risk = score_distance_change(f)
    assert match is not None


def test_score_sex_female_match():
    f = _f(sex="牝")
    _, match, risk = score_sex(f)
    assert match is not None


def test_score_sex_male_neutral():
    f = _f(sex="牡")
    _, match, risk = score_sex(f)
    assert match is None and risk is None


def test_score_age_three_match():
    _, match, risk = score_age(_f(age=3))
    assert match is not None


def test_score_age_old_risk():
    _, match, risk = score_age(_f(age=8))
    assert risk is not None


def test_score_jockey_top_match():
    _, match, risk = score_jockey(_f(jockey="川田将雅"))
    assert match is not None


def test_score_jockey_unknown_neutral():
    _, match, risk = score_jockey(_f(jockey="テスト騎手"))
    assert match is None and risk is None
