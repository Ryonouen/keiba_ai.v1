def test_build_historical_pattern_profile_learns_key_tokens():
    from historical_pattern_engine import build_historical_pattern_profile

    samples = [
        {
            "target_rank": 1,
            "age": 3,
            "gate": 3,
            "past_races": [
                {"race_name": "阪神ジュベナイルF(GI)", "rank": 2, "distance": 1600, "date": "2025/12/08"},
                {"race_name": "アルテミスS(GIII)", "rank": 1, "distance": 1600, "date": "2025/10/26"},
            ],
        },
        {
            "target_rank": 2,
            "age": 3,
            "gate": 2,
            "past_races": [
                {"race_name": "阪神ジュベナイルF(GI)", "rank": 3, "distance": 1600, "date": "2025/12/08"},
            ],
        },
        {
            "target_rank": 8,
            "age": 3,
            "gate": 8,
            "past_races": [
                {"race_name": "フィリーズレビュー(GII)", "rank": 9, "distance": 1400, "date": "2026/03/10"},
            ],
        },
        {
            "target_rank": 11,
            "age": 3,
            "gate": 7,
            "past_races": [
                {"race_name": "アネモネS(L)", "rank": 6, "distance": 1600, "date": "2026/03/15"},
            ],
        },
    ]

    profile = build_historical_pattern_profile(samples)
    token_stats = profile["token_stats"]

    assert token_stats["race_top3:阪神ジュベナイルF"]["score"] > 0
    assert token_stats["age:3"]["score"] > -0.2
    assert token_stats["gate:inner"]["score"] > token_stats["gate:outer"]["score"]


def test_score_historical_patterns_prefers_hanshin_jf_top3_history():
    from historical_pattern_engine import (
        build_feature_pattern_tokens,
        build_historical_pattern_profile,
        score_feature_patterns,
    )

    samples = [
        {
            "target_rank": 1,
            "age": 3,
            "gate": 2,
            "past_races": [
                {"race_name": "阪神ジュベナイルF(GI)", "rank": 1, "distance": 1600, "date": "2025/12/08"},
            ],
        },
        {
            "target_rank": 2,
            "age": 3,
            "gate": 4,
            "past_races": [
                {"race_name": "阪神ジュベナイルF(GI)", "rank": 3, "distance": 1600, "date": "2025/12/08"},
            ],
        },
        {
            "target_rank": 10,
            "age": 3,
            "gate": 8,
            "past_races": [
                {"race_name": "フィリーズレビュー(GII)", "rank": 10, "distance": 1400, "date": "2026/03/10"},
            ],
        },
        {
            "target_rank": 12,
            "age": 3,
            "gate": 7,
            "past_races": [
                {"race_name": "アネモネS(L)", "rank": 8, "distance": 1600, "date": "2026/03/15"},
            ],
        },
    ]

    profile = build_historical_pattern_profile(samples)

    good_feature = {
        "horse_name": "Good",
        "age": 3,
        "gate": 2,
        "past_races": [
            {"race_name": "阪神ジュベナイルF(GI)", "rank": 2, "distance": 1600, "date": "2025/12/08"},
            {"race_name": "アルテミスS(GIII)", "rank": 1, "distance": 1600, "date": "2025/10/26"},
        ],
    }
    bad_feature = {
        "horse_name": "Bad",
        "age": 3,
        "gate": 8,
        "past_races": [
            {"race_name": "フィリーズレビュー(GII)", "rank": 12, "distance": 1400, "date": "2026/03/10"},
        ],
    }

    good_score, good_reasons = score_feature_patterns(good_feature, profile)
    bad_score, bad_reasons = score_feature_patterns(bad_feature, profile)

    assert "race_top3:阪神ジュベナイルF" in build_feature_pattern_tokens(good_feature)
    assert good_score > bad_score
    assert any("阪神ジュベナイルF" in reason for reason in good_reasons)
    # フィリーズレビューは1サンプルのみ→starts<2でフィルタされるため reasons に出ない（正常）
    assert bad_score <= 0.0


def test_low_support_token_is_dampened():
    """A token appearing only once should be dampened to near-zero score."""
    from historical_pattern_engine import build_historical_pattern_profile

    samples = [
        {
            "target_rank": 1,
            "age": 3,
            "gate": 1,
            "past_races": [{"race_name": "2歳新馬", "rank": 1, "distance": 1600, "date": "2025/06/01"}],
        }
    ] + [
        {
            "target_rank": 5 + i,
            "age": 3,
            "gate": i + 2,
            "past_races": [],
        }
        for i in range(9)
    ]

    profile = build_historical_pattern_profile(samples)
    token_stats = profile["token_stats"]
    score = token_stats.get("race_top3:2歳新馬", {}).get("score", 0.0)
    assert abs(score) < 0.08, f"Expected dampened score, got {score}"


def test_race_top3_does_not_dominate_over_distance():
    """race_top3 token weight should be < 2x distance_top3 weight."""
    from historical_pattern_engine import _token_weight
    race_w = _token_weight("race_top3:阪神ジュベナイルF")
    dist_w = _token_weight("distance_top3:1600")
    assert race_w < dist_w * 2.0, f"race_top3 weight {race_w} too dominant over distance {dist_w}"
