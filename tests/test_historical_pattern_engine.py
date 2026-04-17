def _sample(target_rank, gate, race_name, race_rank, distance=1600, date="2025/12/08"):
    return {
        "target_rank": target_rank,
        "age": 3,
        "gate": gate,
        "past_races": [
            {
                "race_name": race_name,
                "rank": race_rank,
                "distance": distance,
                "date": date,
            }
        ],
    }


def test_race_top3_token_requires_min_starts_five():
    from historical_pattern_engine import build_historical_pattern_profile

    samples = [
        *[
            _sample(
                target_rank=rank,
                gate=gate,
                race_name="阪神ジュベナイルF(GI)",
                race_rank=2,
            )
            for rank, gate in [(1, 1), (2, 2), (3, 3), (4, 4)]
        ],
        *[
            _sample(
                target_rank=rank,
                gate=gate,
                race_name="フィリーズレビュー(GII)",
                race_rank=9,
                distance=1400,
                date="2026/03/10",
            )
            for rank, gate in [(8, 5), (9, 6), (10, 7), (11, 8), (12, 9)]
        ],
    ]

    profile = build_historical_pattern_profile(samples)
    token_stats = profile["token_stats"]

    assert "race_top3:阪神ジュベナイルF" not in token_stats

    samples.append(
        _sample(
            target_rank=1,
            gate=5,
            race_name="阪神ジュベナイルF(GI)",
            race_rank=3,
        )
    )
    profile = build_historical_pattern_profile(samples)
    token_stats = profile["token_stats"]

    assert token_stats["race_top3:阪神ジュベナイルF"]["starts"] == 5
    assert token_stats["race_top3:阪神ジュベナイルF"]["score"] > 0


def test_score_historical_patterns_prefers_supported_hanshin_jf_top3_history():
    from historical_pattern_engine import (
        build_feature_pattern_tokens,
        build_historical_pattern_profile,
        score_feature_patterns,
    )

    samples = [
        *[
            _sample(
                target_rank=target_rank,
                gate=gate,
                race_name="阪神ジュベナイルF(GI)",
                race_rank=race_rank,
            )
            for target_rank, gate, race_rank in [
                (1, 1, 1),
                (2, 2, 2),
                (3, 3, 3),
                (1, 4, 2),
                (2, 5, 3),
            ]
        ],
        *[
            _sample(
                target_rank=target_rank,
                gate=gate,
                race_name="フィリーズレビュー(GII)",
                race_rank=race_rank,
                distance=1400,
                date="2026/03/10",
            )
            for target_rank, gate, race_rank in [
                (8, 6, 8),
                (9, 7, 9),
                (10, 8, 10),
                (11, 9, 11),
                (12, 10, 12),
            ]
        ],
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
    assert bad_score <= 0.0


def test_race_any_matches_do_not_contribute_to_score():
    from historical_pattern_engine import score_feature_patterns

    profile = {
        "sample_size": 5,
        "token_stats": {
            "race_any:阪神ジュベナイルF": {
                "starts": 5,
                "wins": 5,
                "top3": 5,
                "score": 0.30,
            },
        },
    }
    feature = {
        "horse_name": "RaceAnyOnly",
        "age": 3,
        "gate": 4,
        "past_races": [
            {"race_name": "阪神ジュベナイルF(GI)", "rank": 9, "distance": 1600, "date": "2025/12/08"},
        ],
    }

    score, reasons = score_feature_patterns(feature, profile)

    assert any("race_any:阪神ジュベナイルF" in reason for reason in reasons)
    assert score == 0.0


def test_token_score_is_capped_at_phase2_limit():
    from historical_pattern_engine import TOKEN_SCORE_CAP, build_historical_pattern_profile

    samples = [
        *[
            _sample(
                target_rank=1,
                gate=gate,
                race_name="阪神ジュベナイルF(GI)",
                race_rank=1,
            )
            for gate in range(1, 6)
        ],
        *[
            _sample(
                target_rank=10 + i,
                gate=6 + i,
                race_name=f"未勝利{i}",
                race_rank=9,
                distance=1400,
                date="2026/03/10",
            )
            for i in range(30)
        ],
    ]

    profile = build_historical_pattern_profile(samples)
    token_score = profile["token_stats"]["race_top3:阪神ジュベナイルF"]["score"]

    assert TOKEN_SCORE_CAP == 0.30
    assert abs(token_score) <= 0.30


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
