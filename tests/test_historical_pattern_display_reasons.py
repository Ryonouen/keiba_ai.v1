def test_display_reason_groups_prioritize_specific_supported_reasons():
    from race_ai_engine import _build_historical_pattern_display_reason_groups

    profile = {
        "token_stats": {
            "race_top3:共同通信杯": {"starts": 17},
            "race_top3:2歳新馬": {"starts": 101},
            "distance_any:1600": {"starts": 41},
            "distance_top3:1800": {"starts": 104},
            "trial_group:gi_prep": {"starts": 65},
            "race_top3:若駒S": {"starts": 6},
        }
    }
    reasons = [
        "race_top3:共同通信杯(+0.300)",
        "race_top3:2歳新馬(+0.288)",
        "distance_any:1600(+0.300)",
        "distance_top3:1800(+0.226)",
        "trial_group:gi_prep(-0.300)",
        "race_top3:若駒S(-0.300)",
    ]

    groups = _build_historical_pattern_display_reason_groups(
        reasons,
        profile,
        route_reasons=["prev_race_name=共同通信杯(+0.450)"],
    )

    assert groups["positive"][0]["token"] == "distance_top3:1800"
    assert groups["negative"][0]["token"] == "trial_group:gi_prep"
    assert "race_top3:" not in groups["positive"][0]["text"]
    assert "trial_group:" not in groups["negative"][0]["text"]
    assert "2歳新馬" not in groups["positive"][0]["text"]


def test_attach_historical_pattern_scores_keeps_raw_reasons_and_adds_display_fields():
    from race_ai_engine import attach_historical_pattern_scores

    features = [
        {
            "horse_name": "TestHorse",
            "age": 3,
            "gate": 4,
            "route_profile_reasons": ["prev_race_name=共同通信杯(+0.450)"],
            "past_races": [
                {"race_name": "共同通信杯(GIII)", "rank": 1, "distance": 1800, "date": "2026/02/15"},
                {"race_name": "2歳新馬", "rank": 1, "distance": 1600, "date": "2025/06/01"},
                {"race_name": "若駒S(L)", "rank": 1, "distance": 2000, "date": "2026/01/24"},
            ],
        }
    ]
    profile = {
        "sample_size": 20,
        "token_stats": {
            "race_top3:共同通信杯": {"starts": 17, "score": 0.30},
            "race_top3:2歳新馬": {"starts": 101, "score": 0.28807},
            "race_top3:若駒S": {"starts": 6, "score": -0.30},
            "distance_top3:1800": {"starts": 104, "score": 0.22581},
            "trial_group:gi_prep": {"starts": 65, "score": -0.30},
        },
    }

    attach_historical_pattern_scores(features, profile, race_id="test")
    feature = features[0]

    assert any("race_top3:" in reason for reason in feature["historical_pattern_reasons"])
    assert feature["historical_pattern_display_reasons"]
    assert feature["historical_pattern_reason_groups"]["positive"]
    assert feature["historical_pattern_reason_groups"]["negative"]
    assert all("race_top3:" not in reason for reason in feature["historical_pattern_display_reasons"])
