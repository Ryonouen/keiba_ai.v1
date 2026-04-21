def test_historical_pattern_ui_prefers_reason_groups_over_legacy_reasons():
    from historical_pattern_ui import get_historical_pattern_ui_reason_groups

    feature = {
        "historical_pattern_reason_groups": {
            "positive": [{"text": "阪神JFで3着以内の実績は好材料"}],
            "negative": [{"text": "前走馬体重440〜459kgは近年傾向ではやや割引"}],
        },
        "historical_pattern_display_reasons": ["プラス要因: 1600mで3着以内の実績は好材料"],
        "historical_pattern_reasons": ["race_top3:阪神ジュベナイルF(+0.300)"],
    }

    groups = get_historical_pattern_ui_reason_groups(feature)

    assert groups["positive"] == ["阪神JFで3着以内の実績は好材料"]
    assert groups["negative"] == ["前走馬体重440〜459kgは近年傾向ではやや割引"]
    assert all("race_top3:" not in reason for reason in groups["positive"])


def test_historical_pattern_ui_falls_back_to_display_then_legacy_reasons():
    from historical_pattern_ui import get_historical_pattern_ui_reason_groups

    display_feature = {
        "historical_pattern_display_reasons": [
            "プラス要因: 1800mで3着以内の実績は好材料",
            "マイナス要因: 近走で直行・非トライアル組に該当する点は近年傾向ではやや割引",
        ],
        "historical_pattern_reasons": ["distance_top3:1800(+0.300)"],
    }
    legacy_feature = {
        "historical_pattern_reasons": [
            "distance_top3:1800(+0.300)",
            "trial_group:other(-0.300)",
        ],
    }

    display_groups = get_historical_pattern_ui_reason_groups(display_feature)
    legacy_groups = get_historical_pattern_ui_reason_groups(legacy_feature)

    assert display_groups["positive"] == ["1800mでの好走実績は距離面の好材料"]
    assert display_groups["negative"] == [
        "近走で直行・非トライアル組に該当する点は近年傾向ではやや割引"
    ]
    assert legacy_groups["positive"] == ["1800mでの好走実績は距離面の好材料"]
    assert legacy_groups["negative"] == [
        "近走で直行・非トライアル組に該当する点は近年傾向ではやや割引"
    ]


def test_historical_pattern_ui_polishes_raw_tokens_in_fallback_reasons():
    from historical_pattern_ui import get_historical_pattern_ui_reason_groups

    feature = {
        "historical_pattern_reasons": [
            "race_top3:共同通信杯(+0.450)",
            "distance_top3:1600(+0.450)",
            "trial_group:gi_prep(-0.300)",
            "prev_race_name=共同通信杯(+0.450)",
            "race_any:新潟2歳S(-0.450)",
            "unknown_token:raw(-0.100)",
        ],
    }

    groups = get_historical_pattern_ui_reason_groups(feature)

    assert groups["positive"] == [
        "共同通信杯で3着以内の実績は好材料",
        "1600mでの好走実績は距離面の好材料",
        "前走共同通信杯組はローテ傾向で好材料",
    ]
    assert groups["negative"] == [
        "近走でトライアル組に該当する点は近年傾向ではやや割引",
    ]
    assert all(":" not in reason and "prev_" not in reason for reason in groups["positive"])
    assert all(":" not in reason and "race_any:" not in reason for reason in groups["negative"])


def test_route_profile_display_reasons_are_human_readable():
    from historical_pattern_ui import get_route_profile_display_reasons

    feature = {
        "route_profile_reasons": [
            "prev_race_name=共同通信杯(+0.450)",
            "prev_distance_bucket=1600(-0.120)",
        ],
    }

    reasons = get_route_profile_display_reasons(feature)

    assert reasons == [
        "前走共同通信杯組はローテ傾向で好材料",
        "前走1600m組はローテ傾向ではやや割引",
    ]
    assert all("prev_" not in reason for reason in reasons)


def test_route_profile_display_reasons_show_exact_distance_buckets():
    from historical_pattern_ui import get_route_profile_display_reasons

    feature = {
        "route_profile_reasons": [
            "prev_distance_bucket=1400(+0.120)",
            "prev_distance_bucket=1600(-0.120)",
            "prev_distance_bucket=1800(+0.120)",
            "prev_distance_bucket=2000(-0.141)",
            "prev_distance_bucket=3200(+0.180)",
        ],
    }

    reasons = get_route_profile_display_reasons(feature, limit=5)

    assert reasons == [
        "前走1400m組はローテ傾向で好材料",
        "前走1600m組はローテ傾向ではやや割引",
        "前走1800m組はローテ傾向で好材料",
        "前走2000m組はローテ傾向ではやや割引",
        "前走3200m組はローテ傾向で好材料",
    ]


def test_route_profile_display_reasons_show_neutral_for_tiny_distance_scores():
    from historical_pattern_ui import get_route_profile_display_reasons

    feature = {
        "route_profile_reasons": [
            "prev_distance_bucket=2000(-0.007)",
            "prev_distance_bucket=2200(+0.012)",
            "prev_race_name=報知弥生ディープ記念(-0.057)",
            "prev_month=3(-0.450)",
        ],
    }

    reasons = get_route_profile_display_reasons(feature, limit=4)

    assert reasons == [
        "前走2000m組はローテ傾向ではほぼ中立",
        "前走2200m組はローテ傾向ではほぼ中立",
        "前走報知弥生ディープ記念組はローテ傾向ではやや割引",
        "3月からの臨戦はローテ傾向ではやや割引",
    ]


def test_historical_pattern_ui_softens_two_year_old_maiden_labels():
    from historical_pattern_ui import get_historical_pattern_ui_reason_groups

    feature = {
        "historical_pattern_reason_groups": {
            "positive": [
                {"text": "2歳新馬で3着以内の実績は好材料"},
                {"text": "2歳未勝利で3着以内の実績は好材料"},
            ],
            "negative": [
                {"text": "2歳未勝利で3着以内の履歴は近年傾向ではやや割引"},
            ],
        },
    }

    groups = get_historical_pattern_ui_reason_groups(feature)

    assert groups["positive"] == [
        "2歳新馬で3着以内の履歴は補助的な好材料（参考度はやや控えめ）",
        "2歳未勝利で3着以内の履歴は補助的な好材料（参考度はやや控えめ）",
    ]
    assert groups["negative"] == [
        "2歳未勝利で3着以内の履歴は近年傾向ではやや割引（参考度はやや控えめ）"
    ]


def test_historical_pattern_ui_expands_listed_and_open_grade_labels():
    from historical_pattern_ui import get_historical_pattern_ui_reason_groups

    feature = {
        "historical_pattern_display_reasons": [
            "プラス要因: Lで3着以内の実績は好材料",
            "マイナス要因: OPで3着以内の履歴は近年傾向ではやや割引",
        ],
    }

    groups = get_historical_pattern_ui_reason_groups(feature)

    assert groups["positive"] == ["リステッドで3着以内の実績は好材料"]
    assert groups["negative"] == ["オープン級で3着以内の履歴は近年傾向ではやや割引"]


def test_historical_pattern_ui_polishes_distance_and_grade_labels():
    from historical_pattern_ui import get_historical_pattern_ui_reason_groups

    feature = {
        "historical_pattern_display_reasons": [
            "プラス要因: 1800mで3着以内の実績は好材料",
            "プラス要因: G2で3着以内の実績は好材料",
            "マイナス要因: 1400m以下で3着以内の履歴は近年傾向ではやや割引",
            "マイナス要因: G3で3着以内の履歴は近年傾向ではやや割引",
        ],
    }

    groups = get_historical_pattern_ui_reason_groups(feature)

    assert groups["positive"] == [
        "1800mでの好走実績は距離面の好材料",
        "G2級での好走実績は近年傾向で好材料",
    ]
    assert groups["negative"] == [
        "1400m以下での好走履歴は近年傾向ではやや割引",
        "G3級での好走履歴は近年傾向ではやや割引",
    ]


def test_audit_historical_pattern_ui_reasons_detects_display_noise():
    from historical_pattern_ui import audit_historical_pattern_ui_reasons

    features = [
        {
            "horse_name": "自然な馬",
            "route_profile_reasons": ["prev_race_name=共同通信杯(+0.450)"],
            "historical_pattern_display_reasons": [
                "プラス要因: 1800mで3着以内の実績は好材料",
                "マイナス要因: 2歳未勝利で3着以内の履歴は近年傾向ではやや割引",
            ],
        },
        {
            "horse_name": "ノイズ馬",
            "historical_pattern_reason_groups": {
                "positive": [{"text": "race_top3:共同通信杯(+0.300)"}],
                "negative": [{"text": "2歳新馬で3着以内の実績は好材料"}],
            },
        },
    ]

    audit = audit_historical_pattern_ui_reasons(features)

    assert audit["checked_horses"] == 2
    assert audit["issue_counts"]["raw_token"] == 0
    assert audit["issue_counts"]["strong_young_reason"] == 0
    assert audit["issue_counts"]["mechanical_phrase"] == 0
    assert audit["examples"]["raw_token"] == []


def test_historical_pattern_ui_polishes_age_labels():
    from historical_pattern_ui import get_historical_pattern_ui_reason_groups

    feature = {
        "historical_pattern_display_reasons": [
            "プラス要因: age:5 は好材料",
            "マイナス要因: age:4 は近年傾向ではやや割引",
            "マイナス要因: age:6以上 は近年傾向ではやや割引",
        ],
    }

    groups = get_historical_pattern_ui_reason_groups(feature)

    assert groups["positive"] == ["5歳は好材料"]
    assert groups["negative"] == [
        "4歳は近年傾向ではやや割引",
        "6歳以上は近年傾向ではやや割引",
    ]


def test_historical_pattern_ui_polishes_other_race_labels():
    from historical_pattern_ui import get_historical_pattern_ui_reason_groups

    feature = {
        "historical_pattern_display_reasons": [
            "プラス要因: OTHERで3着以内の実績は好材料",
            "プラス要因: OTHER出走経験は好材料",
            "マイナス要因: OTHERで3着以内の履歴は近年傾向ではやや割引",
            "マイナス要因: OTHER出走経験は近年傾向では補助的に割引",
        ],
    }

    groups = get_historical_pattern_ui_reason_groups(feature)

    assert groups["positive"] == [
        "その他のレースでの好走実績は好材料",
        "その他のレースでの出走経験は好材料",
    ]
    assert groups["negative"] == [
        "その他のレースでの好走履歴は近年傾向ではやや割引",
        "その他のレースでの出走経験は近年傾向では補助的に割引",
    ]


def test_historical_pattern_ui_softens_low_support_body_weight_reasons():
    from historical_pattern_ui import get_historical_pattern_ui_reason_groups

    feature = {
        "historical_pattern_reason_groups": {
            "positive": [],
            "negative": [
                {
                    "token": "body_weight:460_479",
                    "text": "前走馬体重460〜479kgは近年傾向ではやや割引",
                    "starts": 6,
                },
            ],
        },
    }

    groups = get_historical_pattern_ui_reason_groups(feature)

    assert groups["negative"] == [
        "前走馬体重460〜479kgは近年傾向ではやや割引（参考度はやや控えめ）"
    ]


def test_historical_pattern_ui_limits_body_weight_reasons_to_one_per_horse():
    from historical_pattern_ui import get_historical_pattern_ui_reason_groups

    feature = {
        "historical_pattern_reason_groups": {
            "positive": [],
            "negative": [
                {
                    "token": "body_weight:460_479",
                    "text": "前走馬体重460〜479kgは近年傾向ではやや割引",
                    "starts": 47,
                },
                {
                    "token": "body_weight:440_459",
                    "text": "前走馬体重440〜459kgは近年傾向ではやや割引",
                    "starts": 21,
                },
                {
                    "token": "grade_top3:G2",
                    "text": "G2で3着以内の履歴は近年傾向ではやや割引",
                    "starts": 30,
                },
            ],
        },
    }

    groups = get_historical_pattern_ui_reason_groups(feature)

    assert groups["negative"] == [
        "前走馬体重460〜479kgは近年傾向ではやや割引",
        "G2級での好走履歴は近年傾向ではやや割引",
    ]
