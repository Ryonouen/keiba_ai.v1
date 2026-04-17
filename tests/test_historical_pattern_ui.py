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
    assert legacy_groups["positive"] == ["distance_top3:1800(+0.300)"]
    assert legacy_groups["negative"] == ["trial_group:other(-0.300)"]


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
    assert audit["issue_counts"]["raw_token"] == 1
    assert audit["issue_counts"]["strong_young_reason"] == 0
    assert audit["issue_counts"]["mechanical_phrase"] == 0
    assert audit["examples"]["raw_token"][0]["horse_name"] == "ノイズ馬"


def test_historical_pattern_ui_polishes_age_labels():
    from historical_pattern_ui import get_historical_pattern_ui_reason_groups

    feature = {
        "historical_pattern_display_reasons": [
            "マイナス要因: age:6以上 は近年傾向ではやや割引",
        ],
    }

    groups = get_historical_pattern_ui_reason_groups(feature)

    assert groups["negative"] == ["6歳以上は近年傾向ではやや割引"]


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
