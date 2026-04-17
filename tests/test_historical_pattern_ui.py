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

    assert display_groups["positive"] == ["1800mで3着以内の実績は好材料"]
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
