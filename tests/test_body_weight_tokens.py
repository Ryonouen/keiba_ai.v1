def test_bucket_body_weight_under_440():
    from historical_pattern_engine import bucket_body_weight
    assert bucket_body_weight(430) == "under_440"


def test_bucket_body_weight_440_459():
    from historical_pattern_engine import bucket_body_weight
    assert bucket_body_weight(450) == "440_459"


def test_bucket_body_weight_460_479():
    from historical_pattern_engine import bucket_body_weight
    assert bucket_body_weight(464) == "460_479"


def test_bucket_body_weight_480_plus():
    from historical_pattern_engine import bucket_body_weight
    assert bucket_body_weight(490) == "480_plus"


def test_bucket_body_weight_unknown():
    from historical_pattern_engine import bucket_body_weight
    assert bucket_body_weight(None) == "unknown"
    assert bucket_body_weight("abc") == "unknown"


def test_body_weight_tokens_appear_in_feature_tokens():
    from historical_pattern_engine import build_feature_pattern_tokens
    feature = {
        "age": 3,
        "gate": 4,
        "past_races": [
            {
                "race_name": "チューリップ賞(GII)",
                "rank": 2,
                "distance": 1600,
                "date": "2026/03/08",
                "body_weight": 456,
                "body_weight_change": -4,
            }
        ],
    }
    tokens = build_feature_pattern_tokens(feature)
    assert "body_weight:440_459" in tokens
    assert "body_weight:losing" in tokens


def test_body_weight_gaining_token():
    from historical_pattern_engine import build_feature_pattern_tokens
    feature = {
        "age": 4,
        "gate": 3,
        "past_races": [
            {
                "race_name": "天皇賞春(GI)",
                "rank": 1,
                "distance": 3200,
                "date": "2026/04/27",
                "body_weight": 480,
                "body_weight_change": 6,
            }
        ],
    }
    tokens = build_feature_pattern_tokens(feature)
    assert "body_weight:480_plus" in tokens
    assert "body_weight:gaining" in tokens


def test_parse_body_weight_from_netkeiba_result_column():
    from race_ai_engine import _parse_body_weight_from_result_cols

    class Cell:
        def __init__(self, text):
            self.text = text

        def get_text(self, strip=False):
            return self.text.strip() if strip else self.text

    cols = [Cell("") for _ in range(33)]
    cols[15] = Cell("")
    cols[28] = Cell("490(-10)")

    body_weight, body_weight_change = _parse_body_weight_from_result_cols(
        cols,
        lambda col: col.get_text(strip=True),
    )

    assert body_weight == 490
    assert body_weight_change == -10


def test_no_body_weight_token_when_absent():
    from historical_pattern_engine import build_feature_pattern_tokens
    feature = {
        "age": 3,
        "gate": 4,
        "past_races": [
            {"race_name": "チューリップ賞(GII)", "rank": 2, "distance": 1600, "date": "2026/03/08"},
        ],
    }
    tokens = build_feature_pattern_tokens(feature)
    assert not any(t.startswith("body_weight:") for t in tokens), tokens


def test_classify_trial_group_gi_prep():
    from historical_pattern_engine import classify_trial_group
    assert classify_trial_group(["チューリップ賞(GII)", "アルテミスS(GIII)"]) == "gi_prep"


def test_classify_trial_group_direct():
    from historical_pattern_engine import classify_trial_group
    assert classify_trial_group(["阪神ジュベナイルF(GI)", "新馬"]) == "direct"


def test_classify_trial_group_other():
    from historical_pattern_engine import classify_trial_group
    assert classify_trial_group(["フィリーズレビュー(GII)"]) == "gi_prep"


def test_trial_group_token_in_feature_tokens():
    from historical_pattern_engine import build_feature_pattern_tokens
    feature = {
        "age": 3,
        "gate": 3,
        "past_races": [
            {"race_name": "チューリップ賞(GII)", "rank": 1, "distance": 1600, "date": "2026/03/08"},
            {"race_name": "阪神ジュベナイルF(GI)", "rank": 2, "distance": 1600, "date": "2025/12/08"},
        ],
    }
    tokens = build_feature_pattern_tokens(feature)
    assert "trial_group:gi_prep" in tokens


def test_direct_token_when_no_trial_race():
    from historical_pattern_engine import build_feature_pattern_tokens
    feature = {
        "age": 3,
        "gate": 2,
        "past_races": [
            {"race_name": "阪神ジュベナイルF(GI)", "rank": 1, "distance": 1600, "date": "2025/12/08"},
        ],
    }
    tokens = build_feature_pattern_tokens(feature)
    assert "trial_group:direct" in tokens
