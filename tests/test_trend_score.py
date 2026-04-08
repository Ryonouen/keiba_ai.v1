import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from value_ai import trend_score_adjustment


def _feature(**kw):
    """trend_score_adjustment に渡す最小 feature dict を生成するヘルパー。"""
    base = {
        "running_style":         None,
        "win_odds":              None,
        "gate":                  None,
        "age":                   None,
        "prev_race_class_index": 0.0,
        "prev_race_name":        "",
        "prev_rank":             None,
        "past_races":            [],
        "target_distance":       None,
        "target_course":         None,
    }
    base.update(kw)
    return base


# ── 前走着順補正 ─────────────────────────────────────────────────

def test_prev_rank_win_gives_positive_delta():
    """前走1着 → プラス補正。"""
    f = _feature(prev_rank=1, prev_race_class_index=0.80,
                 past_races=[{"distance": 2000, "course_name": "中山"}],
                 target_distance=2000, target_course="阪神")
    delta = trend_score_adjustment(f, {})
    assert delta > 0, f"expected delta > 0, got {delta}"


def test_prev_rank_top3_gives_smaller_positive_delta():
    """前走3着 → 1着より小さいプラス補正。"""
    f1 = _feature(prev_rank=1, prev_race_class_index=0.80,
                  past_races=[{"distance": 2000, "course_name": "中山"}],
                  target_distance=2000, target_course="阪神")
    f3 = _feature(prev_rank=3, prev_race_class_index=0.80,
                  past_races=[{"distance": 2000, "course_name": "中山"}],
                  target_distance=2000, target_course="阪神")
    assert trend_score_adjustment(f3, {}) < trend_score_adjustment(f1, {})


def test_prev_rank_g1_long_distance_no_penalty():
    """G1 かつ 400m 以上の距離差で大敗 → ペナルティなし（小加点）。
    大阪杯(2000m) 前走 有馬記念(2500m) で 8 着 → 条件差なので減点しない。
    """
    f = _feature(prev_rank=8, prev_race_class_index=0.95,
                 past_races=[{"distance": 2500, "course_name": "中山"}],
                 target_distance=2000, target_course="阪神")
    delta = trend_score_adjustment(f, {})
    assert delta >= 0, f"G1長距離大敗は減点してはいけない。delta={delta}"


def test_prev_rank_non_g1_bad_gives_penalty():
    """非G1重賞以下で 6 着以下 → マイナス補正。
    前走コースは異会場にして着順補正を単独で検証する。
    """
    f = _feature(prev_rank=8, prev_race_class_index=0.65,
                 past_races=[{"distance": 2000, "course_name": "中山"}],  # 異会場
                 target_distance=2000, target_course="阪神")
    delta = trend_score_adjustment(f, {})
    assert delta < 0, f"非G1大敗は減点すべき。delta={delta}"


def test_prev_rank_none_no_crash():
    """前走着順データなし → クラッシュせず float を返す。"""
    f = _feature(prev_rank=None, past_races=[], target_distance=2000, target_course="阪神")
    delta = trend_score_adjustment(f, {})
    assert isinstance(delta, float)


# ── 前走コース補正 ────────────────────────────────────────────────

def test_prev_course_same_venue_gives_bonus():
    """前走と同一競馬場 → 差し引きでプラス（異会場より高い）。"""
    f_same = _feature(past_races=[{"distance": 2000, "course_name": "阪神"}],
                      target_distance=2000, target_course="阪神")
    f_diff = _feature(past_races=[{"distance": 2000, "course_name": "中山"}],
                      target_distance=2000, target_course="阪神")
    assert trend_score_adjustment(f_same, {}) > trend_score_adjustment(f_diff, {})


def test_prev_course_no_past_races_no_crash():
    """past_races が空でもクラッシュしない。"""
    f = _feature(past_races=[], target_course="阪神")
    delta = trend_score_adjustment(f, {})
    assert isinstance(delta, float)
