"""
trend_signal.py
後方互換シム — trend_stats / signal_judge へ移譲

このファイルは既存コードとの互換性維持のみを目的とします。
新規コードは trend_stats.py / signal_judge.py を直接インポートしてください。
"""
from trend_stats import (  # noqa: F401
    bucket_age,
    bucket_gate,
    bucket_style as bucket_style_to_jp,
    bucket_popularity_odds,
    build_condition_stats as build_attribute_stats,
)
from signal_judge import (  # noqa: F401
    judge_signal as calc_signal_strength,
    calc_correction as calc_signal_delta,
    build_horse_signal_details,
    aggregate_signal_result as evaluate_all_signals,
    SIGNAL_JP,
    COND_JP as ATTR_LABELS,
)
