"""Tests for :mod:`app.analytics`: FPS measurement, frame analytics and session stats."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.analytics import (
    FpsCounter,
    FrameAnalytics,
    SessionStats,
    compute_frame_analytics,
    merge_analytics,
)
from app.detector import Detection


# --------------------------------------------------------------------------- #
# FPS counter
# --------------------------------------------------------------------------- #


def test_fps_counter_reports_zero_before_any_interval() -> None:
    counter = FpsCounter()
    assert counter.fps == 0.0
    assert counter.tick(100.0) == 0.0
    assert counter.frame_count == 1


def test_fps_counter_measures_a_known_frame_rate() -> None:
    counter = FpsCounter()
    # Ten frames spaced 0.1 s apart -> 10 FPS.
    for index in range(10):
        counter.tick(index * 0.1)

    assert counter.fps == pytest.approx(10.0, rel=1e-6)
    assert counter.frame_count == 10


def test_fps_counter_uses_a_rolling_window() -> None:
    counter = FpsCounter(window=3)
    for index in range(6):
        counter.tick(index * 0.1)
    assert len(counter._intervals) == 3  # noqa: SLF001 - window size is the behaviour under test
    assert counter.fps == pytest.approx(10.0, rel=1e-6)


def test_fps_counter_ignores_non_positive_intervals() -> None:
    counter = FpsCounter()
    counter.tick(1.0)
    counter.tick(1.0)  # zero delta must not cause a division by zero
    assert counter.fps == 0.0


def test_fps_counter_reset_clears_everything() -> None:
    counter = FpsCounter()
    counter.tick(0.0)
    counter.tick(0.1)
    counter.reset()
    assert counter.fps == 0.0
    assert counter.frame_count == 0


# --------------------------------------------------------------------------- #
# Frame analytics
# --------------------------------------------------------------------------- #


def test_compute_frame_analytics_for_an_empty_frame() -> None:
    analytics = compute_frame_analytics([])

    assert analytics.object_count == 0
    assert analytics.unique_classes == ()
    assert analytics.class_counts == {}
    assert analytics.average_confidence == 0.0
    assert analytics.max_confidence == 0.0
    assert analytics.is_empty is True


def test_compute_frame_analytics_counts_classes_and_confidences(
    sample_detections: list[Detection],
) -> None:
    analytics = compute_frame_analytics(sample_detections)

    assert analytics.object_count == 3
    assert analytics.class_counts == {"person": 2, "cup": 1}
    assert analytics.unique_classes == ("cup", "person")
    assert analytics.unique_class_count == 2
    assert analytics.average_confidence == pytest.approx((0.91 + 0.74 + 0.62) / 3, rel=1e-6)
    assert analytics.max_confidence == pytest.approx(0.91, rel=1e-6)


def test_compute_frame_analytics_accepts_any_object_with_the_right_attributes() -> None:
    class Duck:
        def __init__(self, class_name: str, confidence: float) -> None:
            self.class_name = class_name
            self.confidence = confidence

    analytics = compute_frame_analytics([Duck("bottle", 0.5), Duck("bottle", 0.7)])

    assert analytics.class_counts == {"bottle": 2}
    assert analytics.average_confidence == pytest.approx(0.6, rel=1e-6)


def test_frame_analytics_as_dict_is_json_friendly(sample_analytics: FrameAnalytics) -> None:
    payload = sample_analytics.as_dict()

    assert payload["object_count"] == 3
    assert payload["unique_class_count"] == 2
    assert isinstance(payload["unique_classes"], list)
    assert payload["class_counts"] == {"person": 2, "cup": 1}


# --------------------------------------------------------------------------- #
# Session statistics
# --------------------------------------------------------------------------- #


def test_session_stats_start_empty() -> None:
    stats = SessionStats()

    assert stats.frames_processed == 0
    assert stats.total_detections == 0
    assert stats.peak_simultaneous_detections == 0
    assert stats.most_frequent_class() is None
    assert stats.average_fps == 0.0
    assert stats.average_detections_per_frame == 0.0


def test_session_stats_accumulate_across_frames(sample_analytics: FrameAnalytics) -> None:
    stats = SessionStats()
    stats.update(sample_analytics, fps=20.0)
    stats.update(FrameAnalytics(), fps=24.0)

    assert stats.frames_processed == 2
    assert stats.frames_with_detections == 1
    assert stats.total_detections == 3
    assert stats.class_totals["person"] == 2
    assert stats.class_totals["cup"] == 1
    assert stats.average_detections_per_frame == pytest.approx(1.5, rel=1e-6)
    assert stats.detection_rate == pytest.approx(0.5, rel=1e-6)
    assert stats.average_fps == pytest.approx(22.0, rel=1e-6)


def test_session_stats_track_peak_simultaneous_detections() -> None:
    stats = SessionStats()
    stats.update(FrameAnalytics(object_count=2, class_counts={"person": 2}))
    stats.update(FrameAnalytics(object_count=5, class_counts={"person": 5}))
    stats.update(FrameAnalytics(object_count=1, class_counts={"person": 1}))

    assert stats.peak_simultaneous_detections == 5


def test_session_stats_most_frequent_class() -> None:
    stats = SessionStats()
    stats.update(FrameAnalytics(object_count=3, class_counts={"person": 2, "cup": 1}))
    stats.update(FrameAnalytics(object_count=4, class_counts={"person": 1, "chair": 3}))

    assert stats.most_frequent_class() == ("chair", 3)


def test_session_stats_most_frequent_class_breaks_ties_alphabetically() -> None:
    stats = SessionStats()
    stats.update(FrameAnalytics(object_count=2, class_counts={"zebra": 1, "apple": 1}))

    assert stats.most_frequent_class() == ("apple", 1)


def test_session_stats_average_confidence_is_weighted_by_object_count() -> None:
    stats = SessionStats()
    # Two objects at 1.0 and one object at 0.5 -> (1.0*2 + 0.5*1) / 3 = 0.8333...
    stats.update(FrameAnalytics(object_count=2, class_counts={"person": 2}, average_confidence=1.0))
    stats.update(FrameAnalytics(object_count=1, class_counts={"cup": 1}, average_confidence=0.5))

    assert stats.average_confidence == pytest.approx(2.5 / 3, rel=1e-6)


def test_session_stats_ignore_unmeasured_fps() -> None:
    stats = SessionStats()
    stats.update(FrameAnalytics(), fps=0.0)
    stats.update(FrameAnalytics(), fps=None)  # type: ignore[arg-type]

    assert stats.average_fps == 0.0
    assert stats.peak_fps == 0.0


def test_session_stats_top_classes_are_ordered() -> None:
    stats = SessionStats()
    stats.update(FrameAnalytics(object_count=6, class_counts={"person": 3, "cup": 2, "chair": 1}))

    assert stats.top_classes(2) == [("person", 3), ("cup", 2)]
    assert stats.top_classes() == [("person", 3), ("cup", 2), ("chair", 1)]


def test_session_stats_reset_clears_counters_but_keeps_start_time() -> None:
    start = datetime(2026, 9, 17, 22, 0, 0)
    stats = SessionStats(started_at=start)
    stats.update(FrameAnalytics(object_count=3, class_counts={"person": 3}), fps=10.0)

    stats.reset()

    assert stats.frames_processed == 0
    assert stats.total_detections == 0
    assert stats.class_totals == {}
    assert stats.average_fps == 0.0
    assert stats.started_at == start


def test_session_stats_reset_can_restart_the_clock() -> None:
    stats = SessionStats(started_at=datetime(2020, 1, 1))
    stats.reset(keep_start_time=False)
    assert stats.started_at.year >= 2020


def test_session_stats_duration_uses_the_start_timestamp() -> None:
    stats = SessionStats(started_at=datetime.now() - timedelta(seconds=5))
    assert stats.duration_seconds == pytest.approx(5.0, abs=1.0)


def test_session_stats_as_dict_contains_the_documented_keys() -> None:
    stats = SessionStats()
    stats.update(FrameAnalytics(object_count=2, class_counts={"person": 2}), fps=15.0)

    payload = stats.as_dict()

    for key in (
        "started_at",
        "duration_seconds",
        "frames_processed",
        "total_detections",
        "peak_simultaneous_detections",
        "average_fps",
        "peak_fps",
        "most_frequent_class",
        "class_totals",
    ):
        assert key in payload

    assert payload["frames_processed"] == 1
    assert payload["most_frequent_class"] == "person"
    assert payload["class_totals"] == {"person": 2}


def test_session_stats_summary_lines_are_human_readable() -> None:
    stats = SessionStats()
    stats.update(FrameAnalytics(object_count=1, class_counts={"person": 1}), fps=12.5)

    lines = stats.summary_lines()

    assert any("Frames processed" in line for line in lines)
    assert any("Most frequent class" in line and "person" in line for line in lines)
    assert any("Top classes" in line for line in lines)


def test_merge_analytics_helper() -> None:
    frames = [
        FrameAnalytics(object_count=1, class_counts={"person": 1}),
        FrameAnalytics(object_count=2, class_counts={"cup": 2}),
    ]

    stats = merge_analytics(frames)

    assert stats.frames_processed == 2
    assert stats.total_detections == 3
