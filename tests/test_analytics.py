"""Tests for app.analytics.

FPS is measured from real timestamps, so the tests pass explicit times to
:meth:`FpsCounter.tick` to keep the results predictable.
"""

import pytest

from app.analytics import FpsCounter, SessionStats, count_classes
from app.detector import Detection


def make_detection(class_name: str, confidence: float = 0.8) -> Detection:
    """Build a Detection for use in tests."""
    return Detection(class_id=0, class_name=class_name, confidence=confidence, box=(0, 0, 1, 1))


# -- Class counting --------------------------------------------------------- #


def test_count_classes_counts_each_class() -> None:
    detections = [make_detection("person"), make_detection("person"), make_detection("chair")]

    assert count_classes(detections) == {"person": 2, "chair": 1}


def test_count_classes_of_an_empty_frame_is_empty() -> None:
    assert count_classes([]) == {}


# -- FPS -------------------------------------------------------------------- #


def test_fps_is_zero_before_any_interval_is_measured() -> None:
    assert FpsCounter().tick(0.0) == 0.0


def test_fps_is_the_reciprocal_of_the_frame_interval() -> None:
    counter = FpsCounter()
    counter.tick(0.0)

    assert counter.tick(0.5) == pytest.approx(2.0)


def test_fps_averages_the_recent_intervals() -> None:
    counter = FpsCounter()
    for moment in (0.0, 0.1, 0.2, 0.3):
        fps = counter.tick(moment)

    assert fps == pytest.approx(10.0)


def test_fps_counter_counts_the_frames_it_has_seen() -> None:
    counter = FpsCounter()
    counter.tick(0.0)
    counter.tick(0.1)

    assert counter.frames == 2


def test_fps_counter_reset_clears_everything() -> None:
    counter = FpsCounter()
    counter.tick(0.0)
    counter.tick(0.1)
    counter.reset()

    assert counter.frames == 0
    assert counter.fps == 0.0


# -- Session statistics ----------------------------------------------------- #


def test_session_accumulates_totals_across_frames() -> None:
    stats = SessionStats()
    stats.update([make_detection("person"), make_detection("chair")], fps=10.0)
    stats.update([make_detection("person")], fps=20.0)

    assert stats.frames == 2
    assert stats.total_detections == 3
    assert stats.peak_objects == 2
    assert stats.class_totals["person"] == 2


def test_session_reports_the_most_frequent_class() -> None:
    stats = SessionStats()
    stats.update([make_detection("person"), make_detection("person"), make_detection("chair")], fps=1.0)

    assert stats.most_frequent_class() == ("person", 2)


def test_session_breaks_ties_alphabetically() -> None:
    stats = SessionStats()
    stats.update([make_detection("zebra"), make_detection("apple")], fps=1.0)

    assert stats.most_frequent_class() == ("apple", 1)


def test_session_with_no_detections_stays_at_zero() -> None:
    stats = SessionStats()
    stats.update([], fps=5.0)

    assert stats.total_detections == 0
    assert stats.average_confidence == 0.0
    assert stats.most_frequent_class() is None


def test_session_average_confidence() -> None:
    stats = SessionStats()
    stats.update([make_detection("person", 0.6), make_detection("chair", 0.8)], fps=1.0)

    assert stats.average_confidence == pytest.approx(0.7)


def test_session_average_detections_per_frame() -> None:
    stats = SessionStats()
    stats.update([make_detection("person"), make_detection("chair")], fps=1.0)
    stats.update([], fps=1.0)

    assert stats.average_detections_per_frame == pytest.approx(1.0)


def test_session_average_fps_ignores_unmeasured_samples() -> None:
    stats = SessionStats()
    stats.update([], fps=0.0)
    stats.update([], fps=10.0)

    assert stats.average_fps == pytest.approx(10.0)


def test_session_reset_clears_the_totals() -> None:
    stats = SessionStats()
    stats.update([make_detection("person")], fps=5.0)
    stats.reset()

    assert stats.frames == 0
    assert stats.total_detections == 0


def test_session_summary_dict_has_the_expected_fields() -> None:
    stats = SessionStats()
    stats.update([make_detection("person")], fps=5.0)

    summary = stats.as_dict()

    assert summary["frames_processed"] == 1
    assert summary["total_detections"] == 1
    assert summary["most_frequent_class"] == "person"
    assert summary["class_totals"] == {"person": 1}
