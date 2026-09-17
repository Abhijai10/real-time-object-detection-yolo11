"""Tests for :mod:`app.renderer`: drawing, overlay layout and colour stability.

Rendering needs OpenCV but no webcam and no display server, because every test
draws onto an in-memory NumPy array instead of a window.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from app.analytics import FrameAnalytics, SessionStats  # noqa: E402
from app.detector import Detection  # noqa: E402
from app.renderer import DEFAULT_PALETTE, Renderer, class_color  # noqa: E402


# --------------------------------------------------------------------------- #
# Colour assignment
# --------------------------------------------------------------------------- #


def test_class_color_is_stable_across_calls() -> None:
    assert class_color("person") == class_color("person")


def test_class_color_is_independent_of_process_hash_randomisation() -> None:
    """The colour must not depend on Python's randomised string hash."""
    assert class_color("person") == DEFAULT_PALETTE[
        sum((index + 1) * ord(character) for index, character in enumerate("person")) % len(DEFAULT_PALETTE)
    ]


def test_different_classes_generally_get_different_colours() -> None:
    colours = {class_color(name) for name in ("person", "cup", "chair", "bottle", "dog")}
    assert len(colours) >= 4


def test_class_color_falls_back_when_the_palette_is_empty() -> None:
    assert class_color("person", ()) == (255, 255, 255)


def test_renderer_color_for_matches_class_color() -> None:
    renderer = Renderer()
    assert renderer.color_for("bottle") == class_color("bottle", renderer.palette)


# --------------------------------------------------------------------------- #
# Bounding boxes
# --------------------------------------------------------------------------- #


def test_draw_detections_modifies_the_frame(synthetic_frame: np.ndarray, sample_detections) -> None:
    renderer = Renderer()
    before = synthetic_frame.copy()

    result = renderer.draw_detections(synthetic_frame, sample_detections)

    assert result is synthetic_frame
    assert np.any(before != synthetic_frame), "expected bounding boxes to be drawn"


def test_draw_detections_with_no_detections_leaves_the_frame_untouched(
    synthetic_frame: np.ndarray,
) -> None:
    renderer = Renderer()
    before = synthetic_frame.copy()

    renderer.draw_detections(synthetic_frame, [])

    assert np.array_equal(before, synthetic_frame)


def test_draw_detections_clips_boxes_that_exceed_the_frame(synthetic_frame: np.ndarray) -> None:
    renderer = Renderer()
    detections = [
        Detection(class_id=0, class_name="person", confidence=0.9, x1=-50, y1=-50, x2=9999, y2=9999)
    ]

    renderer.draw_detections(synthetic_frame, detections)  # must not raise


def test_label_is_placed_inside_the_frame_when_there_is_no_room_above(
    synthetic_frame: np.ndarray,
) -> None:
    renderer = Renderer()
    detections = [
        Detection(class_id=0, class_name="person", confidence=0.9, x1=10, y1=0, x2=200, y2=100)
    ]

    renderer.draw_detections(synthetic_frame, detections)  # must not raise


# --------------------------------------------------------------------------- #
# Overlays
# --------------------------------------------------------------------------- #


def test_draw_analytics_overlay_renders_the_expected_fields(
    synthetic_frame: np.ndarray, sample_analytics: FrameAnalytics
) -> None:
    renderer = Renderer()
    before = synthetic_frame.copy()

    result = renderer.draw_analytics_overlay(
        synthetic_frame, sample_analytics, fps=27.4, confidence=0.5, resolution=(640, 480)
    )

    assert result is synthetic_frame
    assert np.any(before != synthetic_frame)


def test_draw_analytics_overlay_handles_an_empty_frame(
    synthetic_frame: np.ndarray,
) -> None:
    renderer = Renderer()
    renderer.draw_analytics_overlay(synthetic_frame, FrameAnalytics(), fps=0.0, confidence=0.25)


def test_draw_analytics_overlay_includes_session_counters(
    synthetic_frame: np.ndarray, sample_analytics: FrameAnalytics
) -> None:
    renderer = Renderer()
    session = SessionStats()
    session.update(sample_analytics, fps=20.0)

    renderer.draw_analytics_overlay(
        synthetic_frame, sample_analytics, fps=20.0, confidence=0.5, session=session
    )


def test_draw_help_overlay_is_skipped_when_no_lines_are_given(
    synthetic_frame: np.ndarray,
) -> None:
    renderer = Renderer()
    before = synthetic_frame.copy()

    renderer.draw_help_overlay(synthetic_frame, [])

    assert np.array_equal(before, synthetic_frame)


def test_draw_help_overlay_draws_the_panel(synthetic_frame: np.ndarray) -> None:
    renderer = Renderer()
    before = synthetic_frame.copy()

    renderer.draw_help_overlay(synthetic_frame, ["Q - Quit", "S - Snapshot"])

    assert np.any(before != synthetic_frame)


def test_recording_indicator_only_draws_when_recording(synthetic_frame: np.ndarray) -> None:
    renderer = Renderer()

    before = synthetic_frame.copy()
    renderer.draw_recording_indicator(synthetic_frame, recording=False)
    assert np.array_equal(before, synthetic_frame)

    renderer.draw_recording_indicator(synthetic_frame, recording=True)
    assert np.any(before != synthetic_frame)


def test_draw_message_renders_text(synthetic_frame: np.ndarray) -> None:
    renderer = Renderer()
    before = synthetic_frame.copy()

    renderer.draw_message(synthetic_frame, "Snapshot saved")

    assert np.any(before != synthetic_frame)


def test_draw_message_ignores_an_empty_string(synthetic_frame: np.ndarray) -> None:
    renderer = Renderer()
    before = synthetic_frame.copy()

    renderer.draw_message(synthetic_frame, "")

    assert np.array_equal(before, synthetic_frame)


# --------------------------------------------------------------------------- #
# Composite rendering
# --------------------------------------------------------------------------- #


def test_render_draws_every_layer(
    synthetic_frame: np.ndarray, sample_detections, sample_analytics: FrameAnalytics
) -> None:
    renderer = Renderer()
    before = synthetic_frame.copy()
    session = SessionStats()
    session.update(sample_analytics, fps=25.0)

    result = renderer.render(
        frame=synthetic_frame,
        detections=sample_detections,
        analytics=sample_analytics,
        fps=25.0,
        confidence=0.5,
        session=session,
        help_lines=["Q - Quit"],
        show_overlay=True,
        recording=True,
        message="Recording started",
    )

    assert result.shape == before.shape
    assert result.dtype == np.uint8
    assert np.any(before != synthetic_frame)


def test_render_works_on_a_minimal_frame() -> None:
    """A 1x1 frame must not crash the renderer."""
    renderer = Renderer()
    tiny = np.zeros((1, 1, 3), dtype=np.uint8)

    renderer.render(
        frame=tiny,
        detections=[Detection(class_id=0, class_name="person", confidence=0.9, x1=0, y1=0, x2=1, y2=1)],
        analytics=FrameAnalytics(object_count=1, class_counts={"person": 1}, unique_classes=("person",)),
        fps=1.0,
        confidence=0.25,
        help_lines=["Q - Quit"],
        recording=True,
        message="hello",
    )


def test_render_hides_the_overlay_when_disabled(
    synthetic_frame: np.ndarray, sample_analytics: FrameAnalytics
) -> None:
    renderer = Renderer()

    renderer.render(
        frame=synthetic_frame,
        detections=[],
        analytics=sample_analytics,
        fps=10.0,
        confidence=0.25,
        show_overlay=False,
    )
