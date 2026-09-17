"""Tests for :mod:`app.utils`: validation, path helpers and output handling."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pytest

from app.utils import (
    MAX_CAMERA_INDEX,
    RecordingError,
    SnapshotError,
    VideoRecorder,
    clamp,
    configure_logging,
    ensure_directory,
    format_duration,
    is_supported_video_extension,
    recording_path,
    safe_divide,
    save_snapshot,
    session_summary_path,
    snapshot_path,
    timestamp_slug,
    validate_camera_index,
    validate_confidence,
    validate_dimension,
    validate_resolution,
    write_json,
)

FIXED_MOMENT = datetime(2026, 9, 17, 22, 45, 1)


# --------------------------------------------------------------------------- #
# Timestamps and output paths
# --------------------------------------------------------------------------- #


def test_timestamp_slug_matches_documented_format() -> None:
    assert timestamp_slug(FIXED_MOMENT) == "2026_09_17_224501"


def test_snapshot_path_uses_timestamp_and_extension(output_dir: Path) -> None:
    path = snapshot_path(output_dir, FIXED_MOMENT)
    assert path.name == "snapshot_2026_09_17_224501.jpg"
    assert path.parent == output_dir


def test_snapshot_path_accepts_extension_without_dot(output_dir: Path) -> None:
    assert snapshot_path(output_dir, FIXED_MOMENT, "png").name.endswith(".png")


def test_recording_path_uses_timestamp(output_dir: Path) -> None:
    assert recording_path(output_dir, FIXED_MOMENT).name == "recording_2026_09_17_224501.mp4"


def test_session_summary_path(output_dir: Path) -> None:
    assert session_summary_path(output_dir).name == "session_summary.json"


# --------------------------------------------------------------------------- #
# Directory handling
# --------------------------------------------------------------------------- #


def test_ensure_directory_creates_nested_path(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c"
    assert ensure_directory(target) == target
    assert target.is_dir()


def test_ensure_directory_rejects_a_file(tmp_path: Path) -> None:
    existing_file = tmp_path / "not_a_dir"
    existing_file.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        ensure_directory(existing_file)


# --------------------------------------------------------------------------- #
# Confidence validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("value", [0.01, 0.25, 0.5, 1.0, "0.75", 1])
def test_validate_confidence_accepts_valid_values(value: object) -> None:
    assert 0.01 <= validate_confidence(value) <= 1.0


@pytest.mark.parametrize("value", [0, -0.1, 1.01, 2, "high", None, "", float("nan")])
def test_validate_confidence_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError):
        validate_confidence(value)


def test_validate_confidence_error_message_is_helpful() -> None:
    with pytest.raises(ValueError, match="between"):
        validate_confidence(3.5)


# --------------------------------------------------------------------------- #
# Camera index validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("value", [0, 1, 2, "0", "3", 32])
def test_validate_camera_index_accepts_valid_values(value: object) -> None:
    assert validate_camera_index(value) >= 0


@pytest.mark.parametrize("value", [-1, -10, MAX_CAMERA_INDEX + 1, 999999, "abc", None])
def test_validate_camera_index_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError):
        validate_camera_index(value)


def test_validate_camera_index_error_mentions_the_limit() -> None:
    with pytest.raises(ValueError, match="unreasonably large"):
        validate_camera_index(5000)


# --------------------------------------------------------------------------- #
# Resolution validation
# --------------------------------------------------------------------------- #


def test_validate_dimension_accepts_positive_integers() -> None:
    assert validate_dimension("1280", "width") == 1280


@pytest.mark.parametrize("value", [0, -5, "tall", None])
def test_validate_dimension_rejects_bad_values(value: object) -> None:
    with pytest.raises(ValueError):
        validate_dimension(value)


def test_validate_resolution_allows_both_missing() -> None:
    assert validate_resolution(None, None) == (None, None)


def test_validate_resolution_requires_both_values() -> None:
    with pytest.raises(ValueError, match="together"):
        validate_resolution(1280, None)
    with pytest.raises(ValueError, match="together"):
        validate_resolution(None, 720)


def test_validate_resolution_returns_pair() -> None:
    assert validate_resolution(1280, 720) == (1280, 720)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("path", "expected"),
    [("clip.mp4", True), ("clip.AVI", True), ("clip.mkv", True), ("photo.jpg", False), ("clip", False)],
)
def test_is_supported_video_extension(path: str, expected: bool) -> None:
    assert is_supported_video_extension(path) is expected


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "0:00:00"), (59, "0:00:59"), (60, "0:01:00"), (3661, "1:01:01")],
)
def test_format_duration(seconds: float, expected: str) -> None:
    assert format_duration(seconds) == expected


def test_format_duration_clamps_negative_values() -> None:
    assert format_duration(-10) == "0:00:00"


def test_safe_divide_handles_zero_denominator() -> None:
    assert safe_divide(5, 0) == 0.0
    assert safe_divide(5, 0, default=-1.0) == -1.0
    assert safe_divide(10, 4) == 2.5


def test_clamp() -> None:
    assert clamp(0.5, 0.0, 1.0) == 0.5
    assert clamp(-3, 0.0, 1.0) == 0.0
    assert clamp(9, 0.0, 1.0) == 1.0


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


def test_configure_logging_does_not_duplicate_handlers() -> None:
    configure_logging()
    configure_logging(verbose=True)
    assert len(logging.getLogger().handlers) == 1
    assert logging.getLogger().level == logging.DEBUG


def test_configure_logging_quiet_wins_over_verbose() -> None:
    configure_logging(verbose=True, quiet=True)
    assert logging.getLogger().level == logging.WARNING


# --------------------------------------------------------------------------- #
# Snapshots
# --------------------------------------------------------------------------- #


def test_save_snapshot_writes_a_readable_image(output_dir: Path, synthetic_frame) -> None:
    cv2 = pytest.importorskip("cv2")

    path = save_snapshot(synthetic_frame, output_dir, FIXED_MOMENT)

    assert path.exists()
    assert path.stat().st_size > 0
    assert path.name == "snapshot_2026_09_17_224501.jpg"

    reloaded = cv2.imread(str(path))
    assert reloaded is not None
    assert reloaded.shape == synthetic_frame.shape


def test_save_snapshot_creates_missing_directory(tmp_path: Path, synthetic_frame) -> None:
    pytest.importorskip("cv2")

    target = tmp_path / "nested" / "outputs"
    path = save_snapshot(synthetic_frame, target, FIXED_MOMENT)

    assert path.exists()
    assert target.is_dir()


def test_save_snapshot_rejects_an_empty_frame(output_dir: Path) -> None:
    pytest.importorskip("cv2")

    import numpy as np

    with pytest.raises(SnapshotError):
        save_snapshot(None, output_dir)
    with pytest.raises(SnapshotError):
        save_snapshot(np.zeros((0, 0, 3), dtype=np.uint8), output_dir)


# --------------------------------------------------------------------------- #
# JSON output
# --------------------------------------------------------------------------- #


def test_write_json_round_trips(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "session_summary.json"
    payload = {"frames": 12, "classes": {"person": 3}, "ratio": 0.5}

    written = write_json(destination, payload)

    assert written == destination
    assert json.loads(destination.read_text(encoding="utf-8")) == payload


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #


def test_video_recorder_write_before_start_returns_false(output_dir: Path, synthetic_frame) -> None:
    recorder = VideoRecorder(output_dir)
    assert recorder.is_recording is False
    assert recorder.write(synthetic_frame) is False
    assert recorder.stop() is None


def test_video_recorder_records_and_stops(output_dir: Path, synthetic_frame) -> None:
    """Record three frames and confirm a non-empty video file is produced."""
    pytest.importorskip("cv2")

    recorder = VideoRecorder(output_dir, fps=20.0)
    height, width = synthetic_frame.shape[:2]

    try:
        path = recorder.start((width, height))
    except RecordingError as exc:  # pragma: no cover - depends on local codecs
        pytest.skip(f"No video codec available in this environment: {exc}")

    assert recorder.is_recording is True
    assert path.name.startswith("recording_")

    for _ in range(3):
        assert recorder.write(synthetic_frame) is True

    assert recorder.frames_written == 3

    finished = recorder.stop()

    assert finished == path
    assert recorder.is_recording is False
    assert finished is not None and finished.exists() and finished.stat().st_size > 0


def test_video_recorder_start_is_idempotent(output_dir: Path, synthetic_frame) -> None:
    pytest.importorskip("cv2")

    recorder = VideoRecorder(output_dir, fps=20.0)
    height, width = synthetic_frame.shape[:2]

    try:
        first = recorder.start((width, height))
    except RecordingError as exc:  # pragma: no cover - depends on local codecs
        pytest.skip(f"No video codec available in this environment: {exc}")

    try:
        assert recorder.start((width, height)) == first
    finally:
        recorder.stop()


def test_video_recorder_rejects_invalid_frame_size(output_dir: Path) -> None:
    pytest.importorskip("cv2")

    recorder = VideoRecorder(output_dir)
    with pytest.raises(RecordingError):
        recorder.start((0, 0))
