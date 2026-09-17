"""Tests for :mod:`app.camera`.

These tests cover camera *validation* and error handling only.  They never
require a physical webcam: opening a real device is skipped when none is
present, and the remaining assertions exercise the deterministic error paths.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.camera import (
    Camera,
    CameraError,
    _capture_native_stderr,
    _denied_by_operating_system,
    capture_backend_name,
    probe_camera,
)


# --------------------------------------------------------------------------- #
# Construction and validation
# --------------------------------------------------------------------------- #


def test_camera_defaults_to_device_index_zero() -> None:
    camera = Camera()

    assert camera.source == 0
    assert camera.is_file_source is False
    assert camera.is_open is False
    assert camera.frames_read == 0


def test_camera_rejects_a_negative_index() -> None:
    with pytest.raises(ValueError):
        Camera(index=-1)


def test_camera_rejects_an_unreasonably_large_index() -> None:
    with pytest.raises(ValueError):
        Camera(index=999999)


def test_camera_accepts_a_numeric_string_index() -> None:
    assert Camera(index="2").source == 2


def test_camera_accepts_a_video_file_path(dummy_video_file: Path) -> None:
    camera = Camera(source=dummy_video_file)

    assert camera.is_file_source is True
    assert camera.source == str(dummy_video_file)


def test_camera_rejects_width_without_height() -> None:
    with pytest.raises(ValueError, match="together"):
        Camera(index=0, width=1280)


def test_camera_stores_the_requested_resolution() -> None:
    camera = Camera(index=0, width=1280, height=720)
    assert (camera.width, camera.height) == (1280, 720)


# --------------------------------------------------------------------------- #
# Descriptions and error messages
# --------------------------------------------------------------------------- #


def test_describe_names_the_camera_index() -> None:
    assert Camera(index=3).describe() == "camera index 3"


def test_describe_names_the_video_file(dummy_video_file: Path) -> None:
    assert "clip.mp4" in Camera(source=dummy_video_file).describe()


def test_open_error_message_for_a_camera_is_actionable() -> None:
    message = Camera(index=0)._open_error_message()  # noqa: SLF001 - message content is the contract

    assert "Unable to open webcam at camera index 0" in message
    assert "--camera 1" in message


def test_open_error_message_for_a_file_mentions_the_path(dummy_video_file: Path) -> None:
    message = Camera(source=dummy_video_file)._open_error_message()  # noqa: SLF001

    assert "Unable to open the video source" in message
    assert "clip.mp4" in message


# --------------------------------------------------------------------------- #
# Operating-system permission denials
# --------------------------------------------------------------------------- #

#: The exact text OpenCV's AVFoundation backend prints on macOS when the OS
#: denies camera access.  Captured from a real run on a machine with a
#: FaceTime HD Camera that had not granted the calling process permission.
_NATIVE_PERMISSION_DENIAL = (
    "OpenCV: not authorized to capture video (status 0), requesting...\n"
    "OpenCV: camera failed to properly initialize!\n"
    "[ WARN:0@0.158] global cap.cpp:477 open VIDEOIO(AVFOUNDATION): backend is "
    "generally available but can't be used to capture by index\n"
)


def test_denied_by_operating_system_detects_a_real_denial() -> None:
    assert _denied_by_operating_system([_NATIVE_PERMISSION_DENIAL]) is True


def test_denied_by_operating_system_ignores_unrelated_output() -> None:
    assert _denied_by_operating_system(["OpenCV: camera failed to properly initialize!"]) is False
    assert _denied_by_operating_system([]) is False


def test_permission_denial_produces_permission_specific_guidance() -> None:
    """A denied permission must not be reported as a missing camera."""
    message = Camera(index=0)._open_error_message([_NATIVE_PERMISSION_DENIAL])  # noqa: SLF001

    assert "denied camera access" in message
    assert "Privacy & Security > Camera" in message
    assert "Quit and reopen that terminal" in message
    # The generic "no webcam is connected" list must NOT appear here.
    assert "No webcam is connected" not in message


def test_generic_failure_keeps_the_cause_list() -> None:
    message = Camera(index=0)._open_error_message()  # noqa: SLF001

    assert "Unable to open webcam at camera index 0" in message
    assert "No webcam is connected" in message
    assert "denied camera access" not in message


def test_file_source_failure_never_reports_a_camera_permission_problem(
    dummy_video_file: Path,
) -> None:
    message = Camera(source=dummy_video_file)._open_error_message(  # noqa: SLF001
        [_NATIVE_PERMISSION_DENIAL]
    )

    assert "Unable to open the video source" in message
    assert "denied camera access" not in message


def test_capture_native_stderr_captures_native_output() -> None:
    with _capture_native_stderr() as captured:
        os.write(2, b"OpenCV: not authorized to capture video (status 0)\n")

    assert len(captured) == 1
    assert "not authorized" in captured[0]


def test_capture_native_stderr_restores_the_original_descriptor() -> None:
    """After the block, file descriptor 2 must point at the real stderr again."""
    before = os.fstat(2)
    with _capture_native_stderr():
        pass
    after = os.fstat(2)

    assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)


def test_capture_backend_name_is_a_non_empty_string() -> None:
    assert isinstance(capture_backend_name(), str)
    assert capture_backend_name()


# --------------------------------------------------------------------------- #
# Lifecycle error handling
# --------------------------------------------------------------------------- #


def test_reading_before_opening_raises_a_clear_error() -> None:
    with pytest.raises(CameraError, match="not open"):
        Camera(index=0).read()


def test_opening_a_missing_video_file_raises(tmp_path: Path) -> None:
    pytest.importorskip("cv2")

    camera = Camera(source=tmp_path / "missing.mp4")

    with pytest.raises(CameraError, match="not found"):
        camera.open()


def test_opening_an_unsupported_extension_raises(tmp_path: Path) -> None:
    pytest.importorskip("cv2")

    unsupported = tmp_path / "clip.txt"
    unsupported.write_text("not a video", encoding="utf-8")

    with pytest.raises(CameraError, match="Unsupported video extension"):
        Camera(source=unsupported).open()


def test_release_is_idempotent_and_safe_before_open() -> None:
    camera = Camera(index=0)
    camera.release()
    camera.release()
    assert camera.is_open is False


def test_properties_are_safe_when_the_camera_is_closed() -> None:
    camera = Camera(index=0)

    assert camera.actual_width == 0
    assert camera.actual_height == 0
    assert camera.reported_fps == 0.0
    assert camera.frame_count_hint == 0


def test_context_manager_releases_the_device_on_error(tmp_path: Path) -> None:
    pytest.importorskip("cv2")

    camera = Camera(source=tmp_path / "missing.mp4")

    with pytest.raises(CameraError):
        with camera:
            pass  # opening fails, __exit__ must still run

    assert camera.is_open is False


# --------------------------------------------------------------------------- #
# Hardware probes (never fail when no camera exists)
# --------------------------------------------------------------------------- #


def test_probe_camera_returns_a_boolean() -> None:
    """The probe must report a boolean and never raise on a camera-less machine."""
    pytest.importorskip("cv2")
    assert isinstance(probe_camera(0), bool)


def test_probe_camera_rejects_an_invalid_index() -> None:
    with pytest.raises(ValueError):
        probe_camera(-1)


@pytest.mark.display
def test_webcam_can_be_opened_and_read() -> None:
    """Hardware smoke test.  Deselected by default; run with ``pytest -m display``."""
    pytest.importorskip("cv2")

    camera = Camera(index=0)
    try:
        camera.open()
    except CameraError as exc:
        pytest.skip(f"No usable webcam in this environment: {exc}")

    try:
        ok, frame = camera.read()
        assert ok is True
        assert frame is not None
        assert frame.shape[0] > 0 and frame.shape[1] > 0
    finally:
        camera.release()
