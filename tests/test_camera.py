"""Tests for app.camera."""

from pathlib import Path

import pytest

from app.camera import Camera, CameraError, backend_name


def test_defaults_to_camera_index_zero() -> None:
    assert Camera().source == 0


def test_accepts_a_video_file_as_the_source(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not a real video")

    camera = Camera(source=clip)

    assert camera.is_file_source
    assert "clip.mp4" in camera.describe()


def test_describe_names_the_camera_index() -> None:
    assert Camera(index=2).describe() == "camera index 2"


def test_reading_before_opening_raises() -> None:
    with pytest.raises(CameraError, match="not open"):
        Camera().read()


def test_opening_a_missing_video_file_raises(tmp_path: Path) -> None:
    with pytest.raises(CameraError, match="not found"):
        Camera(source=tmp_path / "missing.mp4").open()


def test_error_message_for_a_camera_explains_the_likely_causes() -> None:
    message = Camera(index=3)._error_message()

    assert "index 3" in message
    assert "permission" in message.lower()
    assert "--camera 1" in message


def test_error_message_for_a_file_mentions_the_path(tmp_path: Path) -> None:
    message = Camera(source=tmp_path / "clip.mp4")._error_message()

    assert "clip.mp4" in message


def test_release_is_safe_before_opening() -> None:
    Camera().release()  # must not raise


def test_context_manager_releases_the_camera_on_error(tmp_path: Path) -> None:
    camera = Camera(source=tmp_path / "missing.mp4")

    with pytest.raises(CameraError):
        with camera:
            pass

    assert not camera.is_open


def test_backend_name_is_a_non_empty_string() -> None:
    assert isinstance(backend_name(), str)
    assert backend_name()
