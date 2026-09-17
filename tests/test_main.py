"""Tests for the command line interface and the output files."""

import json
from pathlib import Path

import numpy as np
import pytest

from app.analytics import SessionStats
from app.main import parse_args, save_snapshot, write_session_summary


# -- Argument parsing ------------------------------------------------------- #


def test_default_arguments() -> None:
    args = parse_args([])

    assert args.camera == 0
    assert args.model == "yolo11n.pt"
    assert args.conf == pytest.approx(0.25)
    assert args.width is None
    assert args.height is None


def test_parsing_every_option() -> None:
    args = parse_args(
        [
            "--camera", "1",
            "--model", "custom.pt",
            "--conf", "0.40",
            "--width", "1280",
            "--height", "720",
        ]
    )

    assert args.camera == 1
    assert args.model == "custom.pt"
    assert args.conf == pytest.approx(0.40)
    assert args.width == 1280
    assert args.height == 720


def test_confidence_above_the_allowed_range_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--conf", "5.0"])


def test_confidence_below_the_allowed_range_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--conf", "0"])


def test_confidence_must_be_a_number() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--conf", "abc"])


def test_width_without_height_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--width", "1280"])


def test_a_negative_camera_index_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--camera", "-1"])


def test_help_exits_with_code_zero() -> None:
    with pytest.raises(SystemExit) as exit_info:
        parse_args(["--help"])

    assert exit_info.value.code == 0


# -- Output files ----------------------------------------------------------- #


def test_save_snapshot_writes_a_jpeg(tmp_path: Path) -> None:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    path = save_snapshot(frame, tmp_path)

    assert path.exists()
    assert path.suffix == ".jpg"
    assert path.name.startswith("snapshot_")


def test_save_snapshot_creates_the_output_directory(tmp_path: Path) -> None:
    frame = np.zeros((16, 16, 3), dtype=np.uint8)

    path = save_snapshot(frame, tmp_path / "nested" / "outputs")

    assert path.exists()


def test_session_summary_is_valid_json(tmp_path: Path) -> None:
    stats = SessionStats()

    path = write_session_summary(stats, tmp_path)
    written = json.loads(path.read_text())

    assert written["frames_processed"] == 0
    assert written["most_frequent_class"] is None
