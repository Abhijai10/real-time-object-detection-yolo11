"""Tests for :mod:`app.cli`: argument parsing, validation and action handling.

No webcam, no display and no model download is required: the application
controller is exercised through the ``FakeDetector`` test double.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.cli import AppConfig, Application, build_parser, main, parse_args
from app.controls import ControlAction
from app.utils import RecordingError

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def test_parser_builds_without_error() -> None:
    parser = build_parser()
    assert parser.prog == "python -m app"


def test_help_exits_zero_and_documents_the_main_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        parse_args(["--help"])

    assert excinfo.value.code == 0

    output = capsys.readouterr().out
    for option in ("--camera", "--model", "--conf", "--iou", "--imgsz", "--device", "--width", "--height", "--output"):
        assert option in output
    assert "S -" in output  # keyboard controls are documented in the epilog


def test_version_flag_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        parse_args(["--version"])

    assert excinfo.value.code == 0
    assert "1.0.0" in capsys.readouterr().out


def test_defaults_allow_a_zero_argument_start(tmp_path: Path) -> None:
    """``python -m app`` with no arguments must produce a usable configuration."""
    _args, config = parse_args([])

    assert config.camera == 0
    assert config.model == "yolo11n.pt"
    assert config.confidence == pytest.approx(0.25)
    assert config.iou == pytest.approx(0.45)
    assert config.image_size == 640
    assert config.device == "auto"
    assert config.width is None and config.height is None
    assert config.display is True
    assert config.max_frames == 0


def test_parsing_a_full_command_line() -> None:
    _args, config = parse_args(
        [
            "--camera",
            "1",
            "--model",
            "models/custom.pt",
            "--conf",
            "0.40",
            "--iou",
            "0.60",
            "--imgsz",
            "512",
            "--device",
            "cpu",
            "--width",
            "1280",
            "--height",
            "720",
            "--output",
            "out",
            "--max-frames",
            "5",
        ]
    )

    assert config.camera == 1
    assert config.model == "models/custom.pt"
    assert config.confidence == pytest.approx(0.40)
    assert config.iou == pytest.approx(0.60)
    assert config.image_size == 512
    assert config.device == "cpu"
    assert (config.width, config.height) == (1280, 720)
    assert config.output_dir == Path("out")
    assert config.max_frames == 5


# --------------------------------------------------------------------------- #
# Validation through the CLI
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("value", ["0", "-0.5", "1.5", "abc"])
def test_invalid_confidence_is_rejected_with_exit_code_2(value: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        parse_args(["--conf", value])
    assert excinfo.value.code == 2


@pytest.mark.parametrize("value", ["-1", "-99", "999999", "abc"])
def test_invalid_camera_index_is_rejected(value: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        parse_args(["--camera", value])
    assert excinfo.value.code == 2


def test_camera_accepts_an_existing_video_file(dummy_video_file: Path) -> None:
    _args, config = parse_args(["--camera", str(dummy_video_file)])

    assert config.is_file_source is True
    assert Path(str(config.camera)).name == "clip.mp4"


def test_camera_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        parse_args(["--camera", str(tmp_path / "nope.mp4")])
    assert excinfo.value.code == 2


def test_camera_rejects_an_unsupported_extension(tmp_path: Path) -> None:
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("not a video", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        parse_args(["--camera", str(unsupported)])
    assert excinfo.value.code == 2


def test_width_without_height_is_rejected() -> None:
    with pytest.raises(SystemExit) as excinfo:
        parse_args(["--width", "1280"])
    assert excinfo.value.code == 2


@pytest.mark.parametrize("value", ["0", "-640"])
def test_invalid_dimensions_are_rejected(value: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        parse_args(["--width", value, "--height", "720"])
    assert excinfo.value.code == 2


def test_negative_max_frames_is_rejected() -> None:
    with pytest.raises(SystemExit) as excinfo:
        parse_args(["--max-frames", "-1"])
    assert excinfo.value.code == 2


def test_unknown_option_is_rejected() -> None:
    with pytest.raises(SystemExit) as excinfo:
        parse_args(["--not-an-option"])
    assert excinfo.value.code == 2


def test_headless_flags_are_parsed() -> None:
    _args, config = parse_args(["--no-display", "--no-help-overlay", "--no-summary"])

    assert config.display is False
    assert config.show_help is False
    assert config.save_summary is False


# --------------------------------------------------------------------------- #
# AppConfig validation
# --------------------------------------------------------------------------- #


def test_app_config_creates_the_output_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "outputs"
    config = AppConfig(output_dir=target).validate()

    assert target.is_dir()
    assert config.output_dir == target


def test_app_config_rejects_an_unusable_output_directory(tmp_path: Path) -> None:
    blocker = tmp_path / "file.txt"
    blocker.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="Output directory"):
        AppConfig(output_dir=blocker).validate()


@pytest.mark.parametrize("iou", [0.0, -0.1, 1.5])
def test_app_config_rejects_invalid_iou(iou: float) -> None:
    with pytest.raises(ValueError, match="IoU"):
        AppConfig(iou=iou).validate()


def test_app_config_rejects_a_non_positive_inference_size() -> None:
    with pytest.raises(ValueError, match="Inference size"):
        AppConfig(image_size=0).validate()


# --------------------------------------------------------------------------- #
# --list-cameras
# --------------------------------------------------------------------------- #


def test_list_cameras_reports_success_when_a_camera_is_found(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("app.cli.available_camera_indices", lambda limit=5: [0])

    assert main(["--list-cameras"]) == 0
    assert "Usable camera indices: 0" in capsys.readouterr().out


def test_list_cameras_reports_failure_when_none_is_found(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("app.cli.available_camera_indices", lambda limit=5: [])

    assert main(["--list-cameras"]) == 1
    assert "No usable camera was found" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Application controller (with a fake detector)
# --------------------------------------------------------------------------- #


@pytest.fixture
def application(fake_detector, output_dir: Path) -> Application:
    """Return an :class:`Application` wired to fakes and a temporary output dir."""
    config = AppConfig(output_dir=output_dir).validate()
    return Application(config, detector=fake_detector)


def test_application_snapshot_action_writes_a_file(
    application: Application, synthetic_frame, output_dir: Path
) -> None:
    pytest.importorskip("cv2")

    keep_running = application.handle_action(ControlAction.SNAPSHOT, synthetic_frame)

    assert keep_running is True
    snapshots = list(output_dir.glob("snapshot_*.jpg"))
    assert len(snapshots) == 1
    assert snapshots[0].stat().st_size > 0


def test_application_quit_action_stops_the_loop(application: Application, synthetic_frame) -> None:
    assert application.handle_action(ControlAction.QUIT, synthetic_frame) is False


def test_application_confidence_action_updates_the_detector(
    application: Application, synthetic_frame
) -> None:
    application.controls.confidence = 0.80

    application.handle_action(ControlAction.CONFIDENCE_UP, synthetic_frame)

    assert application.detector.confidence == pytest.approx(0.80)


def test_application_reset_action_clears_session_statistics(
    application: Application, synthetic_frame, sample_analytics
) -> None:
    application.session.update(sample_analytics, fps=10.0)
    assert application.session.frames_processed == 1

    application.handle_action(ControlAction.RESET_STATS, synthetic_frame)

    assert application.session.frames_processed == 0
    assert application.session.total_detections == 0


def test_application_recording_toggle_starts_and_stops(
    application: Application, synthetic_frame
) -> None:
    pytest.importorskip("cv2")

    application.controls.recording = True
    application.handle_action(ControlAction.TOGGLE_RECORDING, synthetic_frame)

    if not application.recorder.is_recording:
        pytest.skip("No video codec available in this environment")

    application.controls.recording = False
    application.handle_action(ControlAction.TOGGLE_RECORDING, synthetic_frame)

    assert application.recorder.is_recording is False


def test_application_recording_failure_is_handled_gracefully(
    application: Application, synthetic_frame, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*_args: object, **_kwargs: object) -> None:
        raise RecordingError("simulated codec failure")

    monkeypatch.setattr(application.recorder, "start", explode)

    application.controls.recording = True
    keep_running = application.handle_action(ControlAction.TOGGLE_RECORDING, synthetic_frame)

    assert keep_running is True
    # The flag must be rolled back so the UI does not claim to be recording.
    assert application.controls.recording is False


def test_application_summary_is_printable(
    application: Application, capsys: pytest.CaptureFixture[str]
) -> None:
    application.print_summary()
    assert "Session summary" in capsys.readouterr().out


def test_application_banner_reports_the_configuration(
    application: Application, capsys: pytest.CaptureFixture[str]
) -> None:
    application.print_banner()

    output = capsys.readouterr().out
    assert "Real-Time Object Detection and Scene Analytics Using YOLO11" in output
    assert "yolo11n.pt" in output
