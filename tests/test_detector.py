"""Tests for app.detector.

These tests do not download the YOLO11 weights. Loading the real model is slow
and needs a network connection, so the loading tests here only cover the error
paths.
"""

from pathlib import Path

import pytest

from app.detector import (
    Detection,
    Detector,
    ModelError,
    describe_device,
    filter_by_confidence,
    resolve_device,
)


def make_detection(class_name: str, confidence: float) -> Detection:
    """Build a Detection for use in tests."""
    return Detection(class_id=0, class_name=class_name, confidence=confidence, box=(0, 0, 10, 10))


# -- Detection -------------------------------------------------------------- #


def test_label_combines_class_and_confidence() -> None:
    assert make_detection("person", 0.876).label == "person 0.88"


def test_detection_stores_its_box() -> None:
    assert make_detection("chair", 0.5).box == (0, 0, 10, 10)


# -- Confidence filtering --------------------------------------------------- #


def test_filter_keeps_only_detections_at_or_above_the_threshold() -> None:
    detections = [
        make_detection("person", 0.90),
        make_detection("chair", 0.50),
        make_detection("bottle", 0.70),
    ]

    kept = filter_by_confidence(detections, 0.70)

    assert [detection.class_name for detection in kept] == ["person", "bottle"]


def test_filter_with_a_high_threshold_removes_everything() -> None:
    assert filter_by_confidence([make_detection("person", 0.30)], 0.90) == []


def test_filter_of_nothing_is_empty() -> None:
    assert filter_by_confidence([], 0.25) == []


# -- Detector --------------------------------------------------------------- #


def test_detector_uses_the_default_model() -> None:
    assert Detector().model_path == "yolo11n.pt"


def test_detector_is_not_loaded_before_load_is_called() -> None:
    assert not Detector().is_loaded


def test_detect_before_load_raises() -> None:
    with pytest.raises(ModelError, match="before"):
        Detector().detect(object())


def test_load_reports_a_missing_local_model(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="not found"):
        Detector(model=str(tmp_path / "nope.pt")).load()


def test_set_confidence_updates_the_threshold() -> None:
    detector = Detector()

    assert detector.set_confidence(0.5) == 0.5
    assert detector.confidence == 0.5


# -- Device selection ------------------------------------------------------- #


def test_resolve_device_passes_through_explicit_choices() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("mps") == "mps"


def test_resolve_device_auto_returns_a_supported_device() -> None:
    assert resolve_device("auto") in {"cpu", "mps", "0"}


def test_describe_device_is_readable() -> None:
    assert describe_device("cpu") == "CPU"
    assert "MPS" in describe_device("mps")
