"""Tests for :mod:`app.detector`.

The lightweight tests here use a fake Ultralytics result object, so they run
without downloading the YOLO11 weights.  The one test that really loads the
model is marked ``slow`` and is deselected by default.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.detector import (
    DEFAULT_MODEL,
    Detection,
    Detector,
    DetectionError,
    ModelLoadError,
    describe_device,
    filter_by_confidence,
    resolve_device,
    summarise_detections,
)


# --------------------------------------------------------------------------- #
# Detection value object
# --------------------------------------------------------------------------- #


def test_detection_geometry() -> None:
    detection = Detection(class_id=0, class_name="person", confidence=0.87, x1=10, y1=20, x2=110, y2=220)

    assert detection.bbox == (10, 20, 110, 220)
    assert detection.bbox_int == (10, 20, 110, 220)
    assert detection.width == 100
    assert detection.height == 200
    assert detection.area == 20_000
    assert detection.center == (60.0, 120.0)


def test_detection_label_format() -> None:
    detection = Detection(class_id=0, class_name="person", confidence=0.8712, x1=0, y1=0, x2=1, y2=1)
    assert detection.label == "person 0.87"


def test_detection_as_dict() -> None:
    detection = Detection(class_id=41, class_name="cup", confidence=0.5, x1=1, y1=2, x2=3, y2=4)
    payload = detection.as_dict()

    assert payload["class_id"] == 41
    assert payload["class_name"] == "cup"
    assert payload["bbox"] == [1.0, 2.0, 3.0, 4.0]


def test_detection_dimensions_never_go_negative() -> None:
    inverted = Detection(class_id=0, class_name="person", confidence=0.5, x1=100, y1=100, x2=50, y2=50)
    assert inverted.width == 0.0
    assert inverted.height == 0.0


# --------------------------------------------------------------------------- #
# Confidence filtering
# --------------------------------------------------------------------------- #


def test_filter_by_confidence_keeps_only_confident_detections(sample_detections) -> None:
    # The fixture holds confidences 0.91, 0.74 and 0.62.
    kept = filter_by_confidence(sample_detections, 0.70)

    assert len(kept) == 2
    assert all(detection.confidence >= 0.70 for detection in kept)


def test_filter_by_confidence_can_remove_everything(sample_detections) -> None:
    assert filter_by_confidence(sample_detections, 0.95) == []


def test_filter_by_confidence_is_inclusive_at_the_threshold(sample_detections) -> None:
    kept = filter_by_confidence(sample_detections, 0.62)
    assert any(detection.confidence == pytest.approx(0.62) for detection in kept)


def test_filter_by_confidence_returns_an_empty_list_when_nothing_qualifies(sample_detections) -> None:
    assert filter_by_confidence(sample_detections, 0.99) == []


# --------------------------------------------------------------------------- #
# Device resolution
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("requested", ["cpu", "mps", "0", "1"])
def test_resolve_device_passes_explicit_values_through(requested: str) -> None:
    assert resolve_device(requested) == requested


@pytest.mark.parametrize("requested", [None, "auto", "AUTO", " auto "])
def test_resolve_device_auto_selects_a_supported_device(requested: str | None) -> None:
    assert resolve_device(requested) in {"cpu", "mps", "0"}


def test_resolve_device_cpu_is_honoured_even_with_a_gpu_present() -> None:
    assert resolve_device("cpu") == "cpu"


@pytest.mark.parametrize(
    ("device", "expected_fragment"),
    [("cpu", "CPU"), ("mps", "MPS"), ("0", "CUDA")],
)
def test_describe_device(device: str, expected_fragment: str) -> None:
    assert expected_fragment in describe_device(device)


# --------------------------------------------------------------------------- #
# Detector construction and loading
# --------------------------------------------------------------------------- #


def test_detector_defaults_use_the_lightweight_model() -> None:
    detector = Detector()

    assert detector.model_path == DEFAULT_MODEL == "yolo11n.pt"
    assert detector.is_loaded is False
    assert detector.class_names == {}


def test_detector_accepts_an_official_checkpoint_name_without_touching_the_disk() -> None:
    """A bare official name must not be rejected as a missing local file."""
    detector = Detector(model="yolo11n.pt")
    assert detector.model_path == "yolo11n.pt"
    assert detector.is_loaded is False


def test_detector_rejects_a_missing_local_model_file(tmp_path: Path) -> None:
    detector = Detector(model=tmp_path / "missing_weights.pt")

    with pytest.raises(ModelLoadError, match="not found"):
        detector.load()


def test_detector_rejects_a_missing_model_in_a_subdirectory() -> None:
    detector = Detector(model="models/does_not_exist.pt")

    with pytest.raises(ModelLoadError, match="not found"):
        detector.load()


def test_detector_detect_before_load_raises() -> None:
    detector = Detector()

    with pytest.raises(ModelLoadError, match="before"):
        detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))


def test_detector_set_confidence() -> None:
    detector = Detector(confidence=0.25)
    assert detector.set_confidence(0.7) == pytest.approx(0.7)
    assert detector.confidence == pytest.approx(0.7)


def test_detector_repr_reports_the_load_state() -> None:
    assert "not loaded" in repr(Detector())


# --------------------------------------------------------------------------- #
# Result conversion (fake Ultralytics result, no model download)
# --------------------------------------------------------------------------- #


class _FakeBoxes:
    """Stand-in for ``ultralytics.engine.results.Boxes``."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return len(self.cls)


def _fake_result(rows: list[tuple[list[float], float, int]], names: dict[int, str]):
    xyxy = np.array([row[0] for row in rows], dtype=float).reshape(-1, 4)
    conf = np.array([row[1] for row in rows], dtype=float)
    cls = np.array([row[2] for row in rows], dtype=float)
    return SimpleNamespace(boxes=_FakeBoxes(xyxy, conf, cls), names=names)


def test_to_detections_maps_boxes_classes_and_confidences() -> None:
    detector = Detector()
    result = _fake_result(
        [([10, 20, 110, 220], 0.55, 0), ([30, 40, 60, 80], 0.95, 41)],
        names={0: "person", 41: "cup"},
    )

    detections = detector._to_detections(result)  # noqa: SLF001 - conversion is the unit under test

    assert len(detections) == 2
    # Sorted by descending confidence.
    assert [detection.class_name for detection in detections] == ["cup", "person"]
    assert detections[0].confidence == pytest.approx(0.95)
    assert detections[1].bbox == (10.0, 20.0, 110.0, 220.0)


def test_to_detections_handles_an_empty_result() -> None:
    detector = Detector()
    result = _fake_result([], names={0: "person"})

    assert detector._to_detections(result) == []  # noqa: SLF001


def test_to_detections_falls_back_for_unknown_class_ids() -> None:
    detector = Detector()
    result = _fake_result([([0, 0, 10, 10], 0.5, 7)], names={0: "person"})

    detections = detector._to_detections(result)  # noqa: SLF001

    assert detections[0].class_name == "class_7"


# --------------------------------------------------------------------------- #
# Summary helper
# --------------------------------------------------------------------------- #


def test_summarise_detections_with_no_objects() -> None:
    assert summarise_detections([]) == "0 objects"


def test_summarise_detections_counts_each_class(sample_detections) -> None:
    summary = summarise_detections(sample_detections)

    assert summary.startswith("3 objects:")
    assert "person x2" in summary
    assert "cup x1" in summary


def test_summarise_detections_uses_the_singular_form() -> None:
    single = [Detection(class_id=0, class_name="person", confidence=0.9, x1=0, y1=0, x2=1, y2=1)]
    assert summarise_detections(single).startswith("1 object:")


# --------------------------------------------------------------------------- #
# Integration: real model (deselected by default)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_detector_loads_yolo11n_and_runs_inference(synthetic_frame) -> None:
    """Load the real YOLO11n weights and run one inference pass.

    This test downloads ``yolo11n.pt`` (~5 MB) on first use, so it is marked
    ``slow`` and deselected by default.  Run it with ``pytest -m slow``.

    The synthetic frame contains no real objects, so the test asserts the
    *pipeline* works - it does not assert any particular detection.
    """
    detector = Detector(model="yolo11n.pt", confidence=0.25, device="cpu")

    try:
        detector.load()
    except ModelLoadError as exc:
        pytest.skip(f"YOLO11n could not be loaded in this environment: {exc}")

    assert detector.is_loaded is True
    assert len(detector.class_names) == 80

    detections = detector.detect(synthetic_frame)

    assert isinstance(detections, list)
    for detection in detections:
        assert 0.25 <= detection.confidence <= 1.0
        assert detection.class_name in detector.class_names.values()


@pytest.mark.slow
def test_detector_reports_a_clean_error_for_an_empty_frame() -> None:
    detector = Detector(model="yolo11n.pt", device="cpu")

    try:
        detector.load()
    except ModelLoadError as exc:
        pytest.skip(f"YOLO11n could not be loaded in this environment: {exc}")

    with pytest.raises(DetectionError, match="empty frame"):
        detector.detect(np.zeros((0, 0, 3), dtype=np.uint8))
