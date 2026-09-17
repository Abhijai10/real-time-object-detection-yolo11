"""Shared pytest fixtures for the Real-Time Object Detection test suite.

Every fixture here is hardware-free: no webcam, no display and no model download
is required.  Tests that *do* need the YOLO11 weights or a window are marked
``slow`` / ``display`` and are deselected by default (see ``pyproject.toml``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.analytics import FrameAnalytics, compute_frame_analytics
from app.detector import Detection

# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #


@pytest.fixture
def synthetic_frame() -> np.ndarray:
    """Return a deterministic 480x640 BGR frame with simple shapes drawn on it.

    The frame is generated, not photographed, so it contains no third-party
    imagery and no personal data.
    """
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (40, 45, 60)

    # A few bright rectangles give the frame non-trivial content.
    frame[80:200, 60:180] = (200, 200, 200)
    frame[240:400, 300:460] = (120, 190, 240)
    frame[300:460, 80:220] = (90, 210, 140)
    return frame


@pytest.fixture
def sample_detections() -> list[Detection]:
    """Return a small, deterministic list of detections."""
    return [
        Detection(class_id=0, class_name="person", confidence=0.91, x1=60, y1=80, x2=180, y2=200),
        Detection(class_id=0, class_name="person", confidence=0.74, x1=300, y1=240, x2=460, y2=400),
        Detection(class_id=41, class_name="cup", confidence=0.62, x1=80, y1=300, x2=220, y2=460),
    ]


@pytest.fixture
def sample_analytics(sample_detections: list[Detection]) -> FrameAnalytics:
    """Return the :class:`FrameAnalytics` matching :func:`sample_detections`."""
    return compute_frame_analytics(sample_detections)


# --------------------------------------------------------------------------- #
# Filesystem
# --------------------------------------------------------------------------- #


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Return a temporary directory used as the application's output folder."""
    directory = tmp_path / "outputs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@pytest.fixture
def dummy_video_file(tmp_path: Path) -> Path:
    """Return the path of a placeholder ``.mp4`` file.

    Only the *path* is used: the tests that consume this fixture exercise
    validation logic, not video decoding.
    """
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    return path


@pytest.fixture
def dummy_model_file(tmp_path: Path) -> Path:
    """Return the path of a placeholder ``.pt`` file.

    Used to check that a *missing* local checkpoint produces a clean error
    without downloading anything.
    """
    return tmp_path / "missing_weights.pt"


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #


class FakeDetector:
    """Minimal stand-in for :class:`app.detector.Detector`.

    Lets the application loop be exercised without Ultralytics or the model
    weights.
    """

    def __init__(self, detections: list[Detection] | None = None) -> None:
        self._detections = list(detections or [])
        self.confidence = 0.25
        self.loaded = False

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    def load(self) -> "FakeDetector":
        self.loaded = True
        return self

    def warmup(self, image_size: int | None = None) -> None:
        return None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return list(self._detections)

    def set_confidence(self, confidence: float) -> float:
        self.confidence = float(confidence)
        return self.confidence


@pytest.fixture
def fake_detector(sample_detections: list[Detection]) -> FakeDetector:
    """Return a loaded :class:`FakeDetector` producing :func:`sample_detections`."""
    detector = FakeDetector(sample_detections)
    detector.load()
    return detector
