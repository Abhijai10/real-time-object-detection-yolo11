"""YOLO11 model loading and object detection.

The YOLO11 neural network is **third-party software** provided by Ultralytics.
This module does not re-implement any part of the network; it wraps the official
Ultralytics Python API and converts the raw results into a small, explicit
:class:`Detection` structure that the rest of the application can consume.

``ultralytics`` and ``torch`` are imported lazily inside :meth:`Detector.load`
so that importing this module (and therefore running the pure unit tests) does
not require the heavy dependency to be present or the model to be downloaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.utils import get_logger

__all__ = [
    "Detection",
    "Detector",
    "DetectionError",
    "ModelLoadError",
    "filter_by_confidence",
    "resolve_device",
]

logger = get_logger(__name__)

#: Model used when the user does not override ``--model``.  ``yolo11n`` is the
#: smallest YOLO11 detection model, chosen so the project runs on an ordinary
#: CPU while still benefiting from a GPU when one is present.
DEFAULT_MODEL: str = "yolo11n.pt"


class ModelLoadError(RuntimeError):
    """Raised when the YOLO11 weights cannot be loaded."""


class DetectionError(RuntimeError):
    """Raised when inference fails on a frame."""


# --------------------------------------------------------------------------- #
# Detection value object
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Detection:
    """A single detected object.

    Attributes:
        class_id: Integer class index produced by the network.
        class_name: Human readable class label, e.g. ``"person"``.
        confidence: Detection confidence in ``[0, 1]``.
        x1: Left edge of the bounding box in pixels.
        y1: Top edge of the bounding box in pixels.
        x2: Right edge of the bounding box in pixels.
        y2: Bottom edge of the bounding box in pixels.
    """

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Bounding box as ``(x1, y1, x2, y2)``."""
        return self.x1, self.y1, self.x2, self.y2

    @property
    def bbox_int(self) -> tuple[int, int, int, int]:
        """Bounding box rounded to integer pixel coordinates."""
        return int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2))

    @property
    def width(self) -> float:
        """Width of the bounding box in pixels."""
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        """Height of the bounding box in pixels."""
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        """Area of the bounding box in square pixels."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Centre point of the bounding box."""
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def label(self) -> str:
        """Text rendered next to the bounding box, e.g. ``"person 0.87"``."""
        return f"{self.class_name} {self.confidence:.2f}"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the detection."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(value), 2) for value in self.bbox],
            "width": round(self.width, 2),
            "height": round(self.height, 2),
        }


def filter_by_confidence(
    detections: Iterable[Detection], threshold: float
) -> list[Detection]:
    """Keep only detections whose confidence reaches ``threshold``.

    Args:
        detections: Detections to filter.
        threshold: Minimum accepted confidence in ``[0, 1]``.

    Returns:
        A new list containing the accepted detections, in their original order.
    """
    return [detection for detection in detections if detection.confidence >= threshold]


# --------------------------------------------------------------------------- #
# Device selection
# --------------------------------------------------------------------------- #


def resolve_device(requested: str | None = "auto") -> str:
    """Resolve a user supplied device string into one Ultralytics accepts.

    Args:
        requested: ``"auto"``, ``"cpu"``, ``"mps"`` or a CUDA index such as
            ``"0"``.  ``None`` is treated as ``"auto"``.

    Returns:
        ``"cpu"``, ``"mps"`` or a CUDA device string.  When ``"auto"`` is
        requested the best available device is chosen: CUDA if a GPU is
        available, then Apple Silicon MPS, otherwise CPU.
    """
    requested = (requested or "auto").strip().lower()

    if requested in {"cpu", "mps"}:
        return requested
    if requested == "auto":
        return _best_available_device()
    return requested


def _best_available_device() -> str:
    """Return the best device available on this machine.

    Falls back to ``"cpu"`` when PyTorch is not installed or reports no
    accelerator, which is the expected situation in a lightweight test run.
    """
    try:
        import torch  # noqa: PLC0415 - intentionally lazy
    except ImportError:
        logger.debug("PyTorch not importable; defaulting to CPU")
        return "cpu"

    try:
        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover - defensive, torch internals vary
        logger.debug("Accelerator probe failed; defaulting to CPU", exc_info=True)

    return "cpu"


def describe_device(device: str) -> str:
    """Return a human readable description of a resolved device string.

    Args:
        device: A value previously returned by :func:`resolve_device`.
    """
    if device == "cpu":
        return "CPU"
    if device == "mps":
        return "Apple Silicon GPU (MPS)"
    return f"CUDA device {device}"


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #


class Detector:
    """Thin wrapper around the Ultralytics YOLO11 detection model.

    Args:
        model: Path to a local ``.pt`` file, or the name of an official
            Ultralytics checkpoint such as ``"yolo11n.pt"``.  Official names are
            downloaded automatically by Ultralytics on first use.
        confidence: Confidence threshold passed to the model.
        iou: IoU threshold used by non-maximum suppression.
        device: ``"auto"``, ``"cpu"``, ``"mps"`` or a CUDA index.
        image_size: Inference resolution in pixels.
        max_detections: Upper bound on detections returned per frame.
    """

    def __init__(
        self,
        model: str | Path = DEFAULT_MODEL,
        confidence: float = 0.25,
        iou: float = 0.45,
        device: str | None = "auto",
        image_size: int = 640,
        max_detections: int = 300,
    ) -> None:
        self.model_path = str(model)
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.image_size = int(image_size)
        self.max_detections = int(max_detections)
        self.device = resolve_device(device)
        self._model: Any | None = None

    # -- lifecycle --------------------------------------------------------- #

    @property
    def is_loaded(self) -> bool:
        """``True`` once the weights have been loaded into memory."""
        return self._model is not None

    @property
    def class_names(self) -> dict[int, str]:
        """Mapping of class index to class name reported by the model."""
        if self._model is None:
            return {}
        names = getattr(self._model, "names", {}) or {}
        return {int(key): str(value) for key, value in dict(names).items()}

    def load(self) -> "Detector":
        """Load the YOLO11 weights, downloading them if necessary.

        Returns:
            ``self``, so the call can be chained.

        Raises:
            ModelLoadError: If Ultralytics is not installed, the local weights
                file is missing, or the checkpoint cannot be deserialised.
        """
        if self.is_loaded:
            return self

        path = Path(self.model_path).expanduser()

        # A *local* checkpoint is one that lives in a directory, or that already
        # exists on disk.  A bare name such as "yolo11n.pt" that is not present
        # locally is an official Ultralytics checkpoint, which the library
        # downloads automatically, so it must not be rejected here.
        is_local_file = path.parent != Path(".") or path.exists()

        if is_local_file and not path.exists():
            raise ModelLoadError(
                f"Model file not found: {path}\n"
                "Provide an existing .pt file with --model, or use the official "
                f"checkpoint name '{DEFAULT_MODEL}' which Ultralytics downloads automatically."
            )

        try:
            from ultralytics import YOLO  # noqa: PLC0415 - intentionally lazy
        except ImportError as exc:
            raise ModelLoadError(
                "The 'ultralytics' package is not installed. "
                "Install the project dependencies with: pip install -r requirements.txt"
            ) from exc

        logger.info("Loading YOLO11 model '%s' on %s", self.model_path, describe_device(self.device))
        try:
            self._model = YOLO(str(path) if is_local_file else self.model_path)
        except Exception as exc:  # noqa: BLE001 - surface any loader failure cleanly
            raise ModelLoadError(
                f"Failed to load the YOLO11 model '{self.model_path}': {exc}\n"
                "Check that the file is a valid Ultralytics checkpoint and that you "
                "have network access if the weights must be downloaded."
            ) from exc

        class_count = len(self.class_names)
        logger.info("Model ready - %d object classes available", class_count)
        return self

    def warmup(self, image_size: int | None = None) -> None:
        """Run one dummy inference so the first real frame is not slowed down.

        Args:
            image_size: Optional override of the inference resolution.  A square
                grey image is used, so the result is discarded.
        """
        if not self.is_loaded:
            self.load()

        import numpy as np  # noqa: PLC0415 - lightweight, but kept local for symmetry

        size = int(image_size or self.image_size)
        dummy = np.zeros((size, size, 3), dtype=np.uint8)
        try:
            self._model.predict(
                source=dummy,
                conf=self.confidence,
                iou=self.iou,
                imgsz=self.image_size,
                device=self.device,
                max_det=self.max_detections,
                verbose=False,
            )
            logger.debug("Model warm-up completed at %dx%d", size, size)
        except Exception:  # noqa: BLE001 - warm-up must never be fatal
            logger.warning("Model warm-up failed; continuing anyway", exc_info=True)

    # -- inference --------------------------------------------------------- #

    def detect(self, frame: Any) -> list[Detection]:
        """Run object detection on a single BGR frame.

        Args:
            frame: A NumPy array in OpenCV's BGR layout.

        Returns:
            Detections ordered by descending confidence.

        Raises:
            ModelLoadError: If the model has not been loaded yet.
            DetectionError: If inference fails.
        """
        if not self.is_loaded:
            raise ModelLoadError("Detector.detect() called before Detector.load()")

        if frame is None or getattr(frame, "size", 0) == 0:
            raise DetectionError("Cannot run detection on an empty frame")

        try:
            results = self._model.predict(
                source=frame,
                conf=self.confidence,
                iou=self.iou,
                imgsz=self.image_size,
                device=self.device,
                max_det=self.max_detections,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 - convert to a clean application error
            raise DetectionError(f"YOLO11 inference failed: {exc}") from exc

        if not results:
            return []

        return self._to_detections(results[0])

    def _to_detections(self, result: Any) -> list[Detection]:
        """Convert one Ultralytics ``Results`` object into :class:`Detection`s.

        Args:
            result: A single element of the list returned by ``model.predict``.

        Returns:
            Detections sorted by descending confidence.
        """
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        names = dict(getattr(result, "names", None) or self.class_names)

        coordinates = boxes.xyxy.tolist()
        confidences = boxes.conf.tolist()
        class_ids = boxes.cls.tolist()

        detections: list[Detection] = []
        for (x1, y1, x2, y2), confidence, class_id in zip(coordinates, confidences, class_ids):
            index = int(class_id)
            detections.append(
                Detection(
                    class_id=index,
                    class_name=str(names.get(index, f"class_{index}")),
                    confidence=float(confidence),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                )
            )

        detections.sort(key=lambda detection: detection.confidence, reverse=True)
        return detections

    def set_confidence(self, confidence: float) -> float:
        """Update the confidence threshold used for subsequent frames.

        Args:
            confidence: New threshold in ``[0, 1]``.

        Returns:
            The stored threshold.
        """
        self.confidence = float(confidence)
        logger.debug("Confidence threshold set to %.2f", self.confidence)
        return self.confidence

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        state = "loaded" if self.is_loaded else "not loaded"
        return (
            f"Detector(model={self.model_path!r}, conf={self.confidence}, "
            f"iou={self.iou}, device={self.device!r}, {state})"
        )


def summarise_detections(detections: Sequence[Detection]) -> str:
    """Return a compact one-line description of a frame's detections.

    Args:
        detections: Detections to summarise.

    Returns:
        Example: ``"3 objects: person x2, chair x1"``
    """
    if not detections:
        return "0 objects"

    counts: dict[str, int] = {}
    for detection in detections:
        counts[detection.class_name] = counts.get(detection.class_name, 0) + 1

    rendered = ", ".join(f"{name} x{count}" for name, count in sorted(counts.items()))
    return f"{len(detections)} object{'s' if len(detections) != 1 else ''}: {rendered}"
