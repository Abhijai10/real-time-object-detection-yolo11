"""Object detection using the pretrained YOLO11n model.

Module 2 of the project.

YOLO11 is third-party software written and trained by Ultralytics. This module
does not implement any part of the neural network: it loads the official
pretrained weights through the Ultralytics API and converts the raw output into
a small :class:`Detection` structure for the rest of the application.

No custom training or fine-tuning is performed anywhere in this project.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

__all__ = ["Detection", "Detector", "ModelError", "resolve_device", "DEFAULT_MODEL"]

#: The pretrained model used by default. ``yolo11n`` is the smallest YOLO11
#: detection model, so it runs on an ordinary laptop CPU as well as on a GPU.
DEFAULT_MODEL = "yolo11n.pt"


class ModelError(RuntimeError):
    """Raised when the YOLO11 model cannot be loaded or fails during inference."""


@dataclass
class Detection:
    """A single detected object in a frame.

    Attributes:
        class_id: Integer class index produced by the model.
        class_name: Readable label, e.g. ``"person"``.
        confidence: How sure the model is, between 0 and 1.
        box: Bounding box as ``(x1, y1, x2, y2)`` pixel coordinates.
    """

    class_id: int
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]

    @property
    def label(self) -> str:
        """Text drawn above the bounding box, e.g. ``"person 0.87"``."""
        return f"{self.class_name} {self.confidence:.2f}"


def resolve_device(requested: str = "auto") -> str:
    """Pick the device used for inference.

    Args:
        requested: ``"auto"``, ``"cpu"``, ``"mps"`` or a CUDA index such as
            ``"0"``.

    Returns:
        A device string that Ultralytics accepts. ``"auto"`` chooses CUDA if
        available, then Apple Silicon MPS, otherwise CPU.
    """
    requested = (requested or "auto").strip().lower()
    if requested != "auto":
        return requested

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe_device(device: str) -> str:
    """Return a readable name for a resolved device string."""
    if device == "cpu":
        return "CPU"
    if device == "mps":
        return "Apple Silicon GPU (MPS)"
    return f"CUDA device {device}"


class Detector:
    """Loads YOLO11n and runs detection on individual frames.

    Args:
        model: Checkpoint name such as ``"yolo11n.pt"``, or a path to a local
            ``.pt`` file. Ultralytics downloads official checkpoints on first
            use.
        confidence: Confidence threshold passed to the model.
        device: Inference device, see :func:`resolve_device`.
        image_size: Inference resolution in pixels.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        confidence: float = 0.25,
        device: str = "auto",
        image_size: int = 640,
    ) -> None:
        self.model_path = str(model)
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.device = resolve_device(device)
        self._model: Any | None = None

    @property
    def is_loaded(self) -> bool:
        """``True`` once the weights are in memory."""
        return self._model is not None

    @property
    def class_names(self) -> dict[int, str]:
        """Class index to class name, as reported by the model."""
        if self._model is None:
            return {}
        return {int(key): str(value) for key, value in dict(self._model.names).items()}

    def load(self) -> "Detector":
        """Load the pretrained weights, downloading them if necessary.

        Returns:
            ``self``, so the call can be chained.

        Raises:
            ModelError: If Ultralytics is missing, a local file does not exist,
                or the checkpoint cannot be read.
        """
        if self.is_loaded:
            return self

        path = Path(self.model_path).expanduser()
        # A bare name like "yolo11n.pt" is an official checkpoint that
        # Ultralytics downloads itself, so it must not be treated as a missing
        # local file. Anything with a directory part, or that already exists on
        # disk, is a real local path.
        is_local = path.parent != Path(".") or path.exists()
        if is_local and not path.exists():
            raise ModelError(f"Error: model file not found: {path}")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelError(
                "Error: the 'ultralytics' package is not installed.\n"
                "Run: pip install -r requirements.txt"
            ) from exc

        try:
            self._model = YOLO(str(path) if is_local else self.model_path)
        except Exception as exc:
            raise ModelError(
                f"Error: could not load the model '{self.model_path}': {exc}"
            ) from exc

        return self

    def detect(self, frame: Any) -> list[Detection]:
        """Detect objects in a single BGR frame.

        Args:
            frame: An OpenCV frame (a NumPy array).

        Returns:
            The detections, most confident first.

        Raises:
            ModelError: If the model is not loaded or inference fails.
        """
        if not self.is_loaded:
            raise ModelError("Detector.detect() was called before Detector.load()")
        if frame is None or getattr(frame, "size", 0) == 0:
            raise ModelError("Cannot run detection on an empty frame")

        try:
            results = self._model.predict(
                source=frame,
                conf=self.confidence,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            raise ModelError(f"Error: YOLO11 inference failed: {exc}") from exc

        if not results:
            return []
        return self._convert(results[0])

    def _convert(self, result: Any) -> list[Detection]:
        """Turn one Ultralytics result object into a list of detections."""
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        names = dict(getattr(result, "names", None) or self.class_names)
        detections = [
            Detection(
                class_id=int(class_id),
                class_name=str(names.get(int(class_id), f"class_{int(class_id)}")),
                confidence=float(confidence),
                box=(float(x1), float(y1), float(x2), float(y2)),
            )
            for (x1, y1, x2, y2), confidence, class_id in zip(
                boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
            )
        ]
        detections.sort(key=lambda detection: detection.confidence, reverse=True)
        return detections

    def set_confidence(self, confidence: float) -> float:
        """Update the confidence threshold used from the next frame onwards."""
        self.confidence = float(confidence)
        return self.confidence

    def __repr__(self) -> str:
        state = "loaded" if self.is_loaded else "not loaded"
        return f"Detector(model={self.model_path!r}, device={self.device!r}, {state})"


def filter_by_confidence(
    detections: Sequence[Detection], threshold: float
) -> list[Detection]:
    """Return only the detections whose confidence reaches ``threshold``."""
    return [detection for detection in detections if detection.confidence >= threshold]
