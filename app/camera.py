"""Webcam initialisation, frame capture and cleanup (Module 1 - Camera Capture).

OpenCV is imported defensively so that this module can still be imported - and
its non-hardware helpers unit tested - on a machine where the OpenCV wheel is
unavailable or cannot load its native libraries.
"""

from __future__ import annotations

import contextlib
import os
import platform
from pathlib import Path
from typing import Any, Iterator

from app.utils import (
    SUPPORTED_VIDEO_EXTENSIONS,
    get_logger,
    is_supported_video_extension,
    validate_camera_index,
    validate_resolution,
)

try:  # pragma: no cover - exercised implicitly by every camera test
    import cv2
except Exception:  # noqa: BLE001 - missing native libraries raise ImportError/OSError
    cv2 = None  # type: ignore[assignment]

__all__ = [
    "Camera",
    "CameraError",
    "available_camera_indices",
    "capture_backend_name",
    "probe_camera",
    "require_opencv",
]

logger = get_logger(__name__)


@contextlib.contextmanager
def _silence_native_stderr() -> Iterator[None]:
    """Temporarily hide the OpenCV backend's native stderr chatter.

    When no device is present, OpenCV's video backends print messages such as
    ``OpenCV: camera failed to properly initialize!`` straight from C++, before
    any Python code runs.  Those lines are confusing when they appear next to
    our own clear error messages, so they are suppressed while a device is
    opened or probed.

    This is purely cosmetic: if the file descriptors cannot be redirected the
    context manager degrades to a no-op rather than failing.
    """
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved_stderr = os.dup(2)
    except OSError:  # pragma: no cover - platform dependent
        yield
        return

    try:
        os.dup2(devnull, 2)
        yield
    finally:
        try:
            os.dup2(saved_stderr, 2)
        finally:
            os.close(saved_stderr)
            os.close(devnull)


#: Number of frames to read while probing a device.  Some drivers need a couple
#: of attempts before they return a valid frame.
_PROBE_ATTEMPTS: int = 5


class CameraError(RuntimeError):
    """Raised when a camera or video source cannot be opened or read."""


def require_opencv() -> Any:
    """Return the OpenCV module, raising a helpful error when it is missing.

    Returns:
        The imported :mod:`cv2` module.

    Raises:
        CameraError: If OpenCV could not be imported.
    """
    if cv2 is None:
        raise CameraError(
            "OpenCV is not available. Install the project dependencies with:\n"
            "    pip install -r requirements.txt"
        )
    return cv2


def capture_backend_name() -> str:
    """Return the capture backend used on the current platform.

    Returns:
        A short human readable backend name, e.g. ``"AVFoundation"``.
    """
    system = platform.system()
    return {"Darwin": "AVFoundation", "Windows": "DirectShow / MSMF"}.get(system, "V4L2 / default")


def _resolve_backend() -> int:
    """Return the OpenCV backend flag best suited to this platform."""
    module = require_opencv()
    system = platform.system()
    if system == "Darwin":
        return getattr(module, "CAP_AVFOUNDATION", getattr(module, "CAP_ANY", 0))
    if system == "Windows":
        return getattr(module, "CAP_DSHOW", getattr(module, "CAP_ANY", 0))
    return getattr(module, "CAP_ANY", 0)


class Camera:
    """A webcam (or, for headless development, a video file) capture source.

    Args:
        index: Webcam device index.  The default ``0`` is the built-in camera on
            most systems.
        width: Requested frame width in pixels.  ``None`` keeps the driver
            default.  Drivers may ignore unsupported values.
        height: Requested frame height in pixels.  ``None`` keeps the driver
            default.
        source: Optional explicit source.  When set, ``index`` is ignored.  An
            ``int`` selects a device index, a ``str``/``Path`` selects a video
            file.  This exists so the full detection pipeline can be exercised
            on machines without a physical webcam.
    """

    def __init__(
        self,
        index: int | str = 0,
        width: int | None = None,
        height: int | None = None,
        source: int | str | Path | None = None,
    ) -> None:
        if source is not None:
            self.source: int | str = self._normalise_source(source)
        elif isinstance(index, (str, Path)) and not str(index).isdigit():
            self.source = self._normalise_source(index)
        else:
            self.source = validate_camera_index(index)

        self.width, self.height = validate_resolution(width, height)

        self._capture: Any | None = None
        self._frames_read: int = 0

    @staticmethod
    def _normalise_source(value: int | str | Path) -> int | str:
        """Reduce any accepted source value to an ``int`` index or a ``str`` path.

        Args:
            value: Device index, path string or :class:`pathlib.Path`.

        Returns:
            An ``int`` for a device index, otherwise a ``str`` path.

        Raises:
            ValueError: If an integer index is outside the accepted range.
        """
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, int):
            return validate_camera_index(value)

        text = str(value).strip()
        if text.lstrip("-").isdigit():
            return validate_camera_index(text)
        return text

    # -- properties -------------------------------------------------------- #

    @property
    def is_file_source(self) -> bool:
        """``True`` when the source is a video file rather than a camera."""
        return isinstance(self.source, str)

    @property
    def is_open(self) -> bool:
        """``True`` when the underlying capture device is open."""
        return self._capture is not None and bool(self._capture.isOpened())

    @property
    def frames_read(self) -> int:
        """Number of frames successfully read since :meth:`open`."""
        return self._frames_read

    @property
    def actual_width(self) -> int:
        """Frame width actually delivered by the source, ``0`` when unknown."""
        return self._read_property("CAP_PROP_FRAME_WIDTH")

    @property
    def actual_height(self) -> int:
        """Frame height actually delivered by the source, ``0`` when unknown."""
        return self._read_property("CAP_PROP_FRAME_HEIGHT")

    @property
    def reported_fps(self) -> float:
        """Frame rate reported by the capture device, ``0.0`` when unknown."""
        return self._read_property("CAP_PROP_FPS")

    @property
    def frame_count_hint(self) -> int:
        """Total frame count for a video file, ``0`` for a live camera."""
        return self._read_property("CAP_PROP_FRAME_COUNT")

    def _read_property(self, name: str) -> float:
        if not self.is_open:
            return 0.0
        module = require_opencv()
        prop = getattr(module, name, None)
        if prop is None:
            return 0.0
        try:
            value = float(self._capture.get(prop))
        except Exception:  # noqa: BLE001 - some backends raise on unsupported props
            return 0.0
        return value if value == value else 0.0  # guard against NaN

    # -- lifecycle --------------------------------------------------------- #

    def open(self) -> "Camera":
        """Open the capture source and apply the requested resolution.

        Returns:
            ``self`` so the call can be chained.

        Raises:
            CameraError: If OpenCV is missing, the source does not exist, or the
                device cannot be opened.
        """
        if self.is_open:
            return self

        module = require_opencv()

        if self.is_file_source:
            path = Path(str(self.source)).expanduser()
            if not path.exists():
                raise CameraError(f"Video source not found: {path}")
            if not is_supported_video_extension(path):
                supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
                raise CameraError(
                    f"Unsupported video extension '{path.suffix}'.\nSupported extensions: {supported}"
                )
            logger.info("Opening video source %s", path)
            with _silence_native_stderr():
                capture = module.VideoCapture(str(path))
        else:
            device_index = int(self.source)
            logger.info(
                "Opening webcam at index %d (backend: %s)", device_index, capture_backend_name()
            )
            with _silence_native_stderr():
                capture = module.VideoCapture(device_index, _resolve_backend())
                if not capture.isOpened():
                    # Retry with the default backend; some drivers reject a forced one.
                    capture.release()
                    capture = module.VideoCapture(device_index)

        if not capture.isOpened():
            capture.release()
            raise CameraError(self._open_error_message())

        if self.width is not None and self.height is not None:
            with _silence_native_stderr():
                capture.set(module.CAP_PROP_FRAME_WIDTH, float(self.width))
                capture.set(module.CAP_PROP_FRAME_HEIGHT, float(self.height))
            logger.debug("Requested resolution %dx%d", self.width, self.height)

        self._capture = capture
        self._frames_read = 0

        width, height = self.actual_width, self.actual_height
        if width and height:
            logger.info("Capture ready at %dx%d (%.1f FPS reported)", width, height, self.reported_fps)
        else:
            logger.info("Capture ready")

        return self

    def _open_error_message(self) -> str:
        """Build a clear, actionable message for a failed open."""
        if self.is_file_source:
            return f"Error: Unable to open the video source '{self.source}'."
        return (
            f"Error: Unable to open webcam at camera index {int(self.source)}.\n"
            "Possible causes:\n"
            "  - No webcam is connected to this machine.\n"
            "  - The camera is already in use by another application.\n"
            "  - The operating system has not granted camera permission to the terminal.\n"
            "  - A different device index is required; try --camera 1."
        )

    def read(self) -> tuple[bool, Any | None]:
        """Read the next frame.

        Returns:
            A ``(ok, frame)`` tuple.  ``ok`` is ``False`` when the stream ended
            or the frame could not be decoded.

        Raises:
            CameraError: If the camera is not open.
        """
        if not self.is_open:
            raise CameraError("Camera is not open. Call Camera.open() first.")

        ok, frame = self._capture.read()
        if ok and frame is not None:
            self._frames_read += 1
            return True, frame
        return False, None

    def release(self) -> None:
        """Release the capture device.  Safe to call more than once."""
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:  # noqa: BLE001 - release must never raise
                logger.debug("Ignoring error while releasing the capture device", exc_info=True)
            finally:
                self._capture = None
                logger.debug("Capture device released after %d frames", self._frames_read)

    def describe(self) -> str:
        """Return a short human readable description of the source."""
        if self.is_file_source:
            return f"video file '{self.source}'"
        return f"camera index {int(self.source)}"

    # -- context manager --------------------------------------------------- #

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Camera(source={self.source!r}, open={self.is_open}, frames_read={self._frames_read})"


def probe_camera(index: int = 0) -> bool:
    """Check whether a camera index yields at least one decodable frame.

    Args:
        index: Device index to probe.

    Returns:
        ``True`` when the device can be opened and a frame can be read.

    Raises:
        CameraError: If OpenCV is unavailable or the index is invalid.
    """
    validate_camera_index(index)
    camera = Camera(index=index)

    try:
        camera.open()
    except CameraError:
        logger.debug("Camera probe failed for index %d", index)
        return False

    try:
        with _silence_native_stderr():
            for _ in range(_PROBE_ATTEMPTS):
                ok, frame = camera.read()
                if ok and frame is not None and getattr(frame, "size", 0) > 0:
                    return True
        return False
    finally:
        camera.release()


def available_camera_indices(limit: int = 5) -> list[int]:
    """Probe the first ``limit`` device indices and return the usable ones.

    Args:
        limit: Number of indices to probe, starting at ``0``.

    Returns:
        A list of indices that produced a decodable frame.
    """
    found: list[int] = []
    for index in range(max(0, limit)):
        if probe_camera(index):
            found.append(index)
    return found
