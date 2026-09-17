"""Webcam capture.

Module 1 of the project: open the webcam, hand frames to the detector and
release the device cleanly when the application exits.

The same class also accepts a video file instead of a camera index. That is not
a project feature, it is simply a convenient way to run the pipeline on a
machine that has no webcam attached.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import cv2

__all__ = ["Camera", "CameraError", "backend_name"]


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or a frame cannot be read."""


def backend_name() -> str:
    """Return the OpenCV capture backend used on this operating system."""
    system = platform.system()
    if system == "Darwin":
        return "AVFoundation"
    if system == "Windows":
        return "DirectShow"
    return "V4L2"


def _backend_flag() -> int:
    """Return the OpenCV backend constant matching this platform."""
    if platform.system() == "Darwin":
        return getattr(cv2, "CAP_AVFOUNDATION", 0)
    if platform.system() == "Windows":
        return getattr(cv2, "CAP_DSHOW", 0)
    return getattr(cv2, "CAP_ANY", 0)


class Camera:
    """A webcam capture source.

    Args:
        index: Webcam device index. ``0`` is the built-in camera on most laptops.
        width: Requested frame width, or ``None`` to keep the driver default.
        height: Requested frame height, or ``None`` to keep the driver default.
        source: Optional path to a video file. When given, ``index`` is ignored.
    """

    def __init__(
        self,
        index: int = 0,
        width: int | None = None,
        height: int | None = None,
        source: str | Path | None = None,
    ) -> None:
        self.source: int | str = str(source) if source is not None else int(index)
        self.width = width
        self.height = height
        self._capture: Any | None = None
        self._frames_read = 0

    # -- state ------------------------------------------------------------- #

    @property
    def is_file_source(self) -> bool:
        """``True`` when frames come from a video file rather than a camera."""
        return isinstance(self.source, str)

    @property
    def is_open(self) -> bool:
        """``True`` while the underlying capture device is open."""
        return self._capture is not None and self._capture.isOpened()

    @property
    def frames_read(self) -> int:
        """Number of frames successfully read since :meth:`open`."""
        return self._frames_read

    def describe(self) -> str:
        """Return a short description of the capture source."""
        if self.is_file_source:
            return f"video file '{self.source}'"
        return f"camera index {self.source}"

    # -- lifecycle --------------------------------------------------------- #

    def open(self) -> "Camera":
        """Open the capture source.

        Returns:
            ``self``, so the call can be chained.

        Raises:
            CameraError: If the device or video file cannot be opened.
        """
        if self.is_open:
            return self

        if self.is_file_source:
            path = Path(str(self.source)).expanduser()
            if not path.exists():
                raise CameraError(f"Error: video file not found: {path}")
            capture = cv2.VideoCapture(str(path))
        else:
            capture = cv2.VideoCapture(self.source, _backend_flag())
            if not capture.isOpened():
                # Some drivers reject an explicitly requested backend, so try
                # again and let OpenCV choose.
                capture.release()
                capture = cv2.VideoCapture(self.source)

        if not capture.isOpened():
            capture.release()
            raise CameraError(self._error_message())

        if self.width and self.height:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))

        self._capture = capture
        self._frames_read = 0
        return self

    def _error_message(self) -> str:
        """Build a helpful message explaining why the source would not open."""
        if self.is_file_source:
            return f"Error: could not open the video file '{self.source}'."

        return (
            f"Error: could not open a webcam at camera index {self.source}.\n"
            "Possible reasons:\n"
            "  - No webcam is connected.\n"
            "  - Another application is already using the camera.\n"
            "  - The operating system has not granted camera permission to the\n"
            "    terminal you are running from (macOS: System Settings > Privacy\n"
            "    & Security > Camera; Windows: Settings > Privacy > Camera).\n"
            "  - The wrong index was used; try --camera 1."
        )

    def read(self) -> tuple[bool, Any | None]:
        """Read the next frame from the source.

        Returns:
            A ``(ok, frame)`` pair. ``ok`` is ``False`` when the stream has
            ended or the frame could not be decoded.

        Raises:
            CameraError: If the camera has not been opened.
        """
        if not self.is_open:
            raise CameraError("Camera is not open. Call open() first.")

        ok, frame = self._capture.read()
        if ok and frame is not None:
            self._frames_read += 1
            return True, frame
        return False, None

    def release(self) -> None:
        """Release the camera. Safe to call more than once."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    # -- context manager --------------------------------------------------- #

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc_info: object) -> None:
        self.release()

    def __repr__(self) -> str:
        return f"Camera(source={self.source!r}, open={self.is_open})"
