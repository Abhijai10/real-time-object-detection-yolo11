"""Shared helper utilities for the Real-Time Object Detection application.

This module deliberately contains only small, dependency-light helpers so that
it can be unit tested without a webcam, without OpenCV display support and
without downloading the YOLO11 model.

Contents
--------
* Logging configuration used by every other module.
* Validation helpers for command line values (confidence, camera index, ...).
* Timestamped output-path helpers used for snapshots and recordings.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Final

__all__ = [
    "DEFAULT_LOG_FORMAT",
    "DEFAULT_LOG_DATE_FORMAT",
    "MIN_CONFIDENCE",
    "MAX_CONFIDENCE",
    "MAX_CAMERA_INDEX",
    "SUPPORTED_VIDEO_EXTENSIONS",
    "configure_logging",
    "get_logger",
    "timestamp_slug",
    "ensure_directory",
    "snapshot_path",
    "recording_path",
    "session_summary_path",
    "validate_confidence",
    "validate_camera_index",
    "validate_dimension",
    "validate_resolution",
    "is_supported_video_extension",
    "format_duration",
    "safe_divide",
    "clamp",
    "monotonic_seconds",
    "SnapshotError",
    "RecordingError",
    "save_snapshot",
    "write_json",
    "VideoRecorder",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s"
DEFAULT_LOG_DATE_FORMAT: Final[str] = "%H:%M:%S"

MIN_CONFIDENCE: Final[float] = 0.01
MAX_CONFIDENCE: Final[float] = 1.0

#: Highest camera index we accept.  Most systems expose far fewer devices; the
#: limit exists purely to reject obviously nonsensical values such as ``-3`` or
#: ``999999`` before we hand them to OpenCV.
MAX_CAMERA_INDEX: Final[int] = 32

#: Extensions accepted when a video file is used as the capture source.
SUPPORTED_VIDEO_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".mpg", ".mpeg"}
)

#: Maximum width/height accepted for the camera resolution.
MAX_DIMENSION: Final[int] = 7680

_SLUG_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_\-]+$")


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


def configure_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure the root logger once for the whole application.

    Args:
        verbose: When ``True`` the level is lowered to ``DEBUG``.
        quiet: When ``True`` only warnings and errors are emitted.  ``quiet``
            wins over ``verbose`` if both are supplied.
    """
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    root = logging.getLogger()
    root.setLevel(level)

    # Replace any handler installed by a previous call so repeated invocations
    # inside the same interpreter (e.g. pytest) do not duplicate log lines.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_LOG_DATE_FORMAT))
    root.addHandler(handler)

    # Ultralytics is extremely chatty at INFO level; keep our own output clean.
    logging.getLogger("ultralytics").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)


# --------------------------------------------------------------------------- #
# Output paths
# --------------------------------------------------------------------------- #


def timestamp_slug(moment: datetime | None = None) -> str:
    """Return a filesystem-safe timestamp such as ``2026_09_17_224501``.

    Args:
        moment: Timestamp to format.  Defaults to :func:`datetime.now`.

    Returns:
        The formatted timestamp string.
    """
    moment = datetime.now() if moment is None else moment
    return moment.strftime("%Y_%m_%d_%H%M%S")


def ensure_directory(path: str | Path, create: bool = True) -> Path:
    """Return ``path`` as a :class:`pathlib.Path`, creating it when requested.

    Args:
        path: Directory to normalise.
        create: When ``True`` the directory (and parents) are created.

    Returns:
        The resolved directory path.

    Raises:
        NotADirectoryError: If the path exists but is a file.
        PermissionError: If the directory cannot be created.
    """
    directory = Path(path).expanduser()
    if directory.exists() and not directory.is_dir():
        raise NotADirectoryError(f"Output path exists but is not a directory: {directory}")
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def snapshot_path(output_dir: str | Path, moment: datetime | None = None, extension: str = ".jpg") -> Path:
    """Build a timestamped snapshot path inside ``output_dir``.

    Args:
        output_dir: Directory that will hold the snapshot.
        moment: Optional timestamp used to build the filename.
        extension: Image extension, with or without a leading dot.

    Returns:
        Example: ``outputs/snapshot_2026_09_17_224501.jpg``
    """
    extension = extension if extension.startswith(".") else f".{extension}"
    return Path(output_dir) / f"snapshot_{timestamp_slug(moment)}{extension}"


def recording_path(output_dir: str | Path, moment: datetime | None = None, extension: str = ".mp4") -> Path:
    """Build a timestamped recording path inside ``output_dir``.

    Args:
        output_dir: Directory that will hold the recorded video.
        moment: Optional timestamp used to build the filename.
        extension: Video extension, with or without a leading dot.

    Returns:
        Example: ``outputs/recording_2026_09_17_224501.mp4``
    """
    extension = extension if extension.startswith(".") else f".{extension}"
    return Path(output_dir) / f"recording_{timestamp_slug(moment)}{extension}"


def session_summary_path(output_dir: str | Path) -> Path:
    """Return the path of the session statistics file.

    Args:
        output_dir: Directory that will hold ``session_summary.json``.

    Returns:
        ``<output_dir>/session_summary.json``
    """
    return Path(output_dir) / "session_summary.json"


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate_confidence(value: object) -> float:
    """Validate a confidence threshold.

    Args:
        value: Raw value, typically a string coming from the command line.

    Returns:
        The value as a ``float`` in the closed range ``[0.01, 1.0]``.

    Raises:
        ValueError: If the value is not numeric or falls outside the range.
    """
    try:
        confidence = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Confidence threshold must be a number, received: {value!r}") from exc

    if confidence != confidence:  # NaN check
        raise ValueError("Confidence threshold must be a number, received: NaN")
    if not MIN_CONFIDENCE <= confidence <= MAX_CONFIDENCE:
        raise ValueError(
            f"Confidence threshold must be between {MIN_CONFIDENCE} and {MAX_CONFIDENCE}, "
            f"received: {confidence}"
        )
    return confidence


def validate_camera_index(value: object) -> int:
    """Validate a webcam device index.

    Args:
        value: Raw value, typically a string coming from the command line.

    Returns:
        The index as a non-negative ``int``.

    Raises:
        ValueError: If the value is not an integer or is outside the accepted
            range.
    """
    try:
        index = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Camera index must be an integer, received: {value!r}") from exc

    if index < 0:
        raise ValueError(f"Camera index must be zero or greater, received: {index}")
    if index > MAX_CAMERA_INDEX:
        raise ValueError(
            f"Camera index {index} is unreasonably large (maximum accepted is {MAX_CAMERA_INDEX})"
        )
    return index


def validate_dimension(value: object, name: str = "dimension") -> int:
    """Validate a single frame width or height.

    Args:
        value: Raw value to validate.
        name: Label used in the error message, e.g. ``"width"``.

    Returns:
        The dimension as a positive ``int``.

    Raises:
        ValueError: If the value is not an integer in ``[1, MAX_DIMENSION]``.
    """
    try:
        dimension = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Camera {name} must be an integer, received: {value!r}") from exc

    if dimension <= 0:
        raise ValueError(f"Camera {name} must be greater than zero, received: {dimension}")
    if dimension > MAX_DIMENSION:
        raise ValueError(f"Camera {name} must not exceed {MAX_DIMENSION}, received: {dimension}")
    return dimension


def validate_resolution(width: object | None, height: object | None) -> tuple[int | None, int | None]:
    """Validate an optional ``(width, height)`` pair.

    Both values may be ``None``, in which case the camera default is used.

    Args:
        width: Requested frame width or ``None``.
        height: Requested frame height or ``None``.

    Returns:
        A validated ``(width, height)`` tuple.

    Raises:
        ValueError: If exactly one of the two values is supplied, or if either
            value is invalid.
    """
    if width is None and height is None:
        return None, None
    if (width is None) != (height is None):
        raise ValueError("Camera width and height must be provided together")
    return validate_dimension(width, "width"), validate_dimension(height, "height")


def is_supported_video_extension(path: str | Path) -> bool:
    """Return ``True`` when ``path`` has a known video file extension.

    Args:
        path: File path to inspect.
    """
    return Path(path).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as ``H:MM:SS``.

    Args:
        seconds: Duration in seconds.

    Returns:
        A human readable duration string.
    """
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide two numbers, returning ``default`` when the denominator is zero.

    Args:
        numerator: Dividend.
        denominator: Divisor.
        default: Value returned when the division is undefined.
    """
    if not denominator:
        return default
    return numerator / denominator


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Constrain ``value`` to the closed interval ``[minimum, maximum]``.

    Args:
        value: Value to constrain.
        minimum: Lower bound.
        maximum: Upper bound.
    """
    return max(minimum, min(maximum, value))


def monotonic_seconds() -> float:
    """Return a monotonic clock reading in seconds.

    Wrapped in a helper so that tests can monkeypatch time without touching
    :mod:`time` globally.
    """
    return time.perf_counter()


# --------------------------------------------------------------------------- #
# Output handling: snapshots, recordings, JSON
# --------------------------------------------------------------------------- #


class SnapshotError(RuntimeError):
    """Raised when the current frame cannot be written to disk."""


class RecordingError(RuntimeError):
    """Raised when the annotated video recording cannot be started or written."""


def save_snapshot(
    frame: Any,
    output_dir: str | Path,
    moment: datetime | None = None,
    extension: str = ".jpg",
    jpeg_quality: int = 95,
) -> Path:
    """Write the current annotated frame to a timestamped image file.

    Args:
        frame: BGR frame produced by OpenCV.
        output_dir: Directory that will receive the snapshot.
        moment: Optional timestamp used for the filename.
        extension: Image extension, with or without a leading dot.
        jpeg_quality: JPEG quality in ``[1, 100]`` (ignored for other formats).

    Returns:
        The path of the written file.

    Raises:
        SnapshotError: If OpenCV is unavailable, the directory cannot be
            created, or the write fails.
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        raise SnapshotError("Cannot save a snapshot: the frame is empty")

    try:
        directory = ensure_directory(output_dir)
    except (OSError, NotADirectoryError) as exc:
        raise SnapshotError(f"Cannot create the output directory '{output_dir}': {exc}") from exc

    try:
        import cv2  # noqa: PLC0415 - lazy so this module stays dependency-light
    except ImportError as exc:  # pragma: no cover - OpenCV is a hard dependency
        raise SnapshotError("OpenCV is not installed, so snapshots cannot be written") from exc

    destination = snapshot_path(directory, moment, extension)
    options: list[int] = []
    if extension.lower() in {".jpg", ".jpeg"}:
        options = [int(cv2.IMWRITE_JPEG_QUALITY), int(clamp(jpeg_quality, 1, 100))]

    try:
        written = cv2.imwrite(str(destination), frame, options)
    except Exception as exc:  # noqa: BLE001 - convert any OpenCV failure to a clean error
        raise SnapshotError(f"Failed to write the snapshot '{destination}': {exc}") from exc

    if not written:
        raise SnapshotError(
            f"Failed to write the snapshot '{destination}'. "
            "Check that the directory is writable and that the extension is supported."
        )

    return destination


def write_json(path: str | Path, payload: Any, indent: int = 2) -> Path:
    """Serialise ``payload`` to a JSON file, creating parent directories.

    Args:
        path: Destination file path.
        payload: JSON-serialisable object.
        indent: Indentation used when pretty-printing.

    Returns:
        The path of the written file.

    Raises:
        OSError: If the file cannot be written.
    """
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, ensure_ascii=False, default=str)
        handle.write("\n")
    return destination


class VideoRecorder:
    """Record annotated frames to a video file using OpenCV's ``VideoWriter``.

    Recording is always explicitly started and stopped by the user (the ``R``
    key); nothing is recorded automatically.

    Args:
        output_dir: Directory that will receive the recorded video.
        fps: Frame rate to encode.  The measured capture rate is a good value.
        extension: Container extension, with or without a leading dot.
        fourcc: Preferred four-character codec.
    """

    #: Codec / extension pairs tried in order when the preferred one fails.
    _FALLBACKS: Final[tuple[tuple[str, str], ...]] = (
        ("mp4v", ".mp4"),
        ("avc1", ".mp4"),
        ("MJPG", ".avi"),
        ("XVID", ".avi"),
    )

    def __init__(
        self,
        output_dir: str | Path,
        fps: float = 20.0,
        extension: str = ".mp4",
        fourcc: str = "mp4v",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.fps = float(fps) if fps and fps > 0 else 20.0
        self.extension = extension if extension.startswith(".") else f".{extension}"
        self.fourcc = fourcc

        self._writer: Any | None = None
        self._path: Path | None = None
        self._frame_size: tuple[int, int] | None = None
        self._frames_written: int = 0

    # -- properties -------------------------------------------------------- #

    @property
    def is_recording(self) -> bool:
        """``True`` while a video file is open for writing."""
        return self._writer is not None

    @property
    def path(self) -> Path | None:
        """Path of the file currently being recorded, if any."""
        return self._path

    @property
    def frames_written(self) -> int:
        """Number of frames written to the current recording."""
        return self._frames_written

    # -- lifecycle --------------------------------------------------------- #

    def start(self, frame_size: tuple[int, int], fps: float | None = None) -> Path:
        """Open a new recording.

        Args:
            frame_size: ``(width, height)`` of the frames that will be written.
            fps: Optional override of the encoding frame rate.

        Returns:
            The path of the file being recorded.

        Raises:
            RecordingError: If no codec could be initialised.
        """
        if self.is_recording:
            return self._path  # type: ignore[return-value]

        try:
            import cv2  # noqa: PLC0415 - lazy import, see module note
        except ImportError as exc:  # pragma: no cover
            raise RecordingError("OpenCV is not installed, so recording is unavailable") from exc

        try:
            directory = ensure_directory(self.output_dir)
        except (OSError, NotADirectoryError) as exc:
            raise RecordingError(f"Cannot create the output directory '{self.output_dir}': {exc}") from exc

        width, height = int(frame_size[0]), int(frame_size[1])
        if width <= 0 or height <= 0:
            raise RecordingError(f"Invalid frame size for recording: {frame_size}")

        encoding_fps = float(fps) if fps and fps > 0 else self.fps

        candidates: list[tuple[str, str]] = [(self.fourcc, self.extension)]
        candidates += [pair for pair in self._FALLBACKS if pair != candidates[0]]

        last_error: str = ""
        for codec, extension in candidates:
            destination = recording_path(directory, extension=extension)
            writer = cv2.VideoWriter(
                str(destination),
                cv2.VideoWriter_fourcc(*codec),
                encoding_fps,
                (width, height),
            )
            if writer.isOpened():
                self._writer = writer
                self._path = destination
                self._frame_size = (width, height)
                self._frames_written = 0
                logger = get_logger(__name__)
                logger.info("Recording to %s (%s @ %.1f FPS)", destination, codec, encoding_fps)
                return destination
            writer.release()
            last_error = f"codec '{codec}' with extension '{extension}'"

        raise RecordingError(
            f"Could not start video recording (last attempt: {last_error}).\n"
            "Ensure OpenCV was built with video encoding support, or choose another "
            "output directory with --output."
        )

    def write(self, frame: Any) -> bool:
        """Append a frame to the current recording.

        Args:
            frame: BGR frame to write.  Frames whose size differs from the one
                passed to :meth:`start` are resized, because ``VideoWriter``
                requires a constant frame size.

        Returns:
            ``True`` when a frame was written, ``False`` when not recording or
            when the write failed.
        """
        if not self.is_recording or frame is None:
            return False

        try:
            import cv2  # noqa: PLC0415

            if self._frame_size is not None:
                target_width, target_height = self._frame_size
                if (frame.shape[1], frame.shape[0]) != self._frame_size:
                    frame = cv2.resize(frame, (target_width, target_height))

            self._writer.write(frame)
            self._frames_written += 1
            return True
        except Exception:  # noqa: BLE001 - a dropped frame must not kill the session
            get_logger(__name__).warning("Failed to write a frame to the recording", exc_info=True)
            return False

    def stop(self) -> Path | None:
        """Close the current recording.

        Returns:
            The path of the finished file, or ``None`` if nothing was recorded.
        """
        if not self.is_recording:
            return None

        path = self._path
        frames = self._frames_written
        try:
            self._writer.release()
        except Exception:  # noqa: BLE001 - release must never raise
            get_logger(__name__).debug("Ignoring error while closing the video writer", exc_info=True)
        finally:
            self._writer = None
            self._path = None
            self._frame_size = None

        if frames == 0:
            get_logger(__name__).warning("Recording stopped without writing any frames: %s", path)
        else:
            get_logger(__name__).info("Recording finished: %s (%d frames)", path, frames)

        return path

    def __enter__(self) -> "VideoRecorder":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"VideoRecorder(recording={self.is_recording}, path={self._path!r}, "
            f"frames={self._frames_written})"
        )
