"""Real-time scene analytics.

This module is intentionally free of any heavy dependency (no OpenCV, no
Ultralytics, no NumPy) so that all of the statistics logic can be unit tested
on a machine without a webcam and without the YOLO11 weights.

Two responsibilities live here:

* :class:`FpsCounter` -- measures the *measured* runtime frame rate.  The value
  is always derived from real timestamps; nothing is hard-coded.
* :class:`SessionStats` -- accumulates per-frame analytics into session-level
  statistics that are written to ``outputs/session_summary.json`` at exit.
"""

from __future__ import annotations

import statistics
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Protocol, Sequence

from app.utils import format_duration, safe_divide

__all__ = [
    "DetectionLike",
    "FrameAnalytics",
    "FpsCounter",
    "SessionStats",
    "compute_frame_analytics",
]


class DetectionLike(Protocol):
    """Minimal structural type required by the analytics layer.

    The detector returns :class:`app.detector.Detection` objects, but analytics
    only needs these two attributes.  Declaring a protocol keeps the analytics
    module decoupled from the detection module (and therefore from
    Ultralytics).
    """

    class_name: str
    confidence: float


# --------------------------------------------------------------------------- #
# Per-frame analytics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FrameAnalytics:
    """Analytics computed for a single video frame.

    Attributes:
        object_count: Number of detected objects in the frame.
        class_counts: Mapping of class name to the number of instances.
        unique_classes: Sorted tuple of distinct class names.
        average_confidence: Mean confidence of the detections, ``0.0`` when the
            frame contains no detections.
        max_confidence: Highest confidence in the frame, ``0.0`` when empty.
    """

    object_count: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    unique_classes: tuple[str, ...] = ()
    average_confidence: float = 0.0
    max_confidence: float = 0.0

    @property
    def unique_class_count(self) -> int:
        """Number of distinct object classes present in the frame."""
        return len(self.unique_classes)

    @property
    def is_empty(self) -> bool:
        """``True`` when nothing was detected in the frame."""
        return self.object_count == 0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the frame analytics."""
        return {
            "object_count": self.object_count,
            "unique_classes": list(self.unique_classes),
            "unique_class_count": self.unique_class_count,
            "class_counts": dict(self.class_counts),
            "average_confidence": round(self.average_confidence, 4),
            "max_confidence": round(self.max_confidence, 4),
        }


def compute_frame_analytics(detections: Sequence[DetectionLike]) -> FrameAnalytics:
    """Compute :class:`FrameAnalytics` from a sequence of detections.

    Args:
        detections: Detections for a single frame.  Any object exposing
            ``class_name`` and ``confidence`` attributes is accepted.

    Returns:
        The aggregated analytics for the frame.
    """
    if not detections:
        return FrameAnalytics()

    counter: Counter[str] = Counter()
    confidences: list[float] = []

    for detection in detections:
        counter[str(detection.class_name)] += 1
        confidences.append(float(detection.confidence))

    return FrameAnalytics(
        object_count=len(detections),
        class_counts=dict(counter),
        unique_classes=tuple(sorted(counter)),
        average_confidence=sum(confidences) / len(confidences),
        max_confidence=max(confidences),
    )


# --------------------------------------------------------------------------- #
# FPS measurement
# --------------------------------------------------------------------------- #


class FpsCounter:
    """Measure the frame rate of the capture/detection loop.

    The counter keeps a rolling window of inter-frame intervals and reports the
    reciprocal of their mean.  The very first call to :meth:`tick` only records
    the starting instant, so the reported FPS is ``0.0`` until at least one
    interval has been observed.

    Args:
        window: Number of intervals kept in the rolling average.
    """

    def __init__(self, window: int = 30) -> None:
        self.window = max(1, int(window))
        self._intervals: deque[float] = deque(maxlen=self.window)
        self._last_tick: float | None = None
        self._frame_count = 0

    def tick(self, now: float | None = None) -> float:
        """Record a frame and return the current FPS estimate.

        Args:
            now: Optional monotonic timestamp in seconds.  Supplying it makes
                the counter fully deterministic, which is what the unit tests
                rely on.

        Returns:
            The measured frames per second, or ``0.0`` when not yet known.
        """
        from app.utils import monotonic_seconds

        now = monotonic_seconds() if now is None else float(now)

        if self._last_tick is not None:
            delta = now - self._last_tick
            if delta > 0:
                self._intervals.append(delta)

        self._last_tick = now
        self._frame_count += 1
        return self.fps

    @property
    def fps(self) -> float:
        """The current rolling-average frames per second."""
        if not self._intervals:
            return 0.0
        mean_interval = sum(self._intervals) / len(self._intervals)
        return safe_divide(1.0, mean_interval, 0.0)

    @property
    def frame_count(self) -> int:
        """Number of frames recorded through :meth:`tick`."""
        return self._frame_count

    def reset(self) -> None:
        """Forget all measured intervals and the frame counter."""
        self._intervals.clear()
        self._last_tick = None
        self._frame_count = 0


# --------------------------------------------------------------------------- #
# Session statistics
# --------------------------------------------------------------------------- #


class SessionStats:
    """Accumulate statistics across an entire detection session.

    The class is deliberately small: it tracks exactly the quantities reported
    in ``outputs/session_summary.json`` and nothing else.

    Args:
        started_at: Optional session start timestamp; defaults to *now*.
    """

    def __init__(self, started_at: datetime | None = None) -> None:
        self.started_at: datetime = started_at or datetime.now()
        self._fps_samples: list[float] = []
        self.reset(keep_start_time=True)

    # -- mutation ---------------------------------------------------------- #

    def reset(self, keep_start_time: bool = True) -> None:
        """Clear all accumulated statistics.

        Args:
            keep_start_time: When ``False`` the session start timestamp is also
                reset to *now*.
        """
        if not keep_start_time:
            self.started_at = datetime.now()

        self.frames_processed: int = 0
        self.frames_with_detections: int = 0
        self.total_detections: int = 0
        self.peak_simultaneous_detections: int = 0
        self.class_totals: Counter[str] = Counter()
        self._confidence_sum: float = 0.0
        self._confidence_count: int = 0
        self._fps_samples = []

    def update(self, analytics: FrameAnalytics, fps: float | None = None) -> None:
        """Fold one frame of analytics into the session totals.

        Args:
            analytics: Analytics computed for the frame.
            fps: Measured FPS at the time the frame was processed.  Values that
                are zero (not yet measured) are ignored so they do not drag the
                session average down.
        """
        self.frames_processed += 1
        self.total_detections += analytics.object_count
        self.class_totals.update(analytics.class_counts)

        if analytics.object_count > 0:
            self.frames_with_detections += 1
            self._confidence_sum += analytics.average_confidence * analytics.object_count
            self._confidence_count += analytics.object_count

        if analytics.object_count > self.peak_simultaneous_detections:
            self.peak_simultaneous_detections = analytics.object_count

        if fps and fps > 0:
            self._fps_samples.append(float(fps))

    # -- derived values ---------------------------------------------------- #

    @property
    def duration_seconds(self) -> float:
        """Wall-clock duration of the session in seconds."""
        return max(0.0, (datetime.now() - self.started_at).total_seconds())

    @property
    def average_fps(self) -> float:
        """Mean of all measured non-zero FPS samples."""
        if not self._fps_samples:
            return 0.0
        return statistics.fmean(self._fps_samples)

    @property
    def peak_fps(self) -> float:
        """Highest measured FPS sample."""
        return max(self._fps_samples) if self._fps_samples else 0.0

    @property
    def average_detections_per_frame(self) -> float:
        """Mean number of detections per processed frame."""
        return safe_divide(self.total_detections, self.frames_processed, 0.0)

    @property
    def average_confidence(self) -> float:
        """Mean confidence across every detection observed in the session."""
        return safe_divide(self._confidence_sum, self._confidence_count, 0.0)

    @property
    def detection_rate(self) -> float:
        """Fraction of frames that contained at least one detection."""
        return safe_divide(self.frames_with_detections, self.frames_processed, 0.0)

    def most_frequent_class(self) -> tuple[str, int] | None:
        """Return the most frequently detected class and its total count.

        Ties are broken alphabetically so the result is deterministic.

        Returns:
            ``(class_name, count)`` or ``None`` when nothing was detected.
        """
        if not self.class_totals:
            return None
        name = min(self.class_totals.items(), key=lambda item: (-item[1], item[0]))[0]
        return name, int(self.class_totals[name])

    def top_classes(self, limit: int = 5) -> list[tuple[str, int]]:
        """Return the ``limit`` most frequent classes, most frequent first.

        Args:
            limit: Maximum number of entries to return.
        """
        ordered = sorted(self.class_totals.items(), key=lambda item: (-item[1], item[0]))
        return [(name, int(count)) for name, count in ordered[: max(0, limit)]]

    # -- serialisation ----------------------------------------------------- #

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of the whole session."""
        most_frequent = self.most_frequent_class()
        return {
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "duration_seconds": round(self.duration_seconds, 3),
            "duration_human": format_duration(self.duration_seconds),
            "frames_processed": self.frames_processed,
            "frames_with_detections": self.frames_with_detections,
            "total_detections": self.total_detections,
            "peak_simultaneous_detections": self.peak_simultaneous_detections,
            "average_detections_per_frame": round(self.average_detections_per_frame, 4),
            "detection_rate": round(self.detection_rate, 4),
            "average_confidence": round(self.average_confidence, 4),
            "average_fps": round(self.average_fps, 2),
            "peak_fps": round(self.peak_fps, 2),
            "most_frequent_class": most_frequent[0] if most_frequent else None,
            "most_frequent_class_count": most_frequent[1] if most_frequent else 0,
            "class_totals": dict(sorted(self.class_totals.items())),
        }

    def summary_lines(self, limit: int = 5) -> list[str]:
        """Return a short human readable summary suitable for the terminal.

        Args:
            limit: Maximum number of per-class lines to include.
        """
        most_frequent = self.most_frequent_class()
        lines = [
            f"Frames processed        : {self.frames_processed}",
            f"Total detections        : {self.total_detections}",
            f"Peak simultaneous       : {self.peak_simultaneous_detections}",
            f"Average detections/frame: {self.average_detections_per_frame:.2f}",
            f"Average confidence      : {self.average_confidence:.3f}",
            f"Average FPS             : {self.average_fps:.1f}",
            f"Peak FPS                : {self.peak_fps:.1f}",
            f"Duration                : {format_duration(self.duration_seconds)}",
            f"Most frequent class     : {most_frequent[0] + ' (' + str(most_frequent[1]) + ')' if most_frequent else 'none'}",
        ]

        top = self.top_classes(limit)
        if top:
            rendered = ", ".join(f"{name} x{count}" for name, count in top)
            lines.append(f"Top classes             : {rendered}")

        return lines


def merge_analytics(frames: Iterable[FrameAnalytics]) -> SessionStats:
    """Build :class:`SessionStats` from an iterable of frame analytics.

    Convenience helper used by the tests and by any offline analysis of a
    recorded session.

    Args:
        frames: Iterable of :class:`FrameAnalytics` objects.

    Returns:
        The populated :class:`SessionStats` instance.
    """
    stats = SessionStats()
    for frame in frames:
        stats.update(frame)
    return stats
