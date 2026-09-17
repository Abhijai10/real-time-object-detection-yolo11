"""Basic real-time statistics.

Module 3 of the project (the analytics half).

This module has no dependency on OpenCV, Ultralytics or NumPy, so all of the
counting logic can be tested without a webcam and without the model. Every FPS
value comes from real timestamps; nothing is hard-coded.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Sequence

__all__ = ["FpsCounter", "SessionStats", "count_classes"]


def count_classes(detections: Sequence[Any]) -> dict[str, int]:
    """Count how many detections there are of each class.

    Args:
        detections: Anything with a ``class_name`` attribute.

    Returns:
        A mapping of class name to count, e.g. ``{"person": 2, "chair": 1}``.
    """
    counter: Counter[str] = Counter(detection.class_name for detection in detections)
    return dict(counter)


class FpsCounter:
    """Measure the frame rate of the main loop.

    The counter averages the time between the last few frames and reports the
    reciprocal. The first call only records a starting time, so FPS is ``0.0``
    until at least one interval has been measured.

    Args:
        window: How many recent frame intervals to average over.
    """

    def __init__(self, window: int = 30) -> None:
        self.window = max(1, int(window))
        self._intervals: list[float] = []
        self._last: float | None = None
        self.frames = 0

    def tick(self, now: float) -> float:
        """Record one processed frame and return the current FPS.

        Args:
            now: Current time in seconds, from :func:`time.perf_counter`.

        Returns:
            The measured frames per second, or ``0.0`` if not yet known.
        """
        if self._last is not None:
            elapsed = now - self._last
            if elapsed > 0:
                self._intervals.append(elapsed)
                if len(self._intervals) > self.window:
                    self._intervals.pop(0)

        self._last = now
        self.frames += 1
        return self.fps

    @property
    def fps(self) -> float:
        """The current average frames per second."""
        if not self._intervals:
            return 0.0
        return 1.0 / (sum(self._intervals) / len(self._intervals))

    def reset(self) -> None:
        """Forget all measurements."""
        self._intervals.clear()
        self._last = None
        self.frames = 0


class SessionStats:
    """Totals collected over a whole detection session."""

    def __init__(self) -> None:
        self.started_at = datetime.now()
        self.reset()

    def reset(self) -> None:
        """Clear the accumulated statistics."""
        self.frames = 0
        self.total_detections = 0
        self.peak_objects = 0
        self.class_totals: Counter[str] = Counter()
        self._confidence_sum = 0.0
        self._fps_samples: list[float] = []

    def update(self, detections: Sequence[Any], fps: float) -> None:
        """Add one processed frame to the totals.

        Args:
            detections: The detections found in this frame.
            fps: The measured FPS at the time the frame was processed. A value
                of zero means "not measured yet" and is ignored.
        """
        self.frames += 1
        count = len(detections)

        self.total_detections += count
        self.class_totals.update(detection.class_name for detection in detections)

        if count > self.peak_objects:
            self.peak_objects = count

        for detection in detections:
            self._confidence_sum += float(detection.confidence)

        if fps > 0:
            self._fps_samples.append(float(fps))

    # -- derived values ---------------------------------------------------- #

    @property
    def duration_seconds(self) -> float:
        """How long the session has been running, in seconds."""
        return max(0.0, (datetime.now() - self.started_at).total_seconds())

    @property
    def average_fps(self) -> float:
        """Mean of the measured FPS samples."""
        if not self._fps_samples:
            return 0.0
        return sum(self._fps_samples) / len(self._fps_samples)

    @property
    def peak_fps(self) -> float:
        """Highest measured FPS."""
        return max(self._fps_samples) if self._fps_samples else 0.0

    @property
    def average_confidence(self) -> float:
        """Mean confidence across every detection in the session."""
        if not self.total_detections:
            return 0.0
        return self._confidence_sum / self.total_detections

    @property
    def average_detections_per_frame(self) -> float:
        """Mean number of objects detected per processed frame."""
        if not self.frames:
            return 0.0
        return self.total_detections / self.frames

    def most_frequent_class(self) -> tuple[str, int] | None:
        """Return the most common class and its count, or ``None`` if empty.

        Ties are broken alphabetically so the result is always the same.
        """
        if not self.class_totals:
            return None
        name = min(self.class_totals.items(), key=lambda item: (-item[1], item[0]))[0]
        return name, int(self.class_totals[name])

    # -- output ------------------------------------------------------------ #

    def summary(self) -> list[str]:
        """Return the end-of-session report as a list of text lines."""
        most_common = self.most_frequent_class()
        hours, remainder = divmod(int(self.duration_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)

        return [
            f"Frames processed         : {self.frames}",
            f"Total detections         : {self.total_detections}",
            f"Most objects in one frame: {self.peak_objects}",
            f"Average detections/frame : {self.average_detections_per_frame:.2f}",
            f"Average confidence       : {self.average_confidence:.3f}",
            f"Average FPS              : {self.average_fps:.1f}",
            f"Peak FPS                 : {self.peak_fps:.1f}",
            f"Duration                 : {hours}:{minutes:02d}:{seconds:02d}",
            f"Most frequent class      : {most_common[0] + ' (' + str(most_common[1]) + ')' if most_common else 'none'}",
        ]

    def as_dict(self) -> dict[str, Any]:
        """Return the session statistics as a JSON-serialisable dictionary."""
        most_common = self.most_frequent_class()
        return {
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "duration_seconds": round(self.duration_seconds, 2),
            "frames_processed": self.frames,
            "total_detections": self.total_detections,
            "most_objects_in_one_frame": self.peak_objects,
            "average_detections_per_frame": round(self.average_detections_per_frame, 3),
            "average_confidence": round(self.average_confidence, 4),
            "average_fps": round(self.average_fps, 2),
            "peak_fps": round(self.peak_fps, 2),
            "most_frequent_class": most_common[0] if most_common else None,
            "class_totals": dict(sorted(self.class_totals.items())),
        }
