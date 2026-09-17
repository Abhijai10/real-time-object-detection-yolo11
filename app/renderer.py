"""Drawing of bounding boxes, labels and the analytics overlay (Module 3 - Rendering).

All OpenCV drawing calls live here so that the application loop in
:mod:`app.cli` stays readable and so the overlay layout can be changed in one
place.

The module uses a *stable* colour hash for class names (a CRC-style checksum of
the characters) rather than Python's built-in :func:`hash`, because string
hashing is randomised per process and would otherwise give a different colour to
the same class on every run.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from app.analytics import FrameAnalytics, SessionStats
from app.detector import Detection
from app.utils import get_logger

try:  # pragma: no cover - exercised implicitly whenever rendering happens
    import cv2
except Exception:  # noqa: BLE001 - missing native libraries raise ImportError/OSError
    cv2 = None  # type: ignore[assignment]

__all__ = ["Renderer", "class_color", "DEFAULT_PALETTE"]

logger = get_logger(__name__)

#: Distinct BGR colours assigned to object classes.
DEFAULT_PALETTE: tuple[tuple[int, int, int], ...] = (
    (56, 56, 255),    # red
    (151, 157, 255),  # orange
    (49, 210, 207),   # yellow
    (112, 191, 71),   # green
    (255, 158, 66),   # light blue
    (219, 112, 147),  # purple
    (60, 76, 231),    # deep orange
    (143, 224, 164),  # mint
    (255, 105, 180),  # pink
    (34, 126, 230),   # amber
    (170, 100, 240),  # violet
    (70, 190, 250),   # sky
    (90, 90, 90),     # grey
    (0, 165, 255),    # gold
    (180, 105, 255),  # magenta
    (60, 20, 220),    # crimson
)

_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_PANEL_COLOR = (24, 24, 24)
_RECORD_COLOR = (0, 0, 220)


def _stable_checksum(text: str) -> int:
    """Return a deterministic, process-independent checksum for ``text``.

    Args:
        text: String to checksum.

    Returns:
        A non-negative integer.  Unlike :func:`hash`, this value is identical
        across processes and runs.
    """
    checksum = 0
    for index, character in enumerate(text):
        checksum = (checksum + (index + 1) * ord(character)) % 1_000_003
    return checksum


def class_color(class_name: str, palette: Sequence[tuple[int, int, int]] = DEFAULT_PALETTE) -> tuple[int, int, int]:
    """Return a stable BGR colour for an object class.

    Args:
        class_name: Class label, e.g. ``"person"``.
        palette: Palette to pick from.

    Returns:
        A ``(blue, green, red)`` tuple.
    """
    if not palette:
        return _WHITE
    return palette[_stable_checksum(class_name) % len(palette)]


class Renderer:
    """Draw detections and the analytics overlay onto frames.

    Args:
        palette: Optional custom colour palette.
        font: OpenCV font face used for all text.
        box_thickness: Line thickness of the bounding boxes.
        line_height: Vertical spacing of the overlay text in pixels.
    """

    def __init__(
        self,
        palette: Sequence[tuple[int, int, int]] = DEFAULT_PALETTE,
        box_thickness: int = 2,
        line_height: int = 22,
    ) -> None:
        self.palette = tuple(palette)
        self.box_thickness = int(box_thickness)
        self.line_height = int(line_height)

    # -- helpers ----------------------------------------------------------- #

    @staticmethod
    def _require_cv2() -> Any:
        """Return the OpenCV module or raise a clear error."""
        if cv2 is None:
            raise RuntimeError(
                "OpenCV is not available, so frames cannot be rendered. "
                "Install the dependencies with: pip install -r requirements.txt"
            )
        return cv2

    def color_for(self, class_name: str) -> tuple[int, int, int]:
        """Return the stable colour used for ``class_name``."""
        return class_color(class_name, self.palette)

    @staticmethod
    def _blend_rectangle(
        frame: np.ndarray,
        top_left: tuple[int, int],
        bottom_right: tuple[int, int],
        color: tuple[int, int, int] = _PANEL_COLOR,
        alpha: float = 0.55,
    ) -> None:
        """Draw a translucent filled rectangle in place.

        Args:
            frame: Frame to draw on, modified in place.
            top_left: ``(x, y)`` of the upper-left corner.
            bottom_right: ``(x, y)`` of the lower-right corner.
            color: Fill colour.
            alpha: Opacity of the fill in ``[0, 1]``.
        """
        module = Renderer._require_cv2()
        height, width = frame.shape[:2]
        x1 = max(0, min(int(top_left[0]), width - 1))
        y1 = max(0, min(int(top_left[1]), height - 1))
        x2 = max(0, min(int(bottom_right[0]), width))
        y2 = max(0, min(int(bottom_right[1]), height))
        if x2 <= x1 or y2 <= y1:
            return

        region = frame[y1:y2, x1:x2]
        overlay = np.full_like(region, color, dtype=np.uint8)
        module.addWeighted(overlay, float(alpha), region, 1.0 - float(alpha), 0, region)

    # -- detections -------------------------------------------------------- #

    def draw_detections(self, frame: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
        """Draw bounding boxes and labels for ``detections`` onto ``frame``.

        Args:
            frame: BGR frame, modified in place.
            detections: Detections to draw.

        Returns:
            The same frame, for convenient chaining.
        """
        module = self._require_cv2()
        frame_height, frame_width = frame.shape[:2]

        for detection in detections:
            color = self.color_for(detection.class_name)
            x1, y1, x2, y2 = detection.bbox_int
            x1 = max(0, min(x1, frame_width - 1))
            x2 = max(0, min(x2, frame_width - 1))
            y1 = max(0, min(y1, frame_height - 1))
            y2 = max(0, min(y2, frame_height - 1))

            module.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)

            label = detection.label
            (text_width, text_height), baseline = module.getTextSize(
                label, module.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # Place the label above the box when there is room, otherwise inside it.
            label_top = y1 - text_height - baseline - 4
            if label_top < 0:
                label_top = y1
            label_bottom = label_top + text_height + baseline + 4
            label_right = min(frame_width - 1, x1 + text_width + 8)

            module.rectangle(frame, (x1, label_top), (label_right, label_bottom), color, -1)
            text_y = label_bottom - baseline - 2
            module.putText(
                frame,
                label,
                (x1 + 4, text_y),
                module.FONT_HERSHEY_SIMPLEX,
                0.5,
                _BLACK,
                1,
                module.LINE_AA,
            )

        return frame

    # -- overlay ----------------------------------------------------------- #

    def draw_analytics_overlay(
        self,
        frame: np.ndarray,
        analytics: FrameAnalytics,
        fps: float,
        confidence: float,
        resolution: tuple[int, int] | None = None,
        session: SessionStats | None = None,
        max_classes: int = 6,
    ) -> np.ndarray:
        """Draw the live scene-analytics panel in the top-left corner.

        Args:
            frame: BGR frame, modified in place.
            analytics: Analytics for the current frame.
            fps: Measured frames per second.
            confidence: Active confidence threshold.
            resolution: ``(width, height)`` of the frame.
            session: Optional session statistics used for the summary lines.
            max_classes: Maximum number of per-class lines to draw.

        Returns:
            The same frame, for convenient chaining.
        """
        module = self._require_cv2()

        lines = [
            f"Objects: {analytics.object_count}",
            f"Classes: {', '.join(analytics.unique_classes) if analytics.unique_classes else 'none'}",
            f"FPS: {fps:.1f}",
            f"Confidence: {confidence:.2f}",
        ]

        if resolution:
            lines.append(f"Resolution: {resolution[0]}x{resolution[1]}")
        if analytics.object_count:
            lines.append(f"Avg conf: {analytics.average_confidence:.2f}")

        for class_name, count in list(analytics.class_counts.items())[: max(0, max_classes)]:
            lines.append(f"  - {class_name}: {count}")

        if session is not None:
            lines.append(f"Frames: {session.frames_processed} | Total det: {session.total_detections}")

        self._draw_text_panel(frame, lines, origin=(10, 10))
        return frame

    def draw_help_overlay(self, frame: np.ndarray, lines: Sequence[str]) -> np.ndarray:
        """Draw the keyboard help panel in the bottom-left corner.

        Args:
            frame: BGR frame, modified in place.
            lines: Pre-formatted help lines, e.g. ``"S - Save a snapshot"``.

        Returns:
            The same frame, for convenient chaining.
        """
        module = self._require_cv2()
        if not lines:
            return frame

        height = frame.shape[0]
        text_height = 18
        panel_height = len(lines) * text_height + 12
        top = max(0, height - panel_height - 10)
        bottom = height - 10

        self._blend_rectangle(frame, (10, top), (330, bottom), _PANEL_COLOR, 0.6)

        y = top + text_height
        for line in lines:
            module.putText(
                frame,
                line,
                (18, y),
                module.FONT_HERSHEY_SIMPLEX,
                0.45,
                _WHITE,
                1,
                module.LINE_AA,
            )
            y += text_height

        return frame

    def draw_recording_indicator(self, frame: np.ndarray, recording: bool) -> np.ndarray:
        """Draw the blinking-style REC indicator in the top-right corner.

        Args:
            frame: BGR frame, modified in place.
            recording: Whether recording is currently active.

        Returns:
            The same frame, for convenient chaining.
        """
        if not recording:
            return frame

        module = self._require_cv2()
        width = frame.shape[1]
        label = "REC"
        (text_width, text_height), baseline = module.getTextSize(
            label, module.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        x1 = max(0, width - text_width - 40)
        y1 = 10
        x2 = width - 10
        y2 = y1 + text_height + baseline + 12

        self._blend_rectangle(frame, (x1, y1), (x2, y2), _RECORD_COLOR, 0.7)
        module.circle(frame, (x1 + 14, (y1 + y2) // 2), 6, _WHITE, -1)
        module.putText(
            frame,
            label,
            (x1 + 26, y2 - baseline - 5),
            module.FONT_HERSHEY_SIMPLEX,
            0.6,
            _WHITE,
            2,
            module.LINE_AA,
        )
        return frame

    def draw_message(self, frame: np.ndarray, message: str, duration_hint: bool = True) -> np.ndarray:
        """Draw a transient status message centred near the bottom of the frame.

        Args:
            frame: BGR frame, modified in place.
            message: Message to display.
            duration_hint: Kept for API symmetry; reserved for future use.

        Returns:
            The same frame, for convenient chaining.
        """
        module = self._require_cv2()
        if not message:
            return frame

        height, width = frame.shape[:2]
        (text_width, text_height), baseline = module.getTextSize(
            message, module.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        x1 = max(0, (width - text_width) // 2 - 12)
        y1 = max(0, height - 90)
        x2 = min(width, x1 + text_width + 24)
        y2 = y1 + text_height + baseline + 16

        self._blend_rectangle(frame, (x1, y1), (x2, y2), _PANEL_COLOR, 0.7)
        module.putText(
            frame,
            message,
            (x1 + 12, y2 - baseline - 6),
            module.FONT_HERSHEY_SIMPLEX,
            0.6,
            _WHITE,
            2,
            module.LINE_AA,
        )
        return frame

    def _draw_text_panel(
        self, frame: np.ndarray, lines: Sequence[str], origin: tuple[int, int]
    ) -> None:
        """Draw a translucent panel containing ``lines`` of white text.

        Args:
            frame: BGR frame, modified in place.
            lines: Text lines to draw.
            origin: ``(x, y)`` of the panel's upper-left corner.
        """
        module = self._require_cv2()
        if not lines:
            return

        x, y = origin
        longest = max(len(line) for line in lines)
        panel_width = min(frame.shape[1] - x - 4, int(longest * 8.2) + 24)
        panel_height = len(lines) * self.line_height + 12

        self._blend_rectangle(frame, (x, y), (x + panel_width, y + panel_height), _PANEL_COLOR, 0.55)

        text_y = y + self.line_height - 4
        for line in lines:
            module.putText(
                frame,
                line,
                (x + 10, text_y),
                module.FONT_HERSHEY_SIMPLEX,
                0.55,
                _WHITE,
                1,
                module.LINE_AA,
            )
            text_y += self.line_height

    # -- composite --------------------------------------------------------- #

    def render(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        analytics: FrameAnalytics,
        fps: float,
        confidence: float,
        session: SessionStats | None = None,
        help_lines: Sequence[str] = (),
        show_overlay: bool = True,
        recording: bool = False,
        message: str = "",
    ) -> np.ndarray:
        """Draw every layer for one frame in the correct order.

        Args:
            frame: BGR frame, modified in place.
            detections: Detections to draw.
            analytics: Analytics for the current frame.
            fps: Measured frames per second.
            confidence: Active confidence threshold.
            session: Optional session statistics.
            help_lines: Keyboard help lines, empty to hide the help panel.
            show_overlay: Whether to draw the analytics panel.
            recording: Whether to draw the REC indicator.
            message: Optional transient status message.

        Returns:
            The annotated frame.
        """
        self.draw_detections(frame, detections)

        if show_overlay:
            resolution = (frame.shape[1], frame.shape[0])
            self.draw_analytics_overlay(
                frame, analytics, fps, confidence, resolution, session
            )

        if help_lines:
            self.draw_help_overlay(frame, help_lines)

        self.draw_recording_indicator(frame, recording)

        if message:
            self.draw_message(frame, message)

        return frame
