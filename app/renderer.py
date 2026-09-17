"""Drawing the detection results and the on-screen statistics.

Module 3 of the project (the visualisation half).

All OpenCV drawing lives here so that the main loop stays readable. The colours
are chosen from a fixed palette using a simple checksum of the class name, so a
given class always gets the same colour instead of changing between runs.
"""

from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np

from app.detector import Detection

__all__ = ["Renderer", "class_color", "PALETTE"]

#: Colours assigned to object classes, in BGR order (OpenCV's convention).
PALETTE: tuple[tuple[int, int, int], ...] = (
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
)

_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_PANEL = (28, 28, 28)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def class_color(class_name: str) -> tuple[int, int, int]:
    """Return the drawing colour for a class name.

    The checksum is calculated here rather than using Python's built-in
    ``hash()``, because string hashing is randomised per process and the colour
    would change on every run.
    """
    checksum = 0
    for index, character in enumerate(class_name):
        checksum = (checksum + (index + 1) * ord(character)) % 1_000_003
    return PALETTE[checksum % len(PALETTE)]


def _panel(frame: np.ndarray, top_left: tuple[int, int], bottom_right: tuple[int, int]) -> None:
    """Draw a semi-transparent dark rectangle, used behind text."""
    height, width = frame.shape[:2]
    x1 = max(0, min(int(top_left[0]), width - 1))
    y1 = max(0, min(int(top_left[1]), height - 1))
    x2 = max(0, min(int(bottom_right[0]), width))
    y2 = max(0, min(int(bottom_right[1]), height))
    if x2 <= x1 or y2 <= y1:
        return

    region = frame[y1:y2, x1:x2]
    overlay = np.full_like(region, _PANEL)
    cv2.addWeighted(overlay, 0.55, region, 0.45, 0, region)


class Renderer:
    """Draws bounding boxes, labels and the statistics overlay onto frames."""

    def __init__(self, box_thickness: int = 2) -> None:
        self.box_thickness = int(box_thickness)

    # -- detections -------------------------------------------------------- #

    def draw_detections(self, frame: np.ndarray, detections: Sequence[Detection]) -> None:
        """Draw a box and a label for every detection. Modifies ``frame``."""
        height, width = frame.shape[:2]

        for detection in detections:
            colour = class_color(detection.class_name)
            x1, y1, x2, y2 = (int(round(value)) for value in detection.box)

            # Keep the box inside the frame even if the model predicts outside it.
            x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
            y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))

            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, self.box_thickness)

            label = detection.label
            (text_width, text_height), baseline = cv2.getTextSize(label, _FONT, 0.5, 1)

            # Put the label above the box, unless that would fall off the top.
            label_top = y1 - text_height - baseline - 4
            if label_top < 0:
                label_top = y1
            label_bottom = label_top + text_height + baseline + 4
            label_right = min(width - 1, x1 + text_width + 8)

            cv2.rectangle(frame, (x1, label_top), (label_right, label_bottom), colour, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 4, label_bottom - baseline - 2),
                _FONT,
                0.5,
                _BLACK,
                1,
                cv2.LINE_AA,
            )

    # -- overlays ---------------------------------------------------------- #

    def draw_stats(
        self,
        frame: np.ndarray,
        object_count: int,
        class_counts: dict[str, int],
        fps: float,
        confidence: float,
        session_frames: int = 0,
        session_total: int = 0,
    ) -> None:
        """Draw the live statistics panel in the top-left corner."""
        lines = [
            f"Objects: {object_count}",
            f"Classes: {', '.join(class_counts) if class_counts else 'none'}",
            f"FPS: {fps:.1f}",
            f"Confidence: {confidence:.2f}",
        ]
        for name, count in class_counts.items():
            lines.append(f"  {name}: {count}")
        if session_frames:
            lines.append(f"Frames: {session_frames}  Detections: {session_total}")

        self._draw_lines(frame, lines, origin=(10, 10), font_scale=0.55, line_height=22)

    def draw_help(self, frame: np.ndarray, lines: Sequence[str]) -> None:
        """Draw the keyboard controls in the bottom-left corner."""
        if not lines:
            return
        height = frame.shape[0]
        line_height = 18
        panel_height = len(lines) * line_height + 12
        top = max(0, height - panel_height - 10)

        _panel(frame, (10, top), (280, height - 10))

        y = top + line_height
        for line in lines:
            cv2.putText(frame, line, (18, y), _FONT, 0.45, _WHITE, 1, cv2.LINE_AA)
            y += line_height

    def _draw_lines(
        self,
        frame: np.ndarray,
        lines: Sequence[str],
        origin: tuple[int, int],
        font_scale: float,
        line_height: int,
    ) -> None:
        """Draw a translucent panel containing white text."""
        x, y = origin
        widest = max(len(line) for line in lines)
        panel_width = min(frame.shape[1] - x - 4, int(widest * 8.2) + 24)
        panel_height = len(lines) * line_height + 12

        _panel(frame, (x, y), (x + panel_width, y + panel_height))

        text_y = y + line_height - 4
        for line in lines:
            cv2.putText(frame, line, (x + 10, text_y), _FONT, font_scale, _WHITE, 1, cv2.LINE_AA)
            text_y += line_height
