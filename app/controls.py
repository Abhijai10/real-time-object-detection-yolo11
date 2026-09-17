"""Keyboard controls.

Module 4 of the project.

This module contains no OpenCV code. It only translates the integer that
``cv2.waitKey`` returns into a named action, and clamps the confidence
threshold. Keeping it separate means the key handling can be tested without
opening a window.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "Action",
    "interpret_key",
    "adjust_confidence",
    "CONTROL_HELP",
    "CONFIDENCE_STEP",
    "MIN_CONFIDENCE",
    "MAX_CONFIDENCE",
]

#: Confidence threshold limits the user can move between with ``+`` / ``-``.
MIN_CONFIDENCE = 0.05
MAX_CONFIDENCE = 0.95

#: How much one press of ``+`` or ``-`` changes the threshold.
CONFIDENCE_STEP = 0.05


class Action(Enum):
    """Something the user asked for by pressing a key."""

    NONE = "none"
    QUIT = "quit"
    SNAPSHOT = "snapshot"
    RESET_STATS = "reset_stats"
    CONFIDENCE_UP = "confidence_up"
    CONFIDENCE_DOWN = "confidence_down"


#: ``cv2.waitKey`` returns 27 for the Escape key.
_ESCAPE = 27


def _build_bindings() -> dict[int, Action]:
    """Build the key code to action mapping.

    Both the lower and upper case letter are mapped so the user does not have to
    hold Shift.
    """
    bindings = {_ESCAPE: Action.QUIT}
    letters = {
        "q": Action.QUIT,
        "s": Action.SNAPSHOT,
        "c": Action.RESET_STATS,
    }
    for letter, action in letters.items():
        bindings[ord(letter)] = action
        bindings[ord(letter.upper())] = action

    for key in ("+", "="):
        bindings[ord(key)] = Action.CONFIDENCE_UP
    for key in ("-", "_"):
        bindings[ord(key)] = Action.CONFIDENCE_DOWN
    return bindings


#: Key code to action, used by :func:`interpret_key`.
KEY_BINDINGS = _build_bindings()

#: The controls shown in the window overlay and in the README.
CONTROL_HELP: tuple[tuple[str, str], ...] = (
    ("Q / ESC", "Quit"),
    ("S", "Save a snapshot"),
    ("C", "Clear statistics"),
    ("+ / -", "Change confidence threshold"),
)


def interpret_key(key: int) -> Action:
    """Translate a key code from ``cv2.waitKey`` into an :class:`Action`.

    Args:
        key: Key code. ``-1`` means no key was pressed.

    Returns:
        The matching action, or :attr:`Action.NONE`.
    """
    if key is None or key < 0:
        return Action.NONE
    return KEY_BINDINGS.get(int(key), Action.NONE)


def adjust_confidence(current: float, delta: float) -> float:
    """Return a new confidence threshold, kept inside the allowed range.

    Args:
        current: The threshold in use.
        delta: Amount to add. Use a negative value to lower the threshold.

    Returns:
        The new threshold, clamped to ``[MIN_CONFIDENCE, MAX_CONFIDENCE]``.
    """
    return max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, float(current) + float(delta)))


def control_help_lines() -> list[str]:
    """Return the controls as ``"KEY - description"`` strings."""
    return [f"{key} - {description}" for key, description in CONTROL_HELP]
