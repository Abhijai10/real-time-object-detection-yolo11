"""Keyboard controls and runtime state (Module 4 - User Controls).

The module is deliberately free of OpenCV so the key handling, confidence
arithmetic and state transitions can be unit tested without a window.  Key codes
are plain integers (``ord("s")``, ``27`` for Escape), which is exactly what
``cv2.waitKey`` returns.

The only action this module cannot service on its own is :attr:`ControlAction.SNAPSHOT`,
because writing an image requires the frame itself.  It is therefore returned to
the caller, which performs the file write.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Final

from app.utils import MAX_CONFIDENCE, MIN_CONFIDENCE, clamp

__all__ = [
    "ControlAction",
    "ControlState",
    "KEY_BINDINGS",
    "CONTROL_HELP",
    "ESCAPE_KEY",
    "interpret_key",
    "adjust_confidence",
    "control_help_lines",
]


class ControlAction(Enum):
    """An action requested by the user through the OpenCV window."""

    NONE = "none"
    QUIT = "quit"
    SNAPSHOT = "snapshot"
    TOGGLE_RECORDING = "toggle_recording"
    RESET_STATS = "reset_stats"
    CONFIDENCE_UP = "confidence_up"
    CONFIDENCE_DOWN = "confidence_down"
    TOGGLE_OVERLAY = "toggle_overlay"


#: ``cv2.waitKey`` returns 27 for the Escape key.
ESCAPE_KEY: Final[int] = 27

#: Default confidence step applied by the ``+`` / ``-`` keys.
DEFAULT_CONFIDENCE_STEP: Final[float] = 0.05


def _build_key_bindings() -> dict[int, ControlAction]:
    """Build the key-code to action mapping.

    Returns:
        A mapping of integer key codes to :class:`ControlAction` values.  Both
        the lower and upper case letter are mapped so the user does not need to
        hold Shift.
    """
    bindings: dict[int, ControlAction] = {ESCAPE_KEY: ControlAction.QUIT}

    letter_actions = {
        "q": ControlAction.QUIT,
        "s": ControlAction.SNAPSHOT,
        "r": ControlAction.TOGGLE_RECORDING,
        "c": ControlAction.RESET_STATS,
        "h": ControlAction.TOGGLE_OVERLAY,
    }
    for letter, action in letter_actions.items():
        bindings[ord(letter)] = action
        bindings[ord(letter.upper())] = action

    for key in ("+", "="):
        bindings[ord(key)] = ControlAction.CONFIDENCE_UP
    for key in ("-", "_"):
        bindings[ord(key)] = ControlAction.CONFIDENCE_DOWN

    return bindings


#: Mapping used by :func:`interpret_key`.
KEY_BINDINGS: Final[dict[int, ControlAction]] = _build_key_bindings()

#: Help rows shown in the overlay and in the README, in display order.
CONTROL_HELP: Final[tuple[tuple[str, str], ...]] = (
    ("Q / ESC", "Quit the application"),
    ("S", "Save a snapshot of the annotated frame"),
    ("R", "Start / stop recording the annotated video"),
    ("C", "Clear the accumulated session statistics"),
    ("+ / -", "Increase / decrease the confidence threshold"),
    ("H", "Show / hide the on-screen help overlay"),
)


def interpret_key(key: int) -> ControlAction:
    """Translate a key code into an application action.

    Args:
        key: Key code as returned by ``cv2.waitKey``.  ``-1`` (no key pressed)
            and any unmapped code map to :attr:`ControlAction.NONE`.

    Returns:
        The requested :class:`ControlAction`.
    """
    if key is None or key < 0:
        return ControlAction.NONE
    return KEY_BINDINGS.get(int(key), ControlAction.NONE)


def adjust_confidence(
    current: float,
    delta: float,
    minimum: float = MIN_CONFIDENCE,
    maximum: float = MAX_CONFIDENCE,
) -> float:
    """Return a new confidence threshold, clamped to the allowed range.

    Args:
        current: Current threshold.
        delta: Amount to add; use a negative value to decrease.
        minimum: Lower bound.
        maximum: Upper bound.

    Returns:
        The clamped threshold.
    """
    return clamp(float(current) + float(delta), float(minimum), float(maximum))


@dataclass
class ControlState:
    """Mutable state driven by the keyboard.

    Attributes:
        confidence: Active confidence threshold.
        step: Amount added or removed by the ``+`` / ``-`` keys.
        minimum: Lowest threshold the user may select.
        maximum: Highest threshold the user may select.
        show_overlay: Whether the analytics overlay is drawn.
        recording: Whether the annotated output is currently being recorded.
    """

    confidence: float = 0.25
    step: float = DEFAULT_CONFIDENCE_STEP
    minimum: float = MIN_CONFIDENCE
    maximum: float = MAX_CONFIDENCE
    show_overlay: bool = True
    recording: bool = False
    _history: list[ControlAction] = field(default_factory=list, repr=False)

    def apply(self, action: ControlAction) -> ControlAction:
        """Apply an action to the state and return it for the caller to service.

        ``SNAPSHOT`` and ``QUIT`` require access to the frame or the loop and are
        therefore only recorded here; the application services them.

        Args:
            action: Action produced by :func:`interpret_key`.

        Returns:
            The same ``action``, unchanged, so callers can use a single
            ``match`` statement over the result.
        """
        if action is ControlAction.NONE:
            return action

        if action is ControlAction.CONFIDENCE_UP:
            self.confidence = adjust_confidence(
                self.confidence, self.step, self.minimum, self.maximum
            )
        elif action is ControlAction.CONFIDENCE_DOWN:
            self.confidence = adjust_confidence(
                self.confidence, -self.step, self.minimum, self.maximum
            )
        elif action is ControlAction.TOGGLE_RECORDING:
            self.recording = not self.recording
        elif action is ControlAction.TOGGLE_OVERLAY:
            self.show_overlay = not self.show_overlay

        self._history.append(action)
        return action

    @property
    def history(self) -> tuple[ControlAction, ...]:
        """Actions applied so far, oldest first (useful for debugging/tests)."""
        return tuple(self._history)

    def reset_history(self) -> None:
        """Forget the recorded action history."""
        self._history.clear()

    @property
    def recording_label(self) -> str:
        """``"ON"`` or ``"OFF"``, ready to be drawn on screen."""
        return "ON" if self.recording else "OFF"


def control_help_lines() -> list[str]:
    """Return the keyboard help as ``"KEY - description"`` strings.

    Returns:
        One line per control, in display order.
    """
    return [f"{key} - {description}" for key, description in CONTROL_HELP]
