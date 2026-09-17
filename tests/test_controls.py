"""Tests for app.controls.

The key codes used here are the same integers ``cv2.waitKey`` returns, so no
window is needed.
"""

import pytest

from app.controls import (
    MAX_CONFIDENCE,
    MIN_CONFIDENCE,
    Action,
    adjust_confidence,
    control_help_lines,
    interpret_key,
)

ESCAPE = 27


def test_q_quits() -> None:
    assert interpret_key(ord("q")) is Action.QUIT


def test_escape_quits() -> None:
    assert interpret_key(ESCAPE) is Action.QUIT


def test_s_saves_a_snapshot() -> None:
    assert interpret_key(ord("s")) is Action.SNAPSHOT


def test_c_clears_the_statistics() -> None:
    assert interpret_key(ord("c")) is Action.RESET_STATS


def test_uppercase_letters_work_too() -> None:
    assert interpret_key(ord("Q")) is Action.QUIT
    assert interpret_key(ord("S")) is Action.SNAPSHOT
    assert interpret_key(ord("C")) is Action.RESET_STATS


def test_plus_raises_the_confidence_threshold() -> None:
    assert interpret_key(ord("+")) is Action.CONFIDENCE_UP
    assert interpret_key(ord("=")) is Action.CONFIDENCE_UP


def test_minus_lowers_the_confidence_threshold() -> None:
    assert interpret_key(ord("-")) is Action.CONFIDENCE_DOWN
    assert interpret_key(ord("_")) is Action.CONFIDENCE_DOWN


def test_an_unmapped_key_does_nothing() -> None:
    assert interpret_key(ord("z")) is Action.NONE


def test_no_key_pressed_does_nothing() -> None:
    assert interpret_key(-1) is Action.NONE


def test_adjust_confidence_moves_in_both_directions() -> None:
    assert adjust_confidence(0.50, 0.05) == pytest.approx(0.55)
    assert adjust_confidence(0.50, -0.05) == pytest.approx(0.45)


def test_adjust_confidence_stays_inside_the_allowed_range() -> None:
    assert adjust_confidence(MAX_CONFIDENCE, 0.5) == MAX_CONFIDENCE
    assert adjust_confidence(MIN_CONFIDENCE, -0.5) == MIN_CONFIDENCE


def test_help_lines_are_formatted_for_display() -> None:
    lines = control_help_lines()

    assert len(lines) == 4
    assert any("Quit" in line for line in lines)
