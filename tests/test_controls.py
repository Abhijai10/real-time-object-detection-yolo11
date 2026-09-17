"""Tests for :mod:`app.controls`: key handling, confidence adjustment and state."""

from __future__ import annotations

import pytest

from app.controls import (
    CONTROL_HELP,
    ESCAPE_KEY,
    KEY_BINDINGS,
    ControlAction,
    ControlState,
    adjust_confidence,
    control_help_lines,
    interpret_key,
)


# --------------------------------------------------------------------------- #
# Key interpretation
# --------------------------------------------------------------------------- #


def test_escape_maps_to_quit() -> None:
    assert interpret_key(ESCAPE_KEY) == ControlAction.QUIT


@pytest.mark.parametrize("key", ["q", "Q"])
def test_q_maps_to_quit_in_both_cases(key: str) -> None:
    assert interpret_key(ord(key)) == ControlAction.QUIT


@pytest.mark.parametrize("key", ["s", "S"])
def test_s_maps_to_snapshot(key: str) -> None:
    assert interpret_key(ord(key)) == ControlAction.SNAPSHOT


@pytest.mark.parametrize("key", ["r", "R"])
def test_r_maps_to_recording_toggle(key: str) -> None:
    assert interpret_key(ord(key)) == ControlAction.TOGGLE_RECORDING


@pytest.mark.parametrize("key", ["c", "C"])
def test_c_maps_to_reset(key: str) -> None:
    assert interpret_key(ord(key)) == ControlAction.RESET_STATS


@pytest.mark.parametrize("key", ["h", "H"])
def test_h_maps_to_overlay_toggle(key: str) -> None:
    assert interpret_key(ord(key)) == ControlAction.TOGGLE_OVERLAY


@pytest.mark.parametrize("key", ["+", "="])
def test_plus_keys_map_to_confidence_up(key: str) -> None:
    assert interpret_key(ord(key)) == ControlAction.CONFIDENCE_UP


@pytest.mark.parametrize("key", ["-", "_"])
def test_minus_keys_map_to_confidence_down(key: str) -> None:
    assert interpret_key(ord(key)) == ControlAction.CONFIDENCE_DOWN


@pytest.mark.parametrize("key", [-1, 0, ord("z"), 9999])
def test_unmapped_and_absent_keys_produce_no_action(key: int) -> None:
    assert interpret_key(key) == ControlAction.NONE


def test_every_documented_control_has_a_binding() -> None:
    """The help text and the key map must not drift apart."""
    bound_actions = set(KEY_BINDINGS.values())
    for key, _description in CONTROL_HELP:
        for single_key in key.split("/"):
            single_key = single_key.strip()
            if single_key == "ESC":
                assert interpret_key(ESCAPE_KEY) == ControlAction.QUIT
            elif single_key in {"+", "-"}:
                continue
            else:
                assert interpret_key(ord(single_key)) in bound_actions


# --------------------------------------------------------------------------- #
# Confidence arithmetic
# --------------------------------------------------------------------------- #


def test_adjust_confidence_increases_and_decreases() -> None:
    assert adjust_confidence(0.50, 0.05) == pytest.approx(0.55)
    assert adjust_confidence(0.50, -0.05) == pytest.approx(0.45)


def test_adjust_confidence_clamps_to_the_upper_bound() -> None:
    assert adjust_confidence(0.99, 0.10, maximum=1.0) == pytest.approx(1.0)


def test_adjust_confidence_clamps_to_the_lower_bound() -> None:
    assert adjust_confidence(0.05, -0.50, minimum=0.01) == pytest.approx(0.01)


# --------------------------------------------------------------------------- #
# Control state
# --------------------------------------------------------------------------- #


def test_control_state_starts_from_the_configured_confidence() -> None:
    state = ControlState(confidence=0.40)
    assert state.confidence == pytest.approx(0.40)
    assert state.recording is False
    assert state.show_overlay is True
    assert state.recording_label == "OFF"


def test_control_state_raises_and_lowers_the_threshold() -> None:
    state = ControlState(confidence=0.50, step=0.05)

    state.apply(ControlAction.CONFIDENCE_UP)
    assert state.confidence == pytest.approx(0.55)

    state.apply(ControlAction.CONFIDENCE_DOWN)
    state.apply(ControlAction.CONFIDENCE_DOWN)
    assert state.confidence == pytest.approx(0.45)


def test_control_state_never_leaves_the_valid_confidence_range() -> None:
    state = ControlState(confidence=0.99, step=0.10, maximum=0.95)
    for _ in range(5):
        state.apply(ControlAction.CONFIDENCE_UP)
    assert state.confidence <= 0.95

    for _ in range(50):
        state.apply(ControlAction.CONFIDENCE_DOWN)
    assert state.confidence >= state.minimum


def test_control_state_toggles_recording() -> None:
    state = ControlState()

    state.apply(ControlAction.TOGGLE_RECORDING)
    assert state.recording is True
    assert state.recording_label == "ON"

    state.apply(ControlAction.TOGGLE_RECORDING)
    assert state.recording is False
    assert state.recording_label == "OFF"


def test_control_state_toggles_the_overlay() -> None:
    state = ControlState()
    state.apply(ControlAction.TOGGLE_OVERLAY)
    assert state.show_overlay is False
    state.apply(ControlAction.TOGGLE_OVERLAY)
    assert state.show_overlay is True


def test_control_state_returns_the_action_for_the_caller_to_service() -> None:
    state = ControlState()
    assert state.apply(ControlAction.SNAPSHOT) is ControlAction.SNAPSHOT
    assert state.apply(ControlAction.QUIT) is ControlAction.QUIT


def test_control_state_ignores_none_and_records_history() -> None:
    state = ControlState()

    state.apply(ControlAction.NONE)
    assert state.history == ()

    state.apply(ControlAction.CONFIDENCE_UP)
    state.apply(ControlAction.RESET_STATS)

    assert state.history == (ControlAction.CONFIDENCE_UP, ControlAction.RESET_STATS)

    state.reset_history()
    assert state.history == ()


def test_reset_stats_action_does_not_clear_confidence_or_recording() -> None:
    state = ControlState(confidence=0.60)
    state.apply(ControlAction.TOGGLE_RECORDING)
    state.apply(ControlAction.RESET_STATS)

    assert state.confidence == pytest.approx(0.60)
    assert state.recording is True


# --------------------------------------------------------------------------- #
# Help text
# --------------------------------------------------------------------------- #


def test_control_help_lines_are_formatted_for_the_overlay() -> None:
    lines = control_help_lines()

    assert len(lines) == len(CONTROL_HELP)
    assert any(line.startswith("S -") for line in lines)
    assert all(" - " in line for line in lines)
