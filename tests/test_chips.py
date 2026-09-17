"""Chip availability windows.

The regression this guards against: the planner hardcoded "H1 is GW1-19" for all four chips.
In 2026/27 the first Wildcard and Free Hit only open at GW2, so a GW1 recommendation for
either would be unplayable.
"""

import pytest

from src.fpl_predict.strategy.chips import (
    DEFAULT_H1_DEADLINE,
    DEFAULT_H2_START,
    ChipWindows,
    load_chip_windows,
)


def _chip(name, start, stop):
    return {"name": name, "number": 1, "start_event": start, "stop_event": stop}


# The 2026/27 chips array: WC and FH open at GW2, BB and TC at GW1.
BOOTSTRAP_2026 = {
    "chips": [
        _chip("wildcard", 2, 19),
        _chip("wildcard", 20, 38),
        _chip("freehit", 2, 19),
        _chip("freehit", 20, 38),
        _chip("bboost", 1, 19),
        _chip("bboost", 20, 38),
        _chip("3xc", 1, 19),
        _chip("3xc", 20, 38),
    ]
}


@pytest.fixture
def windows():
    return load_chip_windows(BOOTSTRAP_2026)


def test_halves_are_read_from_the_api(windows):
    assert windows.source == "fpl_api"
    assert windows.h1_deadline == 19
    assert windows.h2_start == 20


def test_wildcard_and_free_hit_are_not_available_in_gw1(windows):
    assert windows.earliest("H1", "WC", 1) == 2
    assert windows.earliest("H1", "FH", 1) == 2


def test_bench_boost_and_triple_captain_are_available_in_gw1(windows):
    assert windows.earliest("H1", "BB", 1) == 1
    assert windows.earliest("H1", "TC", 1) == 1


def test_second_set_starts_in_the_second_half(windows):
    for code in ("WC", "FH", "BB", "TC"):
        assert windows.earliest("H2", code, 20) == 20
        assert windows.latest("H2", code, 38) == 38


def test_first_half_chips_expire_at_the_h1_deadline(windows):
    for code in ("WC", "FH", "BB", "TC"):
        assert windows.latest("H1", code, 19) == 19


def test_a_shifted_split_is_picked_up():
    """If FPL moves the halfway point, the planner must follow it."""
    shifted = {
        "chips": [
            _chip("wildcard", 2, 17),
            _chip("wildcard", 18, 38),
            _chip("freehit", 3, 17),
            _chip("bboost", 1, 17),
            _chip("3xc", 1, 17),
        ]
    }
    w = load_chip_windows(shifted)
    assert w.h1_deadline == 17
    assert w.h2_start == 18
    assert w.earliest("H1", "FH", 1) == 3


def test_missing_chips_array_falls_back_to_the_documented_default():
    w = load_chip_windows({})
    assert w.source == "default"
    assert w.h1_deadline == DEFAULT_H1_DEADLINE
    assert w.h2_start == DEFAULT_H2_START


def test_unknown_chip_names_are_ignored():
    w = load_chip_windows({"chips": [_chip("manager", 1, 19), _chip("3xc", 1, 19)]})
    assert ("H1", "TC") in w.first_available
    assert len(w.first_available) == 1


def test_defaults_report_a_permissive_window():
    w = ChipWindows()
    assert w.earliest("H1", "FH", 1) == 1
    assert w.latest("H1", "FH", 19) == 19


def test_chip_start_never_goes_backwards_from_the_current_gameweek(windows):
    from src.fpl_predict.strategy.chips import ChipStrategy

    strategy = ChipStrategy(windows=windows)
    # Already past the chip's opening gameweek: never suggest a gameweek in the past.
    assert strategy._chip_start("H1", "FH", 7) == 7
    # Before it opens: move forward to the opening gameweek.
    assert strategy._chip_start("H1", "FH", 1) == 2
    assert strategy._chip_start("H1", "BB", 1) == 1
