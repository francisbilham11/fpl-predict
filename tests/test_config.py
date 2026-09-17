"""Season arithmetic and the ingestion window.

The regression these guard against: `.env` pinned FD_START_SEASON=2023 / FD_END_SEASON=2024,
so at the 2026/27 rollover the window still ended at 2024-25 and the 2025-26 season was never
ingested.
"""

import datetime as dt

import pytest

from src.fpl_predict.config import Settings, current_season, season_label


@pytest.mark.parametrize(
    "day,expected",
    [
        (dt.date(2026, 6, 30), 2025),  # still last season in June
        (dt.date(2026, 7, 1), 2026),  # July flips to the new season
        (dt.date(2026, 8, 21), 2026),
        (dt.date(2027, 1, 15), 2026),  # January still belongs to 2026/27
        (dt.date(2027, 5, 30), 2026),
        (dt.date(2027, 7, 2), 2027),
    ],
)
def test_current_season_flips_in_july(day, expected):
    assert current_season(day) == expected


@pytest.mark.parametrize(
    "year,label",
    [(2026, "2026-27"), (2025, "2025-26"), (1999, "1999-00"), (2099, "2099-00")],
)
def test_season_label(year, label):
    assert season_label(year) == label


def _settings(**kw) -> Settings:
    # _env_file=None so the developer's own .env cannot change the result.
    return Settings(_env_file=None, **kw)


def test_unpinned_window_ends_at_last_completed_season():
    s = _settings(HISTORY_SEASONS=3)
    start, end = s.seasons_window()
    assert end == current_season() - 1
    assert start == end - 2
    assert s.history_seasons() == [start, start + 1, end]


def test_stale_pinned_end_is_carried_forward():
    # The exact pins that were sitting in .env.
    s = _settings(FD_START_SEASON=2023, FD_END_SEASON=2024)
    start, end = s.seasons_window()
    assert start == 2023
    assert end == current_season() - 1, "a stale pin must not drop completed seasons"
    assert current_season() - 1 in s.history_seasons()


def test_pinned_start_is_respected():
    s = _settings(FD_START_SEASON=2021)
    start, _ = s.seasons_window()
    assert start == 2021


def test_history_seasons_controls_span_when_unpinned():
    assert len(_settings(HISTORY_SEASONS=5).history_seasons()) == 5
    assert len(_settings(HISTORY_SEASONS=1).history_seasons()) == 1


def test_span_is_never_zero():
    assert len(_settings(HISTORY_SEASONS=0).history_seasons()) >= 1


def test_window_never_includes_the_season_in_progress():
    for span in (1, 3, 5, 8):
        _, end = _settings(HISTORY_SEASONS=span).seasons_window()
        assert end < current_season()
