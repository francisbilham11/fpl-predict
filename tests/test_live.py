"""Featurising an upcoming gameweek.

The risk this guards against is train/serve skew: the live path must produce features that
mean exactly what they meant in training. Three scale mismatches between the archive and the
live API would each pass as a plausible number if got wrong — ownership is a headcount in one
and a percentage in the other, and player ids are renumbered every season.
"""

import pandas as pd
import pytest

from src.fpl_predict.models import live as live_mod
from src.fpl_predict.models.live import (
    next_gameweek,
    upcoming_fixtures,
    upcoming_rows,
)

BOOTSTRAP = {
    "total_players": 1_000_000,
    "events": [
        {"id": 1, "finished": True, "is_next": False},
        {"id": 2, "finished": False, "is_next": True},
        {"id": 3, "finished": False, "is_next": False},
    ],
    "teams": [
        {"id": 1, "name": "Arsenal", "short_name": "ARS"},
        {"id": 2, "name": "Chelsea", "short_name": "CHE"},
        {"id": 3, "name": "Man City", "short_name": "MCI"},
        {"id": 4, "name": "Everton", "short_name": "EVE"},
    ],
    "elements": [
        {
            "id": 10,
            "code": 111,
            "web_name": "Saka",
            "element_type": 3,
            "team": 1,
            "now_cost": 100,
            "selected_by_percent": "25.0",
            "ep_next": "5.0",
        },
        {
            "id": 11,
            "code": 222,
            "web_name": "Palmer",
            "element_type": 3,
            "team": 2,
            "now_cost": 95,
            "selected_by_percent": "10.5",
            "ep_next": "4.0",
        },
        {
            "id": 12,
            "code": 333,
            "web_name": "Haaland",
            "element_type": 4,
            "team": 3,
            "now_cost": 155,
            "selected_by_percent": "70.0",
            "ep_next": "6.0",
        },
        {
            "id": 13,
            "code": 444,
            "web_name": "Pickford",
            "element_type": 1,
            "team": 4,
            "now_cost": 55,
            "selected_by_percent": "8.0",
            "ep_next": "3.0",
        },
    ],
}

# Arsenal v Chelsea in GW2; Man City play twice; Everton have a blank.
FIXTURES = [
    {"id": 100, "event": 2, "team_h": 1, "team_a": 2, "kickoff_time": "2026-08-28T14:00:00Z"},
    {"id": 101, "event": 2, "team_h": 3, "team_a": 1, "kickoff_time": "2026-08-29T14:00:00Z"},
    {"id": 102, "event": 2, "team_h": 2, "team_a": 3, "kickoff_time": "2026-08-30T14:00:00Z"},
    {"id": 103, "event": 3, "team_h": 4, "team_a": 1, "kickoff_time": "2026-09-05T14:00:00Z"},
]


@pytest.fixture(autouse=True)
def _stub_api(monkeypatch):
    monkeypatch.setattr(live_mod, "get_fixtures", lambda: FIXTURES)
    monkeypatch.setattr(live_mod, "current_season", lambda: 2026)


def test_next_gameweek_prefers_the_flagged_one():
    assert next_gameweek(BOOTSTRAP) == 2


def test_next_gameweek_falls_back_to_the_first_unfinished():
    bs = {"events": [{"id": 1, "finished": True}, {"id": 2, "finished": False}]}
    assert next_gameweek(bs) == 2


def test_next_gameweek_is_none_when_the_season_is_over():
    assert next_gameweek({"events": [{"id": 1, "finished": True}]}) is None


def test_upcoming_fixtures_has_two_rows_per_match():
    fx = upcoming_fixtures(2, BOOTSTRAP)
    assert len(fx) == 6  # 3 matches
    assert fx.was_home.sum() == 3


def test_upcoming_fixtures_resolve_to_canonical_slugs():
    fx = upcoming_fixtures(2, BOOTSTRAP)
    assert set(fx.team_slug) == {"arsenal", "chelsea", "man_city"}


def test_a_blank_gameweek_means_no_row():
    """Everton have no GW2 fixture, so Pickford must not appear at all."""
    rows = upcoming_rows(2, BOOTSTRAP)
    assert "Pickford" not in set(rows.player_name)
    assert "everton" not in set(rows.team_slug)


def test_a_double_gameweek_is_one_row_with_a_fixture_count_of_two():
    rows = upcoming_rows(2, BOOTSTRAP)
    haaland = rows[rows.player_name == "Haaland"]
    assert len(haaland) == 1
    assert int(haaland.n_fixtures.iloc[0]) == 2


def test_single_fixtures_get_a_count_of_one():
    rows = upcoming_rows(2, BOOTSTRAP)
    assert (
        int(rows[rows.player_name == "Palmer"].n_fixtures.iloc[0]) == 2
    )  # Chelsea also play twice
    assert int(rows[rows.player_name == "Saka"].n_fixtures.iloc[0]) == 2  # Arsenal too


def test_ownership_is_converted_from_percentage_to_headcount():
    """The archive stores a count of managers; the API gives a percentage."""
    rows = upcoming_rows(2, BOOTSTRAP)
    saka = rows[rows.player_name == "Saka"].iloc[0]
    assert saka.selected == pytest.approx(0.25 * 1_000_000)


def test_price_is_carried_through_unchanged():
    rows = upcoming_rows(2, BOOTSTRAP)
    assert rows[rows.player_name == "Haaland"].value.iloc[0] == 155


def test_rows_are_keyed_on_the_stable_player_code():
    rows = upcoming_rows(2, BOOTSTRAP)
    assert set(rows.player_code) == {111, 222, 333}
    assert set(rows.fpl_id) == {10, 11, 12}


def test_positions_are_mapped_to_short_names():
    rows = upcoming_rows(2, BOOTSTRAP)
    assert set(rows.position) <= {"GKP", "DEF", "MID", "FWD"}
    assert rows[rows.player_name == "Haaland"].position.iloc[0] == "FWD"


def test_no_outcome_is_asserted_for_the_upcoming_gameweek():
    rows = upcoming_rows(2, BOOTSTRAP)
    assert rows["points"].isna().all()
    assert (rows["minutes"] == 0).all()


def test_home_and_away_are_set_per_player():
    rows = upcoming_rows(2, BOOTSTRAP)
    # Arsenal's first GW2 fixture is at home to Chelsea.
    assert bool(rows[rows.player_name == "Saka"].was_home.iloc[0]) is True


def test_no_fixtures_at_all_returns_empty():
    assert upcoming_rows(38, BOOTSTRAP).empty


# --------------------------------------------------------------- full live panel


def _history() -> pd.DataFrame:
    """Two prior seasons of played gameweeks for the same players."""
    rows = []
    for season in (2024, 2025):
        for gw in range(1, 6):
            for code, name, pos, slug in [
                (111, "Saka", "MID", "arsenal"),
                (222, "Palmer", "MID", "chelsea"),
                (333, "Haaland", "FWD", "man_city"),
            ]:
                rows.append(
                    {
                        "season": season,
                        "gw": gw,
                        "player_code": code,
                        "fpl_id": code % 100,
                        "player_name": name,
                        "position": pos,
                        "team_slug": slug,
                        "opponent_slug": "everton",
                        "was_home": gw % 2 == 0,
                        "value": 100,
                        "selected": 200000,
                        "kickoff_time": pd.Timestamp(f"{season}-09-01", tz="UTC"),
                        "fixture_id": gw,
                        "minutes": 90,
                        "total_points": 5.0,
                        "goals": 1.0,
                        "assists": 0.0,
                        "clean_sheets": 0.0,
                        "goals_conceded": 1.0,
                        "saves": 0.0,
                        "bonus": 1.0,
                        "bps": 25.0,
                        "yellow_cards": 0.0,
                        "red_cards": 0.0,
                        "own_goals": 0.0,
                        "penalties_saved": 0.0,
                        "penalties_missed": 0.0,
                        "xg": 0.5,
                        "xa": 0.2,
                        "xgc": 1.0,
                        "defensive_contribution": 3.0,
                        "cbi": 1.0,
                        "tackles": 1.0,
                        "recoveries": 1.0,
                        "starts": 1.0,
                        "fpl_xp": 5.0,
                    }
                )
    return pd.DataFrame(rows)


def test_live_panel_separates_training_rows_from_rows_to_score(monkeypatch):
    monkeypatch.setattr(live_mod, "load_history", lambda seasons=None: _history())
    monkeypatch.setattr(live_mod, "load_history_fixtures", lambda seasons=None: pd.DataFrame())
    monkeypatch.setattr(live_mod, "current_season_rows", lambda bootstrap=None: pd.DataFrame())

    train, score = live_mod.build_live_panel(2, BOOTSTRAP)

    assert len(score) == 3
    assert set(score.gw) == {2}
    assert set(score.season) == {2026}
    # Training rows must never include the gameweek being scored.
    assert not ((train.season == 2026) & (train.gw == 2)).any()
    assert train["points"].notna().all()


def test_live_panel_carries_lagged_form_from_previous_seasons(monkeypatch):
    monkeypatch.setattr(live_mod, "load_history", lambda seasons=None: _history())
    monkeypatch.setattr(live_mod, "load_history_fixtures", lambda seasons=None: pd.DataFrame())
    monkeypatch.setattr(live_mod, "current_season_rows", lambda bootstrap=None: pd.DataFrame())

    _, score = live_mod.build_live_panel(2, BOOTSTRAP)
    saka = score[score.player_name == "Saka"].iloc[0]
    # Every historical gameweek scored 5, so the lagged mean must be 5.
    assert saka.points_l5 == pytest.approx(5.0)
    assert saka.mins_l5 == pytest.approx(90.0)
    assert saka.prev_minutes == pytest.approx(5 * 90)


def test_live_panel_raises_when_nobody_has_a_fixture(monkeypatch):
    monkeypatch.setattr(live_mod, "load_history", lambda seasons=None: _history())
    monkeypatch.setattr(live_mod, "load_history_fixtures", lambda seasons=None: pd.DataFrame())
    monkeypatch.setattr(live_mod, "current_season_rows", lambda bootstrap=None: pd.DataFrame())

    with pytest.raises(ValueError, match="No players have a fixture"):
        live_mod.build_live_panel(38, BOOTSTRAP)
