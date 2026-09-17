"""Elo ratings.

Regressions these guard against:

- Ratings were keyed on whatever spelling the source used, so three seasons of
  football-data results updated keys ("manchestercity") that nobody ever looked up, and the
  FPL-named future fixtures all resolved to the 1500 default. Every upcoming fixture came out
  at identical difficulty.
- Ratings carried across the summer at full strength and every unseen club entered at the
  league mean, so a promoted side looked mid-table in GW1.
"""

import numpy as np
import pandas as pd
import pytest

from src.fpl_predict.fixtures.elo import (
    HOME_ADV,
    MEAN_RATING,
    expected_score,
    match_counts,
    update_elo,
)


def _season(season: int, results: list[tuple[str, str, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": season,
            "date": pd.date_range(f"{season}-08-15", periods=len(results), freq="7D", tz="UTC"),
            "home_team": [r[0] for r in results],
            "away_team": [r[1] for r in results],
            "home_goals": [r[2] for r in results],
            "away_goals": [r[3] for r in results],
        }
    )


def test_expected_score_is_symmetric_and_home_advantaged():
    assert expected_score(1500, 1500, home=False) == pytest.approx(0.5)
    assert expected_score(1500, 1500, home=True) > 0.5
    assert expected_score(1500, 1500 + HOME_ADV, home=True) == pytest.approx(0.5)


def test_winning_raises_a_rating_and_losing_lowers_it():
    df = _season(2025, [("a", "b", 3, 0)])
    r = update_elo(df)
    assert r["a"] > MEAN_RATING
    assert r["b"] < MEAN_RATING
    # Elo is zero sum.
    assert (r["a"] - MEAN_RATING) == pytest.approx(-(r["b"] - MEAN_RATING))


def test_bigger_margins_move_ratings_further():
    narrow = update_elo(_season(2025, [("a", "b", 1, 0)]))
    thumping = update_elo(_season(2025, [("a", "b", 5, 0)]))
    assert thumping["a"] > narrow["a"]


def test_a_draw_between_equals_barely_moves_anything():
    r = update_elo(_season(2025, [("a", "b", 1, 1)]))
    # The home side was favoured, so a draw costs it a little.
    assert r["a"] < MEAN_RATING
    assert abs(r["a"] - MEAN_RATING) < 10


def test_ratings_regress_toward_the_mean_across_a_season_boundary():
    one_season = _season(2024, [("a", "b", 4, 0)] * 5)
    both = pd.concat([one_season, _season(2025, [("c", "d", 0, 0)])], ignore_index=True)

    end_of_2024 = update_elo(one_season)["a"]
    after_summer = update_elo(both)["a"]

    assert MEAN_RATING < after_summer < end_of_2024


def test_a_club_appearing_for_the_first_time_starts_below_the_mean():
    df = pd.concat(
        [
            _season(2024, [("a", "b", 1, 1)]),
            _season(2025, [("a", "promoted", 1, 1)]),
        ],
        ignore_index=True,
    )
    ratings = update_elo(df)
    assert ratings["promoted"] < MEAN_RATING


def test_the_very_first_clubs_seen_start_at_the_mean():
    ratings = update_elo(_season(2025, [("a", "b", 0, 0)]))
    # Nothing is known about anyone yet, so nobody is penalised as promoted.
    assert np.mean([ratings["a"], ratings["b"]]) == pytest.approx(MEAN_RATING)


def test_unplayed_matches_do_not_move_ratings():
    df = _season(2025, [("a", "b", 2, 0)])
    unplayed = df.copy()
    unplayed["home_goals"] = None
    unplayed["away_goals"] = None
    both = pd.concat([df, unplayed], ignore_index=True)
    assert update_elo(both)["a"] == pytest.approx(update_elo(df)["a"])


def test_canonical_keys_mean_multi_season_history_accumulates():
    """This is what the name mismatch used to prevent."""
    two_seasons = pd.concat(
        [
            _season(2024, [("man_city", "burnley", 3, 0)] * 3),
            _season(2025, [("man_city", "burnley", 3, 0)] * 3),
        ],
        ignore_index=True,
    )
    one_season = _season(2025, [("man_city", "burnley", 3, 0)] * 3)

    assert update_elo(two_seasons)["man_city"] > update_elo(one_season)["man_city"]


def test_match_counts_only_counts_played_matches():
    df = _season(2025, [("a", "b", 1, 0), ("b", "a", 2, 2)])
    unplayed = _season(2025, [("a", "c", 0, 0)])
    unplayed["home_goals"] = None
    unplayed["away_goals"] = None

    counts = match_counts(pd.concat([df, unplayed], ignore_index=True))
    assert counts["a"] == 2
    assert counts["b"] == 2
    assert "c" not in counts


def test_no_matches_yields_no_ratings():
    empty = pd.DataFrame(
        columns=["season", "date", "home_team", "away_team", "home_goals", "away_goals"]
    )
    assert update_elo(empty) == {}
