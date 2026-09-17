"""Leak-free player-gameweek feature panel.

One row per (season, player, gameweek) with the outcome for that gameweek and features built
only from information available before its deadline. This is the training table for anything
that wants to predict points, and the evaluation table for `models.backtest`.

**Why the leakage rule is stated so explicitly:** the pipeline this replaces trained on a
single row per player whose target was the mean of that player's last three gameweeks, while
the feature matrix contained `goals_l3`, `goals_l5` and `xg_l3` over the same window. The
model was fitting a smoothed copy of its own input, so cross-validated error looked
respectable and out-of-sample skill was nil.

Three groups of columns, and the distinction is the whole point:

| Group | Examples | Known before the deadline? |
|:------|:---------|:---------------------------|
| Outcome | `points`, `minutes`, `goals`, `bonus` | No. Never a feature. |
| Same-row context | `value`, `selected`, `was_home`, `opponent`, `n_fixtures` | Yes. FPL publishes price, ownership and the fixture before the deadline. |
| Lagged | everything suffixed `_l3`, `_l5`, `_prev`, `_std` | Yes, by construction: shifted one gameweek before any window is taken. |

Doubles are collapsed to one row per player-gameweek with `n_fixtures` recording how many
matches contributed, because that is how FPL scores them.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..utils.cache import PROC
from ..utils.io import read_parquet, write_parquet
from ..utils.logging import get_logger
from ..data.fpl_history import load_history, load_history_fixtures

log = get_logger(__name__)

PANEL_PATH = PROC / "player_gw_panel.parquet"

# Rolling windows, in appearances-or-not gameweeks.
SHORT_WINDOW = 3
LONG_WINDOW = 5
FORM_WINDOW = 10

# Matches used for team-level attack and defence rates.
TEAM_WINDOW = 10

# The outcome and its parts. None of these may ever appear as a feature.
OUTCOME_COLS = [
    "points",
    "minutes",
    "started",
    "goals",
    "assists",
    "clean_sheet",
    "goals_conceded",
    "saves",
    "bonus",
    "bps",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "xg",
    "xa",
    "xgc",
    "defensive_contribution",
    "cbi",
    "tackles",
    "recoveries",
]

# Known before the deadline, so safe to use unlagged.
CONTEXT_COLS = [
    "season",
    "gw",
    "player_code",
    "fpl_id",
    "player_name",
    "position",
    "team_slug",
    "opponent_slug",
    "was_home",
    "value",
    "selected",
    "n_fixtures",
    "kickoff_time",
]


def _collapse_doubles(h: pd.DataFrame) -> pd.DataFrame:
    """One row per player-gameweek, summing a double gameweek the way FPL scores it."""
    h = h.copy()
    h["n_fixtures"] = 1
    sums = {
        "minutes": "sum",
        "points": "sum",
        "goals": "sum",
        "assists": "sum",
        "clean_sheet": "sum",
        "goals_conceded": "sum",
        "saves": "sum",
        "bonus": "sum",
        "bps": "sum",
        "yellow_cards": "sum",
        "red_cards": "sum",
        "own_goals": "sum",
        "penalties_saved": "sum",
        "penalties_missed": "sum",
        "n_fixtures": "sum",
    }

    # Stats that did not exist in every season. `groupby.sum()` defaults to min_count=0 and
    # turns an all-missing group into 0.0, which silently rewrote "this stat was not recorded
    # yet" as "this player recorded none of it". Defensive contribution only exists from
    # 2025-26, so four fifths of the panel became confident zeroes and its measured hit rate
    # collapsed from about 15% to 2.7%.
    def _sum_or_missing(s: pd.Series) -> float:
        return s.sum(min_count=1)

    nullable_sums = {
        col: _sum_or_missing
        for col in (
            "xg",
            "xa",
            "xgc",
            "defensive_contribution",
            "cbi",
            "tackles",
            "recoveries",
            "starts",
        )
    }
    firsts = {
        # FPL publishes one expected-points figure per gameweek and the archive copies it
        # onto every fixture row, so summing double-counts it: Salah's 24.8 for a 2024-25
        # double became 49.6, which inflated the baseline and made it prefer double
        # gameweeks for the wrong reason.
        "fpl_xp": "first",
        "fpl_id": "first",
        "player_name": "first",
        "position": "first",
        "team_slug": "first",
        "opponent_slug": "first",
        "was_home": "first",
        "value": "first",
        "selected": "first",
        "kickoff_time": "min",
        "fixture_id": "first",
    }
    agg = {k: v for k, v in {**sums, **nullable_sums, **firsts}.items() if k in h.columns}
    out = h.groupby(["season", "player_code", "gw"], as_index=False).agg(agg)
    return out


def _team_form(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Attack and defence rates per team per gameweek, from earlier matches only.

    Unplayed fixtures are kept rather than dropped, so a team still gets a row (and therefore
    a pre-gameweek form figure) for a fixture that has not happened yet. That is what lets the
    live path score an upcoming gameweek with the same code that builds the training panel.
    The rolling windows are shifted by one and skip missing values, so a NaN scoreline in the
    current row cannot feed its own rate.
    """
    if fixtures.empty:
        return pd.DataFrame(columns=["season", "gw", "team_slug", "team_gf_l10", "team_ga_l10"])

    played = fixtures.copy()
    home = pd.DataFrame(
        {
            "season": played["season"],
            "gw": played["gw"],
            "team_slug": played["home_slug"],
            "gf": played["home_goals"],
            "ga": played["away_goals"],
        }
    )
    away = pd.DataFrame(
        {
            "season": played["season"],
            "gw": played["gw"],
            "team_slug": played["away_slug"],
            "gf": played["away_goals"],
            "ga": played["home_goals"],
        }
    )
    long = pd.concat([home, away], ignore_index=True).dropna(subset=["team_slug"])
    long = long.sort_values(["team_slug", "season", "gw"], kind="stable")

    grp = long.groupby("team_slug", sort=False)
    for col, out in (("gf", "team_gf_l10"), ("ga", "team_ga_l10")):
        # shift(1) first so a team's own result in this gameweek never feeds its own rate.
        long[out] = grp[col].transform(
            lambda s: s.shift(1).rolling(TEAM_WINDOW, min_periods=3).mean()
        )
    long["team_cs_rate_l10"] = grp["ga"].transform(
        # `ga == 0` on a NaN gives False, which would read an unplayed match as "conceded",
        # so the missing values are reinstated before the window is taken.
        lambda s: (
            (s == 0)
            .astype(float)
            .where(s.notna())
            .shift(1)
            .rolling(TEAM_WINDOW, min_periods=3)
            .mean()
        )
    )

    cols = ["season", "gw", "team_slug", "team_gf_l10", "team_ga_l10", "team_cs_rate_l10"]
    # A team playing twice in one gameweek has two rows here. Keep the first, whose rates are
    # the pre-gameweek ones, so the join stays one-to-one; without this the merge fanned out
    # and the panel came back with more rows than the history it was built from.
    return long[cols].drop_duplicates(subset=["season", "gw", "team_slug"], keep="first")


def _lagged(panel: pd.DataFrame) -> pd.DataFrame:
    """Add per-player lagged features. Every one is shifted before its window is taken."""
    panel = panel.sort_values(["player_code", "season", "gw"], kind="stable").copy()
    grp = panel.groupby("player_code", sort=False)

    def roll(col: str, window: int, how: str = "mean", min_periods: int = 1) -> pd.Series:
        s = grp[col]
        shifted = s.shift(1)
        r = shifted.groupby(panel["player_code"]).rolling(window, min_periods=min_periods)
        out = getattr(r, how)()
        return out.reset_index(level=0, drop=True)

    # Minutes and role
    panel["mins_l3"] = roll("minutes", SHORT_WINDOW)
    panel["mins_l5"] = roll("minutes", LONG_WINDOW)
    panel["mins_l10"] = roll("minutes", FORM_WINDOW)
    panel["mins_std_l5"] = roll("minutes", LONG_WINDOW, "std", min_periods=2)
    panel["started_l5"] = roll("started", LONG_WINDOW)
    panel["played_l5"] = roll("played", LONG_WINDOW)

    # Scoring form
    panel["points_l3"] = roll("points", SHORT_WINDOW)
    panel["points_l5"] = roll("points", LONG_WINDOW)
    panel["points_l10"] = roll("points", FORM_WINDOW)
    panel["bps_l5"] = roll("bps", LONG_WINDOW)
    panel["bonus_l5"] = roll("bonus", LONG_WINDOW)

    # Attacking output
    for col in ("goals", "assists", "xg", "xa"):
        panel[f"{col}_l5"] = roll(col, LONG_WINDOW)
        panel[f"{col}_l10"] = roll(col, FORM_WINDOW)

    # Defensive output
    panel["xgc_l5"] = roll("xgc", LONG_WINDOW)
    panel["cs_l5"] = roll("clean_sheet", LONG_WINDOW)
    panel["conceded_l5"] = roll("goals_conceded", LONG_WINDOW)
    panel["saves_l5"] = roll("saves", LONG_WINDOW)
    panel["dc_l5"] = roll("defensive_contribution", LONG_WINDOW)
    panel["dc_l10"] = roll("defensive_contribution", FORM_WINDOW)

    # Discipline
    panel["yellows_l10"] = roll("yellow_cards", FORM_WINDOW)

    # Per-90 rates over the longer window, which is what a rate model wants
    mins10 = panel["mins_l10"] * FORM_WINDOW
    for col in ("goals", "assists", "xg", "xa"):
        panel[f"{col}_per90_l10"] = np.where(
            mins10 > 0, panel[f"{col}_l10"] * FORM_WINDOW / mins10 * 90.0, np.nan
        )

    # Season-to-date, and previous-season totals. cumsum then shift keeps it strictly prior.
    season_grp = panel.groupby(["player_code", "season"], sort=False)
    panel["gws_played_this_season"] = season_grp.cumcount()
    panel["season_mins_to_date"] = season_grp["minutes"].cumsum() - panel["minutes"]
    panel["season_points_to_date"] = season_grp["points"].cumsum() - panel["points"]
    panel["season_ppg_to_date"] = np.where(
        panel["gws_played_this_season"] > 0,
        panel["season_points_to_date"] / panel["gws_played_this_season"].replace(0, np.nan),
        np.nan,
    )

    return panel


def _previous_season(panel: pd.DataFrame) -> pd.DataFrame:
    """Previous-season totals per player, joined on the stable player code."""
    by_season = (
        panel.groupby(["player_code", "season"])
        .agg(
            prev_minutes=("minutes", "sum"),
            prev_points=("points", "sum"),
            prev_goals=("goals", "sum"),
            prev_assists=("assists", "sum"),
            prev_xg=("xg", "sum"),
            prev_xa=("xa", "sum"),
            prev_starts=("started", "sum"),
            prev_gws=("gw", "size"),
        )
        .reset_index()
    )
    by_season["season"] = by_season["season"] + 1  # attach to the following season
    for col in ("goals", "assists", "xg", "xa"):
        by_season[f"prev_{col}_per90"] = np.where(
            by_season["prev_minutes"] > 0,
            by_season[f"prev_{col}"] / by_season["prev_minutes"] * 90.0,
            np.nan,
        )
    return by_season


def compute_features(rows: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Turn collapsed player-gameweek rows into the full feature panel.

    Shared by the training path and the live path so an upcoming gameweek is featurised by
    exactly the same code that produced the training data. Anything else invites train/serve
    skew, where a feature means one thing in training and another at prediction time.

    `rows` must already be one row per (season, player_code, gw). `fixtures` may include
    unplayed matches, whose team-form rates come from earlier results.
    """
    panel = rows.copy()

    panel["points"] = pd.to_numeric(panel.get("points"), errors="coerce").fillna(0)
    panel["played"] = (panel["minutes"] > 0).astype(float)
    panel["started"] = np.where(
        panel.get("starts").notna() if "starts" in panel.columns else False,
        panel.get("starts", 0),
        (panel["minutes"] >= 60).astype(float),
    )

    panel = _lagged(panel)

    prev = _previous_season(panel)
    panel = panel.merge(prev.drop(columns=["prev_gws"]), on=["player_code", "season"], how="left")

    tf = _team_form(fixtures)
    if not tf.empty:
        panel = panel.merge(tf, on=["season", "gw", "team_slug"], how="left")
        opp = tf.rename(
            columns={
                "team_slug": "opponent_slug",
                "team_gf_l10": "opp_gf_l10",
                "team_ga_l10": "opp_ga_l10",
                "team_cs_rate_l10": "opp_cs_rate_l10",
            }
        )
        panel = panel.merge(opp, on=["season", "gw", "opponent_slug"], how="left")

    panel["is_home"] = panel["was_home"].astype("float")
    panel = panel.sort_values(["season", "gw", "player_code"], kind="stable").reset_index(drop=True)

    # Exactly one row per player-gameweek. A fan-out here would silently duplicate outcomes
    # and inflate every metric computed downstream.
    dupes = panel.duplicated(subset=["season", "player_code", "gw"]).sum()
    if dupes:
        raise AssertionError(f"panel has {dupes} duplicated player-gameweek rows")
    return panel


def collapse_history(h: pd.DataFrame) -> pd.DataFrame:
    """Archive rows to one row per player-gameweek, with our column names."""
    h = h.rename(columns={"total_points": "points", "clean_sheets": "clean_sheet"})
    h = h.dropna(subset=["player_code", "gw"])
    return _collapse_doubles(h)


def build_panel(seasons: List[int] | None = None, save: bool = True) -> pd.DataFrame:
    """Assemble the feature panel from the gameweek archive."""
    h = load_history(seasons)
    if h.empty:
        log.warning("No history available; panel is empty")
        return pd.DataFrame()

    panel = compute_features(collapse_history(h), load_history_fixtures(seasons))

    log.info(
        "Panel: %d player-gameweek rows, seasons %s, %d players, %d feature columns",
        len(panel),
        sorted(panel["season"].unique()),
        panel["player_code"].nunique(),
        len(feature_columns(panel)),
    )
    if save:
        write_parquet(panel, PANEL_PATH)
    return panel


# Feature groups, for ablation. Answering "is this family of features earning its keep"
# needs them named somewhere shared rather than retyped per experiment.
FEATURE_GROUPS: dict[str, List[str]] = {
    "opponent": ["opp_gf_l10", "opp_ga_l10", "opp_cs_rate_l10"],
    "venue": ["is_home"],
    "fixture_count": ["n_fixtures"],
    "team_form": ["team_gf_l10", "team_ga_l10", "team_cs_rate_l10"],
    "market": ["value", "selected"],
    "minutes": [
        "mins_l3",
        "mins_l5",
        "mins_l10",
        "mins_std_l5",
        "started_l5",
        "played_l5",
        "season_mins_to_date",
        "gws_played_this_season",
    ],
    "scoring_form": [
        "points_l3",
        "points_l5",
        "points_l10",
        "bps_l5",
        "bonus_l5",
        "season_points_to_date",
        "season_ppg_to_date",
    ],
    "attacking_rate": [
        "goals_l5",
        "goals_l10",
        "assists_l5",
        "assists_l10",
        "xg_l5",
        "xg_l10",
        "xa_l5",
        "xa_l10",
        "goals_per90_l10",
        "assists_per90_l10",
        "xg_per90_l10",
        "xa_per90_l10",
    ],
    "defensive_rate": ["xgc_l5", "cs_l5", "conceded_l5", "saves_l5", "dc_l5", "dc_l10"],
    "previous_season": [
        "prev_minutes",
        "prev_points",
        "prev_goals",
        "prev_assists",
        "prev_xg",
        "prev_xa",
        "prev_starts",
        "prev_goals_per90",
        "prev_assists_per90",
        "prev_xg_per90",
        "prev_xa_per90",
    ],
}

# The groups that describe *this* fixture rather than the player. If these contribute nothing,
# the model is a pure player-quality estimator.
CONTEXT_GROUPS = ["opponent", "venue", "team_form", "fixture_count"]


def group_features(*groups: str) -> List[str]:
    """Flatten one or more named feature groups."""
    out: List[str] = []
    for g in groups:
        out.extend(FEATURE_GROUPS.get(g, []))
    return out


def feature_columns(panel: pd.DataFrame) -> List[str]:
    """Columns safe to hand a model: everything that is neither an outcome nor an identifier."""
    banned = set(OUTCOME_COLS) | {
        "season",
        "gw",
        "player_code",
        "fpl_id",
        "player_name",
        "position",
        "team_slug",
        "opponent_slug",
        "kickoff_time",
        "fixture_id",
        "was_home",
        "played",
        "starts",
        "fpl_xp",
    }
    return [c for c in panel.columns if c not in banned and pd.api.types.is_numeric_dtype(panel[c])]


def load_panel(seasons: List[int] | None = None, rebuild: bool = False) -> pd.DataFrame:
    """Load the panel, building it if absent."""
    if rebuild or not PANEL_PATH.exists():
        return build_panel(seasons)
    panel = read_parquet(PANEL_PATH)
    if seasons is not None:
        panel = panel[panel["season"].isin([int(s) for s in seasons])]
    return panel
