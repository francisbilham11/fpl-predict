"""Featurise an upcoming gameweek so a model trained on the panel can score it.

The panel is built from matches that have been played. To predict the next gameweek we need
rows that do not exist yet, with the same 55 features attached. The approach is to append
synthetic rows for the upcoming gameweek to the played history and run them through
`panel.compute_features`, the identical function that built the training data.

That is deliberate. Recomputing the features with bespoke live code is the classic source of
train/serve skew, where `mins_l5` quietly means "last five gameweeks" in training and "last
five appearances" in production and nobody notices because both are plausible numbers.

Three scale mismatches between the archive and the live API that have to be reconciled, and
each would pass unnoticed as a plausible number:

| Field | Archive | Live API | Handling |
|:------|:--------|:---------|:---------|
| `selected` | count of managers | `selected_by_percent` | percentage x `total_players` |
| `value` | price in tenths | `now_cost` in tenths | same, used directly |
| player id | season-scoped `element` | season-scoped `id` | both mapped to the stable `code` |

A blank gameweek is represented by the player simply having no row, which is what the archive
does. A double gives `n_fixtures = 2` on one row, matching how FPL scores it.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import current_season, season_label
from ..data.fpl_api import get_bootstrap, get_element_summary, get_fixtures
from ..data.fpl_history import load_history, load_history_fixtures
from ..data.teams import bootstrap_team_slugs
from ..utils.logging import get_logger
from .panel import collapse_history, compute_features

log = get_logger(__name__)

POSITION_BY_TYPE = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def next_gameweek(bootstrap: Optional[dict] = None) -> Optional[int]:
    """The gameweek the next deadline belongs to."""
    bs = bootstrap or get_bootstrap()
    events = bs.get("events", [])
    for ev in events:
        if ev.get("is_next"):
            return int(ev["id"])
    for ev in events:
        if not ev.get("finished"):
            return int(ev["id"])
    return None


def current_season_rows(bootstrap: Optional[dict] = None) -> pd.DataFrame:
    """Played gameweeks of the season in progress, in panel shape.

    Empty before the season's first match, which is the normal August state and means every
    lagged feature comes from previous seasons instead.
    """
    bs = bootstrap or get_bootstrap()
    season = current_season()
    id_to_slug = bootstrap_team_slugs(bs)
    elements = bs.get("elements", [])

    rows: List[dict] = []
    for i, e in enumerate(elements):
        if e.get("code") is None:
            continue
        try:
            summary = get_element_summary(int(e["id"]))
        except Exception as exc:
            log.debug("No summary for player %s: %s", e.get("id"), exc)
            continue
        for g in summary.get("history", []):
            rows.append(
                {
                    "season": season,
                    "gw": g.get("round"),
                    "player_code": int(e["code"]),
                    "fpl_id": int(e["id"]),
                    "player_name": e.get("web_name"),
                    "position": POSITION_BY_TYPE.get(e.get("element_type")),
                    "team_slug": id_to_slug.get(e.get("team"), ""),
                    "opponent_slug": id_to_slug.get(g.get("opponent_team"), ""),
                    "was_home": bool(g.get("was_home")),
                    "value": g.get("value"),
                    "selected": g.get("selected"),
                    "kickoff_time": g.get("kickoff_time"),
                    "fixture_id": g.get("fixture"),
                    "minutes": g.get("minutes", 0),
                    "total_points": g.get("total_points", 0),
                    "goals": g.get("goals_scored", 0),
                    "assists": g.get("assists", 0),
                    "clean_sheets": g.get("clean_sheets", 0),
                    "goals_conceded": g.get("goals_conceded", 0),
                    "saves": g.get("saves", 0),
                    "bonus": g.get("bonus", 0),
                    "bps": g.get("bps", 0),
                    "yellow_cards": g.get("yellow_cards", 0),
                    "red_cards": g.get("red_cards", 0),
                    "own_goals": g.get("own_goals", 0),
                    "penalties_saved": g.get("penalties_saved", 0),
                    "penalties_missed": g.get("penalties_missed", 0),
                    "xg": pd.to_numeric(g.get("expected_goals"), errors="coerce"),
                    "xa": pd.to_numeric(g.get("expected_assists"), errors="coerce"),
                    "xgc": pd.to_numeric(g.get("expected_goals_conceded"), errors="coerce"),
                    "defensive_contribution": g.get("defensive_contribution"),
                    "cbi": g.get("clearances_blocks_interceptions"),
                    "tackles": g.get("tackles"),
                    "recoveries": g.get("recoveries"),
                    "starts": g.get("starts", 0),
                }
            )
        if i and i % 200 == 0:
            log.info("Current-season history: %d/%d players", i, len(elements))

    df = pd.DataFrame(rows)
    if df.empty:
        log.info("No %s gameweeks played yet", season_label(season))
    else:
        df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True, errors="coerce")
        log.info(
            "Current-season history: %d rows across %d gameweeks",
            len(df),
            df["gw"].nunique(),
        )
    return df


def upcoming_fixtures(gw: int, bootstrap: Optional[dict] = None) -> pd.DataFrame:
    """Fixtures for `gw`, one row per team per match, with canonical slugs."""
    bs = bootstrap or get_bootstrap()
    id_to_slug = bootstrap_team_slugs(bs)
    rows = []
    for f in get_fixtures():
        if f.get("event") != gw:
            continue
        h, a = id_to_slug.get(f.get("team_h"), ""), id_to_slug.get(f.get("team_a"), "")
        rows.append(
            {
                "team_slug": h,
                "opponent_slug": a,
                "was_home": True,
                "fixture_id": f.get("id"),
                "kickoff_time": f.get("kickoff_time"),
            }
        )
        rows.append(
            {
                "team_slug": a,
                "opponent_slug": h,
                "was_home": False,
                "fixture_id": f.get("id"),
                "kickoff_time": f.get("kickoff_time"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True, errors="coerce")
        df = df.sort_values("kickoff_time", kind="stable")
    return df


def upcoming_rows(gw: int, bootstrap: Optional[dict] = None) -> pd.DataFrame:
    """Synthetic panel rows for `gw`: pre-deadline context, no outcome."""
    bs = bootstrap or get_bootstrap()
    id_to_slug = bootstrap_team_slugs(bs)
    total_players = float(bs.get("total_players") or 0) or np.nan

    fixtures = upcoming_fixtures(gw, bs)
    if fixtures.empty:
        log.warning("No fixtures found for GW%s", gw)
        return pd.DataFrame()

    per_team: Dict[str, pd.DataFrame] = {s: g for s, g in fixtures.groupby("team_slug")}
    season = current_season()

    rows = []
    for e in bs.get("elements", []):
        if e.get("code") is None:
            continue
        slug = id_to_slug.get(e.get("team"), "")
        team_fixtures = per_team.get(slug)
        if team_fixtures is None or team_fixtures.empty:
            continue  # blank gameweek: no row, exactly as the archive represents it

        first = team_fixtures.iloc[0]
        # Ownership is a percentage live and a headcount in the archive. Converting keeps the
        # feature on the scale the model was trained on.
        pct = pd.to_numeric(e.get("selected_by_percent"), errors="coerce")
        rows.append(
            {
                "season": season,
                "gw": gw,
                "player_code": int(e["code"]),
                "fpl_id": int(e["id"]),
                "player_name": e.get("web_name"),
                "position": POSITION_BY_TYPE.get(e.get("element_type")),
                "team_slug": slug,
                # For a double, the model sees the first opponent and a fixture count of two.
                "opponent_slug": first["opponent_slug"],
                "was_home": bool(first["was_home"]),
                "value": e.get("now_cost"),
                "selected": (pct / 100.0 * total_players) if total_players else np.nan,
                "kickoff_time": first["kickoff_time"],
                "fixture_id": first["fixture_id"],
                "n_fixtures": len(team_fixtures),
                # No outcome: these are what the model is being asked to predict.
                "minutes": 0.0,
                "points": np.nan,
                "starts": np.nan,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    doubles = int((df["n_fixtures"] > 1).sum())
    log.info(
        "GW%d: %d players with a fixture, %d with a double, %d clubs blank",
        gw,
        len(df),
        doubles,
        len(set(id_to_slug.values()) - set(per_team)),
    )
    return df


def build_live_panel(
    gw: Optional[int] = None, bootstrap: Optional[dict] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(training rows, rows to score) for the upcoming gameweek.

    Both come out of `panel.compute_features`, so the features are identical in meaning.
    """
    bs = bootstrap or get_bootstrap()
    gw = gw or next_gameweek(bs)
    if gw is None:
        raise ValueError("No upcoming gameweek found in bootstrap-static")

    played = collapse_history(load_history())
    current = current_season_rows(bs)
    if not current.empty:
        played = pd.concat([played, collapse_history(current)], ignore_index=True)

    future = upcoming_rows(gw, bs)
    if future.empty:
        raise ValueError(f"No players have a fixture in GW{gw}")

    combined = pd.concat([played, future], ignore_index=True)
    fixtures = load_history_fixtures()

    # Give the upcoming matches a fixture row too, unplayed, so team form resolves for them.
    up = upcoming_fixtures(gw, bs)
    if not up.empty:
        home_side = up[up["was_home"]]
        fixtures = pd.concat(
            [
                fixtures,
                pd.DataFrame(
                    {
                        "season": current_season(),
                        "gw": gw,
                        "fixture_id": home_side["fixture_id"].values,
                        "kickoff_time": home_side["kickoff_time"].values,
                        "home_slug": home_side["team_slug"].values,
                        "away_slug": home_side["opponent_slug"].values,
                        "home_goals": np.nan,
                        "away_goals": np.nan,
                        "finished": False,
                    }
                ),
            ],
            ignore_index=True,
        )

    panel = compute_features(combined, fixtures)
    season = current_season()
    is_target = (panel["season"] == season) & (panel["gw"] == gw)
    train = panel[~is_target & panel["points"].notna()]
    score = panel[is_target]

    log.info(
        "Live panel for GW%d: %d rows to score, %d training rows",
        gw,
        len(score),
        len(train),
    )
    return train, score
