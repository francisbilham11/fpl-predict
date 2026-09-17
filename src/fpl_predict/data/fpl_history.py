"""Per-player, per-gameweek FPL history across seasons.

The FPL API only exposes the current season game by game (`element-summary.history`) plus
season totals for earlier years (`history_past`), so the archive at
github.com/vaastav/Fantasy-Premier-League is the only practical source of player-gameweek
rows going back. It publishes one `gws/merged_gw.csv` per season, ~30k rows each.

Two things make the raw CSVs awkward to use directly and are handled here:

- **Element ids are season-scoped.** FPL renumbers players every summer, so `element` 411
  is not the same person across seasons. `players_raw.csv` carries `code`, which is stable,
  and every row is keyed on that instead.
- **`opponent_team` is a season-scoped team id**, renumbered alphabetically each year.
  `teams.csv` resolves it, and the result is stored as a canonical slug.

Column coverage varies by season and is left as NaN rather than zero where a stat did not
exist yet, so a model can tell "not recorded" from "recorded as none":

| Columns                                        | From    |
|:-----------------------------------------------|:--------|
| minutes, goals, assists, points, bps, value    | 2020-21 |
| xG, xA, xGC, expected goal involvements, starts | 2022-23 |
| tackles, CBI, recoveries, defensive contribution | 2025-26 |

`xP` (FPL's own pre-deadline expected points) is kept for every season. It is the baseline
any model here has to beat, so the backtest needs it alongside the outcomes.
"""

from __future__ import annotations

import io
from functools import lru_cache
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests

from ..config import season_label, settings
from ..utils.cache import RAW
from ..utils.io import read_parquet, write_parquet
from ..utils.logging import get_logger
from .teams import canonical_team

log = get_logger(__name__)

ARCHIVE_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
UA = {"User-Agent": "fpl-predict/0.5 (training pipeline)"}

HISTORY_DIR = RAW / "fpl_history"

# Earliest season with the `position` and `team` columns this loader relies on.
EARLIEST_SEASON = 2020

# The archive spells goalkeeper "GK"; the FPL API and everything downstream say "GKP".
POSITION_ALIASES = {"GK": "GKP", "GKP": "GKP", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}

# 2024-25 briefly had an Assistant Manager element type ("AM"). It does not exist in
# 2026/27, so those rows are dropped rather than mixed in with outfield players.
DROPPED_POSITIONS = {"AM"}

# Rename map from archive column -> our schema. Anything not listed is dropped.
_RENAME = {
    "element": "fpl_id",
    "name": "player_name",
    "position": "position",
    "round": "gw",
    "minutes": "minutes",
    "starts": "starts",
    "total_points": "total_points",
    "goals_scored": "goals",
    "assists": "assists",
    "clean_sheets": "clean_sheets",
    "goals_conceded": "goals_conceded",
    "saves": "saves",
    "penalties_saved": "penalties_saved",
    "penalties_missed": "penalties_missed",
    "own_goals": "own_goals",
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
    "bonus": "bonus",
    "bps": "bps",
    "influence": "influence",
    "creativity": "creativity",
    "threat": "threat",
    "ict_index": "ict_index",
    "expected_goals": "xg",
    "expected_assists": "xa",
    "expected_goal_involvements": "xgi",
    "expected_goals_conceded": "xgc",
    "clearances_blocks_interceptions": "cbi",
    "recoveries": "recoveries",
    "tackles": "tackles",
    "defensive_contribution": "defensive_contribution",
    "value": "value",
    "selected": "selected",
    "transfers_in": "transfers_in",
    "transfers_out": "transfers_out",
    "transfers_balance": "transfers_balance",
    "was_home": "was_home",
    "fixture": "fixture_id",
    "kickoff_time": "kickoff_time",
    "xP": "fpl_xp",
}

# Stats that exist in every supported season, so a NaN really means missing data.
_COUNT_COLS = [
    "minutes",
    "total_points",
    "goals",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "penalties_saved",
    "penalties_missed",
    "own_goals",
    "yellow_cards",
    "red_cards",
    "bonus",
    "bps",
]

# Stats introduced part-way through the archive. Left NaN before they existed.
_OPTIONAL_COLS = [
    "starts",
    "xg",
    "xa",
    "xgi",
    "xgc",
    "cbi",
    "recoveries",
    "tackles",
    "defensive_contribution",
]


@lru_cache(maxsize=64)
def _get_csv_cached(url: str) -> Optional[pd.DataFrame]:
    r = requests.get(url, headers=UA, timeout=120)
    if r.status_code != 200 or not r.content:
        log.warning("Archive fetch failed (HTTP %s): %s", r.status_code, url)
        return None
    return pd.read_csv(io.BytesIO(r.content))


def _get_csv(url: str) -> Optional[pd.DataFrame]:
    """Fetch a CSV from the archive. Cached per URL; callers get their own copy."""
    df = _get_csv_cached(url)
    return None if df is None else df.copy()


@lru_cache(maxsize=32)
def _season_team_slugs(season: int) -> Dict[int, str]:
    """Season-scoped FPL team id -> canonical slug."""
    label = season_label(season)
    df = _get_csv(f"{ARCHIVE_BASE}/{label}/teams.csv")
    if df is None or "id" not in df.columns:
        log.warning("No teams.csv for %s; opponent slugs will be blank", label)
        return {}
    return {int(r.id): canonical_team(r.name) for r in df.itertuples()}


@lru_cache(maxsize=32)
def _season_player_codes(season: int) -> Dict[int, int]:
    """Season-scoped element id -> stable player code."""
    label = season_label(season)
    df = _get_csv(f"{ARCHIVE_BASE}/{label}/players_raw.csv")
    if df is None or not {"id", "code"} <= set(df.columns):
        log.warning("No players_raw.csv for %s; falling back to season-scoped ids", label)
        return {}
    return {int(r.id): int(r.code) for r in df.itertuples()}


def fetch_season_gameweeks(season: int) -> pd.DataFrame:
    """Download and normalise one season of player-gameweek rows."""
    label = season_label(season)
    raw = _get_csv(f"{ARCHIVE_BASE}/{label}/gws/merged_gw.csv")
    if raw is None or raw.empty:
        return pd.DataFrame()

    present = {src: dst for src, dst in _RENAME.items() if src in raw.columns}
    df = raw[list(present)].rename(columns=present).copy()

    # `GW` and `round` disagree in a handful of archived rows (rescheduled fixtures); `GW`
    # is the file's own partition key, so prefer it.
    if "GW" in raw.columns:
        df["gw"] = pd.to_numeric(raw["GW"], errors="coerce")

    df["season"] = season
    df["season_label"] = label

    team_slugs = _season_team_slugs(season)
    df["team_slug"] = (
        raw["team"].map(canonical_team) if "team" in raw.columns else pd.Series("", index=df.index)
    )
    df["opponent_slug"] = (
        pd.to_numeric(raw["opponent_team"], errors="coerce").map(team_slugs)
        if "opponent_team" in raw.columns
        else pd.Series("", index=df.index)
    )

    codes = _season_player_codes(season)
    if codes:
        df["player_code"] = pd.to_numeric(df.get("fpl_id"), errors="coerce").map(codes)
    else:
        # No stable code mapping for this season (players_raw.csv fetch failed). Fall back
        # to the season-scoped id itself: not stable across seasons, but unique within this
        # one, which is what matters here — _combine_cached()'s drop_duplicates keys on
        # (season, gw, player_code, fixture_id), and pandas treats NaN == NaN as a match, so
        # leaving this all-NaN (the previous behaviour) silently collapsed every player in a
        # given fixture down to one surviving row for the whole season.
        log.warning("%s: no stable player codes; using season-scoped ids for this season only", label)
        df["player_code"] = pd.to_numeric(df.get("fpl_id"), errors="coerce")
    missing_codes = int(df["player_code"].isna().sum())
    if missing_codes:
        log.warning("%s: %d rows without a stable player code", label, missing_codes)

    for col in _COUNT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in _OPTIONAL_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.NA
    for col in ("influence", "creativity", "threat", "ict_index", "fpl_xp"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["kickoff_time"] = pd.to_datetime(df.get("kickoff_time"), utc=True, errors="coerce")
    df["was_home"] = df.get("was_home").astype("boolean") if "was_home" in df.columns else pd.NA

    raw_positions = df.get("position", pd.Series(dtype="object")).astype("string")
    dropped = raw_positions.isin(DROPPED_POSITIONS)
    if dropped.any():
        log.info("%s: dropping %d manager rows", label, int(dropped.sum()))
        df = df[~dropped]
        raw_positions = raw_positions[~dropped]
    unknown = sorted(set(raw_positions.dropna().unique()) - set(POSITION_ALIASES))
    if unknown:
        log.warning("%s: unrecognised positions %s left as-is", label, unknown)
    df["position"] = raw_positions.map(lambda p: POSITION_ALIASES.get(p, p)).astype("string")

    df["gw"] = pd.to_numeric(df["gw"], errors="coerce").astype("Int64")
    df["fpl_id"] = pd.to_numeric(df["fpl_id"], errors="coerce").astype("Int64")
    df["player_code"] = pd.to_numeric(df["player_code"], errors="coerce").astype("Int64")
    df["fixture_id"] = pd.to_numeric(df.get("fixture_id"), errors="coerce").astype("Int64")

    log.info(
        "%s: %d player-gameweek rows, GW %s-%s, %d players",
        label,
        len(df),
        df["gw"].min(),
        df["gw"].max(),
        df["player_code"].nunique(),
    )
    return df


def fetch_season_fixtures(season: int) -> pd.DataFrame:
    """Fixture list for a season with canonical team slugs."""
    label = season_label(season)
    raw = _get_csv(f"{ARCHIVE_BASE}/{label}/fixtures.csv")
    if raw is None or raw.empty:
        return pd.DataFrame()

    slugs = _season_team_slugs(season)
    out = pd.DataFrame(
        {
            "season": season,
            "fixture_id": pd.to_numeric(raw.get("id"), errors="coerce").astype("Int64"),
            "gw": pd.to_numeric(raw.get("event"), errors="coerce").astype("Int64"),
            "kickoff_time": pd.to_datetime(raw.get("kickoff_time"), utc=True, errors="coerce"),
            "home_slug": pd.to_numeric(raw.get("team_h"), errors="coerce").map(slugs),
            "away_slug": pd.to_numeric(raw.get("team_a"), errors="coerce").map(slugs),
            "home_goals": pd.to_numeric(raw.get("team_h_score"), errors="coerce"),
            "away_goals": pd.to_numeric(raw.get("team_a_score"), errors="coerce"),
            "finished": raw.get("finished").astype("boolean")
            if "finished" in raw.columns
            else pd.NA,
        }
    )
    return out


def _combine_cached() -> pd.DataFrame:
    """Concatenate every per-season file on disk into the combined table.

    Reading from disk rather than from the seasons this call happened to fetch is what stops
    a single-season refresh (`build_history(seasons=[2025], refresh=True)`, which the weekly
    update does) from rewriting the combined file with only that one season.
    """
    frames: List[pd.DataFrame] = []
    for path in sorted(HISTORY_DIR.glob("*/player_gw.parquet")):
        try:
            frames.append(read_parquet(path))
        except Exception as e:
            log.warning("Could not read %s: %s", path, e)
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    # drop_duplicates treats NaN == NaN as a match, which would collapse every row with a
    # missing player_code (e.g. an individual player whose archive row lacked a stable code)
    # down to one survivor per (season, gw, fixture_id) — those rows aren't known duplicates,
    # just unidentifiable, so keep them all and only dedupe the ones with a real key.
    has_code = combined["player_code"].notna()
    keyed = combined[has_code].drop_duplicates(subset=["season", "gw", "player_code", "fixture_id"])
    combined = pd.concat([keyed, combined[~has_code]], ignore_index=True)
    combined = combined.sort_values(["season", "gw", "player_code"], kind="stable")
    write_parquet(combined, HISTORY_DIR / "player_gw.parquet")
    log.info(
        "Player-gameweek history: %d rows across %d seasons (%s)",
        len(combined),
        combined["season"].nunique(),
        ", ".join(season_label(int(s)) for s in sorted(combined["season"].unique())),
    )
    return combined


def build_history(
    seasons: Iterable[int] | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Download (or reuse) player-gameweek history and write the combined table.

    Per-season parquet lands in `data/raw/fpl_history/<season>/`, and the concatenation of
    every cached season in `data/raw/fpl_history/player_gw.parquet`. Seasons already on disk
    are reused unless `refresh` is set, so a mid-season update only re-pulls the season in
    progress.
    """
    if seasons is None:
        seasons = [s for s in settings.history_seasons() if s >= EARLIEST_SEASON]
    seasons = sorted(set(int(s) for s in seasons))

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    for season in seasons:
        gw_path = HISTORY_DIR / str(season) / "player_gw.parquet"
        fx_path = HISTORY_DIR / str(season) / "fixtures.parquet"

        if gw_path.exists() and not refresh:
            log.info("Reusing cached history for %s", season_label(season))
            continue

        df = fetch_season_gameweeks(season)
        if df.empty:
            if gw_path.exists():
                log.warning(
                    "No history returned for %s; keeping the cached file",
                    season_label(season),
                )
            else:
                log.warning("No history available for %s; skipping", season_label(season))
            continue
        write_parquet(df, gw_path)

        fixtures = fetch_season_fixtures(season)
        if not fixtures.empty:
            write_parquet(fixtures, fx_path)

    combined = _combine_cached()
    if combined.empty:
        log.warning("No player-gameweek history assembled")
    return combined


def load_history(seasons: Iterable[int] | None = None) -> pd.DataFrame:
    """Load the combined history from disk, building it if absent."""
    path = HISTORY_DIR / "player_gw.parquet"
    if not path.exists():
        return build_history(seasons)
    df = read_parquet(path)
    if seasons is not None:
        df = df[df["season"].isin([int(s) for s in seasons])]
    return df


def load_history_fixtures(seasons: Iterable[int] | None = None) -> pd.DataFrame:
    """Historical fixture lists with canonical slugs, oldest season first."""
    if seasons is None:
        seasons = [s for s in settings.history_seasons() if s >= EARLIEST_SEASON]
    frames = []
    for season in sorted(set(int(s) for s in seasons)):
        path = HISTORY_DIR / str(season) / "fixtures.parquet"
        if path.exists():
            frames.append(read_parquet(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
