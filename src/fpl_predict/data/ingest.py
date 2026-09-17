from __future__ import annotations

import io
import time
from typing import List, Optional

import pandas as pd
import requests
from requests import HTTPError

from ..config import current_season, season_label, settings
from ..utils.cache import RAW, SAMPLE
from ..utils.io import read_parquet, write_parquet
from ..utils.logging import get_logger
from .fpl_api import get_bootstrap, get_fixtures
from .fpl_history import fetch_season_fixtures
from .teams import bootstrap_team_names, canonical_team, team_display

log = get_logger(__name__)

# Canonical schema for the match-results store. Every source is coerced into this.
MATCH_COLS = [
    "season",
    "date",
    "gameweek",
    "home_team",
    "away_team",
    "home_slug",
    "away_slug",
    "home_goals",
    "away_goals",
    "match_id",
    "status",
]


def bootstrap_raw_from_sample() -> None:
    fx = SAMPLE / "fixtures_sample.csv"
    if not fx.exists():
        log.info("No sample files present at %s; skipping demo bootstrap.", fx.parent)
        return
    fixtures = pd.read_csv(SAMPLE / "fixtures_sample.csv", parse_dates=["date"])
    players = pd.read_csv(SAMPLE / "players_sample.csv")
    events = pd.read_csv(SAMPLE / "events_sample.csv")
    write_parquet(fixtures, RAW / "sample" / "fixtures_sample.parquet")
    write_parquet(players, RAW / "sample" / "players_sample.parquet")
    write_parquet(events, RAW / "sample" / "events_sample.parquet")
    log.info("Demo raw parquet written from bundled samples.")


FD_COMP = {"EPL": "PL", "LaLiga": "PD", "Bundesliga": "BL1", "SerieA": "SA", "Ligue1": "FL1"}
FD_BASE = "https://api.football-data.org/v4"
FD_UK_CODE = {"EPL": "E0", "LaLiga": "SP1", "Bundesliga": "D1", "SerieA": "I1", "Ligue1": "F1"}
FD_UK_BASE = "https://www.football-data.co.uk/mmz4281"


def _finalise(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Coerce any source frame into MATCH_COLS with canonical slugs."""
    if df.empty:
        return pd.DataFrame(columns=MATCH_COLS)
    out = df.copy()
    out["season"] = season
    for col in MATCH_COLS:
        if col not in out.columns:
            out[col] = None
    out["home_slug"] = out["home_team"].map(canonical_team)
    out["away_slug"] = out["away_team"].map(canonical_team)
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
    for col in ("home_goals", "away_goals"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ("gameweek", "match_id"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    out["home_team"] = out["home_team"].astype("string")
    out["away_team"] = out["away_team"].astype("string")
    out["home_slug"] = out["home_slug"].astype("string")
    out["away_slug"] = out["away_slug"].astype("string")
    out["status"] = out["status"].astype("string")
    return out[MATCH_COLS].dropna(subset=["date", "home_slug", "away_slug"])


def _from_archive(season: int) -> pd.DataFrame:
    """Results from the FPL gameweek archive. No API token needed and carries the GW."""
    fx = fetch_season_fixtures(season)
    if fx.empty:
        return pd.DataFrame()
    played = fx[fx["home_goals"].notna() & fx["away_goals"].notna()].copy()
    if played.empty:
        return pd.DataFrame()
    return _finalise(
        pd.DataFrame(
            {
                "date": played["kickoff_time"],
                "gameweek": played["gw"],
                "home_team": played["home_slug"].map(team_display),
                "away_team": played["away_slug"].map(team_display),
                "home_goals": played["home_goals"],
                "away_goals": played["away_goals"],
                "match_id": played["fixture_id"],
                "status": "FINISHED",
            }
        ),
        season,
    )


def _fd_headers() -> dict:
    tok = settings.FOOTBALL_DATA_TOKEN
    return {"X-Auth-Token": tok} if tok else {}


def _fd_uk_csv(lg: str, season: int) -> pd.DataFrame:
    code = FD_UK_CODE[lg]
    yy = f"{season % 100:02d}{(season + 1) % 100:02d}"
    url = f"{FD_UK_BASE}/{yy}/{code}.csv"
    log.info("Fallback CSV for %s %s: %s", lg, season, url)
    try:
        r = requests.get(url, timeout=45)
    except requests.RequestException as e:
        log.warning("CSV fallback unreachable (%s %s): %s", lg, season, e)
        return pd.DataFrame()
    if r.status_code != 200 or not r.content:
        log.warning("CSV fallback not available (%s %s): HTTP %s", lg, season, r.status_code)
        return pd.DataFrame()
    df = pd.read_csv(io.BytesIO(r.content))

    def pick(*c):
        for k in c:
            if k in df.columns:
                return k
        return None

    h = pick("HomeTeam", "Home", "HT")
    a = pick("AwayTeam", "Away", "AT")
    fthg = pick("FTHG", "HG", "HomeGoals")
    ftag = pick("FTAG", "AG", "AwayGoals")
    date_col = pick("Date", "MatchDate")
    if not all([h, a, date_col]):
        log.warning("CSV missing expected cols for %s %s", lg, season)
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce", dayfirst=True, utc=True),
            "home_team": df[h],
            "away_team": df[a],
            "home_goals": df[fthg] if fthg in df.columns else None,
            "away_goals": df[ftag] if ftag in df.columns else None,
            "status": "FINISHED",
        }
    )
    out["match_id"] = range(1, len(out) + 1)
    return _finalise(out, season)


def _fd_matches(code: str, season: int, lg_name: str) -> pd.DataFrame:
    url = f"{FD_BASE}/competitions/{code}/matches"
    try:
        r = requests.get(url, headers=_fd_headers(), params={"season": season}, timeout=45)
        r.raise_for_status()
        js = r.json()
    except HTTPError as e:
        if e.response is not None and e.response.status_code in (401, 403, 404):
            log.warning(
                "football-data.org rejected %s %s (HTTP %s); trying CSV fallback",
                lg_name,
                season,
                e.response.status_code,
            )
            return _fd_uk_csv(lg_name, season)
        raise
    except requests.RequestException as e:
        log.warning("football-data.org unreachable for %s %s: %s", lg_name, season, e)
        return _fd_uk_csv(lg_name, season)

    rows = []
    for m in js.get("matches", []):
        ft = m.get("score", {}).get("fullTime", {})
        rows.append(
            {
                "date": m.get("utcDate"),
                "gameweek": m.get("matchday"),
                "home_team": m.get("homeTeam", {}).get("name"),
                "away_team": m.get("awayTeam", {}).get("name"),
                "home_goals": ft.get("home"),
                "away_goals": ft.get("away"),
                "status": m.get("status"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    df = df[df["status"] == "FINISHED"]
    if df.empty:
        return pd.DataFrame()
    df["match_id"] = range(1, len(df) + 1)
    return _finalise(df, season)


def _completed_season_matches(season: int, league: str) -> pd.DataFrame:
    """Results for a finished season, archive first then football-data."""
    if league == "EPL":
        df = _from_archive(season)
        if not df.empty:
            log.info(
                "%s %s: %d matches from the FPL archive", league, season_label(season), len(df)
            )
            return df
    df = _fd_matches(FD_COMP[league], season, league)
    if not df.empty:
        log.info("%s %s: %d matches from football-data", league, season_label(season), len(df))
    return df


def _current_season_matches(season: int) -> pd.DataFrame:
    """Finished fixtures of the season in progress, from the FPL API.

    Returns empty before the first match of a new season, which is expected rather than an
    error. Callers must not treat empty as a reason to drop stored history.
    """
    try:
        fixtures = get_fixtures()
        bootstrap = get_bootstrap()
    except Exception as e:
        log.warning("Could not fetch current season from FPL API: %s", e)
        return pd.DataFrame()

    names = bootstrap_team_names(bootstrap)
    rows = []
    for f in fixtures:
        if not (f.get("finished") or f.get("finished_provisional")):
            continue
        if f.get("team_h_score") is None or f.get("team_a_score") is None:
            continue
        rows.append(
            {
                "date": f.get("kickoff_time"),
                "gameweek": f.get("event"),
                "home_team": names.get(f.get("team_h")),
                "away_team": names.get(f.get("team_a")),
                "home_goals": f.get("team_h_score"),
                "away_goals": f.get("team_a_score"),
                "match_id": f.get("id"),
                "status": "FINISHED",
            }
        )
    df = _finalise(pd.DataFrame(rows), season)
    log.info("%s: %d finished matches from the FPL API", season_label(season), len(df))
    return df


def _combine_stored(outdir, league: str) -> pd.DataFrame:
    """Rebuild the combined file from every per-season file on disk.

    Reading from disk rather than from this run's fetches is what stops an empty
    current-season fetch (normal in August) from wiping stored seasons.
    """
    frames = []
    for path in sorted(outdir.glob(f"{league}_*_matches.parquet")):
        if path.name == f"{league}_all_matches.parquet":
            continue
        try:
            df = read_parquet(path)
        except Exception as e:
            log.warning("Could not read %s: %s", path.name, e)
            continue
        missing = [c for c in MATCH_COLS if c not in df.columns]
        if missing:
            log.warning("%s predates the current schema (missing %s); skipping", path.name, missing)
            continue
        frames.append(df[MATCH_COLS])
    if not frames:
        return pd.DataFrame(columns=MATCH_COLS)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["season", "date", "home_slug", "away_slug"])
    return combined.sort_values(["season", "date"], kind="stable").reset_index(drop=True)


def ingest_full(
    seasons: Optional[List[int]] = None,
    leagues: Optional[List[str]] = None,
    include_current: bool = True,
) -> pd.DataFrame:
    """Ingest match results into `data/raw/football-data/`.

    Writes one parquet per season plus a combined `<LEAGUE>_all_matches.parquet`. The
    combined file is rebuilt from everything on disk, and a season whose fetch comes back
    empty leaves its stored file untouched.
    """
    seasons = [int(s) for s in seasons] if seasons else settings.history_seasons()
    leagues = leagues or ["EPL"]
    cur = current_season()
    outdir = RAW / "football-data"
    outdir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Ingesting %s; completed seasons %s, current season %s",
        ", ".join(leagues),
        ", ".join(season_label(s) for s in seasons) or "none",
        season_label(cur) if include_current else "skipped",
    )

    for lg in leagues:
        for yr in seasons:
            if yr >= cur:
                continue  # handled by the current-season path
            path = outdir / f"{lg}_{yr}_matches.parquet"
            df = _completed_season_matches(yr, lg)
            if df.empty:
                if path.exists():
                    log.warning(
                        "%s %s returned no matches; keeping the stored file",
                        lg,
                        season_label(yr),
                    )
                else:
                    log.warning("%s %s returned no matches and none stored", lg, season_label(yr))
                continue
            write_parquet(df, path)
            time.sleep(0.5)

        if include_current and lg == "EPL":
            path = outdir / f"{lg}_{cur}_matches.parquet"
            df = _current_season_matches(cur)
            if df.empty:
                log.info(
                    "No finished %s matches yet; leaving %s as-is",
                    season_label(cur),
                    path.name,
                )
            else:
                write_parquet(df, path)

        combined = _combine_stored(outdir, lg)
        if combined.empty:
            log.warning("No stored matches for %s; combined file not written", lg)
            continue
        write_parquet(combined, outdir / f"{lg}_all_matches.parquet")
        by_season = combined.groupby("season").size().to_dict()
        log.info(
            "%s combined: %d matches (%s)",
            lg,
            len(combined),
            ", ".join(f"{season_label(int(s))}={n}" for s, n in sorted(by_season.items())),
        )

    return _combine_stored(outdir, leagues[0])
