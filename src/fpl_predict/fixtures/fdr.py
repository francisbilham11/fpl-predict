from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.fpl_api import get_bootstrap
from ..data.fpl_api import get_fixtures as fpl_fixtures
from ..data.teams import bootstrap_team_slugs, canonical_team, team_display
from ..utils.cache import PROC
from ..utils.io import read_parquet, write_parquet
from ..utils.logging import get_logger
from .elo import MEAN_RATING, expected_score, match_counts, update_elo

log = get_logger(__name__)

# Elo spread implied by one standard deviation of FPL's own team strength ratings.
STRENGTH_ELO_SCALE = 120.0

# Matches of history needed before a team's Elo outweighs the FPL strength prior. At 20, a
# club with a full season behind it sits around two-thirds Elo, and a promoted club with no
# top-flight history falls back entirely to the prior.
SHRINK_MATCHES = 20.0


def _strength_prior(bootstrap: dict, centre: float) -> dict[str, float]:
    """FPL's own team strengths, rescaled onto the Elo scale.

    This is the only signal available for a promoted club before it has played, and it also
    reflects summer transfer activity that last season's results cannot.
    """
    tdf = pd.DataFrame(bootstrap["teams"])
    s = tdf["strength_overall_home"].fillna(0).astype(float) + tdf["strength_overall_away"].fillna(
        0
    ).astype(float)
    sd = s.std()
    z = (s - s.mean()) / (sd if sd else 1.0)
    slugs = [canonical_team(n) for n in tdf["name"].astype(str)]
    return {slug: float(centre + STRENGTH_ELO_SCALE * zi) for slug, zi in zip(slugs, z)}


def compute_team_ratings() -> tuple[dict[str, float], dict[str, int]]:
    """Blended team ratings for the current 20 clubs, keyed on canonical slug.

    Elo over the stored results, shrunk toward the FPL strength prior in proportion to how
    little history a club has. Returns (ratings, matches_used).
    """
    boot = get_bootstrap()
    id_to_slug = bootstrap_team_slugs(boot)
    current = set(id_to_slug.values())

    try:
        past = read_parquet(PROC / "fixtures.parquet")
    except Exception as e:
        log.warning("No stored fixtures to rate teams from (%s)", e)
        past = pd.DataFrame()

    elo: dict[str, float] = {}
    counts: dict[str, int] = {}
    if not past.empty and {"home_slug", "away_slug"} <= set(past.columns):
        keep = [
            c
            for c in ("season", "date", "home_slug", "away_slug", "home_goals", "away_goals")
            if c in past.columns
        ]
        rated = past[keep].rename(columns={"home_slug": "home_team", "away_slug": "away_team"})
        elo = update_elo(rated)
        counts = match_counts(rated)
    elif not past.empty:
        log.warning("fixtures.parquet has no canonical slugs; rebuild features to fix Elo")

    with_history = [elo[s] for s in current if s in elo]
    centre = float(np.mean(with_history)) if with_history else MEAN_RATING
    prior = _strength_prior(boot, centre)

    ratings: dict[str, float] = {}
    used: dict[str, int] = {}
    for slug in sorted(current):
        n = counts.get(slug, 0)
        w = n / (n + SHRINK_MATCHES)
        base = elo.get(slug, centre)
        ratings[slug] = w * base + (1.0 - w) * prior.get(slug, centre)
        used[slug] = n

    spread = float(np.std(list(ratings.values()))) if ratings else 0.0
    log.info(
        "Team ratings: %d clubs, spread %.1f Elo, %d with no stored history",
        len(ratings),
        spread,
        sum(1 for n in used.values() if n == 0),
    )
    if spread < 1.0:
        log.warning("Team ratings are effectively flat; fixture difficulty will carry no signal")
    return ratings, used


def compute_fdr() -> pd.DataFrame:
    """Per-fixture difficulty for played and upcoming matches, keyed on canonical slugs."""
    ratings, used = compute_team_ratings()
    boot = get_bootstrap()
    id_to_slug = bootstrap_team_slugs(boot)

    rows = []

    try:
        past = read_parquet(PROC / "fixtures.parquet")
    except Exception:
        past = pd.DataFrame()

    for m in past.itertuples():
        hs = getattr(m, "home_slug", None) or canonical_team(getattr(m, "home_team", ""))
        as_ = getattr(m, "away_slug", None) or canonical_team(getattr(m, "away_team", ""))
        rh = ratings.get(hs, MEAN_RATING)
        ra = ratings.get(as_, MEAN_RATING)
        p_home = expected_score(rh, ra, home=True)
        rows.append(
            {
                "match_id": getattr(m, "match_id", None),
                "home_slug": hs,
                "away_slug": as_,
                "home_team": team_display(hs),
                "away_team": team_display(as_),
                "fdr_home": float(1.0 - p_home),
                "fdr_away": float(p_home),
                "is_future": False,
                "event": getattr(m, "gameweek", None),
                "kickoff_time": getattr(m, "date", None),
            }
        )

    for fx in fpl_fixtures():
        if fx.get("finished") or fx.get("finished_provisional"):
            continue
        hs = id_to_slug.get(fx.get("team_h"), "")
        as_ = id_to_slug.get(fx.get("team_a"), "")
        rh = ratings.get(hs, MEAN_RATING)
        ra = ratings.get(as_, MEAN_RATING)
        p_home = expected_score(rh, ra, home=True)
        rows.append(
            {
                "match_id": fx.get("id"),
                "home_slug": hs,
                "away_slug": as_,
                "home_team": team_display(hs),
                "away_team": team_display(as_),
                "fdr_home": float(1.0 - p_home),
                "fdr_away": float(p_home),
                "is_future": True,
                "event": fx.get("event"),
                "kickoff_time": fx.get("kickoff_time"),
            }
        )

    fdr = pd.DataFrame(rows)
    if fdr.empty:
        log.warning("No fixtures to compute FDR from")
        write_parquet(fdr, PROC / "fdr.parquet")
        return fdr

    for col in ("home_team", "away_team", "home_slug", "away_slug"):
        fdr[col] = fdr[col].astype("string")
    for col in ("fdr_home", "fdr_away"):
        fdr[col] = pd.to_numeric(fdr[col], errors="coerce")
    for col in ("match_id", "event"):
        fdr[col] = pd.to_numeric(fdr[col], errors="coerce").astype("Int64")
    kt = pd.to_datetime(fdr["kickoff_time"], utc=True, errors="coerce")
    fdr["kickoff_time"] = kt.dt.strftime("%Y-%m-%dT%H:%M:%SZ").astype("string")
    fdr["is_future"] = fdr["is_future"].astype(bool)

    fut = fdr[fdr["is_future"]]
    log.info(
        "FDR computed: %d rows (%d upcoming), upcoming fdr_home spread %.3f",
        len(fdr),
        len(fut),
        float(fut["fdr_home"].std()) if len(fut) > 1 else 0.0,
    )
    write_parquet(fdr, PROC / "fdr.parquet")
    return fdr


def build_player_next5_fdr(current_event: int | None = None, horizon: int = 5) -> pd.DataFrame:
    """Per-player fixture-difficulty multiplier over the next `horizon` gameweeks.

    Averages difficulty across a team's upcoming fixtures, counting every fixture in a
    double gameweek and treating a blank as no fixture, then normalises so the league
    average is 1.0.
    """
    boot = get_bootstrap()
    fdr = read_parquet(PROC / "fdr.parquet")
    fut = fdr[fdr["is_future"] == True].copy() if not fdr.empty else pd.DataFrame()

    if fut.empty:
        out = pd.DataFrame({"player_id": [e["id"] for e in boot["elements"]], "fdr_factor": 1.0})
        out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
        write_parquet(out, PROC / "player_next5_fdr.parquet")
        log.info("player_next5_fdr: no future fixtures; neutral factors for %d players", len(out))
        return out

    fut["event"] = pd.to_numeric(fut["event"], errors="coerce").astype("Int64")
    fut = fut.dropna(subset=["event"])
    if current_event is None:
        current_event = int(fut["event"].min())
    window = set(range(current_event, current_event + horizon))

    # One row per team per upcoming fixture, so a double gameweek contributes twice.
    per_team = pd.concat(
        [
            fut[["home_slug", "fdr_home", "event"]].rename(
                columns={"home_slug": "slug", "fdr_home": "fdr"}
            ),
            fut[["away_slug", "fdr_away", "event"]].rename(
                columns={"away_slug": "slug", "fdr_away": "fdr"}
            ),
        ],
        ignore_index=True,
    )
    per_team = per_team[per_team["event"].isin(window)]

    # Easier fixtures should scale attacking output up, so invert difficulty. Counting
    # fixtures rather than averaging them credits a double gameweek and penalises a blank.
    agg = per_team.groupby("slug").agg(total=("fdr", "sum"), n=("fdr", "size"))
    baseline = float(per_team["fdr"].mean()) if len(per_team) else 0.5
    expected_games = float(agg["n"].median()) if len(agg) else float(horizon)
    if expected_games <= 0:
        expected_games = float(horizon)

    team_fac: dict[str, float] = {}
    for slug, row in agg.iterrows():
        # A team with fewer fixtures in the window gets the league-average difficulty for
        # the games it does not have, so blanks read as neutral rather than as easy.
        filler = max(0.0, expected_games - row["n"]) * baseline
        mean_diff = (row["total"] + filler) / max(expected_games, row["n"])
        team_fac[slug] = baseline / max(mean_diff, 1e-6)

    if team_fac:
        mean_fac = sum(team_fac.values()) / len(team_fac)
        for k in team_fac:
            team_fac[k] /= mean_fac if mean_fac else 1.0

    id_to_slug = bootstrap_team_slugs(boot)
    elements = pd.DataFrame(boot["elements"])[["id", "team"]].rename(
        columns={"id": "player_id", "team": "team_id"}
    )
    elements["slug"] = elements["team_id"].map(id_to_slug)
    elements["fdr_factor"] = elements["slug"].map(team_fac).fillna(1.0)

    out = elements[["player_id", "fdr_factor"]].copy()
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
    out["fdr_factor"] = pd.to_numeric(out["fdr_factor"], errors="coerce")
    write_parquet(out, PROC / "player_next5_fdr.parquet")
    log.info(
        "player_next5_fdr (GW%d-%d): %d players, factor range %.3f-%.3f",
        current_event,
        current_event + horizon - 1,
        len(out),
        out["fdr_factor"].min(),
        out["fdr_factor"].max(),
    )
    return out
