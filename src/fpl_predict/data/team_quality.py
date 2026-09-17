"""Team quality scores from historical match results.

Keyed on canonical team slugs and scoped to the seasons actually on disk, rather than to a
hardcoded pair of season numbers and a hardcoded list of last summer's promoted clubs.
Newly promoted clubs get the score implied by the bottom of the previous table instead of a
fixed constant, so the ranking does not silently rot each August.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from ..config import current_season
from ..utils.cache import PROC, RAW
from ..utils.io import read_parquet, write_parquet
from ..utils.logging import get_logger
from .teams import canonical_team

log = get_logger(__name__)

# Seasons of results used to judge quality. Two is enough to smooth out a single bad year
# without carrying a squad that no longer exists.
QUALITY_SEASONS = 2

# Quality is normalised into this band, with 1.0 as league average.
MIN_SCORE = 0.5
MAX_SCORE = 1.5


def calculate_team_quality_from_data(seasons: int = QUALITY_SEASONS) -> Dict[str, float]:
    """
    Team quality scores from stored match results, keyed on canonical slug.

    Returns:
        Dict mapping team slug to a quality score in [0.5, 1.5], 1.0 being average.
    """
    log.info("Calculating team quality from historical match data...")

    try:
        matches = read_parquet(RAW / "football-data" / "EPL_all_matches.parquet")
    except Exception as e:
        log.warning("No stored matches to judge team quality (%s)", e)
        return {}

    if matches.empty or "season" not in matches.columns:
        log.warning("Stored matches carry no season column; cannot judge team quality")
        return {}

    # Take the most recent completed seasons present, whatever they happen to be.
    available = sorted(int(s) for s in matches["season"].dropna().unique())
    completed = [s for s in available if s < current_season()]
    use = completed[-seasons:] if completed else available[-seasons:]
    recent = matches[matches["season"].isin(use)].copy()

    if recent.empty:
        log.warning("No matches in the selected seasons; cannot judge team quality")
        return {}

    if "home_slug" not in recent.columns:
        recent["home_slug"] = recent["home_team"].map(canonical_team)
    if "away_slug" not in recent.columns:
        recent["away_slug"] = recent["away_team"].map(canonical_team)
    recent = recent.dropna(subset=["home_goals", "away_goals"])

    log.info(
        "Analysing %d matches from %s",
        len(recent),
        ", ".join(str(s) for s in use),
    )

    # One row per team per match, so home and away are handled by the same arithmetic.
    home = pd.DataFrame(
        {
            "slug": recent["home_slug"],
            "gf": recent["home_goals"],
            "ga": recent["away_goals"],
        }
    )
    away = pd.DataFrame(
        {
            "slug": recent["away_slug"],
            "gf": recent["away_goals"],
            "ga": recent["home_goals"],
        }
    )
    long = pd.concat([home, away], ignore_index=True).dropna(subset=["slug"])
    long["points"] = (long.gf > long.ga) * 3 + (long.gf == long.ga) * 1

    agg = long.groupby("slug").agg(
        games=("points", "size"),
        points_per_game=("points", "mean"),
        goals_per_game=("gf", "mean"),
        conceded_per_game=("ga", "mean"),
    )
    agg["gd_per_game"] = agg.goals_per_game - agg.conceded_per_game

    # Composite: half points, a third goal difference, the rest goals scored.
    agg["raw"] = (
        0.5 * (agg.points_per_game / 3.0)
        + 0.3 * ((agg.gd_per_game + 2).clip(0, 4) / 4)
        + 0.2 * (agg.goals_per_game / 3.0).clip(upper=1.0)
    )

    lo, hi = agg.raw.min(), agg.raw.max()
    if hi > lo:
        agg["score"] = MIN_SCORE + (agg.raw - lo) / (hi - lo) * (MAX_SCORE - MIN_SCORE)
    else:
        agg["score"] = 1.0

    scores = agg.score.to_dict()

    # Clubs in the league now with no recent top-flight results are newly promoted. Give them
    # the average of the bottom three rather than a fixed constant, so the figure tracks how
    # weak the bottom of the table actually was.
    promoted_score = float(agg.score.nsmallest(3).mean()) if len(agg) >= 3 else MIN_SCORE
    for slug in _current_league_slugs():
        if slug and slug not in scores:
            scores[slug] = promoted_score
            log.info("%s has no recent top-flight results; scored %.3f", slug, promoted_score)

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    log.info("Team quality, best: %s", ", ".join(f"{t} {s:.2f}" for t, s in ranked[:5]))
    log.info("Team quality, worst: %s", ", ".join(f"{t} {s:.2f}" for t, s in ranked[-5:]))

    write_parquet(
        pd.DataFrame(
            [
                {"team": slug, "quality_score": score, "data_driven": True}
                for slug, score in scores.items()
            ]
        ),
        PROC / "team_quality.parquet",
    )
    log.info("Calculated quality scores for %d teams", len(scores))
    return scores


def _current_league_slugs() -> set[str]:
    """Canonical slugs of the clubs in the league right now."""
    try:
        from .fpl_api import get_bootstrap
        from .teams import bootstrap_team_slugs

        return set(bootstrap_team_slugs(get_bootstrap()).values())
    except Exception as e:
        log.warning("Could not read the current league from the API: %s", e)
        return set()


def get_team_quality_scores(refresh: bool = False) -> Dict[str, float]:
    """
    Team quality scores, from cache unless `refresh` is set.

    Returns:
        Dict mapping team slug to a quality score in [0.5, 1.5].
    """
    if not refresh:
        try:
            path = PROC / "team_quality.parquet"
            if path.exists():
                df = read_parquet(path)
                scores = dict(zip(df["team"], df["quality_score"]))
                log.info("Loaded cached team quality scores for %d teams", len(scores))
                return scores
        except Exception as e:
            log.warning("Could not load cached team quality: %s", e)

    return calculate_team_quality_from_data()


def get_team_tier(team_name: str, quality_scores: Dict[str, float] | None = None) -> int:
    """
    Team tier from the data-driven quality score. Accepts any spelling of the club name.

    Returns:
        Tier: 0 = weak/promoted, 1 = lower-mid, 2 = upper-mid, 3 = top
    """
    if quality_scores is None:
        quality_scores = get_team_quality_scores()

    score = quality_scores.get(canonical_team(team_name))
    if score is None:
        # Also accept a dict that is still keyed on display names.
        score = quality_scores.get(team_name, 1.0)

    if score >= 1.3:
        return 3
    if score >= 1.1:
        return 2
    if score >= 0.8:
        return 1
    return 0


if __name__ == "__main__":
    scores = calculate_team_quality_from_data()
    print(f"Calculated quality scores for {len(scores)} teams")
