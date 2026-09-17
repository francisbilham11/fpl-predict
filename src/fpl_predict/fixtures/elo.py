"""Team strength as Elo ratings over historical results.

Two details matter more than the update rule itself:

- **Ratings must be keyed on canonical team slugs.** The previous version keyed on
  whatever spelling the source used, so football-data's "Manchester City FC" and the FPL
  API's "Man City" built two separate ratings and neither saw the other's matches. Callers
  pass slugs from `data.teams.canonical_team`.
- **Ratings regress toward the mean across a season boundary.** Squads turn over every
  summer, so carrying a full rating from May into August overstates how much last season
  tells us. Newly-arrived clubs enter on a promoted-side prior rather than the league mean,
  which is what stops a promoted team looking like a mid-table one in GW1.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

HOME_ADV = 65.0
K = 20.0
MEAN_RATING = 1500.0

# Fraction of a rating's distance from the mean that survives the summer.
SEASON_CARRY = 0.75

# How far below the league mean a promoted club starts, in Elo points. Roughly the gap
# between a bottom-three side and mid-table.
PROMOTED_PENALTY = 120.0


def expected_score(rating_a: float, rating_b: float, home: bool) -> float:
    ra = rating_a + (HOME_ADV if home else 0.0)
    return 1.0 / (1.0 + 10 ** (-(ra - rating_b) / 400))


def _regress(ratings: Dict[str, float], carry: float) -> None:
    for team, r in list(ratings.items()):
        ratings[team] = MEAN_RATING + carry * (r - MEAN_RATING)


def update_elo(
    df_matches: pd.DataFrame,
    k: float = K,
    season_carry: float = SEASON_CARRY,
    promoted_penalty: float = PROMOTED_PENALTY,
) -> Dict[str, float]:
    """Elo ratings from played matches, oldest first.

    Expects columns `home_team`, `away_team`, `home_goals`, `away_goals`, `date`, and
    optionally `season`. When `season` is present, ratings regress toward the mean at each
    season change and clubs appearing for the first time in a season start below the mean.
    """
    ratings: Dict[str, float] = {}
    current_season: Optional[int] = None
    seasons_seen = 0
    has_season = "season" in df_matches.columns

    df = df_matches.dropna(subset=["home_team", "away_team"])
    sort_cols = ["season", "date"] if has_season else ["date"]
    df = df.sort_values([c for c in sort_cols if c in df.columns], kind="stable")

    for m in df.itertuples():
        if has_season:
            season = getattr(m, "season", None)
            if season != current_season:
                if current_season is not None:
                    _regress(ratings, season_carry)
                current_season = season
                seasons_seen += 1

        h, a = m.home_team, m.away_team
        for team in (h, a):
            if team not in ratings:
                # A club first seen in the earliest season we hold has no history either
                # way, so it starts at the mean. A club first seen later has been promoted
                # into a division we already have ratings for, and starts below it.
                promoted = seasons_seen > 1
                ratings[team] = MEAN_RATING - (promoted_penalty if promoted else 0.0)

        hg = getattr(m, "home_goals", None)
        ag = getattr(m, "away_goals", None)
        if pd.isna(hg) or pd.isna(ag):
            continue
        hg, ag = float(hg), float(ag)

        rh, ra = ratings[h], ratings[a]
        exp_h = expected_score(rh, ra, home=True)
        if hg > ag:
            out = 1.0
        elif hg < ag:
            out = 0.0
        else:
            out = 0.5

        margin = abs(hg - ag)
        kk = k * (1 + 0.1 * margin)
        ratings[h] = rh + kk * (out - exp_h)
        ratings[a] = ra + kk * ((1 - out) - (1 - exp_h))

    return ratings


def match_counts(df_matches: pd.DataFrame) -> Dict[str, int]:
    """Matches per team in the frame, used to decide how far to trust a rating."""
    counts: Dict[str, int] = {}
    played = df_matches.dropna(subset=["home_goals", "away_goals"])
    for col in ("home_team", "away_team"):
        for team, n in played[col].value_counts().items():
            counts[team] = counts.get(team, 0) + int(n)
    return counts
