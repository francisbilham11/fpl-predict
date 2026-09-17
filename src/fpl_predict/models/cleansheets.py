from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Tuple
from sklearn.model_selection import cross_val_score

from ..data.fpl_api import get_bootstrap
from ..data.teams import bootstrap_team_slugs, canonical_team
from ..utils.logging import get_logger

log = get_logger(__name__)

# Matches per team used to measure recent defensive form.
RECENT_MATCH_WINDOW = 10

# Fallback clean-sheet rates when a team has no stored matches, e.g. a promoted club.
DEFAULT_CS_HOME = 0.30
DEFAULT_CS_AWAY = 0.22


class CleanSheetModel:
    """
    Advanced clean sheet prediction model using:
    - Expected goals (xG) based probability
    - Team defensive strength metrics
    - Opposition attacking strength
    - Home/away factors
    - Recent defensive form
    """

    def __init__(self):
        self.model = None
        self.team_defensive_stats = {}
        self.fitted = False

    def _calculate_team_xg_stats(self, fixtures_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calculate xG-based team statistics for clean sheet prediction, keyed on canonical
        team slug.

        Keying matters: `predict_player_cs_probability` receives the FPL numeric team id and
        used to look it up against a dict keyed on team *names*, so the lookup never hit and
        every player fell through to a flat positional default. Both sides now agree on the
        slug from `data.teams`.
        """
        team_stats: Dict[str, Dict[str, float]] = {}

        df = fixtures_df.copy()
        if "home_slug" not in df.columns:
            df["home_slug"] = df["home_team"].map(canonical_team)
        if "away_slug" not in df.columns:
            df["away_slug"] = df["away_team"].map(canonical_team)

        # A window of matches per team rather than a window of days. At the start of a season
        # "the last 70 days" is either empty or a slice of last season's run-in, whereas the
        # last N matches per team always resolves to something usable.
        recent = df.dropna(subset=["date"]).sort_values("date")

        for team in pd.concat([recent["home_slug"], recent["away_slug"]]).dropna().unique():
            if not team:
                continue

            home_games = recent[recent["home_slug"] == team].tail(RECENT_MATCH_WINDOW)
            away_games = recent[recent["away_slug"] == team].tail(RECENT_MATCH_WINDOW)

            # xG conceded (defensive strength)
            home_xg_against = (
                home_games["away_xg"].mean()
                if "away_xg" in home_games and len(home_games) > 0
                else home_games["away_goals"].mean()
                if len(home_games) > 0
                else 1.2
            )
            away_xg_against = (
                away_games["home_xg"].mean()
                if "home_xg" in away_games and len(away_games) > 0
                else away_games["home_goals"].mean()
                if len(away_games) > 0
                else 1.5
            )

            # Clean sheets achieved
            home_cs = (home_games["away_goals"] == 0).mean() if len(home_games) > 0 else 0.3
            away_cs = (away_games["home_goals"] == 0).mean() if len(away_games) > 0 else 0.2

            # Goals conceded
            home_goals_against = home_games["away_goals"].mean() if len(home_games) > 0 else 1.2
            away_goals_against = away_games["home_goals"].mean() if len(away_games) > 0 else 1.5

            # xG for (attacking strength - affects opponent CS probability)
            home_xg_for = (
                home_games["home_xg"].mean()
                if "home_xg" in home_games and len(home_games) > 0
                else home_games["home_goals"].mean()
                if len(home_games) > 0
                else 1.5
            )
            away_xg_for = (
                away_games["away_xg"].mean()
                if "away_xg" in away_games and len(away_games) > 0
                else away_games["away_goals"].mean()
                if len(away_games) > 0
                else 1.0
            )

            team_stats[team] = {
                "xg_against_home": float(home_xg_against),
                "xg_against_away": float(away_xg_against),
                "xg_for_home": float(home_xg_for),
                "xg_for_away": float(away_xg_for),
                "cs_rate_home": float(home_cs),
                "cs_rate_away": float(away_cs),
                "goals_against_home": float(home_goals_against),
                "goals_against_away": float(away_goals_against),
                "defensive_rating": 1.0 / (1.0 + (home_xg_against + away_xg_against) / 2),
                "attacking_rating": (home_xg_for + away_xg_for) / 2,
            }

        return team_stats

    def _poisson_clean_sheet_prob(self, lambda_goals: float) -> float:
        """
        Calculate clean sheet probability using Poisson distribution.
        P(CS) = P(goals_conceded = 0) = exp(-lambda)
        """
        return np.exp(-lambda_goals)

    def _dixon_coles_adjustment(self, xg_for: float, xg_against: float) -> float:
        """
        Apply Dixon-Coles adjustment for low-scoring matches.
        """
        # Adjustment factor for 0-0 scores
        rho = -0.15  # Typical value from research

        # Calculate base probabilities
        p_0_for = np.exp(-xg_for)
        p_0_against = np.exp(-xg_against)

        # Adjusted probability of 0-0
        p_0_0 = p_0_for * p_0_against * (1 - xg_for * xg_against * rho)

        return p_0_0

    def predict_match_cs_probability(
        self, home_team: str, away_team: str, use_dixon_coles: bool = True
    ) -> Tuple[float, float]:
        """
        Predict clean sheet probabilities for both teams in a match.

        Returns:
            (home_cs_prob, away_cs_prob)
        """
        if not self.team_defensive_stats:
            return 0.35, 0.25  # Default probabilities

        home_stats = self.team_defensive_stats.get(home_team, {})
        away_stats = self.team_defensive_stats.get(away_team, {})

        # Get xG values with defaults
        home_xg_for = home_stats.get("xg_for_home", 1.5)
        home_xg_against = home_stats.get("xg_against_home", 1.2)
        away_xg_for = away_stats.get("xg_for_away", 1.0)
        away_xg_against = away_stats.get("xg_against_away", 1.5)

        # Adjust for opponent strength
        home_expected_goals_against = (home_xg_against + away_xg_for) / 2
        away_expected_goals_against = (away_xg_against + home_xg_for) / 2

        if use_dixon_coles:
            home_cs_prob = self._dixon_coles_adjustment(away_xg_for, home_xg_against)
            away_cs_prob = self._dixon_coles_adjustment(home_xg_for, away_xg_against)
        else:
            home_cs_prob = self._poisson_clean_sheet_prob(home_expected_goals_against)
            away_cs_prob = self._poisson_clean_sheet_prob(away_expected_goals_against)

        # Apply historical clean sheet rate as Bayesian prior
        home_cs_rate = home_stats.get("cs_rate_home", 0.35)
        away_cs_rate = away_stats.get("cs_rate_away", 0.25)

        # Weighted average with prior (more weight on model as we have more data)
        weight = 0.7  # Model weight
        home_cs_prob = weight * home_cs_prob + (1 - weight) * home_cs_rate
        away_cs_prob = weight * away_cs_prob + (1 - weight) * away_cs_rate

        return float(home_cs_prob), float(away_cs_prob)

    def create_features_for_player(self, player_row: pd.Series) -> pd.DataFrame:
        """
        Create features for clean sheet prediction for a player.
        """
        features = pd.DataFrame()

        if not self.team_defensive_stats:
            return pd.DataFrame({"cs_prob": [0.3]})

        team = player_row.get("team")
        position = player_row.get("position")

        if pd.isna(team) or team not in self.team_defensive_stats:
            return pd.DataFrame({"cs_prob": [0.3 if position in ["DEF", "GKP"] else 0.0]})

        team_stats = self.team_defensive_stats[team]

        # Base features
        features["defensive_rating"] = [team_stats.get("defensive_rating", 0.5)]
        features["xg_against_avg"] = [
            (team_stats.get("xg_against_home", 1.2) + team_stats.get("xg_against_away", 1.5)) / 2
        ]
        features["cs_rate_avg"] = [
            (team_stats.get("cs_rate_home", 0.35) + team_stats.get("cs_rate_away", 0.25)) / 2
        ]

        # Position-specific adjustments
        if position == "GKP":
            features["position_factor"] = [1.0]
        elif position == "DEF":
            features["position_factor"] = [1.0]
        elif position == "MID":
            features["position_factor"] = [0.25]  # Midfielders get 1 point for CS
        else:
            features["position_factor"] = [0.0]  # Forwards don't get CS points

        return features

    def fit(self, fixtures_df: pd.DataFrame) -> None:
        """
        Fit the clean sheet model on historical fixture data.
        """
        try:
            self.team_defensive_stats = self._calculate_team_xg_stats(fixtures_df)

            # Optional: train an XGBoost model if we have enough data
            if len(fixtures_df) > 100:
                # Prepare training data
                X_train = []
                y_train = []

                slugged = fixtures_df.copy()
                if "home_slug" not in slugged.columns:
                    slugged["home_slug"] = slugged["home_team"].map(canonical_team)
                if "away_slug" not in slugged.columns:
                    slugged["away_slug"] = slugged["away_team"].map(canonical_team)

                for _, match in slugged.iterrows():
                    home_team = match["home_slug"]
                    away_team = match["away_slug"]

                    if (
                        home_team in self.team_defensive_stats
                        and away_team in self.team_defensive_stats
                    ):
                        # Features for home team CS
                        home_feat = [
                            self.team_defensive_stats[home_team].get("defensive_rating", 0.5),
                            self.team_defensive_stats[away_team].get("attacking_rating", 1.2),
                            self.team_defensive_stats[home_team].get("cs_rate_home", 0.35),
                            1,  # Is home
                        ]
                        X_train.append(home_feat)
                        y_train.append(1 if match["away_goals"] == 0 else 0)

                        # Features for away team CS
                        away_feat = [
                            self.team_defensive_stats[away_team].get("defensive_rating", 0.5),
                            self.team_defensive_stats[home_team].get("attacking_rating", 1.5),
                            self.team_defensive_stats[away_team].get("cs_rate_away", 0.25),
                            0,  # Is away
                        ]
                        X_train.append(away_feat)
                        y_train.append(1 if match["home_goals"] == 0 else 0)

                if len(X_train) > 50:
                    X_train = pd.DataFrame(
                        X_train,
                        columns=["def_rating", "opp_att_rating", "historical_cs", "is_home"],
                    )
                    y_train = pd.Series(y_train)

                    self.model = xgb.XGBClassifier(
                        n_estimators=50,
                        max_depth=4,
                        learning_rate=0.1,
                        objective="binary:logistic",
                        random_state=42,
                        verbosity=0,
                    )
                    self.model.fit(X_train, y_train)

                    # Evaluate model
                    cv_scores = cross_val_score(
                        self.model, X_train, y_train, cv=3, scoring="neg_log_loss"
                    )
                    log.info(f"Clean sheet model CV log loss: {-cv_scores.mean():.3f}")

            self.fitted = True
            log.info(f"Clean sheet model fitted with {len(self.team_defensive_stats)} teams")

        except Exception as e:
            log.warning(f"Failed to fit clean sheet model: {e}")
            self.fitted = False

    # Share of a clean sheet's value that reaches each position, reflecting that a
    # midfielder earns 1 point to a defender's 4.
    _POSITION_SHARE = {"GKP": 1.0, "DEF": 1.0, "MID": 0.3, "FWD": 0.0}

    def predict_player_cs_probability(self, features: pd.DataFrame) -> pd.Series:
        """
        Clean sheet probability per player, resolved via the player's team slug.

        `features['team']` holds the FPL numeric team id, so it is translated through the
        bootstrap before the lookup. The previous version compared that integer against a
        dict keyed on team names, missed every time, and returned a flat 0.35 for every
        defender in the league.
        """
        positions = features.get("position", pd.Series("", index=features.index)).astype(str)
        share = positions.map(self._POSITION_SHARE).fillna(0.0)

        if not self.fitted or "team" not in features.columns or not self.team_defensive_stats:
            log.warning("Clean sheet model unfitted; falling back to league-average rates")
            base = (DEFAULT_CS_HOME + DEFAULT_CS_AWAY) / 2
            return share * base

        try:
            id_to_slug = bootstrap_team_slugs(get_bootstrap())
        except Exception as e:
            log.warning("Could not map team ids to slugs (%s); using league-average rates", e)
            base = (DEFAULT_CS_HOME + DEFAULT_CS_AWAY) / 2
            return share * base

        team_cs = {
            slug: (
                stats.get("cs_rate_home", DEFAULT_CS_HOME)
                + stats.get("cs_rate_away", DEFAULT_CS_AWAY)
            )
            / 2
            for slug, stats in self.team_defensive_stats.items()
        }
        league_avg = (
            float(np.mean(list(team_cs.values())))
            if team_cs
            else (DEFAULT_CS_HOME + DEFAULT_CS_AWAY) / 2
        )

        slugs = features["team"].map(id_to_slug)
        rates = slugs.map(team_cs)
        n_missing = int(rates.isna().sum())
        if n_missing:
            log.info(
                "%d players on teams with no stored matches; using the league average %.3f",
                n_missing,
                league_avg,
            )
        rates = rates.fillna(league_avg)

        return (rates * share).astype(float)
