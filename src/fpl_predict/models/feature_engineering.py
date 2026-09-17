from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional
from ..data.fpl_api import get_bootstrap
from ..data.teams import bootstrap_team_slugs, canonical_team
from ..utils.cache import PROC
from ..utils.io import read_parquet
from ..utils.logging import get_logger

log = get_logger(__name__)

# Matches per team used for recent-form features.
RECENT_MATCH_WINDOW = 10


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for FPL predictions including:
    - Opposition strength metrics
    - Home/away performance splits
    - Form trends and momentum
    - Fixture difficulty adjustments
    - Team tactical features
    - Player interaction effects
    """

    def __init__(self):
        self.team_stats = {}
        self.player_stats = {}
        self.fitted = False
        self._slug_cache: Optional[Dict[int, str]] = None

    def _calculate_team_strength(self, fixtures_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Team offensive and defensive strength over recent matches, keyed on canonical slug.

        `transform` maps these onto players via `features['team']`, which is the FPL numeric
        team id, so both sides have to agree on the key. They did not: this dict was keyed on
        team names, the lookup always missed, and `team_offensive_strength_normalized`,
        `team_defensive_strength_normalized` and `overall_strength` were the constant 1.0 for
        every player in the league.
        """
        team_stats: Dict[str, Dict[str, float]] = {}

        df = fixtures_df.copy()
        if "home_slug" not in df.columns:
            df["home_slug"] = df["home_team"].map(canonical_team)
        if "away_slug" not in df.columns:
            df["away_slug"] = df["away_team"].map(canonical_team)

        # Last N matches per team rather than a 70-day window, which is empty before a season
        # starts.
        recent_fixtures = df.dropna(subset=["date"]).sort_values("date")

        for team in (
            pd.concat([recent_fixtures["home_slug"], recent_fixtures["away_slug"]])
            .dropna()
            .unique()
        ):
            if not team:
                continue

            home_games = recent_fixtures[recent_fixtures["home_slug"] == team].tail(
                RECENT_MATCH_WINDOW
            )
            away_games = recent_fixtures[recent_fixtures["away_slug"] == team].tail(
                RECENT_MATCH_WINDOW
            )

            # Offensive strength
            home_goals_for = home_games["home_goals"].mean() if len(home_games) > 0 else 1.5
            away_goals_for = away_games["away_goals"].mean() if len(away_games) > 0 else 1.0

            # Defensive strength (goals conceded)
            home_goals_against = home_games["away_goals"].mean() if len(home_games) > 0 else 1.0
            away_goals_against = away_games["home_goals"].mean() if len(away_games) > 0 else 1.5

            # Expected goals if available
            home_xg = (
                home_games.get("home_xg", pd.Series()).mean()
                if "home_xg" in home_games
                else home_goals_for
            )
            away_xg = (
                away_games.get("away_xg", pd.Series()).mean()
                if "away_xg" in away_games
                else away_goals_for
            )

            team_stats[team] = {
                "offensive_strength_home": float(home_goals_for),
                "offensive_strength_away": float(away_goals_for),
                "defensive_strength_home": float(home_goals_against),
                "defensive_strength_away": float(away_goals_against),
                "xg_home": float(home_xg),
                "xg_away": float(away_xg),
                "overall_strength": float(
                    (home_goals_for + away_goals_for - home_goals_against - away_goals_against) / 4
                ),
            }

        # Normalize strengths relative to league average
        if team_stats:
            avg_off = (
                np.mean(
                    [
                        s["offensive_strength_home"] + s["offensive_strength_away"]
                        for s in team_stats.values()
                    ]
                )
                / 2
            )
            avg_def = (
                np.mean(
                    [
                        s["defensive_strength_home"] + s["defensive_strength_away"]
                        for s in team_stats.values()
                    ]
                )
                / 2
            )

            for team in team_stats:
                team_stats[team]["offensive_strength_normalized"] = (
                    (
                        (
                            team_stats[team]["offensive_strength_home"]
                            + team_stats[team]["offensive_strength_away"]
                        )
                        / 2
                    )
                    / avg_off
                    if avg_off > 0
                    else 1.0
                )

                team_stats[team]["defensive_strength_normalized"] = (
                    (
                        (
                            team_stats[team]["defensive_strength_home"]
                            + team_stats[team]["defensive_strength_away"]
                        )
                        / 2
                    )
                    / avg_def
                    if avg_def > 0
                    else 1.0
                )

        return team_stats

    def _calculate_player_home_away_splits(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate player performance splits for home vs away games.
        """
        if "is_home" not in events_df.columns:
            # Infer from fixture data if possible
            return pd.DataFrame()

        splits = (
            events_df.groupby(["player_id", "is_home"])
            .agg(
                {
                    "minutes": "mean",
                    "goals": "sum",
                    "assists": "sum",
                    "clean_sheets": "sum",
                    "goals_conceded": "sum",
                    "yellow_cards": "sum",
                    "red_cards": "sum",
                    "bonus": "sum",
                    "total_points": "mean",
                }
            )
            .unstack(fill_value=0)
        )

        # Calculate ratios
        player_splits = pd.DataFrame(index=splits.index)

        for metric in ["minutes", "goals", "assists", "total_points"]:
            if metric in splits.columns.get_level_values(0):
                home_val = splits[(metric, True)]
                away_val = splits[(metric, False)]
                total = home_val + away_val
                player_splits[f"{metric}_home_ratio"] = np.where(
                    total > 0,
                    home_val / total,
                    0.5,  # Default to neutral if no data
                )

        return player_splits

    def _calculate_form_momentum(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced form and momentum indicators.
        """
        momentum_features = pd.DataFrame(index=features.index)

        # Form trend (is form improving or declining?)
        if "mins_l3" in features and "mins_l5" in features:
            momentum_features["mins_trend"] = (
                features["mins_l3"] / features["mins_l5"].replace(0, 1)
            ).clip(0, 2)

        # Consistency score (lower variance is better)
        if "form" in features and "selected_by_percent" in features:
            momentum_features["consistency_score"] = features["form"] * np.log1p(
                features["selected_by_percent"]
            )

        # Recent starts momentum
        if "starts_l5" in features:
            momentum_features["starter_confidence"] = features["starts_l5"] / 5.0

        # Fitness and availability trend
        if "chance_next" in features:
            momentum_features["fitness_factor"] = features["chance_next"] / 100.0

        return momentum_features

    def _calculate_fixture_context(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Add context about upcoming fixtures.
        """
        context_features = pd.DataFrame(index=features.index)

        try:
            # Load fixture difficulty ratings
            fdr_df = read_parquet(PROC / "player_next5_fdr.parquet")

            if "player_id" in features.columns and "player_id" in fdr_df.columns:
                fdr_map = dict(zip(fdr_df["player_id"], fdr_df["fdr_factor"]))
                context_features["next5_fdr"] = features["player_id"].map(fdr_map).fillna(1.0)

                # Categorize fixture difficulty
                context_features["easy_fixtures"] = (context_features["next5_fdr"] < 0.9).astype(
                    int
                )
                context_features["hard_fixtures"] = (context_features["next5_fdr"] > 1.1).astype(
                    int
                )
        except Exception as e:
            log.debug(f"Could not load FDR data: {e}")
            context_features["next5_fdr"] = 1.0
            context_features["easy_fixtures"] = 0
            context_features["hard_fixtures"] = 0

        return context_features

    def _calculate_interaction_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate interaction effects between features.
        """
        interactions = pd.DataFrame(index=features.index)

        # Price-performance interaction
        if "now_cost" in features and "form" in features:
            interactions["value_score"] = features["form"] / (features["now_cost"] / 10).clip(
                lower=4.0
            )

        # Minutes-form interaction
        if "mins_l5" in features and "form" in features:
            interactions["active_form"] = features["mins_l5"] * features["form"] / 90

        # Position-specific interactions
        if "position" in features:
            is_def = features["position"].isin(["DEF", "GKP"])
            is_mid = features["position"] == "MID"
            is_fwd = features["position"] == "FWD"

            if "prev_xg_per90" in features:
                interactions["attacking_threat"] = np.where(
                    is_fwd,
                    features["prev_xg_per90"] * 1.2,
                    np.where(
                        is_mid, features["prev_xg_per90"] * 1.0, features["prev_xg_per90"] * 0.8
                    ),
                )

            if "prev_xa_per90" in features:
                interactions["creative_threat"] = np.where(
                    is_mid, features["prev_xa_per90"] * 1.2, features["prev_xa_per90"] * 1.0
                )

        return interactions

    def _calculate_team_form(
        self, features: pd.DataFrame, fixtures_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate team-level form metrics.
        """
        team_form = pd.DataFrame(index=features.index)

        if "team" not in features.columns:
            return team_form

        df = fixtures_df.copy()
        if "home_slug" not in df.columns:
            df["home_slug"] = df["home_team"].map(canonical_team)
        if "away_slug" not in df.columns:
            df["away_slug"] = df["away_team"].map(canonical_team)
        recent_fixtures = df.dropna(subset=["date"]).sort_values("date")

        slugs = self._team_slugs()
        team_results = {}
        for team in features["team"].unique():
            if pd.isna(team):
                continue
            slug = slugs.get(team)
            if not slug:
                continue

            home_games = recent_fixtures[recent_fixtures["home_slug"] == slug].tail(5)
            away_games = recent_fixtures[recent_fixtures["away_slug"] == slug].tail(5)

            # Points from recent games (3 for win, 1 for draw)
            home_points = sum(
                [
                    3
                    if row["home_goals"] > row["away_goals"]
                    else 1
                    if row["home_goals"] == row["away_goals"]
                    else 0
                    for _, row in home_games.iterrows()
                ]
            )

            away_points = sum(
                [
                    3
                    if row["away_goals"] > row["home_goals"]
                    else 1
                    if row["away_goals"] == row["home_goals"]
                    else 0
                    for _, row in away_games.iterrows()
                ]
            )

            total_games = len(home_games) + len(away_games)
            team_results[team] = (home_points + away_points) / max(total_games, 1)

        team_form["team_form"] = (
            features["team"].map(team_results).fillna(1.5)
        )  # Default to mid-table form

        return team_form

    def _team_slugs(self) -> Dict[int, str]:
        """FPL team id -> canonical slug, cached for the life of the engineer."""
        if self._slug_cache is None:
            try:
                self._slug_cache = bootstrap_team_slugs(get_bootstrap())
            except Exception as e:
                log.warning("Could not map team ids to slugs: %s", e)
                self._slug_cache = {}
        return self._slug_cache

    def fit(self, features: pd.DataFrame, fixtures_df: pd.DataFrame = None) -> None:
        """
        Fit the feature engineer on historical data.
        """
        try:
            if fixtures_df is None:
                fixtures_df = read_parquet(PROC / "fixtures.parquet")

            self.team_stats = self._calculate_team_strength(fixtures_df)
            self.fitted = True
            log.info("Feature engineer fitted with %d teams", len(self.team_stats))
        except Exception as e:
            log.warning(f"Failed to fit feature engineer: {e}")
            self.fitted = False

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features with advanced engineering.
        """
        enhanced_features = features.copy()

        # Add team strength features, translating the FPL team id into the slug the stats
        # are keyed on.
        if self.team_stats and "team" in features.columns:
            slugs = self._team_slugs()
            team_slug = features["team"].map(slugs)
            for metric in [
                "offensive_strength_normalized",
                "defensive_strength_normalized",
                "overall_strength",
            ]:
                enhanced_features[f"team_{metric}"] = team_slug.map(
                    lambda s: (
                        self.team_stats.get(s, {}).get(metric, 1.0)
                        if isinstance(s, str) and s
                        else 1.0
                    )
                )
            matched = int(team_slug.map(lambda s: s in self.team_stats).sum())
            log.info(
                "Team strength features attached for %d/%d players", matched, len(enhanced_features)
            )

        # Add momentum features
        momentum_df = self._calculate_form_momentum(features)
        for col in momentum_df.columns:
            enhanced_features[col] = momentum_df[col]

        # Add fixture context
        context_df = self._calculate_fixture_context(features)
        for col in context_df.columns:
            enhanced_features[col] = context_df[col]

        # Add interaction features
        interaction_df = self._calculate_interaction_features(features)
        for col in interaction_df.columns:
            enhanced_features[col] = interaction_df[col]

        # Try to add team form if fixtures available
        try:
            fixtures_df = read_parquet(PROC / "fixtures.parquet")
            team_form_df = self._calculate_team_form(features, fixtures_df)
            for col in team_form_df.columns:
                enhanced_features[col] = team_form_df[col]
        except Exception:
            pass

        # Fill any NaN values with sensible defaults
        numeric_columns = enhanced_features.select_dtypes(include=[np.number]).columns
        enhanced_features[numeric_columns] = enhanced_features[numeric_columns].fillna(0)

        return enhanced_features

    def fit_transform(
        self, features: pd.DataFrame, fixtures_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.
        """
        self.fit(features, fixtures_df)
        return self.transform(features)
