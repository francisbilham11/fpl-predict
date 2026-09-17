"""
Build proper training targets from actual match performance data.

This module aggregates match-level data from FPL API to create meaningful
training targets for goals, assists, minutes, etc.
"""

from __future__ import annotations

import pandas as pd

from ..config import current_season, season_label
from ..utils.cache import PROC
from ..utils.io import read_parquet, write_parquet
from ..utils.logging import get_logger
from .fpl_api import get_bootstrap, get_element_summary

log = get_logger(__name__)


MATCH_DATA_COLS = [
    "player_id",
    "season",
    "gameweek",
    "minutes",
    "goals",
    "assists",
    "total_points",
    "clean_sheets",
    "goals_conceded",
    "bonus",
    "bps",
    "saves",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "expected_goals",
    "expected_assists",
    "defensive_contribution",
    "was_home",
    "starts",
    "fixture_id",
    "opponent_team",
]


def fetch_comprehensive_match_data() -> pd.DataFrame:
    """
    Fetch match-level data for all players in the current season from the FPL API.

    Returns an empty frame with the full schema before the season's first match, which is
    the normal state in August. Previously this built `pd.DataFrame([])` and then read
    `df['player_id']`, raising KeyError and taking the whole training build down with it.
    """
    log.info("Fetching comprehensive match data from FPL API...")

    bootstrap = get_bootstrap()
    elements = bootstrap.get("elements", [])
    season = season_label(current_season())

    all_matches = []

    for i, element in enumerate(elements):
        if i % 100 == 0:
            log.info(f"Processing player {i + 1}/{len(elements)}")

        player_id = element["id"]

        try:
            summary = get_element_summary(player_id)

            # Current season, game by game
            current_history = summary.get("history", [])
            for game in current_history:
                game_data = {
                    "player_id": player_id,
                    "season": season,
                    "gameweek": game.get("round", 0),
                    "minutes": game.get("minutes", 0),
                    "goals": game.get("goals_scored", 0),
                    "assists": game.get("assists", 0),
                    "total_points": game.get("total_points", 0),
                    "clean_sheets": game.get("clean_sheets", 0),
                    "goals_conceded": game.get("goals_conceded", 0),
                    "bonus": game.get("bonus", 0),
                    "bps": game.get("bps", 0),
                    "saves": game.get("saves", 0),
                    "penalties_saved": game.get("penalties_saved", 0),
                    "penalties_missed": game.get("penalties_missed", 0),
                    "yellow_cards": game.get("yellow_cards", 0),
                    "red_cards": game.get("red_cards", 0),
                    "own_goals": game.get("own_goals", 0),
                    "expected_goals": float(game.get("expected_goals", 0)),
                    "expected_assists": float(game.get("expected_assists", 0)),
                    "defensive_contribution": game.get("defensive_contribution", 0),
                    "was_home": game.get("was_home", False),
                    "starts": game.get("starts", 0),
                    "fixture_id": game.get("fixture", 0),
                    "opponent_team": game.get("opponent_team", 0),
                }
                all_matches.append(game_data)

        except Exception as e:
            if i < 10:  # Only log first few errors
                log.debug(f"Error fetching data for player {player_id}: {e}")
            continue

    if not all_matches:
        log.warning(
            "No %s match records yet (season has not started); returning an empty frame",
            season,
        )
        return pd.DataFrame(columns=MATCH_DATA_COLS)

    df = pd.DataFrame(all_matches)
    log.info(f"Collected {len(df)} match records for {df['player_id'].nunique()} players")

    return df


def aggregate_training_targets(match_data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate match data into per-player training targets.

    Creates targets suitable for machine learning models.
    """
    log.info("Aggregating match data into training targets...")

    if match_data.empty:
        log.warning("No match data to aggregate; training targets will be empty")
        return pd.DataFrame(
            columns=["player_id", "target_minutes", "target_goals", "target_assists"]
        )

    # Group by player and calculate aggregates
    player_stats = []

    for player_id, player_matches in match_data.groupby("player_id"):
        # Sort by gameweek to get recent form
        player_matches = player_matches.sort_values("gameweek")

        # Basic aggregates
        total_matches = len(player_matches)
        total_minutes = player_matches["minutes"].sum()
        total_goals = player_matches["goals"].sum()
        total_assists = player_matches["assists"].sum()
        total_starts = player_matches["starts"].sum()

        # Per-90 rates (only if meaningful minutes played)
        if total_minutes >= 45:  # At least half a game
            goals_per90 = (total_goals / total_minutes) * 90
            assists_per90 = (total_assists / total_minutes) * 90
            points_per90 = (player_matches["total_points"].sum() / total_minutes) * 90
        else:
            goals_per90 = assists_per90 = points_per90 = 0.0

        # Recent form (last 5 games)
        recent_5 = player_matches.tail(5)
        recent_minutes = recent_5["minutes"].sum()
        recent_goals = recent_5["goals"].sum()
        recent_assists = recent_5["assists"].sum()
        recent_points = recent_5["total_points"].sum()

        # Recent form (last 3 games)
        recent_3 = player_matches.tail(3)
        recent_3_minutes = recent_3["minutes"].sum()
        recent_3_goals = recent_3["goals"].sum()
        recent_3_assists = recent_3["assists"].sum()

        # Home/away splits
        home_matches = player_matches[player_matches["was_home"] == True]
        away_matches = player_matches[player_matches["was_home"] == False]

        home_goals = home_matches["goals"].sum() if len(home_matches) > 0 else 0
        away_goals = away_matches["goals"].sum() if len(away_matches) > 0 else 0
        home_assists = home_matches["assists"].sum() if len(home_matches) > 0 else 0
        away_assists = away_matches["assists"].sum() if len(away_matches) > 0 else 0

        # Expected stats
        total_xg = player_matches["expected_goals"].sum()
        total_xa = player_matches["expected_assists"].sum()

        # Defensive stats
        total_clean_sheets = player_matches["clean_sheets"].sum()
        total_goals_conceded = player_matches["goals_conceded"].sum()
        total_saves = player_matches["saves"].sum()

        # Form trend (recent vs earlier performance)
        if total_matches >= 3:
            early_matches = player_matches.head(max(1, total_matches - 3))
            recent_avg = recent_3["total_points"].mean() if len(recent_3) > 0 else 0
            early_avg = early_matches["total_points"].mean() if len(early_matches) > 0 else 0
            form_trend = recent_avg - early_avg
        else:
            form_trend = 0.0

        # Target values for next gameweek prediction
        # Use recent form (last 3 games) as proxy for next game performance
        if len(recent_3) > 0 and recent_3["minutes"].sum() > 0:
            target_minutes = recent_3["minutes"].mean()
            target_goals = recent_3_goals / max(1, len(recent_3))
            target_assists = recent_3_assists / max(1, len(recent_3))
        else:
            # Fallback to season averages
            target_minutes = total_minutes / max(1, total_matches)
            target_goals = total_goals / max(1, total_matches)
            target_assists = total_assists / max(1, total_matches)

        player_stats.append(
            {
                "player_id": player_id,
                # Training targets (what models should predict)
                "target_minutes": target_minutes,
                "target_goals": target_goals,
                "target_assists": target_assists,
                # Everything else here is diagnostic context for this aggregation, not a
                # feature to train on directly — it's prefixed `tgt_` because several of
                # these names (goals_l5, assists_l3, goals_per90, home_goals, form_trend, ...)
                # collide with features.parquet's own, differently-computed columns of the
                # same name. Merging two same-named columns makes pandas suffix both `_x`/`_y`,
                # which silently zeroed out the real features.parquet versions in the model's
                # input matrix (neither plain name existed anymore for _build_X_advanced's
                # XCOLS lookup to find). Namespacing here fixes that without touching the
                # legitimate features.parquet columns at all.
                # Season totals (for context)
                "tgt_season_minutes": total_minutes,
                "tgt_season_goals": total_goals,
                "tgt_season_assists": total_assists,
                "tgt_season_points": player_matches["total_points"].sum(),
                "tgt_season_matches": total_matches,
                "tgt_season_starts": total_starts,
                # Per-90 rates (for priors/validation)
                "tgt_goals_per90": goals_per90,
                "tgt_assists_per90": assists_per90,
                "tgt_points_per90": points_per90,
                # Recent form
                "tgt_goals_l5": recent_goals,
                "tgt_assists_l5": recent_assists,
                "tgt_points_l5": recent_points,
                "tgt_minutes_l5": recent_minutes,
                "tgt_goals_l3": recent_3_goals,
                "tgt_assists_l3": recent_3_assists,
                "tgt_minutes_l3": recent_3_minutes,
                # Expected stats
                "tgt_xg_total": total_xg,
                "tgt_xa_total": total_xa,
                "tgt_xg_per90": (total_xg / total_minutes * 90) if total_minutes > 0 else 0,
                "tgt_xa_per90": (total_xa / total_minutes * 90) if total_minutes > 0 else 0,
                # Home/away
                "tgt_home_goals": home_goals,
                "tgt_away_goals": away_goals,
                "tgt_home_assists": home_assists,
                "tgt_away_assists": away_assists,
                # Defensive
                "tgt_clean_sheets": total_clean_sheets,
                "tgt_goals_conceded": total_goals_conceded,
                "tgt_saves": total_saves,
                # Form trend
                "tgt_form_trend": form_trend,
            }
        )

    targets_df = pd.DataFrame(player_stats)
    log.info(f"Created training targets for {len(targets_df)} players")

    return targets_df


def merge_targets_with_features() -> pd.DataFrame:
    """
    Merge training targets with existing features to create complete training dataset.
    """
    log.info("Merging training targets with features...")

    # Load existing features
    features = read_parquet(PROC / "features.parquet")

    # Load or create training targets
    targets_file = PROC / "training_targets.parquet"
    if targets_file.exists():
        targets = read_parquet(targets_file)
    else:
        # Fetch fresh match data
        match_data = fetch_comprehensive_match_data()
        write_parquet(match_data, PROC / "match_data.parquet")

        # Aggregate into targets
        targets = aggregate_training_targets(match_data)
        write_parquet(targets, targets_file)

    target_cols = ["target_minutes", "target_goals", "target_assists"]

    if targets.empty:
        # Keep the target columns present but empty, so downstream code sees "no outcomes"
        # rather than the columns vanishing and silently falling back to zero series.
        merged = features.copy()
        for col in target_cols:
            merged[col] = pd.NA
        log.warning("No training targets available; %d players carry empty targets", len(merged))
        return merged

    merged = features.merge(targets, on="player_id", how="left")

    # Fill missing targets with zeros (for players with no match data)
    for col in target_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    n_targets = sum(1 for c in merged.columns if str(c).startswith("target_"))
    log.info("Merged dataset has %d players and %d target columns", len(merged), n_targets)

    return merged


def build_training_dataset() -> None:
    """
    Main function to build complete training dataset with real match data.
    """
    log.info("Building training dataset with actual match performance data...")

    # Step 1: Fetch comprehensive match data
    match_data = fetch_comprehensive_match_data()
    write_parquet(match_data, PROC / "match_data.parquet")

    # Step 2: Aggregate into training targets
    targets = aggregate_training_targets(match_data)
    write_parquet(targets, PROC / "training_targets.parquet")

    # Step 3: Merge with features
    training_data = merge_targets_with_features()
    write_parquet(training_data, PROC / "training_data.parquet")

    log.info("Training dataset built successfully!")

    non_zero_goals = int((training_data.get("target_goals", pd.Series(dtype=float)) > 0).sum())
    non_zero_assists = int((training_data.get("target_assists", pd.Series(dtype=float)) > 0).sum())
    non_zero_minutes = int((training_data.get("target_minutes", pd.Series(dtype=float)) > 0).sum())

    log.info(
        f"Training targets: {non_zero_goals} players with goals, "
        f"{non_zero_assists} with assists, {non_zero_minutes} with minutes"
    )
    if non_zero_minutes == 0:
        log.warning(
            "No current-season outcomes yet, so the models have nothing to fit. Predictions "
            "will come from previous-season priors and FPL's own ep_next until GW1 is played."
        )


if __name__ == "__main__":
    build_training_dataset()
