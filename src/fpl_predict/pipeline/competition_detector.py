"""Detect competition between players for starting positions."""
from __future__ import annotations
import pandas as pd
import numpy as np
from ..utils.logging import get_logger
from ..data.fpl_api import get_bootstrap

log = get_logger(__name__)

def detect_position_competition():
    """Detect when players are likely competing for the same position.
    
    Uses data-driven approach to identify backup/rotation players based on:
    - Ownership percentage relative to teammates
    - Price relative to teammates in same position
    - Expected squad depth per position
    
    NO HARDCODED NAMES - purely data-driven.
    
    Returns dict mapping player_id to xMins adjustment factor.
    """
    bootstrap = get_bootstrap()
    players = pd.DataFrame(bootstrap['elements'])
    
    adjustments = {}
    
    # Convert ownership to float for calculations
    players['selected_by_percent'] = players['selected_by_percent'].astype(float)

    # How many games has anyone actually started so far this season? Used below as an
    # empirical override: the price/ownership hierarchy this function otherwise relies on
    # assumes a formation-agnostic "4-5 defenders start" average, which misclassifies a
    # genuinely nailed 5th defender on a back-five team as a rotation risk purely for
    # ranking below higher-owned teammates. A player who has started every available game
    # so far is proven to be a starter regardless of what the ownership heuristic guesses.
    games_so_far = int(players['starts'].max()) if 'starts' in players.columns and len(players) else 0

    # Expected regular starters per position (based on common formations)
    expected_starters = {
        1: 1,   # GKP: 1 starter
        2: 4.5, # DEF: 4-5 starters (most teams play 4 or 5 at the back)
        3: 3.5, # MID: 3-4 starters (varies by formation)
        4: 1.5, # FWD: 1-2 starters (some play 2 up top)
    }
    
    # Process each team's squad
    for team_id in players['team'].unique():
        team_players = players[players['team'] == team_id]
        
        for pos_id, expected_count in expected_starters.items():
            position_players = team_players[team_players['element_type'] == pos_id].copy()
            
            if len(position_players) < 2:
                continue  # No competition if only 1 player
            
            # Sort by price (primary) and ownership (secondary) to identify hierarchy
            position_players = position_players.sort_values(
                ['now_cost', 'selected_by_percent'], 
                ascending=[False, False]
            )
            
            # Calculate relative metrics within position group
            max_cost = position_players['now_cost'].max()
            max_ownership = position_players['selected_by_percent'].max()
            
            for idx, player in position_players.iterrows():
                player_id = player['id']
                player_name = player['web_name']
                ownership = player['selected_by_percent']
                cost = player['now_cost']
                
                # Calculate player's rank in pecking order
                rank = position_players.index.get_loc(idx) + 1

                # Empirically nailed: started every game played so far, whatever their
                # price/ownership rank among teammates suggests. Skips every discount path
                # below (rank-based and the low-ownership special case alike).
                if pos_id != 1 and games_so_far > 0 and int(player.get('starts', 0) or 0) >= games_so_far:
                    continue

                # Relative metrics (0-1 scale)
                relative_cost = cost / max_cost if max_cost > 0 else 0
                relative_ownership = ownership / max_ownership if max_ownership > 0 else 0
                
                # Determine playing time factor based on position in hierarchy
                if pos_id == 1:  # Goalkeepers
                    if rank == 1:
                        continue  # Starting GKP, no adjustment
                    else:
                        # Backup GKP - severe reduction
                        factor = 0.0
                        adjustments[player_id] = factor
                        log.info(f"Backup GKP: {player_name} (team {team_id}, rank {rank}, {ownership:.1f}% owned)")
                
                elif pos_id in [2, 3, 4]:  # Outfield players
                    # Combine cost and ownership for overall importance score
                    importance_score = (relative_cost * 0.6 + relative_ownership * 0.4)
                    
                    # Special case: If ownership is significantly lower than top players, likely backup
                    # E.g., Gusto (1.8%) vs Cucurella (18.8%) and James (5.0%)
                    if ownership < 3.0 and max_ownership > 10.0 and ownership < max_ownership / 5:
                        factor = min(0.3, importance_score * 0.4)
                        adjustments[player_id] = factor
                        log.info(f"Low ownership backup: {player_name} ({['GKP','DEF','MID','FWD'][pos_id-1]}, {ownership:.1f}% vs {max_ownership:.1f}% max)")
                        continue
                    
                    # Determine if player is likely starter based on rank and metrics
                    if rank <= np.floor(expected_count):
                        # Likely regular starter
                        continue
                    elif rank <= np.ceil(expected_count):
                        # Rotation player (sometimes starts)
                        if importance_score < 0.5:
                            factor = 0.4 + (importance_score * 0.4)  # 40-60% playing time
                            adjustments[player_id] = factor
                            log.info(f"Rotation player: {player_name} ({['GKP','DEF','MID','FWD'][pos_id-1]}, rank {rank}, {ownership:.1f}% owned, {cost/10:.1f}m)")
                    else:
                        # Backup/bench player
                        if ownership < 2.0 and relative_cost < 0.6:
                            # Clear backup - minimal playing time
                            factor = min(0.2, importance_score * 0.3)
                            adjustments[player_id] = factor
                            log.info(f"Backup: {player_name} ({['GKP','DEF','MID','FWD'][pos_id-1]}, rank {rank}, {ownership:.1f}% owned, {cost/10:.1f}m)")
                        elif ownership < 5.0:
                            # Squad player - limited playing time
                            factor = 0.25 + (importance_score * 0.25)
                            adjustments[player_id] = factor
                            log.info(f"Squad player: {player_name} ({['GKP','DEF','MID','FWD'][pos_id-1]}, rank {rank}, {ownership:.1f}% owned)")
                
                # Additional check: New signings displacing existing players
                # If a cheaper player has much higher ownership, they might be displaced
                if pos_id in [2, 3, 4]:  # Only for outfield
                    higher_owned = position_players[
                        (position_players['selected_by_percent'] > ownership * 3) & 
                        (position_players['now_cost'] >= cost * 0.8)
                    ]
                    if len(higher_owned) > 0:
                        # This player is likely being displaced by new signing
                        if player_id in adjustments:
                            adjustments[player_id] = min(adjustments[player_id], 0.3)
                        else:
                            adjustments[player_id] = 0.3
                        log.info(f"Displaced player: {player_name} (lower ownership than teammates)")
    
    log.info(f"Competition detection complete: {len(adjustments)} players adjusted")
    return adjustments