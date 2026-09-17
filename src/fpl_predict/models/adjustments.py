"""
Adjustments for model predictions to handle edge cases.
"""
import pandas as pd
import numpy as np
from ..utils.logging import get_logger

log = get_logger(__name__)


def apply_new_player_adjustments(
    predictions_df: pd.DataFrame,
    features_df: pd.DataFrame,
    bootstrap_data: dict,
    min_minutes_threshold: int = 180,  # 2 full games
    new_player_floor_multiplier: float = 0.5,  # Minimum 50% of position average
) -> pd.DataFrame:
    """
    Adjust predictions for players with limited Premier League data.
    
    The model can be overly harsh on new signings or players with limited minutes,
    especially if they were subbed early without returns in their first game(s).
    
    Args:
        predictions_df: DataFrame with model predictions
        features_df: DataFrame with player features
        bootstrap_data: FPL bootstrap data
        min_minutes_threshold: Minutes threshold below which we apply adjustments
        new_player_floor_multiplier: Minimum prediction as fraction of position average
    
    Returns:
        Adjusted predictions DataFrame
    """
    adjusted = predictions_df.copy()
    
    # Get player info from bootstrap
    players_map = {p['id']: p for p in bootstrap_data['elements']}
    
    # Calculate position averages for similar-priced players
    position_benchmarks = {}
    for pos in ['GKP', 'DEF', 'MID', 'FWD']:
        pos_players = [p for p in bootstrap_data['elements'] 
                      if p['element_type'] == {'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}[pos]]
        
        # Group by price bracket (±£1m)
        price_brackets = {}
        for p in pos_players:
            price = p['now_cost'] / 10  # Convert to millions
            bracket = round(price)  # Round to nearest million
            if bracket not in price_brackets:
                price_brackets[bracket] = []
            
            # Only include players with reasonable minutes
            if p['minutes'] > min_minutes_threshold:
                ppg = p.get('points_per_game', 0)
                if ppg > 0:
                    price_brackets[bracket].append(float(ppg))
        
        # Calculate average for each bracket
        position_benchmarks[pos] = {}
        for bracket, ppgs in price_brackets.items():
            if ppgs:
                position_benchmarks[pos][bracket] = np.mean(ppgs)
    
    # Apply adjustments
    adjustments_made = []
    
    for idx, row in adjusted.iterrows():
        player_id = row.get('id') or row.get('player_id')
        if player_id and player_id in players_map:
            player = players_map[player_id]
            
            # Check if player needs adjustment
            total_minutes = player.get('minutes', 0)
            if total_minutes < min_minutes_threshold:
                # Get player position and price
                pos_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
                pos = pos_map.get(player['element_type'])
                price = player['now_cost'] / 10
                price_bracket = round(price)
                
                if pos and pos in position_benchmarks:
                    # Get benchmark for similar-priced players in position
                    benchmark = position_benchmarks[pos].get(price_bracket)
                    if not benchmark:
                        # Try adjacent brackets
                        benchmark = (position_benchmarks[pos].get(price_bracket - 1, 0) + 
                                   position_benchmarks[pos].get(price_bracket + 1, 0)) / 2
                    
                    if benchmark > 0:
                        # Calculate minimum acceptable prediction
                        min_prediction = benchmark * new_player_floor_multiplier
                        
                        # Check EP columns and adjust if needed
                        for col in adjusted.columns:
                            if col.startswith('EP') or col == 'EPH':
                                current_val = adjusted.at[idx, col]
                                if pd.notna(current_val) and current_val < min_prediction:
                                    # Apply adjustment with explanation
                                    adjustment_factor = min_prediction / max(current_val, 0.1)
                                    adjusted.at[idx, col] = min_prediction
                                    
                                    if col == 'EPH' or col == 'EP_1':  # Log key adjustments
                                        adjustments_made.append({
                                            'player': player['web_name'],
                                            'position': pos,
                                            'price': price,
                                            'minutes': total_minutes,
                                            'original': current_val,
                                            'adjusted': min_prediction,
                                            'benchmark': benchmark
                                        })
    
    # Log adjustments
    if adjustments_made:
        log.info(f"Applied new player adjustments to {len(adjustments_made)} players:")
        for adj in adjustments_made[:5]:  # Show first 5
            log.info(f"  {adj['player']} ({adj['position']}, £{adj['price']:.1f}m): "
                    f"{adj['original']:.2f} -> {adj['adjusted']:.2f} pts/game "
                    f"(benchmark: {adj['benchmark']:.2f}, mins: {adj['minutes']})")
    
    return adjusted