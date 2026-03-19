"""
FPL 2025/26 Chip Strategy - Updated for Double Chips System (FIXED)

Key Changes:
- 2 sets of chips (8 total): one for H1 (GW1-19), one for H2 (GW20-38)
- H1 chips expire at GW19 deadline - use it or lose it!
- No DGWs/BGWs expected in H1 (not affected by cups)
- DGWs/BGWs mainly in H2 (GW28-38 typically)
- ENSURES ONLY ONE CHIP PER GAMEWEEK
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import pandas as pd
import numpy as np
from collections import defaultdict

from ..utils.cache import PROC
from ..utils.io import read_parquet
from ..utils.logging import get_logger
from ..data.fpl_api import get_bootstrap, get_fixtures

log = get_logger(__name__)


# ----------------------------- Configuration -----------------------------

@dataclass
class ChipStrategy2025Config:
    """Configuration for 2025/26 double chips strategy"""
    
    # First Half (GW1-19) - No DGWs expected, lower thresholds
    h1_tc_min_ep: float = 7.5      # Lower threshold since no DGWs
    h1_bb_min_ep: float = 1.5      # Very low - bench players score less in SGW
    h1_fh_min_gap: float = 6.0     # Lower to ensure usage 
    h1_wc_preferred_gws: Set[int] = field(default_factory=lambda: {8, 9, 14, 15})
    
    # Second Half (GW20-38) - DGWs/BGWs expected, higher thresholds
    h2_tc_min_ep_sgw: float = 9.0
    h2_tc_min_ep_dgw: float = 14.0
    h2_bb_min_ep_sgw: float = 12.0
    h2_bb_min_ep_dgw: float = 18.0
    h2_fh_bgw_threshold: int = 6    # Use FH if <=6 players have fixtures
    h2_wc_preferred_gws: Set[int] = field(default_factory=lambda: {28, 29, 30, 31})
    
    # General settings
    bb_min_xmins: float = 60.0
    bb_min_players: int = 3
    
    # Urgency factors (increase as deadline approaches)
    h1_urgency_boost_gw17: float = 0.15  # 15% threshold reduction
    h1_urgency_boost_gw18: float = 0.25  # 25% threshold reduction  
    h1_urgency_boost_gw19: float = 0.40  # 40% threshold reduction


class ChipType(Enum):
    """Chip types with H1/H2 designation"""
    H1_TRIPLE_CAPTAIN = "H1_TC"
    H1_BENCH_BOOST = "H1_BB"
    H1_FREE_HIT = "H1_FH"
    H1_WILDCARD = "H1_WC"
    H2_TRIPLE_CAPTAIN = "H2_TC"
    H2_BENCH_BOOST = "H2_BB"
    H2_FREE_HIT = "H2_FH"
    H2_WILDCARD = "H2_WC"


@dataclass
class ChipRecommendation:
    """Chip recommendation with reasoning"""
    chip_type: ChipType
    gameweek: int
    expected_value: float
    confidence: float
    urgency: float  # 0-1 how urgent to use (approaching deadline)
    reasons: List[str]
    player_targets: List[str] = field(default_factory=list)


# ----------------------------- Main Strategy Class -----------------------------

class FPL2025ChipStrategy:
    """
    Chip strategy for FPL 2025/26 with double chips system
    """
    
    def __init__(self, config: ChipStrategy2025Config = None):
        self.config = config or ChipStrategy2025Config()
        self.h1_deadline = 19
        self.h2_start = 20
        self._detected_dgws: List[int] = []
        self._detected_bgws: List[int] = []
        self._outstanding_fixtures: Dict[str, int] = {}
        
    def plan_chips(
        self,
        use_myteam: bool = True,
        current_gw: Optional[int] = None,
        explain: bool = True,
        show_teams: bool = False
    ) -> Dict[str, ChipRecommendation]:
        """
        Generate chip strategy for the season

        Returns separate recommendations for H1 and H2 chips
        """

        # Get current gameweek
        if not current_gw:
            current_gw = self._get_current_gw()

        # Load team and player data
        owned_ids = self._load_myteam() if use_myteam else set()
        player_data = self._load_player_data()

        # Get already used chips
        used_chips = self._get_used_chips() if use_myteam else set()

        recommendations = {}

        # Plan H1 chips if still in first half
        if current_gw <= self.h1_deadline:
            h1_recs = self._plan_h1_chips(
                current_gw,
                owned_ids,
                player_data
            )
            recommendations.update(h1_recs)

        # Always plan H2 chips for reference
        h2_recs = self._plan_h2_chips(
            max(current_gw, self.h2_start),
            owned_ids,
            player_data
        )
        recommendations.update(h2_recs)

        # Filter out chips that have already been used
        recommendations = {
            chip_key: rec
            for chip_key, rec in recommendations.items()
            if chip_key not in used_chips
        }

        if explain:
            self._explain_strategy(recommendations, current_gw, show_teams=show_teams, used_chips=used_chips)

        return recommendations
    
    def _plan_h1_chips(
        self,
        current_gw: int,
        owned_ids: Set[int],
        player_data: pd.DataFrame
    ) -> Dict[str, ChipRecommendation]:
        """
        Plan first half chips using fixture-based per-GW EP predictions
        """

        recs = {}
        occupied_gws = set()  # Track which GWs have chips
        urgency = self._calculate_h1_urgency(current_gw)

        # Collect all potential recommendations using new fixture-based methods
        potential_chips = []

        # 1. Find best TC options (top 3 player+GW combinations with haul potential)
        tc_options = self._find_best_tc_options(owned_ids, player_data, current_gw, self.h1_deadline, top_n=3)
        for option in tc_options:
            # Describe haul potential
            haul_desc = {
                (1.5, 2.0): "explosive haul potential",
                (1.2, 1.5): "high haul potential",
                (1.0, 1.2): "good haul potential",
                (0.8, 1.0): "moderate ceiling",
                (0.0, 0.8): "consistent scorer"
            }
            haul_text = next((v for k, v in haul_desc.items() if k[0] <= option['haul_factor'] < k[1]), "")

            potential_chips.append(ChipRecommendation(
                chip_type=ChipType.H1_TRIPLE_CAPTAIN,
                gameweek=option['gw'],
                expected_value=option['tc_score'] * 3,  # Use TC score for sorting
                confidence=min(0.9, option['tc_score'] / 12),
                urgency=urgency,
                reasons=[
                    f"{option['player_name']} expected {option['ep']:.1f} points",
                    f"Haul factor: {option['haul_factor']:.2f}x ({haul_text})",
                    "Best fixture-adjusted TC option"
                ],
                player_targets=[option['player_name']]
            ))

        # 2. Find best BB options (top 3 gameweeks)
        bb_options = self._find_best_bb_options(owned_ids, player_data, current_gw, self.h1_deadline, top_n=3)
        for option in bb_options:
            potential_chips.append(ChipRecommendation(
                chip_type=ChipType.H1_BENCH_BOOST,
                gameweek=option['gw'],
                expected_value=option['bench_ep'],
                confidence=min(0.85, option['bench_ep'] / 8),
                urgency=urgency,
                reasons=[
                    f"Bench expected {option['bench_ep']:.1f} points",
                    f"{option['num_playing']}/4 bench players likely to play"
                ]
            ))

        # 3. Find best FH options (top 3 gameweeks by gap)
        fh_options = self._find_best_fh_options(owned_ids, player_data, current_gw, self.h1_deadline, top_n=3)
        for option in fh_options:
            potential_chips.append(ChipRecommendation(
                chip_type=ChipType.H1_FREE_HIT,
                gameweek=option['gw'],
                expected_value=option['gap'],
                confidence=min(0.9, option['gap'] / 15),
                urgency=urgency,
                reasons=[
                    f"Can gain {option['gap']:.1f} points vs current team",
                    f"Owned XI: {option['owned_xi_ep']:.1f}, Optimal XI: {option['optimal_xi_ep']:.1f}"
                ]
            ))

        # 4. Add WC option
        wc_gw = self._find_h1_wildcard_gw(current_gw)
        if wc_gw:
            potential_chips.append(ChipRecommendation(
                chip_type=ChipType.H1_WILDCARD,
                gameweek=wc_gw,
                expected_value=10,  # Give WC some value for sorting
                confidence=0.8,
                urgency=0.3 if current_gw < 15 else 0.7,
                reasons=[
                    f"International break in GW{wc_gw}" if wc_gw in {4, 8, 12} else "Fixture swing opportunity",
                    "Time to restructure team mid-H1"
                ]
            ))
        
        # Sort by priority: expected_value * confidence
        potential_chips.sort(
            key=lambda x: x.expected_value * x.confidence,
            reverse=True
        )
        
        # Assign chips avoiding conflicts
        chips_assigned = {'TC': False, 'BB': False, 'FH': False, 'WC': False}
        
        for chip in potential_chips:
            if chip.gameweek not in occupied_gws:
                chip_key = chip.chip_type.value.split('_')[1]  # Get TC, BB, FH, or WC
                if not chips_assigned[chip_key]:
                    recs[chip.chip_type.value] = chip
                    occupied_gws.add(chip.gameweek)
                    chips_assigned[chip_key] = True
        
        # If approaching deadline and chips not assigned, force them
        if current_gw >= 17:
            remaining_gws = [gw for gw in range(current_gw, self.h1_deadline + 1) 
                           if gw not in occupied_gws]
            
            # Map short codes to full enum names
            chip_name_map = {
                'TC': 'TRIPLE_CAPTAIN',
                'BB': 'BENCH_BOOST',
                'FH': 'FREE_HIT',
                'WC': 'WILDCARD'
            }
            for chip_key in ['TC', 'BB', 'FH', 'WC']:
                if not chips_assigned[chip_key] and remaining_gws:
                    gw = remaining_gws.pop(0)
                    chip_type = getattr(ChipType, f'H1_{chip_name_map[chip_key]}')
                    recs[chip_type.value] = ChipRecommendation(
                        chip_type=chip_type,
                        gameweek=gw,
                        expected_value=5,
                        confidence=0.5,
                        urgency=1.0,
                        reasons=[
                            f"⚠️ URGENT: Only {self.h1_deadline - current_gw + 1} GWs left!",
                            "Use it or lose it!"
                        ]
                    )
                    occupied_gws.add(gw)
        
        return recs
    
    def _plan_h2_chips(
        self,
        current_gw: int,
        owned_ids: Set[int],
        player_data: pd.DataFrame
    ) -> Dict[str, ChipRecommendation]:
        """
        Plan second half chips using scenario-based sequencing.

        Picks chips in priority order, tracking occupied GWs to prevent
        conflicts and ensure logical sequences (e.g. FH on BGW then WC
        the week after to rebuild).

        Scenarios:
        A: BGWs + DGWs → FH(BGW) → WC(BGW+1) → TC(best DGW) → BB(different DGW)
        B: BGWs only   → FH(BGW) → WC(BGW+1) → TC(HOLD) → BB(HOLD)
        C: DGWs only   → WC(before DGWs) → TC(best DGW) → BB(different DGW) → FH(HOLD)
        D: Neither      → All HOLD
        """
        # Predict DGWs and BGWs
        dgws, bgws = self._predict_dgw_bgw()
        # Store for use in strategy tips
        self._detected_dgws = dgws
        self._detected_bgws = bgws

        has_bgws = len(bgws) > 0
        has_dgws = len(dgws) > 0

        if has_bgws and has_dgws:
            return self._sequence_scenario_a(current_gw, owned_ids, player_data, dgws, bgws)
        elif has_bgws:
            return self._sequence_scenario_b(current_gw, owned_ids, player_data, bgws)
        elif has_dgws:
            return self._sequence_scenario_c(current_gw, owned_ids, player_data, dgws)
        else:
            return self._sequence_scenario_d()

    # ---------- Scenario sequencing methods ----------

    def _score_bgw_for_fh(self, bgws: List[int], owned_ids: Set[int]) -> int:
        """Pick the BGW where the user's squad has the fewest playing players.

        Worst coverage = best Free Hit target.
        """
        if not bgws:
            return bgws[0] if bgws else 33

        try:
            fixtures = get_fixtures()
            boot = get_bootstrap()

            # Map player_id → team_id
            pid_to_team = {p['id']: p['team'] for p in boot.get('elements', [])}
            owned_teams = {pid_to_team[pid] for pid in owned_ids if pid in pid_to_team}

            best_bgw = bgws[0]
            fewest_playing = 999

            for gw in bgws:
                # Teams with a fixture in this GW
                teams_playing = set()
                for f in fixtures:
                    if f.get('event') == gw:
                        teams_playing.add(f['team_h'])
                        teams_playing.add(f['team_a'])

                # How many of our players' teams are playing?
                coverage = len(owned_teams & teams_playing)
                if coverage < fewest_playing:
                    fewest_playing = coverage
                    best_bgw = gw

            return best_bgw
        except Exception:
            # Fallback: pick the first BGW
            return bgws[0]

    def _make_hold(self, chip_type: ChipType, reason: str) -> ChipRecommendation:
        """Create a HOLD recommendation (gameweek=0 sentinel)."""
        return ChipRecommendation(
            chip_type=chip_type,
            gameweek=0,
            expected_value=0,
            confidence=0.0,
            urgency=0.0,
            reasons=[reason]
        )

    def _sequence_scenario_a(
        self,
        current_gw: int,
        owned_ids: Set[int],
        player_data: pd.DataFrame,
        dgws: List[int],
        bgws: List[int]
    ) -> Dict[str, ChipRecommendation]:
        """Scenario A: BGWs + DGWs confirmed.

        Order: FH(BGW) → WC(BGW+1) → TC(best DGW) → BB(different DGW)
        """
        recs = {}
        occupied_gws: Set[int] = set()

        # 1. FH on worst-coverage BGW
        fh_gw = self._score_bgw_for_fh(bgws, owned_ids)
        recs['H2_FH'] = ChipRecommendation(
            chip_type=ChipType.H2_FREE_HIT,
            gameweek=fh_gw,
            expected_value=40,
            confidence=0.85,
            urgency=0.1 if current_gw < fh_gw - 2 else 0.6,
            reasons=[
                f"Blank GW{fh_gw} — limited fixtures",
                "Maximise playing XI on a blank gameweek",
            ]
        )
        occupied_gws.add(fh_gw)

        # 2. WC to rebuild after FH
        wc_gw = self._find_h2_wildcard_gw(dgws, fh_gw=fh_gw, excluded_gws=occupied_gws)
        if wc_gw:
            recs['H2_WC'] = ChipRecommendation(
                chip_type=ChipType.H2_WILDCARD,
                gameweek=wc_gw,
                expected_value=0,
                confidence=0.8,
                urgency=0.2,
                reasons=[
                    f"Rebuild squad after Free Hit on GW{fh_gw}",
                    "Position for upcoming DGWs",
                ]
            )
            occupied_gws.add(wc_gw)

        # 3. TC on best DGW
        best_dgw = self._find_best_captain_dgw(dgws, owned_ids, player_data, excluded_gws=occupied_gws)
        if best_dgw:
            recs['H2_TC'] = ChipRecommendation(
                chip_type=ChipType.H2_TRIPLE_CAPTAIN,
                gameweek=best_dgw['gw'],
                expected_value=best_dgw['ep'] * 3,
                confidence=0.9,
                urgency=0.1 if current_gw < 30 else 0.5,
                reasons=[
                    f"Double gameweek for {best_dgw['player']}",
                    f"Expected {best_dgw['ep']:.1f} points (captained)",
                    "Premium DGW opportunity",
                ],
                player_targets=[best_dgw['player']]
            )
            occupied_gws.add(best_dgw['gw'])
        else:
            recs['H2_TC'] = self._make_hold(
                ChipType.H2_TRIPLE_CAPTAIN,
                "No available DGW for TC — HOLD pending fixture announcements"
            )

        # 4. BB on a different DGW
        best_bb = self._find_best_bench_boost_dgw(dgws, owned_ids, player_data, excluded_gws=occupied_gws)
        if best_bb:
            label = "Double gameweek" if best_bb['dgw_players'] > 0 else "Best remaining gameweek"
            recs['H2_BB'] = ChipRecommendation(
                chip_type=ChipType.H2_BENCH_BOOST,
                gameweek=best_bb['gw'],
                expected_value=best_bb['bench_ep'],
                confidence=0.85 if best_bb['dgw_players'] > 0 else 0.6,
                urgency=0.1 if current_gw < 30 else 0.5,
                reasons=[
                    f"{label} for bench players",
                    f"Bench expected {best_bb['bench_ep']:.1f} points",
                ]
            )
        else:
            recs['H2_BB'] = self._make_hold(
                ChipType.H2_BENCH_BOOST,
                "No suitable GW for BB — HOLD pending fixture announcements"
            )

        return recs

    def _sequence_scenario_b(
        self,
        current_gw: int,
        owned_ids: Set[int],
        player_data: pd.DataFrame,
        bgws: List[int]
    ) -> Dict[str, ChipRecommendation]:
        """Scenario B: BGWs only, no DGWs confirmed.

        Order: FH(BGW) → WC(BGW+1) → TC(HOLD) → BB(HOLD)
        """
        recs = {}
        occupied_gws: Set[int] = set()

        # 1. FH on worst-coverage BGW
        fh_gw = self._score_bgw_for_fh(bgws, owned_ids)
        recs['H2_FH'] = ChipRecommendation(
            chip_type=ChipType.H2_FREE_HIT,
            gameweek=fh_gw,
            expected_value=40,
            confidence=0.85,
            urgency=0.1 if current_gw < fh_gw - 2 else 0.6,
            reasons=[
                f"Blank GW{fh_gw} — limited fixtures",
                "Maximise playing XI on a blank gameweek",
            ]
        )
        occupied_gws.add(fh_gw)

        # 2. WC to rebuild after FH
        wc_gw = self._find_h2_wildcard_gw([], fh_gw=fh_gw, excluded_gws=occupied_gws)
        if wc_gw:
            recs['H2_WC'] = ChipRecommendation(
                chip_type=ChipType.H2_WILDCARD,
                gameweek=wc_gw,
                expected_value=0,
                confidence=0.75,
                urgency=0.2,
                reasons=[
                    f"Rebuild squad after Free Hit on GW{fh_gw}",
                    "No DGWs confirmed yet — rebuild for run-in",
                ]
            )
            occupied_gws.add(wc_gw)

        # 3. TC — HOLD (no DGWs)
        recs['H2_TC'] = self._make_hold(
            ChipType.H2_TRIPLE_CAPTAIN,
            "No DGWs confirmed — HOLD for future DGW announcement"
        )

        # 4. BB — HOLD (no DGWs)
        recs['H2_BB'] = self._make_hold(
            ChipType.H2_BENCH_BOOST,
            "No DGWs confirmed — HOLD for future DGW announcement"
        )

        return recs

    def _sequence_scenario_c(
        self,
        current_gw: int,
        owned_ids: Set[int],
        player_data: pd.DataFrame,
        dgws: List[int]
    ) -> Dict[str, ChipRecommendation]:
        """Scenario C: DGWs only, no BGWs.

        Order: WC(before DGWs) → TC(best DGW) → BB(different DGW) → FH(HOLD)
        """
        recs = {}
        occupied_gws: Set[int] = set()

        # 1. WC to prepare for DGWs
        wc_gw = self._find_h2_wildcard_gw(dgws, excluded_gws=occupied_gws)
        if wc_gw:
            recs['H2_WC'] = ChipRecommendation(
                chip_type=ChipType.H2_WILDCARD,
                gameweek=wc_gw,
                expected_value=0,
                confidence=0.75,
                urgency=0.2,
                reasons=[
                    "Position squad before DGW run",
                    "Build squad for TC/BB potential",
                ]
            )
            occupied_gws.add(wc_gw)

        # 2. TC on best DGW
        best_dgw = self._find_best_captain_dgw(dgws, owned_ids, player_data, excluded_gws=occupied_gws)
        if best_dgw:
            recs['H2_TC'] = ChipRecommendation(
                chip_type=ChipType.H2_TRIPLE_CAPTAIN,
                gameweek=best_dgw['gw'],
                expected_value=best_dgw['ep'] * 3,
                confidence=0.9,
                urgency=0.1 if current_gw < 30 else 0.5,
                reasons=[
                    f"Double gameweek for {best_dgw['player']}",
                    f"Expected {best_dgw['ep']:.1f} points (captained)",
                    "Premium DGW opportunity",
                ],
                player_targets=[best_dgw['player']]
            )
            occupied_gws.add(best_dgw['gw'])

        # 3. BB on different DGW (or SGW fallback)
        best_bb = self._find_best_bench_boost_dgw(dgws, owned_ids, player_data, excluded_gws=occupied_gws)
        if best_bb:
            label = "Double gameweek" if best_bb['dgw_players'] > 0 else "Best remaining gameweek"
            recs['H2_BB'] = ChipRecommendation(
                chip_type=ChipType.H2_BENCH_BOOST,
                gameweek=best_bb['gw'],
                expected_value=best_bb['bench_ep'],
                confidence=0.85 if best_bb['dgw_players'] > 0 else 0.6,
                urgency=0.1 if current_gw < 30 else 0.5,
                reasons=[
                    f"{label} for bench players",
                    f"Bench expected {best_bb['bench_ep']:.1f} points",
                ]
            )

        # 4. FH — HOLD (no BGWs)
        recs['H2_FH'] = self._make_hold(
            ChipType.H2_FREE_HIT,
            "No BGWs confirmed — HOLD for future BGW announcement"
        )

        return recs

    def _sequence_scenario_d(self) -> Dict[str, ChipRecommendation]:
        """Scenario D: No BGWs or DGWs confirmed. All HOLD."""
        return {
            'H2_FH': self._make_hold(
                ChipType.H2_FREE_HIT,
                "No BGWs confirmed — HOLD pending fixture announcements"
            ),
            'H2_WC': self._make_hold(
                ChipType.H2_WILDCARD,
                "No BGWs/DGWs confirmed — HOLD pending fixture announcements"
            ),
            'H2_TC': self._make_hold(
                ChipType.H2_TRIPLE_CAPTAIN,
                "No DGWs confirmed — HOLD pending fixture announcements"
            ),
            'H2_BB': self._make_hold(
                ChipType.H2_BENCH_BOOST,
                "No DGWs confirmed — HOLD pending fixture announcements"
            ),
        }
    
    def _calculate_h1_urgency(self, current_gw: int) -> float:
        """Calculate urgency factor for H1 chips"""
        if current_gw >= 19:
            return 0.5  # Maximum urgency at deadline
        elif current_gw >= 18:
            return self.config.h1_urgency_boost_gw18
        elif current_gw >= 17:
            return self.config.h1_urgency_boost_gw17
        return 0.0
    
    def _predict_dgw_bgw(self) -> Tuple[List[int], List[int]]:
        """
        Detect DGWs and BGWs from actual FPL fixture data.

        A normal gameweek has 10 fixtures (20 teams / 2 per fixture).
        - BGW: fewer than 10 fixtures scheduled
        - DGW: more than 10 fixtures scheduled (some teams play twice)

        Also detects teams with fewer scheduled fixtures than expected,
        which predicts future DGWs when those games are rescheduled.
        """
        try:
            fixtures = get_fixtures()
            boot = get_bootstrap()
            current_gw = self._get_current_gw()
            teams_map = {t["id"]: t["short_name"] for t in boot.get("teams", [])}
            total_gws = len(boot.get("events", []))

            # Count fixtures per gameweek
            fixtures_per_gw: Dict[int, int] = defaultdict(int)
            for f in fixtures:
                gw = f.get("event")
                if gw is not None:
                    fixtures_per_gw[int(gw)] += 1

            likely_dgws = []
            likely_bgws = []

            for gw in sorted(fixtures_per_gw.keys()):
                if gw < current_gw:
                    continue
                count = fixtures_per_gw[gw]
                if count > 10:
                    likely_dgws.append(gw)
                    log.info(f"DGW detected: GW{gw} ({count} fixtures)")
                elif count < 10:
                    likely_bgws.append(gw)
                    log.info(f"BGW detected: GW{gw} ({count} fixtures)")

            # Detect teams with outstanding fixtures (fewer than expected)
            # Each team should play `total_gws` games (38 in a standard season)
            team_scheduled = defaultdict(int)
            unscheduled_fixtures = []
            for f in fixtures:
                if f.get("event") is not None:
                    team_scheduled[f["team_h"]] += 1
                    team_scheduled[f["team_a"]] += 1
                elif not f.get("finished"):
                    unscheduled_fixtures.append(f)

            # Teams with fewer games than expected will get DGWs
            self._outstanding_fixtures = {}
            for tid, count in team_scheduled.items():
                remaining_expected = total_gws
                if count < remaining_expected:
                    outstanding = remaining_expected - count
                    team_name = teams_map.get(tid, f"Team {tid}")
                    self._outstanding_fixtures[team_name] = outstanding
                    log.info(f"{team_name}: {count}/{remaining_expected} fixtures scheduled ({outstanding} outstanding)")

            if unscheduled_fixtures:
                log.info(f"{len(unscheduled_fixtures)} unscheduled fixture(s) — will produce future DGWs")
                for f in unscheduled_fixtures:
                    home = teams_map.get(f.get("team_h"), "?")
                    away = teams_map.get(f.get("team_a"), "?")
                    log.info(f"  Unscheduled: {home} vs {away}")

            if not likely_dgws and not likely_bgws:
                log.info("No DGWs or BGWs detected in remaining fixtures")

            return likely_dgws, likely_bgws

        except Exception as e:
            log.warning(f"Could not fetch fixtures for DGW/BGW detection: {e}")
            self._outstanding_fixtures = {}
            return [], []
    
    def _get_current_gw(self) -> int:
        """Get current gameweek from API"""
        boot = get_bootstrap()
        events = boot.get("events", [])
        
        for ev in events:
            if ev.get("is_next"):
                return int(ev["id"])
            elif ev.get("is_current"):
                return int(ev["id"])
        
        return 1
    
    def _load_myteam(self) -> Set[int]:
        """Load user's team"""
        try:
            with open(PROC / "myteam_latest.json", "r") as f:
                data = json.load(f)
            picks = data.get("picks", [])
            return {int(p["element"]) for p in picks}
        except:
            return set()

    def _get_squad_value(self) -> float:
        """Get total selling value of current squad (for Free Hit budget)"""
        try:
            with open(PROC / "myteam_latest.json", "r") as f:
                data = json.load(f)
            picks = data.get("picks", [])
            # selling_price is in tenths (e.g., 51 = £5.1m)
            total_value = sum(p.get("selling_price", 0) for p in picks) / 10.0
            return total_value if total_value > 0 else 100.0  # Default to £100m if no data
        except:
            return 100.0  # Default to £100m if no team data

    def _get_used_chips(self) -> Set[str]:
        """Get chips that have already been played this season

        Returns set of chip names that are played, e.g., {'H1_WC', 'H1_FH'}
        """
        try:
            with open(PROC / "myteam_latest.json", "r") as f:
                data = json.load(f)
            chips = data.get("chips", [])

            used_chips = set()
            for chip in chips:
                if chip.get("status_for_entry") == "played":
                    name = chip.get("name")
                    # Map API chip names to our internal names
                    # Check if it's H1 or H2 based on start_event
                    start_event = chip.get("start_event", 1)
                    half = "H1" if start_event < 20 else "H2"

                    if name == "wildcard":
                        used_chips.add(f"{half}_WC")
                    elif name == "freehit":
                        used_chips.add(f"{half}_FH")
                    elif name == "bboost":
                        used_chips.add(f"{half}_BB")
                    elif name == "3xc":
                        used_chips.add(f"{half}_TC")

            return used_chips
        except Exception as e:
            log.warning(f"Could not load used chips: {e}")
            return set()
    
    def _load_player_data(self) -> pd.DataFrame:
        """Load player EP data"""
        try:
            ep_df = read_parquet(PROC / "exp_points.parquet")
            xmins_df = read_parquet(PROC / "xmins.parquet")
            
            # Merge data
            df = ep_df.merge(xmins_df, on='player_id', how='left')
            
            # Add player info from bootstrap
            boot = get_bootstrap()
            players = []
            for p in boot.get('elements', []):
                players.append({
                    'player_id': p['id'],
                    'name': p['web_name'],
                    'team': p['team'],
                    'position': p['element_type'],
                    'cost': p['now_cost'] / 10
                })
            
            players_df = pd.DataFrame(players)
            df = df.merge(players_df, on='player_id', how='left')
            
            return df
        except:
            return pd.DataFrame()
    
    def _find_h1_wildcard_gw(self, current_gw: int) -> Optional[int]:
        """Find optimal H1 wildcard GW"""
        # Prefer international breaks or mid-H1
        preferred = [gw for gw in self.config.h1_wc_preferred_gws if gw >= current_gw]
        if preferred:
            return min(preferred)

        # Otherwise suggest around GW12-14
        if current_gw <= 12:
            return 12
        elif current_gw <= 14:
            return 14

        return None

    def _calculate_per_gw_ep(
        self,
        player_data: pd.DataFrame,
        gw_start: int,
        gw_end: int
    ) -> Dict[int, Dict[int, float]]:
        """
        Calculate expected points for each player for each gameweek.

        Returns: {player_id: {gw: ep, ...}, ...}
        """
        try:
            # Load FDR data
            fdr_df = read_parquet(PROC / "fdr.parquet")
            future_fixtures = fdr_df[
                (fdr_df['is_future'] == True) &
                (fdr_df['event'].notna()) &
                (fdr_df['event'] >= gw_start) &
                (fdr_df['event'] <= gw_end)
            ].copy()

            # Map team IDs to names
            boot = get_bootstrap()
            team_id_to_name = {t['id']: t['name'] for t in boot['teams']}

            # Build player_id -> team_name mapping
            player_teams = {}
            for p in boot['elements']:
                team_name = team_id_to_name.get(p['team'])
                if team_name:
                    player_teams[p['id']] = team_name

            # Initialize per-GW EP dict
            per_gw_ep = defaultdict(dict)

            # For each player in player_data
            for _, player in player_data.iterrows():
                pid = player['player_id']
                base_ep = player.get('ep_blend', player.get('ep_adjusted', 0))
                team_name = player_teams.get(pid)

                if not team_name or base_ep <= 0:
                    continue

                # Get fixtures for this player's team
                team_fixtures = future_fixtures[
                    (future_fixtures['home_team'] == team_name) |
                    (future_fixtures['away_team'] == team_name)
                ].copy()

                # For each fixture, calculate adjusted EP
                for _, fixture in team_fixtures.iterrows():
                    gw = int(fixture['event'])
                    is_home = fixture['home_team'] == team_name

                    # FDR ranges from 0 (easiest) to 1 (hardest)
                    # Lower FDR = easier fixture = higher EP
                    # Higher FDR = harder fixture = lower EP
                    fdr = fixture['fdr_home'] if is_home else fixture['fdr_away']

                    # INVERT FDR: 1.0 - fdr so that:
                    # Easy fixture (FDR 0.3) → inverted 0.7 → multiplier 1.12 (+12%)
                    # Medium fixture (FDR 0.5) → inverted 0.5 → multiplier 1.0 (no change)
                    # Hard fixture (FDR 0.7) → inverted 0.3 → multiplier 0.88 (-12%)
                    fdr_inverted = 1.0 - fdr
                    fdr_multiplier = 0.7 + (0.6 * fdr_inverted)  # Range: 0.7 to 1.3

                    # Apply home advantage: +5% for home games
                    venue_multiplier = 1.05 if is_home else 1.0

                    gw_ep = base_ep * fdr_multiplier * venue_multiplier
                    per_gw_ep[pid][gw] = gw_ep

                # For GWs with no fixture (blank gameweeks), EP = 0
                for gw in range(gw_start, gw_end + 1):
                    if gw not in per_gw_ep[pid]:
                        per_gw_ep[pid][gw] = 0.0

            return dict(per_gw_ep)

        except Exception as e:
            log.warning(f"Could not calculate per-GW EP: {e}")
            return {}

    def _upgrade_squad_with_budget(
        self,
        starting_xi: List[Dict],
        bench: List[Dict],
        gkps: List[Dict],
        defs: List[Dict],
        mids: List[Dict],
        fwds: List[Dict],
        remaining_budget: float,
        total_budget: float
    ) -> Tuple[List[Dict], List[Dict], float, float]:
        """
        Upgrade squad to use remaining budget efficiently.
        Prioritizes upgrading starting XI first, then bench.

        Returns: (upgraded_xi, upgraded_bench, total_cost, xi_ep)
        """
        import copy

        # Work with copies
        current_xi = copy.deepcopy(starting_xi)
        current_bench = copy.deepcopy(bench)

        # Get all current player IDs and team counts
        all_current = current_xi + current_bench
        selected_ids = {p['id'] for p in all_current}
        team_counts = {}
        for p in all_current:
            team_counts[p['team']] = team_counts.get(p['team'], 0) + 1

        current_cost = sum(p['cost'] for p in all_current)
        current_xi_ep = sum(p['ep'] for p in current_xi)

        # Pool of available players by position
        position_pools = {
            1: [p for p in gkps if p['id'] not in selected_ids],
            2: [p for p in defs if p['id'] not in selected_ids],
            3: [p for p in mids if p['id'] not in selected_ids],
            4: [p for p in fwds if p['id'] not in selected_ids]
        }

        improved = True
        while improved and current_cost < total_budget - 0.1:
            improved = False
            best_upgrade = None
            best_upgrade_value = 0

            # Try upgrading each player in XI and bench
            for player_list, is_xi in [(current_xi, True), (current_bench, False)]:
                for idx, current_player in enumerate(player_list):
                    position = current_player['position']

                    # Find better alternatives
                    for alt_player in position_pools[position]:
                        # Check if upgrade fits budget
                        cost_diff = alt_player['cost'] - current_player['cost']
                        if current_cost + cost_diff > total_budget + 0.1:
                            continue

                        # Check team constraint (removing old player frees up a slot)
                        old_team_count = team_counts.get(current_player['team'], 0)
                        new_team_count = team_counts.get(alt_player['team'], 0)

                        # If swapping to same team, constraint is maintained
                        # If swapping to different team, new team count must be < 3
                        if alt_player['team'] != current_player['team']:
                            if new_team_count >= 3:
                                continue

                        # Calculate value of upgrade
                        ep_gain = alt_player['ep'] - current_player['ep']

                        # For XI upgrades, we care about EP gain
                        # For bench upgrades, we care about EP gain but lower priority
                        upgrade_value = ep_gain if is_xi else ep_gain * 0.3

                        # Track best upgrade
                        if upgrade_value > best_upgrade_value and ep_gain > 0:
                            best_upgrade_value = upgrade_value
                            best_upgrade = {
                                'list': player_list,
                                'idx': idx,
                                'old_player': current_player,
                                'new_player': alt_player,
                                'is_xi': is_xi,
                                'cost_diff': cost_diff
                            }

            # Apply best upgrade if found
            if best_upgrade:
                old_p = best_upgrade['old_player']
                new_p = best_upgrade['new_player']

                # Update the list
                best_upgrade['list'][best_upgrade['idx']] = new_p

                # Update tracking
                selected_ids.remove(old_p['id'])
                selected_ids.add(new_p['id'])

                team_counts[old_p['team']] -= 1
                team_counts[new_p['team']] = team_counts.get(new_p['team'], 0) + 1

                current_cost += best_upgrade['cost_diff']

                if best_upgrade['is_xi']:
                    current_xi_ep += new_p['ep'] - old_p['ep']

                # Remove new player from available pool
                position_pools[new_p['position']].remove(new_p)

                improved = True

        return current_xi, current_bench, current_cost, current_xi_ep

    def _build_optimal_xi_for_gw(
        self,
        gw: int,
        player_data: pd.DataFrame,
        per_gw_ep: Dict[int, Dict[int, float]],
        budget: float = 100.0
    ) -> Dict:
        """
        Build the optimal 15-man squad for a specific gameweek.
        Enforces: 2 GKP, 5 DEF, 5 MID, 3 FWD, max 3 per club, budget constraint.

        Args:
            gw: Gameweek number
            player_data: DataFrame with player info
            per_gw_ep: Per-gameweek EP predictions
            budget: Total budget (default £100m, or squad selling value for Free Hit)

        Returns: {
            'formation': (n_def, n_mid, n_fwd),
            'starting_xi': [list of 11 player dicts],
            'bench': [list of 4 player dicts],
            'total_ep': float (starting XI only),
            'total_cost': float (all 15 players),
            'budget': float (original budget),
            'remaining_budget': float (budget - total_cost)
        }
        """
        from ..data.fpl_api import get_bootstrap

        # Get team info
        boot = get_bootstrap()
        team_map = {t['id']: t['name'] for t in boot['teams']}

        # Collect all players with their GW EP and team
        all_players = []
        for _, player in player_data.iterrows():
            pid = player['player_id']
            gw_ep = per_gw_ep.get(pid, {}).get(gw, 0)
            if gw_ep > 0:
                # Get team from bootstrap
                boot_player = next((p for p in boot['elements'] if p['id'] == pid), None)
                team = team_map.get(boot_player['team']) if boot_player else 'Unknown'

                all_players.append({
                    'id': pid,
                    'name': player['name'],
                    'position': player['position'],
                    'ep': gw_ep,
                    'cost': player['cost'],
                    'team': team
                })

        # Split by position and sort by EP
        gkps = sorted([p for p in all_players if p['position'] == 1], key=lambda x: x['ep'], reverse=True)
        defs = sorted([p for p in all_players if p['position'] == 2], key=lambda x: x['ep'], reverse=True)
        mids = sorted([p for p in all_players if p['position'] == 3], key=lambda x: x['ep'], reverse=True)
        fwds = sorted([p for p in all_players if p['position'] == 4], key=lambda x: x['ep'], reverse=True)

        # Try all valid formations and pick the best 15-man squad
        formations = [
            (3, 4, 3),  # 3-4-3
            (3, 5, 2),  # 3-5-2
            (4, 4, 2),  # 4-4-2
            (4, 5, 1),  # 4-5-1
            (4, 3, 3),  # 4-3-3
            (5, 4, 1),  # 5-4-1
            (5, 3, 2),  # 5-3-2
        ]

        best_formation = None
        best_starting_xi = None
        best_bench = None
        best_ep = 0
        best_total_cost = 0

        for n_def, n_mid, n_fwd in formations:
            # Build starting XI with max 3 per club constraint
            selected_xi = []
            team_counts = {}

            # Start with best GKP
            if gkps:
                selected_xi.append(gkps[0])
                team_counts[gkps[0]['team']] = 1

            # Add defenders
            for p in defs:
                if len([s for s in selected_xi if s['position'] == 2]) < n_def:
                    if team_counts.get(p['team'], 0) < 3:
                        selected_xi.append(p)
                        team_counts[p['team']] = team_counts.get(p['team'], 0) + 1

            # Add midfielders
            for p in mids:
                if len([s for s in selected_xi if s['position'] == 3]) < n_mid:
                    if team_counts.get(p['team'], 0) < 3:
                        selected_xi.append(p)
                        team_counts[p['team']] = team_counts.get(p['team'], 0) + 1

            # Add forwards
            for p in fwds:
                if len([s for s in selected_xi if s['position'] == 4]) < n_fwd:
                    if team_counts.get(p['team'], 0) < 3:
                        selected_xi.append(p)
                        team_counts[p['team']] = team_counts.get(p['team'], 0) + 1

            # Check if we got a full valid XI
            if len(selected_xi) != 11:
                continue

            # Now build bench (4 players): 1 GKP + 3 outfield
            # Squad must have: 2 GKP, 5 DEF, 5 MID, 3 FWD
            bench = []
            selected_ids = {p['id'] for p in selected_xi}

            # Add backup GKP
            for p in gkps:
                if p['id'] not in selected_ids and team_counts.get(p['team'], 0) < 3:
                    bench.append(p)
                    team_counts[p['team']] = team_counts.get(p['team'], 0) + 1
                    selected_ids.add(p['id'])
                    break

            # Calculate how many more of each position we need for full squad
            xi_def_count = len([s for s in selected_xi if s['position'] == 2])
            xi_mid_count = len([s for s in selected_xi if s['position'] == 3])
            xi_fwd_count = len([s for s in selected_xi if s['position'] == 4])

            need_def = 5 - xi_def_count
            need_mid = 5 - xi_mid_count
            need_fwd = 3 - xi_fwd_count

            # Add remaining 3 outfield bench players
            # Sort by cost (cheapest first) to stay within budget
            cheap_defs = sorted([p for p in defs if p['id'] not in selected_ids], key=lambda x: x['cost'])
            cheap_mids = sorted([p for p in mids if p['id'] not in selected_ids], key=lambda x: x['cost'])
            cheap_fwds = sorted([p for p in fwds if p['id'] not in selected_ids], key=lambda x: x['cost'])

            for p in cheap_defs:
                if need_def > 0 and p['id'] not in selected_ids and team_counts.get(p['team'], 0) < 3:
                    bench.append(p)
                    team_counts[p['team']] = team_counts.get(p['team'], 0) + 1
                    selected_ids.add(p['id'])
                    need_def -= 1

            for p in cheap_mids:
                if need_mid > 0 and p['id'] not in selected_ids and team_counts.get(p['team'], 0) < 3:
                    bench.append(p)
                    team_counts[p['team']] = team_counts.get(p['team'], 0) + 1
                    selected_ids.add(p['id'])
                    need_mid -= 1

            for p in cheap_fwds:
                if need_fwd > 0 and p['id'] not in selected_ids and team_counts.get(p['team'], 0) < 3:
                    bench.append(p)
                    team_counts[p['team']] = team_counts.get(p['team'], 0) + 1
                    selected_ids.add(p['id'])
                    need_fwd -= 1

            # Check if we got a full valid squad (15 players)
            if len(bench) != 4:
                continue

            # Check budget constraint
            total_cost = sum(p['cost'] for p in selected_xi) + sum(p['cost'] for p in bench)
            if total_cost > budget + 0.1:  # Small tolerance for rounding
                continue

            # Calculate starting XI EP
            formation_ep = sum(p['ep'] for p in selected_xi)

            # Pick the formation with highest starting XI EP
            if formation_ep > best_ep:
                best_ep = formation_ep
                best_starting_xi = selected_xi
                best_bench = bench
                best_formation = (n_def, n_mid, n_fwd)
                best_total_cost = total_cost

        if not best_starting_xi:
            return None

        # UPGRADE PHASE: Use remaining budget to improve the squad
        # This maximizes squad quality instead of leaving money on the table
        remaining_budget = budget - best_total_cost

        if remaining_budget > 0.5:  # If we have significant budget left
            best_starting_xi, best_bench, best_total_cost, best_ep = self._upgrade_squad_with_budget(
                best_starting_xi,
                best_bench,
                gkps, defs, mids, fwds,
                remaining_budget,
                budget
            )

        remaining_budget = budget - best_total_cost

        return {
            'formation': best_formation,
            'starting_xi': best_starting_xi,
            'bench': best_bench,
            'players': best_starting_xi,  # For backward compatibility
            'total_ep': best_ep,
            'total_cost': best_total_cost,
            'budget': budget,
            'remaining_budget': remaining_budget
        }

    def _calculate_haul_factor(self, player: pd.Series) -> float:
        """
        Calculate haul potential factor for captaincy decisions.

        Higher haul factor = higher ceiling for big scores (e.g., hat-tricks).
        This prioritizes attackers over defenders for TC.

        Returns: multiplier between 0.5 (low ceiling) and 2.0 (explosive upside)
        """
        # Position-based ceiling (FWD > MID > DEF > GKP)
        position = player.get('position', 0)
        position_weights = {
            4: 1.8,   # FWD - highest ceiling (hat-tricks, multiple returns)
            3: 1.5,   # MID - high ceiling (goals worth more points)
            2: 0.9,   # DEF - low ceiling (mainly clean sheets + occasional goal)
            1: 0.6    # GKP - lowest ceiling (clean sheets + saves)
        }
        base_multiplier = position_weights.get(position, 1.0)

        # Goal threat bonus (xGI90_est indicates attacking output per 90)
        xgi90 = player.get('xgi90_est', 0)
        if xgi90 > 0:
            # xGI90 of 0.8+ = elite attacker, scale up to 1.3x
            # xGI90 of 0.4 = average attacker, scale up to 1.15x
            # xGI90 of 0.1 = occasional threat, scale up to 1.05x
            goal_threat_boost = 1.0 + min(0.3, xgi90 * 0.375)
        else:
            goal_threat_boost = 1.0

        haul_factor = base_multiplier * goal_threat_boost

        # Cap between 0.5 and 2.0
        return max(0.5, min(2.0, haul_factor))

    def _find_best_tc_options(
        self,
        owned_ids: Set[int],
        player_data: pd.DataFrame,
        gw_start: int,
        gw_end: int,
        top_n: int = 3
    ) -> List[Dict]:
        """
        Find best Triple Captain options across all gameweeks.

        Prioritizes high-ceiling players (attackers) who could haul big scores.

        Returns top_n options sorted by haul-adjusted score, each with:
        {gw, player_id, player_name, ep, haul_factor, tc_score}
        """
        per_gw_ep = self._calculate_per_gw_ep(player_data, gw_start, gw_end)
        if not per_gw_ep:
            return []

        # Build list of all (player, gw, ep) combinations for owned players
        options = []
        for pid in owned_ids:
            if pid not in per_gw_ep:
                continue

            player = player_data[player_data['player_id'] == pid].iloc[0]
            player_name = player.get('name', 'Unknown')
            haul_factor = self._calculate_haul_factor(player)

            for gw, ep in per_gw_ep[pid].items():
                if ep > 0:  # Only consider GWs where player has a fixture
                    # TC score = base EP * haul factor (prioritizes high ceiling)
                    tc_score = ep * haul_factor

                    options.append({
                        'gw': gw,
                        'player_id': pid,
                        'player_name': player_name,
                        'ep': ep,
                        'haul_factor': haul_factor,
                        'tc_score': tc_score
                    })

        # Sort by TC score (haul-adjusted) descending
        options.sort(key=lambda x: x['tc_score'], reverse=True)
        return options[:top_n]

    def _find_best_bb_options(
        self,
        owned_ids: Set[int],
        player_data: pd.DataFrame,
        gw_start: int,
        gw_end: int,
        top_n: int = 3
    ) -> List[Dict]:
        """
        Find best Bench Boost options across all gameweeks.

        Returns top_n options sorted by bench EP, each with:
        {gw, bench_ep, bench_players}
        """
        per_gw_ep = self._calculate_per_gw_ep(player_data, gw_start, gw_end)
        if not per_gw_ep or len(owned_ids) < 15:
            return []

        # Get xmins for filtering likely starters
        owned_data = player_data[player_data['player_id'].isin(owned_ids)]

        # For each gameweek, identify bench (lowest 4 by EP)
        bb_options = []
        for gw in range(gw_start, gw_end + 1):
            gw_eps = []
            for pid in owned_ids:
                gw_ep = per_gw_ep.get(pid, {}).get(gw, 0)
                xmins = owned_data[owned_data['player_id'] == pid]['xmins'].iloc[0] if len(owned_data[owned_data['player_id'] == pid]) > 0 else 0
                gw_eps.append({'player_id': pid, 'ep': gw_ep, 'xmins': xmins})

            # Sort by EP descending - top 11 are starters, bottom 4 are bench
            gw_eps.sort(key=lambda x: x['ep'], reverse=True)
            bench = gw_eps[-4:]

            # Only count bench players likely to play (xmins >= 60)
            playing_bench = [p for p in bench if p['xmins'] >= self.config.bb_min_xmins]
            bench_ep = sum(p['ep'] for p in playing_bench)

            if bench_ep > 0:
                bb_options.append({
                    'gw': gw,
                    'bench_ep': bench_ep,
                    'num_playing': len(playing_bench)
                })

        # Sort by bench EP descending
        bb_options.sort(key=lambda x: x['bench_ep'], reverse=True)
        return bb_options[:top_n]

    def _find_best_fh_options(
        self,
        owned_ids: Set[int],
        player_data: pd.DataFrame,
        gw_start: int,
        gw_end: int,
        top_n: int = 3
    ) -> List[Dict]:
        """
        Find best Free Hit options by comparing owned XI vs optimal XI.

        Returns top_n options sorted by gap, each with:
        {gw, owned_xi_ep, optimal_xi_ep, gap}
        """
        per_gw_ep = self._calculate_per_gw_ep(player_data, gw_start, gw_end)
        if not per_gw_ep:
            return []

        fh_options = []
        for gw in range(gw_start, gw_end + 1):
            # Calculate owned XI EP for this GW
            owned_eps = [(pid, per_gw_ep.get(pid, {}).get(gw, 0)) for pid in owned_ids]
            owned_eps.sort(key=lambda x: x[1], reverse=True)
            owned_xi_ep = sum(ep for _, ep in owned_eps[:11])

            # Calculate optimal XI EP for this GW using shared team builder
            # This ensures consistent EP calculations and enforces max 3 per club
            optimal_result = self._build_optimal_xi_for_gw(gw, player_data, per_gw_ep)
            optimal_xi_ep = optimal_result.get('total_ep', 0) if optimal_result else 0

            gap = optimal_xi_ep - owned_xi_ep
            if gap > 0:
                fh_options.append({
                    'gw': gw,
                    'owned_xi_ep': owned_xi_ep,
                    'optimal_xi_ep': optimal_xi_ep,
                    'gap': gap
                })

        # Sort by gap descending
        fh_options.sort(key=lambda x: x['gap'], reverse=True)
        return fh_options[:top_n]

    def _get_best_captain(
        self,
        gw: int,
        owned_ids: Set[int],
        player_data: pd.DataFrame
    ) -> Optional[Dict]:
        """Get best captain for a gameweek from owned players only"""

        # ALWAYS filter by owned players when use_myteam is enabled
        if player_data.empty or not owned_ids:
            return None

        owned = player_data[player_data['player_id'].isin(owned_ids)]

        # Captain from MID/FWD (prefer attackers)
        captains = owned[owned['position'].isin([3, 4])]  # MID=3, FWD=4

        # Fallback to DEF if no MID/FWD available
        if captains.empty:
            captains = owned[owned['position'] == 2]  # DEF=2

        if captains.empty:
            return None

        best = captains.nlargest(1, 'ep_blend').iloc[0]

        return {
            'name': best.get('name', 'Unknown'),
            'ep': best.get('ep_blend', 0),
            'id': best.get('player_id')
        }
    
    def _calculate_bench_ep(
        self,
        gw: int,
        owned_ids: Set[int],
        player_data: pd.DataFrame
    ) -> float:
        """Calculate expected bench points from owned players only"""

        # ALWAYS calculate from owned players when use_myteam is enabled
        if not owned_ids or player_data.empty:
            return 0

        owned = player_data[player_data['player_id'].isin(owned_ids)]

        # Assume bottom 4 players by EP are bench
        if len(owned) < 15:
            return 0

        bench = owned.nsmallest(4, 'ep_blend')

        # Only count bench players likely to play (xmins >= threshold)
        playing = bench[bench['xmins'] >= self.config.bb_min_xmins]

        # Return sum of playing bench players (even if less than 4)
        # BB is still valuable if you have 2-3 good bench options
        return playing['ep_blend'].sum()
    
    def _calculate_fh_gap(
        self,
        gw: int,
        owned_ids: Set[int],
        player_data: pd.DataFrame
    ) -> float:
        """Calculate FH value based on gap between ideal XI and owned XI"""

        # ALWAYS calculate from owned players when use_myteam is enabled
        if not owned_ids or player_data.empty:
            return 0
        
        # Build ideal XI with formation constraints
        # 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD
        gks = player_data[player_data['position'] == 1].nlargest(1, 'ep_blend')
        defs = player_data[player_data['position'] == 2].nlargest(5, 'ep_blend')
        mids = player_data[player_data['position'] == 3].nlargest(5, 'ep_blend')  
        fwds = player_data[player_data['position'] == 4].nlargest(3, 'ep_blend')
        
        # Try different formations and pick best
        formations = [
            (3, 5, 2),  # 352
            (3, 4, 3),  # 343
            (4, 4, 2),  # 442
            (4, 3, 3),  # 433
            (4, 5, 1),  # 451
            (5, 3, 2),  # 532
            (5, 4, 1),  # 541
        ]
        
        best_ideal_ep = 0
        for n_def, n_mid, n_fwd in formations:
            formation_team = pd.concat([
                gks.head(1),
                defs.head(n_def),
                mids.head(n_mid),
                fwds.head(n_fwd)
            ])
            if len(formation_team) == 11:
                ep = formation_team['ep_blend'].sum()
                best_ideal_ep = max(best_ideal_ep, ep)
        
        # Owned XI - best 11 from owned players
        owned = player_data[player_data['player_id'].isin(owned_ids)]
        if len(owned) < 11:
            return best_ideal_ep
        
        # Must have at least 1 GK in owned XI
        owned_gks = owned[owned['position'] == 1].nlargest(1, 'ep_blend')
        owned_others = owned[owned['player_id'].isin(owned_gks['player_id']) == False].nlargest(10, 'ep_blend')
        owned_xi = pd.concat([owned_gks, owned_others])
        owned_xi_ep = owned_xi['ep_blend'].sum()
        
        return max(0, best_ideal_ep - owned_xi_ep)
    
    def _find_best_captain_dgw(
        self,
        dgws: List[int],
        owned_ids: Set[int],
        player_data: pd.DataFrame,
        excluded_gws: Optional[Set[int]] = None
    ) -> Optional[Dict]:
        """Find best captain for DGWs from owned players"""
        if not owned_ids or player_data.empty or not dgws:
            return None

        available_dgws = [gw for gw in dgws if gw not in (excluded_gws or set())]
        if not available_dgws:
            return None

        owned = player_data[player_data['player_id'].isin(owned_ids)]

        # Captain from MID/FWD (prefer attackers for DGW)
        captains = owned[owned['position'].isin([3, 4])]

        # Fallback to DEF if no MID/FWD
        if captains.empty:
            captains = owned[owned['position'] == 2]

        if captains.empty:
            return None

        best = captains.nlargest(1, 'ep_blend').iloc[0]

        # Estimate DGW points as 2x single gameweek (conservative)
        dgw_ep = best.get('ep_blend', 0) * 2.0

        return {
            'gw': available_dgws[0],
            'player': best.get('name', 'Unknown'),
            'ep': dgw_ep
        }
    
    def _find_best_bench_boost_dgw(
        self,
        dgws: List[int],
        owned_ids: Set[int],
        player_data: pd.DataFrame,
        excluded_gws: Optional[Set[int]] = None
    ) -> Optional[Dict]:
        """Find best BB opportunity in DGWs (or best SGW as fallback) from owned players"""
        if not owned_ids or player_data.empty:
            return None

        owned = player_data[player_data['player_id'].isin(owned_ids)]

        # Assume bottom 4 players by EP are bench
        if len(owned) < 15:
            return None

        bench = owned.nsmallest(4, 'ep_blend')

        available_dgws = [gw for gw in dgws if gw not in (excluded_gws or set())]

        if available_dgws:
            # Estimate DGW bench points as 2x single gameweek
            bench_ep = bench['ep_blend'].sum() * 2.0
            return {
                'gw': available_dgws[-1],
                'bench_ep': bench_ep,
                'dgw_players': 4
            }

        # SGW fallback: pick best remaining SGW (GW34-37 range)
        bench_ep = bench['ep_blend'].sum()
        excluded = excluded_gws or set()
        for gw in range(37, 27, -1):
            if gw not in excluded:
                return {
                    'gw': gw,
                    'bench_ep': bench_ep,
                    'dgw_players': 0
                }

        return None
    
    def _find_h2_wildcard_gw(
        self,
        dgws: List[int],
        fh_gw: Optional[int] = None,
        excluded_gws: Optional[Set[int]] = None
    ) -> Optional[int]:
        """Find optimal H2 wildcard timing.

        If a Free Hit GW is provided, place WC the week after (rebuild squad).
        Edge case: if FH is on GW38, place WC the week before instead.
        Otherwise fall back to 2 weeks before first DGW, or GW30 default.
        """
        excluded = excluded_gws or set()

        if fh_gw:
            # Rebuild after Free Hit
            wc_gw = fh_gw + 1 if fh_gw < 38 else fh_gw - 1
            if wc_gw not in excluded:
                return wc_gw

        if dgws:
            # Prepare before DGW run
            candidate = max(28, dgws[0] - 2)
            if candidate not in excluded:
                return candidate

        # Default: GW30
        if 30 not in excluded:
            return 30
        return None
    
    def _explain_strategy(self, recommendations: Dict, current_gw: int, show_teams: bool = False, used_chips: Set[str] = None):
        """Explain the strategy to user"""

        if used_chips is None:
            used_chips = set()

        print("\n" + "=" * 70)
        print("FPL 2025/26 CHIP STRATEGY - DOUBLE CHIPS SYSTEM")
        print("=" * 70)

        print(f"\n📅 Current: GW{current_gw}")
        print(f"⏰ H1 Deadline: GW19 (30 Dec)")
        print(f"🔄 H2 Starts: GW20")

        # Show used chips if any
        if used_chips:
            h1_used = [c for c in used_chips if c.startswith('H1_')]
            h2_used = [c for c in used_chips if c.startswith('H2_')]

            if h1_used:
                chip_names = [c.replace('H1_', '') for c in h1_used]
                print(f"\n✅ H1 Chips Already Used: {', '.join(chip_names)}")
            if h2_used:
                chip_names = [c.replace('H2_', '') for c in h2_used]
                print(f"✅ H2 Chips Already Used: {', '.join(chip_names)}")

        # Separate by half
        h1_chips = {k: v for k, v in recommendations.items() if 'H1_' in k}
        h2_chips = {k: v for k, v in recommendations.items() if 'H2_' in k}

        if current_gw <= 19:
            remaining = 19 - current_gw + 1
            h1_used_count = len([c for c in used_chips if c.startswith('H1_')])
            remaining_chips = 4 - h1_used_count
            print(f"\n⚠️  {remaining} gameweeks left to use {remaining_chips} H1 chips!")

            if remaining <= 3 and remaining_chips > 0:
                print("🚨 URGENT: Use your H1 chips NOW or lose them!")

        print("\n" + "-" * 35 + " FIRST HALF " + "-" * 35)
        print("Must use before GW19 deadline - Use it or lose it!\n")

        for chip_key, rec in h1_chips.items():
            self._print_chip_recommendation(rec)

        if not h1_chips and current_gw <= 19:
            h1_used_count = len([c for c in used_chips if c.startswith('H1_')])
            if h1_used_count == 4:
                print("✅ All H1 chips have been used!")
            else:
                print("⚠️ No specific recommendations yet - monitor your team performance")
                print("💡 Consider using chips around GW15-18 to avoid losing them")

        # Show H1 Free Hit team if requested (skip HOLD chips)
        if show_teams:
            h1_fh = h1_chips.get('H1_FH')
            if h1_fh and h1_fh.gameweek > 0:
                print("\n" + "-" * 60)
                print("📋 H1 FREE HIT TEAM PREVIEW")
                print("-" * 60 + "\n")
                fh_team = generate_free_hit_team(h1_fh.gameweek)
                print(fh_team)

        print("\n" + "-" * 35 + " SECOND HALF " + "-" * 35)
        print("Available from GW20 - Save for DGWs/BGWs!\n")

        for chip_key, rec in h2_chips.items():
            self._print_chip_recommendation(rec)

        # Show H2 Free Hit team if requested (skip HOLD chips)
        if show_teams:
            h2_fh = h2_chips.get('H2_FH')
            if h2_fh and h2_fh.gameweek > 0:
                print("\n" + "-" * 60)
                print("📋 H2 FREE HIT TEAM PREVIEW")
                print("-" * 60 + "\n")
                fh_team = generate_free_hit_team(h2_fh.gameweek)
                print(fh_team)

        # Summary of held chips
        held_chips = [
            rec for rec in recommendations.values()
            if rec.gameweek == 0
        ]
        if held_chips:
            held_names = [rec.chip_type.value.replace('H1_', '').replace('H2_', '') for rec in held_chips]
            print(f"\n⏳ Chips on HOLD ({len(held_chips)}): {', '.join(held_names)}")
            print("   These will be assigned once fixtures are confirmed.")

        print("\n" + "=" * 70)

        # Strategic notes
        print("\n💡 KEY STRATEGY POINTS:")
        if current_gw <= self.h1_deadline:
            print("• H1: Lower thresholds - don't be too greedy waiting for perfect spots")
            print("• H1: GW17-19 urgency - better to use than lose")

        if self._detected_dgws:
            dgw_str = ", ".join(f"GW{gw}" for gw in self._detected_dgws)
            print(f"• H2: Save TC/BB for DGWs ({dgw_str})")
        else:
            print("• H2: Save TC/BB for DGWs (none confirmed yet — monitor fixture announcements)")

        if self._detected_bgws:
            bgw_str = ", ".join(f"GW{gw}" for gw in self._detected_bgws)
            print(f"• H2: Save FH for BGWs ({bgw_str})")
        else:
            print("• H2: Save FH for BGWs (none confirmed yet — monitor fixture announcements)")

        if self._outstanding_fixtures:
            teams_str = ", ".join(
                f"{team} ({n} game{'s' if n > 1 else ''})"
                for team, n in sorted(self._outstanding_fixtures.items(), key=lambda x: -x[1])
            )
            print(f"• ⚠️  Teams with outstanding fixtures (expect future DGWs): {teams_str}")

        print("• WC: Time around international breaks or fixture swings")

        # Add hint about show-teams if not used (only for non-HOLD FH chips)
        if not show_teams:
            fh_recommendations = [
                rec for key, rec in recommendations.items()
                if 'FH' in key and rec.gameweek > 0
            ]
            if fh_recommendations:
                print("\n💡 TIP: Use --show-teams to see the full Free Hit XI for recommended gameweeks")
                print("   Or run: fpl chips free-hit --gw X")
        
    def _print_chip_recommendation(self, rec: ChipRecommendation):
        """Print a single chip recommendation"""

        chip_name = rec.chip_type.value.replace('H1_', '').replace('H2_', '')
        emoji = {'TC': '👑', 'BB': '💪', 'FH': '🎯', 'WC': '🔄'}.get(chip_name, '📌')

        if rec.gameweek == 0:
            # HOLD chip — no gameweek assigned yet
            print(f"{emoji} {chip_name}: HOLD (monitoring)")
            for reason in rec.reasons:
                print(f"   • {reason}")
            print()
            return

        print(f"{emoji} {chip_name}: GW{rec.gameweek}")

        if rec.urgency > 0.3:
            print(f"   ⚠️ Urgency: {'HIGH' if rec.urgency > 0.7 else 'MEDIUM'}")

        print(f"   Expected: {rec.expected_value:.1f} pts")
        print(f"   Confidence: {rec.confidence:.0%}")

        for reason in rec.reasons:
            print(f"   • {reason}")

        if rec.player_targets:
            print(f"   Target: {', '.join(rec.player_targets)}")
        print()


# ----------------------------- Entry Point -----------------------------

def plan_chips_2025(use_myteam: bool = True, explain: bool = True, show_teams: bool = False):
    """
    Generate chip strategy for FPL 2025/26 with double chips
    """
    strategy = FPL2025ChipStrategy()
    return strategy.plan_chips(use_myteam=use_myteam, explain=explain, show_teams=show_teams)


def generate_free_hit_team(gw: int) -> str:
    """
    Generate optimal Free Hit 15-man squad for a specific gameweek

    Args:
        gw: Gameweek number to generate Free Hit team for

    Returns:
        Formatted string with team details
    """
    strategy = FPL2025ChipStrategy()

    # Load player data
    player_data = strategy._load_player_data()

    # Get budget from current squad value
    budget = strategy._get_squad_value()

    # Calculate per-GW EP for this gameweek only
    per_gw_ep = strategy._calculate_per_gw_ep(player_data, gw, gw)

    # Build optimal 15-man squad with actual budget
    result = strategy._build_optimal_xi_for_gw(gw, player_data, per_gw_ep, budget=budget)

    if not result or not result.get('starting_xi'):
        return f"Could not generate Free Hit team for GW{gw}. No valid formation found."

    # Get starting XI and bench
    starting_xi = result['starting_xi']
    bench = result.get('bench', [])
    captain, vice = _select_captain_for_fh(starting_xi)

    # Format output
    output = []
    output.append("=" * 70)
    output.append(f"OPTIMAL FREE HIT TEAM - GW{gw}")
    output.append("=" * 70)
    output.append("")

    formation = result['formation']
    output.append(f"Formation: {formation[0]}-{formation[1]}-{formation[2]}")
    output.append(f"Starting XI Expected Points: {result['total_ep']:.1f}")

    # Calculate bench EP
    bench_ep = sum(p['ep'] for p in bench)
    output.append(f"Bench Expected Points: {bench_ep:.1f}")

    # Budget information
    output.append("")
    output.append(f"Available Budget: £{result['budget']:.1f}m (your squad value)")
    output.append(f"Total Squad Cost: £{result['total_cost']:.1f}m (15 players)")
    output.append(f"Remaining: £{result['remaining_budget']:.1f}m")
    output.append("")

    # Display starting XI
    output.append("─" * 70)
    output.append("STARTING XI")
    output.append("─" * 70)

    # Group starting XI by position
    by_position = {1: [], 2: [], 3: [], 4: []}
    for p in starting_xi:
        by_position[p['position']].append(p)

    position_names = {1: 'GOALKEEPER', 2: 'DEFENDERS', 3: 'MIDFIELDERS', 4: 'FORWARDS'}

    for pos_id in [1, 2, 3, 4]:
        if by_position[pos_id]:
            output.append(f"{position_names[pos_id]}:")
            for p in by_position[pos_id]:
                cap_marker = ""
                if captain and p['id'] == captain['id']:
                    cap_marker = " (C)"
                elif vice and p['id'] == vice['id']:
                    cap_marker = " (VC)"
                output.append(f"  {p['name']} ({p['team']}) - {p['ep']:.2f} EP, £{p['cost']:.1f}m{cap_marker}")
            output.append("")

    # Display bench
    output.append("─" * 70)
    output.append("BENCH")
    output.append("─" * 70)

    bench_by_position = {1: [], 2: [], 3: [], 4: []}
    for p in bench:
        bench_by_position[p['position']].append(p)

    bench_position_names = {1: 'GOALKEEPER', 2: 'DEFENDERS', 3: 'MIDFIELDERS', 4: 'FORWARDS'}

    for pos_id in [1, 2, 3, 4]:
        if bench_by_position[pos_id]:
            output.append(f"{bench_position_names[pos_id]}:")
            for p in bench_by_position[pos_id]:
                output.append(f"  {p['name']} ({p['team']}) - {p['ep']:.2f} EP, £{p['cost']:.1f}m")
            output.append("")

    # Team distribution (all 15 players)
    output.append("─" * 70)
    output.append("SQUAD SUMMARY")
    output.append("─" * 70)

    all_players = starting_xi + bench
    team_counts = {}
    for p in all_players:
        team_counts[p['team']] = team_counts.get(p['team'], 0) + 1

    output.append("Team Distribution:")
    for team, count in sorted(team_counts.items(), key=lambda x: x[1], reverse=True):
        output.append(f"  {team}: {count} players")
    output.append("")

    # Squad composition
    pos_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for p in all_players:
        pos_counts[p['position']] += 1

    output.append("Squad Composition:")
    output.append(f"  Goalkeepers: {pos_counts[1]}")
    output.append(f"  Defenders: {pos_counts[2]}")
    output.append(f"  Midfielders: {pos_counts[3]}")
    output.append(f"  Forwards: {pos_counts[4]}")
    output.append("")

    # Captaincy recommendations
    if captain:
        output.append(f"RECOMMENDED CAPTAIN: {captain['name']} (Haul-adjusted: {captain['haul_score']:.2f})")
    if vice:
        output.append(f"RECOMMENDED VICE-CAPTAIN: {vice['name']} (Haul-adjusted: {vice['haul_score']:.2f})")

    output.append("=" * 70)

    return "\n".join(output)


def _select_captain_for_fh(players: List[Dict]) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Select best captain and vice-captain from Free Hit XI using haul factor logic

    Args:
        players: List of player dicts with position, ep, name, id, xgi90

    Returns:
        Tuple of (captain, vice_captain) player dicts with added haul_score, or (None, None)
    """
    if not players:
        return None, None

    # Position-based haul factors
    position_weights = {
        4: 1.8,   # FWD - highest ceiling
        3: 1.5,   # MID - high ceiling
        2: 0.9,   # DEF - low ceiling
        1: 0.6    # GKP - lowest ceiling
    }

    # Calculate haul scores for all players
    scored_players = []
    for p in players:
        # Base haul factor from position
        base_multiplier = position_weights.get(p['position'], 1.0)

        # Boost from xGI90 if available (goal threat)
        # Note: xgi90 might not be in the player dict from _build_optimal_xi_for_gw
        # So we'll just use position-based for now
        haul_factor = base_multiplier

        # Haul-adjusted score
        haul_score = p['ep'] * haul_factor

        player_copy = p.copy()
        player_copy['haul_score'] = haul_score
        scored_players.append(player_copy)

    # Sort by haul score descending
    scored_players.sort(key=lambda x: x['haul_score'], reverse=True)

    captain = scored_players[0] if len(scored_players) > 0 else None
    vice = scored_players[1] if len(scored_players) > 1 else None

    return captain, vice


def analyze_free_hit_all_gws(gw_start: Optional[int] = None, gw_end: int = 19) -> str:
    """
    Analyze Free Hit value for all gameweeks in a range

    Args:
        gw_start: Starting gameweek (defaults to current GW)
        gw_end: Ending gameweek

    Returns:
        Formatted string with EP comparison table
    """
    strategy = FPL2025ChipStrategy()

    # Get current gameweek if not specified
    if gw_start is None:
        gw_start = strategy._get_current_gw()

    # Load team and player data
    owned_ids = strategy._load_myteam()
    player_data = strategy._load_player_data()

    # Calculate per-GW EP for the range
    per_gw_ep = strategy._calculate_per_gw_ep(player_data, gw_start, gw_end)

    if not per_gw_ep:
        return "Could not calculate EP for gameweeks. Missing data."

    # Build analysis table
    output = []
    output.append("=" * 70)
    output.append(f"FREE HIT ANALYSIS: GW{gw_start} - GW{gw_end}")
    output.append("=" * 70)
    output.append("")
    output.append(f"{'GW':<4} {'Your XI EP':<12} {'Optimal XI EP':<15} {'Delta':<10} {'Worth FH?':<10}")
    output.append("-" * 70)

    results = []
    for gw in range(gw_start, gw_end + 1):
        # Calculate owned XI EP for this GW
        owned_eps = [(pid, per_gw_ep.get(pid, {}).get(gw, 0)) for pid in owned_ids]
        owned_eps.sort(key=lambda x: x[1], reverse=True)
        owned_xi_ep = sum(ep for _, ep in owned_eps[:11])

        # Calculate optimal XI EP using the constraint-respecting builder
        optimal_result = strategy._build_optimal_xi_for_gw(gw, player_data, per_gw_ep)
        optimal_xi_ep = optimal_result.get('total_ep', 0) if optimal_result else 0

        delta = optimal_xi_ep - owned_xi_ep

        # Determine if worth using FH (threshold of ~6+ points gain)
        worth_it = "✓ YES" if delta >= 6.0 else ("Maybe" if delta >= 4.0 else "No")

        results.append({
            'gw': gw,
            'owned': owned_xi_ep,
            'optimal': optimal_xi_ep,
            'delta': delta,
            'worth_it': worth_it
        })

        output.append(f"{gw:<4} {owned_xi_ep:<12.1f} {optimal_xi_ep:<15.1f} {delta:<10.1f} {worth_it:<10}")

    output.append("-" * 70)

    # Find the best gameweek
    best_gw = max(results, key=lambda x: x['delta'])
    output.append("")
    output.append(f"BEST FREE HIT GAMEWEEK: GW{best_gw['gw']}")
    output.append(f"  Your XI:    {best_gw['owned']:.1f} EP")
    output.append(f"  Optimal XI: {best_gw['optimal']:.1f} EP")
    output.append(f"  Gain:       +{best_gw['delta']:.1f} EP")
    output.append("")
    output.append("=" * 70)
    output.append("")
    output.append("💡 NOTES:")
    output.append("  • Delta = Optimal XI EP - Your XI EP")
    output.append("  • Generally worth using Free Hit if delta ≥ 6 points")
    output.append("  • Consider fixtures, injuries, and other chip timing")
    output.append("  • Run 'fpl chips free-hit --gw X' to see the optimal team")

    return "\n".join(output)