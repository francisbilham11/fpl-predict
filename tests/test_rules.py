"""Scoring rules and the appearance-points path into expected points.

Two regressions these guard against:

- Goalkeeper goals were fixed at 6 from a cached YAML snapshot; FPL raised them to 10 for
  2026/27 and the pipeline never noticed.
- `_appearance_points` looked for keys ("any"/"sixty", "appearance_1pt") that neither the
  snapshot nor the scraper produced, so it returned (0, 0) and appearance points were
  missing from every player's expected points.
"""

import pytest

from src.fpl_predict.data.rules_fetcher import (
    POSITIONS,
    parse_defensive_contribution,
    render_scoring_table_md,
    rules_from_bootstrap,
)
from src.fpl_predict.models.aggregate_points import (
    _appearance_points,
    _assist_points,
    _cs_points_for_pos,
    _goal_points_for_pos,
)

# Shape of bootstrap-static.game_config for 2026/27, trimmed to what the parser reads.
BOOTSTRAP = {
    "game_config": {
        "scoring": {
            "long_play": 2,
            "short_play": 1,
            "goals_scored": {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4},
            "assists": 3,
            "clean_sheets": {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0},
            "goals_conceded": {"GKP": -1, "DEF": -1, "MID": 0, "FWD": 0},
            "saves": 1,
            "penalties_saved": 5,
            "penalties_missed": -2,
            "yellow_cards": -1,
            "red_cards": -3,
            "own_goals": -2,
            "bonus": 1,
            "defensive_contribution": {"GKP": 0, "DEF": 2, "MID": 2, "FWD": 2},
        },
        "rules": {
            "squad_squadsize": 15,
            "squad_squadplay": 11,
            "squad_team_limit": 3,
            "squad_total_spend": 1000,
            "max_extra_free_transfers": 4,
            "transfers_sell_on_fee": 0.5,
        },
    },
    "element_types": [
        {"singular_name_short": "GKP", "squad_select": 2},
        {"singular_name_short": "DEF", "squad_select": 5},
        {"singular_name_short": "MID", "squad_select": 5},
        {"singular_name_short": "FWD", "squad_select": 3},
    ],
}


@pytest.fixture
def rules():
    return rules_from_bootstrap(BOOTSTRAP)


def test_goalkeeper_goals_come_from_the_api_not_a_snapshot(rules):
    assert rules["goals"]["GKP"] == 10, "2026/27 raised GKP goals from 6 to 10"
    assert _goal_points_for_pos(rules, "GKP") == 10


@pytest.mark.parametrize("pos,pts", [("GKP", 10), ("DEF", 6), ("MID", 5), ("FWD", 4)])
def test_goal_points_by_position(rules, pos, pts):
    assert _goal_points_for_pos(rules, pos) == pts


@pytest.mark.parametrize("pos,pts", [("GKP", 4), ("DEF", 4), ("MID", 1), ("FWD", 0)])
def test_clean_sheet_points_by_position(rules, pos, pts):
    assert _cs_points_for_pos(rules, pos) == pts


def test_assist_points(rules):
    assert _assist_points(rules) == 3


def test_appearance_points_are_not_zero(rules):
    any_pts, extra_at_60 = _appearance_points(rules)
    assert (any_pts, extra_at_60) == (1.0, 1.0)
    # A 60+ appearance is worth 2 in total, which is what the game awards.
    assert any_pts + extra_at_60 == 2.0


def test_appearance_points_read_the_legacy_snapshot_shape():
    # The scraped YAML snapshot used these labels and produced (0, 0) before the fix.
    legacy = {"minutes": {"1-59": 1, "60+": 2}}
    any_pts, extra_at_60 = _appearance_points(legacy)
    assert any_pts == 1.0
    assert any_pts + extra_at_60 == 2.0


def test_appearance_points_handle_a_hypothetical_rule_change():
    changed = {"minutes": {"any": 1, "sixty": 3, "sixty_extra": 2}}
    assert _appearance_points(changed) == (1.0, 2.0)


def test_squad_rules_are_read_from_the_api(rules):
    assert rules["squad"]["team_limit"] == 3
    assert rules["squad"]["budget"] == 1000
    assert rules["squad"]["max_free_transfers"] == 5  # 1 base + 4 extra
    assert rules["squad_composition"] == {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}


def test_missing_scoring_block_raises_so_callers_can_fall_back():
    with pytest.raises(ValueError):
        rules_from_bootstrap({"game_config": {}})


def test_rendered_table_reflects_the_api_values(rules):
    md = render_scoring_table_md(rules)
    gkp_row = next(line for line in md.splitlines() if "Goal (GKP)" in line)
    assert "10" in gkp_row


def test_rendered_table_includes_the_defcon_rows_once_thresholds_are_merged(rules):
    # `rules_from_bootstrap` covers the scoring table; the thresholds come from the separate
    # explainer scrape, which `fetch_scoring_rules` merges in.
    merged = dict(rules)
    merged.update(parse_defensive_contribution(""))
    md = render_scoring_table_md(merged)
    assert "10 CBIT" in md
    assert "12 CBIRT" in md


def test_defensive_contribution_thresholds_default_when_unparseable():
    dc = parse_defensive_contribution("<html><body>nothing useful</body></html>")[
        "defensive_contribution"
    ]
    assert dc["points"] == 2
    assert dc["defender_threshold_cbit"] == 10
    assert dc["mid_fwd_threshold_cbirt"] == 12


def test_positions_constant_matches_the_api_short_names():
    assert set(POSITIONS) == {"GKP", "DEF", "MID", "FWD"}
