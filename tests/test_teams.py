"""Canonical team identity.

The regression these guard against: football-data.org's "Manchester City FC" and the FPL
API's "Man City" used to normalise to different keys, so Elo ratings, clean-sheet stats and
team form were computed under one key and looked up under another.
"""

import pytest

from src.fpl_predict.data.teams import (
    CANONICAL_TEAMS,
    bootstrap_team_names,
    bootstrap_team_slugs,
    canonical_team,
    normalize,
    team_display,
)

# (football-data.org spelling, FPL API spelling, archive spelling)
SPELLINGS = [
    ("Manchester City FC", "Man City", "Man City"),
    ("Manchester United FC", "Man Utd", "Man Utd"),
    ("Tottenham Hotspur FC", "Spurs", "Spurs"),
    ("Nottingham Forest FC", "Nott'm Forest", "Nott'm Forest"),
    ("Brighton & Hove Albion FC", "Brighton", "Brighton"),
    ("Newcastle United FC", "Newcastle", "Newcastle"),
    ("West Ham United FC", "West Ham", "West Ham"),
    ("Wolverhampton Wanderers FC", "Wolves", "Wolves"),
    ("Leeds United FC", "Leeds", "Leeds"),
    ("Sunderland AFC", "Sunderland", "Sunderland"),
    ("AFC Bournemouth", "Bournemouth", "Bournemouth"),
    ("Sheffield United FC", "Sheffield Utd", "Sheffield Utd"),
    ("Ipswich Town FC", "Ipswich Town", "Ipswich"),
    ("Hull City AFC", "Hull City", "Hull"),
    ("Coventry City FC", "Coventry City", "Coventry"),
    ("Luton Town FC", "Luton Town", "Luton"),
    ("Leicester City FC", "Leicester", "Leicester"),
    ("West Bromwich Albion FC", "West Brom", "West Brom"),
]


@pytest.mark.parametrize("fd,fpl,archive", SPELLINGS)
def test_all_sources_agree_on_one_key(fd, fpl, archive):
    assert canonical_team(fd) == canonical_team(fpl) == canonical_team(archive)


@pytest.mark.parametrize("fd,fpl,archive", SPELLINGS)
def test_resolved_slug_is_a_known_team(fd, fpl, archive):
    assert canonical_team(fpl) in CANONICAL_TEAMS


def test_the_specific_pair_that_used_to_diverge():
    # Under the old normalisation these became "manchestercity" and "mancity".
    assert normalize("Manchester City FC") != normalize("Man City")
    assert canonical_team("Manchester City FC") == canonical_team("Man City") == "man_city"


def test_short_codes_resolve():
    assert canonical_team("MCI") == "man_city"
    assert canonical_team("NFO") == "nottm_forest"


def test_unknown_name_falls_back_to_its_own_key_without_raising():
    assert canonical_team("Barnsley Athletic") == "barnsleyathletic"


def test_blank_names_are_empty_not_an_error():
    assert canonical_team(None) == ""
    assert canonical_team("") == ""
    assert canonical_team("   ") == ""


def test_display_names_round_trip():
    for slug, name in CANONICAL_TEAMS.items():
        assert canonical_team(name) == slug
        assert team_display(slug) == name


def test_display_falls_back_to_the_slug():
    assert team_display("not_a_team") == "not_a_team"


def test_bootstrap_helpers_use_short_name_when_name_is_blank():
    bootstrap = {
        "teams": [
            {"id": 13, "name": "Man City", "short_name": "MCI"},
            {"id": 7, "name": "", "short_name": "CHE"},
        ]
    }
    assert bootstrap_team_slugs(bootstrap) == {13: "man_city", 7: "chelsea"}
    assert bootstrap_team_names(bootstrap) == {13: "Man City", 7: "CHE"}
