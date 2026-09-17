"""Canonical team identity across the data sources this project reads.

Every source spells clubs differently: the FPL API says "Man City", football-data.org
says "Manchester City FC", football-data.co.uk says "Man City" but "Man United", and the
community gameweek archive says "Man Utd". Historically the pipeline normalised names by
stripping punctuation and the "fc"/"afc" suffix, which left eight of the twenty clubs with
two different keys and silently broke every join keyed on team name (Elo ratings, clean
sheet stats, team form).

`canonical_team` maps any of those spellings to a stable slug. Unknown names fall back to
their normalised form with a warning rather than raising, so a newly promoted club whose
spelling we have not seen degrades to "own key" instead of taking the pipeline down.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable

from ..utils.logging import get_logger

log = get_logger(__name__)

# slug -> display name (FPL's short form, which is what reports show)
CANONICAL_TEAMS: Dict[str, str] = {
    "arsenal": "Arsenal",
    "aston_villa": "Aston Villa",
    "bournemouth": "Bournemouth",
    "brentford": "Brentford",
    "brighton": "Brighton",
    "burnley": "Burnley",
    "cardiff": "Cardiff",
    "chelsea": "Chelsea",
    "coventry": "Coventry",
    "crystal_palace": "Crystal Palace",
    "everton": "Everton",
    "fulham": "Fulham",
    "huddersfield": "Huddersfield",
    "hull": "Hull",
    "ipswich": "Ipswich",
    "leeds": "Leeds",
    "leicester": "Leicester",
    "liverpool": "Liverpool",
    "luton": "Luton",
    "man_city": "Man City",
    "man_utd": "Man Utd",
    "middlesbrough": "Middlesbrough",
    "newcastle": "Newcastle",
    "norwich": "Norwich",
    "nottm_forest": "Nott'm Forest",
    "sheffield_utd": "Sheffield Utd",
    "southampton": "Southampton",
    "stoke": "Stoke",
    "sunderland": "Sunderland",
    "spurs": "Spurs",
    "swansea": "Swansea",
    "watford": "Watford",
    "west_brom": "West Brom",
    "west_ham": "West Ham",
    "wolves": "Wolves",
}

# Every spelling seen across the FPL API, football-data.org, football-data.co.uk and the
# vaastav gameweek archive. Keys are matched after `normalize`, so case, punctuation and
# the "fc"/"afc" suffix are already gone by the time we look here.
_ALIAS_SOURCE: Dict[str, Iterable[str]] = {
    "arsenal": ["Arsenal", "Arsenal FC", "ARS"],
    "aston_villa": ["Aston Villa", "Aston Villa FC", "AVL"],
    "bournemouth": ["Bournemouth", "AFC Bournemouth", "BOU"],
    "brentford": ["Brentford", "Brentford FC", "BRE"],
    "brighton": [
        "Brighton",
        "Brighton & Hove Albion",
        "Brighton & Hove Albion FC",
        "Brighton and Hove Albion",
        "BHA",
    ],
    "burnley": ["Burnley", "Burnley FC", "BUR"],
    "cardiff": ["Cardiff", "Cardiff City", "Cardiff City FC", "CAR"],
    "chelsea": ["Chelsea", "Chelsea FC", "CHE"],
    "coventry": ["Coventry", "Coventry City", "Coventry City FC", "COV"],
    "crystal_palace": ["Crystal Palace", "Crystal Palace FC", "CRY"],
    "everton": ["Everton", "Everton FC", "EVE"],
    "fulham": ["Fulham", "Fulham FC", "FUL"],
    "huddersfield": ["Huddersfield", "Huddersfield Town", "Huddersfield Town AFC", "HUD"],
    "hull": ["Hull", "Hull City", "Hull City AFC", "HUL"],
    "ipswich": ["Ipswich", "Ipswich Town", "Ipswich Town FC", "IPS"],
    "leeds": ["Leeds", "Leeds United", "Leeds United FC", "LEE"],
    "leicester": ["Leicester", "Leicester City", "Leicester City FC", "LEI"],
    "liverpool": ["Liverpool", "Liverpool FC", "LIV"],
    "luton": ["Luton", "Luton Town", "Luton Town FC", "LUT"],
    "man_city": ["Man City", "Manchester City", "Manchester City FC", "MCI"],
    "man_utd": [
        "Man Utd",
        "Man United",
        "Manchester United",
        "Manchester United FC",
        "MUN",
    ],
    "middlesbrough": ["Middlesbrough", "Middlesbrough FC", "MID"],
    "newcastle": ["Newcastle", "Newcastle United", "Newcastle United FC", "NEW"],
    "norwich": ["Norwich", "Norwich City", "Norwich City FC", "NOR"],
    "nottm_forest": [
        "Nott'm Forest",
        "Nottingham Forest",
        "Nottingham Forest FC",
        "Nottm Forest",
        "NFO",
    ],
    "sheffield_utd": [
        "Sheffield Utd",
        "Sheffield United",
        "Sheffield United FC",
        "Sheff Utd",
        "SHU",
    ],
    "southampton": ["Southampton", "Southampton FC", "SOU"],
    "stoke": ["Stoke", "Stoke City", "Stoke City FC", "STK"],
    "sunderland": ["Sunderland", "Sunderland AFC", "SUN"],
    "spurs": ["Spurs", "Tottenham", "Tottenham Hotspur", "Tottenham Hotspur FC", "TOT"],
    "swansea": ["Swansea", "Swansea City", "Swansea City AFC", "SWA"],
    "watford": ["Watford", "Watford FC", "WAT"],
    "west_brom": [
        "West Brom",
        "West Bromwich Albion",
        "West Bromwich Albion FC",
        "WBA",
    ],
    "west_ham": ["West Ham", "West Ham United", "West Ham United FC", "WHU"],
    "wolves": [
        "Wolves",
        "Wolverhampton",
        "Wolverhampton Wanderers",
        "Wolverhampton Wanderers FC",
        "WOL",
    ],
}

_SUFFIXES = re.compile(r"\b(fc|afc)\b")


def normalize(name: Any) -> str:
    """Lowercase, drop the club suffix and reduce to alphanumerics."""
    s = str(name or "").lower().replace("&", "and")
    s = _SUFFIXES.sub("", s)
    return re.sub(r"[^a-z0-9]+", "", s)


ALIASES: Dict[str, str] = {}
for _slug, _names in _ALIAS_SOURCE.items():
    ALIASES[_slug] = _slug
    for _n in _names:
        ALIASES[normalize(_n)] = _slug

_warned: set[str] = set()


def canonical_team(name: Any) -> str:
    """Resolve any spelling of a club to its canonical slug.

    Unknown names return their normalised form so joins stay self-consistent, and warn
    once so a new spelling shows up in the logs instead of silently splitting a team in
    two the way the old name handling did.
    """
    key = normalize(name)
    if not key:
        return ""
    slug = ALIASES.get(key)
    if slug:
        return slug
    if key not in _warned:
        _warned.add(key)
        log.warning("Unknown team name %r (normalised %r); using it as its own key", name, key)
    return key


def team_display(slug: str) -> str:
    """Human-readable name for a canonical slug."""
    return CANONICAL_TEAMS.get(slug, slug)


def bootstrap_team_slugs(bootstrap: dict) -> Dict[int, str]:
    """Map FPL team id -> canonical slug for the season the bootstrap describes."""
    out: Dict[int, str] = {}
    for t in bootstrap.get("teams", []):
        name = t.get("name") or t.get("short_name")
        out[int(t["id"])] = canonical_team(name)
    return out


def bootstrap_team_names(bootstrap: dict) -> Dict[int, str]:
    """Map FPL team id -> display name, falling back to short_name when name is blank."""
    out: Dict[int, str] = {}
    for t in bootstrap.get("teams", []):
        name = (t.get("name") or "").strip() or (t.get("short_name") or "")
        out[int(t["id"])] = name
    return out
