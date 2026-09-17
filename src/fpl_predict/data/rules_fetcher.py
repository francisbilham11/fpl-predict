"""Scoring rules for the season in progress.

`bootstrap-static.game_config.scoring` is the authoritative table and changes the moment
FPL changes the game, so it is the primary source here. Scraping the help page is kept as a
fallback for the rare case where the API shape moves, and the YAML snapshot under
`data/rules_cache/` is the last resort.

Two divisors and two thresholds are not in the API and are encoded as constants:

| Rule                                  | Value           | Where it comes from        |
|:--------------------------------------|:----------------|:---------------------------|
| Saves per point                       | 3               | fixed rule, never varied   |
| Goals conceded per penalty            | 2               | fixed rule, never varied   |
| Defensive contribution, DEF threshold | 10 CBIT         | scraped, constant fallback |
| Defensive contribution, MID/FWD       | 12 CBIRT        | scraped, constant fallback |

Position keys are GKP/DEF/MID/FWD throughout, matching `element_types` short names.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup

from ..utils.cache import DATA, RULES_CACHE
from ..utils.io import write_json
from ..utils.logging import get_logger
from .fpl_api import get_bootstrap

log = get_logger(__name__)

PL_DEFCON_URL = "https://www.premierleague.com/en/news/4361991"
FPL_RULES_URL = "https://fantasy.premierleague.com/help/rules"

SAVES_PER_POINT = 3
CONCEDED_PER_PENALTY = 2
DEFCON_DEF_THRESHOLD = 10
DEFCON_MID_FWD_THRESHOLD = 12

POSITIONS = ("GKP", "DEF", "MID", "FWD")


def _get(url: str) -> str:
    headers = {"User-Agent": "fpl-predict/0.5 (+https://example.org)"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.text


def parse_defensive_contribution(html: str) -> dict[str, Any]:
    """Read the defensive contribution thresholds off the Premier League explainer."""
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" \n")
    m_def = re.search(r"defend\w*.{0,200}?combined total of\s*(\d+)", text, re.I | re.S)
    m_mid = re.search(r"midfield\w*\s+and\s+forward\w*.{0,200}?(\d+)", text, re.I | re.S)
    m_pts = re.search(r"(two|2)\s+FPL points", text, re.I)
    return {
        "defensive_contribution": {
            "points": 2 if m_pts else 2,
            "defender_threshold_cbit": int(m_def.group(1)) if m_def else DEFCON_DEF_THRESHOLD,
            "mid_fwd_threshold_cbirt": int(m_mid.group(1)) if m_mid else DEFCON_MID_FWD_THRESHOLD,
            "capped_per_match": True,
        }
    }


def rules_from_bootstrap(bootstrap: dict | None = None) -> dict[str, Any]:
    """Build the scoring table from the FPL API.

    Raises ValueError if `game_config.scoring` is absent or unusable, so callers can fall
    back rather than silently carrying last season's numbers forward.
    """
    bs = bootstrap or get_bootstrap()
    scoring = (bs.get("game_config") or {}).get("scoring")
    if not scoring:
        raise ValueError("bootstrap-static has no game_config.scoring")

    def by_pos(key: str) -> dict[str, int]:
        raw = scoring.get(key)
        if not isinstance(raw, dict):
            raise ValueError(f"game_config.scoring.{key} is not a per-position mapping")
        return {p: int(raw.get(p, 0)) for p in POSITIONS}

    goals = by_pos("goals_scored")
    clean_sheets = by_pos("clean_sheets")
    conceded = by_pos("goals_conceded")
    defcon = by_pos("defensive_contribution")

    short = int(scoring.get("short_play", 1))
    long = int(scoring.get("long_play", 2))

    rules: dict[str, Any] = {
        # Appearance points. `any` is awarded for any minutes, `sixty` is the *total* for
        # 60+, so the incremental value of reaching 60 is sixty - any.
        "minutes": {
            "any": short,
            "sixty": long,
            "sixty_extra": long - short,
            # legacy labels, kept because the README table renderer reads them
            "1-59": short,
            "60+": long,
        },
        "goals": goals,
        "assist": int(scoring.get("assists", 3)),
        "clean_sheet": {p: clean_sheets[p] for p in POSITIONS if clean_sheets[p]},
        "goals_conceded_per_unit": conceded,
        "goals_conceded_unit": CONCEDED_PER_PENALTY,
        "goals_conceded_per_2": {"GKP_DEF": conceded["GKP"]},
        "saves_per_unit": int(scoring.get("saves", 1)),
        "saves_unit": SAVES_PER_POINT,
        "saves_per_3": int(scoring.get("saves", 1)),
        "penalty_save": int(scoring.get("penalties_saved", 5)),
        "penalty_miss": int(scoring.get("penalties_missed", -2)),
        "yellow_card": int(scoring.get("yellow_cards", -1)),
        "red_card": int(scoring.get("red_cards", -3)),
        "own_goal": int(scoring.get("own_goals", -2)),
        "bonus": [1, 2, 3],
        "defensive_contribution_by_pos": defcon,
        "source": "fpl_api",
    }

    # Squad and transfer rules live alongside the scoring table and are worth carrying so
    # the optimizer and chip planner stop hardcoding them.
    game_rules = (bs.get("game_config") or {}).get("rules") or bs.get("game_settings") or {}
    rules["squad"] = {
        "size": int(game_rules.get("squad_squadsize", 15)),
        "play": int(game_rules.get("squad_squadplay", 11)),
        "team_limit": int(game_rules.get("squad_team_limit", 3)),
        "budget": int(game_rules.get("squad_total_spend", 1000)),
        "max_free_transfers": int(game_rules.get("max_extra_free_transfers", 4)) + 1,
        "sell_on_fee": float(game_rules.get("transfers_sell_on_fee", 0.5)),
    }
    rules["squad_composition"] = {
        et.get("singular_name_short", ""): int(et.get("squad_select", 0))
        for et in bs.get("element_types", [])
        if et.get("singular_name_short") in POSITIONS
    }
    return rules


def parse_rules_table(html: str) -> dict[str, Any]:
    """Fallback: scrape the FPL help page."""
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" \n")
    rules: dict[str, Any] = {
        "minutes": {},
        "goals": {},
        "assist": None,
        "clean_sheet": {},
        "saves_per_3": None,
        "penalty_save": None,
        "penalty_miss": None,
        "goals_conceded_per_2": {"GKP_DEF": None},
        "yellow_card": None,
        "red_card": None,
        "own_goal": None,
        "bonus": [1, 2, 3],
        "source": "help_page",
    }

    def m(p):
        return re.search(p, text, re.I)

    k = m(r"1-59 minutes\s*(\+?)(\d+)")
    short = int(k.group(2)) if k else 1
    k = m(r"60 minutes or more\s*(\+?)(\d+)")
    long = int(k.group(2)) if k else 2
    rules["minutes"] = {
        "any": short,
        "sixty": long,
        "sixty_extra": long - short,
        "1-59": short,
        "60+": long,
    }
    for pos, lab in [
        ("GKP", "Goalkeeper"),
        ("DEF", "Defender"),
        ("MID", "Midfielder"),
        ("FWD", "Forward"),
    ]:
        g = m(rf"{lab} goal\s*(\+?)(\d+)")
        if g:
            rules["goals"][pos] = int(g.group(2))
    for pos, lab in [("GKP", "Goalkeeper"), ("DEF", "Defender"), ("MID", "Midfielder")]:
        c = m(rf"{lab} clean sheet\s*(\+?)(\d+)")
        if c:
            rules["clean_sheet"][pos] = int(c.group(2))
    a = m(r"Assist\s*(\+?)(\-?\d+)")
    rules["assist"] = int(a.group(2)) if a else 3
    s = m(r"Save\s*\(every\s*3\)\s*(\+?)(\-?\d+)")
    rules["saves_per_3"] = int(s.group(2)) if s else 1
    ps = m(r"Penalty save\s*(\+?)(\-?\d+)")
    rules["penalty_save"] = int(ps.group(2)) if ps else 5
    pm = m(r"Penalty miss\s*(\+?)(\-?\d+)")
    rules["penalty_miss"] = int(pm.group(2)) if pm else -2
    gc = m(r"Goalkeeper/Defender goals conceded\s*\(every 2\)\s*(\-?\d+)")
    rules["goals_conceded_per_2"]["GKP_DEF"] = int(gc.group(1)) if gc else -1
    y = m(r"Yellow card\s*(\+?)(\-?\d+)")
    rules["yellow_card"] = int(y.group(2)) if y else -1
    r = m(r"Red card\s*(\+?)(\-?\d+)")
    rules["red_card"] = int(r.group(2)) if r else -3
    og = m(r"Own goal\s*(\+?)(\-?\d+)")
    rules["own_goal"] = int(og.group(2)) if og else -2
    rules["saves_unit"] = SAVES_PER_POINT
    rules["goals_conceded_unit"] = CONCEDED_PER_PENALTY
    return rules


def _cached_snapshot() -> dict[str, Any]:
    snaps = sorted(RULES_CACHE.glob("*.yaml"))
    if not snaps:
        raise FileNotFoundError(f"no rules snapshot in {RULES_CACHE}")
    snap = snaps[-1]
    log.warning("Using cached rules snapshot: %s", snap.name)
    compiled = yaml.safe_load(Path(snap).read_text()) or {}
    compiled["source"] = f"snapshot:{snap.stem}"
    return compiled


def fetch_scoring_rules() -> dict[str, Any]:
    """Resolve the scoring rules, API first, and cache them to data/processed/rules.json."""
    compiled: dict[str, Any] = {}

    try:
        compiled = rules_from_bootstrap()
        log.info(
            "Scoring rules from FPL API (GKP goal=%s, DEF goal=%s, MID goal=%s, FWD goal=%s)",
            compiled["goals"]["GKP"],
            compiled["goals"]["DEF"],
            compiled["goals"]["MID"],
            compiled["goals"]["FWD"],
        )
    except Exception as e:
        log.warning("Could not read scoring rules from the API: %s", e)
        try:
            compiled = parse_rules_table(_get(FPL_RULES_URL))
        except Exception as e2:
            log.warning("Rules table parse failed: %s", e2)

    # Thresholds are not in the API, so always try the explainer page.
    try:
        compiled.update(parse_defensive_contribution(_get(PL_DEFCON_URL)))
    except Exception as e:
        log.warning("Defensive contribution parse failed (%s); using known thresholds", e)
        compiled.setdefault(
            "defensive_contribution",
            {
                "points": 2,
                "defender_threshold_cbit": DEFCON_DEF_THRESHOLD,
                "mid_fwd_threshold_cbirt": DEFCON_MID_FWD_THRESHOLD,
                "capped_per_match": True,
            },
        )

    essentials = [
        compiled.get("minutes"),
        compiled.get("goals"),
        compiled.get("assist"),
        compiled.get("clean_sheet"),
        compiled.get("defensive_contribution"),
    ]
    if not all(essentials):
        compiled = _cached_snapshot()

    write_json(compiled, DATA / "processed" / "rules.json")
    return compiled


def render_scoring_table_md(rules: dict[str, Any]) -> str:
    rows = []
    mins = rules.get("minutes", {})
    rows.append(["Minutes 1-59", mins.get("any", mins.get("1-59", "?"))])
    rows.append(["Minutes 60+", mins.get("sixty", mins.get("60+", "?"))])
    for pos in POSITIONS:
        if pos in rules.get("goals", {}):
            rows.append([f"Goal ({pos})", rules["goals"][pos]])
    if rules.get("assist") is not None:
        rows.append(["Assist", rules["assist"]])
    for pos in ("GKP", "DEF", "MID"):
        if pos in rules.get("clean_sheet", {}):
            rows.append([f"Clean sheet ({pos})", rules["clean_sheet"][pos]])
    gc = rules.get("goals_conceded_per_2", {}).get("GKP_DEF")
    if gc is not None:
        rows.append(["Goals conceded (GKP/DEF, per 2)", gc])
    for k in ["saves_per_3", "penalty_save", "penalty_miss", "yellow_card", "red_card", "own_goal"]:
        if rules.get(k) is not None:
            label = {
                "saves_per_3": "Saves (per 3)",
                "penalty_save": "Penalty save",
                "penalty_miss": "Penalty miss",
                "yellow_card": "Yellow card",
                "red_card": "Red card",
                "own_goal": "Own goal",
            }[k]
            rows.append([label, rules[k]])
    dc = rules.get("defensive_contribution", {})
    if dc:
        rows.append(
            [
                "Defensive contribution (DEF threshold)",
                f"{dc.get('points', '?')} @ {dc.get('defender_threshold_cbit', '?')} CBIT",
            ]
        )
        rows.append(
            [
                "Defensive contribution (MID/FWD threshold)",
                f"{dc.get('points', '?')} @ {dc.get('mid_fwd_threshold_cbirt', '?')} CBIRT",
            ]
        )
    df = pd.DataFrame(rows, columns=["Action", "Points"])
    return df.to_markdown(index=False)


def update_readme_scoring_table(readme_path: Path) -> None:
    start, end = "<!-- SCORING_TABLE_START -->", "<!-- SCORING_TABLE_END -->"
    rules = fetch_scoring_rules()
    table_md = render_scoring_table_md(rules)
    md = Path(readme_path).read_text(encoding="utf-8") if Path(readme_path).exists() else ""
    if start not in md:
        md += f"\n\n{start}\n{end}\n"
    pattern = re.compile(f"{re.escape(start)}.*?{re.escape(end)}", re.S)
    new_md = pattern.sub(f"{start}\n\n{table_md}\n\n{end}", md)
    Path(readme_path).write_text(new_md, encoding="utf-8")
    log.info("README scoring table updated.")
