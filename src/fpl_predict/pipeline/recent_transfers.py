"""
Handle recent transfers that need integration time.
Scrapes transfer data to identify recent moves.
"""

import difflib
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import re
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)


def detect_club_changes() -> Dict[int, float]:
    """xMins factors for players who changed club and have not yet featured for the new one.

    Replaces a hardcoded dict of August-2025 transfers ("Gyokeres to Arsenal, 15 days ago"
    and so on) that returned early whenever it matched, so the scraper never ran and every
    later season inherited that window's business.

    A change of club is detected by comparing the player's current club against the club
    they played the most minutes for last season, matched on the stable FPL `code`.

    The penalty is for being *unproven at the new club*, not for the move being recent,
    because "days since transfer" is not knowable from the API and stops being the relevant
    question the moment a player takes the pitch:

    | Situation                                     | Factor            |
    |:----------------------------------------------|:------------------|
    | Season not started                            | 1.0, no penalty   |
    | Has minutes for the new club                  | 1.0, no penalty   |
    | No minutes, 1 gameweek gone                   | 0.75              |
    | No minutes, 2 gameweeks gone                  | 0.55              |
    | No minutes, 3+ gameweeks gone                 | 0.40              |

    A player with no minutes for several gameweeks is a rotation risk on the evidence, so the
    factor keeps biting rather than expiring on a timer.
    """
    factors: Dict[int, float] = {}

    try:
        from ..config import current_season
        from ..data.fpl_api import get_bootstrap
        from ..data.fpl_history import load_history
        from ..data.teams import bootstrap_team_slugs

        bootstrap = get_bootstrap()
        id_to_slug = bootstrap_team_slugs(bootstrap)

        events = bootstrap.get("events", [])
        gws_played = sum(1 for e in events if e.get("finished"))
        if gws_played == 0:
            log.info("Season not started; new signings have had a full pre-season, no penalty")
            return {}

        prev = load_history(seasons=[current_season() - 1])
        if prev is None or prev.empty:
            log.info("No previous-season history available; skipping club-change detection")
            return {}

        played = prev[prev["minutes"] > 0]
        if played.empty:
            return {}
        # Most-appeared-for club, so a mid-season mover is not attributed to whichever club
        # happened to come last in the file.
        last_club = (
            played.groupby(["player_code", "team_slug"])
            .size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
            .drop_duplicates("player_code")
            .set_index("player_code")["team_slug"]
            .to_dict()
        )

        if gws_played >= 3:
            factor = 0.40
        elif gws_played == 2:
            factor = 0.55
        else:
            factor = 0.75

        # During pre-season the bootstrap `minutes` field still holds last season's total
        # rather than zero, so it cannot be read as "minutes this season" without a check.
        # A value equal to last season's total is carryover, not evidence of an appearance.
        prev_minutes = played.groupby("player_code")["minutes"].sum().to_dict()

        for player in bootstrap["elements"]:
            code = player.get("code")
            if code is None:
                continue
            was = last_club.get(int(code))
            if not was:
                continue  # no top-flight minutes last season; handled as a new player
            now = id_to_slug.get(player["team"])
            if not now or was == now:
                continue
            mins = int(player.get("minutes") or 0)
            if mins > 0 and mins != int(prev_minutes.get(int(code), -1)):
                continue  # already featured for the new club, the minutes speak for themselves
            factors[player["id"]] = factor
            log.info(
                "Unproven signing: %s (%s -> %s), no minutes after %d GW, factor %.2f",
                player["web_name"],
                was,
                now,
                gws_played,
                factor,
            )

    except Exception as e:
        log.debug(f"Could not analyse FPL data for club changes: {e}")

    return factors


def scrape_recent_transfers() -> List[Tuple[str, int]]:
    """
    Scrape recent Premier League transfers from reliable sources.
    Uses multiple fallback options.

    Only reached when `detect_club_changes` finds nothing, i.e. mid-season moves that last
    season's archive cannot distinguish from a player staying put.

    Returns:
        List of (player_name, days_since_transfer) tuples
    """
    recent_transfers: List[Tuple[str, int]] = []

    try:
        # Fallback to web scraping
        # Try official Premier League news
        url = "https://www.premierleague.com/transfers"

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")

            # Look for transfer tables
            tables = soup.find_all("table", class_="wikitable")

            # Get current date for comparison
            today = datetime.now()
            cutoff_date = today - timedelta(days=14)  # Only care about last 2 weeks

            for table in tables:
                rows = table.find_all("tr")
                for row in rows[1:]:  # Skip header
                    cells = row.find_all("td")
                    if len(cells) >= 3:
                        # Try to extract player name and date
                        player_cell = cells[0]
                        date_cell = cells[-1]  # Often date is in last column

                        player_name = player_cell.get_text(strip=True)
                        date_text = date_cell.get_text(strip=True)

                        # Parse date (various formats possible)
                        transfer_date = parse_transfer_date(date_text)
                        if transfer_date and transfer_date > cutoff_date:
                            days_ago = (today - transfer_date).days
                            recent_transfers.append((player_name, days_ago))
                            log.info(f"Found recent transfer: {player_name} ({days_ago} days ago)")

        # Fallback: Try Sky Sports Transfer Centre
        if not recent_transfers:
            recent_transfers = scrape_sky_sports_transfers()

        # Additional fallback: BBC Sport
        if not recent_transfers:
            recent_transfers = scrape_bbc_transfers()

    except Exception as e:
        log.warning(f"Failed to scrape transfer data: {e}")

    return recent_transfers


def scrape_sky_sports_transfers() -> List[Tuple[str, int]]:
    """
    Fallback scraper for Sky Sports transfer data.
    """
    transfers = []
    try:
        url = "https://www.skysports.com/premier-league-transfers"
        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")

            # Sky Sports specific parsing
            transfer_items = soup.find_all("div", class_="transfer-item")
            today = datetime.now()

            for item in transfer_items:
                player_elem = item.find("h4", class_="transfer-item__player")
                date_elem = item.find("span", class_="transfer-item__date")

                if player_elem and date_elem:
                    player_name = player_elem.get_text(strip=True)
                    date_text = date_elem.get_text(strip=True)

                    transfer_date = parse_transfer_date(date_text)
                    if transfer_date:
                        days_ago = (today - transfer_date).days
                        if days_ago <= 14:  # Last 2 weeks
                            transfers.append((player_name, days_ago))

    except Exception as e:
        log.debug(f"Sky Sports scraping failed: {e}")

    return transfers


def scrape_bbc_transfers() -> List[Tuple[str, int]]:
    """
    Fallback scraper for BBC Sport transfer data.
    """
    transfers = []
    try:
        # BBC Sport Premier League transfers
        url = "https://www.bbc.com/sport/football/transfers"
        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")

            # BBC specific parsing - they often use data attributes
            transfer_list = soup.find_all("article", {"data-testid": "transfer-item"})
            today = datetime.now()

            for item in transfer_list:
                # Extract player and date from BBC's structure
                player_elem = item.find("h3")
                date_elem = item.find("time")

                if player_elem and date_elem:
                    player_name = player_elem.get_text(strip=True)
                    date_str = date_elem.get("datetime", "")

                    if date_str:
                        try:
                            transfer_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                            days_ago = (today - transfer_date).days
                            if days_ago <= 14:
                                transfers.append((player_name, days_ago))
                        except:
                            pass

    except Exception as e:
        log.debug(f"BBC scraping failed: {e}")

    return transfers


def parse_transfer_date(date_text: str) -> Optional[datetime]:
    """
    Parse various date formats from transfer news.
    """
    if not date_text:
        return None

    # Clean the text
    date_text = date_text.strip()

    # Try different date formats
    formats = [
        "%d %B %Y",  # 15 August 2025
        "%d %b %Y",  # 15 Aug 2025
        "%d/%m/%Y",  # 15/08/2025
        "%Y-%m-%d",  # 2025-08-15
        "%B %d, %Y",  # August 15, 2025
        "%b %d, %Y",  # Aug 15, 2025
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_text, fmt)
        except:
            continue

    # Check for relative dates
    if "today" in date_text.lower():
        return datetime.now()
    elif "yesterday" in date_text.lower():
        return datetime.now() - timedelta(days=1)
    elif "days ago" in date_text.lower():
        # Extract number of days
        match = re.search(r"(\d+)\s*days?\s*ago", date_text.lower())
        if match:
            days = int(match.group(1))
            return datetime.now() - timedelta(days=days)

    return None


def match_transfer_to_fpl(transfer_name: str, fpl_players: list) -> Optional[int]:
    """
    Match a transfer name to an FPL player ID.

    Every tier requires the match to be unambiguous before accepting it: a scraped name that
    matches more than one plausible player (e.g. two players sharing a surname — Silva,
    Fernandes, Rodrigues are all common) is rejected rather than resolved to whichever
    candidate happens to come first in bootstrap element order, since a wrong match applies
    the "just transferred, discount minutes" penalty to an unrelated player.

    Args:
        transfer_name: Name from transfer news
        fpl_players: List of FPL player elements

    Returns:
        Player ID if a single unambiguous match is found, None otherwise
    """
    # Clean the transfer name
    transfer_name = transfer_name.strip()

    # Remove common suffixes
    for suffix in [" (loan)", " (permanent)", " (free)", " (undisclosed)"]:
        transfer_name = transfer_name.replace(suffix, "")

    if not transfer_name:
        return None

    transfer_lower = transfer_name.lower()
    transfer_last = transfer_name.split()[-1].lower()

    def full_name(p: dict) -> str:
        return f"{p.get('first_name', '')} {p.get('second_name', '')}".strip()

    # Tier 1: exact match on web_name (FPL's own display name, unique per player)
    exact = [p for p in fpl_players if p["web_name"].lower() == transfer_lower]
    if len(exact) == 1:
        return exact[0]["id"]

    # Tier 2: exact match on full registered name
    full_matches = [p for p in fpl_players if full_name(p).lower() == transfer_lower]
    if len(full_matches) == 1:
        return full_matches[0]["id"]

    # Tier 3: last-name match, restricted to players not yet widely owned (a just-arrived
    # signing plausibly is; an established player plausibly isn't, which is itself part of
    # the signal here) — but only trusted if exactly one such candidate exists.
    surname_candidates = [
        p for p in fpl_players
        if p.get("second_name", "").lower() == transfer_last
        and float(p.get("selected_by_percent", 0) or 0) < 5.0
    ]
    if len(surname_candidates) == 1:
        return surname_candidates[0]["id"]
    if len(surname_candidates) > 1:
        log.warning(
            "Transfer name '%s' matches %d low-ownership players by surname (%s); "
            "skipping rather than guessing",
            transfer_name,
            len(surname_candidates),
            [p["web_name"] for p in surname_candidates],
        )
        return None

    # Tier 4: fuzzy match on full name among low-ownership players, using a real similarity
    # score rather than a bare substring check — `"Silva" in "Silvao"` or `"Rui" in "Ruiz"`
    # would previously match unrelated players sharing a fragment. Requires both a high
    # absolute similarity and a clear margin over the next-best candidate, so two similarly
    # spelled names don't get resolved to an arbitrary pick either.
    low_ownership = [p for p in fpl_players if float(p.get("selected_by_percent", 0) or 0) < 5.0]
    scored = sorted(
        (
            (difflib.SequenceMatcher(None, transfer_lower, full_name(p).lower()).ratio(), p)
            for p in low_ownership
        ),
        key=lambda t: t[0],
        reverse=True,
    )
    if scored and scored[0][0] >= 0.75:
        if len(scored) == 1 or (scored[0][0] - scored[1][0]) >= 0.15:
            return scored[0][1]["id"]
        log.warning(
            "Transfer name '%s' has multiple close fuzzy matches (top %.2f vs next %.2f: "
            "%s vs %s); skipping rather than guessing",
            transfer_name, scored[0][0], scored[1][0],
            scored[0][1]["web_name"], scored[1][1]["web_name"],
        )

    return None


def get_recent_transfer_adjustments() -> Dict[int, float]:
    """
    Returns adjustment factors for players who changed club and are unproven there.

    The club-change comparison against last season's archive is the primary signal because
    it is exact. Scraping transfer news is only a fallback for mid-season moves the archive
    cannot see yet, since a January signing still shows last-season minutes at the club they
    are already at.

    Returns:
        Dict mapping player_id to adjustment factor (0.0 to 1.0)
    """
    adjustments = detect_club_changes()
    if adjustments:
        log.info("Club-change detection produced %d xMins adjustments", len(adjustments))
        return adjustments

    try:
        # Get FPL player data
        from ..data.fpl_api import get_bootstrap

        bootstrap = get_bootstrap()

        # Scrape recent transfers
        log.info("Scraping recent transfer data...")
        recent_transfers = scrape_recent_transfers()

        if not recent_transfers:
            log.info("No recent transfers found from web scraping")
            # Fallback to manual list if scraping fails completely
            recent_transfers = get_manual_transfers()

        # Match transfers to FPL players and apply adjustments
        for transfer_name, days_ago in recent_transfers:
            player_id = match_transfer_to_fpl(transfer_name, bootstrap["elements"])

            if player_id:
                # Calculate adjustment factor based on days since transfer
                if days_ago <= 3:
                    factor = 0.2  # Very recent: 20% of normal minutes
                elif days_ago <= 7:
                    factor = 0.35  # One week: 35% of normal minutes
                elif days_ago <= 14:
                    factor = 0.6  # Two weeks: 60% of normal minutes
                else:
                    continue  # More than 2 weeks, likely integrated

                adjustments[player_id] = factor

                # Log the adjustment
                for p in bootstrap["elements"]:
                    if p["id"] == player_id:
                        log.info(
                            f"Transfer adjustment: {p['web_name']} "
                            f"({days_ago} days since transfer) - factor={factor:.1f}"
                        )
                        break

    except Exception as e:
        log.warning(f"Could not apply transfer adjustments: {e}")
        # Use manual fallback
        adjustments = get_manual_transfer_fallback()

    return adjustments


def get_manual_transfers() -> List[Tuple[str, int]]:
    """
    Manual fallback list of known recent transfers.
    Update this when scraping fails.
    """
    # Format: (player_name, days_since_transfer)
    # ONLY add VERIFIED transfers
    return [
        # Example: ("Grealish", 5),  # If Grealish actually transferred to Everton
        # User should add actual recent transfers here
    ]


def get_manual_transfer_fallback() -> Dict[int, float]:
    """
    Manual fallback adjustments when scraping fails completely.
    """
    try:
        from ..data.fpl_api import get_bootstrap

        bootstrap = get_bootstrap()

        # Manual list with known player names
        # ONLY add VERIFIED transfers
        manual_adjustments = {
            # Example: "Grealish": 0.3,  # If actually transferred recently
            # User should add actual recent transfers here
        }

        adjustments = {}
        for player_name, factor in manual_adjustments.items():
            for p in bootstrap["elements"]:
                if player_name in p["web_name"]:
                    adjustments[p["id"]] = factor
                    log.info(f"Manual adjustment: {player_name} - factor={factor:.1f}")
                    break

        return adjustments

    except:
        return {}


def apply_transfer_adjustments(xmins_map: Dict[int, float]) -> Dict[int, float]:
    """
    Apply adjustments to xMins for recent transfers.

    Args:
        xmins_map: Current xMins predictions

    Returns:
        Updated xMins map with transfer adjustments
    """
    adjustments = get_recent_transfer_adjustments()

    if not adjustments:
        return xmins_map

    updated = xmins_map.copy()
    for player_id, factor in adjustments.items():
        if player_id in updated:
            original = updated[player_id]
            updated[player_id] = original * factor
            log.info(
                f"Adjusted player {player_id} xMins: {original:.0f} -> {updated[player_id]:.0f}"
            )

    return updated
