# src/fpl_predict/transfer/optimizer.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Set, Optional
import pandas as pd
import numpy as np

import requests

from ..utils.logging import get_logger
from ..utils.io import read_parquet
from ..utils.cache import PROC

log = get_logger(__name__)

# Try to import optimization libraries
try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False
    log.debug("PuLP not available - LP optimization disabled")

# ---- Squad rules ----
POSITIONS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_CLUB = 3
BUDGET = 1000  # tenths of a million, i.e. £100.0m

FORMATIONS = {
    "343": {"GKP": 1, "DEF": 3, "MID": 4, "FWD": 3},
    "352": {"GKP": 1, "DEF": 3, "MID": 5, "FWD": 2},
    "442": {"GKP": 1, "DEF": 4, "MID": 4, "FWD": 2},
    "451": {"GKP": 1, "DEF": 4, "MID": 5, "FWD": 1},
    "433": {"GKP": 1, "DEF": 4, "MID": 3, "FWD": 3},
    "541": {"GKP": 1, "DEF": 5, "MID": 4, "FWD": 1},
    "532": {"GKP": 1, "DEF": 5, "MID": 3, "FWD": 2},
}

# mild depth heuristics to avoid obvious backups
EXPECTED_TEAM_STARTERS = {"DEF": 4, "MID": 4, "FWD": 2}


@dataclass
class Player:
    id: int
    name: str
    pos: str
    team: int
    cost: int              # tenths of a million
    xmins: float           # 0..90
    ep_base: float         # single-GW, minutes-adjusted baseline
    xgi90: float           # optional per-90 attack involvement
    team_att: float        # optional team attack strength ~1.0
    ep_seq: List[float]    # EP per GW across horizon H
    ep1: float             # ep_seq[0]
    eph: float             # horizon-weighted sum
    
    # Additional fields for LP optimization
    form: float = 0.0
    selected_by: float = 0.0  # Ownership %
    value_score: float = 0.0
    is_differential: bool = False
    injury_risk: float = 0.0
    rotation_risk: float = 0.0
    team_strength: float = 1.0  # Team quality multiplier


# ------------------- data loaders -------------------
def _fetch_bootstrap() -> dict:
    r = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=30)
    r.raise_for_status()
    return r.json()


def _get_next_gameweek() -> int:
    """Get the next gameweek number from FPL API."""
    try:
        bootstrap = _fetch_bootstrap()
        # Find the next gameweek (is_next=True) or current gameweek (is_current=True)
        next_gw = next((e['id'] for e in bootstrap.get('events', []) if e.get('is_next')), None)
        if next_gw:
            return next_gw
        # Fallback to current gameweek if next is not set
        current_gw = next((e['id'] for e in bootstrap.get('events', []) if e.get('is_current')), None)
        if current_gw:
            return current_gw
        # Final fallback
        return 1
    except Exception as e:
        log.warning(f"Could not get current gameweek: {e}")
        return 1


def _fetch_fixtures() -> list:
    r = requests.get("https://fantasy.premierleague.com/api/fixtures/", timeout=30)
    r.raise_for_status()
    return r.json()


def _load_xmins() -> Dict[int, float]:
    try:
        df = read_parquet(PROC / "xmins.parquet")
        return {int(pid): float(xm) for pid, xm in zip(df["player_id"], df["xmins"])}
    except Exception as e:
        log.warning("xMins parquet missing; defaulting to 70 mins. (%s)", e)
        return {}


def _load_ep_extras() -> tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """Return ep_adjusted (or ep_blend or ep_model), xgi90_est (optional), team_att (optional)."""
    try:
        df = read_parquet(PROC / "exp_points.parquet")
        # Prioritize ep_adjusted which accounts for playing time
        if "ep_adjusted" in df.columns:
            ep_col = "ep_adjusted"
        elif "ep_blend" in df.columns:
            ep_col = "ep_blend"
        else:
            ep_col = "ep_model"
        ep_map = {int(r.player_id): float(getattr(r, ep_col)) for _, r in df.iterrows()}
        xgi_map = {int(r.player_id): float(getattr(r, "xgi90_est", 0.0)) for _, r in df.iterrows()}
        tatt_map = {int(r.player_id): float(getattr(r, "team_att", 1.0)) for _, r in df.iterrows()}
        return ep_map, xgi_map, tatt_map
    except Exception:
        return {}, {}, {}


def _pergw_factors(team_id: int, H: int) -> list[float]:
    """Fixture difficulty → factors ~[0.84..1.16]."""
    try:
        fx = _fetch_fixtures()
    except Exception:
        return [1.0] * H
    ups = [f for f in fx if (not f.get("finished")) and f.get("event") and (f.get("team_h")==team_id or f.get("team_a")==team_id)]
    ups.sort(key=lambda f: f.get("event", 999))
    facs = []
    for f in ups[:H]:
        if f.get("team_h") == team_id:
            diff = int(f.get("team_h_difficulty") or 3)
        else:
            diff = int(f.get("team_a_difficulty") or 3)
        fac = 1.0 + (3 - diff) * 0.08
        facs.append(max(0.80, min(1.20, fac)))
    while len(facs) < H:
        facs.append(1.0)
    return facs


def _parse_weights(hweights: str, H: int) -> list[float]:
    if hweights:
        try:
            ws = [float(x.strip()) for x in hweights.split(",") if x.strip()]
            if len(ws) >= H:
                return ws[:H]
            while len(ws) < H:
                ws.append(ws[-1] if ws else 1.0)
            return ws
        except Exception:
            pass
    w = [1.0]
    for _ in range(1, H):
        w.append(w[-1] * 0.9)
    return w


def _candidate_pool(
    js: dict,
    xmins_map: dict[int, float],
    H: int,
    weights: list[float],
    fdr_weight: float,
) -> List[Player]:
    ep_blend, xgi90_est, team_att = _load_ep_extras()

    pool: List[Player] = []
    for e in js["elements"]:
        pid = int(e["id"])
        pos = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}[e["element_type"]]
        team_id = int(e["team"])
        # ep_blend now contains ep_adjusted which already accounts for minutes
        ep0 = float(ep_blend.get(pid, float(e.get("ep_next") or 0.0)))
        xm = float(xmins_map.get(pid, 70.0))
        
        # STRUCTURAL CONSTRAINT FIX: Prevent forcing weak bench players into XI
        # Key insight: We must field 11 players, so having 3+ cheap players from
        # weak teams forces us to start at least one (limiting flexibility)
        cost = int(e.get("now_cost") or 0)
        
        # Mark players as "bench-only" if they're cheap + low EP
        # This helps the optimizer understand they shouldn't be in the XI
        is_bench_quality = (cost <= 40 and ep0 < 2.0 and pos != "GKP")
        
        # Skip non-playing fodder entirely
        if cost <= 45 and xm < 60.0 and pos != "GKP":
            continue
        
        # Skip ultra-cheap players with very low EP (likely not playing)
        if cost <= 40 and ep0 < 1.2 and pos != "GKP":
            continue
            
        # Skip backup GKPs with 0 playing time (ep_adjusted handles this)
        if pos == "GKP" and ep0 == 0.0:
            continue
        
        # Note: ep0 is already minutes-adjusted via ep_adjusted column
        xgi90 = float(xgi90_est.get(pid, 0.0))
        t_att = float(team_att.get(pid, 1.0))

        facs = _pergw_factors(team_id, H)
        ep_seq = [max(0.0, ep0 * (1.0 + fdr_weight * (f - 1.0))) for f in facs]
        wsum = sum(w * v for w, v in zip(weights, ep_seq))

        # Add a penalty for bench-quality players to discourage starting them
        # This addresses the structural constraint issue
        ep_penalty = 1.0
        if is_bench_quality:
            # Apply 30% penalty to EP for optimization purposes
            # This makes the optimizer prefer other options for the starting XI
            ep_penalty = 0.7
            ep_seq = [ep * ep_penalty for ep in ep_seq]
            wsum = sum(w * v for w, v in zip(weights, ep_seq))
        
        pool.append(
            Player(
                id=pid,
                name=e["web_name"],
                pos=pos,
                team=team_id,
                cost=int(e.get("now_cost") or 0),
                xmins=xm,
                ep_base=ep0,
                xgi90=xgi90,
                team_att=t_att,
                ep_seq=ep_seq,
                ep1=ep_seq[0] if ep_seq else 0.0,
                eph=wsum,
            )
        )
    return pool


# ------------------- helpers -------------------
def _rank_key(p: Player) -> float:
    return 1.0 * p.ep_base + 0.2 * (p.xmins / 90.0) + 0.1 * p.xgi90


def _adjust_xmins_with_depth(pool: List[Player], nonstarter_xmins: float, gk_backup_xmins: float) -> None:
    by_team: Dict[int, List[Player]] = {}
    for p in pool:
        by_team.setdefault(p.team, []).append(p)

    for players in by_team.values():
        # GK: one clear starter → others trimmed
        gks = [p for p in players if p.pos == "GKP"]
        if gks:
            starter = max(gks, key=_rank_key)
            for g in gks:
                if g.id != starter.id:
                    g.xmins = min(g.xmins, gk_backup_xmins)

        # Outfield: trim deepest bench
        for pos in ["DEF", "MID", "FWD"]:
            ps = [p for p in players if p.pos == pos]
            if not ps:
                continue
            ps.sort(key=_rank_key, reverse=True)
            starters = EXPECTED_TEAM_STARTERS[pos]
            for idx, p in enumerate(ps):
                if idx >= starters:
                    p.xmins = min(p.xmins, nonstarter_xmins)

    # ultra-cheap nonstarters → tiny minutes
    for p in pool:
        if p.pos != "GKP" and p.cost <= 45 and p.ep_base <= 2.0:
            p.xmins = min(p.xmins, max(10.0, 0.5 * nonstarter_xmins))


def _club_counts(players: List[Player]) -> Dict[int, int]:
    cc: Dict[int, int] = {}
    for p in players:
        cc[p.team] = cc.get(p.team, 0) + 1
    return cc


def _total_spent(players: List[Player]) -> int:
    return sum(p.cost for p in players)


# ------------------- bench template (NEW: guarantees cap) -------------------
def _cheapest_by_pos(pool: List[Player], pos: str, exclude_ids: Set[int], used_clubs: Dict[int, int]) -> Optional[Player]:
    cands = [p for p in pool if p.pos == pos and p.id not in exclude_ids]
    cands.sort(key=lambda z: (z.cost, -z.ep1))
    for q in cands:
        if used_clubs.get(q.team, 0) >= MAX_PER_CLUB:
            continue
        return q
    return None


def _choose_cheap_bench_template(pool: List[Player], bench_budget: int) -> List[Player]:
    """
    Pick 4 cheap bench players under club caps:
      - exactly 1 GK (cheapest)
      - 3 outfielders across DEF/MID/FWD with minimum total cost
    Return list of 4 Players. Guarantees cost <= bench_budget if feasible in the pool.
    """
    used: List[Player] = []
    exclude: Set[int] = set()
    clubs: Dict[int, int] = {}

    # 1) cheapest GK
    gk = _cheapest_by_pos(pool, "GKP", exclude, clubs)
    if not gk:
        raise RuntimeError("No goalkeepers in pool.")
    used.append(gk); exclude.add(gk.id); clubs[gk.team] = clubs.get(gk.team, 0) + 1

    # 2) pick cheapest 3 outfielders (any of DEF/MID/FWD) under club caps
    # STRUCTURAL FIX: Prioritize better cheap defenders to avoid triple weak teams
    outfield = [p for p in pool if p.pos in {"DEF", "MID", "FWD"} and p.id not in exclude]
    
    # Sort with smarter logic: still cheap but prefer higher EP to avoid forcing weak players into XI
    # This addresses the "must field 11 players" constraint
    def bench_score(p):
        # For £4.0m players, heavily weight EP to avoid weak teams
        # For £4.5m+ players, still prefer cheaper
        if p.cost <= 40:
            return (p.cost, -p.ep1 * 10)  # Heavily weight EP for £4.0m
        else:
            return (p.cost, -p.ep1)
    
    outfield.sort(key=bench_score)
    
    # Track how many from potentially weak teams (low EP defenders)
    weak_team_players = 0
    
    for q in outfield:
        if len(used) == 4:
            break
        if clubs.get(q.team, 0) >= MAX_PER_CLUB:
            continue
        
        # STRUCTURAL CONSTRAINT: If this is a £4.0m defender with <2.0 EP
        # and we already have 2 such players, skip to avoid triple weak team
        if q.pos == "DEF" and q.cost <= 40 and q.ep1 < 2.0:
            if weak_team_players >= 2:
                continue  # Look for slightly better option
            weak_team_players += 1
        
        used.append(q)
        exclude.add(q.id)
        clubs[q.team] = clubs.get(q.team, 0) + 1

    # If still < 4, fill regardless of club caps (should be rare)
    if len(used) < 4:
        for q in outfield:
            if q.id in exclude:
                continue
            used.append(q)
            if len(used) == 4:
                break

    if len(used) < 4:
        raise RuntimeError("Unable to construct a 4-man bench template from pool.")

    cost = _total_spent(used)
    if cost > bench_budget:
        # Try to improve: swap the most expensive outfielder down if possible
        tries = 0
        while cost > bench_budget and tries < 50:
            tries += 1
            # find most expensive outfielder among the 3 outfielders
            ofs = [p for p in used if p.pos in {"DEF", "MID", "FWD"}]
            worst = max(ofs, key=lambda z: z.cost)
            # search a cheaper same-pos alt not yet used, under club caps
            cands = [p for p in outfield if p.pos == worst.pos and p.id not in {pp.id for pp in used}]
            cands.sort(key=lambda z: (z.cost, -z.ep1))
            swapped = False
            for q in cands:
                if q.cost >= worst.cost:
                    break
                # club caps
                clubs_now = _club_counts([pp for pp in used if pp.id != worst.id])
                if clubs_now.get(q.team, 0) >= MAX_PER_CLUB:
                    continue
                used = [q if pp.id == worst.id else pp for pp in used]
                cost = _total_spent(used)
                swapped = True
                break
            if not swapped:
                break

    return used


# ------------------- greedy selection (respects bench template) -------------------
def _min_costs_baseline(pool: List[Player]) -> Dict[str, int]:
    out = {}
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        costs = [p.cost for p in pool if p.pos == pos]
        out[pos] = min(costs) if costs else 0
    return out


def _pick15_with_bench_template(pool: List[Player], bench_template: List[Player]) -> List[Player]:
    """Pick 15 ensuring the 4 bench-template players are in the squad; fill the rest greedily."""
    template_ids = {p.id for p in bench_template}

    def score(p: Player) -> float:
        # Slightly de-prioritize template guys for XI (they're meant to be cheap bench)
        bench_pen = 0.9 if p.id in template_ids else 1.0
        price = max(35.0, float(p.cost))
        return bench_pen * ((0.7 * p.eph + 0.3 * p.ep1) / price + 0.02 * (p.xmins / 90.0))

    cand = sorted(pool, key=score, reverse=True)

    need = dict(POSITIONS)
    picked: List[Player] = []
    clubs: Dict[int, int] = {}
    spent = 0
    min_cost = _min_costs_baseline(pool)

    # First, add the 4 bench-template players (they still count to totals)
    for p in bench_template:
        if need[p.pos] <= 0:
            # if template conflicts with totals (very rare), skip it
            continue
        if clubs.get(p.team, 0) >= MAX_PER_CLUB:
            continue
        if (spent + p.cost) > BUDGET:
            continue
        picked.append(p)
        need[p.pos] -= 1
        spent += p.cost
        clubs[p.team] = clubs.get(p.team, 0) + 1

    # Then fill the rest greedily
    def can_afford(cost_add: int, need_after: Dict[str, int]) -> bool:
        remaining_min = sum(need_after[pos] * min_cost.get(pos, 0) for pos in need_after)
        return (spent + cost_add + remaining_min) <= BUDGET

    for p in cand:
        if p in picked:
            continue
        if need[p.pos] <= 0:
            continue
        if clubs.get(p.team, 0) >= MAX_PER_CLUB:
            continue
        after = need.copy()
        after[p.pos] -= 1
        if not can_afford(p.cost, after):
            continue
        picked.append(p)
        need[p.pos] -= 1
        spent += p.cost
        clubs[p.team] = clubs.get(p.team, 0) + 1
        if sum(need.values()) == 0:
            break

    # fill if somehow short
    if sum(need.values()) > 0:
        cheap = sorted([q for q in pool if q not in picked], key=lambda q: (q.cost, -q.eph))
        for q in cheap:
            if need[q.pos] <= 0:
                continue
            if clubs.get(q.team, 0) >= MAX_PER_CLUB:
                continue
            if spent + q.cost > BUDGET:
                continue
            picked.append(q)
            need[q.pos] -= 1
            spent += q.cost
            clubs[q.team] = clubs.get(q.team, 0) + 1
            if sum(need.values()) == 0:
                break

    assert len(picked) == 15, f"picked={len(picked)}"
    assert sum(1 for p in picked if p.pos == "GKP") == 2
    assert sum(1 for p in picked if p.pos == "DEF") == 5
    assert sum(1 for p in picked if p.pos == "MID") == 5
    assert sum(1 for p in picked if p.pos == "FWD") == 3
    return picked


def _bench_cost(bench: List[Player]) -> int:
    return sum(p.cost for p in bench)


def _xi_for_formation(picked: List[Player], form_key: str) -> List[Player]:
    req = FORMATIONS[form_key]
    xi: List[Player] = []

    # GK: pick higher ep1 as starter
    gks = sorted([p for p in picked if p.pos == "GKP"], key=lambda p: p.ep1, reverse=True)
    xi.extend(gks[: req["GKP"]])

    # DEF/MID/FWD by ep1
    for pos in ["DEF", "MID", "FWD"]:
        needn = req[pos]
        cands = [p for p in picked if p.pos == pos and p not in xi]
        cands.sort(key=lambda p: p.ep1, reverse=True)
        xi.extend(cands[:needn])

    if len(xi) < 11:
        rem = [p for p in picked if p not in xi]
        rem.sort(key=lambda p: p.ep1, reverse=True)
        xi.extend(rem[: 11 - len(xi)])

    return xi[:11]


def _repair_bench_budget(xi: List[Player], picked: List[Player], bench_budget: int) -> Tuple[List[Player], List[Player]]:
    """
    Make bench ≤ bench_budget via same-position swaps (bench ↔ XI).
    Also ensure cheaper GK is on the bench.
    """
    bench = [p for p in picked if p not in xi][:4]

    # Ensure CHEAPER GK is on the bench
    xi_gk = [p for p in xi if p.pos == "GKP"]
    bench_gk = [p for p in bench if p.pos == "GKP"]
    if xi_gk and bench_gk and bench_gk[0].cost > xi_gk[0].cost:
        xi.remove(xi_gk[0])
        bench.remove(bench_gk[0])
        xi.append(bench_gk[0])
        bench.append(xi_gk[0])

    # Swap bench with XI (same-position) to push expensive pieces into XI
    tries = 0
    while _bench_cost(bench) > bench_budget and tries < 40:
        tries += 1
        best = None  # (cost_saved, -ep_loss, b_idx, i_idx)
        bench_by_pos: Dict[str, List[int]] = {}
        xi_by_pos: Dict[str, List[int]] = {}
        for idx, p in enumerate(bench):
            bench_by_pos.setdefault(p.pos, []).append(idx)
        for idx, p in enumerate(xi):
            xi_by_pos.setdefault(p.pos, []).append(idx)

        for pos in ["GKP", "DEF", "MID", "FWD"]:
            for b_idx in bench_by_pos.get(pos, []):
                b = bench[b_idx]
                for i_idx in xi_by_pos.get(pos, []):
                    i = xi[i_idx]
                    cost_saved = b.cost - i.cost  # reduce bench cost if positive
                    if cost_saved <= 0:
                        continue
                    ep_loss = max(0.0, i.ep1 - b.ep1)
                    cand = (cost_saved, -ep_loss, b_idx, i_idx)
                    if (best is None) or (cand > best):
                        best = cand

        if best is None:
            break
        _, _, b_idx, i_idx = best
        b = bench[b_idx]
        i = xi[i_idx]
        xi[i_idx] = b
        bench[b_idx] = i

    return xi, bench


def _downgrade_bench_until_cap(
    pool: List[Player],
    picked: List[Player],
    form: str,
    bench_budget: int,
) -> Tuple[List[Player], List[Player]]:
    """
    Replace expensive bench players with cheaper **same-position** pool candidates,
    respecting total budget and club caps. Rebuild XI/bench each time.
    """
    xi = _xi_for_formation(picked, form)
    xi, bench = _repair_bench_budget(xi, picked, bench_budget)
    if _bench_cost(bench) <= bench_budget:
        return xi, bench

    # pool by position (not already picked)
    pool_by_pos: Dict[str, List[Player]] = {}
    picked_ids = {p.id for p in picked}
    for p in pool:
        if p.id in picked_ids:
            continue
        pool_by_pos.setdefault(p.pos, []).append(p)
    for pos in pool_by_pos:
        pool_by_pos[pos].sort(key=lambda z: (z.cost, -z.ep1))

    tries = 0
    while _bench_cost(bench) > bench_budget and tries < 80:
        tries += 1
        b_idx_sorted = sorted(range(len(bench)), key=lambda i: bench[i].cost, reverse=True)
        made = False
        for b_idx in b_idx_sorted:
            b = bench[b_idx]
            current_spent = _total_spent(picked)
            clubs = _club_counts(picked)
            for q in pool_by_pos.get(b.pos, []):
                if q.cost >= b.cost:
                    break  # can't reduce spend with this q
                # club caps after swap
                clubs_after = dict(clubs)
                clubs_after[b.team] = clubs_after.get(b.team, 0) - 1
                if clubs_after[b.team] <= 0:
                    clubs_after.pop(b.team, None)
                clubs_after[q.team] = clubs_after.get(q.team, 0) + 1
                if clubs_after[q.team] > MAX_PER_CLUB:
                    continue
                if (current_spent - b.cost + q.cost) > BUDGET:
                    continue

                # perform replacement in picked
                picked = [q if (pp.id == b.id) else pp for pp in picked]
                xi = _xi_for_formation(picked, form)
                xi, bench = _repair_bench_budget(xi, picked, bench_budget)
                made = True
                break
            if made:
                break
        if not made:
            break

    return xi, bench


def _choose_xi_and_bench(
    picked: List[Player],
    formations_allowed: List[str],
    bench_budget: int,
    pool: Optional[List[Player]] = None,
) -> Tuple[List[Player], List[Player], str]:
    """
    Try each formation; repair bench; if still over, try downgrades using pool.
    """
    best = None  # (feasible(bool), xi_score, form, xi, bench)
    for form in formations_allowed:
        xi0 = _xi_for_formation(picked, form)
        xi, bench = _repair_bench_budget(xi0, picked, bench_budget)
        feasible = _bench_cost(bench) <= bench_budget
        score = sum(p.ep1 for p in xi)
        key = (feasible, score)
        if best is None or key > (best[0], best[1]):
            best = (feasible, score, form, xi, bench)

    feasible, _, form, xi, bench = best
    if feasible:
        return xi, bench, form

    if pool is not None:
        xi2, bench2 = _downgrade_bench_until_cap(pool, picked, form, bench_budget)
        return xi2, bench2, form

    return xi, bench, form


# ------------------- helpers for advanced optimizer -------------------
def _format_advanced_result(result: Dict[str, Any], horizon: int, bench_budget: int) -> Dict[str, Any]:
    """Format advanced optimizer result to match expected output format."""
    squad = result.get("squad", [])
    xi = result.get("starting_xi", [])
    bench = result.get("bench", [])
    captain = result.get("captain")
    formation = result.get("formation", "442")
    
    def fmt(p) -> str:
        return f" - {p.name} ({p.position}) — ep1 {p.ep_next:.2f}, next{horizon} {sum(p.ep_horizon[:horizon]):.2f}, xMins {p.xmins:.0f}, £{p.price:.1f}m"
    
    human = []
    next_gw = _get_next_gameweek()
    human.append(f"Suggested 15-man squad (LP Optimized), formation {formation}:")
    human.append(f"Expected Points (GW{next_gw}): {result.get('expected_points_gw1', 0):.1f}")
    human.append(f"Optimization Score: {result.get('optimization_score', 0):.1f}")
    human.append("")
    
    human.append("Full Squad:")
    for p in squad:
        human.append(fmt(p))
    human.append("")
    
    human.append("Starting XI:")
    for p in xi:
        marker = " (C)" if captain and p.id == captain.id else ""
        human.append(fmt(p) + marker)
    human.append("")
    
    human.append("Bench:")
    for i, p in enumerate(bench, 1):
        human.append(f"{i}. {p.name} ({p.position}) — {p.ep_next:.2f} pts")
    human.append("")
    
    bench_cost = sum(p.price for p in bench)
    human.append(f"Bench value: £{bench_cost:.1f}m (target max £{bench_budget/10:.1f}m)")
    
    if captain:
        human.append(f"Captain: {captain.name} ({captain.position}) - {captain.ep_next:.2f} pts")
    
    # Add transfer suggestions if present
    if result.get("transfers_in") or result.get("transfers_out"):
        human.append("")
        human.append("Recommended Transfers:")
        for p in result.get("transfers_in", []):
            human.append(f"  IN: {p.name}")
        for pid in result.get("transfers_out", []):
            human.append(f"  OUT: Player {pid}")
    
    return {
        "human_readable": "\n".join(human),
        "picked_ids": [p.id for p in squad],
        "xi_ids": [p.id for p in xi],
        "bench_ids": [p.id for p in bench],
        "formation": formation,
        "bench_cost": bench_cost,
        "bench_budget": bench_budget / 10.0,
        "captain_id": captain.id if captain else None,
        "expected_points": result.get("expected_points_gw1", 0),
        "optimization_score": result.get("optimization_score", 0),
        "table": [
            {"player_id": p.id, "name": p.name, "pos": p.position, "team_id": p.team,
             "cost": p.price, "xmins": p.xmins, "ep1": p.ep_next, 
             f"next{horizon}": sum(p.ep_horizon[:horizon])}
            for p in squad
        ],
    }


# ------------------- Two-Stage Transfer Optimization -------------------
def evaluate_transfer_with_optimal_xi(
    current_squad: List[int],
    transfer_out_id: int,
    transfer_in_id: int,
    players: List[Player],
    horizon: int,
    bench_weight: float
) -> float:
    """
    Evaluate a single transfer by optimizing XI selection after the transfer.
    Returns the total objective value after the transfer.
    """
    # Create new squad after transfer
    new_squad_ids = [pid if pid != transfer_out_id else transfer_in_id for pid in current_squad]
    
    # Get squad players
    squad_players = [p for p in players if p.id in new_squad_ids]
    
    # Create LP for XI selection with this squad
    prob = pulp.LpProblem("XI_Selection", pulp.LpMaximize)
    
    # Variables: who is in XI
    xi_vars = {p.id: pulp.LpVariable(f"xi_{p.id}", cat="Binary") for p in squad_players}

    # Objective: maximize EP (XI gets full, bench gets bench_weight + xmins bonus)
    # Bench players get bonus for having high xMins (important for Bench Boost)
    bench_xmins_weight = 0.5  # Weight for xMins on bench (0.5 means 45 xMins = +0.25 pts)
    prob += pulp.lpSum([
        xi_vars[p.id] * p.eph + (1 - xi_vars[p.id]) * (p.eph * bench_weight + bench_xmins_weight * (p.xmins / 90.0))
        for p in squad_players
    ])
    
    # Exactly 11 in XI
    prob += pulp.lpSum(xi_vars.values()) == 11
    
    # Formation constraints - at least satisfy minimum requirements
    prob += pulp.lpSum([xi_vars[p.id] for p in squad_players if p.pos == "GKP"]) == 1
    prob += pulp.lpSum([xi_vars[p.id] for p in squad_players if p.pos == "DEF"]) >= 3
    prob += pulp.lpSum([xi_vars[p.id] for p in squad_players if p.pos == "DEF"]) <= 5
    prob += pulp.lpSum([xi_vars[p.id] for p in squad_players if p.pos == "MID"]) >= 2
    prob += pulp.lpSum([xi_vars[p.id] for p in squad_players if p.pos == "MID"]) <= 5
    prob += pulp.lpSum([xi_vars[p.id] for p in squad_players if p.pos == "FWD"]) >= 1
    prob += pulp.lpSum([xi_vars[p.id] for p in squad_players if p.pos == "FWD"]) <= 3
    
    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    if prob.status == pulp.LpStatusOptimal:
        return pulp.value(prob.objective)
    else:
        return -float('inf')  # Invalid solution


def optimize_transfers_two_stage(
    current_squad: List[int],
    players: List[Player],
    max_transfers: int,
    horizon: int,
    bench_weight: float,
    budget_limit: int
) -> Dict[str, Any]:
    """
    Two-stage transfer optimization:
    1. Enumerate all valid transfer combinations
    2. For each, optimize XI and calculate total objective
    3. Select the best transfer combination
    """
    log.info(f"Starting two-stage transfer optimization for {max_transfers} transfer(s)")

    # Get current squad players
    current_players = [p for p in players if p.id in current_squad]

    # Identify dead players (EP=0 or xmins=0) - these should be prioritized for transfer out
    dead_player_ids = set()
    for p in current_players:
        if p.ep_base == 0 or p.xmins == 0:
            dead_player_ids.add(p.id)
            log.info(f"Dead player identified: {p.name} (ID={p.id}, EP={p.ep_base:.2f}, xMins={p.xmins:.0f})")

    if dead_player_ids:
        log.info(f"Found {len(dead_player_ids)} dead player(s) to prioritize for transfer out")

    # CRITICAL: Check for club limit violations (max 3 per team)
    # This takes priority over everything else - must be fixed first
    club_violation_player_ids = set()
    club_violation_info = {}  # {team_id: count} for teams over limit
    team_player_counts = {}
    team_players = {}
    for p in current_players:
        team_player_counts[p.team] = team_player_counts.get(p.team, 0) + 1
        if p.team not in team_players:
            team_players[p.team] = []
        team_players[p.team].append(p)

    for team_id, count in team_player_counts.items():
        if count > MAX_PER_CLUB:
            club_violation_info[team_id] = count
            # Find players from this team and mark them for potential transfer out
            team_name = team_players[team_id][0].name.split()[0] if team_players[team_id] else f"Team {team_id}"
            log.warning(f"🚨 CLUB LIMIT VIOLATION: {count} players from team {team_id} (max {MAX_PER_CLUB})")
            for p in team_players[team_id]:
                club_violation_player_ids.add(p.id)
                log.info(f"  Must consider transferring out: {p.name} (ID={p.id})")

    if club_violation_player_ids:
        log.warning(f"Found {len(club_violation_player_ids)} player(s) from clubs over limit - MUST transfer one out")

    # Calculate current budget (selling prices + bank)
    import json
    import os
    selling_prices = {}
    bank = 0
    myteam_file = "data/processed/myteam_latest.json"
    if os.path.exists(myteam_file):
        with open(myteam_file) as f:
            myteam_data = json.load(f)
        for pick in myteam_data.get("picks", []):
            selling_prices[pick["element"]] = pick.get("selling_price", pick.get("purchase_price", 0))
        bank = myteam_data.get("transfers", {}).get("bank", 0)
        log.info(f"Bank available: £{bank/10:.1f}m")

    # Group players by position for valid transfers
    players_by_pos = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
    for p in players:
        if p.id not in current_squad:  # Only consider players not in squad
            players_by_pos[p.pos].append(p)
    
    # Sort by EP to prioritize better players
    for pos in players_by_pos:
        players_by_pos[pos].sort(key=lambda p: p.eph, reverse=True)
    
    # Generate transfer options based on max_transfers
    transfer_options = []
    
    if max_transfers == 1:
        # Generate single transfer options
        for p_out in current_players:
            selling_price = selling_prices.get(p_out.id, p_out.cost)
            available_budget = selling_price + bank  # Include bank in budget

            # Only consider top N candidates per position to reduce computation
            candidates = players_by_pos[p_out.pos][:20]  # Top 20 by EP

            for p_in in candidates:
                # Check budget constraint (include bank)
                if p_in.cost > available_budget:
                    continue  # Can't afford

                # Check club constraint (max 3 per team)
                new_squad_ids = [pid if pid != p_out.id else p_in.id for pid in current_squad]
                team_counts = {}
                for pid in new_squad_ids:
                    p = next((pl for pl in players if pl.id == pid), None)
                    if p:
                        team_counts[p.team] = team_counts.get(p.team, 0) + 1

                if team_counts.get(p_in.team, 0) > 3:
                    continue  # Would exceed club limit

                # Calculate EP gain (simple heuristic for initial filtering)
                ep_gain = p_in.eph - p_out.eph

                # CRITICAL: Add HUGE bonus for fixing club limit violations
                # This is a rule violation that MUST be fixed - highest priority
                fixes_club_violation = p_out.id in club_violation_player_ids
                if fixes_club_violation:
                    # Check that this transfer actually fixes the violation
                    # (i.e., the incoming player is not from the same over-limit club)
                    if p_in.team != p_out.team:
                        club_violation_bonus = 100.0  # Highest priority
                        ep_gain += club_violation_bonus
                        log.info(f"🚨 Club violation fix: {p_out.name} -> {p_in.name}, base gain={ep_gain - club_violation_bonus:.2f}, with bonus={ep_gain:.2f}")

                # Add large bonus for removing dead players (lower priority than club violations)
                is_removing_dead = p_out.id in dead_player_ids
                if is_removing_dead and not fixes_club_violation:
                    dead_player_bonus = 50.0
                    ep_gain += dead_player_bonus
                    log.info(f"Dead player transfer: {p_out.name} -> {p_in.name}, base gain={ep_gain - dead_player_bonus:.2f}, with bonus={ep_gain:.2f}")

                transfer_options.append({
                    'transfers': [(p_out, p_in)],
                    'ep_gain': ep_gain,
                    'budget_saved': selling_price - p_in.cost,
                    'removes_dead': is_removing_dead,
                    'fixes_club_violation': fixes_club_violation
                })
        
        # Sort by EP gain and take top candidates
        transfer_options.sort(key=lambda x: x['ep_gain'], reverse=True)
        transfer_options = transfer_options[:50]  # Evaluate top 50 single transfers
        
    elif max_transfers == 2:
        # Generate double transfer options
        # This allows for downgrade/upgrade combinations

        # First, generate promising single transfers
        single_transfers = []
        for p_out in current_players:
            # p_out is a Player object, and selling_prices keys are player IDs
            selling_price = selling_prices.get(p_out.id, p_out.cost)
            available_budget = selling_price + bank  # Include bank
            # For 2-transfers, include both upgrades AND downgrades
            # Sort by EP but also include some cheaper players for downgrade options
            all_candidates = players_by_pos[p_out.pos]

            # Take top players by EP
            top_by_ep = all_candidates[:25]

            # Also specifically include good cheaper options for downgrades
            # These help fund upgrades elsewhere
            cheaper = [p for p in all_candidates if p.cost < selling_price]
            cheaper.sort(key=lambda p: p.eph, reverse=True)

            # Include some premium options too (for premium downgrades like Salah->Palmer)
            premium_downgrades = [p for p in all_candidates
                                 if selling_price > 100 and  # If selling premium (>£10m)
                                 p.cost < selling_price and  # Cheaper option
                                 p.cost > selling_price - 50]  # But not too cheap (within £5m)
            premium_downgrades.sort(key=lambda p: p.eph, reverse=True)

            # Combine all sets (avoid duplicates by ID)
            seen_ids = set()
            candidates = []
            for p in top_by_ep + cheaper[:15] + premium_downgrades[:10]:
                if p.id not in seen_ids:
                    candidates.append(p)
                    seen_ids.add(p.id)

            for p_in in candidates:
                # Skip if same player
                if p_in.id == p_out.id:
                    continue

                # For 2-transfer combos, be more flexible with individual costs
                # since one transfer can fund another (include bank)
                if p_in.cost > available_budget + 50:  # Allow up to £5m more expensive with savings elsewhere
                    continue

                ep_gain = p_in.eph - p_out.eph

                # CRITICAL: Add HUGE bonus for fixing club limit violations
                fixes_club_violation = p_out.id in club_violation_player_ids
                if fixes_club_violation and p_in.team != p_out.team:
                    ep_gain += 100.0  # Highest priority

                # Add bonus for removing dead players (lower priority than club violations)
                is_removing_dead = p_out.id in dead_player_ids
                if is_removing_dead and not fixes_club_violation:
                    ep_gain += 50.0  # Large bonus to prioritize dead player removal

                budget_diff = selling_price - p_in.cost
                single_transfers.append({
                    'out': p_out,
                    'in': p_in,
                    'ep_gain': ep_gain,
                    'budget_diff': budget_diff,
                    'removes_dead': is_removing_dead,
                    'fixes_club_violation': fixes_club_violation
                })
        
        # Sort single transfers by EP gain
        single_transfers.sort(key=lambda x: x['ep_gain'], reverse=True)
        
        log.info(f"Generated {len(single_transfers)} single transfer options for pairing")
        
        # Debug: Show a few top single transfers
        if single_transfers:
            log.info(f"Top single transfer: {single_transfers[0]['out'].name} -> {single_transfers[0]['in'].name}, EP gain: {single_transfers[0]['ep_gain']:.2f}")
        
        # Separate downgrades and upgrades for better pairing
        downgrades = [t for t in single_transfers if t['budget_diff'] > 0]  # Saves money
        upgrades = [t for t in single_transfers if t['budget_diff'] < 0]  # Costs money
        neutral = [t for t in single_transfers if t['budget_diff'] == 0]  # Same price
        
        # Sort by their value (EP gain per pound saved/spent)
        downgrades.sort(key=lambda x: x['ep_gain'] / max(x['budget_diff'], 1), reverse=True)
        upgrades.sort(key=lambda x: x['ep_gain'] / max(-x['budget_diff'], 1), reverse=True)
        
        log.info(f"Found {len(downgrades)} downgrades, {len(upgrades)} upgrades, {len(neutral)} neutral")
        
        # Generate double transfer combinations
        combinations_tried = 0
        combinations_valid = 0
        budget_fails = 0
        club_fails = 0
        remove_fails = 0
        
        # Try pairing each downgrade with each upgrade
        for t1 in downgrades[:20]:  # Top 20 downgrades
            for t2 in upgrades[:20]:  # Top 20 upgrades
                    
                combinations_tried += 1
                    
                # Skip if same player involved
                if t1['out'].id == t2['out'].id or t1['in'].id == t2['in'].id:
                    continue
                if t1['out'].id == t2['in'].id or t1['in'].id == t2['out'].id:
                    continue
                    
                # Check combined budget
                # Get the actual selling prices from our myteam data
                # t1['out'] and t2['out'] are Player objects, and their selling prices are in selling_prices dict
                selling_price_1 = selling_prices.get(t1['out'].id, t1['out'].cost)
                selling_price_2 = selling_prices.get(t2['out'].id, t2['out'].cost)
                total_out = selling_price_1 + selling_price_2
                total_in = t1['in'].cost + t2['in'].cost
                
                # Debug first few combinations
                if combinations_tried <= 5:
                    log.info(f"Combo {combinations_tried}: {t1['out'].name}->{t1['in'].name} + {t2['out'].name}->{t2['in'].name}")
                    log.info(f"  Budget: Out £{total_out/10:.1f}m, In £{total_in/10:.1f}m, Valid: {total_in <= total_out}")
                
                # Allow the transfer if we can afford it with our budget
                # (selling both players gives us their combined value to spend)
                if total_in > total_out:
                    budget_fails += 1
                    continue  # Can't afford both
                
                # Check club constraints
                new_squad_ids = current_squad.copy()
                try:
                    new_squad_ids.remove(t1['out'].id)
                    new_squad_ids.remove(t2['out'].id)
                except ValueError:
                    # One of the players not in squad - shouldn't happen but skip
                    remove_fails += 1
                    continue
                new_squad_ids.append(t1['in'].id)
                new_squad_ids.append(t2['in'].id)
                
                team_counts = {}
                for pid in new_squad_ids:
                    p = next((pl for pl in players if pl.id == pid), None)
                    if p:
                        team_counts[p.team] = team_counts.get(p.team, 0) + 1
                
                if any(count > 3 for count in team_counts.values()):
                    club_fails += 1
                    continue  # Would exceed club limit
                
                # Calculate combined EP gain
                combined_ep_gain = t1['ep_gain'] + t2['ep_gain']
                
                transfer_options.append({
                    'transfers': [(t1['out'], t1['in']), (t2['out'], t2['in'])],
                    'ep_gain': combined_ep_gain,
                    'budget_saved': total_out - total_in
                })
                combinations_valid += 1
        
        # Also try neutral transfers with others
        for t1 in neutral[:10]:
            for t2 in single_transfers[:20]:
                if t1['out'].id == t2['out'].id or t1['in'].id == t2['in'].id:
                    continue
                if t1['out'].id == t2['in'].id or t1['in'].id == t2['out'].id:
                    continue
                    
                combinations_tried += 1
                
                selling_price_1 = selling_prices.get(t1['out'].id, t1['out'].cost)
                selling_price_2 = selling_prices.get(t2['out'].id, t2['out'].cost)
                total_out = selling_price_1 + selling_price_2
                total_in = t1['in'].cost + t2['in'].cost
                
                if total_in > total_out:
                    budget_fails += 1
                    continue
                
                # Check club constraints
                new_squad_ids = current_squad.copy()
                try:
                    new_squad_ids.remove(t1['out'].id)
                    new_squad_ids.remove(t2['out'].id)
                except ValueError:
                    remove_fails += 1
                    continue
                new_squad_ids.append(t1['in'].id)
                new_squad_ids.append(t2['in'].id)
                
                team_counts = {}
                for pid in new_squad_ids:
                    p = next((pl for pl in players if pl.id == pid), None)
                    if p:
                        team_counts[p.team] = team_counts.get(p.team, 0) + 1
                
                if any(count > 3 for count in team_counts.values()):
                    club_fails += 1
                    continue
                
                combined_ep_gain = t1['ep_gain'] + t2['ep_gain']
                
                transfer_options.append({
                    'transfers': [(t1['out'], t1['in']), (t2['out'], t2['in'])],
                    'ep_gain': combined_ep_gain,
                    'budget_saved': total_out - total_in
                })
                combinations_valid += 1
        
        # Also try best single transfers alone (for 1-transfer option within 2-transfer budget)
        for t in single_transfers[:20]:
            if t['budget_diff'] >= 0:  # Only if we can afford it
                transfer_options.append({
                    'transfers': [(t['out'], t['in'])],
                    'ep_gain': t['ep_gain'],
                    'budget_saved': t['budget_diff']
                })
        
        log.info(f"Tried {combinations_tried} combinations, found {combinations_valid} valid 2-transfer options")
        log.info(f"Added {min(20, len([t for t in single_transfers[:20] if t['budget_diff'] >= 0]))} single transfer options")
        log.info(f"Failures - Budget: {budget_fails}, Club: {club_fails}, Remove: {remove_fails}")
        
        # Sort by combined EP gain and take top candidates
        transfer_options.sort(key=lambda x: x['ep_gain'], reverse=True)
        transfer_options = transfer_options[:100]  # Evaluate top 100 double transfers
    else:
        # For 3+ transfers, fall back to regular LP optimization
        # Two-stage optimization is complex for 3+ transfers and would require
        # evaluating too many combinations
        log.info(f"Two-stage optimization not implemented for {max_transfers} transfers, using regular LP")
        # Return None to signal fallback to regular LP
        return None
    
    log.info(f"Evaluating {len(transfer_options)} transfer options")
    
    # Evaluate each transfer combination with optimal XI selection
    best_option = None
    best_objective = -float('inf')
    
    # First evaluate no transfer (baseline)
    baseline_objective = evaluate_transfer_with_optimal_xi(
        current_squad, -1, -1, players, horizon, bench_weight
    )
    log.info(f"Baseline (no transfer): {baseline_objective:.2f}")
    
    for i, option in enumerate(transfer_options):
        # Apply transfers to create new squad
        new_squad_ids = current_squad.copy()
        
        if i < 10:  # Log top 10 for debugging
            if len(option['transfers']) == 1:
                p_out, p_in = option['transfers'][0]
                log.info(f"Evaluating: {p_out.name} -> {p_in.name} (EP gain: {option['ep_gain']:.2f})")
            else:
                transfers_str = ", ".join([f"{out.name}->{inp.name}" for out, inp in option['transfers']])
                log.info(f"Evaluating: {transfers_str} (EP gain: {option['ep_gain']:.2f})")
        
        # Apply all transfers in this option
        for p_out, p_in in option['transfers']:
            idx = new_squad_ids.index(p_out.id)
            new_squad_ids[idx] = p_in.id
        
        # Get squad players after transfers
        squad_players = [p for p in players if p.id in new_squad_ids]
        
        # Optimize XI for this squad
        prob = pulp.LpProblem("XI_Selection", pulp.LpMaximize)
        xi_vars = {p.id: pulp.LpVariable(f"xi_{p.id}", cat="Binary") for p in squad_players}

        # Objective: maximize EP (XI gets full, bench gets bench_weight + xmins bonus)
        # Bench players get bonus for having high xMins (important for Bench Boost)
        bench_xmins_weight = 0.5  # Weight for xMins on bench
        prob += pulp.lpSum([
            xi_vars[p.id] * p.eph + (1 - xi_vars[p.id]) * (p.eph * bench_weight + bench_xmins_weight * (p.xmins / 90.0))
            for p in squad_players
        ])
        
        # Constraints
        prob += pulp.lpSum(xi_vars.values()) == 11
        prob += pulp.lpSum([xi_vars[p.id] for p in squad_players if p.pos == "GKP"]) == 1
        prob += pulp.lpSum([xi_vars[p.id] for p in squad_players if p.pos == "DEF"]) >= 3
        prob += pulp.lpSum([xi_vars[p.id] for p in squad_players if p.pos == "DEF"]) <= 5
        prob += pulp.lpSum([xi_vars[p.id] for p in squad_players if p.pos == "MID"]) >= 2
        prob += pulp.lpSum([xi_vars[p.id] for p in squad_players if p.pos == "MID"]) <= 5
        prob += pulp.lpSum([xi_vars[p.id] for p in squad_players if p.pos == "FWD"]) >= 1
        prob += pulp.lpSum([xi_vars[p.id] for p in squad_players if p.pos == "FWD"]) <= 3
        
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == pulp.LpStatusOptimal:
            objective = pulp.value(prob.objective)
            
            if i < 10:
                log.info(f"  Objective after transfer(s): {objective:.2f} (gain: {objective - baseline_objective:.2f})")
            
            if objective > best_objective:
                best_objective = objective
                best_option = option
    
    if best_option and best_objective > baseline_objective:
        if len(best_option['transfers']) == 1:
            p_out, p_in = best_option['transfers'][0]
            log.info(f"Best transfer: {p_out.name} -> {p_in.name}")
        else:
            transfers_str = ", ".join([f"{out.name}->{inp.name}" for out, inp in best_option['transfers']])
            log.info(f"Best transfers: {transfers_str}")
        log.info(f"Objective improvement: {best_objective - baseline_objective:.2f}")
        
        # Create the new squad with the transfers
        new_squad_ids = current_squad.copy()
        for p_out, p_in in best_option['transfers']:
            idx = new_squad_ids.index(p_out.id)
            new_squad_ids[idx] = p_in.id
        
        # Now optimize XI for the final squad
        squad_players = [p for p in players if p.id in new_squad_ids]
        
        # Create final LP for XI selection
        prob = pulp.LpProblem("Final_XI", pulp.LpMaximize)
        xi_vars = {p.id: pulp.LpVariable(f"xi_{p.id}", cat="Binary") for p in squad_players}
        captain_vars = {p.id: pulp.LpVariable(f"cap_{p.id}", cat="Binary") for p in squad_players}
        
        # Objective with captain bonus
        prob += pulp.lpSum([
            xi_vars[p.id] * p.eph + 
            captain_vars[p.id] * p.ep1 +  # Captain bonus
            (1 - xi_vars[p.id]) * p.eph * bench_weight
            for p in squad_players
        ])
        
        # Constraints
        prob += pulp.lpSum(xi_vars.values()) == 11
        prob += pulp.lpSum(captain_vars.values()) == 1
        
        # Captain must be in XI
        for p in squad_players:
            prob += captain_vars[p.id] <= xi_vars[p.id]
        
        # Formation
        prob += pulp.lpSum([xi_vars[p.id] for p in squad_players if p.pos == "GKP"]) == 1
        prob += pulp.lpSum([xi_vars[p.id] for p in squad_players if p.pos == "DEF"]) >= 3
        prob += pulp.lpSum([xi_vars[p.id] for p in squad_players if p.pos == "DEF"]) <= 5
        prob += pulp.lpSum([xi_vars[p.id] for p in squad_players if p.pos == "MID"]) >= 2
        prob += pulp.lpSum([xi_vars[p.id] for p in squad_players if p.pos == "MID"]) <= 5
        prob += pulp.lpSum([xi_vars[p.id] for p in squad_players if p.pos == "FWD"]) >= 1
        prob += pulp.lpSum([xi_vars[p.id] for p in squad_players if p.pos == "FWD"]) <= 3
        
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract solution
        xi = []
        captain = None
        for p in squad_players:
            if xi_vars[p.id].varValue > 0.5:
                xi.append(p)
                if captain_vars[p.id].varValue > 0.5:
                    captain = p
        
        bench = [p for p in squad_players if p not in xi]
        bench.sort(key=lambda p: -p.ep1)
        
        # Determine formation
        pos_counts = {"GKP": 0, "DEF": 0, "MID": 0, "FWD": 0}
        for p in xi:
            pos_counts[p.pos] += 1
        
        formation = "442"  # Default
        for form, reqs in FORMATIONS.items():
            if all(pos_counts[pos] == count for pos, count in reqs.items()):
                formation = form
                break
        
        # Prepare transfer details based on single or double transfers
        if len(best_option['transfers']) == 1:
            p_out, p_in = best_option['transfers'][0]
            transfers_out = [p_out]
            transfers_in = [p_in]
        else:
            transfers_out = [t[0] for t in best_option['transfers']]
            transfers_in = [t[1] for t in best_option['transfers']]
        
        return {
            'transfers_out': transfers_out,
            'transfers_in': transfers_in,
            'transfer_out': transfers_out[0] if len(transfers_out) == 1 else None,  # For backward compatibility
            'transfer_in': transfers_in[0] if len(transfers_in) == 1 else None,
            'squad': squad_players,
            'xi': xi,
            'bench': bench,
            'captain': captain,
            'formation': formation,
            'objective_improvement': best_objective - baseline_objective,
            # Metadata for verbose reporting
            'dead_players': [{'id': p.id, 'name': p.name, 'pos': p.pos} for p in current_players if p.id in dead_player_ids],
            'club_violation_info': club_violation_info,  # {team_id: count} for teams over limit
            'removes_dead': best_option.get('removes_dead', False),
            'fixes_club_violation': best_option.get('fixes_club_violation', False),
            'num_transfers': len(best_option['transfers']),
        }
    else:
        log.info("No beneficial transfer found")
        return None


# ------------------- LP Optimization -------------------
def optimize_with_lp(
    horizon: int = 5,
    bench_weight: float = 0.1,
    bench_budget: int = 180,
    formations: str = "343,352,442,451,433",
    differential_bonus: float = 0.1,
    risk_penalty: float = 0.05,
    value_weight: float = 0.3,
    current_squad: Optional[List[int]] = None,
    max_transfers: int = 2,
    wildcard: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Linear Programming based optimization using PuLP.
    Now uses two-stage approach for transfers.
    """
    if not HAS_PULP:
        return {"error": "PuLP not installed"}
    
    try:
        # Load player pool
        js = _fetch_bootstrap()
        teams_map = {t["id"]: t["name"] for t in js.get("teams", [])}
        xmins_map = _load_xmins()
        ep_map, xgi_map, team_att_map = _load_ep_extras()
        
        # Load data-driven team quality scores once
        from ..data.team_quality import get_team_quality_scores
        team_quality_scores = get_team_quality_scores()
        
        # Load FDR data for fixture difficulty
        try:
            from ..utils.cache import PROC
            from ..utils.io import read_parquet
            fdr_df = read_parquet(PROC / "player_next5_fdr.parquet")
            fdr_map = dict(zip(fdr_df['player_id'], fdr_df['fdr_factor']))
            log.info(f"Loaded FDR data for {len(fdr_map)} players")
        except Exception as e:
            log.warning(f"FDR data not available: {e}")
            fdr_map = {}
        
        # USE TWO-STAGE OPTIMIZATION FOR TRANSFERS
        if current_squad and max_transfers > 0 and not wildcard:
            log.info("Using two-stage optimization for transfers")
            
            # Load selling prices for current squad
            selling_prices = {}
            import json
            import os
            myteam_file = "data/processed/myteam_latest.json"
            budget_limit = 1000  # Default
            if os.path.exists(myteam_file):
                with open(myteam_file) as f:
                    myteam_data = json.load(f)
                for pick in myteam_data.get("picks", []):
                    selling_prices[pick["element"]] = pick.get("selling_price", pick.get("purchase_price", 0))
                
                # Calculate total budget
                selling_value = sum(selling_prices.get(pid, 0) for pid in current_squad)
                bank = myteam_data.get("transfers", {}).get("bank", 0)
                budget_limit = selling_value + bank
                log.info(f"Budget: Selling value={selling_value/10:.1f}m + Bank={bank/10:.1f}m = {budget_limit/10:.1f}m")
            
            # Create player objects
            players = []
            for e in js["elements"]:
                pid = int(e["id"])
                pos = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}[e["element_type"]]
                team_id = int(e["team"])
                
                # Use selling price for current squad, current price for others
                if pid in current_squad and pid in selling_prices:
                    cost = selling_prices[pid]
                else:
                    cost = int(e.get("now_cost", 0))
                
                xmins = float(xmins_map.get(pid, 70.0))
                ep_base = float(ep_map.get(pid, float(e.get("ep_next", 0))))
                
                # Skip ultra-cheap fodder unless in current squad
                if pos != "GKP" and pid not in current_squad:
                    if cost < 40 or (cost <= 45 and (ep_base < 1.5 or xmins < 30)):
                        continue
                
                # Apply FDR adjustment
                fdr_raw = fdr_map.get(pid, 3.0)
                fdr_multiplier = 1.0 + (3.0 - fdr_raw) * 0.05
                adjusted_eph = ep_base * horizon * fdr_multiplier
                
                # Get team quality
                team_name = teams_map.get(team_id, "")
                team_strength = team_quality_scores.get(team_name, 1.0)
                
                players.append(Player(
                    id=pid,
                    name=e.get("web_name", f"Player_{pid}"),
                    pos=pos,
                    team=team_id,
                    cost=cost,
                    xmins=xmins,
                    ep_base=ep_base,
                    xgi90=float(xgi_map.get(pid, 0)),
                    team_att=float(team_att_map.get(pid, 1.0)),
                    ep_seq=[ep_base * fdr_multiplier for _ in range(horizon)],
                    ep1=ep_base,
                    eph=adjusted_eph,
                    form=float(e.get("form", 0)),
                    selected_by=float(e.get("selected_by_percent", 0)),
                    value_score=ep_base / (cost / 10) if cost > 0 else 0,
                    is_differential=float(e.get("selected_by_percent", 0)) < 10.0 and ep_base > 3.0,
                    injury_risk=0.1 if e.get("status") != "a" else 0.0,
                    rotation_risk=max(0, (90 - xmins) / 90) if xmins < 60 else 0.0,
                    team_strength=team_strength
                ))
            
            # Run two-stage optimization
            result = optimize_transfers_two_stage(
                current_squad=current_squad,
                players=players,
                max_transfers=max_transfers,
                horizon=horizon,
                bench_weight=bench_weight,
                budget_limit=budget_limit
            )
            
            # If two-stage returns None (e.g., for 3+ transfers), fall back to regular LP
            if result is None:
                log.info(f"Two-stage optimization not available for {max_transfers} transfers, falling back to regular LP")
                # Don't use two-stage for this case - continue to regular LP below
            elif result is not None and result:
                # Find vice captain (best non-captain in XI)
                vice_captain = None
                for p in result['xi']:
                    if result['captain'] and p.id != result['captain'].id:
                        if vice_captain is None or p.ep1 > vice_captain.ep1:
                            vice_captain = p
                
                # Format the result
                output = []
                output.append(f"=== TRANSFER RECOMMENDATION ===")
                
                # Handle single or multiple transfers
                if 'transfer_out' in result and result['transfer_out']:
                    # Single transfer - show both next GW and horizon total
                    p_out = result['transfer_out']
                    p_in = result['transfer_in']
                    output.append(f"OUT: {p_out.name} ({p_out.pos}) - £{p_out.cost/10:.1f}m, EP: {p_out.eph:.1f} over {horizon} GWs ({p_out.ep1:.2f}/wk)")
                    output.append(f"IN:  {p_in.name} ({p_in.pos}) - £{p_in.cost/10:.1f}m, EP: {p_in.eph:.1f} over {horizon} GWs ({p_in.ep1:.2f}/wk)")
                else:
                    # Multiple transfers
                    for i, (p_out, p_in) in enumerate(zip(result['transfers_out'], result['transfers_in']), 1):
                        output.append(f"Transfer {i}:")
                        output.append(f"  OUT: {p_out.name} ({p_out.pos}) - £{p_out.cost/10:.1f}m, EP: {p_out.eph:.1f} over {horizon} GWs ({p_out.ep1:.2f}/wk)")
                        output.append(f"  IN:  {p_in.name} ({p_in.pos}) - £{p_in.cost/10:.1f}m, EP: {p_in.eph:.1f} over {horizon} GWs ({p_in.ep1:.2f}/wk)")

                output.append(f"EP gain over {horizon} GWs: {result['objective_improvement']:.2f}")
                output.append("")
                output.append(f"=== OPTIMAL LINEUP (Formation {result['formation']}) ===")
                output.append("Starting XI:")
                for p in sorted(result['xi'], key=lambda x: ["GKP", "DEF", "MID", "FWD"].index(x.pos)):
                    status = ""
                    if result['captain'] and p.id == result['captain'].id:
                        status = " (C)"
                    elif vice_captain and p.id == vice_captain.id:
                        status = " (VC)"
                    output.append(f" - {p.name} ({p.pos}) — {p.ep1:.2f} pts{status}")
                output.append("")
                output.append("Bench:")
                for i, p in enumerate(result['bench'], 1):
                    output.append(f" {i}. {p.name} ({p.pos}) — {p.ep1:.2f} pts")
                output.append("")
                if result['captain']:
                    output.append(f"Captain: {result['captain'].name} - {result['captain'].ep1:.2f} pts (doubled = {result['captain'].ep1*2:.2f})")
                if vice_captain:
                    output.append(f"Vice-Captain: {vice_captain.name} - {vice_captain.ep1:.2f} pts")
                
                total_cost = sum(p.cost for p in result['squad'])
                bench_value = sum(p.cost for p in result['bench'])
                expected_points = sum(p.ep1 for p in result['xi'])
                if result['captain']:
                    expected_points += result['captain'].ep1
                
                return {
                    "squad": [
                        {
                            "id": p.id,
                            "name": p.name,
                            "pos": p.pos,
                            "team": p.team,
                            "cost": p.cost / 10,
                            "ep1": p.ep1,
                            "eph": p.eph,
                            "xmins": p.xmins,
                            "is_captain": result['captain'] and p.id == result['captain'].id,
                            "in_xi": p in result['xi']
                        }
                        for p in result['squad']
                    ],
                    "transfers_out": [p.id for p in result.get('transfers_out', [result.get('transfer_out')])] if result.get('transfers_out') or result.get('transfer_out') else [],
                    "transfers_in": [p.id for p in result.get('transfers_in', [result.get('transfer_in')])] if result.get('transfers_in') or result.get('transfer_in') else [],
                    "formation": result['formation'],
                    "total_cost": total_cost / 10,
                    "bench_value": bench_value / 10,
                    "expected_points_gw1": expected_points,
                    "expected_points_total": expected_points,  # Add this for compatibility
                    "optimization_score": result['objective_improvement'],
                    "solver": "Two-Stage Transfer Optimization",
                    "human_readable": "\n".join(output),
                    # Metadata for verbose reporting
                    "dead_players": result.get('dead_players', []),
                    "club_violation_info": result.get('club_violation_info', {}),
                    "removes_dead": result.get('removes_dead', False),
                    "fixes_club_violation": result.get('fixes_club_violation', False),
                    "num_transfers": result.get('num_transfers', 0),
                    "horizon": horizon,
                }
            elif result is not None:
                # result is False/empty - no beneficial transfer found
                return {
                    "error": "No beneficial transfer found",
                    "human_readable": "Two-stage optimization found no beneficial transfers within constraints"
                }
        
        # For non-transfer cases (new squad or wildcard), continue with regular LP optimization
        
        # Load selling prices for current squad if doing transfers (but NOT for wildcard)
        selling_prices = {}
        if current_squad and not wildcard:
            import json
            import os
            myteam_file = "data/processed/myteam_latest.json"
            if os.path.exists(myteam_file):
                with open(myteam_file) as f:
                    myteam_data = json.load(f)
                for pick in myteam_data.get("picks", []):
                    selling_prices[pick["element"]] = pick.get("selling_price", pick.get("purchase_price", 0))

        # Create player objects with enhanced data
        players = []
        for e in js["elements"]:
            pid = int(e["id"])
            pos = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}[e["element_type"]]
            team_id = int(e["team"])

            # CRITICAL: Use selling price for current squad players (but NOT on wildcard)
            # On wildcard, all players are bought at current market price
            if current_squad and not wildcard and pid in current_squad and pid in selling_prices:
                cost = selling_prices[pid]
            else:
                cost = int(e.get("now_cost", 0))
            
            xmins = float(xmins_map.get(pid, 70.0))
            ep_base = float(ep_map.get(pid, float(e.get("ep_next", 0))))
            
            # Skip ultra-cheap fodder unless it's a GKP or in current squad
            # But keep some reasonable bench options and all current squad players
            if pos != "GKP" and not (current_squad and pid in current_squad):
                if cost < 40:  # Less than £4.0m
                    continue
                # Skip players with very low EP and minutes (not viable bench options)
                if cost <= 45 and (ep_base < 1.5 or xmins < 30):
                    continue
            
            # Calculate value score
            value_score = ep_base / (cost / 10) if cost > 0 else 0
            
            # Check if differential
            ownership = float(e.get("selected_by_percent", 0))
            is_differential = ownership < 10.0 and ep_base > 3.0
            
            # Risk assessments
            injury_risk = 0.1 if e.get("status") != "a" else 0.0
            rotation_risk = max(0, (90 - xmins) / 90) if xmins < 60 else 0.0
            
            # Apply FDR adjustment to expected points over horizon
            # FDR represents fixture difficulty over next 5 games (1=easy, 5=hard)
            # We'll apply a small adjustment: easy fixtures boost EP, hard fixtures reduce it
            fdr_raw = fdr_map.get(pid, 3.0)  # Default to neutral (3.0) if no FDR data
            # Convert FDR (1-5 scale) to multiplier (1.1 to 0.9)
            # FDR 1 (easiest) -> 1.1x, FDR 3 (neutral) -> 1.0x, FDR 5 (hardest) -> 0.9x
            fdr_multiplier = 1.0 + (3.0 - fdr_raw) * 0.05  # ±10% max adjustment
            adjusted_eph = ep_base * horizon * fdr_multiplier
            
            # Use data-driven team quality scores (already loaded above)
            team_name = teams_map.get(team_id, "")
            
            # Get team quality score (0.5-1.5 range where 1.0 is average)
            team_strength = team_quality_scores.get(team_name, 1.0)
            
            players.append(Player(
                id=pid,
                name=e.get("web_name", f"Player_{pid}"),
                pos=pos,
                team=team_id,
                cost=cost,
                xmins=xmins,
                ep_base=ep_base,
                xgi90=float(xgi_map.get(pid, 0)),
                team_att=float(team_att_map.get(pid, 1.0)),
                ep_seq=[ep_base * fdr_multiplier for _ in range(horizon)],
                ep1=ep_base,
                eph=adjusted_eph,
                form=float(e.get("form", 0)),
                selected_by=ownership,
                value_score=value_score,
                is_differential=is_differential,
                injury_risk=injury_risk,
                rotation_risk=rotation_risk,
                team_strength=team_strength
            ))
        
        if not players:
            return {"error": "No valid players found"}
        
        # Apply adjustments for players with limited minutes (like Gyökeres, Frimpong)
        try:
            from ..models.adjustments import apply_new_player_adjustments
            from ..data.fpl_api import get_bootstrap
            
            # Create DataFrame for adjustments
            player_data = []
            for p in players:
                player_data.append({
                    'id': p.id,
                    'player_id': p.id,
                    'EPH': p.eph,
                    'position': p.pos,
                    'cost': p.cost
                })
            
            predictions_df = pd.DataFrame(player_data)
            bootstrap = get_bootstrap()
            
            # Apply adjustments with a reasonable floor
            adjusted_df = apply_new_player_adjustments(
                predictions_df,
                features_df=pd.DataFrame(),
                bootstrap_data=bootstrap,
                min_minutes_threshold=180,  # < 2 full games
                new_player_floor_multiplier=0.6  # At least 60% of position average for same price
            )
            
            # Update player EPH values
            adjustments_made = []
            for p in players:
                adj_row = adjusted_df[adjusted_df['id'] == p.id]
                if not adj_row.empty:
                    new_eph = adj_row.iloc[0].get('EPH', p.eph)
                    if new_eph != p.eph and new_eph > p.eph:
                        # Only adjust upward (don't make good players worse)
                        adjustments_made.append(f"{p.name}: {p.eph:.2f} -> {new_eph:.2f}")
                        ratio = new_eph / p.eph if p.eph > 0 else 1
                        p.eph = new_eph
                        p.ep_seq = [ep * ratio for ep in p.ep_seq]
                        p.ep1 = p.ep_seq[0] if p.ep_seq else new_eph
            
            if adjustments_made:
                log.info(f"Applied new player adjustments to {len(adjustments_made)} players")
                for adj in adjustments_made[:5]:  # Show first 5
                    log.info(f"  {adj}")
        except Exception as e:
            log.debug(f"Could not apply new player adjustments: {e}")
        
        # Create LP problem
        prob = pulp.LpProblem("FPL_Squad", pulp.LpMaximize)
        
        # Debug: Check if key players are in the pool
        if current_squad and max_transfers > 0:
            virgil = next((p for p in players if p.id == 373), None)
            xhaka = next((p for p in players if p.id == 668), None)
            frimpong = next((p for p in players if p.id == 370), None)
            reijnders = next((p for p in players if p.id == 427), None)
            
            log.info(f"Key players in LP pool (total={len(players)}):")
            if virgil:
                log.info(f"  Virgil (373): EPH={virgil.eph:.2f}, Cost=£{virgil.cost/10:.1f}m - IN POOL")
            else:
                log.info(f"  Virgil (373): NOT IN POOL")
            if xhaka:
                log.info(f"  Xhaka (668): EPH={xhaka.eph:.2f}, Cost=£{xhaka.cost/10:.1f}m - IN POOL")
            else:
                log.info(f"  Xhaka (668): NOT IN POOL")
            if frimpong:
                log.info(f"  Frimpong (370): EPH={frimpong.eph:.2f}, Cost=£{frimpong.cost/10:.1f}m - IN POOL")
            else:
                log.info(f"  Frimpong (370): NOT IN POOL")
            if reijnders:
                log.info(f"  Reijnders (427): EPH={reijnders.eph:.2f}, Cost=£{reijnders.cost/10:.1f}m - IN POOL")
            else:
                log.info(f"  Reijnders (427): NOT IN POOL")
        
        # Decision variables
        squad_vars = {p.id: pulp.LpVariable(f"squad_{p.id}", cat="Binary") for p in players}
        xi_vars = {p.id: pulp.LpVariable(f"xi_{p.id}", cat="Binary") for p in players}
        captain_var = {p.id: pulp.LpVariable(f"cap_{p.id}", cat="Binary") for p in players}
        
        # Objective function
        objective_terms = []
        
        # Debug logging for key players if doing transfers
        debug_players = []
        if current_squad and max_transfers > 0:
            debug_ids = [370, 373, 427, 668]  # Frimpong, Virgil, Reijnders, Xhaka
            debug_players = [p for p in players if p.id in debug_ids]
            
        for p in players:
            # Base expected points
            ep_contrib = p.eph
            
            # REMOVED: Form and team quality adjustments
            # These are already accounted for:
            # 1. Team quality: Models should learn from historical performance
            # 2. FDR: Already applied when creating eph (line 783)
            # 3. Form: Too volatile (1 game), causes overreaction
            # We should trust the model predictions + FDR adjustment only
            
            # Value bonus - DISABLED because we care about points, not savings!
            value_bonus = 0  # p.value_score * value_weight
            
            # Differential bonus - reduced to avoid chasing low-owned players
            diff_bonus = differential_bonus * ep_contrib * 0.5 if p.is_differential else 0
            
            # Risk penalties
            risk_pen = (p.injury_risk + p.rotation_risk) * risk_penalty * ep_contrib
            
            # XI bonus removed - let the optimizer decide based on pure EP
            xi_bonus = 0  # Was causing bias towards certain formations
            
            # Captain bonus (double points)
            cap_bonus = p.ep1
            
            # XI players get full EP contribution
            # Bench players get small contribution (10% of EP for bench strength)
            bench_contrib = (squad_vars[p.id] - xi_vars[p.id]) * ep_contrib * bench_weight
            
            # Calculate total objective contribution for this player
            xi_contrib = ep_contrib + value_bonus + diff_bonus - risk_pen
            
            # Debug logging for key players
            if p in debug_players:
                log.info(f"Objective for {p.name} ({p.id}): EPH={p.eph:.2f}, " +
                        f"EP_contrib={ep_contrib:.2f}, XI_contrib={xi_contrib:.2f}")
            
            objective_terms.append(
                xi_vars[p.id] * xi_contrib +
                captain_var[p.id] * cap_bonus +
                bench_contrib
            )
        
        prob += pulp.lpSum(objective_terms)
        
        # Constraints
        # Squad size = 15
        prob += pulp.lpSum([squad_vars[p.id] for p in players]) == 15
        
        # Starting XI = 11
        prob += pulp.lpSum([xi_vars[p.id] for p in players]) == 11
        
        # XI must be in squad
        for p in players:
            prob += xi_vars[p.id] <= squad_vars[p.id]
        
        # Exactly 1 captain
        prob += pulp.lpSum([captain_var[p.id] for p in players]) == 1
        
        # Captain must be in XI
        for p in players:
            prob += captain_var[p.id] <= xi_vars[p.id]
        
        # Position constraints for SQUAD (15 players)
        for pos, count in POSITIONS.items():
            prob += pulp.lpSum([squad_vars[p.id] for p in players if p.pos == pos]) == count

        # Formation constraints for STARTING XI
        # CRITICAL: Always enforce valid formation for XI, even when doing transfers
        prob += pulp.lpSum([xi_vars[p.id] for p in players if p.pos == "GKP"]) == 1
        prob += pulp.lpSum([xi_vars[p.id] for p in players if p.pos == "DEF"]) >= 3
        prob += pulp.lpSum([xi_vars[p.id] for p in players if p.pos == "DEF"]) <= 5
        prob += pulp.lpSum([xi_vars[p.id] for p in players if p.pos == "MID"]) >= 2
        prob += pulp.lpSum([xi_vars[p.id] for p in players if p.pos == "MID"]) <= 5
        prob += pulp.lpSum([xi_vars[p.id] for p in players if p.pos == "FWD"]) >= 1
        prob += pulp.lpSum([xi_vars[p.id] for p in players if p.pos == "FWD"]) <= 3

        # Additional formation selection for initial squad (not needed for transfers)
        if not current_squad:
            # Only apply specific formation selection when building a new squad from scratch
            formation_vars = {f: pulp.LpVariable(f"form_{f}", cat="Binary") for f in FORMATIONS}
            prob += pulp.lpSum(formation_vars.values()) == 1

            for formation, reqs in FORMATIONS.items():
                for pos, required in reqs.items():
                    prob += pulp.lpSum([xi_vars[p.id] for p in players if p.pos == pos]) >= required - 100 * (1 - formation_vars[formation])
                    prob += pulp.lpSum([xi_vars[p.id] for p in players if p.pos == pos]) <= required + 100 * (1 - formation_vars[formation])
        
        # Budget constraint - check if budget_limit was passed (e.g., from wildcard mode)
        # Otherwise calculate from current squad
        budget_limit = kwargs.get('budget_limit')

        if budget_limit:
            # Budget already calculated (e.g., wildcard mode)
            log.info(f"Using pre-calculated budget: £{budget_limit/10:.1f}m")
        elif current_squad:
            # Load myteam data to get actual selling prices and bank
            import json
            import os
            myteam_file = "data/processed/myteam_latest.json"
            if os.path.exists(myteam_file):
                with open(myteam_file) as f:
                    myteam_data = json.load(f)

                # Regular transfers: budget = selling value + bank
                # Calculate total selling value of current squad + bank
                selling_value = 0
                for pick in myteam_data.get("picks", []):
                    if pick["element"] in current_squad:
                        selling_value += pick.get("selling_price", pick.get("purchase_price", 0))

                bank = myteam_data.get("transfers", {}).get("bank", 0)
                budget_limit = selling_value + bank

                log.info(f"Budget calculation: Selling value={selling_value/10:.1f}m + Bank={bank/10:.1f}m = Total={budget_limit/10:.1f}m")

                # For no transfers, allow keeping current squad even if over budget
                if max_transfers == 0:
                    current_cost = sum(p.cost for p in players if p.id in current_squad)
                    budget_limit = max(budget_limit, current_cost)
            else:
                log.warning(f"myteam file not found at {myteam_file}, using default budget")
                budget_limit = BUDGET
        else:
            # For new squads (no current_squad), use standard budget
            budget_limit = BUDGET
        
        prob += pulp.lpSum([squad_vars[p.id] * p.cost for p in players]) <= budget_limit
        
        # Bench budget constraint
        # Skip this constraint if we're keeping current squad with no transfers
        if not (current_squad and max_transfers == 0):
            bench_cost = pulp.lpSum([squad_vars[p.id] * p.cost for p in players]) - pulp.lpSum([xi_vars[p.id] * p.cost for p in players])
            prob += bench_cost <= bench_budget
        
        # Minimum bench quality - at least one bench player should have decent EP
        # This ensures viable substitution options
        bench_quality_vars = {p.id: pulp.LpVariable(f"bench_{p.id}", cat="Binary") for p in players}
        for p in players:
            # Player is on bench if in squad but not in XI
            prob += bench_quality_vars[p.id] <= squad_vars[p.id]
            prob += bench_quality_vars[p.id] <= 1 - xi_vars[p.id]
            prob += bench_quality_vars[p.id] >= squad_vars[p.id] - xi_vars[p.id]
        
        # At least 2 bench players should have EP > 1.5 (viable options)
        viable_bench = [p for p in players if p.ep_base > 1.5 and p.pos != "GKP"]
        if viable_bench:
            prob += pulp.lpSum([bench_quality_vars[p.id] for p in viable_bench]) >= 2
        
        # Max 3 per club
        # Note: This constraint is applied to the FINAL squad after transfers
        for team_id in set(p.team for p in players):
            prob += pulp.lpSum([squad_vars[p.id] for p in players if p.team == team_id]) <= MAX_PER_CLUB
        
        # Transfer constraints if current squad provided
        if current_squad and not wildcard:
            # Create transfer variables for tracking changes
            in_current = {p.id: (1 if p.id in current_squad else 0) for p in players}
            
            if max_transfers == 0:
                # No transfers allowed - must keep exact current squad
                for p in players:
                    prob += squad_vars[p.id] == in_current[p.id]
            else:
                # Track transfers in and out
                transfer_out_vars = {}
                transfer_in_vars = {}
                
                for p in players:
                    # Transfer out: was in current squad but not selected
                    transfer_out_vars[p.id] = pulp.LpVariable(f"out_{p.id}", cat='Binary')
                    # Transfer in: not in current squad but selected  
                    transfer_in_vars[p.id] = pulp.LpVariable(f"in_{p.id}", cat='Binary')
                    
                    # Link transfer variables to squad selection
                    if p.id in current_squad:
                        # Can only transfer out if not selected
                        prob += transfer_out_vars[p.id] >= in_current[p.id] - squad_vars[p.id]
                        prob += transfer_out_vars[p.id] <= 1
                        prob += transfer_in_vars[p.id] == 0  # Can't transfer in a player we already have
                        
                        # CRITICAL: If transferred out, CANNOT be in XI
                        # Already handled by xi_vars[p.id] <= squad_vars[p.id] constraint earlier
                    else:
                        # Can only transfer in if selected
                        prob += transfer_in_vars[p.id] >= squad_vars[p.id] - in_current[p.id]
                        prob += transfer_in_vars[p.id] <= squad_vars[p.id]
                        prob += transfer_out_vars[p.id] == 0  # Can't transfer out a player we don't have
                        
                        # New players can be in XI if transferred in
                        prob += xi_vars[p.id] <= squad_vars[p.id]
                
                # Total transfers = players out (or equivalently, players in)
                total_out = pulp.lpSum([transfer_out_vars[p.id] for p in players])
                total_in = pulp.lpSum([transfer_in_vars[p.id] for p in players])
                
                # Transfers in must equal transfers out
                prob += total_out == total_in
                
                # Limit total transfers
                prob += total_out <= max_transfers
                
                # CRUCIAL: Position balance - transfers must maintain squad structure
                # For each position, transfers out must equal transfers in
                for pos in ['GKP', 'DEF', 'MID', 'FWD']:
                    transfers_out_pos = pulp.lpSum([transfer_out_vars[p.id] for p in players if p.pos == pos])
                    transfers_in_pos = pulp.lpSum([transfer_in_vars[p.id] for p in players if p.pos == pos])
                    prob += transfers_out_pos == transfers_in_pos
                
                # CRITICAL FIX: Force optimal XI selection for transfers
                # The issue is that the LP might keep lower-EP players in XI after transfers
                # We need to ensure transferred-in players with higher EP go into XI
                
                # CRITICAL: If we transfer in a high-EP player, they MUST be in the XI
                # This fixes the issue where the LP transfers in Reinildo but keeps him on bench
                # while not considering Virgil who would go straight into XI
                if max_transfers > 0:
                    # For each transferred-in player, if their EP is high, force them into XI
                    for p in players:
                        if p.id not in current_squad:  # Potential transfer target
                            # If this player has higher EP than current squad average for their position,
                            # and they get transferred in, they should be in XI
                            pos_players_current = [pl for pl in players if pl.pos == p.pos and pl.id in current_squad]
                            if pos_players_current:
                                avg_ep_current = sum(pl.eph for pl in pos_players_current) / len(pos_players_current)
                                if p.eph > avg_ep_current * 1.2:  # 20% better than average
                                    # If transferred in (transfer_in = 1), must be in XI (xi = 1)
                                    prob += xi_vars[p.id] >= transfer_in_vars[p.id]
                                    
                    # Also ensure that if we transfer OUT a player from XI, we must transfer IN someone better
                    # This prevents the LP from downgrading the XI through transfers
        
        # Solve
        if current_squad and max_transfers > 0:
            # Write LP to file for debugging
            prob.writeLP("debug_transfer_lp.lp")
            log.info("Wrote LP problem to debug_transfer_lp.lp")
        
        # Use CBC solver with options for finding optimal solution
        # CBC always finds global optimum for LP problems
        # We can add a small timeout to ensure it doesn't run forever
        solver = pulp.PULP_CBC_CMD(
            msg=1 if current_squad and max_transfers > 0 else 0,  # Show solver output for debugging transfers
            timeLimit=30,  # Allow up to 30 seconds
            gapRel=0.0  # Require exact optimum (0% gap)
        )
        
        prob.solve(solver)
        
        # Log optimization details for debugging
        if current_squad and max_transfers > 0:
            if prob.status == pulp.LpStatusOptimal:
                log.info(f"Transfer optimization completed successfully. Objective value: {pulp.value(prob.objective):.2f}")
            elif prob.status == pulp.LpStatusInfeasible:
                log.warning("LP problem is infeasible - no valid solution exists")
            elif prob.status == pulp.LpStatusUnbounded:
                log.warning("LP problem is unbounded - objective can be infinitely improved")
            elif prob.status == pulp.LpStatusNotSolved:
                log.warning("LP problem was not solved")
            else:
                log.warning(f"LP solver returned status: {pulp.LpStatus[prob.status]}")
            
            # Log current squad defenders for debugging
            current_defs = [p for p in players if p.id in current_squad and p.pos == "DEF"]
            log.info(f"Current squad defenders ({len(current_defs)}):")
            for p in current_defs:
                selected = "KEPT" if squad_vars[p.id].varValue > 0.5 else "TRANSFERRED OUT"
                log.info(f"  {p.name} ({p.team}) ID={p.id} - EP={p.ep_base:.2f}, EPH={p.eph:.2f}, Cost=£{p.cost/10}m -> {selected}")
            
            # Log key Liverpool defenders for debugging Frimpong/Virgil issue
            liverpool_defs = [p for p in players if p.pos == "DEF" and p.team == 12]  # Liverpool team ID
            log.info(f"Liverpool defenders available ({len(liverpool_defs)}):")
            for p in sorted(liverpool_defs, key=lambda x: x.eph, reverse=True)[:5]:  # Top 5 by EPH
                in_squad = "CURRENT" if p.id in current_squad else "AVAILABLE"
                selected = "SELECTED" if squad_vars[p.id].varValue > 0.5 else "NOT SELECTED"
                log.info(f"  {p.name} (ID:{p.id}) - EP={p.ep_base:.2f}, EPH={p.eph:.2f}, Team strength={p.team_strength:.2f}, Cost=£{p.cost/10}m - {in_squad} -> {selected}")
            
            # Check Liverpool player count
            liverpool_count = sum(1 for p in players if p.team == 12 and squad_vars[p.id].varValue > 0.5)
            log.info(f"Liverpool players in final squad: {liverpool_count}/3 max")
            liverpool_in_squad = [p.name for p in players if p.team == 12 and squad_vars[p.id].varValue > 0.5]
            log.info(f"Liverpool players: {', '.join(liverpool_in_squad)}")
            
            # Log best transfer opportunities by EP gain
            log.info("Top transfer opportunities (by EP gain):")
            transfer_opportunities = []
            for p_out in [p for p in players if p.id in current_squad]:
                for p_in in [p for p in players if p.id not in current_squad and p.pos == p_out.pos]:
                    if p_in.cost <= p_out.cost:  # Can afford
                        ep_gain = p_in.eph - p_out.eph
                        if ep_gain > 0:
                            transfer_opportunities.append((ep_gain, p_out, p_in))
            
            for ep_gain, p_out, p_in in sorted(transfer_opportunities, key=lambda x: x[0], reverse=True)[:10]:
                log.info(f"  {p_out.name} -> {p_in.name}: +{ep_gain:.2f} EP (£{p_out.cost/10:.1f}m -> £{p_in.cost/10:.1f}m)")
            
            # Check which players were transferred
            transfers_made = []
            
            # First log the XI composition BEFORE transfers
            log.info("XI before transfers:")
            for p in players:
                if p.id in current_squad and xi_vars[p.id].varValue and xi_vars[p.id].varValue > 0.5:
                    log.info(f"  {p.name} ({p.pos}) ID={p.id} - EPH={p.eph:.2f}")
            
            # Check transfer variables directly
            log.info("Transfer variables that are set to 1:")
            for p in players:
                if p.id in current_squad:
                    if transfer_out_vars[p.id].varValue and transfer_out_vars[p.id].varValue > 0.5:
                        was_xi = "XI" if xi_vars[p.id].varValue > 0.5 else "BENCH"
                        log.info(f"  transfer_out[{p.id}] = 1: {p.name} ({p.pos}) was {was_xi}")
                else:
                    if transfer_in_vars[p.id].varValue and transfer_in_vars[p.id].varValue > 0.5:
                        is_xi = "XI" if xi_vars[p.id].varValue > 0.5 else "BENCH"
                        log.info(f"  transfer_in[{p.id}] = 1: {p.name} ({p.pos}) goes to {is_xi}")
            
            for p in players:
                if p.id in current_squad and squad_vars[p.id].varValue < 0.5:
                    transfers_made.append(("OUT", p))
                    log.info(f"OUT: {p.name} ({p.pos}) ID={p.id} - EP over {horizon} GWs: {p.eph:.2f}, Cost: £{p.cost/10:.1f}m")
                elif p.id not in current_squad and squad_vars[p.id].varValue > 0.5:
                    transfers_made.append(("IN", p))
                    log.info(f"IN: {p.name} ({p.pos}) ID={p.id} - EP over {horizon} GWs: {p.eph:.2f}, Cost: £{p.cost/10:.1f}m")
            
            # Calculate EP gain from transfers
            if transfers_made:
                ep_out = sum(p.eph for direction, p in transfers_made if direction == "OUT")
                ep_in = sum(p.eph for direction, p in transfers_made if direction == "IN")
                log.info(f"Transfer EP gain: {ep_in:.2f} - {ep_out:.2f} = {ep_in - ep_out:.2f}")
        
        # Debug: Check formation when using formation variables (new squad from scratch)
        if not current_squad:
            selected_formation = None
            for f, var in formation_vars.items():
                if var.varValue and var.varValue > 0.5:
                    selected_formation = f
                    log.info(f"Selected formation: {f}")
                    
                    # Calculate what the XI EP would be for different formations
                    xi_players = [p for p in players if xi_vars[p.id].varValue and xi_vars[p.id].varValue > 0.5]
                    total_xi_ep = sum(p.ep1 for p in xi_players)
                    log.info(f"Total XI EP with {f}: {total_xi_ep:.2f}")
                    
                    # Show position breakdown
                    pos_breakdown = {}
                    for p in xi_players:
                        pos_breakdown[p.pos] = pos_breakdown.get(p.pos, 0) + 1
                    log.info(f"Position breakdown: {pos_breakdown}")
                    break
        else:
            # For transfers, just log the XI composition
            xi_players = [p for p in players if xi_vars[p.id].varValue and xi_vars[p.id].varValue > 0.5]
            if xi_players:
                total_xi_ep = sum(p.ep1 for p in xi_players)
                pos_breakdown = {}
                for p in xi_players:
                    pos_breakdown[p.pos] = pos_breakdown.get(p.pos, 0) + 1
                log.info(f"XI composition after transfers: {pos_breakdown}, Total EP: {total_xi_ep:.2f}")
        
        if prob.status != pulp.LpStatusOptimal:
            # Debug infeasible problems
            if prob.status == pulp.LpStatusInfeasible:
                log.warning("LP optimization infeasible - debugging constraints:")
                if current_squad:
                    current_positions = {"GKP": 0, "DEF": 0, "MID": 0, "FWD": 0}
                    current_total_cost = 0
                    missing_players = []
                    for pid in current_squad:
                        found = False
                        for p in players:
                            if p.id == pid:
                                current_positions[p.pos] += 1
                                current_total_cost += p.cost
                                found = True
                                break
                        if not found:
                            missing_players.append(pid)
                    
                    log.warning(f"Current squad positions: {current_positions}")
                    log.warning(f"Current squad cost: £{current_total_cost/10:.1f}m")
                    log.warning(f"Budget limit: £{budget_limit/10:.1f}m")
                    log.warning(f"Max transfers: {max_transfers}")
                    if missing_players:
                        log.warning(f"Missing players in pool: {missing_players}")
                    
                    # Check if squad meets basic requirements
                    if current_positions != POSITIONS:
                        log.warning(f"Position mismatch! Required: {POSITIONS}, Current: {current_positions}")
            return {"error": f"Optimization failed with status: {pulp.LpStatus[prob.status]}"}
        
        # Extract solution
        squad = []
        xi = []
        captain_id = None
        
        for p in players:
            if squad_vars[p.id].varValue > 0.5:
                squad.append(p)
                if xi_vars[p.id].varValue > 0.5:
                    xi.append(p)
                if captain_var[p.id].varValue > 0.5:
                    captain_id = p.id
        
        # Sort XI by position
        xi.sort(key=lambda p: ["GKP", "DEF", "MID", "FWD"].index(p.pos))
        
        # Determine formation
        pos_counts = {"GKP": 0, "DEF": 0, "MID": 0, "FWD": 0}
        for p in xi:
            pos_counts[p.pos] += 1
        
        formation = "unknown"
        for form, reqs in FORMATIONS.items():
            if all(pos_counts[pos] == count for pos, count in reqs.items()):
                formation = form
                break
        
        # Format output
        bench = [p for p in squad if p not in xi]
        bench.sort(key=lambda p: -p.ep1)  # Best bench players first
        
        # Find vice captain (best non-captain in XI)
        vice = None
        for p in xi:
            if p.id != captain_id:
                if vice is None or p.ep1 > vice.ep1:
                    vice = p
        
        total_cost = sum(p.cost for p in squad)
        bench_value = sum(p.cost for p in bench)
        expected_points = sum(p.ep1 for p in xi) + (next(p.ep1 for p in xi if p.id == captain_id) if captain_id else 0)
        
        # Build human-readable output
        next_gw = _get_next_gameweek()
        output = []
        output.append(f"Suggested 15-man squad (LP Optimized), formation {formation}:")
        output.append(f"Expected Points (GW{next_gw}): {expected_points:.1f}")
        output.append(f"Optimization Score: {pulp.value(prob.objective):.1f}")
        output.append("")
        
        output.append("Full Squad:")
        for p in squad:
            status = ""
            if p.id == captain_id:
                status = " (C)"
            elif vice and p.id == vice.id:
                status = " (V)"
            output.append(f" - {p.name} ({p.pos}) — ep1 {p.ep1:.2f}, eph {p.eph:.2f}, xMins {p.xmins:.0f}, £{p.cost/10:.1f}m{status}")
        
        output.append(f"\nStarting XI:")
        for p in xi:
            status = ""
            if p.id == captain_id:
                status = " (C)"
            elif vice and p.id == vice.id:
                status = " (V)"
            output.append(f" - {p.name} ({p.pos}) — ep1 {p.ep1:.2f}, eph {p.eph:.2f}, xMins {p.xmins:.0f}, £{p.cost/10:.1f}m{status}")
        
        output.append(f"\nBench:")
        for i, p in enumerate(bench, 1):
            output.append(f"{i}. {p.name} ({p.pos}) — {p.ep1:.2f} pts")
        
        output.append(f"\nBench value: £{bench_value/10:.1f}m (target max £{bench_budget/10:.1f}m)")
        if captain_id:
            cap = next(p for p in xi if p.id == captain_id)
            output.append(f"Captain: {cap.name} ({cap.pos}) - {cap.ep1:.2f} pts")
        
        return {
            "squad": [
                {
                    "id": p.id,
                    "name": p.name,
                    "pos": p.pos,
                    "team": p.team,
                    "cost": p.cost / 10,
                    "ep1": p.ep1,
                    "eph": p.eph,
                    "xmins": p.xmins,
                    "is_captain": p.id == captain_id,
                    "is_vice": p.id == vice.id if vice else False,
                    "in_xi": p in xi
                }
                for p in squad
            ],
            "formation": formation,
            "total_cost": total_cost / 10,
            "bench_value": bench_value / 10,
            "expected_points_gw1": expected_points,
            "optimization_score": pulp.value(prob.objective),
            "solver": "Linear Programming (PuLP)",
            "human_readable": "\n".join(output)
        }
        
    except Exception as e:
        log.error(f"LP optimization error: {e}")
        return {"error": str(e)}


# ------------------- public API -------------------
def optimize_transfers(
    use_myteam: bool = False,               # (kept for CLI compat)
    horizon: int = 5,
    bench_weight: float = 0.10,             # (unused in greedy path)
    bench_budget: int = 180,                # <-- HARD CAP (tenths of a million)
    formations: str = "343,352,442,451,433",
    nonstarter_xmins: float = 20.0,
    gk_backup_xmins: float = 0.0,
    captain_positions: str = "MID,FWD",
    vice_positions: str = "MID,FWD,DEF,GKP",
    bench_min_xmins: float = 45.0,          # (depth trimming handles this loosely)
    use_model_ep: bool = True,
    fdr_weight: float = 0.10,
    hweights: str = "",
    explain: bool = True,
    json_out: Optional[str] = None,
    use_advanced: bool = False,             # Use LP-based optimization
    **kwargs,
) -> Dict[str, Any]:
    """
    Squad optimizer. Uses advanced LP solver if use_advanced=True,
    otherwise falls back to greedy approach.
    """
    # Load current squad if use_myteam is specified
    # BUT: on wildcard, we don't want current_squad to constrain the optimization
    # We only need it to calculate the budget
    current_squad = kwargs.get('current_squad')
    wildcard = kwargs.get('wildcard', False)

    if use_myteam and not current_squad:
        # Load current squad from myteam data (for budget calculation)
        import json
        import os
        myteam_file = "data/processed/myteam_latest.json"
        if os.path.exists(myteam_file):
            with open(myteam_file) as f:
                myteam_data = json.load(f)
            current_squad = [pick["element"] for pick in myteam_data.get("picks", [])]

            # On wildcard, calculate budget but don't pass current_squad to optimizer
            # This allows building a completely fresh team
            if wildcard:
                # Calculate wildcard budget from selling prices
                selling_value = sum(pick.get("selling_price", pick.get("purchase_price", 0))
                                   for pick in myteam_data.get("picks", []))
                bank = myteam_data.get("transfers", {}).get("bank", 0)
                wildcard_budget = selling_value + bank
                kwargs['budget_limit'] = wildcard_budget
                log.info(f"Wildcard mode: Budget = £{wildcard_budget/10:.1f}m (selling value £{selling_value/10:.1f}m + bank £{bank/10:.1f}m)")
                # Don't pass current_squad - build fresh team
                current_squad = None
            else:
                # Regular transfers - use current squad
                kwargs['current_squad'] = current_squad
                log.info(f"Loaded current squad from myteam: {len(current_squad)} players")
        else:
            log.warning(f"use_myteam=True but myteam file not found at {myteam_file}")

    # Special case: No transfers with current squad - just evaluate current squad
    max_transfers = kwargs.get('max_transfers', 2)
    
    if current_squad and max_transfers == 0:
        # Just evaluate the current squad without optimization
        log.info("No transfers allowed - evaluating current squad")
        js = _fetch_bootstrap()
        xmins_map = _load_xmins()
        H = max(1, int(horizon))
        weights = _parse_weights(hweights, H)
        
        # Get player data for current squad
        players = []
        for element in js["elements"]:
            if element["id"] in current_squad:
                pid = element["id"]
                xmins = xmins_map.get(pid, element.get("minutes", 0) / 38 * 90 if element.get("minutes") else 45.0)
                
                # Calculate expected points
                ep_blend, _, _ = _load_ep_extras()
                ep1 = ep_blend.get(pid, element.get("ep_next", 0))
                
                players.append({
                    "id": pid,
                    "name": element["web_name"],
                    "team": element["team"],
                    "position": ["GKP", "DEF", "MID", "FWD"][element["element_type"] - 1],
                    "cost": element["now_cost"] / 10,
                    "xmins": xmins,
                    "ep": ep1,
                    "selected_by": element["selected_by_percent"],
                })
        
        # Calculate total expected points
        total_ep = sum(p["ep"] for p in players)

        # Generate full lineup output for current squad (same format as transfers)
        # This ensures lineup is displayed when banking/holding

        # Sort players by expected points to determine captain and lineup
        sorted_players = sorted(players, key=lambda x: -x["ep"])

        # Find the best captain (highest EP)
        captain = sorted_players[0] if sorted_players else None
        captain_id = captain["id"] if captain else None

        # Find vice captain (second highest EP)
        vice_captain = sorted_players[1] if len(sorted_players) > 1 else None
        vice_captain_id = vice_captain["id"] if vice_captain else None

        # Determine the best formation and lineup
        # Group players by position
        by_position = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
        for p in players:
            by_position[p["position"]].append(p)

        # Sort each position by EP
        for pos in by_position:
            by_position[pos].sort(key=lambda x: -x["ep"])

        # Common formations to try
        formations_to_try = ["442", "433", "343", "352", "451", "541", "532"]
        best_formation = None
        best_xi = []
        best_ep = 0

        for form in formations_to_try:
            n_def = int(form[0])
            n_mid = int(form[1])
            n_fwd = int(form[2])

            # Check if we have enough players for this formation
            if (len(by_position["GKP"]) >= 1 and
                len(by_position["DEF"]) >= n_def and
                len(by_position["MID"]) >= n_mid and
                len(by_position["FWD"]) >= n_fwd):

                # Build XI for this formation
                xi = []
                xi.extend(by_position["GKP"][:1])  # 1 GKP
                xi.extend(by_position["DEF"][:n_def])
                xi.extend(by_position["MID"][:n_mid])
                xi.extend(by_position["FWD"][:n_fwd])

                # Calculate EP for this XI
                xi_ep = sum(p["ep"] for p in xi)
                # Add captain bonus
                if captain and captain in xi:
                    xi_ep += captain["ep"]

                if xi_ep > best_ep:
                    best_ep = xi_ep
                    best_xi = xi
                    best_formation = form

        if not best_formation:
            # Fallback to a default formation if none work
            best_formation = "442"
            best_xi = []
            best_xi.extend(by_position["GKP"][:1])
            best_xi.extend(by_position["DEF"][:4])
            best_xi.extend(by_position["MID"][:4])
            best_xi.extend(by_position["FWD"][:2])

        # Determine bench (everyone not in XI)
        xi_ids = {p["id"] for p in best_xi}
        bench = [p for p in players if p["id"] not in xi_ids]
        bench.sort(key=lambda x: -x["ep"])  # Sort bench by EP

        # Build human readable output with full lineup
        output = []
        output.append(f"Current squad evaluated: {total_ep:.1f} expected points")
        output.append("")
        output.append(f"=== OPTIMAL LINEUP (Formation {best_formation}) ===")
        output.append("Starting XI:")

        # Sort XI by position for display
        for p in sorted(best_xi, key=lambda x: ["GKP", "DEF", "MID", "FWD"].index(x["position"])):
            status = ""
            if p["id"] == captain_id:
                status = " (C)"
            elif p["id"] == vice_captain_id:
                status = " (VC)"
            output.append(f" - {p['name']} ({p['position']}) — {p['ep']:.2f} pts{status}")

        output.append("")
        output.append("Bench:")
        for i, p in enumerate(bench, 1):
            output.append(f" {i}. {p['name']} ({p['position']}) — {p['ep']:.2f} pts")

        output.append("")
        if captain:
            output.append(f"Captain: {captain['name']} - {captain['ep']:.2f} pts (doubled = {captain['ep']*2:.2f})")
        if vice_captain:
            output.append(f"Vice-Captain: {vice_captain['name']} - {vice_captain['ep']:.2f} pts")

        # Compute dead players and club violations for metadata
        dead_players_meta = []
        for p in players:
            if p["ep"] == 0 or p["xmins"] == 0:
                dead_players_meta.append({"id": p["id"], "name": p["name"], "pos": p["position"]})

        club_violation_meta = {}
        team_counts_meta = {}
        for p in players:
            tid = p["team"]
            team_counts_meta[tid] = team_counts_meta.get(tid, 0) + 1
        teams_map_meta = {t["id"]: t["name"] for t in js.get("teams", [])}
        for tid, cnt in team_counts_meta.items():
            if cnt > MAX_PER_CLUB:
                club_violation_meta[tid] = cnt

        return {
            "squad": players,
            "expected_points_gw1": total_ep,
            "total_cost": sum(p["cost"] for p in players),
            "human_readable": "\n".join(output),
            # Metadata for verbose reporting
            "dead_players": dead_players_meta,
            "club_violation_info": club_violation_meta,
            "removes_dead": False,
            "fixes_club_violation": False,
            "horizon": horizon,
        }
    
    # Use LP optimizer if requested and available
    if use_advanced and HAS_PULP:
        log.info("Using advanced LP-based optimization")
        
        result = optimize_with_lp(
            horizon=horizon,
            bench_weight=bench_weight,
            bench_budget=bench_budget,
            formations=formations,
            differential_bonus=kwargs.get('differential_bonus', 0.1),
            risk_penalty=kwargs.get('risk_penalty', 0.05),
            value_weight=kwargs.get('value_weight', 0.3),
            current_squad=current_squad,
            max_transfers=max_transfers,
            wildcard=kwargs.get('wildcard', False),
            budget_limit=kwargs.get('budget_limit')  # Pass wildcard budget if set
        )
        
        if result and "error" not in result:
            return result
        else:
            if result:
                log.warning(f"LP optimization failed: {result.get('error', 'Unknown error')}. Falling back to greedy.")
            else:
                log.warning("LP optimization returned None. Falling back to greedy.")
    elif use_advanced and not HAS_PULP:
        log.warning("PuLP not installed. Install with 'pip install pulp' for LP optimization. Using greedy approach.")
    
    # Original greedy implementation
    js = _fetch_bootstrap()
    xmins_map = _load_xmins()

    H = max(1, int(horizon))
    weights = _parse_weights(hweights, H)
    if explain:
        wtxt = ",".join(f"{w:.2f}" for w in weights)
        log.info(
            "Objective: next %d GWs, weights=[%s], bench_weight=%.2f, fdr_weight=%.2f, bench_budget=£%.1fm",
            H, wtxt, bench_weight, fdr_weight, bench_budget / 10.0
        )

    pool = _candidate_pool(js, xmins_map, H=H, weights=weights, fdr_weight=fdr_weight)
    _adjust_xmins_with_depth(pool, nonstarter_xmins=float(nonstarter_xmins), gk_backup_xmins=float(gk_backup_xmins))

    if not pool:
        return {"human_readable": "No candidates found from FPL bootstrap.", "picked_ids": []}

    # Build a cheap bench upfront (guarantees cap if feasible)
    bench_template = _choose_cheap_bench_template(pool, bench_budget)
    bt_cost = _total_spent(bench_template)
    if explain:
        log.info("Bench template cost: £%.1fm (cap £%.1fm)", bt_cost / 10.0, bench_budget / 10.0)

    allowed_forms = [f.strip() for f in formations.split(",") if f.strip() in FORMATIONS]
    if not allowed_forms:
        allowed_forms = ["343", "352", "442"]

    picked = _pick15_with_bench_template(pool, bench_template)
    xi, bench, form = _choose_xi_and_bench(picked, allowed_forms, bench_budget, pool=pool)

    # Final safety: if still over cap (should be rare), force cheapest same-pos replacements until ≤ cap
    if _bench_cost(bench) > bench_budget:
        bench_sorted_idx = sorted(range(len(bench)), key=lambda i: bench[i].cost, reverse=True)
        clubs = _club_counts(picked)
        picked_ids = {p.id for p in picked}
        for idx in bench_sorted_idx:
            if _bench_cost(bench) <= bench_budget:
                break
            b = bench[idx]
            candidates = [q for q in pool if (q.pos == b.pos and q.id not in picked_ids)]
            candidates.sort(key=lambda z: (z.cost, -z.ep1))
            for q in candidates:
                if q.cost >= b.cost:
                    break
                clubs_after = dict(clubs)
                clubs_after[b.team] = clubs_after.get(b.team, 0) - 1
                if clubs_after[b.team] <= 0:
                    clubs_after.pop(b.team, None)
                clubs_after[q.team] = clubs_after.get(q.team, 0) + 1
                if clubs_after[q.team] > MAX_PER_CLUB:
                    continue
                if (_total_spent(picked) - b.cost + q.cost) > BUDGET:
                    continue
                picked = [q if (pp.id == b.id) else pp for pp in picked]
                picked_ids = {p.id for p in picked}
                clubs = clubs_after
                xi = _xi_for_formation(picked, form)
                xi, bench = _repair_bench_budget(xi, picked, bench_budget)
                break

    # captain/vice by ep1 * availability
    # Simplified: Pick highest EP for captain, second highest for vice (considering position restrictions)
    def p_play(p: Player) -> float:
        return min(1.0, max(0.0, p.xmins / 60.0))

    allowed_c_pos = {p.strip().upper() for p in captain_positions.split(",") if p.strip()} or {"MID", "FWD"}
    allowed_v_pos = {p.strip().upper() for p in vice_positions.split(",") if p.strip()} or {"MID", "FWD", "DEF"}

    # Sort XI by expected points * playing probability
    xi_sorted = sorted(xi, key=lambda p: p.ep1 * p_play(p), reverse=True)
    
    # Pick captain from allowed positions
    cap = None
    for p in xi_sorted:
        if p.pos in allowed_c_pos:
            cap = p
            break
    if cap is None:
        cap = xi_sorted[0]  # Fallback to highest EP overall
    
    # Pick vice from allowed positions (excluding captain)
    vice = None
    for p in xi_sorted:
        if p.id != cap.id and p.pos in allowed_v_pos:
            vice = p
            break
    if vice is None:
        # Fallback to second highest EP overall
        for p in xi_sorted:
            if p.id != cap.id:
                vice = p
                break

    def fmt(p: Player) -> str:
        return f" - {p.name} ({p.pos}) — ep1 {p.ep1:.2f}, next{H} {p.eph:.2f}, xMins {p.xmins:.0f}, £{p.cost/10:.1f}m"

    human = []
    human.append(f"Suggested 15-man squad (budget 100.0m), formation {form}:")
    for p in picked:
        human.append(fmt(p))
    human.append("")
    human.append("Starting XI:")
    for p in xi:
        human.append(fmt(p))
    human.append("")
    human.append("Bench:")
    for p in bench:
        human.append(fmt(p))
    human.append("")
    human.append(f"Bench spend: £{_bench_cost(bench)/10:.1f}m (cap £{bench_budget/10:.1f}m)")
    human.append(f"Captain: {cap.name} ({cap.pos})")
    human.append(f"Vice: {vice.name} ({vice.pos})")

    return {
        "human_readable": "\n".join(human),
        "picked_ids": [p.id for p in picked],
        "xi_ids": [p.id for p in xi],
        "bench_ids": [p.id for p in bench],
        "formation": form,
        "bench_cost": _bench_cost(bench) / 10.0,
        "bench_budget": bench_budget / 10.0,
        "captain_id": cap.id,
        "vice_id": vice.id,
        "table": [
            {"player_id": p.id, "name": p.name, "pos": p.pos, "team_id": p.team,
             "cost": p.cost/10.0, "xmins": p.xmins, "ep1": p.ep1, f"next{H}": p.eph}
            for p in picked
        ],
    }