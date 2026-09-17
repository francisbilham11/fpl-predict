from __future__ import annotations

import pathlib
from typing import List, Dict, Optional
import pandas as pd

from ..data.fpl_api import get_bootstrap, get_fixtures
from ..utils.cache import PROC
from ..utils.io import read_parquet
from ..utils.logging import get_logger

log = get_logger(__name__)


def _default_weights(H: int) -> list[float]:
    """Near-week emphasis; gentle decay after week 5."""
    base = [1.00, 0.90, 0.80, 0.65, 0.55]
    if H <= len(base):
        return base[:H]
    return base + [0.50] * (H - len(base))


def _pergw_factors_from_fpl(team_id: int, H: int) -> list[float]:
    """Return up to H difficulty factors (>=0), one per upcoming GW for this team.
    We map FPL difficulty 1..5 to a multiplicative factor around 1.0.
    diff=3 -> 1.00, diff=1 -> ~1.16 (easy), diff=5 -> ~0.84 (hard)."""
    try:
        fx = get_fixtures()
    except Exception:
        return [1.0] * H

    ups = [
        f for f in fx
        if (not f.get("finished")) and f.get("event") and (f.get("team_h") == team_id or f.get("team_a") == team_id)
    ]
    ups.sort(key=lambda f: f.get("event", 999))

    factors = []
    for f in ups[:H]:
        if f.get("team_h") == team_id:
            diff = int(f.get("team_h_difficulty") or 3)
        else:
            diff = int(f.get("team_a_difficulty") or 3)
        fac = 1.0 + (3 - diff) * 0.08          # ~[1.16, 1.08, 1.00, 0.92, 0.84]
        fac = max(0.80, min(1.20, fac))        # clamp for stability
        factors.append(fac)
    while len(factors) < H:
        factors.append(1.0)
    return factors


def _bootstrap_maps() -> tuple[Dict[int, str], Dict[int, str], Dict[int, str]]:
    """Return (player_id->name, player_id->position, team_id->team_name) from FPL bootstrap."""
    js = get_bootstrap()
    pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

    id2name = {int(e["id"]): e["web_name"] for e in js["elements"]}
    id2pos = {int(e["id"]): pos_map.get(e["element_type"], "?") for e in js["elements"]}
    team2name = {int(t["id"]): t["name"] for t in js["teams"]}
    return id2name, id2pos, team2name


def export_expected_points_table(
    horizon: int = 5,
    out_path: Optional[str | pathlib.Path] = None,
    fmt: Optional[str] = None,
    fdr_weight: float = 0.25,
    weights_csv: str = "",
    include_pergw: bool = True,
) -> pd.DataFrame:
    """
    Build a table with player name + expected points over the next `horizon` gameweeks,
    minutes-adjusted, and fixture-difficulty–adjusted. Writes CSV/Parquet if `out_path` given.

    Data sources:
      - PROC/exp_points.parquet (prefers 'ep_blend', else 'ep_model' as base EP per GW)
      - PROC/xmins.parquet (expected minutes per player, single value reused for horizon)
      - FPL bootstrap (names/positions/teams)
      - FPL fixtures (per-GW difficulty → factors)
    """

    # Load model EP (or fallback)
    ep_df = read_parquet(PROC / "exp_points.parquet")
    ep_col = "ep_blend" if "ep_blend" in ep_df.columns else ("ep_model" if "ep_model" in ep_df.columns else None)
    if ep_col is None:
        raise RuntimeError("exp_points.parquet is missing 'ep_blend'/'ep_model' columns.")
    ep_map = {int(r.player_id): float(getattr(r, ep_col)) for _, r in ep_df.iterrows()}

    # Load xMins (minutes per GW; reused across horizon)
    xm_df = read_parquet(PROC / "xmins.parquet")
    xmins_map = {int(r.player_id): float(r.xmins) for _, r in xm_df.iterrows()}

    # Names/positions/teams
    id2name, id2pos, team2name = _bootstrap_maps()

    # Which team a player belongs to? Use the latest features parquet for 'team'
    # (fall back to bootstrap 'team' if you store it there instead)
    try:
        feats = read_parquet(PROC / "features.parquet")
        pid2team = {int(r.player_id): int(r.team) for _, r in feats.iterrows()}
        cost_map = {int(r.player_id): int(r.now_cost) for _, r in feats.iterrows()}  # tenths of £m
    except Exception:
        pid2team = {}
        cost_map = {}

    # Weighting across horizon
    H = max(1, int(horizon))
    weights = [float(w) for w in weights_csv.split(",")] if weights_csv.strip() else _default_weights(H)
    if len(weights) < H:
        weights += [weights[-1]] * (H - len(weights))
    elif len(weights) > H:
        weights = weights[:H]

    # Compute per-player EP across H weeks
    rows = []
    for pid, ep_base in ep_map.items():
        name = id2name.get(pid, f"#{pid}")
        pos = id2pos.get(pid, "?")
        team_id = pid2team.get(pid)
        team_nm = team2name.get(team_id, str(team_id) if team_id is not None else "?")
        cost = cost_map.get(pid, None)
        xmins = xmins_map.get(pid, 70.0)
        xm_fac = max(0.0, min(1.0, xmins / 90.0))

        if team_id is None:
            pergw = [1.0] * H
        else:
            pergw = _pergw_factors_from_fpl(team_id, H)

        ep_pergw = []
        total = 0.0
        for k in range(H):
            fac = pergw[k]
            # minutes-adjusted + FDR-adjusted
            ep_k = ep_base * (1.0 + fdr_weight * (fac - 1.0)) * xm_fac
            ep_pergw.append(ep_k)
            total += weights[k] * ep_k

        row = {
            "player_id": pid,
            "player": name,
            "position": pos,
            "team_id": team_id,
            "team": team_nm,
            "cost_tenths": cost,
            "xmins": xmins,
            "ep_base": ep_base,
            "ep_nextH": total,
        }
        if include_pergw:
            for i, v in enumerate(ep_pergw, start=1):
                row[f"ep_gw{i}"] = v
        rows.append(row)

    out = pd.DataFrame(rows)
    # Nice sort: by ep_nextH desc
    out = out.sort_values("ep_nextH", ascending=False).reset_index(drop=True)

    # Write if requested
    if out_path:
        out_path = pathlib.Path(out_path)
        if fmt is None:
            fmt = "parquet" if out_path.suffix.lower() in {".parquet", ".arrow", ".pq"} else "csv"
        if fmt.lower() == "csv":
            out.to_csv(out_path, index=False)
        elif fmt.lower() == "parquet":
            out.to_parquet(out_path, index=False)
        else:
            raise ValueError(f"Unsupported fmt: {fmt}")
        log.info("Wrote expected-points table: %s (rows=%d)", out_path, len(out))

    return out