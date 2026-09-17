from __future__ import annotations
import pandas as pd
from ..utils.cache import RAW, PROC
from ..utils.io import read_parquet, write_parquet
from ..utils.logging import get_logger
from .fpl_api import get_bootstrap, get_element_summary
from .teams import canonical_team

log = get_logger(__name__)
FPL_BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"


def _seed_features_from_fpl() -> pd.DataFrame:
    js = get_bootstrap()
    els = js.get("elements", [])
    pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

    rows = []
    player_summaries = {}  # Cache summaries for reuse

    for i, e in enumerate(els, 1):
        pid = int(e["id"])

        # ---- per-player recent minutes (current season 2025-26) ----
        mins_l5 = mins_l3 = 0.0
        starts_l5 = bench_l5 = 0
        # ---- Defensive contributions (NEW) ----
        dc_l5 = dc_l3 = 0.0
        tackles_l5 = clearances_blocks_interceptions_l5 = recoveries_l5 = 0.0
        # ---- previous-season priors (from history_past) ----
        mins_prev = starts_prev = 0.0
        prev_goals_per90 = prev_assists_per90 = 0.0
        prev_xg_per90 = prev_xa_per90 = 0.0

        # Performance data (NEW!)
        goals_l5 = assists_l5 = goals_l3 = assists_l3 = 0.0
        xg_l5 = xa_l5 = xg_l3 = xa_l3 = 0.0
        bonus_l5 = bps_l5 = ict_l5 = 0.0
        clean_sheets_l5 = goals_conceded_l5 = saves_l5 = 0.0
        goals_per90 = assists_per90 = xg_per90 = xa_per90 = 0.0
        home_goals = away_goals = home_xg = away_xg = 0.0
        total_minutes = 0.0
        total_starts = 0
        avg_points = form_trend = 0.0

        try:
            summ = get_element_summary(pid)
            player_summaries[pid] = summ  # Cache for later

            # current season history - this includes GW1 of 2025-26!
            hist = summ.get("history") or []
            if hist and i <= 5:  # Log first few players to confirm we have 2025-26 data
                log.debug(f"Player {e.get('web_name')} has {len(hist)} gameweeks of 2025-26 data")
            last5 = hist[-5:]
            mins_l5 = sum(h.get("minutes") or 0 for h in last5) / max(1, len(last5))
            last3 = last5[-3:] if last5 else []
            mins_l3 = sum(h.get("minutes") or 0 for h in last3) / max(1, len(last3))
            starts_l5 = sum(1 for h in last5 if (h.get("minutes") or 0) >= 60)
            bench_l5 = sum(1 for h in last5 if 1 <= (h.get("minutes") or 0) < 60)

            # Extract defensive contributions
            dc_l5 = sum(h.get("defensive_contribution") or 0 for h in last5) / max(1, len(last5))
            dc_l3 = sum(h.get("defensive_contribution") or 0 for h in last3) / max(1, len(last3))
            tackles_l5 = sum(h.get("tackles") or 0 for h in last5) / max(1, len(last5))
            clearances_blocks_interceptions_l5 = sum(
                h.get("clearances_blocks_interceptions") or 0 for h in last5
            ) / max(1, len(last5))
            recoveries_l5 = sum(h.get("recoveries") or 0 for h in last5) / max(1, len(last5))

            # Extract actual performance data (CRITICAL!)
            goals_l5 = sum(h.get("goals_scored", 0) for h in last5)
            assists_l5 = sum(h.get("assists", 0) for h in last5)
            goals_l3 = sum(h.get("goals_scored", 0) for h in last3)
            assists_l3 = sum(h.get("assists", 0) for h in last3)

            xg_l5 = sum(float(h.get("expected_goals", 0)) for h in last5)
            xa_l5 = sum(float(h.get("expected_assists", 0)) for h in last5)
            xg_l3 = sum(float(h.get("expected_goals", 0)) for h in last3)
            xa_l3 = sum(float(h.get("expected_assists", 0)) for h in last3)

            bonus_l5 = sum(h.get("bonus", 0) for h in last5)
            bps_l5 = sum(h.get("bps", 0) for h in last5) / max(1, len(last5))
            ict_l5 = sum(float(h.get("ict_index", 0)) for h in last5) / max(1, len(last5))

            clean_sheets_l5 = sum(h.get("clean_sheets", 0) for h in last5)
            goals_conceded_l5 = sum(h.get("goals_conceded", 0) for h in last5)
            saves_l5 = sum(h.get("saves", 0) for h in last5)

            # Calculate per90 stats from all games
            total_minutes = sum(h.get("minutes", 0) for h in hist)
            total_starts = sum(h.get("starts", 0) for h in hist)
            total_goals = sum(h.get("goals_scored", 0) for h in hist)
            total_assists = sum(h.get("assists", 0) for h in hist)
            total_xg = sum(float(h.get("expected_goals", 0)) for h in hist)
            total_xa = sum(float(h.get("expected_assists", 0)) for h in hist)

            if total_minutes > 0:
                goals_per90 = (total_goals / total_minutes) * 90
                assists_per90 = (total_assists / total_minutes) * 90
                xg_per90 = (total_xg / total_minutes) * 90
                xa_per90 = (total_xa / total_minutes) * 90

            # Home/away splits
            home_games = [h for h in hist if h.get("was_home", False)]
            away_games = [h for h in hist if not h.get("was_home", False)]

            if home_games:
                home_goals = sum(h.get("goals_scored", 0) for h in home_games) / len(home_games)
                home_xg = sum(float(h.get("expected_goals", 0)) for h in home_games) / len(
                    home_games
                )

            if away_games:
                away_goals = sum(h.get("goals_scored", 0) for h in away_games) / len(away_games)
                away_xg = sum(float(h.get("expected_goals", 0)) for h in away_games) / len(
                    away_games
                )

            # Average points and form trend
            if hist:
                avg_points = sum(h.get("total_points", 0) for h in hist) / len(hist)

                if len(hist) >= 3:
                    recent_pts = sum(h.get("total_points", 0) for h in hist[-3:]) / 3
                    earlier_pts = sum(h.get("total_points", 0) for h in hist[:-3]) / max(
                        1, len(hist[:-3])
                    )
                    form_trend = recent_pts - earlier_pts

            # previous season rollup (most recent)
            past = summ.get("history_past") or []
            last = past[-1] if past else {}
            mins_prev = float(last.get("minutes") or 0.0)
            # some seasons expose 'starts', some just 'appearances'
            starts_prev = float(last.get("starts") or last.get("appearances") or 0.0)
            g_prev = float(last.get("goals_scored") or 0.0)
            a_prev = float(last.get("assists") or 0.0)

            # xG/xA may be totals or per90 depending on vintage; handle both
            xg_prev_tot = last.get("expected_goals")
            xa_prev_tot = last.get("expected_assists")
            xg_prev_p90 = last.get("expected_goals_per_90")
            xa_prev_p90 = last.get("expected_assists_per_90")

            def per90(val):
                return (
                    (float(val) / mins_prev * 90.0) if mins_prev and float(mins_prev) > 0 else 0.0
                )

            prev_goals_per90 = per90(g_prev)
            prev_assists_per90 = per90(a_prev)
            if xg_prev_p90 is not None and xa_prev_p90 is not None:
                prev_xg_per90 = float(xg_prev_p90 or 0.0)
                prev_xa_per90 = float(xa_prev_p90 or 0.0)
            else:
                prev_xg_per90 = per90(xg_prev_tot or 0.0)
                prev_xa_per90 = per90(xa_prev_tot or 0.0)

        except Exception:
            # leave defaults (zeros) if the per-player call fails
            pass

        chance_next = e.get("chance_of_playing_next_round")
        if chance_next is None:
            chance_next = 100 if (e.get("status") or "a") == "a" else 0

        # Also get defensive stats from bootstrap data
        dc_per90 = float(e.get("defensive_contribution_per_90") or 0.0)
        dc_total = float(e.get("defensive_contribution") or 0.0)
        cbi_total = float(e.get("clearances_blocks_interceptions") or 0.0)
        tackles_total = float(e.get("tackles") or 0.0)
        recoveries_total = float(e.get("recoveries") or 0.0)

        rows.append(
            {
                "player_id": pid,
                "position": pos_map.get(e.get("element_type")),
                "team": e.get("team"),
                "now_cost": e.get("now_cost"),
                "form": float(e.get("form") or 0.0),
                "ep_next": float(e.get("ep_next") or 0.0),
                "selected_by_percent": float(e.get("selected_by_percent") or 0.0),
                "status": e.get("status"),
                "chance_next": float(chance_next),
                # current-season signals
                "mins_l5": float(mins_l5),
                "mins_l3": float(mins_l3),
                "starts_l5": int(starts_l5),
                "bench_l5": int(bench_l5),
                # Defensive contributions (NEW!)
                "dc_l5": float(dc_l5),
                "dc_l3": float(dc_l3),
                "dc_per90": dc_per90,
                "dc_total": dc_total,
                "tackles_l5": float(tackles_l5),
                "tackles_total": tackles_total,
                "cbi_l5": float(clearances_blocks_interceptions_l5),
                "cbi_total": cbi_total,
                "recoveries_l5": float(recoveries_l5),
                "recoveries_total": recoveries_total,
                # ACTUAL PERFORMANCE DATA (CRITICAL!)
                "goals_l5": float(goals_l5),
                "assists_l5": float(assists_l5),
                "goals_l3": float(goals_l3),
                "assists_l3": float(assists_l3),
                "xg_l5": float(xg_l5),
                "xa_l5": float(xa_l5),
                "xg_l3": float(xg_l3),
                "xa_l3": float(xa_l3),
                "bonus_l5": float(bonus_l5),
                "bps_l5": float(bps_l5),
                "ict_l5": float(ict_l5),
                "clean_sheets_l5": float(clean_sheets_l5),
                "goals_conceded_l5": float(goals_conceded_l5),
                "saves_l5": float(saves_l5),
                "goals_per90": float(goals_per90),
                "assists_per90": float(assists_per90),
                "xg_per90": float(xg_per90),
                "xa_per90": float(xa_per90),
                "home_goals": float(home_goals),
                "away_goals": float(away_goals),
                "home_xg": float(home_xg),
                "away_xg": float(away_xg),
                "total_minutes": float(total_minutes),
                "total_starts": int(total_starts),
                "avg_points": float(avg_points),
                "form_trend": float(form_trend),
                # previous-season priors
                "prev_minutes": float(mins_prev),
                "prev_starts": float(starts_prev),
                "prev_goals_per90": float(prev_goals_per90),
                "prev_assists_per90": float(prev_assists_per90),
                "prev_xg_per90": float(prev_xg_per90),
                "prev_xa_per90": float(prev_xa_per90),
            }
        )

        if i % 100 == 0:
            log.info("FPL per-player summaries processed: %d / %d", i, len(els))

    df = pd.DataFrame(rows)

    # minimal dtype hygiene
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df["position"] = df["position"].astype("string")
    df["team"] = pd.to_numeric(df["team"], errors="coerce").astype("Int64")

    write_parquet(df, PROC / "features.parquet")
    log.info(
        "Processed features seeded from FPL bootstrap + per-player history: %d players", len(df)
    )
    return df


FIXTURE_COLS = [
    "season",
    "match_id",
    "date",
    "gameweek",
    "home_team",
    "away_team",
    "home_slug",
    "away_slug",
    "home_goals",
    "away_goals",
]


def build_features() -> None:
    epl_all = RAW / "football-data" / "EPL_all_matches.parquet"
    if epl_all.exists():
        fx = read_parquet(epl_all).copy()
        # Slugs are what Elo, the clean-sheet model and team form join on, so backfill them
        # rather than letting a pre-slug file through and silently flattening those joins.
        if "home_slug" not in fx.columns:
            fx["home_slug"] = fx["home_team"].map(canonical_team)
        if "away_slug" not in fx.columns:
            fx["away_slug"] = fx["away_team"].map(canonical_team)
        for c in FIXTURE_COLS:
            if c not in fx.columns:
                fx[c] = None
        write_parquet(fx[FIXTURE_COLS], PROC / "fixtures.parquet")
        log.info("fixtures.parquet: %d matches with canonical slugs", len(fx))

        _seed_features_from_fpl()
        return

    # DEMO fallback
    fixtures = read_parquet(RAW / "sample" / "fixtures_sample.parquet")
    events = read_parquet(RAW / "sample" / "events_sample.parquet")
    feats = (
        events.groupby("player_id").agg(
            minutes=("minutes", "sum"), goals=("goals", "sum"), assists=("assists", "sum")
        )
    ).reset_index()
    write_parquet(fixtures, PROC / "fixtures.parquet")
    write_parquet(feats, PROC / "features.parquet")
    log.info("Processed features written (DEMO).")
