# src/fpl_predict/cli.py
from __future__ import annotations

import json
import click

from .pipeline.update_weekly import update_weekly_data
from .myteam.ingest import sync_myteam
from .myteam.prices import show_price_changes
from .transfer.optimizer import optimize_transfers
from .reporting.exports import export_expected_points_table
from .strategy.chips_2025_final import plan_chips_2025
from .auth.login import set_token_env, set_cookie_env, test_auth, pw_login


@click.group()
def main() -> None:
    """FPL Predict CLI"""


# --------------------- update --------------------- #
@main.command("update")
@click.option("--run", is_flag=True, help="Run the weekly update.")
@click.option("--demo", is_flag=True, help="Use bundled sample data only.")
@click.option("--advanced", is_flag=True, help="Use advanced ML models (XGBoost/ensemble).")
def update_cmd(run: bool, demo: bool, advanced: bool) -> None:
    if advanced:
        import os
        os.environ['FPL_USE_ADVANCED_MODELS'] = 'true'
        click.echo("Advanced models enabled (XGBoost, ensemble, uncertainty quantification)")
    if run:
        update_weekly_data(demo_mode=demo)
    else:
        click.echo("Use --run to execute the weekly update.")


# --------------------- auth --------------------- #
@main.group("auth")
def auth_group() -> None:
    """Authentication helpers."""


@auth_group.command("set-token")
@click.option("--token", prompt=True, hide_input=True,
              help="Paste the full x-api-authorization Bearer token (or the part after 'Bearer ').")
@click.option("--save-env/--no-save-env", default=True, show_default=True)
def auth_set_token(token: str, save_env: bool) -> None:
    set_token_env(token, save_env_path=".env" if save_env else None)
    click.echo("Saved FPL_AUTH_TOKEN.")


@auth_group.command("set-cookie")
@click.option("--cookie", prompt=True,
              help="Paste the full Cookie header value from a logged-in request to fantasy.premierleague.com.")
@click.option("--save-env/--no-save-env", default=True, show_default=True)
def auth_set_cookie(cookie: str, save_env: bool) -> None:
    set_cookie_env(cookie, save_env_path=".env" if save_env else None)
    click.echo("Saved FPL_SESSION.")


@auth_group.command("test")
@click.option("--entry", type=int, required=True, help="Your FPL entry id.")
def auth_test(entry: int) -> None:
    code = test_auth(entry)
    click.echo(f"GET /api/my-team/{entry}/ → HTTP {code}")
    if code == 200:
        click.echo("Looks good ✅")
    elif code == 403:
        click.echo("Forbidden (403). Your token/cookie is invalid or expired.")
    else:
        click.echo("Check your auth values.")


@auth_group.command("pw-login")
@click.option("--email", type=str, default=None, help="Override email (else uses .env via settings).")
@click.option("--password", type=str, default=None, help="Override password (else uses .env via settings).")
@click.option("--save-env/--no-save-env", default=True, show_default=True, help="Write FPL_SESSION to .env")
def auth_pw_login(email: str | None, password: str | None, save_env: bool) -> None:
    from .config import settings

    cookie = pw_login(
        email=email or settings.FPL_EMAIL,
        password=password or settings.FPL_PASSWORD,
        save_env_path=".env" if save_env else None,
    )
    click.echo("Got FPL_SESSION:")
    click.echo(cookie[:80] + ("…" if len(cookie) > 80 else ""))


# --------------------- myteam --------------------- #
@main.group("myteam")
def myteam_group() -> None:
    """My-team utilities."""


@myteam_group.command("sync")
@click.option("--entry", type=int, required=True, help="Your FPL entry id.")
def myteam_sync_cmd(entry: int) -> None:
    sync_myteam(entry_id=entry)
    click.echo("My-team snapshot written → data/processed/myteam_latest.json")


@myteam_group.command("prices")
def myteam_prices() -> None:
    show_price_changes()


# --------------------- export-ep --------------------- #
@main.command("export-ep")
@click.option("--horizon", type=int, default=5, show_default=True,
              help="Number of upcoming gameweeks to aggregate.")
@click.option("--out", "out_path", type=str, default="reports/ep_next5.parquet", show_default=True,
              help="Output file path (.parquet or .csv).")
@click.option("--fmt", type=click.Choice(["parquet", "csv"]), default=None,
              help="Explicit format; inferred from extension if omitted.")
@click.option("--fdr-weight", type=float, default=0.25, show_default=True,
              help="How strongly to scale by fixture difficulty per GW.")
@click.option("--weights", "weights_csv", type=str, default="",
              help="Optional comma weights for the horizon (e.g. '1,0.9,0.8,0.65,0.55').")
@click.option("--no-pergw", "include_pergw", is_flag=True, flag_value=False, default=True,
              help="If set, omit ep_gw1..ep_gwH columns.")
def export_ep_cmd(horizon: int, out_path: str, fmt: str | None,
                  fdr_weight: float, weights_csv: str, include_pergw: bool) -> None:
    """Write a table with player name + expected points over next H GWs."""
    df = export_expected_points_table(
        horizon=horizon,
        out_path=out_path,
        fmt=fmt,
        fdr_weight=fdr_weight,
        weights_csv=weights_csv,
        include_pergw=include_pergw,
    )
    click.echo(f"Wrote {len(df)} rows → {out_path}")


# --------------------- transfers --------------------- #
@main.group("transfers")
def transfers_group() -> None:
    """Transfer optimization and recommendations."""


@transfers_group.command("recommend")
@click.option("--entry", type=int, help="FPL team ID (uses myteam_latest.json if not provided)")
@click.option("--max-transfers", type=int, default=1, show_default=True,
              help="Maximum transfers to consider (1 or 2)")
@click.option("--horizon", type=int, default=5, show_default=True,
              help="Planning horizon in gameweeks")
@click.option("--consider-hits/--no-hits", default=False, show_default=True,
              help="Consider taking a -4 hit for 2 transfers")
@click.option("--no-banking", is_flag=True, default=False,
              help="Disable banking strategy evaluation")
def transfers_recommend(entry: int | None, max_transfers: int, horizon: int, consider_hits: bool, no_banking: bool) -> None:
    """Recommend transfers for your existing team."""
    from .transfer.recommend import recommend_weekly_transfers
    
    try:
        recommendation = recommend_weekly_transfers(
            max_transfers=max_transfers,
            planning_horizon=horizon,
            consider_hits=consider_hits,
            entry_id=entry,
            evaluate_banking=not no_banking  # Enable banking by default
        )
        
        # The recommendation dict always has human_readable from format_recommendation_output
        if "human_readable" in recommendation and recommendation["human_readable"]:
            click.echo(recommendation["human_readable"])
        elif "error" in recommendation:
            click.echo(f"Error: {recommendation['error']}")
        else:
            click.echo("No recommendation could be generated.")
        
    except FileNotFoundError as e:
        click.echo(f"Error: {e}")
        click.echo("Run 'fpl myteam sync --entry YOUR_ID' first to download your current team.")
    except Exception as e:
        click.echo(f"Error generating recommendations: {e}")


@transfers_group.command("optimize")
@click.option("--use-myteam/--no-use-myteam", default=False, show_default=True,
              help="Optimize using your live team (requires FPL_SESSION).")
@click.option("--horizon", type=int, default=5, show_default=True,
              help="Projection horizon in gameweeks (display only).")
@click.option("--bench-weight", type=float, default=0.10, show_default=True,
              help="Weight for bench points in objective (0..1).")
@click.option("--bench-budget", type=int, default=180, show_default=True,
              help="Max spend on bench (in tenths of a million).")
@click.option("--formations", type=str, default="343,352,442,451,433", show_default=True,
              help="Allowed formations, comma-separated.")
@click.option("--nonstarter-xmins", type=float, default=20.0, show_default=True,
              help="Expected minutes cap for non-starters (outfield).")
@click.option("--gk-backup-xmins", type=float, default=0.0, show_default=True,
              help="Expected minutes cap for backup goalkeepers.")
@click.option("--bench-min-xmins", type=float, default=45.0, show_default=True,
              help="Minimum xMins to be eligible for the bench.")
@click.option("--captain-positions", type=str, default="MID,FWD", show_default=True,
              help="Eligible captain positions (comma-separated).")
@click.option("--vice-positions", type=str, default="MID,FWD,DEF,GKP", show_default=True,
              help="Eligible vice-captain positions (comma-separated).")
@click.option("--use-model-ep/--no-use-model-ep", default=True, show_default=True,
              help="Use model-based expected points when available.")
@click.option("--fdr-weight", type=float, default=0.10, show_default=True,
              help="Blend weight for next-5 FDR into EP (0..1).")
@click.option("--hweights", type=str, default="", show_default=False,
              help="Comma weights for horizon, e.g. '1,0.8,0.6,0.5,0.4'.")
@click.option("--explain/--no-explain", default=True, show_default=True,
              help="Print a short breakdown of the optimization objective.")
@click.option("--json-out", type=click.Path(dir_okay=False, writable=True), default=None,
              help="Write the raw plan JSON to this file.")
# NEW role/avoid knobs
@click.option("--mid-floor-xgi90", type=float, default=0.22, show_default=True,
              help="MIDs with prior xGI/90 below this are damped.")
@click.option("--mid-floor-damp", type=float, default=0.25, show_default=True,
              help="Proportional damp (e.g. 0.25 = -25%) applied to such MIDs.")
@click.option("--avoid", type=str, default="", show_default=False,
              help="Comma-separated player surnames to exclude, e.g. 'Enzo,Szoboszlai'.")
@click.option("--use-lp/--no-use-lp", default=False, show_default=True,
              help="Use advanced Linear Programming optimizer instead of greedy algorithm.")
@click.option("--differential-bonus", type=float, default=0.1, show_default=True,
              help="Bonus weight for low-ownership high-upside players.")
@click.option("--risk-penalty", type=float, default=0.05, show_default=True,
              help="Penalty for injury/rotation risk.")
@click.option("--value-weight", type=float, default=0.3, show_default=True,
              help="Weight for value (points per million) in objective.")
@click.option("--wildcard/--no-wildcard", default=False, show_default=True,
              help="Use wildcard chip (all players at current market price).")
def transfers_optimize_cmd(
    use_myteam: bool,
    horizon: int,
    bench_weight: float,
    bench_budget: int,
    formations: str,
    nonstarter_xmins: float,
    gk_backup_xmins: float,
    captain_positions: str,
    vice_positions: str,
    bench_min_xmins: float,
    use_model_ep: bool,
    fdr_weight: float,
    hweights: str,
    explain: bool,
    json_out: str | None,
    mid_floor_xgi90: float,
    mid_floor_damp: float,
    avoid: str,
    use_lp: bool,
    differential_bonus: float,
    risk_penalty: float,
    value_weight: float,
    wildcard: bool,
) -> None:
    """Suggest a 15-man squad + XI (and captain/vice) under budget/constraints."""
    plan = optimize_transfers(
        use_myteam=use_myteam,
        horizon=horizon,
        bench_weight=bench_weight,
        bench_budget=bench_budget,
        formations=formations,
        nonstarter_xmins=nonstarter_xmins,
        gk_backup_xmins=gk_backup_xmins,
        captain_positions=captain_positions,
        vice_positions=vice_positions,
        bench_min_xmins=bench_min_xmins,
        use_model_ep=use_model_ep,
        fdr_weight=fdr_weight,
        hweights=hweights,
        explain=explain,
        json_out=json_out,
        mid_floor_xgi90=mid_floor_xgi90,
        mid_floor_damp=mid_floor_damp,
        avoid=avoid,
        use_advanced=use_lp,
        differential_bonus=differential_bonus,
        risk_penalty=risk_penalty,
        value_weight=value_weight,
        wildcard=wildcard,
    )

    click.echo(plan.get("human_readable", "No plan generated."))

    if json_out:
        with open(json_out, "w") as f:
            json.dump(plan, f, indent=2)
        click.echo(f"\nWrote JSON plan → {json_out}")


# --------------------- chips --------------------- #
@main.group("chips")
def chips_group() -> None:
    """Chip planning commands."""


@chips_group.command("plan")
@click.option(
    "--horizon",
    type=click.Choice(["H1", "H2", "season"], case_sensitive=False),
    default="H1",
    show_default=True,
    help="Window to plan over: first half, second half, or whole season.",
)
@click.option(
    "--use-myteam/--no-use-myteam",
    default=False,
    show_default=True,
    help="Make recommendations based on your synced squad (fpl myteam sync).",
)
@click.option("--tc-min-ep", type=float, default=8.0, show_default=True, help="Min owned captain EP to TC.")
@click.option("--bb-min-ep", type=float, default=12.0, show_default=True, help="Min total bench EP to BB.")
@click.option("--fh-delta-min", type=float, default=12.0, show_default=True, help="Min (ideal XI − owned XI) to FH.")
@click.option("--bench-min-xmins", type=float, default=45.0, show_default=True, help="Bench players must have xMins ≥ this.")
@click.option("--explain/--no-explain", default=True, show_default=True, help="Log rationale and write metrics CSV.")
def chips_plan_cmd(
    horizon: str,
    use_myteam: bool,
    tc_min_ep: float,
    bb_min_ep: float,
    fh_delta_min: float,
    bench_min_xmins: float,
    explain: bool,
) -> None:
    from .strategy.chips_2025_final import plan_chips_2025

    plan_chips(
        horizon=horizon,
        use_myteam=use_myteam,
        tc_min_ep=tc_min_ep,
        bb_min_ep=bb_min_ep,
        fh_delta_min=fh_delta_min,
        bench_min_xmins=bench_min_xmins,
        explain=explain,
    )


@chips_group.command("plan-2025")
@click.option(
    "--use-myteam/--no-use-myteam",
    default=True,
    show_default=True,
    help="Make recommendations based on your synced squad (fpl myteam sync).",
)
@click.option("--explain/--no-explain", default=True, show_default=True, help="Show detailed strategy explanation.")
@click.option("--show-teams", is_flag=True, default=False, help="Show full Free Hit team for recommended gameweeks.")
def chips_plan_2025_cmd(
    use_myteam: bool,
    explain: bool,
    show_teams: bool,
) -> None:
    """Plan chips using 2025/26 double chips system (8 total chips)"""
    from .strategy.chips_2025_final import plan_chips_2025

    click.echo("📌 FPL 2025/26 Double Chips Strategy")
    click.echo("Using new rules: 2 sets of chips (H1 expires GW19, H2 starts GW20)")
    click.echo("")

    plan_chips_2025(
        use_myteam=use_myteam,
        explain=explain,
        show_teams=show_teams,
    )


@chips_group.command("free-hit")
@click.option("--gw", type=int, required=True, help="Gameweek to generate Free Hit team for")
def chips_free_hit_cmd(gw: int) -> None:
    """Generate optimal Free Hit XI for a specific gameweek"""
    from .strategy.chips_2025_final import generate_free_hit_team

    click.echo(f"Generating optimal Free Hit team for GW{gw}...")
    click.echo("")

    result = generate_free_hit_team(gw)
    click.echo(result)


@chips_group.command("free-hit-analysis")
@click.option("--gw-start", type=int, default=None, help="Starting gameweek (defaults to current GW)")
@click.option("--gw-end", type=int, default=19, show_default=True, help="Ending gameweek")
def chips_free_hit_analysis_cmd(gw_start: int, gw_end: int) -> None:
    """Analyze Free Hit value across all gameweeks"""
    from .strategy.chips_2025_final import analyze_free_hit_all_gws

    result = analyze_free_hit_all_gws(gw_start, gw_end)
    click.echo(result)


if __name__ == "__main__":
    main()