from __future__ import annotations

from ..config import current_season, season_label
from ..data.fpl_api import get_bootstrap
from ..data.fpl_history import build_history
from ..data.ingest import bootstrap_raw_from_sample, ingest_full
from ..data.process import build_features
from ..data.rules_fetcher import update_readme_scoring_table
from ..fixtures.fdr import build_player_next5_fdr, compute_fdr
from ..models.train import train_all
from ..utils.cache import DATA, MODELS, PROC, ROOT
from ..utils.io import read_parquet, write_json, write_parquet
from ..utils.logging import get_logger
from ..utils.time import now_utc_str
from .competition_detector import detect_position_competition
from .recent_transfers import apply_transfer_adjustments

log = get_logger(__name__)


def fix_squad_roles_post_training():
    """Apply pecking-order, transfer and availability adjustments to xMins, then rescale EP.

    Everything here is derived from the current squad rather than from hand-maintained name
    lists. The previous version carried a dict of backup keeper surnames and a dict of
    backup defenders with fixed minute caps, both captured from the 2024/25 squads; they
    survived two transfer windows and were pinning the wrong players to 0 and 10 minutes.
    """
    log.info("Applying squad role fixes post-training...")

    try:
        xmins_df = read_parquet(PROC / "xmins.parquet")
        bootstrap = get_bootstrap()
        # The pre-adjustment figure, so EP can be scaled by how far xMins moved rather than
        # recomputed from scratch. Recomputing assumed EP was a per-90 rate, which is true of
        # the shipped model but not of the component model, whose expected minutes are already
        # inside its prediction; the recompute would have discounted it a second time.
        original_xmins = dict(zip(xmins_df["player_id"], xmins_df["xmins"].astype(float)))
        xmins_df = xmins_df.set_index("player_id")

        n_changes = 0

        log.info("Detecting position competition...")
        competition_adjustments = detect_position_competition()
        for pid, factor in competition_adjustments.items():
            if pid not in xmins_df.index:
                continue
            before = float(xmins_df.at[pid, "xmins"])
            after = before * factor
            if abs(after - before) > 1e-9:
                xmins_df.at[pid, "xmins"] = after
                n_changes += 1
                log.debug(
                    "Player %s: %.0f -> %.0f xMins (competition %.2f)", pid, before, after, factor
                )

        log.info("Applying recent transfer adjustments...")
        xmins_map = apply_transfer_adjustments(dict(xmins_df["xmins"]))
        for pid, new_xmins in xmins_map.items():
            if pid in xmins_df.index and float(xmins_df.at[pid, "xmins"]) != float(new_xmins):
                xmins_df.at[pid, "xmins"] = float(new_xmins)
                n_changes += 1

        xmins_df = xmins_df.reset_index()

        log.info("Applying availability adjustments...")
        from ..models.availability import apply_availability_adjustments

        xmins_df = apply_availability_adjustments(xmins_df, bootstrap)
        write_parquet(xmins_df, PROC / "xmins.parquet")

        ep_df = read_parquet(PROC / "exp_points.parquet")
        ep_df = apply_availability_adjustments(ep_df, bootstrap)

        # Scale EP by the proportional change in expected minutes. A player cut from 90 to 45
        # keeps half their expected points; one cut to 0 loses all of them. This is equivalent
        # to the old recompute whenever EP really was a per-90 rate, and correct when it is
        # not.
        adjusted = dict(zip(xmins_df["player_id"], xmins_df["xmins"].astype(float)))
        ratio = ep_df["player_id"].map(
            lambda pid: (
                adjusted.get(pid, 0.0) / original_xmins[pid]
                if original_xmins.get(pid, 0.0) > 0
                else (0.0 if adjusted.get(pid, 0.0) <= 0 else 1.0)
            )
        )
        ep_df["ep_adjusted"] = ep_df["ep_adjusted"] * ratio.clip(lower=0.0, upper=1.0)
        write_parquet(ep_df, PROC / "exp_points.parquet")

        log.info(
            "Applied %d squad role and competition fixes; %d players scaled to zero EP",
            n_changes,
            int((ratio <= 0).sum()),
        )

    except Exception as e:
        log.warning("Squad role fixes failed: %s", e, exc_info=True)


def update_weekly_data(demo_mode: bool = False, model: str = "component") -> None:
    """Refresh data and produce expected points for the upcoming gameweek.

    `model` selects which expected-points model writes `exp_points.parquet`:

    | Value | Model | Measured XI capture |
    |:------|:------|:--------------------|
    | `component` | per-component model on the gameweek panel | 39.5% |
    | `shipped` | the original pipeline | 34.3% |

    Both figures come from `fpl backtest` over 151 gameweeks. `component` is the default
    because it beats the shipped design in 64.9% of gameweeks by +7.7 XI points.
    """
    log.info("Starting weekly update (demo=%s, model=%s)", demo_mode, model)

    # Step 1: Ingest match results
    if demo_mode:
        bootstrap_raw_from_sample()
    else:
        ingest_full()

        # Player-gameweek history for previous seasons. Completed seasons are cached on disk
        # and reused; only the season in progress is re-pulled.
        try:
            cur = current_season()
            build_history()
            build_history(seasons=[cur - 1], refresh=True)
        except Exception as e:
            log.warning("Could not refresh player-gameweek history: %s", e)

    # Step 2: Build features from the fresh data
    build_features()
    update_readme_scoring_table(ROOT / "README.md")
    compute_fdr()
    build_player_next5_fdr()

    # get_team_quality_scores() (used by the optimizer for team_att/team_strength) defaults
    # to reusing team_quality.parquet if it exists and is never called with refresh=True
    # anywhere else — once written, it would otherwise never be recalculated again, silently
    # freezing every future update onto whichever seasons happened to be "the last two
    # completed" the first time this ever ran, rather than sliding forward as seasons end.
    try:
        from ..data.team_quality import calculate_team_quality_from_data

        calculate_team_quality_from_data()
    except Exception as e:
        log.warning("Could not refresh team quality scores: %s", e)

    # Step 3: Drop stale training data so it is rebuilt against current player stats
    training_data_file = PROC / "training_data.parquet"
    if training_data_file.exists():
        log.info("Removing old training_data.parquet to force rebuild with current player stats")
        training_data_file.unlink()

    # Step 4: Expected points
    if model == "component" and not demo_mode:
        from ..models.panel import build_panel
        from ..models.points import train_and_predict_gameweek

        build_panel()
        train_and_predict_gameweek()
        write_json({"saved_at": now_utc_str(), "models": ["component"]}, MODELS / "latest.json")
    else:
        models = train_all()
        write_json(
            {"saved_at": now_utc_str(), "models": list(models.keys())}, MODELS / "latest.json"
        )

    # Step 5: Apply post-training fixes for squad roles
    fix_squad_roles_post_training()

    write_json(
        {
            "updated_at": now_utc_str(),
            "season": season_label(current_season()),
            "ep_model": model,
        },
        DATA / "processed" / "weekly_changelog.json",
    )
    log.info(
        "Weekly update complete for %s using the %s model.", season_label(current_season()), model
    )


