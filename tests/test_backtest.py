"""The backtest harness.

This is the measuring instrument, so it gets tested harder than the things it measures. It
replaces a one-line stub that returned `{"MAE": 1.2, "RMSE": 1.8}` and was never called,
while the README quoted those constants as measured performance.

One bug already found here in practice: FPL's `xP` column is missing (recorded as 0 for every
player) in several 2025-26 gameweeks, which made the baseline's XI selection arbitrary and
dropped its apparent capture from ~62% to 23%.
"""

import numpy as np
import pandas as pd
import pytest

from src.fpl_predict.models.backtest import (
    ColumnPredictor,
    CurrentArchitecturePredictor,
    PositionMeanPredictor,
    _best_xi_points,
    evaluate_gameweek,
    rolling_origin_backtest,
    summarise,
    usable_xp_gameweeks,
)


def _panel(n_gw: int = 10, seasons=(2024, 2025), players_per_pos: int = 8) -> pd.DataFrame:
    """A panel where a known subset of players always outscores the rest.

    The standouts are placed *last* within each position. Ties in a prediction are broken by
    row order, so a constant predictor must not stumble onto them by accident, otherwise a
    useless predictor scores full marks and the metric tests prove nothing.
    """
    rows = []
    for season in seasons:
        for gw in range(1, n_gw + 1):
            code = 0
            for pos in ("GKP", "DEF", "MID", "FWD"):
                for k in range(players_per_pos):
                    code += 1
                    good = k == players_per_pos - 1  # one standout per position, sorted last
                    rows.append(
                        {
                            "season": season,
                            "gw": gw,
                            "player_code": code,
                            "position": pos,
                            "team_slug": f"team{k % 4}",
                            "points": (8.0 if good else 2.0) + (gw % 3),
                            "minutes": 90.0,
                            "mins_l5": 90.0,
                            "played_l5": 1.0,
                            "points_l5": 8.0 if good else 2.0,
                            "goals_per90_l10": 0.5 if good else 0.05,
                            "assists_per90_l10": 0.2 if good else 0.05,
                            "dc_l5": 5.0,
                            "team_cs_rate_l10": 0.3,
                            "fpl_xp": (8.0 if good else 2.0),
                        }
                    )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- XI selection


def test_best_xi_picks_a_legal_eleven():
    pred = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1])
    actual = pred.copy()
    positions = pd.Series(["GKP", "GKP"] + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 2)
    # 14 players, so only some formations fit; the function must still return a total.
    got = _best_xi_points(pred, actual, positions)
    assert np.isfinite(got)


def test_best_xi_prefers_the_highest_predicted_players():
    positions = pd.Series(["GKP"] + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3)
    # Predictions and outcomes agree, so the chosen XI should be the best 11 available.
    pred = pd.Series([5.0] + [4.0] * 5 + [3.0] * 5 + [9.0] * 3)
    got = _best_xi_points(pred, pred, positions)
    # 1 GKP(5) + 3 FWD(27) + best remaining 7 from DEF/MID under a legal shape
    assert got == pytest.approx(5 + 27 + 4 * 4 + 3 * 3)


def test_best_xi_returns_nan_without_a_goalkeeper():
    positions = pd.Series(["DEF"] * 6 + ["MID"] * 6)
    pred = pd.Series(range(12), dtype=float)
    assert np.isnan(_best_xi_points(pred, pred, positions))


def test_best_xi_ignores_rows_with_no_prediction():
    positions = pd.Series(["GKP"] + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3)
    pred = pd.Series([5.0] + [4.0] * 5 + [3.0] * 5 + [9.0] * 3)
    with_nan = pred.copy()
    with_nan.iloc[-1] = np.nan
    assert _best_xi_points(with_nan, pred, positions) <= _best_xi_points(pred, pred, positions)


# --------------------------------------------------------------------- per-gameweek metrics


def test_a_perfect_predictor_captures_everything():
    p = _panel(n_gw=1, seasons=(2025,))
    rows = p[p.gw == 1].copy()
    res = evaluate_gameweek(rows["points"].astype(float), rows, 2025, 1)
    assert res.mae == pytest.approx(0.0)
    assert res.xi_capture == pytest.approx(1.0)


def test_a_constant_predictor_captures_little():
    p = _panel(n_gw=1, seasons=(2025,))
    rows = p[p.gw == 1].copy()
    flat = pd.Series(2.0, index=rows.index)
    res = evaluate_gameweek(flat, rows, 2025, 1)
    assert res.xi_capture < 1.0
    assert res.mae > 0


def test_captain_points_is_the_top_ranked_players_actual_score():
    rows = pd.DataFrame(
        {
            "points": [2.0, 11.0, 5.0],
            "position": ["MID", "FWD", "DEF"],
        }
    )
    pred = pd.Series([1.0, 9.0, 3.0])
    res = evaluate_gameweek(pred, rows, 2025, 1)
    assert res.captain_points == pytest.approx(11.0)


def test_oracle_is_independent_of_the_predictor():
    p = _panel(n_gw=1, seasons=(2025,))
    rows = p[p.gw == 1].copy()
    a = evaluate_gameweek(rows["points"].astype(float), rows, 2025, 1)
    b = evaluate_gameweek(pd.Series(1.0, index=rows.index), rows, 2025, 1)
    assert a.xi_oracle == pytest.approx(b.xi_oracle)


# --------------------------------------------------------------------- walk-forward hygiene


class _SpyPredictor:
    """Records the gameweeks it was shown during fit, so leakage is detectable."""

    name = "spy"

    def __init__(self):
        self.seen: list[tuple[int, int]] = []
        self.fit_calls = 0

    def fit(self, train: pd.DataFrame) -> None:
        self.fit_calls += 1
        self.seen = sorted(set(zip(train.season.astype(int), train.gw.astype(int))))

    def predict(self, rows: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=rows.index)


def test_training_data_never_includes_the_gameweek_being_scored():
    panel = _panel(n_gw=10)
    spy = _SpyPredictor()
    results = rolling_origin_backtest(
        spy, panel=panel, min_train_gws=5, refit_every=1, score_gameweeks=None
    )
    assert not results.empty
    # After the final fit, the last scored gameweek must not be in the training set.
    last = results.iloc[-1]
    assert (int(last.season), int(last.gw)) not in spy.seen


def test_training_data_only_contains_earlier_gameweeks():
    panel = _panel(n_gw=8, seasons=(2025,))

    order = []

    class Recorder(_SpyPredictor):
        def predict(self, rows):
            gw = int(rows.gw.iloc[0])
            # Everything seen in fit must precede the gameweek being predicted.
            assert all(g < gw for _, g in self.seen), f"fit saw {self.seen} before GW{gw}"
            order.append(gw)
            return pd.Series(1.0, index=rows.index)

    rolling_origin_backtest(Recorder(), panel=panel, min_train_gws=3, refit_every=1)
    assert order == sorted(order), "gameweeks were not scored in order"


def test_min_train_gws_is_respected():
    panel = _panel(n_gw=10, seasons=(2025,))
    results = rolling_origin_backtest(
        ColumnPredictor(column="points_l5", name="x"), panel=panel, min_train_gws=6
    )
    assert results.gw.min() >= 7


def test_refit_every_controls_how_often_fit_is_called():
    panel = _panel(n_gw=10, seasons=(2024, 2025))
    often, rarely = _SpyPredictor(), _SpyPredictor()
    rolling_origin_backtest(often, panel=panel, min_train_gws=5, refit_every=1)
    rolling_origin_backtest(rarely, panel=panel, min_train_gws=5, refit_every=100)
    assert often.fit_calls > rarely.fit_calls
    assert rarely.fit_calls == 1


# --------------------------------------------------------------------- xP data-quality gate


def test_gameweeks_with_no_recorded_xp_are_excluded():
    panel = _panel(n_gw=6, seasons=(2025,))
    panel.loc[panel.gw == 4, "fpl_xp"] = 0.0  # the archive gap, reproduced
    usable = usable_xp_gameweeks(panel)
    assert (2025, 4) not in set(zip(usable.season, usable.gw))
    assert (2025, 3) in set(zip(usable.season, usable.gw))


def test_scattered_zero_xp_does_not_disqualify_a_gameweek():
    """A benched keeper legitimately has xP of 0; only a mostly-zero gameweek is a gap."""
    panel = _panel(n_gw=4, seasons=(2025,))
    gw2 = panel.index[panel.gw == 2]
    panel.loc[gw2[: len(gw2) // 4], "fpl_xp"] = 0.0
    usable = usable_xp_gameweeks(panel)
    assert (2025, 2) in set(zip(usable.season, usable.gw))


def test_the_gate_applies_to_every_predictor_equally():
    panel = _panel(n_gw=8, seasons=(2024, 2025))
    panel.loc[(panel.season == 2025) & (panel.gw == 5), "fpl_xp"] = 0.0
    usable = usable_xp_gameweeks(panel)

    a = rolling_origin_backtest(
        ColumnPredictor(column="fpl_xp", name="xp"),
        panel=panel,
        min_train_gws=8,
        score_gameweeks=usable,
    )
    b = rolling_origin_backtest(
        PositionMeanPredictor(), panel=panel, min_train_gws=8, score_gameweeks=usable
    )
    assert list(zip(a.season, a.gw)) == list(zip(b.season, b.gw))


# --------------------------------------------------------------------- reimplemented pipeline


def test_current_architecture_omits_what_the_shipped_model_omits():
    """It must stay a faithful stand-in: no saves, conceded, bonus or cards."""
    rows = pd.DataFrame(
        {
            "position": ["GKP", "DEF"],
            "mins_l5": [90.0, 90.0],
            "goals_per90_l10": [0.0, 0.0],
            "assists_per90_l10": [0.0, 0.0],
            "dc_l5": [0.0, 0.0],
            "team_cs_rate_l10": [0.0, 0.0],
        }
    )
    model = CurrentArchitecturePredictor()
    model.fit(pd.DataFrame({"team_cs_rate_l10": [0.3], "points": [2.0]}))
    pred = model.predict(rows)
    # With no attacking output, no clean sheet and no defensive contribution, all that is
    # left is appearance points. A keeper's saves would otherwise show up here.
    assert pred.tolist() == pytest.approx([2.0, 2.0])


def test_current_architecture_credits_a_clean_sheet_by_position():
    rows = pd.DataFrame(
        {
            "position": ["DEF", "MID", "FWD"],
            "mins_l5": [90.0] * 3,
            "goals_per90_l10": [0.0] * 3,
            "assists_per90_l10": [0.0] * 3,
            "dc_l5": [0.0] * 3,
            "team_cs_rate_l10": [1.0] * 3,
        }
    )
    model = CurrentArchitecturePredictor()
    model.fit(pd.DataFrame({"team_cs_rate_l10": [0.3], "points": [2.0]}))
    pred = model.predict(rows)
    assert pred.iloc[0] == pytest.approx(2.0 + 4.0)
    assert pred.iloc[1] == pytest.approx(2.0 + 1.0)
    assert pred.iloc[2] == pytest.approx(2.0)


# --------------------------------------------------------------------- summary


def test_summary_ranks_by_xi_points():
    panel = _panel(n_gw=8, seasons=(2024, 2025))
    good = rolling_origin_backtest(
        ColumnPredictor(column="points_l5", name="good"), panel=panel, min_train_gws=8
    )
    bad = rolling_origin_backtest(PositionMeanPredictor(), panel=panel, min_train_gws=8)
    s = summarise({"good": good, "position mean": bad})
    assert s.iloc[0]["predictor"] == "good"


def test_summary_of_nothing_is_empty():
    assert summarise({"a": pd.DataFrame()}).empty


# --------------------------------------------------------------------- ablation


def test_ablation_reports_one_row_per_experiment_plus_the_control():
    from src.fpl_predict.models.backtest import ablation_report

    panel = _panel(n_gw=12, seasons=(2024, 2025), players_per_pos=10)
    # Give the model the columns its sub-models need.
    for col in (
        "minutes",
        "goals",
        "assists",
        "clean_sheet",
        "goals_conceded",
        "saves",
        "bonus",
        "bps",
        "yellow_cards",
        "red_cards",
        "defensive_contribution",
        "n_fixtures",
        "value",
        "selected",
        "is_home",
        "opp_gf_l10",
        "opp_ga_l10",
        "opp_cs_rate_l10",
        "team_gf_l10",
        "team_ga_l10",
    ):
        if col not in panel.columns:
            panel[col] = 1.0

    report = ablation_report(
        groups=["opponent", "venue"], panel=panel, min_train_gws=10, refit_every=100
    )
    assert set(report.experiment) == {"full model", "drop opponent", "drop venue"}


def test_ablation_report_is_empty_without_a_panel():
    from src.fpl_predict.models.backtest import ablation_report

    assert ablation_report(panel=pd.DataFrame()).empty


def test_format_ablation_handles_nothing():
    from src.fpl_predict.models.backtest import format_ablation

    assert "No ablation" in format_ablation(pd.DataFrame())


def test_format_ablation_renders_the_deltas():
    from src.fpl_predict.models.backtest import format_ablation

    report = pd.DataFrame(
        [
            {
                "experiment": "full model",
                "mae": 1.0,
                "mae_worse_by": 0.0,
                "t": 0.0,
                "spearman": 0.7,
                "spearman_drop": 0.0,
                "xi_points": 55.0,
                "xi_drop": 0.0,
            },
            {
                "experiment": "drop opponent",
                "mae": 1.01,
                "mae_worse_by": 0.01,
                "t": 5.8,
                "spearman": 0.69,
                "spearman_drop": 0.01,
                "xi_points": 54.0,
                "xi_drop": 1.0,
            },
        ]
    )
    md = format_ablation(report)
    assert "drop opponent" in md
    assert "MAE worse by" in md
