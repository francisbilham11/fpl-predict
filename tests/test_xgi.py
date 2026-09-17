"""Goals/assists prediction.

The regression this guards against, and it is the big one: `predict_with_uncertainty` scaled
its feature perturbation by `X_noisy[col].std()` computed on a *single-row* DataFrame, which
is NaN. Every feature became NaN, XGBoost routed the all-missing row down its default branch,
and every player in a position came back with an identical prediction with std exactly 0.

The stored artefact confirmed it: `prediction_uncertainty.parquet` had `goals_std == 0.0` for
all 822 players and exactly one distinct `goals_lower` per position.
"""

import numpy as np
import pandas as pd
import pytest

from src.fpl_predict.models.xgi import XGIModel

RNG = np.random.default_rng(0)
N = 240


@pytest.fixture(scope="module")
def cohort():
    """A cohort where goals genuinely depend on the features."""
    positions = np.array((["FWD"] * 60) + (["MID"] * 60) + (["DEF"] * 60) + (["GKP"] * 60))
    xg = RNG.gamma(2.0, 0.15, N)
    xa = RNG.gamma(2.0, 0.10, N)
    mins = RNG.uniform(0, 90, N)
    X = pd.DataFrame(
        {
            "prev_xg_per90": xg,
            "prev_xa_per90": xa,
            "prev_goals_per90": xg * 0.9,
            "prev_assists_per90": xa * 0.9,
            "mins_l5": mins,
            "mins_l3": mins * RNG.uniform(0.8, 1.2, N),
            "now_cost": RNG.uniform(40, 150, N),
            "form": RNG.uniform(0, 8, N),
            "selected_by_percent": RNG.uniform(0, 60, N),
            "chance_next": 100.0,
            "saves_l5": RNG.uniform(0, 4, N),
        }
    )
    pos_multiplier = pd.Series(positions).map({"FWD": 1.0, "MID": 0.7, "DEF": 0.2, "GKP": 0.0})
    y_g = pd.Series(xg * pos_multiplier.to_numpy() * (mins / 90.0))
    y_a = pd.Series(xa * (mins / 90.0))
    return X, y_g, y_a, pd.Series(positions)


@pytest.fixture(scope="module")
def fitted(cohort):
    X, y_g, y_a, positions = cohort
    model = XGIModel(n_trials=0, cv_splits=2)
    model.fit(X, y_g, y_a, positions)
    return model


def test_model_fits_a_model_per_position(fitted):
    assert set(fitted.models_g) == {"FWD", "MID", "DEF", "GKP"}


def test_predictions_differ_between_players(fitted, cohort):
    X, _, _, positions = cohort
    out = fitted.predict_with_uncertainty(X, positions, n_iterations=5)
    goals = out["goals"]["mean"]
    # The whole failure mode was one distinct value per position, so four overall.
    assert goals.nunique() > 20, (
        f"only {goals.nunique()} distinct predictions across {len(X)} players"
    )


def test_predictions_differ_within_a_single_position(fitted, cohort):
    X, _, _, positions = cohort
    out = fitted.predict_with_uncertainty(X, positions, n_iterations=5)
    for pos in ("FWD", "MID"):
        within = out["goals"]["mean"][(positions == pos).to_numpy()]
        assert within.nunique() > 5, f"{pos} predictions are near-constant"


def test_predictions_are_not_nan(fitted, cohort):
    X, _, _, positions = cohort
    out = fitted.predict_with_uncertainty(X, positions, n_iterations=5)
    assert out["goals"]["mean"].notna().all()
    assert out["assists"]["mean"].notna().all()


def test_predictions_are_non_negative(fitted, cohort):
    X, _, _, positions = cohort
    out = fitted.predict_with_uncertainty(X, positions, n_iterations=5)
    assert (out["goals"]["mean"] >= 0).all()
    assert (out["assists"]["mean"] >= 0).all()


def test_predictions_track_the_underlying_signal(fitted, cohort):
    X, y_g, _, positions = cohort
    out = fitted.predict_with_uncertainty(X, positions, n_iterations=3)
    corr = np.corrcoef(out["goals"]["mean"].to_numpy(), y_g.to_numpy())[0, 1]
    assert corr > 0.5, f"predictions barely correlate with the target (r={corr:.2f})"


def test_forwards_outscore_defenders(fitted, cohort):
    X, _, _, positions = cohort
    out = fitted.predict_with_uncertainty(X, positions, n_iterations=3)
    means = out["goals"]["mean"]
    assert (
        means[(positions == "FWD").to_numpy()].mean()
        > means[(positions == "DEF").to_numpy()].mean()
    )


def test_uncertainty_is_a_real_spread(fitted, cohort):
    X, _, _, positions = cohort
    out = fitted.predict_with_uncertainty(X, positions, n_iterations=20)
    std = out["goals"]["std"]
    assert (std > 0).mean() > 0.5, "std was exactly 0 for every player under the NaN bug"
    assert (out["goals"]["upper"] >= out["goals"]["lower"]).all()


def test_first_iteration_is_the_unperturbed_prediction(fitted, cohort):
    X, _, _, positions = cohort
    one = fitted.predict_with_uncertainty(X, positions, n_iterations=1)["goals"]["mean"]
    many = fitted.predict_with_uncertainty(X, positions, n_iterations=30)["goals"]["mean"]
    # Perturbation is symmetric, so the mean should stay close to the point prediction.
    assert np.corrcoef(one.to_numpy(), many.to_numpy())[0, 1] > 0.95


def test_predict_is_consistent_with_predict_with_uncertainty(fitted, cohort):
    X, _, _, positions = cohort
    g, a = fitted.predict(X, positions)
    assert len(g) == len(X) and len(a) == len(X)
    assert g.notna().all() and a.notna().all()


def test_unfitted_model_returns_flat_priors(cohort):
    X, _, _, positions = cohort
    out = XGIModel().predict_with_uncertainty(X, positions, n_iterations=3)
    assert out["goals"]["mean"].nunique() == 1


def test_positions_with_a_different_index_still_align(fitted, cohort):
    X, _, _, positions = cohort
    shifted = positions.copy()
    shifted.index = range(1000, 1000 + len(shifted))
    out = fitted.predict_with_uncertainty(X, shifted, n_iterations=3)
    assert out["goals"]["mean"].nunique() > 20
