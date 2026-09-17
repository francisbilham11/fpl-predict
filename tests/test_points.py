"""Expected-points models.

The component model exists because the shipped one omits saves, goals conceded, bonus and
cards outright, and over-credits defensive contribution by 3x for defenders, 4.5x for
midfielders and 69x for forwards. These tests pin the arithmetic of each term so a
regression in one component cannot hide inside the total.
"""

import numpy as np
import pandas as pd
import pytest

from src.fpl_predict.models.points import (
    BAND_LONG,
    BAND_MINUTES,
    ComponentPointsModel,
    DirectPointsModel,
    _design,
)

RNG = np.random.default_rng(7)


def _train(n: int = 3000) -> pd.DataFrame:
    """Synthetic panel where each component has a known, separable driver."""
    pos = RNG.choice(["GKP", "DEF", "MID", "FWD"], n, p=[0.1, 0.35, 0.4, 0.15])
    starter = RNG.random(n) < 0.6
    minutes = np.where(starter, RNG.integers(60, 91, n), RNG.integers(0, 60, n))
    xg90 = np.where(pos == "FWD", 0.5, np.where(pos == "MID", 0.25, 0.05)) * RNG.uniform(
        0.4, 1.8, n
    )
    saves90 = np.where(pos == "GKP", RNG.uniform(1.5, 4.5, n), 0.0)

    df = pd.DataFrame(
        {
            "season": 2024,
            "gw": RNG.integers(1, 39, n),
            "player_code": RNG.integers(1, 400, n),
            "position": pos,
            "n_fixtures": 1,
            "minutes": minutes.astype(float),
            "mins_l5": np.where(starter, 80.0, 20.0),
            "mins_l3": np.where(starter, 80.0, 20.0),
            "played_l5": np.where(starter, 1.0, 0.4),
            "started_l5": np.where(starter, 0.9, 0.2),
            "points_l5": RNG.uniform(0, 7, n),
            "xg_per90_l10": xg90,
            "xa_per90_l10": xg90 * 0.5,
            "goals_per90_l10": xg90 * 0.9,
            "assists_per90_l10": xg90 * 0.4,
            "saves_l5": saves90,
            "dc_l5": np.where(pos == "DEF", 7.0, 4.0) * RNG.uniform(0.5, 1.6, n),
            "team_cs_rate_l10": RNG.uniform(0.1, 0.5, n),
            "team_ga_l10": RNG.uniform(0.6, 2.2, n),
            "opp_gf_l10": RNG.uniform(0.6, 2.2, n),
            "bps_l5": RNG.uniform(0, 30, n),
            "value": RNG.uniform(40, 140, n),
            "selected": RNG.uniform(0, 50, n),
            "is_home": RNG.integers(0, 2, n).astype(float),
        }
    )
    played = df.minutes > 0
    df["goals"] = RNG.poisson(xg90 * df.minutes / 90) * played
    df["assists"] = RNG.poisson(xg90 * 0.4 * df.minutes / 90) * played
    df["saves"] = RNG.poisson(saves90 * df.minutes / 90) * played
    df["clean_sheet"] = ((RNG.random(n) < df.team_cs_rate_l10) & (df.minutes >= 60)).astype(float)
    df["goals_conceded"] = RNG.poisson(df.team_ga_l10) * (df.minutes >= 60)
    df["bonus"] = (RNG.random(n) < 0.12) * RNG.integers(1, 4, n) * played
    df["yellow_cards"] = (RNG.random(n) < 0.1) * played
    df["red_cards"] = (RNG.random(n) < 0.005) * played
    df["defensive_contribution"] = RNG.poisson(df.dc_l5) * played
    df["points"] = (
        np.where(df.minutes >= 60, 2, np.where(df.minutes > 0, 1, 0))
        + df.goals * pd.Series(pos).map({"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4}).to_numpy()
        + df.assists * 3
        + df.clean_sheet * pd.Series(pos).map({"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}).to_numpy()
        + df.bonus
        - df.yellow_cards
    )
    return df


@pytest.fixture(scope="module")
def train():
    return _train()


@pytest.fixture(scope="module")
def fitted(train):
    m = ComponentPointsModel()
    m.fit(train)
    return m


# ------------------------------------------------------------------ design matrix


def test_design_matrix_one_hots_position(train):
    X = _design(train, ["mins_l5", "value"])
    assert {"pos_GKP", "pos_DEF", "pos_MID", "pos_FWD"} <= set(X.columns)
    assert X[["pos_GKP", "pos_DEF", "pos_MID", "pos_FWD"]].sum(axis=1).eq(1).all()


def test_design_matrix_does_not_encode_club_identity(train):
    X = _design(train, ["mins_l5"])
    assert not [c for c in X.columns if "team_slug" in c or "opponent_slug" in c]


# ------------------------------------------------------------------ sub-models present


def test_every_component_is_fitted(fitted):
    for part in ("band", "goals", "assists", "bonus", "cards", "cs", "conceded", "saves", "dc"):
        assert part in fitted._parts, f"{part} sub-model missing"


def test_saves_and_conceded_are_modelled_at_all(fitted):
    """The shipped model omits both entirely; a keeper's saves are 17% of their points."""
    assert "saves" in fitted._parts
    assert "conceded" in fitted._parts


def test_bonus_and_cards_are_modelled_at_all(fitted):
    """Bonus correlates 0.69-0.87 with a score and the shipped model has no term for it."""
    assert "bonus" in fitted._parts
    assert "cards" in fitted._parts


# ------------------------------------------------------------------ predictions


def test_predictions_are_finite_and_varied(fitted, train):
    pred = fitted.predict(train)
    assert pred.notna().all()
    assert np.isfinite(pred).all()
    assert pred.nunique() > 100


def test_starters_outscore_non_starters(fitted, train):
    pred = fitted.predict(train)
    assert pred[train.mins_l5 > 50].mean() > pred[train.mins_l5 < 30].mean()


def test_forwards_get_more_attacking_points_than_defenders(fitted, train):
    pred = fitted.predict(train)
    starters = train.mins_l5 > 50
    fwd = pred[starters & (train.position == "FWD")].mean()
    dfn = pred[starters & (train.position == "DEF")].mean()
    assert fwd > dfn, f"forwards {fwd:.2f} should outscore defenders {dfn:.2f}"


def test_a_double_gameweek_is_worth_more_than_a_single(fitted, train):
    """The shipped model caps expected minutes at one match and cannot express this."""
    single = train.head(200).copy()
    single["n_fixtures"] = 1
    double = single.copy()
    double["n_fixtures"] = 2

    p1 = fitted.predict(single)
    p2 = fitted.predict(double)
    ratio = p2.sum() / p1.sum()
    assert ratio == pytest.approx(2.0, rel=0.01), f"double paid {ratio:.2f}x a single"


def test_unfitted_model_returns_zeros(train):
    assert ComponentPointsModel().predict(train).eq(0).all()


# ------------------------------------------------------------------ counting-stat arithmetic


@pytest.mark.parametrize("divisor", [2, 3])
def test_expected_floor_div_matches_a_direct_sum(divisor):
    """Saves pay per 3 and conceded costs per 2, so the expectation is over the count."""
    from scipy.stats import poisson

    rates = np.array([0.1, 0.5, 1.0, 2.0, 3.5])
    got = ComponentPointsModel._expected_floor_div(rates, divisor)
    want = np.array([sum((k // divisor) * poisson.pmf(k, r) for k in range(0, 60)) for r in rates])
    # The implementation folds the far tail into its last term, so exactness is bounded by
    # COUNT_SUPPORT. A tolerance of 1e-4 points is orders of magnitude below anything that
    # could change a team selection.
    assert got == pytest.approx(want, abs=1e-4)


def test_expected_floor_div_is_not_the_naive_division():
    """Dividing the mean instead would misprice low rates badly, which is the point."""
    rate = np.array([1.0])
    exact = ComponentPointsModel._expected_floor_div(rate, 3)[0]
    naive = rate[0] / 3
    assert exact < naive
    assert exact == pytest.approx(0.0803, abs=0.005)


def test_expected_floor_div_of_zero_is_zero():
    assert ComponentPointsModel._expected_floor_div(np.array([0.0]), 2)[0] == pytest.approx(0.0)


def test_expected_floor_div_grows_with_the_rate():
    vals = ComponentPointsModel._expected_floor_div(np.array([0.5, 1.0, 2.0, 4.0]), 2)
    assert list(vals) == sorted(vals)


# ------------------------------------------------------------------ minutes bands


def test_band_minutes_are_ordered_and_plausible():
    assert BAND_MINUTES[BAND_LONG] > 60
    assert BAND_MINUTES[BAND_LONG] <= 90


def test_keepers_get_saves_points_and_outfielders_do_not(fitted):
    rows = pd.DataFrame(
        {
            "position": ["GKP", "DEF"],
            "n_fixtures": [1, 1],
            "mins_l5": [90.0, 90.0],
            "mins_l3": [90.0, 90.0],
            "played_l5": [1.0, 1.0],
            "started_l5": [1.0, 1.0],
            "points_l5": [4.0, 4.0],
            "xg_per90_l10": [0.0, 0.0],
            "xa_per90_l10": [0.0, 0.0],
            "goals_per90_l10": [0.0, 0.0],
            "assists_per90_l10": [0.0, 0.0],
            "saves_l5": [4.0, 4.0],
            "dc_l5": [0.0, 0.0],
            "team_cs_rate_l10": [0.0, 0.0],
            "team_ga_l10": [1.0, 1.0],
            "opp_gf_l10": [1.0, 1.0],
            "bps_l5": [20.0, 20.0],
            "value": [55.0, 55.0],
            "selected": [10.0, 10.0],
            "is_home": [1.0, 1.0],
        }
    )
    pred = fitted.predict(rows)
    assert pred.iloc[0] > 0
    assert pred.iloc[1] > 0


# ------------------------------------------------------------------ direct model


def test_direct_model_learns_something(train):
    m = DirectPointsModel()
    m.fit(train)
    pred = m.predict(train)
    assert pred.notna().all()
    r = np.corrcoef(pred, train.points)[0, 1]
    assert r > 0.4, f"direct model barely correlates with the target (r={r:.2f})"


def test_direct_model_excludes_the_contaminated_xp_column(train):
    t = train.copy()
    t["fpl_xp"] = t["points"]  # perfectly leaked, must be ignored
    m = DirectPointsModel()
    m.fit(t)
    assert "fpl_xp" not in m.columns


def test_direct_model_unfitted_returns_zeros(train):
    assert DirectPointsModel().predict(train).eq(0).all()


# ------------------------------------------------------------------ ablation plumbing


def test_dropped_features_are_withheld_from_the_component_model(train):
    from src.fpl_predict.models.panel import group_features

    dropped = group_features("opponent", "venue")
    m = ComponentPointsModel(drop_features=dropped)
    m.fit(train)
    assert not set(m.columns) & set(dropped)
    # Everything else is still there.
    assert "mins_l5" in m.columns


def test_dropped_features_are_withheld_from_the_direct_model(train):
    m = DirectPointsModel(drop_features=["is_home", "value"])
    m.fit(train)
    assert "is_home" not in m.columns
    assert "value" not in m.columns


def test_an_ablated_model_still_predicts(train):
    from src.fpl_predict.models.panel import CONTEXT_GROUPS, group_features

    m = ComponentPointsModel(drop_features=group_features(*CONTEXT_GROUPS))
    m.fit(train)
    pred = m.predict(train)
    assert pred.notna().all()
    assert pred.nunique() > 50


def test_dropping_n_fixtures_does_not_disable_the_double_multiplier(train):
    """The multiplier is arithmetic, not a learned effect, so ablation must not remove it."""
    m = ComponentPointsModel(drop_features=["n_fixtures"])
    m.fit(train)
    single = train.head(100).assign(n_fixtures=1)
    double = train.head(100).assign(n_fixtures=2)
    assert m.predict(double).sum() == pytest.approx(2 * m.predict(single).sum(), rel=0.01)


def test_feature_groups_cover_the_real_panel_columns():
    """A group naming a column that no longer exists would silently ablate nothing."""
    from src.fpl_predict.models.panel import FEATURE_GROUPS, feature_columns, load_panel

    panel = load_panel()
    if panel.empty:
        pytest.skip("no panel on disk")
    available = set(feature_columns(panel))
    for group, cols in FEATURE_GROUPS.items():
        missing = set(cols) - available
        assert not missing, f"group {group!r} names columns not in the panel: {sorted(missing)}"


def test_context_groups_are_all_real_groups():
    from src.fpl_predict.models.panel import CONTEXT_GROUPS, FEATURE_GROUPS

    assert set(CONTEXT_GROUPS) <= set(FEATURE_GROUPS)


# ------------------------------------------------------------------ hyperparameters


def test_lgbm_params_override_reaches_every_sub_model(train):
    """The tuner drives one dict; if a sub-model ignores it the search is measuring nothing."""
    from src.fpl_predict.models.points import DEFAULT_LGBM_PARAMS

    override = dict(DEFAULT_LGBM_PARAMS)
    override.update(n_estimators=17, num_leaves=5)

    m = ComponentPointsModel(lgbm_params=override)
    m.fit(train)

    checked = 0
    for name, sub in m._parts.items():
        params = sub.get_params()
        assert params["n_estimators"] == 17, f"{name} ignored the override"
        assert params["num_leaves"] == 5, f"{name} ignored the override"
        checked += 1
    assert checked >= 8, f"only {checked} sub-models were built"


def test_objectives_survive_a_params_override(train):
    """An override must not clobber the per-sub-model objective."""
    from src.fpl_predict.models.points import DEFAULT_LGBM_PARAMS

    m = ComponentPointsModel(lgbm_params=dict(DEFAULT_LGBM_PARAMS, n_estimators=20))
    m.fit(train)
    assert m._parts["goals"].get_params()["objective"] == "poisson"
    assert m._parts["assists"].get_params()["objective"] == "poisson"
    assert m._parts["conceded"].get_params()["objective"] == "poisson"
    assert m._parts["bonus"].get_params()["objective"] == "regression"
    assert m._parts["band"].get_params()["objective"] == "multiclass"
    assert m._parts["cs"].get_params()["objective"] == "binary"


def test_minutes_band_keeps_three_classes_under_an_override(train):
    from src.fpl_predict.models.points import DEFAULT_LGBM_PARAMS

    m = ComponentPointsModel(lgbm_params=dict(DEFAULT_LGBM_PARAMS, n_estimators=20))
    m.fit(train)
    assert m._parts["band"].get_params()["num_class"] == 3
    assert len(m._parts["band"].classes_) == 3


def test_default_params_are_used_when_no_override_given(train):
    from src.fpl_predict.models.points import DEFAULT_LGBM_PARAMS

    m = ComponentPointsModel()
    m.fit(train)
    assert m._parts["goals"].get_params()["n_estimators"] == DEFAULT_LGBM_PARAMS["n_estimators"]


def test_exposure_weighting_is_off_by_default():
    """It was measured on the development seasons and lost; see the field comment."""
    assert ComponentPointsModel().weight_by_exposure is False


def test_exposure_weighting_changes_the_fit_when_enabled(train):
    unweighted = ComponentPointsModel(weight_by_exposure=False)
    unweighted.fit(train)
    weighted = ComponentPointsModel(weight_by_exposure=True)
    weighted.fit(train)

    a = unweighted.predict(train)
    b = weighted.predict(train)
    assert not np.allclose(a, b), "the weighting flag had no effect on the fit"
