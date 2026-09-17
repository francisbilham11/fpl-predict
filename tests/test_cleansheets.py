"""Clean sheet probability.

The regression this guards against: `_calculate_team_xg_stats` keyed its output on team
*names* from fixtures.parquet, while `predict_player_cs_probability` looked up
`features['team']`, which is the FPL numeric team id. The lookup never hit, so every
goalkeeper and defender in the league received the same flat 0.35.
"""

import pandas as pd
import pytest

from src.fpl_predict.models import cleansheets as cs_module
from src.fpl_predict.models.cleansheets import CleanSheetModel

BOOTSTRAP = {
    "teams": [
        {"id": 1, "name": "Arsenal", "short_name": "ARS"},
        {"id": 13, "name": "Man City", "short_name": "MCI"},
        {"id": 3, "name": "Burnley", "short_name": "BUR"},
        {"id": 7, "name": "Coventry City", "short_name": "COV"},
    ]
}


def _fixtures() -> pd.DataFrame:
    """Arsenal keep clean sheets, Burnley concede. Coventry have no matches at all."""
    rows = []
    date = pd.Timestamp("2025-08-15", tz="UTC")
    for i in range(10):
        date += pd.Timedelta(days=7)
        # Arsenal at home, always a clean sheet
        rows.append(("Arsenal", "Man City", 2, 0, date))
        # Burnley at home, always concede
        rows.append(("Burnley", "Arsenal", 0, 3, date))
        # Man City away at Burnley, mixed
        rows.append(("Man City", "Burnley", 1, 1, date))
    return pd.DataFrame(
        rows, columns=["home_team", "away_team", "home_goals", "away_goals", "date"]
    )


@pytest.fixture
def fitted(monkeypatch):
    monkeypatch.setattr(cs_module, "get_bootstrap", lambda: BOOTSTRAP)
    model = CleanSheetModel()
    model.fit(_fixtures())
    return model


def test_stats_are_keyed_on_canonical_slugs(fitted):
    assert "man_city" in fitted.team_defensive_stats
    assert "arsenal" in fitted.team_defensive_stats
    # Not on display names, which is what broke the lookup.
    assert "Man City" not in fitted.team_defensive_stats


def test_probability_varies_by_team(fitted):
    features = pd.DataFrame(
        {"player_id": [1, 2, 3], "team": [1, 3, 13], "position": ["DEF", "DEF", "DEF"]}
    )
    probs = fitted.predict_player_cs_probability(features)
    assert probs.nunique() > 1, "every defender used to get the same flat probability"


def test_a_good_defence_beats_a_bad_one(fitted):
    features = pd.DataFrame({"player_id": [1, 2], "team": [1, 3], "position": ["DEF", "DEF"]})
    probs = fitted.predict_player_cs_probability(features)
    assert probs.iloc[0] > probs.iloc[1], "Arsenal should out-rate Burnley for clean sheets"


def test_position_scaling(fitted):
    features = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "team": [1, 1, 1, 1],
            "position": ["GKP", "DEF", "MID", "FWD"],
        }
    )
    probs = fitted.predict_player_cs_probability(features)
    gkp, dfn, mid, fwd = probs.tolist()
    assert gkp == dfn, "keepers and defenders both score 4 for a clean sheet"
    assert 0 < mid < dfn, "midfielders score 1, so they carry a fraction of the value"
    assert fwd == 0.0, "forwards get nothing for a clean sheet"


def test_teams_with_no_stored_matches_get_the_league_average(fitted):
    features = pd.DataFrame({"player_id": [1, 2], "team": [7, 1], "position": ["DEF", "DEF"]})
    probs = fitted.predict_player_cs_probability(features)
    assert probs.notna().all()
    assert 0.0 < probs.iloc[0] < 1.0


def test_probabilities_stay_in_range(fitted):
    features = pd.DataFrame(
        {
            "player_id": range(8),
            "team": [1, 3, 13, 7, 1, 3, 13, 7],
            "position": ["GKP", "DEF", "MID", "FWD", "DEF", "GKP", "DEF", "MID"],
        }
    )
    probs = fitted.predict_player_cs_probability(features)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_unfitted_model_returns_position_scaled_defaults():
    features = pd.DataFrame(
        {"player_id": [1, 2, 3], "team": [1, 3, 13], "position": ["DEF", "MID", "FWD"]}
    )
    probs = CleanSheetModel().predict_player_cs_probability(features)
    assert probs.iloc[2] == 0.0
    assert probs.iloc[0] > probs.iloc[1] > 0


def test_prediction_survives_a_bootstrap_failure(fitted, monkeypatch):
    def boom():
        raise RuntimeError("API down")

    monkeypatch.setattr(cs_module, "get_bootstrap", boom)
    features = pd.DataFrame({"player_id": [1], "team": [1], "position": ["DEF"]})
    probs = fitted.predict_player_cs_probability(features)
    assert probs.notna().all()
    assert 0.0 < probs.iloc[0] < 1.0


def test_slug_columns_are_used_when_already_present(monkeypatch):
    monkeypatch.setattr(cs_module, "get_bootstrap", lambda: BOOTSTRAP)
    fx = _fixtures()
    fx["home_slug"] = fx["home_team"].str.lower().str.replace(" ", "_")
    fx["away_slug"] = fx["away_team"].str.lower().str.replace(" ", "_")
    model = CleanSheetModel()
    model.fit(fx)
    assert "man_city" in model.team_defensive_stats
