"""Feature panel, with the leakage guarantee tested rather than asserted in a docstring.

The regression this guards against is the reason the panel exists. The pipeline it replaces
trained one row per player on a target that was the mean of that player's last three
gameweeks, while feeding it `goals_l3`/`goals_l5`/`xg_l3` computed over the same window. It
was fitting a smoothed copy of its own input, so cross-validated error looked fine and
out-of-sample skill was zero.
"""

import numpy as np
import pandas as pd
import pytest

from src.fpl_predict.models import panel as panel_mod
from src.fpl_predict.models.panel import (
    OUTCOME_COLS,
    build_panel,
    feature_columns,
)


def pts(player: int, gw: int) -> float:
    """Deterministic synthetic score.

    Deliberately not monotone in gameweek: a linear score would correlate perfectly with
    `gws_played_this_season`, which is a legitimate pre-deadline feature, and the
    perfect-correlation check below would flag it as leakage.
    """
    return float((gw * 7 + player * 3) % 13)


def _history(n_players: int = 4, seasons=(2024, 2025), gws: int = 8) -> pd.DataFrame:
    """A small synthetic archive where each player's output is a known function of gameweek."""
    rows = []
    for season in seasons:
        for p in range(n_players):
            for gw in range(1, gws + 1):
                rows.append(
                    {
                        "season": season,
                        "gw": gw,
                        "player_code": 1000 + p,
                        "fpl_id": p + 1,
                        "player_name": f"Player {p}",
                        "position": ["GKP", "DEF", "MID", "FWD"][p % 4],
                        "team_slug": "arsenal" if p % 2 else "chelsea",
                        "opponent_slug": "chelsea" if p % 2 else "arsenal",
                        "was_home": bool(gw % 2),
                        "value": 50 + p,
                        "selected": 1000 * (p + 1),
                        "kickoff_time": pd.Timestamp(f"{season}-08-15", tz="UTC")
                        + pd.Timedelta(days=7 * gw),
                        "fixture_id": gw * 10 + p,
                        "minutes": 90,
                        # Outcome is a known function of (player, gameweek), so a leaked
                        # feature shows up immediately.
                        "total_points": pts(p, gw),
                        "goals": pts(p, gw) / 5,
                        "assists": 0.0,
                        "clean_sheets": 0.0,
                        "goals_conceded": 0.0,
                        "saves": 0.0,
                        "bonus": 0.0,
                        "bps": 0.0,
                        "yellow_cards": 0.0,
                        "red_cards": 0.0,
                        "own_goals": 0.0,
                        "penalties_saved": 0.0,
                        "penalties_missed": 0.0,
                        "xg": pts(p, gw) / 50,
                        "xa": 0.0,
                        "xgc": 0.0,
                        "defensive_contribution": 0.0,
                        "cbi": 0.0,
                        "tackles": 0.0,
                        "recoveries": 0.0,
                        "starts": 1.0,
                        "fpl_xp": pts(p, gw),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic(monkeypatch):
    monkeypatch.setattr(panel_mod, "load_history", lambda seasons=None: _history())
    monkeypatch.setattr(panel_mod, "load_history_fixtures", lambda seasons=None: pd.DataFrame())
    return build_panel(save=False)


def test_one_row_per_player_gameweek(synthetic):
    assert not synthetic.duplicated(subset=["season", "player_code", "gw"]).any()


def test_lagged_points_exclude_the_current_gameweek(synthetic):
    """points_l3 at gameweek g must be the mean of g-3..g-1, never touching g."""
    p = synthetic[(synthetic.player_code == 1000) & (synthetic.season == 2025)].sort_values("gw")
    row = p[p.gw == 6].iloc[0]
    assert row.points == pts(0, 6)
    assert row.points_l3 == pytest.approx(np.mean([pts(0, g) for g in (3, 4, 5)]))
    assert row.points_l5 == pytest.approx(np.mean([pts(0, g) for g in (1, 2, 3, 4, 5)]))


def test_first_gameweek_of_the_first_season_has_no_history(synthetic):
    first = synthetic[(synthetic.season == 2024) & (synthetic.gw == 1)]
    for col in ("points_l3", "points_l5", "mins_l5", "goals_l5"):
        assert first[col].isna().all(), f"{col} invented history for the first gameweek"


def test_lagged_features_carry_across_a_season_boundary(synthetic):
    """A player's form in GW1 of a new season should reflect the end of the last one."""
    row = synthetic[
        (synthetic.player_code == 1000) & (synthetic.season == 2025) & (synthetic.gw == 1)
    ].iloc[0]
    # The last three gameweeks of 2024 are what should be in the window.
    assert row.points_l3 == pytest.approx(np.mean([pts(0, g) for g in (6, 7, 8)]))


def test_previous_season_totals_come_from_the_previous_season(synthetic):
    row = synthetic[
        (synthetic.player_code == 1000) & (synthetic.season == 2025) & (synthetic.gw == 3)
    ].iloc[0]
    assert row.prev_points == pytest.approx(sum(pts(0, g) for g in range(1, 9)))
    assert row.prev_minutes == pytest.approx(8 * 90)


def test_previous_season_totals_absent_for_the_earliest_season(synthetic):
    first = synthetic[synthetic.season == 2024]
    assert first["prev_points"].isna().all()


def test_season_to_date_excludes_the_current_gameweek(synthetic):
    p = synthetic[(synthetic.player_code == 1000) & (synthetic.season == 2025)]
    row = p[p.gw == 5].iloc[0]
    assert row.season_points_to_date == pytest.approx(sum(pts(0, g) for g in (1, 2, 3, 4)))
    assert row.gws_played_this_season == 4


def test_no_outcome_column_is_offered_as_a_feature(synthetic):
    features = set(feature_columns(synthetic))
    leaked = features & set(OUTCOME_COLS)
    assert not leaked, f"outcome columns exposed as features: {sorted(leaked)}"


def test_fpl_xp_is_not_a_feature(synthetic):
    """FPL's own expected points is the baseline to beat, not an input to beat it with."""
    assert "fpl_xp" not in feature_columns(synthetic)


def test_no_feature_perfectly_predicts_the_outcome(synthetic):
    """A leaked feature shows up as a correlation of 1 with the target it should not see."""
    features = feature_columns(synthetic)
    sub = synthetic.dropna(subset=["points"])
    for col in features:
        vals = sub[col]
        if vals.notna().sum() < 20 or vals.nunique() < 2:
            continue
        pair = pd.DataFrame({"x": vals, "y": sub["points"]}).dropna()
        if len(pair) < 20 or pair.x.nunique() < 2:
            continue
        r = abs(np.corrcoef(pair.x, pair.y)[0, 1])
        assert r < 0.999, f"{col} correlates {r:.4f} with points, which implies leakage"


def test_doubles_are_collapsed_and_summed(monkeypatch):
    h = _history(n_players=1, seasons=(2025,), gws=3)
    # Give the player a second fixture in gameweek 2.
    extra = h[h.gw == 2].copy()
    extra["fixture_id"] = 999
    extra["total_points"] = 5.0
    extra["minutes"] = 45
    monkeypatch.setattr(panel_mod, "load_history", lambda seasons=None: pd.concat([h, extra]))
    monkeypatch.setattr(panel_mod, "load_history_fixtures", lambda seasons=None: pd.DataFrame())
    p = build_panel(save=False)

    gw2 = p[p.gw == 2].iloc[0]
    assert gw2.n_fixtures == 2
    assert gw2.points == pytest.approx(pts(0, 2) + 5.0)
    assert gw2.minutes == pytest.approx(90 + 45)
    assert len(p[p.gw == 2]) == 1


def test_a_stat_that_did_not_exist_stays_missing_rather_than_becoming_zero(monkeypatch):
    """`groupby.sum()` turns an all-NaN group into 0.0, which is a silent data corruption.

    Defensive contribution only exists from 2025-26. Read as 0 for earlier seasons it means
    "this player recorded none", and four fifths of the panel became confident zeroes: the
    measured hit rate fell from about 15% to 2.7% and the model trained on it accordingly.
    """
    h = _history(n_players=2, seasons=(2024, 2025), gws=4)
    h.loc[h.season == 2024, "defensive_contribution"] = np.nan
    monkeypatch.setattr(panel_mod, "load_history", lambda seasons=None: h)
    monkeypatch.setattr(panel_mod, "load_history_fixtures", lambda seasons=None: pd.DataFrame())
    p = build_panel(save=False)

    assert p.loc[p.season == 2024, "defensive_contribution"].isna().all()
    assert p.loc[p.season == 2025, "defensive_contribution"].notna().all()


def test_real_panel_keeps_defensive_contribution_missing_before_it_existed():
    from src.fpl_predict.models.panel import load_panel

    p = load_panel()
    if p.empty or "defensive_contribution" not in p.columns:
        pytest.skip("no history on disk")
    coverage = p.groupby("season")["defensive_contribution"].apply(lambda s: s.notna().mean())
    older = coverage[coverage.index < 2025]
    if len(older):
        assert (older == 0).all(), "a season before the rule existed reports the stat"


def test_real_panel_is_leak_free_and_the_right_shape():
    """Same checks against the real archive, which is where leakage would actually bite."""
    from src.fpl_predict.models.panel import load_panel

    p = load_panel()
    if p.empty:
        pytest.skip("no history on disk")

    assert not p.duplicated(subset=["season", "player_code", "gw"]).any()

    features = feature_columns(p)
    assert "fpl_xp" not in features
    assert not set(features) & set(OUTCOME_COLS)

    # Nothing may correlate near-perfectly with points.
    sub = p[p.minutes > 0]
    worst = ("", 0.0)
    for col in features:
        pair = sub[[col, "points"]].dropna()
        if len(pair) < 500 or pair[col].nunique() < 2:
            continue
        r = abs(np.corrcoef(pair[col], pair["points"])[0, 1])
        if r > worst[1]:
            worst = (col, r)
    assert worst[1] < 0.9, f"{worst[0]} correlates {worst[1]:.3f} with points"

    # Opening gameweek of the earliest season cannot have prior form.
    earliest = p.season.min()
    opener = p[(p.season == earliest) & (p.gw == 1)]
    assert opener["points_l5"].isna().all()
