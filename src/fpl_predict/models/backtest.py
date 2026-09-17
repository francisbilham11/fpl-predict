"""Rolling-origin backtest for expected-points predictors.

Replaces a one-line stub that returned `{"MAE": 1.2, "RMSE": 1.8}` and was never called by
anything, while the README quoted those numbers as measured performance.

**Protocol.** Walk forward one gameweek at a time. At gameweek *g* a predictor may see every
panel row before *g* and the pre-deadline context of *g* itself, never its outcome. Refitting
every gameweek over five seasons is slow and buys little, so models are refit every
`refit_every` gameweeks and reused in between; a predictor that needs no fitting ignores this.

**Metrics.** MAE alone is a poor guide here. Two thirds of the selection universe scores 0-2,
so predicting "everyone gets 2" scores well on MAE and is useless for picking a team. What
matters is whether the ranking is right at the top, so the headline numbers are:

| Metric | What it answers |
|:-------|:----------------|
| `spearman` | Does the whole ranking hold up? |
| `xi_points` | Points actually scored by the best legal XI the predictor's ranking implies |
| `xi_capture` | That, as a share of the best XI available in hindsight |
| `captain_points` | Points scored by its single highest-ranked player |
| `mae` / `rmse` | Calibration of the magnitude, reported but not the target |

`xi_points` is the number to care about: it is the same decision the optimizer makes, scored
against what actually happened.

**On the archive's `xP` column: do not use it as a baseline.** It looks like FPL's
pre-deadline expected points and it is tempting to treat it as the number to beat, but it is
contaminated with post-match information. `leakage_report` reproduces the evidence; the
decisive figures, over established starters (`mins_l5 >= 70`):

| Check | `fpl_xp` | Best lagged feature | A real forecast should |
|:------|:---------|:--------------------|:-----------------------|
| Correlation with this gameweek's points, players who did play | 0.613 | 0.187 | be nearer the lagged figure |
| Correlation with this gameweek's *bonus* | 0.434 | — | be near zero, bonus is decided in-match |
| AUC for whether a starter played at all | 0.826 | 0.615 | be nearer the lagged figure |
| Correlation with previous gameweek's points | 0.674 | — | exceed its correlation with the *next* |

A forecast cannot know the outcome better than it knows the history it was built from, and no
pre-match model predicts a single match's bonus points at 0.43. The realistic ceiling for
correlation with one gameweek's score is roughly 0.25-0.35.

Note this says nothing about the live API's `ep_next`, which is a genuine forward-looking
figure published before its gameweek. It cannot be evaluated here because past values are not
recoverable, so the pipeline's habit of blending it in is unmeasured rather than discredited.

**Baselines that are valid** are the naive rules (`points_l5`, minutes-weighted form,
position mean) and `CurrentArchitecturePredictor`, which reproduces the shipped model's
structure on historical features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from .panel import load_panel

log = get_logger(__name__)

# Legal FPL formations as (DEF, MID, FWD); one keeper is always required.
FORMATIONS = [(3, 4, 3), (3, 5, 2), (4, 4, 2), (4, 5, 1), (4, 3, 3), (5, 4, 1), (5, 3, 2)]

# A player needs some evidence of being in the picture before their prediction is
# decision-relevant. Metrics are reported over the whole universe and over this subset.
STARTER_MIN_MINS_L5 = 30.0

# Share of a gameweek's rows that may be exactly zero before its `fpl_xp` is treated as
# missing rather than as a forecast of nothing. Individual zeros are normal (FPL expects
# nothing from a benched keeper); a gameweek that is *mostly* zero means the archive failed
# to capture the column that week. Several 2025-26 gameweeks are zero for every player.
XP_ZERO_SHARE_LIMIT = 0.5


def usable_xp_gameweeks(panel: pd.DataFrame) -> pd.DataFrame:
    """The (season, gw) pairs where FPL's own xP was actually recorded.

    Without this the baseline is scored on gameweeks where every xP is 0, its XI selection
    becomes arbitrary, and it looks far worse than it is: capture fell from ~62% to 23% in
    2025-26 purely because of missing data.
    """
    if "fpl_xp" not in panel.columns:
        return panel[["season", "gw"]].drop_duplicates()
    stats = (
        panel.assign(_zero=panel["fpl_xp"].fillna(0) == 0)
        .groupby(["season", "gw"], as_index=False)
        .agg(zero_share=("_zero", "mean"))
    )
    usable = stats[stats.zero_share <= XP_ZERO_SHARE_LIMIT][["season", "gw"]]
    dropped = len(stats) - len(usable)
    if dropped:
        by_season = stats[stats.zero_share > XP_ZERO_SHARE_LIMIT].groupby("season").size().to_dict()
        log.warning(
            "Excluding %d gameweeks with no recorded FPL xP (by season: %s)", dropped, by_season
        )
    return usable


class Predictor(Protocol):
    """Anything that can score a gameweek's players."""

    name: str

    def fit(self, train: pd.DataFrame) -> None:
        """Fit on rows strictly before the gameweek being predicted."""

    def predict(self, rows: pd.DataFrame) -> pd.Series:
        """Expected points per row, indexed like `rows`."""


@dataclass
class GameweekResult:
    season: int
    gw: int
    n_players: int
    mae: float
    rmse: float
    spearman: float
    xi_points: float
    xi_oracle: float
    xi_capture: float
    captain_points: float
    top11_points: float


def _best_xi_points(pred: pd.Series, actual: pd.Series, positions: pd.Series) -> float:
    """Actual points of the best legal XI implied by `pred`.

    Picks the top-ranked players per position for each formation and keeps the formation with
    the highest predicted total, then reports what that XI actually scored. Budget and the
    three-per-club limit are ignored: this compares rankings, and adding constraints would
    mix the optimizer's behaviour into a measurement of the predictions.
    """
    frame = pd.DataFrame({"pred": pred, "actual": actual, "pos": positions}).dropna(subset=["pred"])
    if frame.empty:
        return float("nan")

    by_pos = {p: g.sort_values("pred", ascending=False) for p, g in frame.groupby("pos")}
    if "GKP" not in by_pos or by_pos["GKP"].empty:
        return float("nan")

    best_pred, best_actual = -np.inf, float("nan")
    gkp = by_pos["GKP"].iloc[0]
    for n_def, n_mid, n_fwd in FORMATIONS:
        picks = [gkp]
        ok = True
        for pos, n in (("DEF", n_def), ("MID", n_mid), ("FWD", n_fwd)):
            group = by_pos.get(pos)
            if group is None or len(group) < n:
                ok = False
                break
            picks.extend(group.iloc[i] for i in range(n))
        if not ok:
            continue
        total_pred = sum(p["pred"] for p in picks)
        if total_pred > best_pred:
            best_pred = total_pred
            best_actual = sum(p["actual"] for p in picks)
    return float(best_actual)


def _spearman(a: pd.Series, b: pd.Series) -> float:
    pair = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(pair) < 10 or pair.a.nunique() < 2 or pair.b.nunique() < 2:
        return float("nan")
    return float(pair.a.rank().corr(pair.b.rank()))


def evaluate_gameweek(pred: pd.Series, rows: pd.DataFrame, season: int, gw: int) -> GameweekResult:
    """Score one gameweek's predictions against what happened."""
    actual = rows["points"].astype(float)
    err = (pred - actual).dropna()

    oracle = _best_xi_points(actual, actual, rows["position"])
    chosen = _best_xi_points(pred, actual, rows["position"])

    order = pred.sort_values(ascending=False)
    captain = float(actual.loc[order.index[0]]) if len(order) else float("nan")
    top11 = float(actual.loc[order.index[:11]].sum()) if len(order) >= 11 else float("nan")

    return GameweekResult(
        season=season,
        gw=gw,
        n_players=len(rows),
        mae=float(err.abs().mean()) if len(err) else float("nan"),
        rmse=float(np.sqrt((err**2).mean())) if len(err) else float("nan"),
        spearman=_spearman(pred, actual),
        xi_points=chosen,
        xi_oracle=oracle,
        xi_capture=chosen / oracle
        if oracle and np.isfinite(oracle) and oracle > 0
        else float("nan"),
        captain_points=captain,
        top11_points=top11,
    )


def rolling_origin_backtest(
    predictor: Predictor,
    panel: Optional[pd.DataFrame] = None,
    seasons: Optional[Sequence[int]] = None,
    min_train_gws: int = 38,
    refit_every: int = 6,
    starters_only: bool = False,
    score_gameweeks: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Walk forward through the panel, scoring `predictor` one gameweek at a time.

    Returns one row per gameweek. `min_train_gws` gameweeks of history are held back before
    scoring starts, so the first evaluated gameweek has a full season behind it. Gameweeks
    outside `score_gameweeks` still contribute to training but are not scored, which is how
    the FPL-xP data gaps are excluded from every predictor equally.
    """
    panel = load_panel() if panel is None else panel
    if panel.empty:
        log.warning("No panel data; nothing to backtest")
        return pd.DataFrame()

    if seasons is not None:
        panel = panel[panel["season"].isin([int(s) for s in seasons])]

    panel = panel.sort_values(["season", "gw"], kind="stable")
    keys = panel[["season", "gw"]].drop_duplicates().reset_index(drop=True)

    scoreable = (
        None
        if score_gameweeks is None
        else set(zip(score_gameweeks.season.astype(int), score_gameweeks.gw.astype(int)))
    )

    results: List[GameweekResult] = []
    fitted_at = -(10**9)

    for i, key in keys.iterrows():
        season, gw = int(key.season), int(key.gw)
        if i < min_train_gws:
            continue
        if scoreable is not None and (season, gw) not in scoreable:
            continue

        earlier = keys.iloc[:i]
        train_mask = panel.set_index(["season", "gw"]).index.isin(
            list(zip(earlier.season, earlier.gw))
        )
        train = panel[train_mask]
        rows = panel[(panel.season == season) & (panel.gw == gw)]
        if starters_only:
            rows = rows[rows["mins_l5"].fillna(0) >= STARTER_MIN_MINS_L5]
        if len(rows) < 20:
            continue

        if i - fitted_at >= refit_every:
            predictor.fit(train)
            fitted_at = i

        pred = predictor.predict(rows)
        pred = pd.Series(np.asarray(pred, dtype=float), index=rows.index)
        results.append(evaluate_gameweek(pred, rows, season, gw))

    df = pd.DataFrame([r.__dict__ for r in results])
    if not df.empty:
        log.info(
            "%s: %d gameweeks, MAE %.3f, spearman %.3f, XI capture %.1f%%",
            predictor.name,
            len(df),
            df.mae.mean(),
            df.spearman.mean(),
            100 * df.xi_capture.mean(),
        )
    return df


def summarise(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Collapse per-gameweek results into one row per predictor."""
    rows = []
    for name, df in results.items():
        if df.empty:
            continue
        rows.append(
            {
                "predictor": name,
                "gameweeks": len(df),
                "mae": df.mae.mean(),
                "rmse": df.rmse.mean(),
                "spearman": df.spearman.mean(),
                "xi_points": df.xi_points.mean(),
                "xi_capture": 100 * df.xi_capture.mean(),
                "captain_points": df.captain_points.mean(),
                "oracle_xi": df.xi_oracle.mean(),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("xi_points", ascending=False) if not out.empty else out


# --------------------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------------------


@dataclass
class ColumnPredictor:
    """Rank by an existing panel column. Used for FPL's own xP and for naive form."""

    column: str
    name: str
    fill: float = 0.0

    def fit(self, train: pd.DataFrame) -> None:  # noqa: D102 - nothing to fit
        return

    def predict(self, rows: pd.DataFrame) -> pd.Series:
        if self.column not in rows.columns:
            return pd.Series(self.fill, index=rows.index)
        return rows[self.column].astype(float).fillna(self.fill)


@dataclass
class PositionMeanPredictor:
    """Every player gets their position's historical mean. A floor for the other metrics."""

    name: str = "position mean"
    _means: Dict[str, float] = field(default_factory=dict)
    _overall: float = 2.0

    def fit(self, train: pd.DataFrame) -> None:
        self._means = train.groupby("position")["points"].mean().to_dict()
        self._overall = float(train["points"].mean())

    def predict(self, rows: pd.DataFrame) -> pd.Series:
        return rows["position"].map(self._means).astype(float).fillna(self._overall)


@dataclass
class MinutesWeightedFormPredictor:
    """Recent points per appearance, scaled by how likely the player is to start.

    A deliberately simple model of the right shape: separate the chance of playing from the
    rate of scoring when they do.
    """

    name: str = "minutes-weighted form"

    def fit(self, train: pd.DataFrame) -> None:
        return

    def predict(self, rows: pd.DataFrame) -> pd.Series:
        played_rate = rows["played_l5"].fillna(0.0).clip(0, 1)
        mins = rows["mins_l5"].fillna(0.0).clip(0, 90)
        per_app = np.where(
            played_rate > 0,
            rows["points_l5"].fillna(0.0) / played_rate.replace(0, np.nan),
            0.0,
        )
        return pd.Series(np.nan_to_num(per_app) * (mins / 90.0), index=rows.index)


@dataclass
class CurrentArchitecturePredictor:
    """Reimplementation of the shipped `expected_points_df` on historical features.

    The live pipeline cannot be replayed: it depends on `ep_next` and bootstrap state that is
    not recoverable for past gameweeks. This reproduces its *structure* instead, so the
    backtest measures the design rather than a particular day's API response:

        appearance + goals x goal_points + assists x 3 + P(clean sheet) x cs_points + defcon

    with expected minutes from recent minutes, goal and assist rates from recent per-90
    output, clean-sheet probability from the team's recent rate, and the defensive
    contribution term using the shipped `min(rate / threshold, 0.8)` formula. Everything the
    shipped model omits is omitted here too: no saves, no goals conceded, no bonus, no cards.
    """

    name: str = "current architecture"
    goal_points: Dict[str, float] = field(
        default_factory=lambda: {"GKP": 10.0, "DEF": 6.0, "MID": 5.0, "FWD": 4.0}
    )
    cs_points: Dict[str, float] = field(
        default_factory=lambda: {"GKP": 4.0, "DEF": 4.0, "MID": 1.0, "FWD": 0.0}
    )
    dc_threshold: Dict[str, float] = field(
        default_factory=lambda: {"GKP": 99.0, "DEF": 10.0, "MID": 12.0, "FWD": 12.0}
    )

    def fit(self, train: pd.DataFrame) -> None:
        # Fallback clean-sheet rate for teams with no recent record.
        self._cs_default = float(
            train.get("team_cs_rate_l10", pd.Series([0.28])).dropna().mean() or 0.28
        )

    def predict(self, rows: pd.DataFrame) -> pd.Series:
        pos = rows["position"].astype(str)
        xmins = rows["mins_l5"].fillna(0.0).clip(0, 90)

        appearance = (xmins > 0).astype(float) * 1.0 + (xmins >= 60).astype(float) * 1.0

        mu_g = rows["goals_per90_l10"].fillna(0.0) * (xmins / 90.0)
        mu_a = rows["assists_per90_l10"].fillna(0.0) * (xmins / 90.0)
        attack = mu_g * pos.map(self.goal_points).fillna(0.0) + mu_a * 3.0

        cs_rate = rows.get("team_cs_rate_l10", pd.Series(np.nan, index=rows.index))
        cs_rate = cs_rate.fillna(getattr(self, "_cs_default", 0.28))
        clean_sheet = cs_rate * pos.map(self.cs_points).fillna(0.0)

        dc_rate = rows.get("dc_l5", pd.Series(np.nan, index=rows.index)).fillna(0.0)
        threshold = pos.map(self.dc_threshold).fillna(12.0)
        dc_prob = (dc_rate / threshold).clip(upper=0.8)
        defcon = 2.0 * dc_prob * (xmins / 90.0)
        defcon = defcon.where(xmins >= 30, 0.0)
        defcon = defcon.where(pos != "GKP", 0.0)

        return (appearance + attack + clean_sheet + defcon).astype(float)


def leakage_report(panel: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Evidence that the archive's `xP` is contaminated, recomputed from the panel.

    Kept as code rather than a note so the claim can be rechecked whenever the archive is
    refreshed, and so nobody reinstates `fpl_xp` as a baseline on the assumption that a
    column named "expected points" must be a forecast.
    """
    from sklearn.metrics import roc_auc_score

    panel = load_panel() if panel is None else panel
    usable = usable_xp_gameweeks(panel)
    keys = set(zip(usable.season.astype(int), usable.gw.astype(int)))
    p = panel[[(s, g) in keys for s, g in zip(panel.season, panel.gw)]].copy()

    starters = p[p["mins_l5"].fillna(0) >= 70]
    played = starters[starters["minutes"] > 0]

    def corr(a: pd.Series, b: pd.Series) -> float:
        pair = pd.DataFrame({"a": a, "b": b}).dropna()
        return float(np.corrcoef(pair.a, pair.b)[0, 1]) if len(pair) > 100 else float("nan")

    lagged = ["points_l5", "season_ppg_to_date", "xg_per90_l10", "bps_l5"]
    best_lagged = max(
        (corr(played[c], played["points"]) for c in lagged if c in played.columns),
        default=float("nan"),
    )
    y = (starters["minutes"] > 0).astype(int)

    rows = [
        {
            "check": "corr with this GW points (players who played)",
            "fpl_xp": corr(played["fpl_xp"], played["points"]),
            "best_lagged": best_lagged,
        },
        {
            "check": "corr with this GW bonus (decided in-match)",
            "fpl_xp": corr(played["fpl_xp"], played["bonus"]),
            "best_lagged": float("nan"),
        },
        {
            "check": "AUC for whether a starter played",
            "fpl_xp": float(roc_auc_score(y, starters["fpl_xp"].fillna(0))),
            "best_lagged": float(roc_auc_score(y, starters["mins_l5"].fillna(0))),
        },
    ]
    return pd.DataFrame(rows)


def default_baselines() -> List[Predictor]:
    """Valid comparison set: two naive rules, a floor, and the shipped design.

    `fpl_xp` is deliberately absent. See the module docstring and `leakage_report`.
    """
    return [
        ColumnPredictor(column="points_l5", name="mean points, last 5"),
        MinutesWeightedFormPredictor(),
        PositionMeanPredictor(),
        CurrentArchitecturePredictor(),
    ]


def run_backtest(
    predictors: Optional[Sequence[Predictor]] = None,
    seasons: Optional[Sequence[int]] = None,
    starters_only: bool = False,
    min_train_gws: int = 38,
    refit_every: int = 6,
    require_xp: bool = False,
) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Backtest a set of predictors over the same gameweeks and summarise them.

    `require_xp` restricts scoring to gameweeks where the archive recorded `xP`. It defaults
    off: `xP` is no longer used as a baseline, so there is no reason to throw away the 33
    gameweeks that lack it.
    """
    panel = load_panel()
    if panel.empty:
        log.warning("No panel; run `fpl backtest --rebuild-panel` first")
        return pd.DataFrame(), {}

    score_gameweeks = usable_xp_gameweeks(panel) if require_xp else None

    predictors = list(predictors) if predictors is not None else default_baselines()
    per_gw: Dict[str, pd.DataFrame] = {}
    for p in predictors:
        per_gw[p.name] = rolling_origin_backtest(
            p,
            panel=panel,
            seasons=seasons,
            min_train_gws=min_train_gws,
            refit_every=refit_every,
            starters_only=starters_only,
            score_gameweeks=score_gameweeks,
        )
    return summarise(per_gw), per_gw


def ablation_report(
    groups: Optional[Sequence[str]] = None,
    panel: Optional[pd.DataFrame] = None,
    min_train_gws: int = 38,
    refit_every: int = 18,
) -> pd.DataFrame:
    """Refit the component model with each feature family withheld and measure the loss.

    Answers "is this family earning its keep", which a single headline number cannot.

    Read the MAE and Spearman columns, not XI points. XI capture is a blunt instrument for
    this: a fixture-adjusted top eleven overlaps about 75% with a quality-only one, so even
    perfect fixture modelling can reorder at most three of the eleven. A family can carry real
    signal and barely move the XI metric.
    """
    from .panel import CONTEXT_GROUPS, FEATURE_GROUPS, group_features
    from .points import ComponentPointsModel

    panel = load_panel() if panel is None else panel
    if panel.empty:
        return pd.DataFrame()

    names = list(groups) if groups else list(FEATURE_GROUPS)
    experiments: Dict[str, List[str]] = {"full model": []}
    for g in names:
        experiments[f"drop {g}"] = group_features(g)
    if not groups:
        experiments["drop all fixture context"] = group_features(*CONTEXT_GROUPS)

    per_gw: Dict[str, pd.DataFrame] = {}
    for label, drop in experiments.items():
        model = ComponentPointsModel(drop_features=list(drop))
        model.name = label
        per_gw[label] = rolling_origin_backtest(
            model, panel=panel, min_train_gws=min_train_gws, refit_every=refit_every
        )

    base = per_gw["full model"].set_index(["season", "gw"])
    rows = []
    for label, df in per_gw.items():
        if df.empty:
            continue
        other = df.set_index(["season", "gw"])
        paired = pd.DataFrame({"mae_a": base.mae, "mae_b": other.mae}).dropna()
        diff = paired.mae_b - paired.mae_a
        se = diff.std() / np.sqrt(len(diff)) if len(diff) > 1 else np.nan
        rows.append(
            {
                "experiment": label,
                "mae": df.mae.mean(),
                "mae_worse_by": df.mae.mean() - base.mae.mean(),
                "t": (diff.mean() / se) if se else np.nan,
                "spearman": df.spearman.mean(),
                "spearman_drop": base.spearman.mean() - df.spearman.mean(),
                "xi_points": df.xi_points.mean(),
                "xi_drop": base.xi_points.mean() - df.xi_points.mean(),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("mae_worse_by", ascending=False).reset_index(drop=True)


def format_ablation(report: pd.DataFrame) -> str:
    """Render the ablation as a markdown table."""
    if report.empty:
        return "No ablation results."
    out = report.copy()
    out.columns = [
        "Experiment",
        "MAE",
        "MAE worse by",
        "t",
        "Spearman",
        "Spearman drop",
        "XI points",
        "XI drop",
    ]
    return out.to_markdown(index=False, floatfmt=".4f")


def format_summary(summary: pd.DataFrame) -> str:
    """Render the summary as a markdown table."""
    if summary.empty:
        return "No backtest results."
    out = summary.copy()
    out.columns = [
        "Predictor",
        "GWs",
        "MAE",
        "RMSE",
        "Spearman",
        "XI points",
        "XI capture %",
        "Captain pts",
        "Oracle XI",
    ]
    return out.to_markdown(index=False, floatfmt=".3f")
