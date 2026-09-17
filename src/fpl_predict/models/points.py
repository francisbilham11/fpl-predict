"""Expected-points models trained on the player-gameweek panel.

Two shapes, both measurable through `models.backtest`:

- `DirectPointsModel` regresses total points on the lagged features in one step. Cheap, and
  a strong baseline now that the panel is leak-free.
- `ComponentPointsModel` predicts each scoring component on its own scale and adds them up
  using the season's actual scoring table. More code, but it can express things a single
  regressor struggles with, and every term is inspectable.

The component model exists because of what the shipped pipeline leaves out. Measured over
45,787 appearances from 2022-23 to 2025-26 under the current scoring table:

| Component              | In the shipped model | Real mean points per appearance |
|:-----------------------|:---------------------|:--------------------------------|
| Saves                  | no                   | 0.66 for GKP, 17% of their total |
| Goals conceded         | no                   | -0.50 GKP, -0.40 DEF            |
| Bonus                  | no                   | 0.17-0.36, and the strongest predictor of a score after goals |
| Cards                  | no                   | -0.07 to -0.17                  |
| Defensive contribution | over-credited        | 3x for DEF, 4.5x for MID, 69x for FWD |

The defensive-contribution term is the clearest example of why these are learned rather than
assumed. The shipped formula is `min(rate / threshold, 0.8)`, which treats a player averaging
the threshold as near-certain to clear it; the real hit rate at that average is about 50%, and
across all appearances it is 21% for defenders, 11% for midfielders and 1% for forwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from .panel import feature_columns

log = get_logger(__name__)

POSITIONS = ("GKP", "DEF", "MID", "FWD")

# Minutes bands. Appearance points and clean sheets both hinge on reaching 60.
BAND_NONE, BAND_SHORT, BAND_LONG = 0, 1, 2

# Representative minutes within each band, used to convert a per-90 rate into an expectation.
BAND_MINUTES = {BAND_NONE: 0.0, BAND_SHORT: 30.0, BAND_LONG: 82.0}

# Goals conceded and saves are scored in whole units of 2 and 3, so the expectation has to be
# taken over the count distribution rather than applied to the mean. The sum runs to this many
# events; anything beyond is folded into the last term, so it needs to sit well above the
# highest plausible rate (about 8 conceded, about 12 saves) or the tail is under-counted.
COUNT_SUPPORT = 30

# Ceiling on a per-90 rate target. With exposure weighting this only trims genuine outliers;
# without it, it was the only thing standing between the model and a target of 90 goals per 90.
RATE_CLIP = 6.0


# Shared LightGBM settings. Every sub-model starts from these; `ComponentPointsModel` can
# override them wholesale via `lgbm_params`, which is what the tuner drives.
#
# Tuned by 12 Optuna TPE trials maximising Spearman on the development seasons (2021-22 to
# 2024-25), then evaluated once on 2025-26. Eight of the twelve trials beat the previous
# hand-picked settings, from scattered corners of the space, so this is a plateau rather than
# a lucky point.
#
#              dev (chose these)      holdout 2025-26 (did not)
#   XI points  56.46 -> 58.98 (+2.5)  49.63 -> 53.55 (+3.9, t=+2.5)
#   Spearman   0.6865 -> 0.6911       0.7138 -> 0.7149
#   MAE        1.0437 -> 1.0463       0.9902 -> 1.0082 (worse, t=+7.5)
#
# The gain grew on the holdout rather than evaporating, which is the opposite of the feature
# prune that was rejected. MAE gets worse, which is expected and fine: across the 13 runs,
# MAE correlates +0.63 with XI points, and since lower MAE is better that means better average
# error goes with *worse* team selection. Spearman correlates +0.94 with XI, which is why it
# was the search objective.
#
# The previous values were n_estimators=400, learning_rate=0.05, num_leaves=31,
# min_child_samples=40, colsample_bytree=0.8, reg_lambda=1.0. The direction of travel is
# firmly toward more regularisation: smaller trees, fewer of them, larger leaves. It also fits
# roughly three times faster, and across the search slower models were reliably worse
# (fit time against XI correlates -0.84).
DEFAULT_LGBM_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.036,
    num_leaves=15,
    min_child_samples=56,
    subsample=0.914,
    subsample_freq=1,
    colsample_bytree=0.600,
    reg_lambda=1.525,
    random_state=42,
    verbosity=-1,
)


def _lgbm(objective: str, overrides: Optional[Dict] = None, **kw):
    import lightgbm as lgb

    params = dict(DEFAULT_LGBM_PARAMS)
    if overrides:
        params.update(overrides)
    params["objective"] = objective
    params.update(kw)
    return lgb.LGBMRegressor(**params)


def _lgbm_classifier(objective: str, overrides: Optional[Dict] = None, **kw):
    import lightgbm as lgb

    params = dict(DEFAULT_LGBM_PARAMS)
    if overrides:
        params.update(overrides)
    params["objective"] = objective
    params.update(kw)
    return lgb.LGBMClassifier(**params)


def _design(rows: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Numeric design matrix with position one-hot encoded.

    Team and opponent are deliberately not encoded as identities: their strength is already
    carried by `team_gf_l10`, `team_ga_l10` and the opponent equivalents, and identity
    features would tie the model to the clubs that happen to be in the division.
    """
    X = rows.reindex(columns=columns).astype(float)
    pos = rows["position"].astype(str)
    for p in POSITIONS:
        X[f"pos_{p}"] = (pos == p).astype(float)
    return X


@dataclass
class DirectPointsModel:
    """One regressor from lagged features to total points."""

    name: str = "direct GBM"
    drop_features: List[str] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    _model: object = None

    def fit(self, train: pd.DataFrame) -> None:
        dropped = set(self.drop_features)
        self.columns = [c for c in feature_columns(train) if c != "fpl_xp" and c not in dropped]
        X = _design(train, self.columns)
        y = train["points"].astype(float)
        self._model = _lgbm("regression")
        self._model.fit(X, y)

    def predict(self, rows: pd.DataFrame) -> pd.Series:
        if self._model is None:
            return pd.Series(0.0, index=rows.index)
        pred = self._model.predict(_design(rows, self.columns))
        return pd.Series(np.clip(pred, -2.0, None), index=rows.index)


@dataclass
class ComponentPointsModel:
    """Predict each scoring component separately, then add them up.

    Sub-models, and why each is shaped the way it is:

    | Component | Model | Note |
    |:----------|:------|:-----|
    | minutes band | 3-class classifier | separates "will they play" from "how well" |
    | goals, assists | Poisson on per-90 rate | scaled by expected minutes |
    | clean sheet | classifier on team form | requires 60+ minutes to pay out |
    | goals conceded | Poisson count | scored per 2, so summed over the distribution |
    | saves | Poisson count | scored per 3, keepers only |
    | bonus | regressor | omitted entirely by the shipped model |
    | cards | regressor | omitted entirely by the shipped model |
    | defensive contribution | classifier on hitting the threshold | replaces `min(rate/thr, 0.8)` |
    """

    name: str = "component model"
    goal_points: Dict[str, float] = field(
        default_factory=lambda: {"GKP": 10.0, "DEF": 6.0, "MID": 5.0, "FWD": 4.0}
    )
    cs_points: Dict[str, float] = field(
        default_factory=lambda: {"GKP": 4.0, "DEF": 4.0, "MID": 1.0, "FWD": 0.0}
    )
    conceded_positions: tuple = ("GKP", "DEF")
    dc_threshold: Dict[str, float] = field(
        default_factory=lambda: {"DEF": 10.0, "MID": 12.0, "FWD": 12.0}
    )
    dc_points: float = 2.0
    penalty_save_points: float = 5.0
    penalty_miss_points: float = -2.0
    own_goal_points: float = -2.0
    # Feature names to withhold, for ablation. Note that `n_fixtures` still drives the
    # double-gameweek multiplier in `predict`, because that is structural arithmetic rather
    # than something the model learns; dropping it here only hides it from the sub-models.
    drop_features: List[str] = field(default_factory=list)
    # Weight the goals and assists models by minutes played, so a short cameo cannot assert a
    # huge per-90 rate. Equivalent to a Poisson offset of log(minutes), and the textbook
    # treatment of exposure.
    #
    # Off by default because it was measured and it lost. On the development seasons it is
    # consistently but trivially worse (MAE +0.0069, t=+14.1; XI -1.20, t=-1.8; Spearman
    # unchanged), and it is better calibrated on goals in isolation (predicted/actual 1.007
    # against 1.048), so the loss is not a calibration effect and the mechanism is unexplained.
    # Kept as a flag rather than deleted: it is the more defensible formulation and worth
    # revisiting if the minutes model changes, since the two interact through expected minutes.
    weight_by_exposure: bool = False
    # Override the shared LightGBM settings. Set by the tuner; None means DEFAULT_LGBM_PARAMS.
    lgbm_params: Optional[Dict] = None
    columns: List[str] = field(default_factory=list)
    _parts: Dict[str, object] = field(default_factory=dict)
    _own_goal_rate_per90: float = 0.0

    # ------------------------------------------------------------------ fitting

    def fit(self, train: pd.DataFrame) -> None:

        dropped = set(self.drop_features)
        self.columns = [c for c in feature_columns(train) if c != "fpl_xp" and c not in dropped]
        X = _design(train, self.columns)
        self._parts = {}

        # Everything is modelled *per match* and multiplied by the fixture count at predict
        # time. Without this the model cannot express a double gameweek at all: expected
        # minutes cap at one match, so a player with two fixtures is predicted to score half
        # what they should, and doubles are exactly the gameweeks where the big scores are.
        fixtures = train["n_fixtures"].clip(lower=1)
        mins_per_match = train["minutes"] / fixtures

        # Minutes band, over every row: learning who will not play is half the job.
        band = np.where(
            mins_per_match >= 60, BAND_LONG, np.where(mins_per_match > 0, BAND_SHORT, BAND_NONE)
        )
        clf = _lgbm_classifier("multiclass", self.lgbm_params, num_class=3)
        clf.fit(X, band)
        self._parts["band"] = clf

        # Everything else is conditional on appearing, and expressed per 90 minutes so the
        # minutes model is the only thing deciding how much game time to expect. Counting
        # stats are divided by the fixture count so a double gameweek does not look like an
        # unusually productive single match.
        played = train[train["minutes"] > 0]
        if len(played) < 500:
            log.warning("Only %d appearances to fit component models", len(played))
            return
        Xp = _design(played, self.columns)
        played_fixtures = played["n_fixtures"].clip(lower=1)
        per90 = 90.0 / played["minutes"].clip(lower=1)

        # Rate models weighted by exposure. Modelling a per-90 rate with weight proportional
        # to minutes is the standard equivalent of modelling the raw count with log(minutes)
        # as a Poisson offset, and it is the difference between a usable model and a broken
        # one here: a goal in a one-minute cameo is a target of 90 goals per 90, and 23% of
        # appearances are under half an hour. Unweighted, those rows have a standard deviation
        # of 3.46 against 0.36 for a full match, and they drown the signal.
        exposure = played["minutes"].clip(lower=1) / 90.0
        for key, col in (("goals", "goals"), ("assists", "assists")):
            m = _lgbm("poisson", self.lgbm_params)
            rate = (played[col].astype(float) * per90).clip(upper=RATE_CLIP)
            if self.weight_by_exposure:
                m.fit(Xp, rate, sample_weight=exposure)
            else:
                m.fit(Xp, rate)
            self._parts[key] = m

        m = _lgbm("regression", self.lgbm_params)
        m.fit(Xp, played["bonus"].astype(float) / played_fixtures)
        self._parts["bonus"] = m

        m = _lgbm("regression", self.lgbm_params)
        m.fit(
            Xp,
            (played["yellow_cards"].astype(float) + 3 * played["red_cards"].astype(float))
            / played_fixtures,
        )
        self._parts["cards"] = m

        # Clean sheets and goals conceded only pay out for players on the pitch at 60+.
        long_apps = train[mins_per_match >= 60]
        if len(long_apps) >= 500:
            Xl = _design(long_apps, self.columns)
            long_fixtures = long_apps["n_fixtures"].clip(lower=1)
            cs = _lgbm_classifier("binary", self.lgbm_params)
            # Per match, so a double with one clean sheet reads as a 50% rate.
            cs.fit(Xl, (long_apps["clean_sheet"] / long_fixtures >= 0.5).astype(int))
            self._parts["cs"] = cs

            gc = _lgbm("poisson", self.lgbm_params)
            gc.fit(Xl, (long_apps["goals_conceded"].astype(float) / long_fixtures).clip(upper=8))
            self._parts["conceded"] = gc

        keepers = played[played["position"] == "GKP"]
        if len(keepers) >= 200:
            m = _lgbm("poisson", self.lgbm_params)
            m.fit(
                _design(keepers, self.columns),
                (keepers["saves"].astype(float) * per90.loc[keepers.index]).clip(upper=15),
            )
            self._parts["saves"] = m

            if "penalties_saved" in keepers.columns and keepers["penalties_saved"].sum() > 0:
                m = _lgbm("poisson", self.lgbm_params)
                m.fit(
                    _design(keepers, self.columns),
                    (keepers["penalties_saved"].astype(float) * per90.loc[keepers.index]).clip(
                        upper=2
                    ),
                )
                self._parts["penalty_saves"] = m

        # Penalty misses: rare, and there is no "who takes penalties" feature, but a rate
        # model still lets the players who actually take (and occasionally miss) them show up
        # as a small extra risk rather than pricing every position identically at zero.
        if "penalties_missed" in played.columns and played["penalties_missed"].sum() > 0:
            m = _lgbm("poisson", self.lgbm_params)
            m.fit(
                Xp, (played["penalties_missed"].astype(float) * per90).clip(upper=2)
            )
            self._parts["penalty_misses"] = m

        # Own goals are close enough to a random per-appearance event (no real skill signal,
        # unlike goals/assists/DC) that a learned per-player rate would mostly be fitting
        # noise. A single league-wide per-90 rate keeps the small negative EV term honest
        # about how predictable it actually is, rather than pretending otherwise.
        if "own_goals" in played.columns:
            total_og = float(played["own_goals"].astype(float).sum())
            total_mins = float(played["minutes"].astype(float).sum())
            self._own_goal_rate_per90 = (total_og / total_mins * 90.0) if total_mins > 0 else 0.0
        else:
            self._own_goal_rate_per90 = 0.0

        # Defensive contribution exists only from 2025-26, so this is fitted on whatever rows
        # actually carry it rather than on the whole panel.
        dc_rows = played[played["defensive_contribution"].notna() & (played["position"] != "GKP")]
        if len(dc_rows) >= 500:
            threshold = dc_rows["position"].map(self.dc_threshold).fillna(12.0)
            dc_per_match = dc_rows["defensive_contribution"] / dc_rows["n_fixtures"].clip(lower=1)
            hit = (dc_per_match >= threshold).astype(int)
            if hit.nunique() > 1:
                m = _lgbm_classifier("binary", self.lgbm_params)
                m.fit(_design(dc_rows, self.columns), hit)
                self._parts["dc"] = m
                log.info(
                    "Defensive contribution fitted on %d rows, base hit rate %.1f%%",
                    len(dc_rows),
                    100 * hit.mean(),
                )

    # ------------------------------------------------------------------ prediction

    def _band_probabilities(self, X: pd.DataFrame, n: int) -> tuple[np.ndarray, np.ndarray]:
        """(P(1-59 mins), P(60+ mins)) per row, per match."""
        model = self._parts["band"]
        proba = model.predict_proba(X)
        classes = list(model.classes_)

        def p(b: int) -> np.ndarray:
            return proba[:, classes.index(b)] if b in classes else np.zeros(n)

        return p(BAND_SHORT), p(BAND_LONG)

    def expected_minutes(self, rows: pd.DataFrame) -> pd.Series:
        """Expected minutes in a single match, 0-90.

        Per match rather than per gameweek, because everything downstream that reads xMins is
        asking "is this player a starter", which a doubled figure would answer wrongly.
        """
        if "band" not in self._parts:
            return pd.Series(0.0, index=rows.index)
        X = _design(rows, self.columns)
        p_short, p_long = self._band_probabilities(X, len(rows))
        minutes = p_short * BAND_MINUTES[BAND_SHORT] + p_long * BAND_MINUTES[BAND_LONG]
        return pd.Series(np.clip(minutes, 0.0, 90.0), index=rows.index)

    def expected_xgi90(self, rows: pd.DataFrame) -> pd.Series:
        """Expected goals plus assists per 90, which the optimizer uses for role heuristics."""
        if "goals" not in self._parts:
            return pd.Series(0.0, index=rows.index)
        X = _design(rows, self.columns)
        goals = np.clip(np.asarray(self._parts["goals"].predict(X), dtype=float), 0, None)
        assists = np.clip(np.asarray(self._parts["assists"].predict(X), dtype=float), 0, None)
        return pd.Series(goals + assists, index=rows.index)

    @staticmethod
    def _expected_floor_div(rate: np.ndarray, divisor: int) -> np.ndarray:
        """E[floor(N / divisor)] for N ~ Poisson(rate).

        Needed because saves pay 1 point per 3 and goals conceded cost 1 per 2. Applying the
        divisor to the mean instead would misprice both, badly at low rates.
        """
        from scipy.stats import poisson

        rate = np.clip(np.asarray(rate, dtype=float), 0, None)
        total = np.zeros_like(rate)
        for k in range(1, COUNT_SUPPORT):
            total += (k // divisor) * poisson.pmf(k, rate)
        tail = 1.0 - poisson.cdf(COUNT_SUPPORT - 1, rate)
        total += ((COUNT_SUPPORT - 1) // divisor) * tail
        return total

    def predict(self, rows: pd.DataFrame) -> pd.Series:
        if "band" not in self._parts:
            return pd.Series(0.0, index=rows.index)

        X = _design(rows, self.columns)
        pos = rows["position"].astype(str)

        p_short, p_long = self._band_probabilities(X, len(rows))
        p_play = p_short + p_long
        exp_minutes = p_short * BAND_MINUTES[BAND_SHORT] + p_long * BAND_MINUTES[BAND_LONG]
        minutes_factor = exp_minutes / 90.0

        # Every term below is per match. A double gameweek pays out twice, and without this
        # multiplier the model caps out at one match's worth of points for the very gameweeks
        # where the largest scores happen.
        fixtures = rows["n_fixtures"].fillna(1).clip(lower=1).to_numpy(dtype=float)

        # Appearance
        total = p_short * 1.0 + p_long * 2.0

        def part(key: str) -> Optional[np.ndarray]:
            m = self._parts.get(key)
            return None if m is None else np.asarray(m.predict(X), dtype=float)

        def part_or_zeros(key: str) -> np.ndarray:
            # `part(key) or zeros` would raise: numpy arrays have no truth value.
            got = part(key)
            return np.zeros(len(rows)) if got is None else got

        # Attacking returns, as per-90 rates scaled by expected game time
        goals90 = np.clip(part_or_zeros("goals"), 0, None)
        assists90 = np.clip(part_or_zeros("assists"), 0, None)
        total += goals90 * minutes_factor * pos.map(self.goal_points).fillna(0.0).to_numpy()
        total += assists90 * minutes_factor * 3.0

        # Bonus and cards are per appearance, so they scale with the chance of playing
        bonus = part("bonus")
        if bonus is not None:
            total += np.clip(bonus, 0, None) * p_play
        cards = part("cards")
        if cards is not None:
            total -= np.clip(cards, 0, None) * p_play

        # Clean sheet, only from a 60+ appearance
        cs_model = self._parts.get("cs")
        if cs_model is not None:
            p_cs = cs_model.predict_proba(X)[:, 1]
            total += p_cs * p_long * pos.map(self.cs_points).fillna(0.0).to_numpy()

        # Goals conceded, scored per 2, keepers and defenders only
        conceded = part("conceded")
        if conceded is not None:
            penalty = self._expected_floor_div(conceded, 2)
            applies = pos.isin(self.conceded_positions).to_numpy().astype(float)
            total -= penalty * p_long * applies

        # Saves, scored per 3, keepers only
        saves90 = part("saves")
        if saves90 is not None:
            save_pts = self._expected_floor_div(np.clip(saves90, 0, None) * minutes_factor, 3)
            total += save_pts * (pos == "GKP").to_numpy().astype(float)

        # Defensive contribution, learned hit probability rather than an assumed one
        dc_model = self._parts.get("dc")
        if dc_model is not None:
            p_dc = dc_model.predict_proba(X)[:, 1]
            eligible = (pos != "GKP").to_numpy().astype(float)
            total += self.dc_points * p_dc * p_play * eligible

        # Penalty saves, keepers only
        pen_save90 = part("penalty_saves")
        if pen_save90 is not None:
            total += (
                np.clip(pen_save90, 0, None)
                * minutes_factor
                * self.penalty_save_points
                * (pos == "GKP").to_numpy().astype(float)
            )

        # Penalty misses — no "who takes penalties" feature exists, so this rides on whatever
        # the rate model infers from goals/minutes/role; small and mostly-zero for most players
        # is the correct answer here, not a bug.
        pen_miss90 = part("penalty_misses")
        if pen_miss90 is not None:
            total += np.clip(pen_miss90, 0, None) * minutes_factor * self.penalty_miss_points

        # Own goals: a single league-wide per-90 rate rather than a per-player model — see fit().
        if self._own_goal_rate_per90:
            total += (self._own_goal_rate_per90 / 90.0 * exp_minutes) * self.own_goal_points

        return pd.Series(total * fixtures, index=rows.index)


def candidate_models() -> List[object]:
    """The models this project is trying to make work, for the backtest to rank."""
    return [DirectPointsModel(), ComponentPointsModel()]


def _live_scoring_kwargs() -> Dict[str, object]:
    """ComponentPointsModel field overrides sourced from the live FPL rules.

    Without this, live prediction used ComponentPointsModel's hardcoded dataclass defaults
    unconditionally — every instantiation of the model ignored rules_fetcher.py's live-fetched
    scoring table entirely, even though that table is fetched fresh every `fpl update --run`
    and correctly keeps the README's scoring section current. The numbers happen to agree
    today, but a mid-season or new-season scoring change (FPL has changed goal/clean-sheet/DC
    values before) would silently diverge: the README would show the new rule, the model would
    keep scoring by the old one, with no warning anywhere that the two had split.

    Falls back to ComponentPointsModel's own defaults for anything the live fetch doesn't
    provide (e.g. `fetch_scoring_rules()` failed entirely, or a field genuinely isn't in the
    API and only the scraped/hardcoded fallback is available), rather than raising — a model
    with slightly-possibly-stale defaults is better than the pipeline crashing.
    """
    defaults = ComponentPointsModel()
    try:
        from ..data.rules_fetcher import fetch_scoring_rules

        rules = fetch_scoring_rules()
    except Exception as e:
        log.warning("Could not fetch live scoring rules (%s); using built-in defaults", e)
        return {}

    goals = rules.get("goals") or {}
    goal_points = {p: float(goals[p]) for p in ("GKP", "DEF", "MID", "FWD") if p in goals}

    # rules_from_bootstrap drops a position from `clean_sheet` entirely when its value is 0
    # (FWD normally is), so start from the model's own defaults and only overlay what the
    # live table actually reports.
    cs_points = dict(defaults.cs_points)
    cs_points.update({p: float(v) for p, v in (rules.get("clean_sheet") or {}).items()})

    dc = rules.get("defensive_contribution") or {}
    def_thr = dc.get("defender_threshold_cbit")
    mf_thr = dc.get("mid_fwd_threshold_cbirt")
    dc_threshold = dict(defaults.dc_threshold)
    if def_thr is not None:
        dc_threshold["DEF"] = float(def_thr)
    if mf_thr is not None:
        dc_threshold["MID"] = dc_threshold["FWD"] = float(mf_thr)

    kwargs: Dict[str, object] = {}
    if goal_points:
        kwargs["goal_points"] = {**defaults.goal_points, **goal_points}
    if cs_points:
        kwargs["cs_points"] = cs_points
    if dc_threshold:
        kwargs["dc_threshold"] = dc_threshold
    if dc.get("points") is not None:
        kwargs["dc_points"] = float(dc["points"])
    if rules.get("penalty_save") is not None:
        kwargs["penalty_save_points"] = float(rules["penalty_save"])
    if rules.get("penalty_miss") is not None:
        kwargs["penalty_miss_points"] = float(rules["penalty_miss"])
    if rules.get("own_goal") is not None:
        kwargs["own_goal_points"] = float(rules["own_goal"])

    return kwargs


def train_and_predict_gameweek(gw: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """Fit the component model on everything played and score the upcoming gameweek.

    Writes the two files the optimizer reads, in the schema it already expects:

    | File | Column | Meaning |
    |:-----|:-------|:--------|
    | `xmins.parquet` | `xmins` | expected minutes in a single match, 0-90 |
    | `exp_points.parquet` | `ep_adjusted` | expected points for the gameweek, minutes already included |
    | | `ep_model` | same figure, kept so both names resolve |
    | | `ep_blend` | same figure; nothing is blended in, see below |
    | | `ep_fpl` | the API's `ep_next`, carried for comparison only |
    | | `xgi90_est` | expected goals plus assists per 90 |

    `ep_blend` is not a blend here. The shipped pipeline mixed its own output with FPL's
    `ep_next` and calibrated to its median, which is why so much of its ranking was FPL's
    rather than its own. This model stands on its own and the column is retained only so
    existing readers keep working.
    """
    from ..data.fpl_api import get_bootstrap
    from ..utils.cache import PROC
    from ..utils.io import write_parquet
    from .live import build_live_panel, next_gameweek
    from .train import _team_attack_strength

    bootstrap = get_bootstrap()
    gw = gw or next_gameweek(bootstrap)
    train, score = build_live_panel(gw, bootstrap)

    model = ComponentPointsModel(**_live_scoring_kwargs())
    log.info("Fitting the component model on %d player-gameweek rows...", len(train))
    model.fit(train)

    ep = model.predict(score)
    xmins = model.expected_minutes(score)
    xgi90 = model.expected_xgi90(score)

    ep_next = {int(e["id"]): float(e.get("ep_next") or 0.0) for e in bootstrap.get("elements", [])}
    team_att = _team_attack_strength(bootstrap)
    team_by_id = {int(e["id"]): int(e["team"]) for e in bootstrap.get("elements", [])}

    player_id = score["fpl_id"].astype(int)
    xmins_df = pd.DataFrame({"player_id": player_id.values, "xmins": xmins.values})
    ep_df = pd.DataFrame(
        {
            "player_id": player_id.values,
            "ep_model": ep.values,
            "ep_fpl": player_id.map(ep_next).fillna(0.0).values,
            "ep_blend": ep.values,
            "ep_adjusted": ep.values,
            "xgi90_est": xgi90.values,
            "team_att": player_id.map(
                lambda pid: team_att.get(team_by_id.get(pid, -1), 1.0)
            ).values,
        }
    )

    write_parquet(xmins_df, PROC / "xmins.parquet")
    write_parquet(ep_df, PROC / "exp_points.parquet")

    q = ep_df["ep_adjusted"].quantile([0.5, 0.9, 0.99]).to_dict()
    log.info(
        "GW%d component-model EP: p50=%.2f p90=%.2f p99=%.2f max=%.2f (%d players)",
        gw,
        q[0.5],
        q[0.9],
        q[0.99],
        ep_df["ep_adjusted"].max(),
        len(ep_df),
    )
    by_pos = (
        score.assign(ep=ep.values)
        .groupby("position")["ep"]
        .agg(["mean", "max"])
        .round(2)
        .to_dict("index")
    )
    log.info("Mean/max EP by position: %s", by_pos)
    return {"exp_points": ep_df, "xmins": xmins_df, "scored": score.assign(ep=ep.values)}
