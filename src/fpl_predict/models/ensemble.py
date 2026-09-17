from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, PoissonRegressor
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin

from ..utils.logging import get_logger

log = get_logger(__name__)


class EnsembleModel(BaseEstimator, RegressorMixin):
    """
    Ensemble model that combines multiple algorithms with weighted averaging.
    Includes automatic weight optimization based on cross-validation performance.
    """

    def __init__(
        self, use_models: List[str] = None, cv_folds: int = 5, optimize_weights: bool = True
    ):
        """
        Args:
            use_models: List of model types to use. Options:
                ['xgboost', 'lightgbm', 'random_forest', 'gradient_boost', 'ridge', 'poisson']
            cv_folds: Number of cross-validation folds for weight optimization
            optimize_weights: Whether to optimize ensemble weights
        """
        if use_models is None:
            use_models = ["xgboost", "lightgbm", "random_forest", "poisson"]

        self.use_models = use_models
        self.cv_folds = cv_folds
        self.optimize_weights = optimize_weights
        self.models = {}
        self.weights = {}
        self.fitted = False

    def _create_models(self) -> Dict[str, BaseEstimator]:
        """Create instances of selected models."""
        model_dict = {}

        if "xgboost" in self.use_models:
            model_dict["xgboost"] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="count:poisson",
                random_state=42,
                verbosity=0,
            )

        if "lightgbm" in self.use_models:
            model_dict["lightgbm"] = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="poisson",
                random_state=42,
                verbosity=-1,
            )

        if "random_forest" in self.use_models:
            model_dict["random_forest"] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
            )

        if "gradient_boost" in self.use_models:
            model_dict["gradient_boost"] = GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8, random_state=42
            )

        if "ridge" in self.use_models:
            model_dict["ridge"] = Ridge(alpha=1.0, random_state=42)

        if "poisson" in self.use_models:
            model_dict["poisson"] = PoissonRegressor(alpha=0.1, max_iter=1000)

        return model_dict

    def _optimize_ensemble_weights(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Optimize ensemble weights using cross-validation performance.
        """
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        model_scores = {name: [] for name in self.models.keys()}

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            for name, model in self.models.items():
                # Clone and fit model
                model_clone = model.__class__(**model.get_params())
                model_clone.fit(X_train, y_train)

                # Predict and calculate MAE
                preds = model_clone.predict(X_val)
                mae = np.mean(np.abs(y_val - preds))
                model_scores[name].append(mae)

        # Calculate average scores
        avg_scores = {name: np.mean(scores) for name, scores in model_scores.items()}

        # Convert to weights (inverse of error, normalized)
        inverse_scores = {name: 1.0 / (score + 1e-10) for name, score in avg_scores.items()}
        total_weight = sum(inverse_scores.values())
        weights = {name: w / total_weight for name, w in inverse_scores.items()}

        log.info(f"Optimized ensemble weights: {weights}")
        return weights

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit all ensemble models, dropping any that fail."""
        if len(X) == 0:
            return

        self.models = self._create_models()

        # Collect failures and delete afterwards. Deleting inside the loop raised
        # "dictionary changed size during iteration", which aborted the whole fit as soon as
        # any single model failed, so one model's problem took down the ensemble.
        failed: List[str] = []
        for name, model in self.models.items():
            try:
                model.fit(X, y)
                log.debug(f"Fitted {name} model")
            except Exception as e:
                log.warning(f"Failed to fit {name}: {e}")
                failed.append(name)
        for name in failed:
            del self.models[name]

        if not self.models:
            log.warning("No ensemble model could be fitted; predictions will fall back to zero")
            self.weights = {}
            self.fitted = False
            return

        if self.optimize_weights and len(self.models) > 1:
            self.weights = self._optimize_ensemble_weights(X, y)
        else:
            self.weights = {name: 1.0 / len(self.models) for name in self.models}

        self.fitted = True
        log.info("Ensemble fitted with models: %s", list(self.models))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions."""
        if not self.fitted:
            return np.zeros(len(X))

        predictions = []
        weights = []

        for name, model in self.models.items():
            try:
                preds = model.predict(X)
                predictions.append(preds)
                weights.append(self.weights[name])
            except Exception as e:
                log.warning(f"Prediction failed for {name}: {e}")

        if not predictions:
            return np.zeros(len(X))

        # Weighted average
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Renormalize

        return np.average(predictions, axis=0, weights=weights)

    def predict_with_std(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with standard deviation across models.
        Useful for uncertainty estimation.
        """
        if not self.fitted:
            return np.zeros(len(X)), np.ones(len(X)) * 0.1

        predictions = []

        for name, model in self.models.items():
            try:
                preds = model.predict(X)
                predictions.append(preds)
            except Exception:
                pass

        if not predictions:
            return np.zeros(len(X)), np.ones(len(X)) * 0.1

        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)

        return mean_pred, std_pred
