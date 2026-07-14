import logging
import warnings
from typing import Dict, Tuple, Any, Optional, List

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

logging.getLogger("lightgbm").setLevel(logging.ERROR)


class MacroFinanceModels:
    """Quantitative regression framework for macro-finance forecasting."""

    def __init__(self, random_state: int = 42, cv_splits: int = 5):
        self.random_state = random_state
        self.cv = TimeSeriesSplit(n_splits=cv_splits)
        self.models: Dict[str, Any] = {
            "Baseline": DummyRegressor(strategy="mean"),
            "OLS": LinearRegression(),
            "RidgeCV": RidgeCV(
                alphas=np.logspace(-2, 8, 80),
                cv=self.cv,
            ),
            "ElasticNetCV": ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.9],
                alphas=np.logspace(-4, 1, 40),
                cv=self.cv,
                max_iter=5000,
                n_jobs=-1,
                random_state=self.random_state,
            ),
            "MLP": MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=self.random_state,
            ),
            "XGBoost": GridSearchCV(
                estimator=XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=self.random_state,
                    n_jobs=1,
                ),
                param_grid={
                    "max_depth": [2, 3],
                    "learning_rate": [0.03, 0.05],
                    "n_estimators": [200, 300],
                    "subsample": [0.8],
                    "colsample_bytree": [0.8],
                    "reg_alpha": [0.0, 0.1],
                    "reg_lambda": [1.0, 2.0],
                },
                cv=self.cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=1,
            ),
            "LightGBM": GridSearchCV(
                estimator=LGBMRegressor(
                    objective="regression",
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=self.random_state,
                    n_jobs=1,
                    verbosity=-1,
                ),
                param_grid={
                    "max_depth": [2, 3],
                    "learning_rate": [0.03, 0.05],
                    "n_estimators": [200, 300],
                    "subsample": [0.8],
                    "colsample_bytree": [0.8],
                    "reg_alpha": [0.0, 0.1],
                    "reg_lambda": [1.0, 2.0],
                },
                cv=self.cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=1,
            ),
        }
        self.trained_models: Dict[str, Any] = {}

    def _metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        directional_accuracy = float(np.mean(np.sign(y_true_arr) == np.sign(y_pred_arr)) * 100.0)

        return {
            "R2": float(r2_score(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "Directional Accuracy": directional_accuracy,
        }

    def fit_and_evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        model_names: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fit selected models and return evaluation metrics plus out-of-sample predictions."""
        metrics = []
        predictions_df = pd.DataFrame(index=y_test.index)
        predictions_df["Actual"] = y_test.values

        for name, model in self.models.items():
            if model_names and name not in model_names:
                continue

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", message="No further splits with positive gain.*")
                model.fit(X_train, y_train)
            self.trained_models[name] = model

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            predictions_df[f"{name}_Pred"] = test_pred

            train_metrics = self._metrics(y_train, train_pred)
            test_metrics = self._metrics(y_test, test_pred)

            row = {
                "Model": name,
                "Train_R2": train_metrics["R2"],
                "Test_R2": test_metrics["R2"],
                "Train_RMSE": train_metrics["RMSE"],
                "Test_RMSE": test_metrics["RMSE"],
                "Train_MAE": train_metrics["MAE"],
                "Test_MAE": test_metrics["MAE"],
                "Train_Directional_Accuracy": train_metrics["Directional Accuracy"],
                "Test_Directional_Accuracy": test_metrics["Directional Accuracy"],
            }

            if name == "RidgeCV":
                row["Alpha"] = float(model.alpha_)
            if name == "ElasticNetCV":
                row["Alpha"] = float(model.alpha_)
                row["L1_ratio"] = float(model.l1_ratio_)
            if name in {"XGBoost", "LightGBM"}:
                row["Best_Params"] = str(model.best_params_)

            metrics.append(row)

        metrics_df = pd.DataFrame(metrics)
        metrics_df = metrics_df.sort_values(by="Test_R2", ascending=False).reset_index(drop=True)
        return metrics_df, predictions_df

    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Return predictions for a trained model name."""
        model = self.trained_models.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' is not trained yet.")
        return model.predict(X)

    def get_model(self, name: str) -> Any:
        """Return a trained model instance by name."""
        model = self.trained_models.get(name)
        if model is None:
            raise ValueError(f"Model '{name}' has not been trained.")
        return model

    def get_estimator(self, name: str) -> Any:
        """Return the underlying estimator for a fitted model, including GridSearchCV results."""
        model = self.get_model(name)
        return getattr(model, "best_estimator_", model)
