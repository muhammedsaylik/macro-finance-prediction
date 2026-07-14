import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional, List
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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
        }
        self.trained_models: Dict[str, Any] = {}

    def _metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            "R2": r2_score(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE": mean_absolute_error(y_true, y_pred),
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
            }

            if name == "RidgeCV":
                row["Alpha"] = float(model.alpha_)
            if name == "ElasticNetCV":
                row["Alpha"] = float(model.alpha_)
                row["L1_ratio"] = float(model.l1_ratio_)

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
