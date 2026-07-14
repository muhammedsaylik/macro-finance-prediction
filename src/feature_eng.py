import os
import pandas as pd
from typing import Optional, Tuple, List
from sklearn.preprocessing import StandardScaler

# High-VIF macro bases excluded from the modeling feature set (single source of truth).
REDUNDANT_MACRO_BASES = [
    "real_rate_proxy",
    "CPI_acceleration",
    "CPI_inflation_yoy",
    "FEDFUNDS_diff",
]


class FeatureEngineer:
    """Prepares model-ready datasets from engineered macro-finance features."""

    def __init__(
        self,
        processed_data_path: Optional[str] = None,
        target_column: str = "target_next_ret",
    ):
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.data_path = (
            processed_data_path
            if processed_data_path
            else os.path.join(base_path, "data", "processed", "macro_finance_engineered.csv")
        )
        self.target_column = target_column
        self.inf_producing_bases = list(REDUNDANT_MACRO_BASES)
        self.default_excluded_columns = [
            "FEDFUNDS",
            "CPIAUCSL",
            "T10Y2Y",
            "UNRATE",
            "SP500",
            "SP500_log",
            self.target_column,
        ]

    def _get_redundant_columns(self, df: pd.DataFrame) -> List[str]:
        """Dynamically identifies all redundant base columns and their lagged variants."""
        redundant = []
        for base in self.inf_producing_bases:
            if base in df.columns:
                redundant.append(base)
            for lag in [1, 2, 3]:
                lag_col = f"{base}_lag{lag}"
                if lag_col in df.columns:
                    redundant.append(lag_col)
        return redundant

    def load_processed_data(self) -> pd.DataFrame:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Processed dataset not found at: {self.data_path}")
        return pd.read_csv(self.data_path, index_col=0, parse_dates=True).sort_index()

    def get_feature_columns(self, df: pd.DataFrame, exclude_raw_levels: bool = True) -> List[str]:
        redundant_cols = self._get_redundant_columns(df)
        excluded_cols = self.default_excluded_columns.copy()
        if exclude_raw_levels:
            excluded_cols += redundant_cols
        return [col for col in df.columns if col not in excluded_cols]

    def prepare_modeling_data(
        self,
        test_size: float = 0.2,
        exclude_raw_levels: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], StandardScaler]:
        """Loads processed data, removes redundant features, splits by time, and scales train/test sets."""
        df = self.load_processed_data()
        feature_cols = self.get_feature_columns(df, exclude_raw_levels=exclude_raw_levels)

        if not feature_cols:
            raise ValueError("No features remain after exclusion and multicollinearity filtering.")

        X = df[feature_cols]
        y = df[self.target_column]

        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

        return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler
