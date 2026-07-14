import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from src.data_loader import DataLoader

LOGGER = logging.getLogger(__name__)


class FeaturePipeline:
    """Constructs model-ready feature sets from raw macro-finance inputs."""

    DEFAULT_OUTPUT_FILENAME = "macro_finance_engineered.csv"
    REQUIRED_COLUMNS = DataLoader.DEFAULT_FRED_SERIES + [DataLoader.SP500_LABEL]

    def __init__(
        self,
        api_key: Optional[str] = None,
        output_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError("FRED_API_KEY must be provided via constructor or environment variable.")

        base_path = Path(__file__).resolve().parent.parent
        self.output_dir = Path(output_dir) if output_dir else base_path / "data" / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.loader = DataLoader(api_key=self.api_key)
        self.logger = logger or LOGGER

    def processed_file_path(self) -> Path:
        return self.output_dir / self.DEFAULT_OUTPUT_FILENAME

    def load_processed_dataset(self) -> pd.DataFrame:
        path = self.processed_file_path()
        if not path.exists():
            raise FileNotFoundError(f"Processed dataset not found at {path}")
        return pd.read_csv(path, parse_dates=["date"]).set_index("date")

    def _validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise KeyError(f"Processed raw dataframe is missing required columns: {missing}")
        return df

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._validate_dataframe(df).copy()
        df = df.sort_index()

        # Base returns and log price
        df["SP500_log"] = np.log(df["SP500"])
        df["SP500_ret"] = df["SP500_log"].diff()

        # Target: next period return, aligned with current feature values
        df["target_next_ret"] = df["SP500_ret"].shift(-1)

        # Trend and momentum (1m momentum is identical to SP500_ret, so omitted)
        df["SP500_mom_3m"] = df["SP500_log"].diff(3)
        df["SP500_mom_6m"] = df["SP500_log"].diff(6)
        df["SP500_mom_12m"] = df["SP500_log"].diff(12)

        # Volatility
        df["SP500_vol_3m"] = df["SP500_ret"].rolling(window=3, min_periods=2).std()
        df["SP500_vol_6m"] = df["SP500_ret"].rolling(window=6, min_periods=4).std()
        df["SP500_vol_12m"] = df["SP500_ret"].rolling(window=12, min_periods=8).std()

        # Macro dynamics
        df["FEDFUNDS_diff"] = df["FEDFUNDS"].diff()
        df["CPI_inflation_mom"] = df["CPIAUCSL"].pct_change()
        df["CPI_inflation_yoy"] = df["CPIAUCSL"].pct_change(12)
        df["CPI_acceleration"] = df["CPI_inflation_mom"].diff()
        df["UNRATE_diff"] = df["UNRATE"].diff()
        df["T10Y2Y_diff"] = df["T10Y2Y"].diff()

        # Inflation gap proxy (yield curve level is already captured via T10Y2Y_diff)
        df["real_rate_proxy"] = df["FEDFUNDS"] - df["CPI_inflation_yoy"]

        # Lagged macro signals to prevent leakage
        lag_columns = [
            "FEDFUNDS_diff",
            "CPI_inflation_mom",
            "CPI_inflation_yoy",
            "CPI_acceleration",
            "UNRATE_diff",
            "T10Y2Y_diff",
            "real_rate_proxy",
        ]
        for col in lag_columns:
            for lag in [1, 2, 3]:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)

        df = df.dropna()
        return df

    def run_pipeline(self, start_date: str = "2000-01-01") -> pd.DataFrame:
        self.logger.info("Starting pipeline run with start_date=%s", start_date)
        raw_df = self.loader.build_raw_dataset(start_date=start_date)
        processed_df = self._compute_features(raw_df)

        path = self.processed_file_path()
        processed_df.to_csv(path, index=True)
        self.logger.info("Processed dataset saved to %s", path)
        return processed_df

    def refresh(self, start_date: str = "2000-01-01") -> pd.DataFrame:
        """Refresh raw and engineered datasets from the latest source data."""
        return self.run_pipeline(start_date=start_date)

