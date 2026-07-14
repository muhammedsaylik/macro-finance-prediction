import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

LOGGER = logging.getLogger(__name__)


class DataLoader:
    """Responsible for raw data ingestion and persistently storing downstream-ready CSVs."""

    DEFAULT_FRED_SERIES = ["FEDFUNDS", "CPIAUCSL", "T10Y2Y", "UNRATE"]
    RAW_FILENAME = "macro_finance_monthly.csv"
    FRED_ENDPOINT = "https://api.stlouisfed.org/fred/series/observations"
    SP500_LABEL = "SP500"

    def __init__(
        self,
        api_key: Optional[str] = None,
        data_dir: str = "data",
        logger: Optional[logging.Logger] = None,
    ):
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError("FRED_API_KEY must be provided via constructor or environment variable.")

        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.base_dir = os.path.join(base_path, data_dir)
        self.raw_dir = os.path.join(self.base_dir, "raw")
        os.makedirs(self.raw_dir, exist_ok=True)

        self.session = self._create_session()
        self.logger = logger or LOGGER

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            backoff_factor=0.5,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        return session

    def _validate_response(self, response: requests.Response) -> Dict:
        response.raise_for_status()
        payload = response.json()
        if "observations" not in payload:
            raise ValueError("Unexpected FRED response schema.")
        return payload

    def fetch_series(self, series_id: str, start_date: str = "2000-01-01") -> pd.DataFrame:
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date,
        }
        self.logger.debug("Fetching FRED series %s from %s", series_id, start_date)
        response = self.session.get(self.FRED_ENDPOINT, params=params, timeout=20)
        payload = self._validate_response(response)

        df = pd.DataFrame(payload["observations"])
        df = df[["date", "value"]].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"].replace(".", np.nan), errors="coerce")
        df = df.rename(columns={"value": series_id}).set_index("date")

        if df.index.hasnans:
            raise ValueError(f"Invalid dates while fetching {series_id}")

        return df

    def fetch_multiple_fred(
        self,
        series_list: Optional[List[str]] = None,
        start_date: str = "2000-01-01",
        series_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        # Accept both old and new keyword names for backward compatibility.
        series_list = series_list or series_ids or self.DEFAULT_FRED_SERIES
        self.logger.info("Downloading %d FRED series", len(series_list))
        frames = [self.fetch_series(series_id, start_date) for series_id in series_list]
        return pd.concat(frames, axis=1, join="outer")

    def fetch_sp500_daily(self, start_date: str = "2000-01-01") -> pd.DataFrame:
        self.logger.info("Fetching S&P 500 daily history from Yahoo Finance")
        period1 = int(pd.to_datetime(start_date).timestamp())
        period2 = int(pd.Timestamp.now().timestamp())
        url = (
            "https://query1.finance.yahoo.com/v8/finance/chart/^GSPC"
            f"?period1={period1}&period2={period2}&interval=1d"
        )
        headers = {"User-Agent": "Mozilla/5.0"}
        response = self.session.get(url, headers=headers, timeout=20)
        response.raise_for_status()

        payload = response.json()
        chart_data = payload.get("chart", {}).get("result")
        if not chart_data:
            raise ValueError("Unexpected Yahoo Finance response schema.")

        chart_data = chart_data[0]
        timestamps = chart_data["timestamp"]
        closes = chart_data["indicators"]["quote"][0]["close"]
        df = pd.DataFrame({"date": pd.to_datetime(timestamps, unit="s"), self.SP500_LABEL: closes})
        df = df.dropna(subset=[self.SP500_LABEL]).set_index("date")
        df.index = df.index.tz_localize(None)
        return df

    def raw_file_path(self) -> str:
        return os.path.join(self.raw_dir, self.RAW_FILENAME)

    def load_raw_dataset(self) -> pd.DataFrame:
        raw_path = self.raw_file_path()
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw dataset not found at {raw_path}")
        return pd.read_csv(raw_path, parse_dates=["date"]).set_index("date")

    def build_raw_dataset(
        self,
        start_date: str = "2000-01-01",
        series_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        df_fred = self.fetch_multiple_fred(series_ids=series_ids, start_date=start_date)
        df_sp500 = self.fetch_sp500_daily(start_date=start_date)

        df_all = pd.concat([df_fred, df_sp500], axis=1, join="outer")
        df_monthly = df_all.resample("MS").last()
        df_monthly = df_monthly.sort_index().ffill()

        raw_path = self.raw_file_path()
        df_monthly.to_csv(raw_path, index=True)
        self.logger.info("Raw dataset refreshed and saved to %s", raw_path)
        return df_monthly

    def build_macro_dataset(self, start_date: str = "2000-01-01", series_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """Backward-compatible alias for older notebook names and API references."""
        return self.build_raw_dataset(start_date=start_date, series_ids=series_ids)
   