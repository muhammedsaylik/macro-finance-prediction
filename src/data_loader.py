import os
import ssl
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

ssl._create_default_https_context = ssl._create_unverified_context
load_dotenv()

class DataLoader:
    def __init__(self, api_key: str, data_dir: str = "data"):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.raw_dir = os.path.join(base_path, data_dir, "raw")
        os.makedirs(self.raw_dir, exist_ok=True)

    def fetch_series(self, series_id: str, start_date: str = "2000-01-01") -> pd.DataFrame:
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date
        }
        
        response = requests.get(self.base_url, params=params)
        if response.status_code != 200:
            raise Exception(f"FRED API Error! Code: {response.status_code}, Message: {response.text}")
            
        data = response.json()
        df = pd.DataFrame(data["observations"])
        df = df[["date", "value"]]
        
        df["value"] = df["value"].replace(".", np.nan)
        df["date"] = pd.to_datetime(df["date"])
        df[series_id] = df["value"].astype(float)
        
        return df[["date", series_id]].set_index("date")

    def fetch_multiple_fred(self, series_list: list, start_date: str = "2000-01-01") -> pd.DataFrame:
        dfs = []
        for s_id in series_list:
            print(f"Downloading FRED series: {s_id}...")
            df_temp = self.fetch_series(s_id, start_date)
            dfs.append(df_temp)
        
        return pd.concat(dfs, axis=1, join="outer")

    def fetch_sp500_daily(self, start_date: str = "2000-01-01") -> pd.DataFrame:
        print("Fetching S&P 500 data directly from Yahoo Finance API...")
        
        period1 = int(pd.to_datetime(start_date).timestamp())
        period2 = int(pd.Timestamp.now().timestamp())
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/^GSPC?period1={period1}&period2={period2}&interval=1d"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        try:
            response = requests.get(url, headers=headers, verify=False)
            if response.status_code != 200:
                raise Exception(f"Yahoo API Did Not Respond! Code: {response.status_code}")
                
            data = response.json()
            chart_data = data["chart"]["result"][0]
            timestamps = chart_data["timestamp"]
            closes = chart_data["indicators"]["quote"][0]["close"]
            
            df_spy = pd.DataFrame({
                "date": pd.to_datetime(timestamps, unit="s"),
                "SP500": closes
            }).set_index("date")
            df_spy.index = df_spy.index.tz_localize(None)
            
            return df_spy.dropna()
            
        except Exception as e:
            raise Exception(f"S&P 500 fetch failed! Details: {e}")
    
    def build_macro_dataset(self, start_date: str = "2000-01-01") -> pd.DataFrame:
        fred_ids = ["FEDFUNDS", "CPIAUCSL", "T10Y2Y", "UNRATE"]
        
        df_fred = self.fetch_multiple_fred(fred_ids, start_date)
        df_sp500 = self.fetch_sp500_daily(start_date)
        
        df_all = pd.concat([df_fred, df_sp500], axis=1, join="outer")
        df_monthly = df_all.resample("MS").last()
        df_monthly = df_monthly.ffill().dropna()
        
        raw_path = os.path.join(self.raw_dir, "macro_finance_monthly.csv")
        df_monthly.to_csv(raw_path)
        print(f"[SUCCESS] Saved raw dataset: {raw_path}")
        
        return df_monthly
   