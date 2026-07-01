from __future__ import annotations

import json
import urllib.request
from typing import Optional

import pandas as pd

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
DBNOMICS_API = "https://api.db.nomics.world/v22/series/{provider}/{dataset}/{series}?observations=1"
SP500_CSV = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"


def fetch_fred_series(series_id: str, start: str = "1990-01-01") -> pd.Series:
    url = FRED_CSV.format(series_id=series_id)
    df = pd.read_csv(url, na_values=["."])
    date_col, val_col = df.columns[0], df.columns[1]
    s = pd.Series(
        pd.to_numeric(df[val_col], errors="coerce").values,
        index=pd.to_datetime(df[date_col]),
        name=series_id,
    ).dropna()
    s = s[s.index >= pd.Timestamp(start)]
    if s.empty:
        raise ValueError(f"FRED returned no data for {series_id} since {start}")
    return s


def fetch_dbnomics_series(
    provider: str,
    dataset: str,
    series: str,
    min_valid: Optional[float] = None,
    max_valid: Optional[float] = None,
) -> pd.Series:

    url = DBNOMICS_API.format(provider=provider, dataset=dataset, series=series)
    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    doc = payload["series"]["docs"][0]
    s = pd.Series(
        pd.to_numeric(pd.Series(doc["value"]), errors="coerce").values,
        index=pd.to_datetime(doc["period_start_day"]),
        name=f"{provider}/{dataset}/{series}",
    ).dropna()
    if min_valid is not None:
        s = s[s >= min_valid]
    if max_valid is not None:
        s = s[s <= max_valid]
    if s.empty:
        raise ValueError(f"DBnomics returned no valid data for {provider}/{dataset}/{series}")
    return s


def fetch_sp500_monthly(start: str = "1990-01-01") -> pd.Series:
    df = pd.read_csv(SP500_CSV)
    if "Date" not in df.columns or "SP500" not in df.columns:
        raise ValueError(f"Unexpected S&P CSV layout: {list(df.columns)}")
    s = pd.Series(
        pd.to_numeric(df["SP500"], errors="coerce").values,
        index=pd.to_datetime(df["Date"]),
        name="SP500",
    ).dropna().sort_index()
    s = s[s.index >= pd.Timestamp(start)]
    if s.empty:
        raise ValueError(f"No S&P 500 data since {start}")
    return s
