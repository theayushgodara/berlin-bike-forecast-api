"""Daily weather for Berlin from Open-Meteo (free, no API key).

Historical data is used for training; forecast data is used by the API.

Run:  python -m bikeforecast.weather --start 2012-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd
import requests

from bikeforecast import config

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def _to_frame(payload: dict) -> pd.DataFrame:
    df = pd.DataFrame(payload["daily"]).rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    # Sunshine comes in seconds; hours are easier to read.
    df["sunshine_duration"] = df["sunshine_duration"] / 3600
    return df


def _params(**extra) -> dict:
    return {
        "latitude": config.BERLIN_LAT,
        "longitude": config.BERLIN_LON,
        "daily": ",".join(config.WEATHER_VARS),
        "timezone": config.TIMEZONE,
        **extra,
    }


def fetch_history(start: date, end: date) -> pd.DataFrame:
    resp = requests.get(
        ARCHIVE_URL,
        params=_params(start_date=start.isoformat(), end_date=end.isoformat()),
        timeout=120,
    )
    resp.raise_for_status()
    return _to_frame(resp.json())


def fetch_forecast(day: date) -> dict:
    """Return the weather forecast for a single day as a dict of floats."""
    resp = requests.get(
        FORECAST_URL,
        params=_params(start_date=day.isoformat(), end_date=day.isoformat()),
        timeout=30,
    )
    resp.raise_for_status()
    row = _to_frame(resp.json()).iloc[0]
    return {var: float(row[var]) for var in config.WEATHER_VARS}


def load_weather() -> pd.DataFrame:
    return pd.read_csv(config.WEATHER_FILE, parse_dates=["date"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2012, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    args = parser.parse_args()
    df = fetch_history(args.start, args.end)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.WEATHER_FILE, index=False)
    print(f"Wrote {len(df):,} days of weather to {config.WEATHER_FILE}")


if __name__ == "__main__":
    main()
