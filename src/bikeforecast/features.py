"""Feature engineering shared by training and the API.

Only features known in advance are used (calendar + weather forecast), so the
model can predict any day for which a weather forecast exists.
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from bikeforecast import config

CALENDAR_FEATURES = [
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "is_bridge_day",
    "month",
    "doy_sin",
    "doy_cos",
    "year",
]
FEATURES = ["station_code", *CALENDAR_FEATURES, *config.WEATHER_VARS]


def _berlin_holidays(years) -> set:
    return set(holidays.Germany(subdiv="BE", years=list(years)).keys())


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar columns derived from the `date` column."""
    df = df.copy()
    dates = pd.to_datetime(df["date"])
    hol = _berlin_holidays(range(dates.dt.year.min() - 1, dates.dt.year.max() + 2))
    day = dates.dt.date

    df["day_of_week"] = dates.dt.day_of_week
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday"] = day.isin(hol).astype(int)
    # A workday squeezed between a holiday and a weekend, e.g. Friday after
    # Ascension Day. Many people take it off.
    yesterday = dates - pd.Timedelta(days=1)
    tomorrow = dates + pd.Timedelta(days=1)
    is_workday = (df["is_weekend"] == 0) & (df["is_holiday"] == 0)
    holiday_then_weekend = yesterday.dt.date.isin(hol) & (tomorrow.dt.day_of_week >= 5)
    weekend_then_holiday = (yesterday.dt.day_of_week >= 5) & tomorrow.dt.date.isin(hol)
    df["is_bridge_day"] = (is_workday & (holiday_then_weekend | weekend_then_holiday)).astype(int)
    df["month"] = dates.dt.month
    doy = dates.dt.day_of_year
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["year"] = dates.dt.year
    return df


def build_feature_frame(
    df: pd.DataFrame, station_codes: dict[str, int]
) -> pd.DataFrame:
    """Return the model input matrix for rows with `station_id`, `date` and weather."""
    df = add_calendar_features(df)
    df["station_code"] = df["station_id"].map(station_codes)
    return df[FEATURES]
