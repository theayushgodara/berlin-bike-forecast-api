import pandas as pd

from bikeforecast.data import hourly_sheet_to_long, hourly_to_daily
from bikeforecast.features import FEATURES, add_calendar_features, build_feature_frame
from bikeforecast import config


def test_calendar_features_holidays_and_bridge_days():
    df = pd.DataFrame({"date": pd.to_datetime([
        "2025-05-29",  # Thursday, Ascension Day (holiday)
        "2025-05-30",  # Friday after Ascension -> bridge day
        "2025-03-08",  # Saturday, International Women's Day (Berlin holiday)
        "2025-06-03",  # ordinary Tuesday
    ])})
    out = add_calendar_features(df)
    assert out["is_holiday"].tolist() == [1, 0, 1, 0]
    assert out["is_bridge_day"].tolist() == [0, 1, 0, 0]
    assert out["is_weekend"].tolist() == [0, 0, 1, 0]
    assert out["day_of_week"].tolist() == [3, 4, 5, 1]


def test_build_feature_frame_columns():
    row = {"station_id": "A", "date": pd.Timestamp("2025-06-03"),
           **{v: 1.0 for v in config.WEATHER_VARS}}
    X = build_feature_frame(pd.DataFrame([row]), {"A": 0})
    assert list(X.columns) == FEATURES
    assert X["station_code"].iloc[0] == 0


def test_hourly_to_daily_drops_incomplete_and_zero_days():
    hours = pd.date_range("2025-01-01", periods=72, freq="h")
    sheet = pd.DataFrame({
        "Zählstelle        Inbetriebnahme": hours,
        "A-1\n01.01.2020": [10] * 24 + [0] * 24 + [5] * 24,   # day 2 is all zeros
        "B-2 01.01.2020": [1] * 24 + [None] * 24 + [2] * 24,  # day 2 missing; old header style
    })
    daily = hourly_to_daily(hourly_sheet_to_long(sheet))
    assert set(daily["station_id"]) == {"A-1", "B-2"}
    a = daily[daily["station_id"] == "A-1"]
    assert a["cyclists"].tolist() == [240, 120]
    b = daily[daily["station_id"] == "B-2"]
    assert b["cyclists"].tolist() == [24, 48]
