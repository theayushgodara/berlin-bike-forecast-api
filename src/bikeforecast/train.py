"""Train and evaluate the forecasting model.

One global gradient-boosting model covers all stations (the station is a
categorical feature). The target is log(1 + cyclists), which keeps busy
stations from dominating the loss.

Evaluation: train on every year before the test year, test on the test year,
and compare against a naive baseline (average for that station + weekday).

Run:  python -m bikeforecast.train
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from bikeforecast import config
from bikeforecast.data import load_daily_counts, load_stations
from bikeforecast.features import FEATURES, build_feature_frame
from bikeforecast.weather import load_weather


def make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.1,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        categorical_features=[FEATURES.index("station_code")],
        random_state=42,
    )


def drop_counter_faults(counts: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    """Remove days far below the station's usual level (partial counter outages).

    A day is dropped when it has less than `threshold` x the station's
    centred 29-day rolling median. Real low days (heavy snow, Christmas) are
    rarely below 10% of normal, while broken counters often are.
    """
    counts = counts.sort_values(["station_id", "date"])
    typical = counts.groupby("station_id")["cyclists"].transform(
        lambda s: s.rolling(29, center=True, min_periods=7).median()
    )
    keep = typical.isna() | (counts["cyclists"] >= threshold * typical)
    return counts[keep]


def prepare(counts: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    counts = drop_counter_faults(counts.dropna(subset=["cyclists"]))
    return counts.merge(weather, on="date", how="inner")


def mape(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(y_true, 1)) * 100)


def evaluate(df: pd.DataFrame, station_codes: dict[str, int], test_year: int) -> dict:
    train = df[df["date"].dt.year < test_year]
    test = df[df["date"].dt.year == test_year]
    # Only score stations the model has seen in training.
    test = test[test["station_id"].isin(train["station_id"].unique())]

    model = make_model()
    model.fit(build_feature_frame(train, station_codes), np.log1p(train["cyclists"]))
    pred = np.expm1(model.predict(build_feature_frame(test, station_codes)))

    # Baseline: mean count for this station and weekday over the last 2 training years.
    recent = train[train["date"].dt.year >= test_year - 2]
    base = recent.assign(dow=recent["date"].dt.day_of_week).groupby(["station_id", "dow"])["cyclists"].mean()
    base_pred = test.assign(dow=test["date"].dt.day_of_week).set_index(["station_id", "dow"]).index.map(base)
    base_pred = np.asarray(base_pred, dtype=float)
    ok = ~np.isnan(base_pred)

    y = test["cyclists"].to_numpy()
    return {
        "test_year": test_year,
        "test_rows": int(len(test)),
        "model_mae": round(mean_absolute_error(y, pred), 1),
        "model_mape": round(mape(y, pred), 1),
        "baseline_mae": round(mean_absolute_error(y[ok], base_pred[ok]), 1),
        "baseline_mape": round(mape(y[ok], base_pred[ok]), 1),
    }


def train(test_year: int) -> dict:
    counts = load_daily_counts()
    weather = load_weather()
    stations = load_stations()
    df = prepare(counts, weather)

    station_codes = {sid: i for i, sid in enumerate(sorted(stations["station_id"]))}

    metrics = evaluate(df, station_codes, test_year)
    print("Hold-out evaluation:", metrics)

    # Final model: train on everything.
    model = make_model()
    model.fit(build_feature_frame(df, station_codes), np.log1p(df["cyclists"]))

    last_seen = df.groupby("station_id")["date"].max().dt.strftime("%Y-%m-%d").to_dict()
    bundle = {
        "model": model,
        "station_codes": station_codes,
        "stations": stations.assign(last_data=stations["station_id"].map(last_seen)).to_dict("records"),
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_until": df["date"].max().strftime("%Y-%m-%d"),
    }
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, config.MODEL_FILE)
    print(f"Saved model to {config.MODEL_FILE}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-year", type=int, default=2025)
    args = parser.parse_args()
    train(args.test_year)


if __name__ == "__main__":
    main()
