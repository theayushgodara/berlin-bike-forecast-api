"""HTTP API: how many cyclists are expected at a Berlin station on a given day?

Run:  uvicorn bikeforecast.api:app --reload
Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Callable
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel

from bikeforecast import config
from bikeforecast.features import build_feature_frame
from bikeforecast.weather import fetch_forecast

app = FastAPI(
    title="Berlin Bike Forecast API",
    description="Forecast daily bicycle traffic at Berlin counting stations.",
    version="1.0.0",
)


class Station(BaseModel):
    station_id: str
    name: str
    lat: float
    lon: float
    last_data: str | None = None


class Forecast(BaseModel):
    station_id: str
    station_name: str
    date: date
    predicted_cyclists: int
    weather: dict[str, float]


class Health(BaseModel):
    status: str
    model_trained_at: str
    data_until: str
    metrics: dict


@lru_cache
def get_bundle() -> dict:
    if not config.MODEL_FILE.exists():
        raise RuntimeError(
            f"No model at {config.MODEL_FILE}. Run `python -m bikeforecast.train` first."
        )
    return joblib.load(config.MODEL_FILE)


def get_weather_fetcher() -> Callable[[date], dict]:
    return fetch_forecast


def berlin_today() -> date:
    return datetime.now(ZoneInfo(config.TIMEZONE)).date()


@app.get("/health", response_model=Health)
def health(bundle: dict = Depends(get_bundle)):
    return Health(
        status="ok",
        model_trained_at=bundle["trained_at"],
        data_until=bundle["data_until"],
        metrics=bundle["metrics"],
    )


@app.get("/stations", response_model=list[Station])
def stations(bundle: dict = Depends(get_bundle)):
    return [Station(**{k: s.get(k) for k in Station.model_fields}) for s in bundle["stations"]]


@app.get("/forecast/{station_id}", response_model=Forecast)
def forecast(
    station_id: str,
    day: date | None = Query(
        None, alias="date", description="YYYY-MM-DD, defaults to tomorrow (Berlin time)"
    ),
    bundle: dict = Depends(get_bundle),
    fetch_weather: Callable[[date], dict] = Depends(get_weather_fetcher),
):
    station = next((s for s in bundle["stations"] if s["station_id"] == station_id), None)
    if station is None:
        raise HTTPException(404, f"Unknown station '{station_id}'. See /stations.")

    today = berlin_today()
    day = day or today + timedelta(days=1)
    if not today <= day <= today + timedelta(days=config.MAX_FORECAST_DAYS):
        raise HTTPException(
            422, f"date must be between {today} and {today + timedelta(days=config.MAX_FORECAST_DAYS)}"
        )

    try:
        weather = fetch_weather(day)
    except Exception as exc:  # network problems, API changes, ...
        raise HTTPException(503, f"Weather forecast unavailable: {exc}") from exc

    row = pd.DataFrame([{"station_id": station_id, "date": pd.Timestamp(day), **weather}])
    X = build_feature_frame(row, bundle["station_codes"])
    pred = float(np.expm1(bundle["model"].predict(X)[0]))

    return Forecast(
        station_id=station_id,
        station_name=station["name"],
        date=day,
        predicted_cyclists=max(0, round(pred)),
        weather=weather,
    )
