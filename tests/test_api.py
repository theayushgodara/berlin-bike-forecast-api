from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from bikeforecast import api, config
from bikeforecast.features import build_feature_frame
from bikeforecast.train import make_model

FAKE_WEATHER = {
    "temperature_2m_max": 22.0,
    "temperature_2m_min": 12.0,
    "precipitation_sum": 0.0,
    "precipitation_hours": 0.0,
    "sunshine_duration": 10.0,
    "wind_speed_10m_max": 12.0,
}


@pytest.fixture(scope="module")
def bundle():
    """A tiny model trained on synthetic data so tests run offline and fast."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", "2024-12-31")
    rows = []
    for sid, base in [("S1", 3000), ("S2", 500)]:
        for d in dates:
            temp = 10 + 10 * np.sin(2 * np.pi * d.day_of_year / 365)
            rows.append({"station_id": sid, "date": d, **FAKE_WEATHER,
                         "temperature_2m_max": temp,
                         "cyclists": base * (1 + temp / 30) * (0.6 if d.day_of_week >= 5 else 1)
                         + rng.normal(0, 20)})
    df = pd.DataFrame(rows)
    codes = {"S1": 0, "S2": 1}
    model = make_model().set_params(max_iter=50)
    model.fit(build_feature_frame(df, codes), np.log1p(df["cyclists"].clip(lower=1)))
    return {
        "model": model,
        "station_codes": codes,
        "stations": [
            {"station_id": "S1", "name": "Busy Bridge", "lat": 52.5, "lon": 13.4, "last_data": "2024-12-31"},
            {"station_id": "S2", "name": "Quiet Street", "lat": 52.4, "lon": 13.3, "last_data": "2024-12-31"},
        ],
        "metrics": {"model_mae": 1.0},
        "trained_at": "2025-01-01T00:00:00+00:00",
        "data_until": "2024-12-31",
    }


@pytest.fixture
def client(bundle):
    api.app.dependency_overrides[api.get_bundle] = lambda: bundle
    api.app.dependency_overrides[api.get_weather_fetcher] = lambda: (lambda day: FAKE_WEATHER)
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_stations(client):
    r = client.get("/stations")
    assert r.status_code == 200
    assert [s["station_id"] for s in r.json()] == ["S1", "S2"]


def test_forecast_defaults_to_tomorrow(client):
    r = client.get("/forecast/S1")
    assert r.status_code == 200
    body = r.json()
    assert body["date"] == str(api.berlin_today() + timedelta(days=1))
    assert body["station_name"] == "Busy Bridge"
    assert body["predicted_cyclists"] > 0
    assert body["weather"] == FAKE_WEATHER


def test_busier_station_gets_higher_forecast(client):
    busy = client.get("/forecast/S1").json()["predicted_cyclists"]
    quiet = client.get("/forecast/S2").json()["predicted_cyclists"]
    assert busy > quiet


def test_unknown_station_is_404(client):
    assert client.get("/forecast/NOPE").status_code == 404


def test_date_out_of_range_is_422(client):
    too_far = api.berlin_today() + timedelta(days=config.MAX_FORECAST_DAYS + 1)
    assert client.get(f"/forecast/S1?date={too_far}").status_code == 422
    past = api.berlin_today() - timedelta(days=1)
    assert client.get(f"/forecast/S1?date={past}").status_code == 422


def test_weather_failure_is_503(client, bundle):
    def broken(day):
        raise ConnectionError("offline")
    api.app.dependency_overrides[api.get_weather_fetcher] = lambda: broken
    assert client.get("/forecast/S1").status_code == 503
