# Berlin Bike Traffic Forecasting API

Forecasts daily bicycle traffic at Berlin's permanent counting stations.

**Question:** How many cyclists are expected at a selected Berlin station tomorrow?

## How it works

1. **Data** – hourly counts from all 33 permanent counters, published by the Berlin Senate
   ([source](https://www.berlin.de/sen/uvk/mobilitaet-und-verkehr/verkehrsplanung/radverkehr/weitere-radinfrastruktur/zaehlstellen-und-fahrradbarometer/)),
   summed into daily totals (2012–2025, ~96k station-days). Incomplete days and
   counter faults (days below 10% of the station's usual level) are removed.
2. **Weather** – daily Berlin weather from [Open-Meteo](https://open-meteo.com/)
   (history for training, forecast for predictions; no API key needed).
3. **Features** – station, weekday, weekend, Berlin public holiday, bridge day,
   season (day-of-year), year, temperature, rain, sunshine, wind.
   Only things known in advance are used, so any day with a weather forecast
   (up to 15 days ahead) can be predicted.
4. **Model** – one scikit-learn `HistGradientBoostingRegressor` across all
   stations, trained on `log(1 + cyclists)`.

## Results (trained on 2012–2024, tested on 2025)

| | MAE (cyclists/day) | MAPE |
|---|---|---|
| Baseline: station × weekday average | 668 | 74% |
| **Model** | **385** | **54%** |

The model cuts the average error by ~42% compared to the baseline.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
pip install -e .
```

## Build the data and model

```bash
python -m bikeforecast.data      # download counts (~18 MB) and build data/processed/
python -m bikeforecast.weather   # download historical weather
python -m bikeforecast.train     # evaluate on 2025, then train final model -> models/model.joblib
```

## Run the API

```bash
uvicorn bikeforecast.api:app --reload
```

Interactive docs: http://127.0.0.1:8000/docs

| Endpoint | Description |
|---|---|
| `GET /health` | Model status, training date and evaluation metrics |
| `GET /stations` | All counting stations with id, name and coordinates |
| `GET /forecast/{station_id}` | Forecast for tomorrow |
| `GET /forecast/{station_id}?date=YYYY-MM-DD` | Forecast for a date up to 15 days ahead |

Example:

```bash
curl http://127.0.0.1:8000/forecast/05-FK-OBB-O
```

```json
{
  "station_id": "05-FK-OBB-O",
  "station_name": "Oberbaumbrücke Ost",
  "date": "2026-11-13",
  "predicted_cyclists": 6071,
  "weather": {"temperature_2m_max": 8.0, "precipitation_sum": 0.0, "...": "..."}
}
```

## Docker

```bash
docker build -t berlin-bike-forecast .
docker run -p 8000:8000 berlin-bike-forecast
```

(Train the model first – the image copies `models/model.joblib`.)

## Tests

```bash
pytest
```

API tests use a small synthetic model and fake weather, so they run offline.

## Project structure

```
src/bikeforecast/
  config.py     paths and constants
  data.py       download + clean bike counts
  weather.py    Open-Meteo history and forecast
  features.py   calendar/weather features (shared by training and API)
  train.py      evaluation + final model training
  api.py        FastAPI app
tests/          pytest suite
notebooks/      exploration
```

## Limitations

- One weather point for the whole city.
- The official data is published with a delay, so the model has no information about
  recent trends (e.g. construction work closing a street near a counter).
- School holidays are not yet a feature.
