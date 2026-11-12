"""Paths and constants shared across the project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

RAW_COUNTS_FILE = RAW_DIR / "gesamtdatei-stundenwerte.xlsx"
DAILY_COUNTS_FILE = PROCESSED_DIR / "daily_counts.csv"
STATIONS_FILE = PROCESSED_DIR / "stations.csv"
WEATHER_FILE = PROCESSED_DIR / "weather_daily.csv"
MODEL_FILE = MODELS_DIR / "model.joblib"

# Berlin Senate open data: hourly counts for all permanent bike counters.
COUNTS_URL = (
    "https://www.berlin.de/sen/uvk/_assets/verkehr/verkehrsplanung/radverkehr/"
    "weitere-radinfrastruktur/zaehlstellen-und-fahrradbarometer/"
    "gesamtdatei-stundenwerte.xlsx"
)

# Weather is taken for central Berlin; the city is small enough that one
# point is a reasonable approximation for every station.
BERLIN_LAT = 52.52
BERLIN_LON = 13.405
TIMEZONE = "Europe/Berlin"

# Daily weather variables requested from Open-Meteo (historical + forecast).
WEATHER_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "precipitation_hours",
    "sunshine_duration",
    "wind_speed_10m_max",
]

# Open-Meteo's forecast API covers today + 15 days.
MAX_FORECAST_DAYS = 15
