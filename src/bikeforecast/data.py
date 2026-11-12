"""Download and clean the Berlin bike counter data.

Run:  python -m bikeforecast.data
"""

from __future__ import annotations

import argparse

import pandas as pd
import requests

from bikeforecast import config


def download_counts(force: bool = False) -> None:
    """Download the raw Excel file (~18 MB) unless it is already present."""
    if config.RAW_COUNTS_FILE.exists() and not force:
        print(f"Raw file already present: {config.RAW_COUNTS_FILE}")
        return
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading bike counts from berlin.de ...")
    resp = requests.get(config.COUNTS_URL, timeout=120)
    resp.raise_for_status()
    config.RAW_COUNTS_FILE.write_bytes(resp.content)
    print(f"Saved {len(resp.content) / 1e6:.1f} MB to {config.RAW_COUNTS_FILE}")


def parse_stations(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the 'Standortdaten' sheet into a tidy station table."""
    df = df.iloc[:, :5].copy()
    df.columns = ["station_id", "name", "lat", "lon", "installed"]
    df = df.dropna(subset=["station_id"])
    df["station_id"] = df["station_id"].astype(str).str.strip()
    df["installed"] = pd.to_datetime(df["installed"]).dt.date
    return df.reset_index(drop=True)


def hourly_sheet_to_long(sheet: pd.DataFrame) -> pd.DataFrame:
    """Convert one 'Jahresdatei YYYY' sheet (wide, hourly) to long format.

    Column headers look like '02-MI-JAN-N\\n01.04.2015' (newer sheets) or
    '02-MI-JAN-N 01.04.2015' (older sheets): id + install date. Only the id
    is kept.
    """
    sheet = sheet.rename(columns={sheet.columns[0]: "timestamp"})
    sheet = sheet.rename(columns=lambda c: c if c == "timestamp" else str(c).split()[0])
    sheet["timestamp"] = pd.to_datetime(sheet["timestamp"], errors="coerce")
    sheet = sheet.dropna(subset=["timestamp"])
    long = sheet.melt(id_vars="timestamp", var_name="station_id", value_name="count")
    long["count"] = pd.to_numeric(long["count"], errors="coerce")
    return long


def hourly_to_daily(long: pd.DataFrame) -> pd.DataFrame:
    """Sum hourly counts into daily totals, keeping only trustworthy days.

    A day is kept only if it has at least 23 hourly values (allowing for the
    DST switch) and a total above zero; zero days mean the counter was broken.
    """
    long = long.assign(date=long["timestamp"].dt.date)
    daily = long.groupby(["station_id", "date"])["count"].agg(["sum", "count"]).reset_index()
    daily = daily[(daily["count"] >= 23) & (daily["sum"] > 0)]
    daily = daily.rename(columns={"sum": "cyclists"}).drop(columns="count")
    daily["cyclists"] = daily["cyclists"].astype(int)
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values(["station_id", "date"]).reset_index(drop=True)


def build_processed() -> None:
    """Read the raw Excel file and write stations.csv + daily_counts.csv."""
    print("Reading Excel file (this takes a minute) ...")
    sheets = pd.read_excel(config.RAW_COUNTS_FILE, sheet_name=None)

    stations = parse_stations(sheets["Standortdaten"])
    year_sheets = [s for name, s in sheets.items() if name.startswith("Jahresdatei")]
    long = pd.concat([hourly_sheet_to_long(s) for s in year_sheets], ignore_index=True)
    daily = hourly_to_daily(long)

    # Keep only stations that appear in both tables.
    daily = daily[daily["station_id"].isin(stations["station_id"])]
    stations = stations[stations["station_id"].isin(daily["station_id"])]

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stations.to_csv(config.STATIONS_FILE, index=False)
    daily.to_csv(config.DAILY_COUNTS_FILE, index=False)
    print(
        f"Wrote {len(stations)} stations and {len(daily):,} station-days "
        f"({daily['date'].min():%Y-%m-%d} to {daily['date'].max():%Y-%m-%d})"
    )


def load_daily_counts() -> pd.DataFrame:
    return pd.read_csv(config.DAILY_COUNTS_FILE, parse_dates=["date"])


def load_stations() -> pd.DataFrame:
    return pd.read_csv(config.STATIONS_FILE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    download_counts(force=args.force_download)
    build_processed()


if __name__ == "__main__":
    main()
