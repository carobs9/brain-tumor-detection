# DATA SOURE: https://s3.amazonaws.com/tripdata/index.html

import pandas as pd
import geopandas as gpd
from pathlib import Path

INPUT_DIR = Path("data/raw")
OUT_DIR = Path("data/interim")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.concat(
    [pd.read_csv(f, low_memory=False) for f in INPUT_DIR.glob("*citibike-tripdata*.csv")],
    ignore_index=True,
)

# ensure both cols are pd.datetime
df['started_at'] = pd.to_datetime(df['started_at'])
df['ended_at'] = pd.to_datetime(df['ended_at'])


hourly_long = ( # one row per station-hour that had at least one ride
    df.assign(hour=df["started_at"].dt.floor("1h"))
      .groupby(["start_station_id", "hour"])
      .size()
      .rename("rides")
      .reset_index()
)

full_range = pd.date_range(
    hourly_long["hour"].min(), hourly_long["hour"].max(), freq="1h"
)

hourly_wide = ( # station × hour matrix
    hourly_long.pivot(index="start_station_id", columns="hour", values="rides")
               .reindex(columns=full_range)
               .fillna(0)
               .astype(int)
)

print("long:", hourly_long.shape, " wide (stations x hours):", hourly_wide.shape)

hourly_long.to_parquet(OUT_DIR / "rides_long.parquet")
hourly_wide.to_parquet(OUT_DIR / "rides_wide.parquet")