from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def make_synthetic_hourly(start="2024-01-01", days=120, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=days * 24, freq="h")
    df = pd.DataFrame({"timestamp": idx})

    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    daily = 5 * np.sin(2 * np.pi * df["hour"] / 24)
    seasonal = 8 * np.sin(2 * np.pi * (df["timestamp"].dt.dayofyear) / 365)
    noise = rng.normal(0, 1.5, size=len(df))
    df["temperature_c"] = 15 + daily + seasonal + noise

    base = 30000
    hour_peak = 2500 * np.sin(2 * np.pi * (df["hour"] - 7) / 24) + 1500 * np.sin(4 * np.pi * df["hour"] / 24)
    weekend_drop = -2000 * df["is_weekend"]
    temp_effect = 120 * (20 - df["temperature_c"]).abs()
    demand_noise = rng.normal(0, 600, size=len(df))

    df["demand_mw"] = base + hour_peak + weekend]()
