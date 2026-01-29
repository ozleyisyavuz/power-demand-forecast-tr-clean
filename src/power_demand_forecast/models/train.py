from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-6
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100)


def main() -> None:
    data_path = Path("data/processed/demand.csv")
    if not data_path.exists():
        raise FileNotFoundError(
            "Veri yok. Önce üret: python -m power_demand_forecast.data.make_dataset"
        )

    df = pd.read_csv(data_path, parse_dates=["timestamp"]).sort_values("timestamp")

    features = ["hour", "dayofweek", "is_weekend", "temperature_c"]
    target = "demand_mw"

    X = df[features]
    y = df[target].to_numpy()

    # Son 14 gün validasyon
    val_hours = 14 * 24
    X_train, X_val = X.iloc[:-val_hours], X.iloc[-val_hours:]
    y_train, y_val = y[:-val_hours], y[-val_hours:]

    pre = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), ["dayofweek"])],
        remainder="passthrough",
    )

    model = HistGradientBoostingRegressor(random_state=42)
    pipe = Pipeline([("pre", pre), ("model", model)])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_val)

    score = mape(y_val, pred)
    print(f"Validation MAPE (%): {score:.2f}")

    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / "model.joblib")
    print(f"Saved model: {out_dir / 'model.joblib'}")


if __name__ == "__main__":
    main()
