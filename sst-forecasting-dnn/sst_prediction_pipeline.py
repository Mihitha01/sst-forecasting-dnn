"""
Simple SST prediction pipeline for a fisheries project.

Task:
1. Use real NOAA ERSST monthly SST data.
2. Select one equatorial Indian Ocean point: latitude 0, longitude 90E.
3. Train a DNN using data before 2000.
4. Predict SST from 2000 to 2025.
5. Compare predicted SST with actual observed SST.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import callbacks, layers


# -----------------------------------------------------------------------------
# Project settings
# -----------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "sst.mnmean.nc"
OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILES = [
    "dnn_sst_point_model.keras",
    "fig1_selected_point_sst.png",
    "fig2_dnn_prediction_vs_actual.png",
    "fig3_training_history.png",
    "fig4_prediction_error.png",
    "hindcast_2000_2025_comparison.csv",
    "results_summary.json",
]

TARGET_LAT = 0.0
TARGET_LON = 90.0
WINDOW_SIZE = 12
TRAIN_END = "1999-12-01"
TEST_START = "2000-01-01"
TEST_END = "2025-12-01"


# -----------------------------------------------------------------------------
# 1. Load one SST point from the real NOAA dataset
# -----------------------------------------------------------------------------
def load_point_sst(
    nc_path: Path = DATA_PATH,
    lat: float = TARGET_LAT,
    lon: float = TARGET_LON,
) -> tuple[pd.Series, dict]:
    """Return monthly SST at the selected lat/lon point as a pandas Series."""
    if not nc_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {nc_path}\n"
            "Place sst.mnmean.nc inside the data folder."
        )

    ds = xr.open_dataset(nc_path, engine="netcdf4")
    point = ds["sst"].sel(lat=lat, lon=lon, method="nearest")

    selected_lat = float(point["lat"].values)
    selected_lon = float(point["lon"].values)
    sst = point.to_series().astype(float).ffill().bfill()

    location = {
        "requested_lat": lat,
        "requested_lon": lon,
        "selected_lat": selected_lat,
        "selected_lon": selected_lon,
        "description": "Central equatorial Indian Ocean point",
    }

    print("Selected SST point")
    print(f"  requested : lat={lat}, lon={lon}E")
    print(f"  selected  : lat={selected_lat}, lon={selected_lon}E")
    print(f"  date range: {sst.index[0].date()} to {sst.index[-1].date()}")
    print(f"  months    : {len(sst)}")
    print(f"  missing   : {int(sst.isna().sum())}")

    return sst, location


# -----------------------------------------------------------------------------
# 2. Convert the time series into DNN input/output data
# -----------------------------------------------------------------------------
def make_supervised_data(
    sst: pd.Series,
    window: int = WINDOW_SIZE,
    test_start: str = TEST_START,
    test_end: str = TEST_END,
) -> dict:
    """Create 12-month input windows and split them by date."""
    values = sst.to_numpy(dtype=np.float32)
    dates = pd.DatetimeIndex(sst.index)

    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    target_dates: list[pd.Timestamp] = []

    for i in range(window, len(values)):
        x_rows.append(values[i - window : i])
        y_rows.append(values[i])
        target_dates.append(dates[i])

    x_raw = np.asarray(x_rows, dtype=np.float32)
    y_raw = np.asarray(y_rows, dtype=np.float32)
    y_dates = pd.DatetimeIndex(target_dates)

    train_mask = y_dates < pd.Timestamp(test_start)
    test_mask = (y_dates >= pd.Timestamp(test_start)) & (y_dates <= pd.Timestamp(test_end))

    x_train_raw = x_raw[train_mask]
    y_train_raw = y_raw[train_mask]
    x_test_raw = x_raw[test_mask]
    y_test_raw = y_raw[test_mask]
    train_dates = y_dates[train_mask]
    test_dates = y_dates[test_mask]

    if len(x_train_raw) == 0 or len(x_test_raw) == 0:
        raise ValueError("Train/test split produced empty data. Check the date settings.")

    # Fit the scaler on training data only to avoid data leakage.
    scaler = StandardScaler()
    scaler.fit(np.concatenate([x_train_raw.ravel(), y_train_raw]).reshape(-1, 1))

    x_train = scaler.transform(x_train_raw.reshape(-1, 1)).reshape(x_train_raw.shape)
    y_train = scaler.transform(y_train_raw.reshape(-1, 1)).ravel()
    x_test = scaler.transform(x_test_raw.reshape(-1, 1)).reshape(x_test_raw.shape)
    y_test = scaler.transform(y_test_raw.reshape(-1, 1)).ravel()

    print("Data split")
    print(f"  train: {train_dates[0].date()} to {train_dates[-1].date()} ({len(train_dates)} samples)")
    print(f"  test : {test_dates[0].date()} to {test_dates[-1].date()} ({len(test_dates)} samples)")

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "y_test_actual": y_test_raw,
        "train_dates": train_dates,
        "test_dates": test_dates,
        "scaler": scaler,
        "sst": sst,
    }


# -----------------------------------------------------------------------------
# 3. Build and train the DNN
# -----------------------------------------------------------------------------
def build_dnn(input_size: int = WINDOW_SIZE) -> keras.Model:
    """Small feed-forward neural network for monthly SST prediction."""
    model = keras.Sequential(
        [
            layers.Input(shape=(input_size,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.20),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.10),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_dnn(data: dict) -> tuple[keras.Model, keras.callbacks.History]:
    """Train the DNN using only the pre-2000 training data."""
    model = build_dnn(input_size=data["x_train"].shape[1])

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
    )

    history = model.fit(
        data["x_train"],
        data["y_train"],
        validation_split=0.20,
        epochs=300,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0,
    )

    model.save(OUTPUT_DIR / "dnn_sst_point_model.keras")
    return model, history


# -----------------------------------------------------------------------------
# 4. Predict 2000-2025 recursively and compare with actual SST
# -----------------------------------------------------------------------------
def predict_one_month(model: keras.Model, scaler: StandardScaler, window_values: np.ndarray) -> float:
    """Predict one month of SST from the previous 12 months."""
    x_norm = scaler.transform(window_values.reshape(-1, 1)).ravel().reshape(1, -1)
    pred_norm = model.predict(x_norm, verbose=0)[0, 0]
    return float(scaler.inverse_transform([[pred_norm]])[0, 0])


def recursive_hindcast(model: keras.Model, data: dict) -> np.ndarray:
    """
    Predict every month from 2000-2025 recursively.

    Recursive means: after predicting one month, that prediction is used as part
    of the next 12-month input window. This simulates forecasting forward from
    the year 2000 without using actual future SST as input.
    """
    sst = data["sst"]
    test_dates = data["test_dates"]
    scaler = data["scaler"]

    first_test_date = test_dates[0]
    window_values = sst.loc[sst.index < first_test_date].tail(WINDOW_SIZE).to_numpy(dtype=np.float32)

    predictions: list[float] = []
    for _ in test_dates:
        pred = predict_one_month(model, scaler, window_values)
        predictions.append(pred)
        window_values = np.append(window_values[1:], pred)

    return np.asarray(predictions, dtype=np.float32)


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Calculate common prediction error metrics."""
    mse = mean_squared_error(actual, predicted)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return {
        "mse": float(mse),
        "rmse": rmse,
        "mae": float(mae),
        "r2": float(r2),
    }


def build_comparison_table(data: dict, predicted: np.ndarray) -> pd.DataFrame:
    """Create a month-by-month actual vs predicted table."""
    actual = data["y_test_actual"]
    return pd.DataFrame(
        {
            "date": data["test_dates"].strftime("%Y-%m-%d"),
            "actual_sst": actual,
            "predicted_sst": predicted,
            "error_actual_minus_predicted": actual - predicted,
        }
    )


# -----------------------------------------------------------------------------
# 5. Save plots and result files
# -----------------------------------------------------------------------------
def plot_sst_series(sst: pd.Series, train_end: str = TRAIN_END) -> None:
    """Plot the selected point SST time series."""
    fig, ax = plt.subplots(figsize=(14, 4))
    dates = pd.DatetimeIndex(sst.index)
    split_date = pd.Timestamp(train_end)

    ax.plot(dates, sst.to_numpy(dtype=float), color="#1f77b4", lw=0.9, label="Monthly SST")
    ax.plot(dates, sst.rolling(12, center=True).mean(), color="#d62728", lw=2, label="12-month average")
    ax.axvline(split_date, color="black", ls="--", lw=1, label="Training ends")
    ax.set_title("SST at 0 deg latitude, 90E longitude")
    ax.set_xlabel("Year")
    ax.set_ylabel("SST (C)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_selected_point_sst.png", dpi=150)
    plt.close(fig)


def plot_prediction_comparison(comparison: pd.DataFrame, metrics: dict) -> None:
    """Plot actual SST vs DNN predicted SST for 2000-2025."""
    dates = pd.to_datetime(comparison["date"])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, comparison["actual_sst"], label="Actual SST", color="#1f77b4", lw=1.5)
    ax.plot(dates, comparison["predicted_sst"], label="DNN predicted SST", color="#2ca02c", lw=1.5, ls="--")
    ax.set_title(f"DNN SST Prediction vs Actual (RMSE = {metrics['rmse']:.3f} C)")
    ax.set_xlabel("Year")
    ax.set_ylabel("SST (C)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_dnn_prediction_vs_actual.png", dpi=150)
    plt.close(fig)


def plot_training_history(history: keras.callbacks.History) -> None:
    """Plot DNN training and validation loss."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history["loss"], label="Training loss")
    ax.plot(history.history["val_loss"], label="Validation loss")
    ax.set_title("DNN Training History")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_training_history.png", dpi=150)
    plt.close(fig)


def plot_residuals(comparison: pd.DataFrame) -> None:
    """Plot prediction error over time."""
    dates = pd.to_datetime(comparison["date"])
    error = comparison["error_actual_minus_predicted"]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dates, error, color="#9467bd", lw=1)
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_title("Prediction Error: Actual SST minus Predicted SST")
    ax.set_xlabel("Year")
    ax.set_ylabel("Error (C)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_prediction_error.png", dpi=150)
    plt.close(fig)


def save_results(location: dict, data: dict, metrics: dict, comparison: pd.DataFrame) -> None:
    """Save CSV and JSON outputs."""
    comparison.to_csv(OUTPUT_DIR / "hindcast_2000_2025_comparison.csv", index=False)

    summary = {
        "project": "DNN SST prediction for fisheries-focused point analysis",
        "dataset": str(DATA_PATH),
        "location": location,
        "method": {
            "model": "Dense Neural Network (DNN)",
            "input_window_months": WINDOW_SIZE,
            "training_period": {
                "start": str(data["train_dates"][0].date()),
                "end": str(data["train_dates"][-1].date()),
            },
            "prediction_period": {
                "start": str(data["test_dates"][0].date()),
                "end": str(data["test_dates"][-1].date()),
            },
            "prediction_style": "recursive hindcast from 2000 to 2025",
        },
        "metrics": {key: round(value, 6) for key, value in metrics.items()},
    }

    with open(OUTPUT_DIR / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def clean_previous_outputs() -> None:
    """Remove this script's old output files before creating fresh results."""
    for filename in OUTPUT_FILES:
        path = OUTPUT_DIR / filename
        if path.exists():
            path.unlink()


def main() -> None:
    print("SST prediction at 0 deg latitude, 90E longitude")
    print("Using real NOAA ERSST data and a DNN model.\n")

    sst, location = load_point_sst()
    data = make_supervised_data(sst)
    print("Training DNN...")
    model, history = train_dnn(data)
    print(f"Training stopped after {len(history.history['loss'])} epochs")

    predicted = recursive_hindcast(model, data)
    comparison = build_comparison_table(data, predicted)
    metrics = calculate_metrics(comparison["actual_sst"].to_numpy(), predicted)

    print("\nDNN recursive prediction results, 2000-2025")
    print(f"  RMSE: {metrics['rmse']:.4f} C")
    print(f"  MAE : {metrics['mae']:.4f} C")
    print(f"  R2  : {metrics['r2']:.4f}")

    plot_sst_series(sst)
    plot_prediction_comparison(comparison, metrics)
    plot_training_history(history)
    plot_residuals(comparison)
    save_results(location, data, metrics, comparison)

    print(f"\nSaved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
