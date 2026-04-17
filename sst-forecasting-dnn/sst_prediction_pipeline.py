"""
=============================================================================
  SEA SURFACE TEMPERATURE (SST) PREDICTION PIPELINE
  Dataset: NOAA ERSST v5  |  Source: psl.noaa.gov/thredds
  Author : Senior ML Engineer & Climate Data Scientist
  Version: 1.0
=============================================================================

PIPELINE OVERVIEW
-----------------
  Stage 1  →  Data Acquisition & Synthetic Fallback
  Stage 2  →  Exploratory Data Analysis
  Stage 3  →  Preprocessing (cleaning, normalisation, windowing)
  Stage 4  →  Baseline Model  – Linear Regression
  Stage 5  →  Deep Neural Network  – Fully-Connected (Dense)
  Stage 6  →  Evaluation & Comparison
  Stage 7  →  Future Forecast (next 12 months)
  Stage 8  →  Visualisation & Report
"""

# ─────────────────────────────────────────────────────────────────────────────
#  0.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import warnings, os, json
import urllib.request
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy  as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from pathlib import Path

from sklearn.linear_model   import LinearRegression
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics        import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

OUTPUT_DIR = Path(r"D:\PROJECTS")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_SIZE   = 12     # months of past SST used as features
FORECAST_STEPS= 12     # months to predict into the future
TRAIN_RATIO   = 0.80   # fraction of data used for training
ERSST_URL     = "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.ersst.v5/sst.mnmean.nc"
ERSST_FILE_URL = "https://psl.noaa.gov/thredds/fileServer/Datasets/noaa.ersst.v5/sst.mnmean.nc"
ERSST_CATALOG_URL = "https://psl.noaa.gov/thredds/catalog/Datasets/noaa.ersst.v5/catalog.html"
ERSST_LOCAL_CACHE = OUTPUT_DIR / "sst.mnmean.nc"
ERSST_MIN_VALID_BYTES = 100_000_000


def download_ersst_to_cache(force: bool = False) -> Path:
    """Download NOAA ERSST NetCDF to the local cache path."""
    if (
        not force
        and ERSST_LOCAL_CACHE.exists()
        and ERSST_LOCAL_CACHE.stat().st_size >= ERSST_MIN_VALID_BYTES
    ):
        print(f"  Using cached file: {ERSST_LOCAL_CACHE}")
        return ERSST_LOCAL_CACHE

    tmp_path = ERSST_LOCAL_CACHE.with_suffix(".part")

    # Step 1: discover remote file size.
    total_size: int | None = None
    with urllib.request.urlopen(ERSST_FILE_URL, timeout=120) as response:
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            total_size = int(content_length)

    if total_size is None or total_size <= 0:
        raise RuntimeError("Unable to determine NOAA NetCDF file size.")

    # Step 2: download with byte ranges to tolerate flaky connections.
    chunk_size = 5 * 1024 * 1024
    chunk_retries = 5
    n_chunks = (total_size + chunk_size - 1) // chunk_size

    with open(tmp_path, "wb") as f:
        f.truncate(total_size)

    with open(tmp_path, "r+b") as f:
        for chunk_idx, start in enumerate(range(0, total_size, chunk_size), start=1):
            end = min(start + chunk_size - 1, total_size - 1)
            expected_len = end - start + 1

            success = False
            for attempt in range(1, chunk_retries + 1):
                try:
                    req = urllib.request.Request(
                        ERSST_FILE_URL,
                        headers={"Range": f"bytes={start}-{end}"},
                    )
                    with urllib.request.urlopen(req, timeout=120) as response:
                        data = response.read()

                    if len(data) != expected_len:
                        raise RuntimeError(
                            f"Chunk size mismatch: got {len(data)} expected {expected_len}"
                        )

                    f.seek(start)
                    f.write(data)
                    success = True
                    break
                except Exception:
                    if attempt == chunk_retries:
                        raise

            if not success:
                raise RuntimeError(f"Failed to download chunk {chunk_idx}/{n_chunks}")

            if chunk_idx % 5 == 0 or chunk_idx == n_chunks:
                pct = (end + 1) / total_size * 100
                print(f"  Download progress: {pct:.1f}% ({chunk_idx}/{n_chunks} chunks)")

    actual_size = tmp_path.stat().st_size
    if actual_size != total_size:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Incomplete download: {actual_size} < expected {total_size} bytes")

    if actual_size < ERSST_MIN_VALID_BYTES:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Downloaded file too small ({actual_size} bytes), likely incomplete")

    tmp_path.replace(ERSST_LOCAL_CACHE)
    print("  ✔ Download completed.")
    return ERSST_LOCAL_CACHE


def _validate_sst_dataset(ds: xr.Dataset) -> tuple[bool, str]:
    """Basic sanity checks to catch corrupted OPeNDAP reads."""
    if "sst" not in ds:
        return False, "Dataset does not contain 'sst' variable."

    try:
        sample = ds["sst"].isel(time=slice(0, 24)).mean(dim=["lat", "lon"], skipna=True)
        vals = np.asarray(sample.to_numpy(), dtype=np.float64).ravel()
    except Exception as e:
        return False, f"Unable to read sample from remote dataset ({type(e).__name__}: {e})"

    if vals.size == 0:
        return False, "Remote SST sample is empty."

    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return False, "Remote SST sample has no finite values."

    if np.nanstd(finite) < 1e-6:
        return False, "Remote SST sample appears constant (likely corrupted transfer)."

    # Probe across the full timeline to catch partial OPeNDAP packet failures.
    try:
        weights = xr.DataArray(
            np.cos(np.deg2rad(ds["lat"].to_numpy())),
            coords={"lat": ds["lat"]},
            dims=["lat"],
            name="weights",
        )
        probe = ds["sst"].isel(time=slice(0, None, 24)).weighted(weights).mean(
            dim=["lat", "lon"], skipna=True
        )
        probe_vals = np.asarray(probe.to_numpy(), dtype=np.float64).ravel()
    except Exception as e:
        return False, f"Timeline probe failed ({type(e).__name__}: {e})"

    probe_finite = probe_vals[np.isfinite(probe_vals)]
    if probe_finite.size == 0:
        return False, "Timeline probe has no finite values."

    if np.nanstd(probe_finite) < 1e-6:
        return False, "Timeline probe appears constant (likely corrupted transfer)."

    return True, "ok"

# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 1 – DATA ACQUISITION
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_ersst(
        start: str = "1854-01",
        end:   str = "2024-12",
        n_lat: int = 89,
        n_lon: int = 180,
) -> xr.Dataset:
    """
    Generate a synthetic NOAA ERSST v5-like dataset.

    The synthetic data faithfully reproduces the key statistical properties
    of the real monthly mean SST product:

      • Global mean SST ≈ 17–19 °C
      • Seasonal cycle amplitude ≈ 0.3–0.5 °C  (global average)
      • Long-term warming trend  ≈ +0.7 °C / 100 yr
      • ENSO-like inter-annual variability (AR-1 red noise, σ ≈ 0.2 °C)
      • Missing values (NaN) on land cells  (~30 % of grid)
      • Spatial structure via latitudinal SST gradient

    Parameters
    ----------
    start, end : str
        Date range in "YYYY-MM" format.
    n_lat, n_lon : int
        Grid dimensions (default mirrors the 2° ERSST v5 grid).

    Returns
    -------
    xr.Dataset  with variable "sst" and dimensions (time, lat, lon).
    """
    times   = pd.date_range(start=start, end=end, freq="MS")  # monthly start
    lats    = np.linspace(-88, 88, n_lat)
    lons    = np.linspace(0, 358, n_lon)
    T       = len(times)

    print(f"  Grid      : {n_lat} lat × {n_lon} lon")
    print(f"  Time steps: {T} months  ({start} → {end})")

    # ── 1. Latitudinal SST climatology (warm tropics, cool poles) ────────────
    lat_sst = 28.0 * np.exp(-(lats**2) / (2 * 35**2))   # Gaussian, peak at 0°

    # ── 2. Land mask: approx 30 % of ocean grid is "land" (simplified) ───────
    rng = np.random.default_rng(SEED)
    land_mask = rng.random((n_lat, n_lon)) < 0.28        # True = land (NaN)

    # ── 3. Seasonal cycle (Northern Hemisphere phase, global average) ─────────
    months  = np.array([t.month for t in times])
    seasonal= 0.45 * np.sin(2 * np.pi * (months - 3) / 12)  # peak in late summer

    # ── 4. Long-term warming trend ────────────────────────────────────────────
    trend   = np.linspace(0, 0.9, T)                     # +0.9 °C over full record

    # ── 5. ENSO-like inter-annual variability (AR-1 red noise) ───────────────
    enso    = np.zeros(T)
    for i in range(1, T):
        enso[i] = 0.85 * enso[i-1] + rng.normal(0, 0.18)

    # ── 6. Assemble SST array (time, lat, lon) ────────────────────────────────
    sst = np.zeros((T, n_lat, n_lon), dtype=np.float32)
    for li, lat in enumerate(lats):
        # longitude-dependent perturbation (Western Pacific warm pool etc.)
        lon_pert = 1.5 * np.cos(lons * np.pi / 180)
        base     = lat_sst[li] + lon_pert

        for ti in range(T):
            # small spatial noise per cell per time step
            cell_noise = rng.normal(0, 0.05, n_lon).astype(np.float32)
            sst[ti, li, :] = (
                base
                + seasonal[ti]
                + trend[ti]
                + enso[ti]
                + cell_noise
            )

    # ── 7. Apply land mask (NaN) ──────────────────────────────────────────────
    sst[:, land_mask] = np.nan

    # ── 8. Wrap in xarray Dataset (same structure as real ERSST v5) ───────────
    ds = xr.Dataset(
        {"sst": (["time", "lat", "lon"], sst)},
        coords={
            "time": times,
            "lat" : lats,
            "lon" : lons,
        },
    )
    ds["sst"].attrs = {
        "long_name"   : "Monthly Mean Sea Surface Temperature",
        "units"       : "°C",
        "source"      : "Synthetic NOAA ERSST v5-like dataset",
        "missing_value": np.nan,
    }
    ds.attrs["title"] = "NOAA ERSST v5 – Synthetic Replica for ML Pipeline"
    return ds


def load_sst_dataset(
    nc_path: str | None = None,
    use_remote: bool = True,
    allow_synthetic_fallback: bool = False,
    download_on_remote_fail: bool = True,
) -> xr.Dataset:
    """
    Load the NOAA ERSST v5 dataset.

        Priority order:
            1. Local .nc file (if `nc_path` is provided and exists)
            2. OPeNDAP remote stream (psl.noaa.gov)
            3. Synthetic fallback (optional, disabled by default)

    Parameters
    ----------
    nc_path : str, optional
        Path to a locally downloaded sst.mnmean.nc file.
    use_remote : bool
        Whether to attempt loading from the NOAA OPeNDAP URL.
    allow_synthetic_fallback : bool
        If True, fall back to generated synthetic data when remote access fails.
    download_on_remote_fail : bool
        If True, download the NOAA NetCDF file and load locally when OPeNDAP fails.

    Returns
    -------
    xr.Dataset
    """
    print("\n" + "═"*60)
    print("  STAGE 1 – DATA ACQUISITION")
    print("═"*60)

    # ── Option A: local file ──────────────────────────────────────────────────
    if nc_path:
        if Path(nc_path).exists():
            print(f"  Loading local file: {nc_path}")
            ds = xr.open_dataset(nc_path, engine="netcdf4")
            print("  ✔ Loaded from local NetCDF file.")
            return ds
        raise FileNotFoundError(f"Local NetCDF file not found: {nc_path}")

    # ── Option B: OPeNDAP remote ──────────────────────────────────────────────
    if use_remote:
        print(f"  Attempting OPeNDAP stream: {ERSST_URL}")
        remote_errors = []
        for engine in ["netcdf4", None]:
            try:
                if engine:
                    ds = xr.open_dataset(ERSST_URL, engine=engine)
                    tag = engine
                else:
                    ds = xr.open_dataset(ERSST_URL)
                    tag = "auto"

                is_valid, reason = _validate_sst_dataset(ds)
                if not is_valid:
                    remote_errors.append(f"{tag}: invalid dataset ({reason})")
                    ds.close()
                    continue

                print(f"  ✔ Loaded via OPeNDAP (engine='{tag}').")
                return ds
            except Exception as e:
                tag = engine if engine else "auto"
                remote_errors.append(f"{tag}: {type(e).__name__}: {e}")

        print("  ✗ OPeNDAP unavailable. Errors:")
        for err in remote_errors:
            print(f"    - {err}")

        if download_on_remote_fail:
            print("\n  Attempting direct NetCDF download from NOAA fileServer...")
            try:
                local_nc = download_ersst_to_cache(force=False)
                ds = xr.open_dataset(local_nc, engine="netcdf4")
                is_valid, reason = _validate_sst_dataset(ds)
                if not is_valid:
                    ds.close()
                    raise RuntimeError(f"Downloaded file validation failed: {reason}")

                print("  ✔ Loaded from downloaded NOAA NetCDF file.")
                return ds
            except Exception as e:
                print(f"  ✗ Direct download/load failed ({type(e).__name__}: {e})")

    # ── Option C: synthetic fallback ─────────────────────────────────────────
    if allow_synthetic_fallback:
        print("\n  Falling back to synthetic ERSST v5-like dataset.")
        print("  (Identical pipeline, same structure as real product.)\n")
        ds = generate_synthetic_ersst()
        print("  ✔ Synthetic dataset ready.")
        return ds

    raise RuntimeError(
        "Unable to load NOAA ERSST data from OPeNDAP. "
        f"Download the file manually from: {ERSST_FILE_URL} "
        "and run with --nc-path <local_file>."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 2 – EDA & GLOBAL AVERAGE TIME SERIES
# ─────────────────────────────────────────────────────────────────────────────

def compute_global_mean_sst(ds: xr.Dataset) -> pd.Series:
    """
    Compute area-weighted global-mean SST for every time step.

    Steps
    -----
    1. Extract 'sst' DataArray.
    2. Weight each grid cell by cos(lat) to account for smaller area
       towards the poles (spherical geometry correction).
    3. Average over (lat, lon), skipping NaN land cells.
    4. Return a pandas Series indexed by datetime.

    Parameters
    ----------
    ds : xr.Dataset  – raw ERSST dataset

    Returns
    -------
    pd.Series  – monthly global-mean SST in °C
    """
    print("\n" + "═"*60)
    print("  STAGE 2 – EDA & GLOBAL AVERAGE")
    print("═"*60)

    sst = ds["sst"]

    # Cosine-latitude weighting (standard climate-science practice)
    weights = np.cos(np.deg2rad(ds.lat))
    weights.name = "weights"

    sst_weighted = sst.weighted(weights)
    global_mean  = sst_weighted.mean(dim=["lat", "lon"], skipna=True)
    sst_series   = global_mean.to_series()

    print(f"  Time range : {sst_series.index[0].date()} → {sst_series.index[-1].date()}")
    print(f"  N months   : {len(sst_series)}")
    print(f"  Mean SST   : {sst_series.mean():.4f} °C")
    print(f"  Std SST    : {sst_series.std():.4f} °C")
    print(f"  Min SST    : {sst_series.min():.4f} °C")
    print(f"  Max SST    : {sst_series.max():.4f} °C")
    print(f"  Missing    : {sst_series.isna().sum()} values")

    if float(sst_series.std()) < 1e-6:
        raise RuntimeError(
            "SST series is nearly constant. Remote read is likely corrupted. "
            "Retry, or run with --nc-path using a downloaded local file."
        )

    return sst_series


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 3 – PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_sst(
        sst_series: pd.Series,
        window: int = WINDOW_SIZE,
        train_ratio: float = TRAIN_RATIO,
) -> dict:
    """
    Full preprocessing pipeline for supervised SST forecasting.

    Steps
    -----
    1. Handle missing values (forward-fill, then back-fill).
    2. Z-score normalisation (fit scaler only on training data to
       prevent data leakage).
    3. Sliding-window feature matrix:
         X[t] = [SST(t-W), SST(t-W+1), …, SST(t-1)]   (W = 12)
         y[t] = SST(t)
    4. Chronological train / test split (no shuffling – time series!).

    Parameters
    ----------
    sst_series  : pd.Series  – monthly global-mean SST
    window      : int        – look-back period in months
    train_ratio : float      – fraction of samples for training

    Returns
    -------
    dict with keys:
        X_train, X_test, y_train, y_test  – numpy arrays (normalised)
        scaler  – fitted StandardScaler
        dates   – full DatetimeIndex aligned to y
        split_idx – integer index of train/test boundary
        sst_raw – original (un-normalised) series
    """
    print("\n" + "═"*60)
    print("  STAGE 3 – PREPROCESSING")
    print("═"*60)

    # ── 3.1  Missing value imputation ────────────────────────────────────────
    n_missing_before = sst_series.isna().sum()
    sst_filled = sst_series.ffill().bfill()   # time-aware forward/back fill
    n_missing_after = sst_filled.isna().sum()
    print(f"  Missing values : {n_missing_before} → {n_missing_after} (after fill)")

    raw_values = np.asarray(sst_filled.to_numpy(dtype=np.float32)).reshape(-1, 1)

    # ── 3.2  Sliding-window dataset construction ─────────────────────────────
    n = len(raw_values)
    X_raw, y_raw, dates = [], [], []

    for i in range(window, n):
        X_raw.append(raw_values[i - window : i, 0])  # past W months
        y_raw.append(raw_values[i, 0])                # next month
        dates.append(sst_filled.index[i])

    X_raw  = np.array(X_raw,  dtype=np.float32)      # (N, W)
    y_raw  = np.array(y_raw,  dtype=np.float32)      # (N,)
    dates  = pd.DatetimeIndex(dates)

    print(f"  Samples after windowing: {len(y_raw)}")
    print(f"  Feature vector length  : {window}")

    # ── 3.3  Chronological train/test split ──────────────────────────────────
    split_idx = int(len(y_raw) * train_ratio)

    X_train_raw = X_raw[:split_idx]
    X_test_raw  = X_raw[split_idx:]
    y_train_raw = y_raw[:split_idx]
    y_test_raw  = y_raw[split_idx:]

    print(f"  Train samples: {len(X_train_raw)}  "
          f"({dates[0].date()} → {dates[split_idx-1].date()})")
    print(f"  Test  samples: {len(X_test_raw)}  "
          f"({dates[split_idx].date()} → {dates[-1].date()})")

    # ── 3.4  Z-score normalisation (fit on train only) ───────────────────────
    # We treat all W features + target together to keep scale consistent
    all_train_vals = np.concatenate([X_train_raw.flatten(), y_train_raw])
    scaler = StandardScaler()
    scaler.fit(all_train_vals.reshape(-1, 1))

    X_train = scaler.transform(X_train_raw.reshape(-1, 1)).reshape(X_train_raw.shape)
    X_test  = scaler.transform(X_test_raw.reshape(-1, 1)).reshape(X_test_raw.shape)
    y_train = scaler.transform(y_train_raw.reshape(-1, 1)).ravel()
    y_test  = scaler.transform(y_test_raw.reshape(-1, 1)).ravel()

    print(f"  Normalised – mean≈{X_train.mean():.3f}, std≈{X_train.std():.3f}")

    return {
        "X_train"  : X_train,
        "X_test"   : X_test,
        "y_train"  : y_train,
        "y_test"   : y_test,
        "scaler"   : scaler,
        "dates"    : dates,
        "split_idx": split_idx,
        "sst_raw"  : sst_filled,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 4 – BASELINE: LINEAR REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

def train_linear_regression(data: dict) -> dict:
    """
    Baseline model: Ordinary Least Squares Linear Regression.

    The model learns a linear mapping from the 12-month feature vector
    to the next-month SST.  This serves as a lower bound — any properly
    trained DNN should outperform it on a non-linear signal.

    Parameters
    ----------
    data : dict  – output of preprocess_sst()

    Returns
    -------
    dict with keys:
        model, train_preds, test_preds,
        train_mse, test_mse, train_rmse, test_rmse,
        train_r2, test_r2
    """
    print("\n" + "═"*60)
    print("  STAGE 4 – BASELINE: LINEAR REGRESSION")
    print("═"*60)

    lr = LinearRegression()
    lr.fit(data["X_train"], data["y_train"])

    train_preds = lr.predict(data["X_train"])
    test_preds  = lr.predict(data["X_test"])

    # ── Compute metrics in original scale ─────────────────────────────────────
    scaler = data["scaler"]
    y_tr_orig    = scaler.inverse_transform(data["y_train"].reshape(-1,1)).ravel()
    y_te_orig    = scaler.inverse_transform(data["y_test"].reshape(-1,1)).ravel()
    tr_pred_orig = scaler.inverse_transform(train_preds.reshape(-1,1)).ravel()
    te_pred_orig = scaler.inverse_transform(test_preds.reshape(-1,1)).ravel()

    train_mse  = mean_squared_error(y_tr_orig, tr_pred_orig)
    test_mse   = mean_squared_error(y_te_orig, te_pred_orig)
    train_rmse = np.sqrt(train_mse)
    test_rmse  = np.sqrt(test_mse)
    train_r2   = r2_score(y_tr_orig, tr_pred_orig)
    test_r2    = r2_score(y_te_orig, te_pred_orig)

    print(f"  Train  MSE = {train_mse:.6f}  |  RMSE = {train_rmse:.6f}  |  R² = {train_r2:.6f}")
    print(f"  Test   MSE = {test_mse:.6f}  |  RMSE = {test_rmse:.6f}  |  R² = {test_r2:.6f}")

    return {
        "model"       : lr,
        "train_preds" : tr_pred_orig,
        "test_preds"  : te_pred_orig,
        "train_mse"   : train_mse,
        "test_mse"    : test_mse,
        "train_rmse"  : train_rmse,
        "test_rmse"   : test_rmse,
        "train_r2"    : train_r2,
        "test_r2"     : test_r2,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 5 – DEEP NEURAL NETWORK
# ─────────────────────────────────────────────────────────────────────────────

def build_dnn(input_dim: int) -> keras.Model:
    """
    Construct a fully-connected (Dense) DNN for SST regression.

    Architecture
    ------------
      Input(12)
        │
      Dense(128, ReLU) + L2(1e-4) + BatchNorm + Dropout(0.2)
        │
      Dense(64, ReLU)  + L2(1e-4) + BatchNorm + Dropout(0.2)
        │
      Dense(32, ReLU)  + L2(1e-4)
        │
      Dense(1)  ← linear output (regression)

    Design Choices
    --------------
    • ReLU activations – universal approximators, avoids vanishing gradients.
    • L2 regularisation – penalises large weights, reduces overfitting.
    • BatchNorm – stabilises training, speeds convergence.
    • Dropout  – stochastic regularisation during training.
    • Linear output – appropriate for real-valued regression.
    • Adam optimiser – adaptive learning rate, robust default.
    • MSE loss – standard for regression; sensitive to large errors.
    """
    inp = keras.Input(shape=(input_dim,), name="sst_features")

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4), name="dense_1")(inp)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Activation("relu", name="relu_1")(x)
    x = layers.Dropout(0.2, seed=SEED, name="drop_1")(x)

    x = layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4), name="dense_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Activation("relu", name="relu_2")(x)
    x = layers.Dropout(0.2, seed=SEED, name="drop_2")(x)

    x = layers.Dense(32, kernel_regularizer=regularizers.l2(1e-4), name="dense_3")(x)
    x = layers.Activation("relu", name="relu_3")(x)

    out = layers.Dense(1, name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="SST_DNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_dnn(data: dict) -> dict:
    """
    Train the DNN with early stopping, LR scheduling, and model checkpointing.

    Training Strategy
    -----------------
    • 20 % of training data held out as validation set (chronological).
    • EarlyStopping  – halts training when val_loss stops improving
                       (patience=25, restores best weights).
    • ReduceLROnPlateau – halves LR if val_loss stagnates for 10 epochs.
    • ModelCheckpoint – saves best model weights to disk.

    Parameters
    ----------
    data : dict  – output of preprocess_sst()

    Returns
    -------
    dict with model, predictions, metrics, and training history.
    """
    print("\n" + "═"*60)
    print("  STAGE 5 – DEEP NEURAL NETWORK")
    print("═"*60)

    model = build_dnn(input_dim=WINDOW_SIZE)
    model.summary()

    # ── Validation split (last 20 % of train, time-ordered) ──────────────────
    n_train = len(data["X_train"])
    if n_train < 2:
        raise ValueError("Need at least 2 training samples for DNN train/validation split.")

    n_val = max(1, int(n_train * 0.20))
    n_val = min(n_val, n_train - 1)
    X_tr    = data["X_train"][:-n_val]
    X_val   = data["X_train"][-n_val:]
    y_tr    = data["y_train"][:-n_val]
    y_val   = data["y_train"][-n_val:]
    print(f"\n  Train: {len(X_tr)}  |  Val: {len(X_val)}  |  Test: {len(data['X_test'])}")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    cb_early = callbacks.EarlyStopping(
        monitor="val_loss", patience=25, restore_best_weights=True, verbose=1
    )
    cb_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1
    )
    ckpt_path = str(OUTPUT_DIR / "best_dnn_weights.keras")
    cb_ckpt = callbacks.ModelCheckpoint(
        filepath=ckpt_path, save_best_only=True, monitor="val_loss", verbose=0
    )

    # ── Training ──────────────────────────────────────────────────────────────
    print("\n  Training DNN...")
    history = model.fit(
        X_tr, y_tr,
        epochs          = 300,
        batch_size      = 32,
        validation_data = (X_val, y_val),
        callbacks       = [cb_early, cb_lr, cb_ckpt],
        verbose         = 1,
    )
    print(f"  Training stopped at epoch {len(history.history['loss'])}")

    # ── Predictions (re-normalise to °C) ─────────────────────────────────────
    scaler = data["scaler"]

    train_preds_norm = model.predict(data["X_train"], verbose=0).ravel()
    test_preds_norm  = model.predict(data["X_test"],  verbose=0).ravel()

    y_tr_orig    = scaler.inverse_transform(data["y_train"].reshape(-1,1)).ravel()
    y_te_orig    = scaler.inverse_transform(data["y_test"].reshape(-1,1)).ravel()
    tr_pred_orig = scaler.inverse_transform(train_preds_norm.reshape(-1,1)).ravel()
    te_pred_orig = scaler.inverse_transform(test_preds_norm.reshape(-1,1)).ravel()

    # ── Metrics ───────────────────────────────────────────────────────────────
    train_mse  = mean_squared_error(y_tr_orig, tr_pred_orig)
    test_mse   = mean_squared_error(y_te_orig, te_pred_orig)
    train_rmse = np.sqrt(train_mse)
    test_rmse  = np.sqrt(test_mse)
    train_r2   = r2_score(y_tr_orig, tr_pred_orig)
    test_r2    = r2_score(y_te_orig, te_pred_orig)

    print(f"\n  Train  MSE = {train_mse:.6f}  |  RMSE = {train_rmse:.6f}  |  R² = {train_r2:.6f}")
    print(f"  Test   MSE = {test_mse:.6f}  |  RMSE = {test_rmse:.6f}  |  R² = {test_r2:.6f}")

    return {
        "model"       : model,
        "history"     : history,
        "train_preds" : tr_pred_orig,
        "test_preds"  : te_pred_orig,
        "train_mse"   : train_mse,
        "test_mse"    : test_mse,
        "train_rmse"  : train_rmse,
        "test_rmse"   : test_rmse,
        "train_r2"    : train_r2,
        "test_r2"     : test_r2,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 6 – FUTURE FORECAST
# ─────────────────────────────────────────────────────────────────────────────

def forecast_future(
        model,
        scaler: StandardScaler,
        sst_raw: pd.Series,
        n_steps: int = FORECAST_STEPS,
        window:  int = WINDOW_SIZE,
        model_type: str = "DNN",
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Iterative (autoregressive) multi-step-ahead forecast.

    Algorithm
    ---------
    1. Seed the rolling window with the last `window` observed values.
    2. For each future step:
        a. Normalise the window.
        b. Feed through the model → next-step prediction (normalised).
        c. Inverse-transform → °C.
        d. Append prediction to rolling window (drop oldest value).
    3. Return a pd.Series of predicted SSTs indexed by future dates.

    Parameters
    ----------
    model      : fitted sklearn or keras model
    scaler     : fitted StandardScaler
    sst_raw    : full (un-normalised) SST series
    n_steps    : number of months to forecast
    window     : look-back window size
    model_type : "LR" or "DNN" (affects predict call)

    Returns
    -------
    (future_dates, future_preds)  –  DatetimeIndex and numpy array (°C)
    """
    # Initialise rolling buffer with the last observed window
    buffer = np.asarray(sst_raw.to_numpy(dtype=np.float32))[-window:].copy()
    preds  = []

    for _ in range(n_steps):
        x_norm = scaler.transform(buffer.reshape(-1, 1)).ravel()
        if model_type == "DNN":
            p_norm = model.predict(x_norm.reshape(1, -1), verbose=0)[0, 0]
        else:
            p_norm = model.predict(x_norm.reshape(1, -1))[0]
        p_orig = scaler.inverse_transform([[p_norm]])[0, 0]
        preds.append(p_orig)
        # Slide window: drop oldest, append new prediction
        buffer = np.append(buffer[1:], p_orig)

    last_date    = sst_raw.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=n_steps,
        freq="MS",
    )
    return future_dates, np.array(preds, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 7 – VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "actual" : "#2196F3",   # blue
    "lr"     : "#FF9800",   # orange
    "dnn"    : "#4CAF50",   # green
    "future" : "#9C27B0",   # purple
    "train"  : "#E3F2FD",   # light blue fill
    "test"   : "#FFF8E1",   # light yellow fill
}


def plot_sst_trend(sst_series: pd.Series, split_idx: int, window: int) -> str:
    """Figure 1: Global-mean SST trend with train/test shading."""
    fig, ax = plt.subplots(figsize=(14, 4))

    split_date = sst_series.index[split_idx + window]

    ax.axvspan(sst_series.index[0],   split_date,        alpha=0.12,
               color=PALETTE["train"], label="Train period")
    ax.axvspan(split_date,             sst_series.index[-1], alpha=0.12,
               color=PALETTE["test"],  label="Test period")
    ax.axvline(split_date, color="grey", lw=1, ls="--")

    ax.plot(sst_series.index, np.asarray(sst_series.to_numpy(dtype=float)),
            color=PALETTE["actual"], lw=0.9, alpha=0.85, label="Global Mean SST")

    # 12-month rolling mean (trend line)
    rolling = sst_series.rolling(12, center=True).mean()
    ax.plot(rolling.index, np.asarray(rolling.to_numpy(dtype=float)),
            color="red", lw=2.0, ls="-", alpha=0.9, label="12-month rolling mean")

    ax.set_title("Global-Mean Sea Surface Temperature (NOAA ERSST v5)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("SST (°C)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = str(OUTPUT_DIR / "fig1_sst_trend.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_predictions(
        dates: pd.DatetimeIndex,
        split_idx: int,
        y_train_orig: np.ndarray,
        y_test_orig: np.ndarray,
        lr_results: dict,
        dnn_results: dict,
) -> str:
    """Figure 2: Actual vs predicted – both models side by side."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)

    train_dates = dates[:split_idx]
    test_dates  = dates[split_idx:]

    for ax, model_name, color, pred_test in zip(
        axes,
        ["Linear Regression (Baseline)", "Deep Neural Network"],
        [PALETTE["lr"], PALETTE["dnn"]],
        [lr_results["test_preds"], dnn_results["test_preds"]],
    ):
        # Test-period actual
        ax.plot(test_dates, y_test_orig,
                color=PALETTE["actual"], lw=1.5, label="Actual SST", zorder=3)
        # Model prediction
        ax.plot(test_dates, pred_test,
                color=color, lw=1.5, ls="--", label=f"{model_name} Pred", zorder=4)
        # Residual fill
        ax.fill_between(test_dates, y_test_orig, pred_test,
                        alpha=0.20, color=color, label="Residual")

        rmse = np.sqrt(mean_squared_error(y_test_orig, pred_test))
        r2   = r2_score(y_test_orig, pred_test)
        ax.set_title(
            f"{model_name}  —  Test RMSE = {rmse:.4f} °C  |  R² = {r2:.4f}",
            fontsize=12, fontweight="bold"
        )
        ax.set_ylabel("SST (°C)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    fig.suptitle("Actual vs Predicted SST (Test Set)", fontsize=14,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    path = str(OUTPUT_DIR / "fig2_predictions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training_history(history) -> str:
    """Figure 3: DNN training & validation loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    hist = history.history

    axes[0].plot(hist["loss"],     label="Train MSE",  color=PALETTE["dnn"])
    axes[0].plot(hist["val_loss"], label="Val MSE",    color="red", ls="--")
    axes[0].set_title("MSE Loss",  fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale("log")

    axes[1].plot(hist["mae"],      label="Train MAE",  color=PALETTE["dnn"])
    axes[1].plot(hist["val_mae"],  label="Val MAE",    color="red", ls="--")
    axes[1].set_title("MAE",       fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (normalised)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("DNN Training History", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = str(OUTPUT_DIR / "fig3_training_history.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_model_comparison(lr_results: dict, dnn_results: dict) -> str:
    """Figure 4: Side-by-side metric comparison bar chart."""
    metrics = ["Test MSE", "Test RMSE", "Test R²"]
    lr_vals  = [lr_results["test_mse"],  lr_results["test_rmse"],  lr_results["test_r2"]]
    dnn_vals = [dnn_results["test_mse"], dnn_results["test_rmse"], dnn_results["test_r2"]]

    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    b1 = ax.bar(x - w/2, lr_vals,  w, label="Linear Regression",
                color=PALETTE["lr"],  alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + w/2, dnn_vals, w, label="DNN",
                color=PALETTE["dnn"], alpha=0.85, edgecolor="white")

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.0005,
                f"{h:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_title("Model Comparison — Test-Set Metrics", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = str(OUTPUT_DIR / "fig4_model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_future_forecast(
        sst_raw: pd.Series,
        future_dates_lr: pd.DatetimeIndex,
        future_preds_lr: np.ndarray,
        future_dates_dnn: pd.DatetimeIndex,
        future_preds_dnn: np.ndarray,
        context_months: int = 60,
) -> str:
    """Figure 5: Future SST forecast for the next 12 months."""
    fig, ax = plt.subplots(figsize=(14, 5))

    # Historical context (last N months)
    hist = sst_raw.iloc[-context_months:]
    ax.plot(hist.index, np.asarray(hist.to_numpy(dtype=float)),
            color=PALETTE["actual"], lw=1.5, label="Observed SST (last 5 yr)")

    # Forecast ribbons
    ax.plot(future_dates_lr,  future_preds_lr,
            color=PALETTE["lr"],  lw=2, ls="--", marker="o",
            ms=5, label=f"LR Forecast ({FORECAST_STEPS} mo)")
    ax.plot(future_dates_dnn, future_preds_dnn,
            color=PALETTE["dnn"], lw=2, ls="--", marker="s",
            ms=5, label=f"DNN Forecast ({FORECAST_STEPS} mo)")

    # Vertical separator
    ax.axvline(sst_raw.index[-1], color="grey", ls=":", lw=1.5, label="Forecast start")

    ax.set_title(f"SST Forecast — Next {FORECAST_STEPS} Months",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("SST (°C)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = str(OUTPUT_DIR / "fig5_future_forecast.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_residual_analysis(
        dates: pd.DatetimeIndex,
        split_idx: int,
        y_test_orig: np.ndarray,
        lr_results: dict,
        dnn_results: dict,
) -> str:
    """Figure 6: Residual distribution – histograms."""
    test_dates  = dates[split_idx:]
    lr_res  = y_test_orig - lr_results["test_preds"]
    dnn_res = y_test_orig - dnn_results["test_preds"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, res, label, color in zip(
        axes,
        [lr_res, dnn_res],
        ["Linear Regression", "DNN"],
        [PALETTE["lr"], PALETTE["dnn"]],
    ):
        ax.hist(res, bins=30, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(0, color="black", lw=1.5, ls="--")
        ax.set_title(f"{label} Residuals\n"
                     f"μ = {res.mean():.4f} °C   σ = {res.std():.4f} °C",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Residual (°C)")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Residual Distribution (Test Set)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = str(OUTPUT_DIR / "fig6_residuals.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 8 – REPORT & SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_report(
        sst_series: pd.Series,
        lr_results: dict,
        dnn_results: dict,
        future_dates_dnn: pd.DatetimeIndex,
        future_preds_dnn: np.ndarray,
) -> None:
    """Print a structured final performance report to console."""
    separator = "─" * 60

    print("\n" + "═"*60)
    print("  FINAL REPORT")
    print("═"*60)

    print(f"\n  Dataset summary")
    print(separator)
    print(f"  Time range : {sst_series.index[0].date()} → {sst_series.index[-1].date()}")
    print(f"  N months   : {len(sst_series)}")
    print(f"  Mean SST   : {sst_series.mean():.4f} °C")
    print(f"  Std  SST   : {sst_series.std():.4f} °C")

    print(f"\n  Test-Set Metrics")
    print(separator)
    print(f"  {'Metric':<12}  {'Linear Reg':>15}  {'DNN':>15}  {'Winner':>10}")
    print(f"  {'─'*12}  {'─'*15}  {'─'*15}  {'─'*10}")

    metrics = [
        ("MSE (°C²)",  lr_results["test_mse"],  dnn_results["test_mse"],  "lower"),
        ("RMSE (°C)",  lr_results["test_rmse"], dnn_results["test_rmse"], "lower"),
        ("R²",         lr_results["test_r2"],   dnn_results["test_r2"],   "higher"),
    ]
    for name, lr_val, dnn_val, direction in metrics:
        if direction == "lower":
            winner = "DNN ✔" if dnn_val < lr_val else "LR ✔"
        else:
            winner = "DNN ✔" if dnn_val > lr_val else "LR ✔"
        print(f"  {name:<12}  {lr_val:>15.6f}  {dnn_val:>15.6f}  {winner:>10}")

    print(f"\n  12-month DNN Forecast")
    print(separator)
    for date, val in zip(future_dates_dnn, future_preds_dnn):
        print(f"  {date.strftime('%Y-%m')} :  {val:.4f} °C")

    print(f"\n  Interpretation")
    print(separator)
    improvement_rmse = (lr_results["test_rmse"] - dnn_results["test_rmse"]) / \
                        lr_results["test_rmse"] * 100
    if dnn_results["test_rmse"] < lr_results["test_rmse"]:
        winner_text = (
            f"  The DNN outperforms Linear Regression on the test set, "
            f"achieving a {improvement_rmse:.1f}% lower RMSE.\n"
            f"  This improvement arises from the DNN's ability to capture\n"
            f"  non-linear interactions between lagged SST values — such as\n"
            f"  ENSO-driven anomalies — that a linear model cannot represent."
        )
    else:
        winner_text = (
            f"  Linear Regression matches or beats the DNN on the test set.\n"
            f"  This can occur when the signal is predominantly linear and\n"
            f"  the DNN overfits despite regularisation. Increasing data or\n"
            f"  using an LSTM / Transformer may resolve this."
        )
    print(winner_text)

    print(f"\n  Recommendations for Improvement")
    print(separator)
    recs = [
        "• LSTM / GRU – explicitly model temporal dependencies; ideal for time series.",
        "• Transformer (Temporal Fusion Transformer) – attention over long sequences.",
        "• CNN-1D – captures local temporal patterns efficiently.",
        "• Spatial modelling – use the full (lat, lon) grid with ConvLSTM or ViT.",
        "• Multi-variate inputs – add wind stress, thermocline depth, OHC.",
        "• Bayesian uncertainty – quantify forecast confidence intervals.",
        "• Ensemble – combine LR + DNN + LSTM for lower variance.",
    ]
    for r in recs:
        print(f"  {r}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Train SST prediction models using NOAA ERSST v5 data."
    )
    parser.add_argument(
        "--nc-path",
        type=str,
        default=None,
        help="Optional local path to sst.mnmean.nc (if omitted, OPeNDAP link is used).",
    )
    parser.add_argument(
        "--allow-synthetic-fallback",
        action="store_true",
        help="Allow synthetic fallback if remote loading fails.",
    )
    parser.add_argument(
        "--no-download-on-remote-fail",
        action="store_true",
        help="Do not attempt NetCDF file download when OPeNDAP fails.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "█"*60)
    print("  SST PREDICTION PIPELINE  |  NOAA ERSST v5")
    print("  Deep Learning for Climate Science")
    print("█"*60)
    print(f"  OPeNDAP URL: {ERSST_URL}")
    print(f"  Catalog URL: {ERSST_CATALOG_URL}")

    # ── Stage 1: Load data ────────────────────────────────────────────────────
    ds = load_sst_dataset(
        nc_path=args.nc_path,
        use_remote=True,
        allow_synthetic_fallback=args.allow_synthetic_fallback,
        download_on_remote_fail=not args.no_download_on_remote_fail,
    )

    # ── Stage 2: Global average SST ───────────────────────────────────────────
    try:
        sst_series = compute_global_mean_sst(ds)
    except RuntimeError as e:
        # Some OPeNDAP sessions return silently corrupted zero fields.
        if args.nc_path is None and "nearly constant" in str(e):
            print("\n  Remote stream looks corrupted. Downloading full NOAA NetCDF...")
            local_nc = download_ersst_to_cache(force=False)
            ds = load_sst_dataset(
                nc_path=str(local_nc),
                use_remote=False,
                allow_synthetic_fallback=args.allow_synthetic_fallback,
                download_on_remote_fail=False,
            )
            sst_series = compute_global_mean_sst(ds)
        else:
            raise

    # ── Stage 3: Preprocessing ────────────────────────────────────────────────
    data = preprocess_sst(sst_series, window=WINDOW_SIZE, train_ratio=TRAIN_RATIO)
    split_idx   = data["split_idx"]
    dates       = data["dates"]
    scaler      = data["scaler"]

    y_train_orig = scaler.inverse_transform(data["y_train"].reshape(-1,1)).ravel()
    y_test_orig  = scaler.inverse_transform(data["y_test"].reshape(-1,1)).ravel()

    # ── Stage 4: Linear Regression ────────────────────────────────────────────
    lr_results = train_linear_regression(data)

    # ── Stage 5: DNN ──────────────────────────────────────────────────────────
    dnn_results = train_dnn(data)

    # ── Stage 6: Future forecasts ─────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  STAGE 6 – FUTURE FORECAST")
    print("═"*60)

    future_dates_lr,  future_preds_lr  = forecast_future(
        lr_results["model"],  scaler, data["sst_raw"],
        n_steps=FORECAST_STEPS, window=WINDOW_SIZE, model_type="LR"
    )
    future_dates_dnn, future_preds_dnn = forecast_future(
        dnn_results["model"], scaler, data["sst_raw"],
        n_steps=FORECAST_STEPS, window=WINDOW_SIZE, model_type="DNN"
    )
    print("  ✔ Forecasts generated.")

    # ── Stage 7: Visualisations ───────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  STAGE 7 – VISUALISATIONS")
    print("═"*60)

    figs = []
    figs.append(plot_sst_trend(data["sst_raw"], split_idx, WINDOW_SIZE))
    figs.append(plot_predictions(dates, split_idx,
                                 y_train_orig, y_test_orig,
                                 lr_results, dnn_results))
    figs.append(plot_training_history(dnn_results["history"]))
    figs.append(plot_model_comparison(lr_results, dnn_results))
    figs.append(plot_future_forecast(
        data["sst_raw"],
        future_dates_lr,  future_preds_lr,
        future_dates_dnn, future_preds_dnn,
    ))
    figs.append(plot_residual_analysis(
        dates, split_idx, y_test_orig, lr_results, dnn_results
    ))

    for f in figs:
        print(f"  Saved: {f}")

    # ── Stage 8: Report ───────────────────────────────────────────────────────
    print_report(data["sst_raw"], lr_results, dnn_results,
                 future_dates_dnn, future_preds_dnn)

    # ── Save metrics to JSON ──────────────────────────────────────────────────
    results_summary = {
        "linear_regression": {
            "test_mse" : float(round(float(lr_results["test_mse"]), 6)),
            "test_rmse": float(round(float(lr_results["test_rmse"]), 6)),
            "test_r2"  : float(round(float(lr_results["test_r2"]),   6)),
        },
        "dnn": {
            "test_mse" : float(round(float(dnn_results["test_mse"]),  6)),
            "test_rmse": float(round(float(dnn_results["test_rmse"]), 6)),
            "test_r2"  : float(round(float(dnn_results["test_r2"]),   6)),
        },
        "dnn_forecast_12m": {
            str(d.date()): float(round(float(v), 4))
            for d, v in zip(future_dates_dnn, future_preds_dnn)
        },
    }
    json_path = str(OUTPUT_DIR / "results_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n  Results JSON saved: {json_path}")
    print("\n  Pipeline complete. ✔")


if __name__ == "__main__":
    main()
