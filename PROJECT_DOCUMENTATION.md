# SST Prediction Project Documentation

## 1. Project Goal

This project predicts Sea Surface Temperature (SST) for one selected ocean point in the central equatorial Indian Ocean.

The selected point is:

- Latitude: 0.0
- Longitude: 90.0E
- Region meaning: central equatorial Indian Ocean

This location was chosen because the project is for a fisheries-related study. Fisheries work is usually more useful when it focuses on a specific ocean region instead of the whole global ocean.

## 2. Dataset Used

The project uses the real NOAA ERSST monthly SST dataset:

```text
data/sst.mnmean.nc
```

This is a NetCDF file. NetCDF is a common file format used for climate and ocean data.

The code does not use mock or synthetic data.

## 3. What We Did

The workflow is:

1. Load the NOAA ERSST dataset.
2. Select only the SST values at latitude 0.0 and longitude 90.0E.
3. Convert that point into a monthly time series.
4. Use data before 2000 for training.
5. Train a Deep Neural Network (DNN).
6. Predict SST from 2000 to 2025.
7. Compare the predicted SST with actual observed SST from 2000 to 2025.
8. Save plots, a CSV comparison table, and a JSON summary.

## 4. Training and Testing Periods

The model is trained using historical data before the year 2000.

```text
Training period: 1855-01 to 1999-12
Prediction period: 2000-01 to 2025-12
```

The first year in training becomes 1855 instead of 1854 because the model uses the previous 12 months to predict the next month. It needs 12 months of history before it can create the first training example.

## 5. How the DNN Predicts

The model uses the previous 12 months of SST to predict the next month.

Example:

```text
Input : SST from Jan 1999 to Dec 1999
Output: SST for Jan 2000
```

For the 2000-2025 prediction, the code uses recursive prediction.

Recursive prediction means:

1. Predict Jan 2000.
2. Use that predicted value as part of the next input window.
3. Predict Feb 2000.
4. Continue month by month until Dec 2025.

This is harder than normal one-step prediction, but it better matches the idea of training on old data and forecasting into the future.

## 6. Important Code Sections

The main file is:

```text
sst-forecasting-dnn/sst_prediction_pipeline.py
```

### Project Settings

At the top of the file, these values control the experiment:

```python
TARGET_LAT = 0.0
TARGET_LON = 90.0
WINDOW_SIZE = 12
TRAIN_END = "1999-12-01"
TEST_START = "2000-01-01"
TEST_END = "2025-12-01"
```

Meaning:

- `TARGET_LAT` and `TARGET_LON` select the ocean point.
- `WINDOW_SIZE = 12` means the model uses the previous 12 months.
- `TEST_START` and `TEST_END` define the prediction/comparison period.

### `load_point_sst()`

This function loads the NetCDF file and extracts SST at 0.0, 90.0E.

Important line:

```python
point = ds["sst"].sel(lat=lat, lon=lon, method="nearest")
```

This selects the closest dataset grid point to the requested latitude and longitude. In this dataset, the exact point 0.0 and 90.0E exists.

### `make_supervised_data()`

This function converts the SST time series into machine-learning input and output.

For every month, it creates:

```text
X = previous 12 months of SST
y = next month SST
```

It also separates the data into:

- training data before 2000
- testing/comparison data from 2000 to 2025

It uses `StandardScaler` so the DNN trains more smoothly.

### `build_dnn()`

This function defines the neural network.

The DNN has:

- input layer with 12 values
- dense hidden layers
- dropout layers to reduce overfitting
- output layer with 1 value: predicted SST

### `train_dnn()`

This function trains the DNN using the pre-2000 data only.

It also uses early stopping. Early stopping prevents the model from training too long when validation performance stops improving.

### `recursive_hindcast()`

This is the main prediction function for 2000-2025.

It predicts month by month from Jan 2000 to Dec 2025. Each prediction becomes part of the next prediction input.

### `calculate_metrics()`

This function calculates the model error:

- MSE: Mean Squared Error
- RMSE: Root Mean Squared Error
- MAE: Mean Absolute Error
- R2: coefficient of determination

For a simple explanation, RMSE is usually the easiest metric to discuss. It tells the average prediction error in degrees Celsius.

### Plot Functions

The script creates four plots:

```text
fig1_selected_point_sst.png
fig2_dnn_prediction_vs_actual.png
fig3_training_history.png
fig4_prediction_error.png
```

These are saved in the `results` folder.

## 7. Output Files

After running the script, important outputs are saved in:

```text
results/
```

Important files:

```text
results_summary.json
hindcast_2000_2025_comparison.csv
fig1_selected_point_sst.png
fig2_dnn_prediction_vs_actual.png
fig3_training_history.png
fig4_prediction_error.png
dnn_sst_point_model.keras
```

### `hindcast_2000_2025_comparison.csv`

This is the easiest file to open in Excel. It contains:

- date
- actual SST
- predicted SST
- error

## 8. How to Run the Project

First install the dependencies:

```powershell
pip install -r requirement.txt
```

Then run the script:

```powershell
python sst-forecasting-dnn/sst_prediction_pipeline.py
```

Make sure this file exists before running:

```text
data/sst.mnmean.nc
```

## 9. Simple Explanation for the Report

This study used monthly NOAA ERSST sea surface temperature data at the central equatorial Indian Ocean point 0.0 latitude and 90.0E longitude. A deep neural network was trained using historical SST data before 2000. The trained model then predicted SST from 2000 to 2025. The predictions were compared with actual observed SST values from the same period to evaluate model performance.

The model used the previous 12 months of SST as input to predict the next month. This allowed the DNN to learn seasonal and temporal SST patterns at the selected ocean point.

## 10. Notes and Limitations

- This model uses only SST from one location.
- It does not use other ocean or climate variables such as wind, salinity, chlorophyll, rainfall, or ocean currents.
- Recursive prediction from 2000 to 2025 is difficult because errors can accumulate over time.
- For better fisheries forecasting, future work could add more environmental variables and nearby grid points.
