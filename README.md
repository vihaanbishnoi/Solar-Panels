# Solar Panel Fault Detection

This project focuses on two machine learning tasks for solar panel systems:

- time-series forecasting of solar energy generation
- vision-based fault classification from solar panel images

The codebase is organized so that notebook experimentation and normal Python code are separated cleanly.

## Project Structure

```text
Solar Panel Fault Detection/
  backend/
    artifacts/
    data/
    notebooks/
    timeseries/
    vision/
    README.md
  .gitignore
  README.md
```

## Main Modules

### Time-Series

The time-series module converts the Jupyter notebook workflow into reusable Python code.

- `backend/timeseries/pipeline.py`
  Contains data loading, preprocessing, feature engineering, training, evaluation, and saving logic
- `backend/timeseries/train.py`
  Runs the full forecasting pipeline
- `backend/timeseries/predict.py`
  Loads a trained model and generates predictions from prepared feature data
- `backend/timeseries/predict_one_day_input.py`
  Takes interactive user input for one row (one day/time point) and predicts output

### Vision

The vision module handles image-based solar panel fault detection.

- `backend/vision/augmentation.py`
- `backend/vision/data_loader.py`
- `backend/vision/model.py`
- `backend/vision/split_dataset.py`
- `backend/vision/train.py`
- `backend/vision/test.py`
- `backend/vision/infer_image.py`

## Run the Project

From the project root:

### Time-Series Training

```powershell
.\venv\Scripts\python.exe -m backend.timeseries.train
```

### Time-Series Prediction

```powershell
.\venv\Scripts\python.exe -m backend.timeseries.predict backend\artifacts\timeseries\forecasts\energy_forecast_test_predictions.csv
```

### Time-Series One-Row User Input Prediction

```powershell
.\venv\Scripts\python.exe -m backend.timeseries.predict_one_day_input
```

### Vision Dataset Split

```powershell
.\venv\Scripts\python.exe -m backend.vision.split_dataset
```

### Vision Training

```powershell
.\venv\Scripts\python.exe -m backend.vision.train
```

### Vision Testing

```powershell
.\venv\Scripts\python.exe -m backend.vision.test
```

## Notes

- `backend/notebooks/` contains the Jupyter notebooks used for experimentation
- `backend/data/` stores datasets and is ignored by Git
- `backend/artifacts/` stores generated models, metrics, and predictions and is also ignored by Git
- notebook files are ignored in `.gitignore`

For more backend-specific details, see `backend/README.md`.
