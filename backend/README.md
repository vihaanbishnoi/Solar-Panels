# Solar Panel Fault Detection Backend

This backend contains two ML workflows:

- `timeseries/` for solar energy forecasting from weather and generation data
- `vision/` for solar panel fault classification from images

The notebook work lives under `notebooks/`, and the production-style Python code lives in feature-based folders.

## Folder Structure

```text
backend/
  artifacts/
    timeseries/
      checkpoints/
      forecasts/
      metrics/
    vision/
      checkpoints/
      metrics/
  data/
    timeseries/
      raw/
      processed/
      features/
    vision/
      raw/
      processed/
  notebooks/
    timeseries/
    vision/
  timeseries/
    pipeline.py
    train.py
    predict.py
  vision/
    augmentation.py
    data_loader.py
    model.py
    split_dataset.py
    train.py
    test.py
    infer_image.py
```

## Time-Series Module

The time-series workflow is the Python version of the Jupyter notebook.

- `timeseries/pipeline.py`
  Contains the reusable logic from the notebook:
  loading data, preprocessing solar data, preprocessing weather data, feature engineering, train/test split, model training, evaluation, and saving outputs.
- `timeseries/train.py`
  Runs the full forecasting pipeline end to end.
  This is the script equivalent of "Run all cells" for the training part of the notebook.
- `timeseries/predict.py`
  Loads a trained model and generates predictions from a prepared feature CSV.

### Run Time-Series Training

From the project root:

```powershell
.\venv\Scripts\python.exe -m backend.timeseries.train
```

Outputs are saved to:

- `backend/artifacts/timeseries/checkpoints/`
- `backend/artifacts/timeseries/metrics/`
- `backend/artifacts/timeseries/forecasts/`

### Run Time-Series Prediction

```powershell
.\venv\Scripts\python.exe -m backend.timeseries.predict backend\artifacts\timeseries\forecasts\energy_forecast_test_predictions.csv
```

## Vision Module

The vision workflow handles solar panel image classification.

- `vision/augmentation.py`
  Image transforms for training and evaluation.
- `vision/data_loader.py`
  Loads image datasets from the processed train/val/test folders.
- `vision/model.py`
  Defines the ResNet50-based classifier.
- `vision/split_dataset.py`
  Splits raw images into train/val/test folders.
- `vision/train.py`
  Trains the classifier and saves the best checkpoint.
- `vision/test.py`
  Evaluates the trained model on the test set.
- `vision/infer_image.py`
  Predicts the class of a single image.

### Run Vision Dataset Split

```powershell
.\venv\Scripts\python.exe -m backend.vision.split_dataset
```

### Run Vision Training

```powershell
.\venv\Scripts\python.exe -m backend.vision.train
```

### Run Vision Evaluation

```powershell
.\venv\Scripts\python.exe -m backend.vision.test
```

### Run Single Image Inference

```powershell
.\venv\Scripts\python.exe -m backend.vision.infer_image path\to\image.jpg
```

## Data and Artifacts

- `data/` stores input datasets
- `artifacts/` stores generated outputs such as models, metrics, and forecasts
- `notebooks/` stores Jupyter notebooks for experimentation and explanation

## Git Notes

- Notebook files are ignored by `.gitignore`
- `backend/data/` is ignored because datasets are usually large
- `backend/artifacts/` is ignored because trained models and outputs are generated files
