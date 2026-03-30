# Backend Structure

- `src/`: reusable code modules (`model`, `data_loader`, `augmentation`)
- `scripts/`: runnable entrypoints (`train`, `test`, `split_dataset`, `infer_image`, `train_timeseries`, `predict_energy`)
- `data/`: dataset (`raw` and `clean`)
- `artifacts/`: saved model weights, metrics, and forecasts
- `notebooks/`: exploratory notebooks
- `docs/`: notes and TODO files

## Run Commands

From project root:

```powershell
python -m backend.scripts.train
python -m backend.scripts.test
python -m backend.scripts.infer_image <image_path>
python -m backend.scripts.split_dataset
python -m backend.scripts.train_timeseries
python -m backend.scripts.predict_energy <prepared_feature_csv>
```

## Website Handoff

For the vision model, share the classifier checkpoint at `backend/artifacts/vision/checkpoints/best_model.pth` plus the class labels from the image dataset folders.

For the energy model, first run:

```powershell
python -m backend.scripts.train_timeseries
```

That will produce:

- `backend/artifacts/timeseries/checkpoints/energy_forecast_model.joblib`
- `backend/artifacts/timeseries/metrics/energy_forecast_metrics.json`
- `backend/artifacts/timeseries/forecasts/energy_forecast_test_predictions.csv`

For website integration, your friend mainly needs the trained model file, the required feature columns from `backend/src/timeseries.py`, and a small prediction wrapper like `backend/scripts/predict_energy.py`.
