# Backend Structure

- `src/vision/`: image classification pipeline (train/evaluate/infer/split)
- `src/timeseries/`: energy prediction and residual-based fault detection
- `src/common/`: shared helpers
- `scripts/vision/`: explicit vision entrypoints
- `scripts/timeseries/`: explicit timeseries entrypoints
- `data/vision/`: image data (`raw`, `processed`)
- `data/timeseries/`: timeseries data (`raw`, `processed`, `features`)
- `artifacts/vision/`: vision checkpoints and metrics
- `artifacts/timeseries/`: timeseries checkpoints, forecasts, and metrics
- `configs/`: task config templates
- `notebooks/vision` and `notebooks/timeseries`: task-specific notebooks

## Vision Commands

```powershell
python -m backend.scripts.vision.train_vision
python -m backend.scripts.vision.evaluate_vision
python -m backend.scripts.vision.infer_vision "backend\data\vision\processed\test\Bird-drop\Bird (3).jpg"
python -m backend.scripts.vision.split_dataset
```

## Timeseries Commands

```powershell
python -m backend.scripts.timeseries.preprocess_timeseries --generation generation.csv --irradiance irradiance.csv --weather weather.csv
python -m backend.scripts.timeseries.train_timeseries
python -m backend.scripts.timeseries.infer_timeseries
python -m backend.scripts.timeseries.run_fault_detection
```

## Backward Compatibility

Older commands still work:

```powershell
python -m backend.scripts.train
python -m backend.scripts.test
python -m backend.scripts.infer_image "<image_path>"
python -m backend.scripts.split_dataset
```
