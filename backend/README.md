# Backend Structure

- `src/`: reusable code modules (`model`, `data_loader`, `augmentation`)
- `scripts/`: runnable entrypoints (`train`, `test`, `split_dataset`, `infer_image`)
- `data/`: dataset (`raw` and `clean`)
- `checkpoints/`: saved model weights
- `notebooks/`: exploratory notebooks
- `docs/`: notes and TODO files

## Run Commands

From project root:

```powershell
python -m backend.scripts.train
python -m backend.scripts.test
python -m backend.scripts.infer_image <image_path>
python -m backend.scripts.split_dataset
```
