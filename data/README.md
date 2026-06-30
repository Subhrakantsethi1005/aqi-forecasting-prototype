# Data

Place the full dataset here as:

```text
data/aqi_dataset.csv
```

The original notebook used Google Drive file ID:

```text
1-JdY0p0PHt34VHlULz8RkR9B62k-Futb
```

The source metadata lives in `dataset_sources.json`. If redistribution is allowed, you can download it with:

```bash
pip install gdown
python scripts/download_dataset.py
```

The CSV should include a datetime column plus pollutant and weather columns.

The repository includes `sample_aqi_dataset.csv` only as a tiny format example. It is not enough for serious model training.

Large datasets should not be committed directly unless the license allows redistribution and the file size is reasonable. Prefer GitHub Releases, Kaggle, Zenodo, Google Drive, or Hugging Face Datasets.
