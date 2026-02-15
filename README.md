# GeoGuessr ML Bot (Geolocation Baseline)

This repository now includes a working baseline pipeline for:

1. Downloading geotagged Mapillary images
2. Running metadata/image quality checks
3. Building deterministic train/val/test splits with grid-cell labels
4. Training a baseline image classifier for geolocation

## Requirements

- Python 3.11+
- `uv` (recommended) or another virtual environment manager

## Install

```bash
uv venv
uv sync
```

## Data Pipeline

### 1. Download Mapillary data

`MAPILLARY_TOKEN` must be set in `.env` or your shell.

```bash
python scripts/download_mapillary.py \
  --bbox=13.5,42.0,19.5,46.6 \
  --target 5000 \
  --samples 3000 \
  --search-limit 200
```

Important downloader features:

- Retries with exponential backoff (`--max-retries`, `--retry-backoff`)
- Structured error log (`--error-csv`)
- Schema validation/migration for metadata
- Relative file paths in metadata
- Sequence-level cap (`--max-per-sequence`)

### 2. Run quality control

```bash
geobot qc \
  --metadata-csv data/raw/mapillary/metadata.csv \
  --cleaned-csv data/processed/mapillary/metadata_clean.csv \
  --report-json data/processed/mapillary/qc_report.json
```

This removes unusable rows and writes a QC report with counts for:

- duplicates
- invalid coordinates
- missing files
- unreadable images

### 3. Prepare labeled dataset

```bash
geobot prepare \
  --metadata-csv data/processed/mapillary/metadata_clean.csv \
  --output-csv data/processed/mapillary/dataset.csv \
  --cell-size-deg 1.0 \
  --min-class-count 5 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42
```

This produces:

- `data/processed/mapillary/dataset.csv`
- `data/processed/mapillary/label_map.json`
- `data/processed/mapillary/dataset_summary.json`

### 4. Train baseline classifier

```bash
geobot train \
  --dataset-csv data/processed/mapillary/dataset.csv \
  --out-dir models/classifier_baseline \
  --model-name resnet18 \
  --epochs 10 \
  --batch-size 32 \
  --image-size 224 \
  --device auto
```

Training artifacts:

- `models/classifier_baseline/best.pt`
- `models/classifier_baseline/history.csv`
- `models/classifier_baseline/metrics.json`

`metrics.json` includes:

- top-1 accuracy
- median geolocation error (km) using class centroids
- threshold accuracy (`<=25km`, `<=50km`, `<=200km`, `<=750km`)

## Recommended Iteration Loop

1. Increase dataset coverage (more countries/regions)
2. Tune `--cell-size-deg` and `--min-class-count`
3. Try stronger backbones (`--model-name`, e.g. `efficientnet_b0`, `convnext_tiny`)
4. Compare augmentations, batch size, learning rate, and epochs
5. Track results by keeping `metrics.json` per run directory
