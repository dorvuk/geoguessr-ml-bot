# GeoGuessr ML Bot (Geolocation Baseline)

This repository includes a working pipeline for:

1. Downloading geotagged Mapillary images
2. Running metadata/image quality checks
3. Building deterministic train/val/test splits with grid-cell labels
4. Training classifier + retrieval geolocation models
5. Running a Streamlit app for interactive predictions

## Requirements

- `Python 3.11+`
- `uv` (pip install uv)
- `git-lfs`

## Install

```bash
uv venv
uv sync
```

## Remote App Setup

1. Clone the repo:

```bash
git clone https://github.com/dorvuk/geoguessr-ml-bot.git
cd geoguessr-ml-bot
```

2. Install and pull LFS model files:

```bash
git lfs install
git lfs pull
```

3. Install Python dependencies:

```bash
uv venv
uv sync
```

4. Launch the app:

```bash
uv run streamlit run app/app.py
```

## What Gets Pulled Via LFS

The repo tracks serving artifacts in Git LFS, including:

- `models/classifier_fine_effb0_50k/best.pt` (primary)
- `models/retrieval/fine_effb0_50k_train.faiss`
- `models/retrieval/fine_effb0_50k_train_meta.npz`
- `models/classifier_fine_effb0_v1/best.pt` (fallback/alternative)
- `models/retrieval/fine_effb0_train.faiss`
- `models/retrieval/fine_effb0_train_meta.npz`

You can verify on any machine:

```bash
git lfs ls-files
```

## App Model Selection

In the app sidebar:

- `Primary Model Profile`: default is `fine_effb0_50k (best)`
- `Enable Auto Fallback Model`: optional switch to `fine_effb0_20k` when confidence is low
- `Low-confidence warning (<)`: threshold that warns user prediction is uncertain
- `Override ... paths`: point app to custom checkpoints/indexes if you train new models

## Troubleshooting Remote Setup

- `missing file` errors in app:
  Run `git lfs pull` and verify files under `models/`.
- `git lfs` command not found:
  Install Git LFS first, then run `git lfs install`.
- app starts but predictions fail:
  Confirm model paths in sidebar match real files.
- no GPU on remote machine:
  Set app device to `cpu` in sidebar.

## Data Pipeline

### 1. Download Mapillary data

`MAPILLARY_TOKEN` must be set in `.env` or your shell.

```bash
python scripts/download_mapillary.py \
  --bbox=13.5,42.0,19.5,46.6 \
  --target 5000 \
  --samples 3000 \
  --half-size-deg 0.01 \
  --search-limit 200
```

Global preset (popular GeoGuessr countries):

```bash
python scripts/download_mapillary.py \
  --country-preset popular \
  --target 12000 \
  --samples 10000 \
  --half-size-deg 0.01 \
  --search-limit 150
```

Broader worldwide preset (adds more India/China/Africa/SouthAmerica):

```bash
python scripts/download_mapillary.py \
  --country-preset global_balanced \
  --target 50000 \
  --samples 120000 \
  --half-size-deg 0.01 \
  --search-limit 250 \
  --per-search-max-new 80 \
  --download-workers 24
```

Custom global subset:

```bash
python scripts/download_mapillary.py \
  --countries us,ca,gb,de,fr,es,it,jp,au,nz,br \
  --target 8000 \
  --samples 7000 \
  --half-size-deg 0.01 \
  --search-limit 150
```

List built-in country codes:

```bash
python scripts/download_mapillary.py --list-countries
```

Important downloader features:

- Single `/images` search call returns metadata + thumbnail URLs (no per-image detail call)
- Global country sampling (`--country-preset` or `--countries`)
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

### 5. Build retrieval baseline (FAISS)

This gives you a nearest-neighbor geolocation baseline from the trained classifier features.

Extract embeddings for all splits:

```bash
uv run geobot embed --dataset-csv data/processed/mapillary/dataset.csv --checkpoint models/classifier_baseline/best.pt --out-npz models/retrieval/embeddings_all.npz --splits train,val,test --image-size 224 --batch-size 64 --device cuda --amp
```

Build a FAISS index from train split embeddings:

```bash
uv run geobot index --embeddings-npz models/retrieval/embeddings_all.npz --index-out models/retrieval/index.faiss --meta-out models/retrieval/index_meta.npz --summary-json models/retrieval/index_summary.json --splits train --metric cosine
```

Evaluate retrieval on test split:

```bash
uv run geobot eval-retrieval --query-embeddings-npz models/retrieval/embeddings_all.npz --index-path models/retrieval/index.faiss --index-meta-npz models/retrieval/index_meta.npz --query-splits test --k 1 --metric auto --out-json models/retrieval/retrieval_metrics.json --per-query-csv models/retrieval/retrieval_predictions_test.csv
```

Outputs:

- `models/retrieval/retrieval_metrics.json` (aggregate retrieval metrics)
- `models/retrieval/retrieval_predictions_test.csv` (per-image errors)
- `models/retrieval/index.faiss` and `models/retrieval/index_meta.npz` (index artifacts)

### 6. Run single-image prediction

```bash
uv run geobot predict --image path/to/query.jpg --checkpoint models/classifier_fine_effb0_50k/best.pt --index-path models/retrieval/fine_effb0_50k_train.faiss --index-meta-npz models/retrieval/fine_effb0_50k_train_meta.npz --image-size 224 --k 1 --show-neighbors 5 --device cuda --amp --json-out models/retrieval/last_prediction.json
```

Outputs:

- terminal prediction (`pred_lat`, `pred_lon`, confidence, nearest neighbors)
- optional JSON file (`--json-out`)

### 7. Launch simple UI (Streamlit)

```bash
uv run streamlit run app/app.py
```
