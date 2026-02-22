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

Broader worldwide preset (adds more India/China/Africa/South-America):

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
uv add streamlit
uv run streamlit run app/app.py
```

## Recommended Iteration Loop

1. Increase dataset coverage (more countries/regions)
2. Tune `--cell-size-deg` and `--min-class-count`
3. Try stronger backbones (`--model-name`, e.g. `efficientnet_b0`, `convnext_tiny`)
4. Compare augmentations, batch size, learning rate, and epochs
5. Track results by keeping `metrics.json` per run directory
