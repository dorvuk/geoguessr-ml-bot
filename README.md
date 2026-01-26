# GeoGuessr ML Bot (Image Geolocation)

A machine learning project that predicts the geographic location of a “Street View–style” screenshot.
The goal is to build a robust geolocation system that performs well in both urban and rural environments by combining:

- **Region classification** (coarse prediction: country/region/grid cell)
- **Retrieval (embedding + nearest neighbors)** for fine-grained refinement
- Optionally, **regression** to estimate latitude/longitude offsets

This repository is the starting point for a thesis project and will evolve during development.

## Core Idea

Pure classification can be fast but too coarse, while pure retrieval can be brittle.  
This project explores **hybrid geolocation**:

1. A classifier produces a **rough guess** (e.g., country/region or a geographic grid cell).
2. An embedding model retrieves **similar geotagged images** from an indexed database.
3. The final prediction combines both signals for improved accuracy and robustness.

## Data (Planned)

We plan to use **geotagged, legally usable datasets** such as:
- Flickr YFCC100M, Im2GPS
- Mapillary / OpenStreetCam or other open street-level imagery sources

If any Google Street View / GeoGuessr-like imagery is used, it must be done **strictly within Terms of Service and licensing constraints** (e.g., via permitted APIs and usage rules).

Data is not stored in this repository.

## Methods (Planned)

- **Baseline**: transfer learning (e.g., ResNet/EfficientNet/ViT) for classification on a discrete geographic grid (e.g., S2 cells / geohash).
- **Retrieval**: CLIP or ViT-based embeddings + nearest-neighbor search (FAISS) over geotagged reference images.
- **Hybrid**: combine classifier output with retrieval results.
- Optional: multi-task learning (classification + coordinate regression).

## Evaluation (Planned)

Metrics may include:
- Country / region accuracy
- Accuracy within distance thresholds (e.g., 25/50/200/750 km)
- Median geolocation error (km)
- Top-k retrieval accuracy
- Separate reporting for **urban vs rural** subsets

## Tech Stack

- Python
- PyTorch + Hugging Face ecosystem
- FAISS for retrieval indexing/search
- Docker (optional) for reproducibility
- Simple demo UI (Streamlit or Flask)

## Repository Setup

### Requirements
- Python 3.11+

### Install (uv)
```bash
# create venv + install deps
uv venv
uv sync
