import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from geobot.io_utils import resolve_path, write_json
from geobot.retrieval_utils import (
    build_feature_model,
    choose_device,
    extract_embeddings,
    normalize_rows,
    weighted_geo_mean,
)


def register_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("predict", help="Predict image geolocation using retrieval index")
    parser.add_argument("--repo-root", default=".", help="Repository root directory")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--checkpoint", default="models/classifier_baseline/best.pt")
    parser.add_argument("--index-path", default="models/retrieval/index.faiss")
    parser.add_argument("--index-meta-npz", default="models/retrieval/index_meta.npz")
    parser.add_argument("--model-name", default=None, help="Optional timm model name override")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--k", type=int, default=1, help="Neighbors for final geolocation estimate")
    parser.add_argument("--show-neighbors", type=int, default=5, help="How many neighbors to print")
    parser.add_argument("--metric", choices=["auto", "cosine", "l2"], default="auto")
    parser.add_argument(
        "--cosine-temperature",
        type=float,
        default=20.0,
        help="Only used when --metric cosine and k>1",
    )
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument("--amp", action="store_true", help="Enable AMP on CUDA")
    parser.add_argument("--json-out", default="", help="Optional JSON output path")
    parser.set_defaults(func=run)


def load_index_meta(path: Path) -> dict[str, Any]:
    required = ["image_id", "path", "lat", "lon", "metric"]
    with np.load(path, allow_pickle=False) as payload:
        missing = [k for k in required if k not in payload]
        if missing:
            raise RuntimeError(f"index meta npz missing keys: {missing}")
        out = {k: payload[k] for k in payload.files}

    out["image_id"] = np.asarray(out["image_id"]).astype(str)
    out["path"] = np.asarray(out["path"]).astype(str)
    out["lat"] = np.asarray(out["lat"], dtype=np.float64)
    out["lon"] = np.asarray(out["lon"], dtype=np.float64)
    out["metric"] = str(np.asarray(out["metric"]).astype(str).reshape(-1)[0]).strip().lower()
    return out


def weights_from_scores(scores: np.ndarray, metric: str, cosine_temperature: float) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    if metric == "cosine":
        logits = (s - float(np.max(s))) * cosine_temperature
        logits = np.clip(logits, -60.0, 60.0)
        w = np.exp(logits)
    else:
        d = np.maximum(s, 0.0)
        w = 1.0 / (d + 1e-6)
    if not np.any(np.isfinite(w)) or float(w.sum()) <= 0.0:
        return np.ones_like(s)
    return w


def confidence_from_scores(scores: np.ndarray, metric: str) -> tuple[float, float, float]:
    s = np.asarray(scores, dtype=np.float64)
    top1 = float(s[0])
    top2 = float(s[1]) if s.size >= 2 else top1

    if metric == "cosine":
        # Heuristic confidence, not calibrated probability.
        similarity = np.clip((top1 + 1.0) / 2.0, 0.0, 1.0)
        margin = np.clip(top1 - top2, 0.0, 1.0)
        confidence = float(0.7 * similarity + 0.3 * margin)
    else:
        # For L2, lower distance means better match.
        dist_strength = float(1.0 / (1.0 + max(top1, 0.0)))
        margin = float(max(top2 - top1, 0.0) / (1.0 + max(top2 - top1, 0.0)))
        confidence = float(0.7 * dist_strength + 0.3 * margin)
    return confidence, top1, top2


def run(args: argparse.Namespace) -> int:
    if sys.platform == "darwin":
        # macOS wheels for torch/faiss may ship separate OpenMP runtimes.
        # Allow coexistence to avoid aborting during second runtime init.
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    try:
        import timm
        import torch
        from torchvision import transforms
        import faiss
    except ImportError as e:
        raise RuntimeError("Missing dependency for prediction. Run `uv sync`.") from e

    if args.k <= 0:
        raise ValueError("--k must be >= 1")
    if args.show_neighbors <= 0:
        raise ValueError("--show-neighbors must be >= 1")

    repo_root = resolve_path(args.repo_root, Path.cwd())
    image_path = resolve_path(args.image, repo_root)
    checkpoint = resolve_path(args.checkpoint, repo_root)
    index_path = resolve_path(args.index_path, repo_root)
    index_meta_npz = resolve_path(args.index_meta_npz, repo_root)
    json_out = resolve_path(args.json_out, repo_root) if args.json_out else None

    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not index_path.exists():
        raise FileNotFoundError(f"index not found: {index_path}")
    if not index_meta_npz.exists():
        raise FileNotFoundError(f"index meta not found: {index_meta_npz}")

    device = choose_device(args.device, torch)
    use_amp = bool(args.amp and device.type == "cuda")

    model_name, model, _ = build_feature_model(
        timm_mod=timm,
        torch_mod=torch,
        checkpoint_path=checkpoint,
        model_name_override=args.model_name,
        device=device,
    )

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    with Image.open(image_path) as img:
        x = transform(img.convert("RGB")).unsqueeze(0)
    x = x.to(device, non_blocking=True)

    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=use_amp):
            emb = extract_embeddings(model, x, torch)
    query_vec = emb.detach().cpu().numpy().astype(np.float32)

    index_meta = load_index_meta(index_meta_npz)
    index = faiss.read_index(str(index_path))
    if int(index.ntotal) != int(index_meta["image_id"].shape[0]):
        raise RuntimeError(
            f"index/meta size mismatch: index.ntotal={index.ntotal}, meta_rows={index_meta['image_id'].shape[0]}"
        )

    index_metric = str(index_meta["metric"])
    metric = index_metric if args.metric == "auto" else args.metric
    if metric != index_metric:
        raise RuntimeError(f"metric mismatch: requested={metric}, index_meta={index_metric}")

    if metric == "cosine":
        query_vec = normalize_rows(query_vec).astype(np.float32)

    search_k = min(int(index.ntotal), max(args.k, args.show_neighbors))
    if search_k <= 0:
        raise RuntimeError("index has no vectors")

    scores, neighbors = index.search(query_vec, search_k)
    scores = np.asarray(scores[0], dtype=np.float64)
    neighbors = np.asarray(neighbors[0], dtype=np.int64)
    valid = neighbors >= 0
    scores = scores[valid]
    neighbors = neighbors[valid]
    if neighbors.size == 0:
        raise RuntimeError("no neighbors returned by index")

    est_k = min(args.k, neighbors.size)
    est_idx = neighbors[:est_k]
    est_scores = scores[:est_k]
    est_lat = np.asarray(index_meta["lat"], dtype=np.float64)[est_idx]
    est_lon = np.asarray(index_meta["lon"], dtype=np.float64)[est_idx]
    weights = weights_from_scores(est_scores, metric=metric, cosine_temperature=args.cosine_temperature)
    pred_lat, pred_lon = weighted_geo_mean(est_lat, est_lon, weights)

    conf, top1, top2 = confidence_from_scores(scores[: max(1, min(2, scores.size))], metric=metric)
    metric_key = "similarity" if metric == "cosine" else "distance"
    margin = float(top1 - top2) if metric == "cosine" else float(top2 - top1)

    shown = min(args.show_neighbors, neighbors.size)
    rows = []
    for rank in range(shown):
        idx = int(neighbors[rank])
        rows.append(
            {
                "rank": rank + 1,
                "image_id": str(index_meta["image_id"][idx]),
                "path": str(index_meta["path"][idx]),
                "lat": float(index_meta["lat"][idx]),
                "lon": float(index_meta["lon"][idx]),
                metric_key: float(scores[rank]),
            }
        )

    output = {
        "image": str(image_path),
        "checkpoint": str(checkpoint),
        "model_name": model_name,
        "device": str(device),
        "index_path": str(index_path),
        "index_meta_npz": str(index_meta_npz),
        "metric": metric,
        "k": int(est_k),
        "pred_lat": float(pred_lat),
        "pred_lon": float(pred_lon),
        "confidence": float(conf),
        "confidence_note": "Heuristic confidence, not calibrated probability.",
        "top1_score": float(top1),
        "top2_score": float(top2),
        "top12_margin": float(margin),
        "neighbors": rows,
    }

    print(f"Predicted lat/lon: {pred_lat:.6f}, {pred_lon:.6f}")
    print(f"Confidence (heuristic): {conf:.3f}")
    print(f"Top-1 {metric_key}: {top1:.6f}")
    if scores.size >= 2:
        print(f"Top-2 {metric_key}: {top2:.6f}")
    print("Nearest neighbors:")
    for row in rows:
        score_val = row[metric_key]
        print(
            f"  #{row['rank']} id={row['image_id']} "
            f"{metric_key}={score_val:.6f} "
            f"lat={row['lat']:.5f} lon={row['lon']:.5f}"
        )

    if json_out is not None:
        write_json(json_out, output)
        print(f"Prediction JSON: {json_out}")
    return 0
