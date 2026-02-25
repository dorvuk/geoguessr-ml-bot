import argparse
import csv
from pathlib import Path

import numpy as np

from geobot.constants import DEFAULT_DISTANCE_THRESHOLDS_KM
from geobot.geo import haversine_km
from geobot.io_utils import resolve_path, write_json
from geobot.retrieval_utils import (
    filter_embeddings_by_splits,
    load_embeddings_npz,
    normalize_rows,
    parse_splits,
    weighted_geo_mean,
)


def register_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("eval-retrieval", help="Evaluate retrieval geolocation from FAISS index")
    parser.add_argument("--repo-root", default=".", help="Repository root directory")
    parser.add_argument("--query-embeddings-npz", default="models/retrieval/embeddings_all.npz")
    parser.add_argument("--index-path", default="models/retrieval/index.faiss")
    parser.add_argument("--index-meta-npz", default="models/retrieval/index_meta.npz")
    parser.add_argument("--query-splits", default="test", help="Comma-separated query splits")
    parser.add_argument("--k", type=int, default=1, help="Number of neighbors to aggregate")
    parser.add_argument("--metric", choices=["auto", "cosine", "l2"], default="auto")
    parser.add_argument(
        "--cosine-temperature",
        type=float,
        default=20.0,
        help="Softmax temperature for cosine score weighting",
    )
    parser.add_argument(
        "--exclude-image-id",
        dest="exclude_image_id",
        action="store_true",
        default=True,
        help="Ignore neighbors with the same image_id as the query",
    )
    parser.add_argument("--include-image-id", dest="exclude_image_id", action="store_false")
    parser.add_argument("--out-json", default="models/retrieval/retrieval_metrics.json")
    parser.add_argument(
        "--per-query-csv",
        default="",
        help="Optional CSV path for per-query predictions/errors",
    )
    parser.set_defaults(func=run)


def load_index_meta(path: Path) -> dict[str, np.ndarray | str]:
    required = ["image_id", "path", "lat", "lon", "split", "label_id", "metric"]
    with np.load(path, allow_pickle=False) as payload:
        missing = [k for k in required if k not in payload]
        if missing:
            raise RuntimeError(f"index meta npz missing keys: {missing}")
        out = {k: payload[k] for k in payload.files}

    out["image_id"] = np.asarray(out["image_id"]).astype(str)
    out["path"] = np.asarray(out["path"]).astype(str)
    out["lat"] = np.asarray(out["lat"], dtype=np.float64)
    out["lon"] = np.asarray(out["lon"], dtype=np.float64)
    out["split"] = np.asarray(out["split"]).astype(str)
    out["label_id"] = np.asarray(out["label_id"], dtype=np.int64)
    metric = str(np.asarray(out["metric"]).astype(str).reshape(-1)[0]).strip().lower()
    out["metric"] = metric
    return out


def summarize_distance_metrics(distances: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(distances)
    if not np.any(valid):
        return {"count": 0}
    d = distances[valid]
    out: dict[str, float | int] = {
        "count": int(d.size),
        "median_km": float(np.median(d)),
        "mean_km": float(np.mean(d)),
    }
    for t in DEFAULT_DISTANCE_THRESHOLDS_KM:
        out[f"acc_{t}km"] = float(np.mean(d <= t))
    return out


def weights_from_scores(scores: np.ndarray, metric: str, cosine_temperature: float) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    if metric == "cosine":
        logits = (s - float(np.max(s))) * cosine_temperature
        logits = np.clip(logits, -60.0, 60.0)
        weights = np.exp(logits)
    else:
        d = np.maximum(s, 0.0)
        weights = 1.0 / (d + 1e-6)
    if not np.any(np.isfinite(weights)) or float(np.sum(weights)) <= 0.0:
        return np.ones_like(s, dtype=np.float64)
    return weights


def run(args: argparse.Namespace) -> int:
    try:
        import faiss
    except ImportError as e:
        raise RuntimeError("Missing FAISS dependency. Run `uv sync`.") from e

    if args.k <= 0:
        raise ValueError("--k must be >= 1")

    repo_root = resolve_path(args.repo_root, Path.cwd())
    query_embeddings_npz = resolve_path(args.query_embeddings_npz, repo_root)
    index_path = resolve_path(args.index_path, repo_root)
    index_meta_npz = resolve_path(args.index_meta_npz, repo_root)
    out_json = resolve_path(args.out_json, repo_root)
    per_query_csv = resolve_path(args.per_query_csv, repo_root) if args.per_query_csv else None
    query_splits = parse_splits(args.query_splits)

    if not query_embeddings_npz.exists():
        raise FileNotFoundError(f"query embeddings not found: {query_embeddings_npz}")
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not index_meta_npz.exists():
        raise FileNotFoundError(f"index metadata not found: {index_meta_npz}")

    query_payload = load_embeddings_npz(query_embeddings_npz)
    query_payload = filter_embeddings_by_splits(query_payload, query_splits)
    q_emb = np.asarray(query_payload["embeddings"], dtype=np.float32)
    if q_emb.ndim != 2 or q_emb.shape[0] == 0:
        raise RuntimeError("query embeddings are empty after split filtering")

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
        q_emb = normalize_rows(q_emb).astype(np.float32)

    internal_k = args.k + (1 if args.exclude_image_id else 0)
    internal_k = min(int(index.ntotal), internal_k)
    if internal_k <= 0:
        raise RuntimeError("index has no vectors")

    scores, neighbors = index.search(q_emb, internal_k)

    n = q_emb.shape[0]
    true_lat = np.asarray(query_payload["lat"], dtype=np.float64)
    true_lon = np.asarray(query_payload["lon"], dtype=np.float64)
    pred_lat = np.full(n, np.nan, dtype=np.float64)
    pred_lon = np.full(n, np.nan, dtype=np.float64)
    nn1_lat = np.full(n, np.nan, dtype=np.float64)
    nn1_lon = np.full(n, np.nan, dtype=np.float64)
    nn1_score = np.full(n, np.nan, dtype=np.float64)
    top1_neighbor_id: list[str] = [""] * n
    neighbor_count = np.zeros(n, dtype=np.int32)

    index_image_id = np.asarray(index_meta["image_id"]).astype(str)
    index_lat = np.asarray(index_meta["lat"], dtype=np.float64)
    index_lon = np.asarray(index_meta["lon"], dtype=np.float64)
    query_image_id = np.asarray(query_payload["image_id"]).astype(str)

    for i in range(n):
        keep_idx: list[int] = []
        keep_scores: list[float] = []
        qid = query_image_id[i]
        for pos in range(internal_k):
            idx = int(neighbors[i, pos])
            if idx < 0:
                continue
            if args.exclude_image_id and index_image_id[idx] == qid:
                continue
            keep_idx.append(idx)
            keep_scores.append(float(scores[i, pos]))
            if len(keep_idx) >= args.k:
                break

        if not keep_idx:
            continue

        idx_arr = np.asarray(keep_idx, dtype=np.int64)
        score_arr = np.asarray(keep_scores, dtype=np.float64)
        neighbor_count[i] = int(idx_arr.size)

        local_lat = index_lat[idx_arr]
        local_lon = index_lon[idx_arr]
        w = weights_from_scores(score_arr, metric=metric, cosine_temperature=args.cosine_temperature)
        out_lat, out_lon = weighted_geo_mean(local_lat, local_lon, w)
        pred_lat[i] = out_lat
        pred_lon[i] = out_lon

        nn1_lat[i] = float(local_lat[0])
        nn1_lon[i] = float(local_lon[0])
        nn1_score[i] = float(score_arr[0])
        top1_neighbor_id[i] = index_image_id[idx_arr[0]]

    dist_weighted = haversine_km(true_lat, true_lon, pred_lat, pred_lon)
    dist_nn1 = haversine_km(true_lat, true_lon, nn1_lat, nn1_lon)

    weighted_metrics = summarize_distance_metrics(dist_weighted)
    nn1_metrics = summarize_distance_metrics(dist_nn1)

    out_payload: dict[str, object] = {
        "query_embeddings_npz": str(query_embeddings_npz),
        "index_path": str(index_path),
        "index_meta_npz": str(index_meta_npz),
        "query_splits": query_splits,
        "metric": metric,
        "k": int(args.k),
        "exclude_image_id": bool(args.exclude_image_id),
        "num_queries": int(n),
        "num_predictions": int(np.isfinite(dist_weighted).sum()),
        "weighted_metrics": weighted_metrics,
        "nearest_neighbor_metrics": nn1_metrics,
    }
    if metric == "cosine":
        out_payload["cosine_temperature"] = float(args.cosine_temperature)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_json, out_payload)

    if per_query_csv is not None:
        per_query_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(per_query_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image_id",
                    "split",
                    "path",
                    "true_lat",
                    "true_lon",
                    "pred_lat",
                    "pred_lon",
                    "error_km",
                    "nn1_lat",
                    "nn1_lon",
                    "nn1_error_km",
                    "nn1_score",
                    "top1_neighbor_id",
                    "neighbor_count",
                ],
            )
            writer.writeheader()
            for i in range(n):
                writer.writerow(
                    {
                        "image_id": query_image_id[i],
                        "split": str(query_payload["split"][i]),
                        "path": str(query_payload["path"][i]),
                        "true_lat": float(true_lat[i]),
                        "true_lon": float(true_lon[i]),
                        "pred_lat": "" if not np.isfinite(pred_lat[i]) else float(pred_lat[i]),
                        "pred_lon": "" if not np.isfinite(pred_lon[i]) else float(pred_lon[i]),
                        "error_km": "" if not np.isfinite(dist_weighted[i]) else float(dist_weighted[i]),
                        "nn1_lat": "" if not np.isfinite(nn1_lat[i]) else float(nn1_lat[i]),
                        "nn1_lon": "" if not np.isfinite(nn1_lon[i]) else float(nn1_lon[i]),
                        "nn1_error_km": "" if not np.isfinite(dist_nn1[i]) else float(dist_nn1[i]),
                        "nn1_score": "" if not np.isfinite(nn1_score[i]) else float(nn1_score[i]),
                        "top1_neighbor_id": top1_neighbor_id[i],
                        "neighbor_count": int(neighbor_count[i]),
                    }
                )
        print(f"Per-query CSV written: {per_query_csv}")

    weighted_median = weighted_metrics.get("median_km")
    nn1_median = nn1_metrics.get("median_km")
    print(f"Retrieval metrics written: {out_json}")
    if isinstance(weighted_median, float):
        print(f"Weighted median error: {weighted_median:.2f} km")
    if isinstance(nn1_median, float):
        print(f"Top-1 NN median error: {nn1_median:.2f} km")
    return 0
