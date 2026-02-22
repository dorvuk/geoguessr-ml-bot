import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from geobot.constants import DEFAULT_DISTANCE_THRESHOLDS_KM
from geobot.geo import haversine_km
from geobot.retrieval_utils import (
    filter_embeddings_by_splits,
    load_embeddings_npz,
    normalize_rows,
    parse_splits,
    weighted_geo_mean,
)


def parse_int_list(raw: str) -> list[int]:
    out = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not out:
        raise ValueError("list is empty")
    return out


def parse_float_list(raw: str) -> list[float]:
    out = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not out:
        raise ValueError("list is empty")
    return out


def parse_str_list(raw: str) -> list[str]:
    out = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not out:
        raise ValueError("list is empty")
    return out


def load_index_meta(path: Path) -> dict[str, np.ndarray | str]:
    required = ["image_id", "lat", "lon", "metric"]
    with np.load(path, allow_pickle=False) as payload:
        missing = [k for k in required if k not in payload]
        if missing:
            raise RuntimeError(f"index meta npz missing keys: {missing}")
        out = {k: payload[k] for k in payload.files}
    out["image_id"] = np.asarray(out["image_id"]).astype(str)
    out["lat"] = np.asarray(out["lat"], dtype=np.float64)
    out["lon"] = np.asarray(out["lon"], dtype=np.float64)
    out["metric"] = str(np.asarray(out["metric"]).astype(str).reshape(-1)[0]).strip().lower()
    return out


def summarize_distance(dist_km: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(dist_km)
    if not np.any(valid):
        return {
            "count": 0.0,
            "median_km": float("nan"),
            "mean_km": float("nan"),
            "acc_25km": float("nan"),
            "acc_50km": float("nan"),
            "acc_200km": float("nan"),
            "acc_750km": float("nan"),
        }
    d = dist_km[valid]
    out = {
        "count": float(d.size),
        "median_km": float(np.median(d)),
        "mean_km": float(np.mean(d)),
    }
    for t in DEFAULT_DISTANCE_THRESHOLDS_KM:
        out[f"acc_{t}km"] = float(np.mean(d <= t))
    return out


def weights_from_scores(
    scores: np.ndarray,
    metric: str,
    mode: str,
    temperature: float,
) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        return s

    if mode == "top1":
        w = np.zeros_like(s)
        w[0] = 1.0
        return w
    if mode == "uniform":
        return np.ones_like(s)
    if mode == "softmax":
        logits = (s - float(np.max(s))) * float(temperature)
        logits = np.clip(logits, -60.0, 60.0)
        w = np.exp(logits)
        if float(np.sum(w)) <= 0.0:
            return np.ones_like(s)
        return w
    if mode == "inverse":
        if metric == "cosine":
            d = np.maximum(1.0 - s, 1e-6)
        else:
            d = np.maximum(s, 1e-6)
        return 1.0 / d
    raise ValueError(f"unknown weighting mode: {mode}")


def macro_region(lat: float, lon: float) -> str:
    if 110.0 <= lon <= 180.0 and -50.0 <= lat < 0.0:
        return "oceania"
    if -170.0 <= lon <= -30.0 and lat >= 12.0:
        return "north_america"
    if -90.0 <= lon <= -30.0 and lat < 12.0:
        return "south_america"
    if -25.0 <= lon <= 45.0 and 34.0 <= lat <= 72.0:
        return "europe"
    if -20.0 <= lon <= 55.0 and -35.0 <= lat <= 38.0:
        return "africa"
    if 45.0 <= lon <= 180.0 and -10.0 <= lat <= 82.0:
        return "asia"
    return "other"


def evaluate_setting(
    keep_scores: list[np.ndarray],
    keep_neighbors: list[np.ndarray],
    k: int,
    metric: str,
    weighting: str,
    temperature: float,
    margin_threshold: float,
    index_lat: np.ndarray,
    index_lon: np.ndarray,
    true_lat: np.ndarray,
    true_lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(keep_scores)
    pred_lat = np.full(n, np.nan, dtype=np.float64)
    pred_lon = np.full(n, np.nan, dtype=np.float64)
    top1_scores = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        scores = keep_scores[i]
        neigh = keep_neighbors[i]
        if scores.size == 0:
            continue

        use_n = min(k, scores.size)
        local_scores = scores[:use_n]
        local_neigh = neigh[:use_n]
        top1_scores[i] = float(local_scores[0])

        # Confidence gate: if top1-top2 margin is tiny, avoid averaging and keep top1.
        if margin_threshold > 0.0 and local_scores.size >= 2 and metric == "cosine":
            margin = float(local_scores[0] - local_scores[1])
            if margin < margin_threshold:
                local_scores = local_scores[:1]
                local_neigh = local_neigh[:1]

        lat = index_lat[local_neigh]
        lon = index_lon[local_neigh]
        w = weights_from_scores(local_scores, metric=metric, mode=weighting, temperature=temperature)
        out_lat, out_lon = weighted_geo_mean(lat, lon, w)
        pred_lat[i] = out_lat
        pred_lon[i] = out_lon

    dist = haversine_km(true_lat, true_lon, pred_lat, pred_lon)
    return pred_lat, pred_lon, dist


def write_confidence_bins(path: Path, top1_scores: np.ndarray, dist_km: np.ndarray, metric: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bins = [0.0, 0.45, 0.55, 0.65, 0.75, 1.01]
    labels = ["<0.45", "0.45-0.55", "0.55-0.65", "0.65-0.75", ">=0.75"]
    if metric != "cosine":
        # Convert l2/squared distance to a bounded strength for binning.
        top1_scores = 1.0 / (1.0 + np.maximum(top1_scores, 0.0))
    valid = np.isfinite(top1_scores) & np.isfinite(dist_km)
    s = top1_scores[valid]
    d = dist_km[valid]

    rows = []
    for lo, hi, label in zip(bins[:-1], bins[1:], labels):
        if label.startswith(">="):
            mask = s >= lo
        else:
            mask = (s >= lo) & (s < hi)
        if not np.any(mask):
            rows.append({"bin": label, "count": 0, "median_km": "", "acc_50km": "", "acc_200km": ""})
            continue
        sub = d[mask]
        rows.append(
            {
                "bin": label,
                "count": int(sub.size),
                "median_km": float(np.median(sub)),
                "acc_50km": float(np.mean(sub <= 50.0)),
                "acc_200km": float(np.mean(sub <= 200.0)),
            }
        )

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bin", "count", "median_km", "acc_50km", "acc_200km"])
        writer.writeheader()
        writer.writerows(rows)


def write_region_breakdown(path: Path, true_lat: np.ndarray, true_lon: np.ndarray, dist_km: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    regions = np.asarray([macro_region(float(la), float(lo)) for la, lo in zip(true_lat, true_lon)])
    valid = np.isfinite(dist_km)
    rows = []
    for region in sorted(set(regions.tolist())):
        mask = (regions == region) & valid
        if not np.any(mask):
            continue
        d = dist_km[mask]
        rows.append(
            {
                "region": region,
                "count": int(d.size),
                "median_km": float(np.median(d)),
                "acc_50km": float(np.mean(d <= 50.0)),
                "acc_200km": float(np.mean(d <= 200.0)),
            }
        )
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["region", "count", "median_km", "acc_50km", "acc_200km"])
        writer.writeheader()
        writer.writerows(rows)


def iter_settings(
    k_values: Iterable[int],
    weighting_modes: Iterable[str],
    temperatures: Iterable[float],
    margin_thresholds: Iterable[float],
) -> Iterable[tuple[int, str, float, float]]:
    for k in k_values:
        for weighting in weighting_modes:
            if weighting == "softmax":
                for t in temperatures:
                    for mt in margin_thresholds:
                        yield k, weighting, float(t), float(mt)
            else:
                for mt in margin_thresholds:
                    yield k, weighting, float("nan"), float(mt)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep retrieval settings and compare metrics.")
    ap.add_argument("--query-embeddings-npz", required=True)
    ap.add_argument("--index-path", required=True)
    ap.add_argument("--index-meta-npz", required=True)
    ap.add_argument("--query-splits", default="test")
    ap.add_argument("--k-values", default="1,3,5,10")
    ap.add_argument("--weighting-modes", default="top1,softmax,uniform,inverse")
    ap.add_argument("--temperatures", default="5,10,20,40")
    ap.add_argument("--margin-thresholds", default="0.0,0.01,0.02")
    ap.add_argument("--exclude-image-id", action="store_true", default=True)
    ap.add_argument("--include-image-id", dest="exclude_image_id", action="store_false")
    ap.add_argument("--out-csv", default="models/retrieval/sweep_results.csv")
    ap.add_argument("--best-json", default="models/retrieval/sweep_best.json")
    ap.add_argument("--best-confidence-csv", default="models/retrieval/sweep_best_confidence_bins.csv")
    ap.add_argument("--best-region-csv", default="models/retrieval/sweep_best_regions.csv")
    args = ap.parse_args()

    try:
        import faiss
    except ImportError as e:
        raise RuntimeError("Missing FAISS dependency. Run `uv sync`.") from e

    query_embeddings_npz = Path(args.query_embeddings_npz).expanduser()
    index_path = Path(args.index_path).expanduser()
    index_meta_npz = Path(args.index_meta_npz).expanduser()
    out_csv = Path(args.out_csv).expanduser()
    best_json = Path(args.best_json).expanduser()
    best_confidence_csv = Path(args.best_confidence_csv).expanduser()
    best_region_csv = Path(args.best_region_csv).expanduser()

    query_splits = parse_splits(args.query_splits)
    k_values = sorted(set(parse_int_list(args.k_values)))
    weighting_modes = parse_str_list(args.weighting_modes)
    temperatures = parse_float_list(args.temperatures)
    margin_thresholds = parse_float_list(args.margin_thresholds)

    query_payload = load_embeddings_npz(query_embeddings_npz)
    query_payload = filter_embeddings_by_splits(query_payload, query_splits)
    q_emb = np.asarray(query_payload["embeddings"], dtype=np.float32)
    true_lat = np.asarray(query_payload["lat"], dtype=np.float64)
    true_lon = np.asarray(query_payload["lon"], dtype=np.float64)
    query_image_ids = np.asarray(query_payload["image_id"]).astype(str)

    index_meta = load_index_meta(index_meta_npz)
    index = faiss.read_index(str(index_path))
    metric = str(index_meta["metric"])
    if metric not in {"cosine", "l2"}:
        raise RuntimeError(f"unsupported index metric: {metric}")
    if metric == "cosine":
        q_emb = normalize_rows(q_emb).astype(np.float32)

    max_k = max(k_values)
    search_k = max_k + (1 if args.exclude_image_id else 0)
    search_k = min(search_k, int(index.ntotal))
    if search_k <= 0:
        raise RuntimeError("index has no vectors")

    scores_all, neigh_all = index.search(q_emb, search_k)
    index_ids = np.asarray(index_meta["image_id"]).astype(str)
    index_lat = np.asarray(index_meta["lat"], dtype=np.float64)
    index_lon = np.asarray(index_meta["lon"], dtype=np.float64)

    keep_scores: list[np.ndarray] = []
    keep_neighbors: list[np.ndarray] = []
    for i in range(q_emb.shape[0]):
        scores = np.asarray(scores_all[i], dtype=np.float64)
        neigh = np.asarray(neigh_all[i], dtype=np.int64)
        valid = neigh >= 0
        if args.exclude_image_id:
            valid &= index_ids[neigh] != query_image_ids[i]
        keep_scores.append(scores[valid])
        keep_neighbors.append(neigh[valid])

    rows = []
    best_row = None
    best_pred_dist = None
    best_top1_scores = None
    for k, weighting, temperature, margin_thr in iter_settings(
        k_values, weighting_modes, temperatures, margin_thresholds
    ):
        pred_lat, pred_lon, dist = evaluate_setting(
            keep_scores=keep_scores,
            keep_neighbors=keep_neighbors,
            k=k,
            metric=metric,
            weighting=weighting,
            temperature=temperature if np.isfinite(temperature) else 0.0,
            margin_threshold=margin_thr,
            index_lat=index_lat,
            index_lon=index_lon,
            true_lat=true_lat,
            true_lon=true_lon,
        )
        _ = pred_lat, pred_lon
        metrics = summarize_distance(dist)
        row = {
            "k": int(k),
            "weighting": weighting,
            "temperature": "" if not np.isfinite(temperature) else float(temperature),
            "margin_threshold": float(margin_thr),
            "metric": metric,
            "median_km": float(metrics["median_km"]),
            "mean_km": float(metrics["mean_km"]),
            "acc_25km": float(metrics["acc_25km"]),
            "acc_50km": float(metrics["acc_50km"]),
            "acc_200km": float(metrics["acc_200km"]),
            "acc_750km": float(metrics["acc_750km"]),
            "count": int(metrics["count"]),
        }
        rows.append(row)

        if best_row is None:
            best_row = row
            best_pred_dist = dist
            best_top1_scores = np.asarray([s[0] if s.size > 0 else np.nan for s in keep_scores], dtype=np.float64)
        else:
            current_key = (row["median_km"], -row["acc_50km"], -row["acc_200km"])
            best_key = (best_row["median_km"], -best_row["acc_50km"], -best_row["acc_200km"])
            if current_key < best_key:
                best_row = row
                best_pred_dist = dist
                best_top1_scores = np.asarray([s[0] if s.size > 0 else np.nan for s in keep_scores], dtype=np.float64)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "k",
                "weighting",
                "temperature",
                "margin_threshold",
                "metric",
                "median_km",
                "mean_km",
                "acc_25km",
                "acc_50km",
                "acc_200km",
                "acc_750km",
                "count",
            ],
        )
        writer.writeheader()
        writer.writerows(
            sorted(
                rows,
                key=lambda r: (r["median_km"], -r["acc_50km"], -r["acc_200km"]),
            )
        )

    best_payload = {
        "query_embeddings_npz": str(query_embeddings_npz),
        "index_path": str(index_path),
        "index_meta_npz": str(index_meta_npz),
        "query_splits": query_splits,
        "best_setting": best_row,
        "num_settings": len(rows),
        "results_csv": str(out_csv),
        "confidence_bins_csv": str(best_confidence_csv),
        "region_breakdown_csv": str(best_region_csv),
    }
    best_json.parent.mkdir(parents=True, exist_ok=True)
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2)

    if best_pred_dist is not None and best_top1_scores is not None:
        write_confidence_bins(best_confidence_csv, best_top1_scores, best_pred_dist, metric=metric)
        write_region_breakdown(best_region_csv, true_lat=true_lat, true_lon=true_lon, dist_km=best_pred_dist)

    print(f"Sweep results CSV: {out_csv}")
    print(f"Best setting JSON: {best_json}")
    if best_row is not None:
        print(
            "Best setting: "
            f"k={best_row['k']} weighting={best_row['weighting']} "
            f"temp={best_row['temperature']} margin_thr={best_row['margin_threshold']} "
            f"median_km={best_row['median_km']:.2f} acc50={best_row['acc_50km']:.4f}"
        )


if __name__ == "__main__":
    main()
