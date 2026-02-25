import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from geobot.constants import DEFAULT_DISTANCE_THRESHOLDS_KM
from geobot.data_utils import class_centroids
from geobot.geo import haversine_km
from geobot.io_utils import resolve_path
from geobot.retrieval_utils import build_feature_model, choose_device


class ImageDataset:
    def __init__(self, frame: pd.DataFrame, repo_root: Path, transform) -> None:
        self.frame = frame.reset_index(drop=True)
        self.repo_root = repo_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        path = Path(str(row["path"])).expanduser()
        if not path.is_absolute():
            path = self.repo_root / path
        with Image.open(path) as img:
            image = img.convert("RGB")
        x = self.transform(image)
        return x, str(row["image_id"])


def parse_float_list(raw: str) -> list[float]:
    out = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not out:
        raise ValueError("list is empty")
    return out


def parse_dynamic_thresholds(raw: str) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    if not raw.strip():
        return out
    for token in [t.strip() for t in raw.split(",") if t.strip()]:
        if ":" not in token:
            raise ValueError(f"invalid threshold pair: {token}")
        lo, hi = token.split(":", 1)
        lo_f = float(lo)
        hi_f = float(hi)
        if hi_f <= lo_f:
            raise ValueError(f"dynamic threshold pair requires hi>lo: {token}")
        out.append((lo_f, hi_f))
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


def blend_two_points(
    cls_lat: np.ndarray,
    cls_lon: np.ndarray,
    ret_lat: np.ndarray,
    ret_lon: np.ndarray,
    alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # spherical blend via xyz averaging avoids dateline artifacts
    cls_lat_r = np.radians(cls_lat)
    cls_lon_r = np.radians(cls_lon)
    ret_lat_r = np.radians(ret_lat)
    ret_lon_r = np.radians(ret_lon)

    cx = np.cos(cls_lat_r) * np.cos(cls_lon_r)
    cy = np.cos(cls_lat_r) * np.sin(cls_lon_r)
    cz = np.sin(cls_lat_r)

    rx = np.cos(ret_lat_r) * np.cos(ret_lon_r)
    ry = np.cos(ret_lat_r) * np.sin(ret_lon_r)
    rz = np.sin(ret_lat_r)

    a = np.clip(alpha, 0.0, 1.0)
    wc = 1.0 - a
    wr = a

    x = wc * cx + wr * rx
    y = wc * cy + wr * ry
    z = wc * cz + wr * rz
    norm = np.sqrt(x * x + y * y + z * z)
    norm = np.maximum(norm, 1e-12)
    x /= norm
    y /= norm
    z /= norm

    lat = np.degrees(np.arctan2(z, np.sqrt(x * x + y * y)))
    lon = np.degrees(np.arctan2(y, x))
    return lat.astype(np.float64), lon.astype(np.float64)


def run_classifier_predictions(
    dataset_csv: Path,
    checkpoint: Path,
    repo_root: Path,
    model_name_override: str | None,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device_name: str,
    amp: bool,
) -> pd.DataFrame:
    try:
        import timm
        import torch
        from torch.utils.data import DataLoader
        from torchvision import transforms
    except ImportError as e:
        raise RuntimeError("Missing ML dependency. Run `uv sync`.") from e

    df = pd.read_csv(dataset_csv, dtype=str, keep_default_na=False)
    required = {"path", "lat", "lon", "label_id", "split", "image_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"dataset missing required columns: {missing}")

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["label_id"] = pd.to_numeric(df["label_id"], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "label_id"]).copy()
    df["label_id"] = df["label_id"].astype(int)
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
    if train_df.empty or test_df.empty:
        raise RuntimeError("dataset must contain non-empty train and test splits")

    centers = class_centroids(train_df, label_col="label_id")
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    device = choose_device(device_name, torch_mod=torch)
    use_amp = bool(amp and device.type == "cuda")
    _, model, _ = build_feature_model(
        timm_mod=timm,
        torch_mod=torch,
        checkpoint_path=checkpoint,
        model_name_override=model_name_override,
        device=device,
    )
    model.eval()

    ds = ImageDataset(test_df, repo_root=repo_root, transform=transform)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    pred_image_id: list[str] = []
    pred_lat: list[float] = []
    pred_lon: list[float] = []
    with torch.no_grad():
        for images, image_ids in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
            labels = logits.argmax(dim=1).detach().cpu().numpy().astype(int)
            for img_id, label in zip(image_ids, labels):
                center = centers.get(int(label))
                if center is None:
                    continue
                pred_image_id.append(str(img_id))
                pred_lat.append(float(center[0]))
                pred_lon.append(float(center[1]))

    out = pd.DataFrame(
        {
            "image_id": pred_image_id,
            "cls_pred_lat": pred_lat,
            "cls_pred_lon": pred_lon,
        }
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate classifier+retrieval hybrid blending.")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--dataset-csv", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--retrieval-csv", required=True)
    ap.add_argument("--model-name", default=None, help="Optional timm model override")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--alpha-values", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument(
        "--dynamic-thresholds",
        default="",
        help="Optional retrieval score ranges for dynamic alpha. Example: '0.50:0.65,0.55:0.70'",
    )
    ap.add_argument("--out-csv", default="models/retrieval/hybrid_sweep.csv")
    ap.add_argument("--best-json", default="models/retrieval/hybrid_best.json")
    ap.add_argument("--best-predictions-csv", default="models/retrieval/hybrid_best_predictions.csv")
    args = ap.parse_args()

    repo_root = resolve_path(args.repo_root, Path.cwd())
    dataset_csv = resolve_path(args.dataset_csv, repo_root)
    checkpoint = resolve_path(args.checkpoint, repo_root)
    retrieval_csv = resolve_path(args.retrieval_csv, repo_root)
    out_csv = resolve_path(args.out_csv, repo_root)
    best_json = resolve_path(args.best_json, repo_root)
    best_predictions_csv = resolve_path(args.best_predictions_csv, repo_root)

    alpha_values = parse_float_list(args.alpha_values)
    dynamic_thresholds = parse_dynamic_thresholds(args.dynamic_thresholds)

    cls_df = run_classifier_predictions(
        dataset_csv=dataset_csv,
        checkpoint=checkpoint,
        repo_root=repo_root,
        model_name_override=args.model_name,
        image_size=int(args.image_size),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device_name=args.device,
        amp=bool(args.amp),
    )
    if cls_df.empty:
        raise RuntimeError("classifier predictions are empty")

    ret_df = pd.read_csv(retrieval_csv, dtype=str, keep_default_na=False)
    required_ret = {"image_id", "true_lat", "true_lon", "pred_lat", "pred_lon", "nn1_score"}
    missing_ret = sorted(required_ret - set(ret_df.columns))
    if missing_ret:
        raise RuntimeError(f"retrieval csv missing required columns: {missing_ret}")

    for col in ["true_lat", "true_lon", "pred_lat", "pred_lon", "nn1_score"]:
        ret_df[col] = pd.to_numeric(ret_df[col], errors="coerce")
    ret_df = ret_df.dropna(subset=["true_lat", "true_lon", "pred_lat", "pred_lon", "nn1_score"]).copy()
    if ret_df.empty:
        raise RuntimeError("retrieval CSV has no usable rows")

    merged = ret_df.merge(cls_df, on="image_id", how="inner")
    merged = merged.dropna(subset=["cls_pred_lat", "cls_pred_lon"]).copy()
    if merged.empty:
        raise RuntimeError("no overlap between retrieval rows and classifier predictions")

    true_lat = merged["true_lat"].to_numpy(dtype=np.float64)
    true_lon = merged["true_lon"].to_numpy(dtype=np.float64)
    ret_lat = merged["pred_lat"].to_numpy(dtype=np.float64)
    ret_lon = merged["pred_lon"].to_numpy(dtype=np.float64)
    cls_lat = merged["cls_pred_lat"].to_numpy(dtype=np.float64)
    cls_lon = merged["cls_pred_lon"].to_numpy(dtype=np.float64)
    nn1_score = merged["nn1_score"].to_numpy(dtype=np.float64)

    rows = []
    best_row = None
    best_pred = None

    # static alpha sweeps
    for alpha in alpha_values:
        a = np.full_like(true_lat, fill_value=float(alpha), dtype=np.float64)
        pred_lat, pred_lon = blend_two_points(cls_lat, cls_lon, ret_lat, ret_lon, a)
        dist = haversine_km(true_lat, true_lon, pred_lat, pred_lon)
        m = summarize_distance(dist)
        row = {
            "mode": "static",
            "alpha": float(alpha),
            "low": "",
            "high": "",
            "median_km": float(m["median_km"]),
            "mean_km": float(m["mean_km"]),
            "acc_25km": float(m["acc_25km"]),
            "acc_50km": float(m["acc_50km"]),
            "acc_200km": float(m["acc_200km"]),
            "acc_750km": float(m["acc_750km"]),
            "count": int(m["count"]),
        }
        rows.append(row)
        key = (row["median_km"], -row["acc_50km"], -row["acc_200km"])
        if best_row is None or key < (best_row["median_km"], -best_row["acc_50km"], -best_row["acc_200km"]):
            best_row = row
            best_pred = (pred_lat, pred_lon, dist)

    # dynamic alpha from retrieval confidence score
    for low, high in dynamic_thresholds:
        a = (nn1_score - low) / (high - low)
        a = np.clip(a, 0.0, 1.0)
        pred_lat, pred_lon = blend_two_points(cls_lat, cls_lon, ret_lat, ret_lon, a)
        dist = haversine_km(true_lat, true_lon, pred_lat, pred_lon)
        m = summarize_distance(dist)
        row = {
            "mode": "dynamic_score",
            "alpha": "",
            "low": float(low),
            "high": float(high),
            "median_km": float(m["median_km"]),
            "mean_km": float(m["mean_km"]),
            "acc_25km": float(m["acc_25km"]),
            "acc_50km": float(m["acc_50km"]),
            "acc_200km": float(m["acc_200km"]),
            "acc_750km": float(m["acc_750km"]),
            "count": int(m["count"]),
        }
        rows.append(row)
        key = (row["median_km"], -row["acc_50km"], -row["acc_200km"])
        if best_row is None or key < (best_row["median_km"], -best_row["acc_50km"], -best_row["acc_200km"]):
            best_row = row
            best_pred = (pred_lat, pred_lon, dist)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "alpha",
                "low",
                "high",
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
        "dataset_csv": str(dataset_csv),
        "checkpoint": str(checkpoint),
        "retrieval_csv": str(retrieval_csv),
        "best_row": best_row,
        "num_rows_merged": int(len(merged)),
        "out_csv": str(out_csv),
        "best_predictions_csv": str(best_predictions_csv),
    }
    best_json.parent.mkdir(parents=True, exist_ok=True)
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2)

    if best_pred is not None:
        pred_lat, pred_lon, dist = best_pred
        best_df = merged.copy()
        best_df["hybrid_pred_lat"] = pred_lat
        best_df["hybrid_pred_lon"] = pred_lon
        best_df["hybrid_error_km"] = dist
        best_predictions_csv.parent.mkdir(parents=True, exist_ok=True)
        best_df.to_csv(best_predictions_csv, index=False)

    print(f"Hybrid sweep CSV: {out_csv}")
    print(f"Best hybrid JSON: {best_json}")
    if best_row is not None:
        print(
            f"Best hybrid: mode={best_row['mode']} alpha={best_row['alpha']} "
            f"low={best_row['low']} high={best_row['high']} "
            f"median_km={best_row['median_km']:.2f} acc50={best_row['acc_50km']:.4f}"
        )


if __name__ == "__main__":
    main()
