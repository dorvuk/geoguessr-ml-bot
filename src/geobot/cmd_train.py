import argparse
import csv
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from PIL import Image

from geobot.constants import DEFAULT_DISTANCE_THRESHOLDS_KM
from geobot.data_utils import class_centroids
from geobot.geo import haversine_km
from geobot.io_utils import resolve_path, write_json


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
        label = int(row["label_id"])
        lat = float(row["lat"])
        lon = float(row["lon"])
        return x, label, lat, lon


def register_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("train", help="Train baseline geolocation classifier")
    parser.add_argument("--repo-root", default=".", help="Repository root directory")
    parser.add_argument("--dataset-csv", default="data/processed/mapillary/dataset.csv")
    parser.add_argument("--out-dir", default="models/classifier_baseline")
    parser.add_argument("--model-name", default="resnet18", help="timm model name")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument("--amp", action="store_true", help="Enable AMP on CUDA")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(func=run)


def set_seed(seed: int, torch_mod) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_mod.manual_seed(seed)
    if torch_mod.cuda.is_available():
        torch_mod.cuda.manual_seed_all(seed)


def choose_device(requested: str, torch_mod):
    if requested == "auto":
        return torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
    return torch_mod.device(requested)


def evaluate(
    model,
    loader,
    device,
    criterion,
    torch_mod,
    centers: Dict[int, tuple[float, float]],
    thresholds_km: list[int],
) -> Dict[str, float]:
    if loader is None:
        return {}

    model.eval()
    total = 0
    total_loss = 0.0
    total_correct = 0
    pred_labels: list[int] = []
    true_lats: list[float] = []
    true_lons: list[float] = []

    with torch_mod.no_grad():
        for images, labels, lats, lons in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            batch_size = int(labels.size(0))
            total += batch_size
            total_loss += float(loss.item()) * batch_size
            total_correct += int((preds == labels).sum().item())

            pred_labels.extend(preds.detach().cpu().numpy().astype(int).tolist())
            true_lats.extend(np.asarray(lats, dtype=float).tolist())
            true_lons.extend(np.asarray(lons, dtype=float).tolist())

    if total == 0:
        return {}

    metrics: Dict[str, float] = {
        "loss": total_loss / total,
        "top1": total_correct / total,
    }

    pred_lat = []
    pred_lon = []
    for label in pred_labels:
        center = centers.get(int(label))
        if center is None:
            pred_lat.append(float("nan"))
            pred_lon.append(float("nan"))
        else:
            pred_lat.append(center[0])
            pred_lon.append(center[1])

    dist = haversine_km(true_lats, true_lons, pred_lat, pred_lon)
    valid = ~np.isnan(dist)
    if np.any(valid):
        dist_valid = dist[valid]
        metrics["median_km"] = float(np.median(dist_valid))
        for t in thresholds_km:
            metrics[f"acc_{t}km"] = float(np.mean(dist_valid <= t))

    return metrics


def run(args: argparse.Namespace) -> int:
    try:
        import timm
        import torch
        from torch.utils.data import DataLoader
        from torchvision import transforms
    except ImportError as e:
        raise RuntimeError(
            "Missing training dependency. Install with `uv sync` before running `geobot train`."
        ) from e

    repo_root = resolve_path(args.repo_root, Path.cwd())
    dataset_csv = resolve_path(args.dataset_csv, repo_root)
    out_dir = resolve_path(args.out_dir, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed, torch)
    device = choose_device(args.device, torch)
    use_amp = bool(args.amp and device.type == "cuda")

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
    df = df[df["split"].isin(["train", "val", "test"])].copy()
    if df.empty:
        raise RuntimeError(f"dataset has no usable rows: {dataset_csv}")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    if train_df.empty:
        raise RuntimeError("dataset has no train split rows")

    num_classes = int(train_df["label_id"].nunique())
    if num_classes < 2:
        raise RuntimeError(f"need at least 2 classes for training, found {num_classes}")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_ds = ImageDataset(train_df, repo_root=repo_root, transform=train_transform)
    val_ds = ImageDataset(val_df, repo_root=repo_root, transform=eval_transform) if not val_df.empty else None
    test_ds = ImageDataset(test_df, repo_root=repo_root, transform=eval_transform) if not test_df.empty else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        if val_ds is not None
        else None
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        if test_ds is not None
        else None
    )

    model = timm.create_model(args.model_name, pretrained=args.pretrained, num_classes=num_classes)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    centers = class_centroids(train_df, label_col="label_id")
    thresholds = DEFAULT_DISTANCE_THRESHOLDS_KM

    history_path = out_dir / "history.csv"
    best_path = out_dir / "best.pt"
    history_rows: list[dict[str, float | int]] = []

    best_metric = -1.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        total = 0
        total_loss = 0.0
        total_correct = 0

        for images, labels, _, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = logits.argmax(dim=1)
            bs = int(labels.size(0))
            total += bs
            total_loss += float(loss.item()) * bs
            total_correct += int((preds == labels).sum().item())

        scheduler.step()

        train_metrics = {
            "loss": total_loss / max(total, 1),
            "top1": total_correct / max(total, 1),
        }
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            criterion,
            torch_mod=torch,
            centers=centers,
            thresholds_km=thresholds,
        )
        selection_metric = float(val_metrics.get("top1", train_metrics["top1"]))
        if selection_metric > best_metric:
            best_metric = selection_metric
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": args.model_name,
                    "num_classes": num_classes,
                    "epoch": epoch,
                },
                best_path,
            )

        row: dict[str, float | int] = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_metrics["loss"]),
            "train_top1": float(train_metrics["top1"]),
            "seconds": float(time.time() - epoch_start),
        }
        for key, value in val_metrics.items():
            row[f"val_{key}"] = float(value)
        history_rows.append(row)

        print(
            f"epoch={epoch}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_top1={train_metrics['top1']:.4f} "
            f"val_top1={val_metrics.get('top1', float('nan')):.4f}"
        )

    with open(history_path, "w", encoding="utf-8", newline="") as f:
        if history_rows:
            fieldnames = sorted({k for row in history_rows for k in row.keys()})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in history_rows:
                writer.writerow(row)

    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        criterion,
        torch_mod=torch,
        centers=centers,
        thresholds_km=thresholds,
    )
    val_metrics = evaluate(
        model,
        val_loader,
        device,
        criterion,
        torch_mod=torch,
        centers=centers,
        thresholds_km=thresholds,
    )

    metrics = {
        "dataset_csv": str(dataset_csv),
        "out_dir": str(out_dir),
        "model_name": args.model_name,
        "num_classes": num_classes,
        "seed": int(args.seed),
        "device": str(device),
        "epochs": int(args.epochs),
        "best_epoch": int(best_epoch),
        "best_selection_metric": float(best_metric),
        "split_sizes": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "val_metrics": {k: float(v) for k, v in val_metrics.items()},
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "history_csv": str(history_path),
        "best_checkpoint": str(best_path),
    }
    write_json(out_dir / "metrics.json", metrics)

    print(f"Training complete. Best epoch: {best_epoch}")
    print(f"Metrics JSON: {out_dir / 'metrics.json'}")
    return 0
