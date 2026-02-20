import argparse
import time
from pathlib import Path

import numpy as np

from geobot.io_utils import resolve_path
from geobot.retrieval_utils import (
    EmbeddingDataset,
    build_feature_model,
    choose_device,
    extract_embeddings,
    load_dataset_frame,
    normalize_rows,
    parse_splits,
)


def register_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("embed", help="Extract image embeddings from a trained classifier checkpoint")
    parser.add_argument("--repo-root", default=".", help="Repository root directory")
    parser.add_argument("--dataset-csv", default="data/processed/mapillary/dataset.csv")
    parser.add_argument("--checkpoint", default="models/classifier_baseline/best.pt")
    parser.add_argument("--out-npz", default="models/retrieval/embeddings_all.npz")
    parser.add_argument("--splits", default="train,val,test", help="Comma-separated splits to embed")
    parser.add_argument("--model-name", default=None, help="Optional timm model name override")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument("--amp", action="store_true", help="Enable AMP on CUDA")
    parser.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        default=True,
        help="L2-normalize embeddings before saving",
    )
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    try:
        import timm
        import torch
        from torch.utils.data import DataLoader
        from torchvision import transforms
    except ImportError as e:
        raise RuntimeError("Missing dependency for embedding extraction. Run `uv sync`.") from e

    repo_root = resolve_path(args.repo_root, Path.cwd())
    dataset_csv = resolve_path(args.dataset_csv, repo_root)
    checkpoint = resolve_path(args.checkpoint, repo_root)
    out_npz = resolve_path(args.out_npz, repo_root)
    splits = parse_splits(args.splits)

    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    device = choose_device(args.device, torch)
    use_amp = bool(args.amp and device.type == "cuda")
    frame = load_dataset_frame(dataset_csv, repo_root=repo_root, splits=splits)

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ds = EmbeddingDataset(frame, transform=transform)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model_name, model, num_classes = build_feature_model(
        timm_mod=timm,
        torch_mod=torch,
        checkpoint_path=checkpoint,
        model_name_override=args.model_name,
        device=device,
    )

    chunks: list[np.ndarray] = []
    image_ids: list[str] = []
    paths: list[str] = []
    lats: list[float] = []
    lons: list[float] = []
    split_names: list[str] = []
    label_ids: list[int] = []

    start = time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader, start=1):
            images, batch_ids, batch_paths, batch_lats, batch_lons, batch_splits, batch_labels = batch
            images = images.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                emb = extract_embeddings(model, images, torch)

            emb_np = emb.detach().cpu().numpy().astype(np.float32)
            chunks.append(emb_np)

            image_ids.extend([str(x) for x in batch_ids])
            paths.extend([str(x) for x in batch_paths])
            lats.extend(np.asarray(batch_lats, dtype=np.float64).tolist())
            lons.extend(np.asarray(batch_lons, dtype=np.float64).tolist())
            split_names.extend([str(x) for x in batch_splits])
            label_ids.extend(np.asarray(batch_labels, dtype=np.int64).tolist())

            if i % 20 == 0:
                done = min(i * args.batch_size, len(ds))
                print(f"Embedded {done}/{len(ds)} images")

    if not chunks:
        raise RuntimeError("no embeddings were generated")

    embeddings = np.concatenate(chunks, axis=0).astype(np.float32)
    if args.normalize:
        embeddings = normalize_rows(embeddings).astype(np.float32)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        embeddings=embeddings,
        image_id=np.asarray(image_ids, dtype=str),
        path=np.asarray(paths, dtype=str),
        lat=np.asarray(lats, dtype=np.float64),
        lon=np.asarray(lons, dtype=np.float64),
        split=np.asarray(split_names, dtype=str),
        label_id=np.asarray(label_ids, dtype=np.int64),
        model_name=np.asarray([model_name], dtype=str),
        num_classes=np.asarray([num_classes], dtype=np.int64),
        checkpoint=np.asarray([str(checkpoint)], dtype=str),
        dataset_csv=np.asarray([str(dataset_csv)], dtype=str),
        image_size=np.asarray([int(args.image_size)], dtype=np.int32),
        normalized=np.asarray([1 if args.normalize else 0], dtype=np.int8),
    )

    elapsed = time.time() - start
    print(f"Wrote embeddings: {out_npz}")
    print(f"Rows: {embeddings.shape[0]}, Dim: {embeddings.shape[1]}")
    print(f"Model: {model_name}, Device: {device}")
    print(f"Elapsed: {elapsed:.1f}s")
    return 0
