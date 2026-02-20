import argparse
from pathlib import Path

import numpy as np

from geobot.io_utils import resolve_path, write_json
from geobot.retrieval_utils import (
    filter_embeddings_by_splits,
    load_embeddings_npz,
    normalize_rows,
    parse_splits,
)


def register_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("index", help="Build a FAISS index from saved embeddings")
    parser.add_argument("--repo-root", default=".", help="Repository root directory")
    parser.add_argument("--embeddings-npz", default="models/retrieval/embeddings_all.npz")
    parser.add_argument("--index-out", default="models/retrieval/index.faiss")
    parser.add_argument("--meta-out", default="models/retrieval/index_meta.npz")
    parser.add_argument("--summary-json", default="models/retrieval/index_summary.json")
    parser.add_argument("--splits", default="train", help="Comma-separated splits to include in index")
    parser.add_argument("--metric", choices=["cosine", "l2"], default="cosine")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    try:
        import faiss
    except ImportError as e:
        raise RuntimeError("Missing FAISS dependency. Run `uv sync`.") from e

    repo_root = resolve_path(args.repo_root, Path.cwd())
    embeddings_npz = resolve_path(args.embeddings_npz, repo_root)
    index_out = resolve_path(args.index_out, repo_root)
    meta_out = resolve_path(args.meta_out, repo_root)
    summary_json = resolve_path(args.summary_json, repo_root)
    splits = parse_splits(args.splits)

    if not embeddings_npz.exists():
        raise FileNotFoundError(f"embeddings file not found: {embeddings_npz}")

    payload = load_embeddings_npz(embeddings_npz)
    payload = filter_embeddings_by_splits(payload, splits=splits)

    vectors = np.asarray(payload["embeddings"], dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise RuntimeError("filtered embeddings are empty")

    if args.metric == "cosine":
        vectors = normalize_rows(vectors).astype(np.float32)
        index = faiss.IndexFlatIP(vectors.shape[1])
    else:
        index = faiss.IndexFlatL2(vectors.shape[1])

    index.add(vectors)

    index_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_out))
    np.savez_compressed(
        meta_out,
        image_id=np.asarray(payload["image_id"]).astype(str),
        path=np.asarray(payload["path"]).astype(str),
        lat=np.asarray(payload["lat"], dtype=np.float64),
        lon=np.asarray(payload["lon"], dtype=np.float64),
        split=np.asarray(payload["split"]).astype(str),
        label_id=np.asarray(payload["label_id"], dtype=np.int64),
        metric=np.asarray([args.metric], dtype=str),
        dim=np.asarray([vectors.shape[1]], dtype=np.int32),
        count=np.asarray([vectors.shape[0]], dtype=np.int64),
    )

    summary = {
        "embeddings_npz": str(embeddings_npz),
        "index_out": str(index_out),
        "meta_out": str(meta_out),
        "metric": args.metric,
        "splits": splits,
        "count": int(vectors.shape[0]),
        "dim": int(vectors.shape[1]),
    }
    write_json(summary_json, summary)

    print(f"Index written: {index_out}")
    print(f"Index metadata: {meta_out}")
    print(f"Rows: {vectors.shape[0]}, Dim: {vectors.shape[1]}, Metric: {args.metric}")
    return 0
