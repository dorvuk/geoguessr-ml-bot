import argparse
from pathlib import Path

from geobot.data_utils import (
    add_cell_column,
    assign_group_split,
    filter_min_class_count,
    load_metadata,
    with_label_ids,
)
from geobot.io_utils import resolve_path, write_json


def register_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("prepare", help="Build labeled train/val/test metadata")
    parser.add_argument("--repo-root", default=".", help="Repository root directory")
    parser.add_argument("--metadata-csv", default="data/processed/mapillary/metadata_clean.csv")
    parser.add_argument("--output-csv", default="data/processed/mapillary/dataset.csv")
    parser.add_argument("--labels-json", default="data/processed/mapillary/label_map.json")
    parser.add_argument("--summary-json", default="data/processed/mapillary/dataset_summary.json")
    parser.add_argument("--cell-size-deg", type=float, default=1.0, help="Grid size in degrees")
    parser.add_argument("--min-class-count", type=int, default=5, help="Minimum rows per cell")
    parser.add_argument("--group-col", default="sequence", help="Grouping key used for split isolation")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    repo_root = resolve_path(args.repo_root, Path.cwd())
    metadata_csv = resolve_path(args.metadata_csv, repo_root)
    output_csv = resolve_path(args.output_csv, repo_root)
    labels_json = resolve_path(args.labels_json, repo_root)
    summary_json = resolve_path(args.summary_json, repo_root)

    df = load_metadata(metadata_csv, repo_root=repo_root, require_files=True, dedupe_image_ids=True)
    if df.empty:
        raise RuntimeError(f"no usable rows after loading metadata: {metadata_csv}")

    df = add_cell_column(df, cell_size_deg=args.cell_size_deg, cell_col="cell_id")
    pre_filter_rows = len(df)
    df = filter_min_class_count(df, class_col="cell_id", min_count=args.min_class_count)
    if df.empty:
        raise RuntimeError(
            f"all rows were filtered by min_class_count={args.min_class_count}. "
            "Lower it or download more data."
        )

    group_col = args.group_col if args.group_col in df.columns else "image_id"
    df["split"] = assign_group_split(
        df,
        group_col=group_col,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    df, label_map = with_label_ids(df, class_col="cell_id")

    output_cols = [
        "image_id",
        "path",
        "lat",
        "lon",
        "cell_id",
        "label_id",
        "split",
        "captured_at",
        "compass_angle",
        "sequence",
        "source",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df[output_cols].to_csv(output_csv, index=False)

    write_json(labels_json, label_map)
    split_counts = {k: int(v) for k, v in df["split"].value_counts().to_dict().items()}
    summary = {
        "input_rows": int(pre_filter_rows),
        "output_rows": int(len(df)),
        "num_classes": int(len(label_map)),
        "cell_size_deg": float(args.cell_size_deg),
        "min_class_count": int(args.min_class_count),
        "group_col_used": group_col,
        "split_counts": split_counts,
        "output_csv": str(output_csv),
        "labels_json": str(labels_json),
    }
    write_json(summary_json, summary)

    print(f"Prepared rows: {len(df)}")
    print(f"Prepared classes: {len(label_map)}")
    print(f"Prepared split counts: {split_counts}")
    print(f"Dataset CSV: {output_csv}")
    return 0
