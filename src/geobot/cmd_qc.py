import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

from geobot.constants import METADATA_COLUMNS
from geobot.data_utils import canonicalize_metadata_columns, to_absolute_path
from geobot.io_utils import resolve_path, write_json


def _to_repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def register_subcommand(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("qc", help="Validate and clean downloaded metadata")
    parser.add_argument("--repo-root", default=".", help="Repository root directory")
    parser.add_argument("--metadata-csv", default="data/raw/mapillary/metadata.csv")
    parser.add_argument("--cleaned-csv", default="data/processed/mapillary/metadata_clean.csv")
    parser.add_argument("--report-json", default="data/processed/mapillary/qc_report.json")
    parser.add_argument(
        "--max-image-checks",
        type=int,
        default=0,
        help="Limit expensive image decode checks (0 means check all candidate rows)",
    )
    parser.add_argument(
        "--skip-image-verify",
        action="store_true",
        help="Skip opening image files during QC",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    repo_root = resolve_path(args.repo_root, Path.cwd())
    metadata_csv = resolve_path(args.metadata_csv, repo_root)
    cleaned_csv = resolve_path(args.cleaned_csv, repo_root)
    report_json = resolve_path(args.report_json, repo_root)

    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata file not found: {metadata_csv}")

    raw_df = pd.read_csv(metadata_csv, dtype=str, keep_default_na=False)
    canonical = canonicalize_metadata_columns(raw_df)
    total_rows = int(len(canonical))

    duplicate_count = int(canonical.duplicated(subset=["image_id"], keep="first").sum())
    canonical = canonical.drop_duplicates(subset=["image_id"], keep="first").copy()

    canonical["lat"] = pd.to_numeric(canonical["lat"], errors="coerce")
    canonical["lon"] = pd.to_numeric(canonical["lon"], errors="coerce")

    missing_required = (
        (canonical["image_id"] == "")
        | (canonical["path"] == "")
        | canonical["lat"].isna()
        | canonical["lon"].isna()
    )
    invalid_coords = ~canonical["lat"].between(-90.0, 90.0) | ~canonical["lon"].between(-180.0, 180.0)

    canonical["abs_path"] = [to_absolute_path(p, repo_root) for p in canonical["path"]]
    missing_files = ~canonical["abs_path"].map(Path.exists)

    unreadable = pd.Series(False, index=canonical.index)
    candidate = canonical.index[(~missing_required) & (~invalid_coords) & (~missing_files)]
    if args.max_image_checks and args.max_image_checks > 0:
        candidate = candidate[: args.max_image_checks]

    if not args.skip_image_verify:
        for idx in candidate:
            img_path = canonical.at[idx, "abs_path"]
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except (UnidentifiedImageError, OSError):
                unreadable.at[idx] = True

    valid = (~missing_required) & (~invalid_coords) & (~missing_files) & (~unreadable)
    cleaned = canonical.loc[valid, METADATA_COLUMNS].copy()
    cleaned["path"] = [_to_repo_relative(p, repo_root) for p in canonical.loc[valid, "abs_path"]]

    cleaned_csv.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(cleaned_csv, index=False)

    report = {
        "input_metadata": str(metadata_csv),
        "output_cleaned_metadata": str(cleaned_csv),
        "total_rows": total_rows,
        "rows_after_dedup": int(len(canonical)),
        "valid_rows": int(len(cleaned)),
        "duplicate_image_ids": duplicate_count,
        "missing_required_rows": int(missing_required.sum()),
        "invalid_coordinate_rows": int(invalid_coords.sum()),
        "missing_file_rows": int(missing_files.sum()),
        "unreadable_image_rows": int(unreadable.sum()),
        "image_verify_checked_rows": int(len(candidate)) if not args.skip_image_verify else 0,
        "image_verify_skipped": bool(args.skip_image_verify),
        "columns_expected": METADATA_COLUMNS,
        "columns_found": [str(col) for col in raw_df.columns.tolist()],
    }
    write_json(report_json, report)

    print(f"QC input rows: {total_rows}")
    print(f"QC valid rows: {len(cleaned)}")
    print(f"QC report: {report_json}")
    print(f"Cleaned metadata: {cleaned_csv}")
    return 0
