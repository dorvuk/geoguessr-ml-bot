import hashlib
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from geobot.constants import METADATA_COLUMNS
from geobot.geo import build_cell_ids


def canonicalize_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = {col: col.strip().lower() for col in df.columns if isinstance(col, str)}
    out = df.rename(columns=normalized).copy()
    if "sequence_id" in out.columns and "sequence" not in out.columns:
        out["sequence"] = out["sequence_id"]
    for col in METADATA_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out = out[METADATA_COLUMNS].copy()
    for col in out.columns:
        out[col] = out[col].fillna("").astype(str).str.strip()
    return out


def to_absolute_path(raw_path: str, repo_root: Path) -> Path:
    p = Path(raw_path).expanduser()
    if not p.is_absolute():
        p = repo_root / p
    return p


def load_metadata(
    metadata_csv: Path,
    repo_root: Path,
    require_files: bool = True,
    dedupe_image_ids: bool = True,
) -> pd.DataFrame:
    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata file not found: {metadata_csv}")

    df = pd.read_csv(metadata_csv, dtype=str, keep_default_na=False)
    if df.empty:
        out = pd.DataFrame(columns=[*METADATA_COLUMNS, "abs_path"])
        return out

    df = canonicalize_metadata_columns(df)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df[df["lat"].between(-90.0, 90.0) & df["lon"].between(-180.0, 180.0)].copy()

    df = df[df["image_id"] != ""].copy()
    df = df[df["path"] != ""].copy()

    df["path"] = df["path"].str.replace("\\", "/", regex=False)
    df["abs_path"] = [to_absolute_path(p, repo_root) for p in df["path"]]

    if require_files:
        exists_mask = [p.exists() for p in df["abs_path"]]
        df = df[exists_mask].copy()

    if dedupe_image_ids:
        df = df.drop_duplicates(subset=["image_id"], keep="first").copy()

    df = df.reset_index(drop=True)
    return df


def add_cell_column(df: pd.DataFrame, cell_size_deg: float, cell_col: str = "cell_id") -> pd.DataFrame:
    out = df.copy()
    out[cell_col] = build_cell_ids(out["lat"], out["lon"], cell_size_deg=cell_size_deg)
    return out


def filter_min_class_count(df: pd.DataFrame, class_col: str, min_count: int) -> pd.DataFrame:
    if min_count <= 1:
        return df.copy()
    counts = df[class_col].value_counts()
    keep = counts[counts >= min_count].index
    return df[df[class_col].isin(keep)].copy()


def assign_group_split(
    df: pd.DataFrame,
    group_col: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.Series:
    if val_ratio < 0.0 or test_ratio < 0.0:
        raise ValueError("val_ratio and test_ratio must be >= 0")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1")

    groups = df[group_col].fillna("").astype(str).str.strip()
    fallback = df["image_id"].astype(str)
    groups = groups.where(groups != "", fallback)

    def unit_hash(value: str) -> float:
        payload = f"{seed}:{value}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        as_int = int.from_bytes(digest, "big")
        return as_int / float((1 << 64) - 1)

    hashed = groups.map(unit_hash)
    split = pd.Series("train", index=df.index, dtype=str)
    split = split.mask(hashed < test_ratio, "test")
    split = split.mask((hashed >= test_ratio) & (hashed < test_ratio + val_ratio), "val")
    return split


def with_label_ids(df: pd.DataFrame, class_col: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    classes = sorted(df[class_col].astype(str).unique().tolist())
    mapping = {cls: idx for idx, cls in enumerate(classes)}
    out = df.copy()
    out["label_id"] = out[class_col].map(mapping).astype(int)
    return out, mapping


def class_centroids(df: pd.DataFrame, label_col: str = "label_id") -> Dict[int, Tuple[float, float]]:
    grouped = df.groupby(label_col, dropna=False)[["lat", "lon"]].mean()
    return {int(label): (float(row["lat"]), float(row["lon"])) for label, row in grouped.iterrows()}
