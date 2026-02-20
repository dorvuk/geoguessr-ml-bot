from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from PIL import Image


def choose_device(requested: str, torch_mod):
    if requested == "auto":
        return torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
    return torch_mod.device(requested)


def parse_splits(raw: str) -> list[str]:
    splits = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not splits:
        raise ValueError("splits list is empty")
    return splits


def load_dataset_frame(dataset_csv: Path, repo_root: Path, splits: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(dataset_csv, dtype=str, keep_default_na=False)
    required = {"image_id", "path", "lat", "lon", "split"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"dataset missing required columns: {missing}")

    df["split"] = df["split"].astype(str).str.strip().str.lower()
    split_set = set(splits)
    df = df[df["split"].isin(split_set)].copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    if "label_id" in df.columns:
        df["label_id"] = pd.to_numeric(df["label_id"], errors="coerce").fillna(-1).astype(int)
    else:
        df["label_id"] = -1

    df = df.dropna(subset=["lat", "lon"]).copy()
    df["path"] = df["path"].astype(str).str.replace("\\", "/", regex=False)
    df["abs_path"] = [resolve_abs_path(Path(p), repo_root) for p in df["path"]]
    exists_mask = [p.exists() for p in df["abs_path"]]
    df = df[exists_mask].copy()
    if df.empty:
        raise RuntimeError("no rows available for selected splits after filtering missing files")

    return df.reset_index(drop=True)


def resolve_abs_path(path: Path, repo_root: Path) -> Path:
    p = path.expanduser()
    if not p.is_absolute():
        p = repo_root / p
    return p


class EmbeddingDataset:
    def __init__(self, frame: pd.DataFrame, transform) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        with Image.open(row["abs_path"]) as img:
            image = img.convert("RGB")
        x = self.transform(image)
        return (
            x,
            str(row["image_id"]),
            str(row["path"]),
            float(row["lat"]),
            float(row["lon"]),
            str(row["split"]),
            int(row["label_id"]),
        )


def _strip_module_prefix(state: dict[str, Any]) -> dict[str, Any]:
    if not state:
        return state
    keys = list(state.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module.") :]: v for k, v in state.items()}
    return state


def build_feature_model(
    timm_mod: Any,
    torch_mod: Any,
    checkpoint_path: Path,
    model_name_override: Optional[str],
    device,
):
    checkpoint = torch_mod.load(checkpoint_path, map_location="cpu")
    model_name = model_name_override or checkpoint.get("model_name")
    if not model_name:
        raise RuntimeError("Unable to infer model name from checkpoint; pass --model-name explicitly.")

    num_classes = int(checkpoint.get("num_classes", 1000))
    model = timm_mod.create_model(model_name, pretrained=False, num_classes=num_classes)

    model_state = checkpoint.get("model_state", checkpoint)
    if not isinstance(model_state, dict):
        raise RuntimeError("checkpoint format unsupported: expected state dict")
    model_state = _strip_module_prefix(model_state)

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys when loading checkpoint: {unexpected[:8]}")
    if missing:
        print(f"Warning: missing keys when loading checkpoint ({len(missing)} keys)")

    model = model.to(device)
    model.eval()
    return model_name, model, num_classes


def extract_embeddings(model, images, torch_mod):
    if hasattr(model, "forward_features"):
        feats = model.forward_features(images)
    else:
        feats = model(images)

    if hasattr(model, "forward_head"):
        try:
            emb = model.forward_head(feats, pre_logits=True)
        except TypeError:
            emb = model.forward_head(feats)
    else:
        emb = feats

    if not isinstance(emb, torch_mod.Tensor):
        emb = torch_mod.as_tensor(emb)
    if emb.ndim > 2:
        emb = emb.flatten(start_dim=1)
    return emb


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def load_embeddings_npz(path: Path) -> dict[str, np.ndarray]:
    required = ["embeddings", "image_id", "path", "lat", "lon", "split", "label_id"]
    with np.load(path, allow_pickle=False) as payload:
        missing = [k for k in required if k not in payload]
        if missing:
            raise RuntimeError(f"embeddings npz missing keys: {missing}")

        out: dict[str, np.ndarray] = {}
        for k in payload.files:
            out[k] = payload[k]

    out["embeddings"] = np.asarray(out["embeddings"], dtype=np.float32)
    if out["embeddings"].ndim != 2:
        raise RuntimeError("embeddings must have shape [N, D]")

    n = int(out["embeddings"].shape[0])
    for key in ["image_id", "path", "split"]:
        out[key] = np.asarray(out[key]).astype(str)
        if out[key].shape[0] != n:
            raise RuntimeError(f"{key} length does not match embeddings")
    for key in ["lat", "lon"]:
        out[key] = np.asarray(out[key], dtype=np.float64)
        if out[key].shape[0] != n:
            raise RuntimeError(f"{key} length does not match embeddings")
    out["label_id"] = np.asarray(out["label_id"], dtype=np.int64)
    if out["label_id"].shape[0] != n:
        raise RuntimeError("label_id length does not match embeddings")
    return out


def select_rows(payload: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    n = int(payload["embeddings"].shape[0])
    for key, value in payload.items():
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] == n:
            out[key] = arr[mask]
        else:
            out[key] = arr
    return out


def filter_embeddings_by_splits(
    payload: dict[str, np.ndarray], splits: Iterable[str]
) -> dict[str, np.ndarray]:
    split_set = {s.lower() for s in splits}
    row_splits = np.asarray(payload["split"]).astype(str)
    mask = np.isin(np.char.lower(row_splits), list(split_set))
    if not np.any(mask):
        raise RuntimeError(f"no rows matched requested splits: {sorted(split_set)}")
    return select_rows(payload, mask)


def weighted_geo_mean(lat: np.ndarray, lon: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    if lat.size == 0 or lon.size == 0:
        raise ValueError("lat/lon arrays must be non-empty")
    if lat.size != lon.size or lat.size != weights.size:
        raise ValueError("lat/lon/weights must have matching lengths")

    w = np.asarray(weights, dtype=np.float64)
    w = np.maximum(w, 0.0)
    if float(w.sum()) <= 0.0:
        w = np.ones_like(w)

    lat_rad = np.radians(np.asarray(lat, dtype=np.float64))
    lon_rad = np.radians(np.asarray(lon, dtype=np.float64))

    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    xw = float(np.sum(w * x))
    yw = float(np.sum(w * y))
    zw = float(np.sum(w * z))
    norm = float(np.sqrt(xw * xw + yw * yw + zw * zw))
    if norm <= 1e-12:
        return float(np.mean(lat)), float(np.mean(lon))

    xw /= norm
    yw /= norm
    zw /= norm

    out_lat = float(np.degrees(np.arctan2(zw, np.sqrt(xw * xw + yw * yw))))
    out_lon = float(np.degrees(np.arctan2(yw, xw)))
    return out_lat, out_lon
