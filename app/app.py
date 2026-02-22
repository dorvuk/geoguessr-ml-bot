from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image


def _streamlit_version_tuple() -> tuple[int, int, int]:
    raw = getattr(st, "__version__", "0.0.0")
    parts = raw.split(".")
    nums = []
    for p in parts[:3]:
        token = "".join(ch for ch in p if ch.isdigit())
        nums.append(int(token) if token else 0)
    while len(nums) < 3:
        nums.append(0)
    return nums[0], nums[1], nums[2]


def st_image_compat(image_obj, caption: str) -> None:
    try:
        st.image(image_obj, caption=caption, use_container_width=True)
    except TypeError:
        st.image(image_obj, caption=caption, use_column_width=True)


def st_dataframe_compat(df: pd.DataFrame) -> None:
    # Older Streamlit (for example 1.19.x) can fail on modern pandas/pyarrow
    # with `LargeUtf8` during Arrow serialization.
    if _streamlit_version_tuple() <= (1, 24, 0):
        st.markdown(df.to_html(index=False), unsafe_allow_html=True)
        return

    try:
        st.dataframe(df, use_container_width=True)
    except TypeError:
        st.dataframe(df)


def st_primary_button_compat(label: str) -> bool:
    try:
        return st.button(label, type="primary")
    except TypeError:
        return st.button(label)


def run_geobot_predict(
    image_bytes: bytes,
    image_suffix: str,
    repo_root: Path,
    checkpoint: str,
    index_path: str,
    index_meta_npz: str,
    image_size: int,
    k: int,
    show_neighbors: int,
    metric: str,
    cosine_temperature: float,
    device: str,
    amp: bool,
) -> tuple[dict, str]:
    with tempfile.TemporaryDirectory(prefix="geobot_ui_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        image_path = tmp_dir_path / f"query{image_suffix or '.jpg'}"
        out_json = tmp_dir_path / "prediction.json"
        image_path.write_bytes(image_bytes)

        cmd = [
            sys.executable,
            "-m",
            "geobot",
            "predict",
            "--repo-root",
            str(repo_root),
            "--image",
            str(image_path),
            "--checkpoint",
            checkpoint,
            "--index-path",
            index_path,
            "--index-meta-npz",
            index_meta_npz,
            "--image-size",
            str(image_size),
            "--k",
            str(k),
            "--show-neighbors",
            str(show_neighbors),
            "--metric",
            metric,
            "--cosine-temperature",
            str(cosine_temperature),
            "--device",
            device,
            "--json-out",
            str(out_json),
        ]
        if amp:
            cmd.append("--amp")

        completed = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        if completed.returncode != 0:
            raise RuntimeError(output.strip() or f"predict command failed (code {completed.returncode})")
        if not out_json.exists():
            raise RuntimeError("predict command did not produce JSON output")

        payload = json.loads(out_json.read_text(encoding="utf-8"))
        return payload, output


def main() -> None:
    st.set_page_config(page_title="GeoBot Predictor", layout="wide")
    st.title("GeoBot Image Geolocation")
    st.caption("Upload one street-view image and run retrieval-based geolocation.")

    repo_root = Path(__file__).resolve().parents[1]
    default_checkpoint = repo_root / "models" / "classifier_fine_effb0_50k" / "best.pt"
    default_index = repo_root / "models" / "retrieval" / "fine_effb0_50k_train.faiss"
    default_index_meta = repo_root / "models" / "retrieval" / "fine_effb0_50k_train_meta.npz"

    st.sidebar.header("Model Settings")
    checkpoint = st.sidebar.text_input("Checkpoint", value=str(default_checkpoint))
    index_path = st.sidebar.text_input("FAISS Index", value=str(default_index))
    index_meta_npz = st.sidebar.text_input("Index Metadata NPZ", value=str(default_index_meta))

    st.sidebar.header("Predict Settings")
    image_size = st.sidebar.number_input("Image Size", min_value=64, max_value=1024, value=224, step=32)
    k = st.sidebar.number_input("K Neighbors (Estimate)", min_value=1, max_value=100, value=1, step=1)
    show_neighbors = st.sidebar.number_input("Shown Neighbors", min_value=1, max_value=100, value=5, step=1)
    metric = st.sidebar.selectbox("Metric", options=["auto", "cosine", "l2"], index=0)
    cosine_temperature = st.sidebar.number_input("Cosine Temperature", min_value=0.1, max_value=200.0, value=20.0)
    device = st.sidebar.selectbox("Device", options=["auto", "cuda", "cpu"], index=0)
    amp = st.sidebar.checkbox("AMP (CUDA only)", value=True)

    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is None:
        st.info("Upload an image to run prediction.")
        return

    image_bytes = uploaded.getvalue()
    suffix = Path(uploaded.name).suffix.lower()
    pil_img = Image.open(uploaded)

    left, right = st.columns([1, 1])
    with left:
        st_image_compat(pil_img, caption=uploaded.name)
    with right:
        st.write("Ready to predict with the current settings.")
        run_clicked = st_primary_button_compat("Run Prediction")

    if not run_clicked:
        return

    try:
        payload, command_output = run_geobot_predict(
            image_bytes=image_bytes,
            image_suffix=suffix,
            repo_root=repo_root,
            checkpoint=checkpoint,
            index_path=index_path,
            index_meta_npz=index_meta_npz,
            image_size=int(image_size),
            k=int(k),
            show_neighbors=int(show_neighbors),
            metric=metric,
            cosine_temperature=float(cosine_temperature),
            device=device,
            amp=amp,
        )
    except Exception as e:
        st.error(str(e))
        return

    st.success("Prediction complete.")
    metric_name = "Similarity" if payload.get("metric") == "cosine" else "Distance"
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Latitude", f"{payload['pred_lat']:.6f}")
    c2.metric("Predicted Longitude", f"{payload['pred_lon']:.6f}")
    c3.metric("Confidence (heuristic)", f"{payload['confidence']:.3f}")

    st.write(
        f"Top-1 {metric_name.lower()}: `{payload['top1_score']:.6f}` | "
        f"Top-2 {metric_name.lower()}: `{payload['top2_score']:.6f}` | "
        f"Margin: `{payload['top12_margin']:.6f}`"
    )

    map_df = pd.DataFrame(
        [
            {"label": "prediction", "lat": payload["pred_lat"], "lon": payload["pred_lon"]},
            *[
                {"label": f"nn_{row['rank']}", "lat": row["lat"], "lon": row["lon"]}
                for row in payload.get("neighbors", [])
            ],
        ]
    )
    st.subheader("Map")
    st.map(map_df[["lat", "lon"]])

    neighbors_df = pd.DataFrame(payload.get("neighbors", []))
    st.subheader("Nearest Neighbors")
    st_dataframe_compat(neighbors_df)

    with st.expander("Raw Prediction JSON"):
        st.json(payload)

    with st.expander("CLI Output"):
        st.code(command_output.strip(), language="text")


if __name__ == "__main__":
    main()
