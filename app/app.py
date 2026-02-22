from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

MODEL_PROFILES = {
    "fine_effb0_50k (best)": {
        "checkpoint": "models/classifier_fine_effb0_50k/best.pt",
        "index": "models/retrieval/fine_effb0_50k_train.faiss",
        "meta": "models/retrieval/fine_effb0_50k_train_meta.npz",
    },
    "fine_effb0_20k (fallback)": {
        "checkpoint": "models/classifier_fine_effb0_v1/best.pt",
        "index": "models/retrieval/fine_effb0_train.faiss",
        "meta": "models/retrieval/fine_effb0_train_meta.npz",
    },
}


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
    st.sidebar.header("Model Settings")
    profile_names = list(MODEL_PROFILES.keys())
    primary_profile = st.sidebar.selectbox("Primary Model Profile", options=profile_names, index=0)
    primary_defaults = MODEL_PROFILES[primary_profile]

    use_custom_primary = st.sidebar.checkbox("Override primary paths", value=False)
    if use_custom_primary:
        checkpoint = st.sidebar.text_input(
            "Primary Checkpoint",
            value=str(repo_root / Path(primary_defaults["checkpoint"])),
        )
        index_path = st.sidebar.text_input(
            "Primary FAISS Index",
            value=str(repo_root / Path(primary_defaults["index"])),
        )
        index_meta_npz = st.sidebar.text_input(
            "Primary Index Metadata NPZ",
            value=str(repo_root / Path(primary_defaults["meta"])),
        )
    else:
        checkpoint = str(repo_root / Path(primary_defaults["checkpoint"]))
        index_path = str(repo_root / Path(primary_defaults["index"]))
        index_meta_npz = str(repo_root / Path(primary_defaults["meta"]))
        st.sidebar.caption("Primary model files")
        st.sidebar.code(
            f"ckpt: {primary_defaults['checkpoint']}\n"
            f"index: {primary_defaults['index']}\n"
            f"meta: {primary_defaults['meta']}",
            language="text",
        )

    enable_fallback = st.sidebar.checkbox("Enable Auto Fallback Model", value=True)
    fallback_profile = None
    fallback_checkpoint = ""
    fallback_index = ""
    fallback_meta = ""
    fallback_trigger = 0.55
    fallback_min_gain = 0.02
    if enable_fallback:
        fallback_candidates = [x for x in profile_names if x != primary_profile]
        fallback_profile = st.sidebar.selectbox(
            "Fallback Model Profile",
            options=fallback_candidates,
            index=0,
        )
        fallback_defaults = MODEL_PROFILES[fallback_profile]
        use_custom_fallback = st.sidebar.checkbox("Override fallback paths", value=False)
        fallback_trigger = float(
            st.sidebar.number_input(
                "Fallback Trigger (confidence <)",
                min_value=0.0,
                max_value=1.0,
                value=0.55,
                step=0.01,
            )
        )
        fallback_min_gain = float(
            st.sidebar.number_input(
                "Min confidence gain to switch",
                min_value=0.0,
                max_value=1.0,
                value=0.02,
                step=0.01,
            )
        )
        if use_custom_fallback:
            fallback_checkpoint = st.sidebar.text_input(
                "Fallback Checkpoint",
                value=str(repo_root / Path(fallback_defaults["checkpoint"])),
            )
            fallback_index = st.sidebar.text_input(
                "Fallback FAISS Index",
                value=str(repo_root / Path(fallback_defaults["index"])),
            )
            fallback_meta = st.sidebar.text_input(
                "Fallback Index Metadata NPZ",
                value=str(repo_root / Path(fallback_defaults["meta"])),
            )
        else:
            fallback_checkpoint = str(repo_root / Path(fallback_defaults["checkpoint"]))
            fallback_index = str(repo_root / Path(fallback_defaults["index"]))
            fallback_meta = str(repo_root / Path(fallback_defaults["meta"]))
            st.sidebar.caption("Fallback model files")
            st.sidebar.code(
                f"ckpt: {fallback_defaults['checkpoint']}\n"
                f"index: {fallback_defaults['index']}\n"
                f"meta: {fallback_defaults['meta']}",
                language="text",
            )

    st.sidebar.header("Predict Settings")
    image_size = st.sidebar.number_input("Image Size", min_value=64, max_value=1024, value=224, step=32)
    k = st.sidebar.number_input("K Neighbors (Estimate)", min_value=1, max_value=100, value=1, step=1)
    show_neighbors = st.sidebar.number_input("Shown Neighbors", min_value=1, max_value=100, value=5, step=1)
    metric = st.sidebar.selectbox("Metric", options=["auto", "cosine", "l2"], index=0)
    cosine_temperature = st.sidebar.number_input("Cosine Temperature", min_value=0.1, max_value=200.0, value=20.0)
    device = st.sidebar.selectbox("Device", options=["auto", "cuda", "cpu"], index=0)
    amp = st.sidebar.checkbox("AMP (CUDA only)", value=True)
    low_conf_threshold = float(
        st.sidebar.number_input(
            "Low-confidence warning (<)",
            min_value=0.0,
            max_value=1.0,
            value=0.65,
            step=0.01,
        )
    )

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
        primary_payload, primary_output = run_geobot_predict(
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

    chosen_payload = primary_payload
    chosen_output = primary_output
    chosen_profile = primary_profile
    fallback_attempted = False
    fallback_switched = False
    fallback_payload = None
    fallback_output = ""

    if enable_fallback and float(primary_payload.get("confidence", 0.0)) < fallback_trigger:
        fallback_attempted = True
        try:
            fallback_payload, fallback_output = run_geobot_predict(
                image_bytes=image_bytes,
                image_suffix=suffix,
                repo_root=repo_root,
                checkpoint=fallback_checkpoint,
                index_path=fallback_index,
                index_meta_npz=fallback_meta,
                image_size=int(image_size),
                k=int(k),
                show_neighbors=int(show_neighbors),
                metric=metric,
                cosine_temperature=float(cosine_temperature),
                device=device,
                amp=amp,
            )
            primary_conf = float(primary_payload.get("confidence", 0.0))
            fallback_conf = float(fallback_payload.get("confidence", 0.0))
            if fallback_conf >= primary_conf + fallback_min_gain:
                fallback_switched = True
                chosen_payload = fallback_payload
                chosen_output = fallback_output
                chosen_profile = fallback_profile or "fallback"
        except Exception as e:
            st.warning(f"Fallback model failed: {e}")

    st.success("Prediction complete.")
    st.write(f"Active model profile: `{chosen_profile}`")
    if fallback_attempted and fallback_switched:
        st.warning("Primary confidence was low. Switched to fallback model.")
    elif fallback_attempted:
        st.info("Primary confidence was low, but fallback was not more confident.")

    metric_name = "Similarity" if chosen_payload.get("metric") == "cosine" else "Distance"
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Latitude", f"{chosen_payload['pred_lat']:.6f}")
    c2.metric("Predicted Longitude", f"{chosen_payload['pred_lon']:.6f}")
    c3.metric("Confidence (heuristic)", f"{chosen_payload['confidence']:.3f}")

    if float(chosen_payload.get("confidence", 0.0)) < low_conf_threshold:
        st.warning(
            "Low-confidence prediction. Use nearest neighbors as alternatives "
            "and consider manual map refinement."
        )

    st.write(
        f"Top-1 {metric_name.lower()}: `{chosen_payload['top1_score']:.6f}` | "
        f"Top-2 {metric_name.lower()}: `{chosen_payload['top2_score']:.6f}` | "
        f"Margin: `{chosen_payload['top12_margin']:.6f}`"
    )

    map_df = pd.DataFrame(
        [
            {"label": "prediction", "lat": chosen_payload["pred_lat"], "lon": chosen_payload["pred_lon"]},
            *[
                {"label": f"nn_{row['rank']}", "lat": row["lat"], "lon": row["lon"]}
                for row in chosen_payload.get("neighbors", [])
            ],
        ]
    )
    st.subheader("Map")
    st.map(map_df[["lat", "lon"]])

    neighbors_df = pd.DataFrame(chosen_payload.get("neighbors", []))
    st.subheader("Nearest Neighbors")
    st_dataframe_compat(neighbors_df)

    if fallback_attempted and fallback_payload is not None:
        with st.expander("Fallback comparison"):
            st.write(
                f"Primary confidence: `{primary_payload.get('confidence', float('nan')):.3f}` | "
                f"Fallback confidence: `{fallback_payload.get('confidence', float('nan')):.3f}`"
            )
            if fallback_switched:
                st.write("Fallback was selected as final output.")
            else:
                st.write("Primary output was kept.")

    with st.expander("Raw Prediction JSON"):
        st.json(chosen_payload)

    with st.expander("CLI Output"):
        st.code(chosen_output.strip(), language="text")
        if fallback_attempted and fallback_output:
            st.code(fallback_output.strip(), language="text")


if __name__ == "__main__":
    main()
