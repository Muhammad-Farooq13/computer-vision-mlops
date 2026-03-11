"""
Computer Vision MLOps — Streamlit Dashboard
Interactive demo for the vehicle detection/classification pipeline.

Runs entirely on CPU-safe packages (no torch, no cv2 required).
"""
from __future__ import annotations

import io
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageOps

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CV MLOps Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CLASSES = ["Ambulance", "Bus", "Car", "Motorcycle", "Truck"]
CLASS_COLORS = {
    "Ambulance": "#ff4b4b",
    "Bus": "#ffa421",
    "Car": "#00d4ff",
    "Motorcycle": "#21c354",
    "Truck": "#7d5bf5",
}
REPO_ROOT = Path(__file__).parent


def _parse_annotations(filepath: Path) -> pd.DataFrame:
    """Parse annotation file format:  filename  x1,y1,x2,y2,class_id ..."""
    records = []
    if not filepath.exists():
        return pd.DataFrame(columns=["filename", "class_id", "class_name",
                                     "x1", "y1", "x2", "y2", "width", "height"])

    for line in filepath.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        fname = parts[0]
        for box_str in parts[1:]:
            try:
                coords = list(map(int, box_str.split(",")))
                if len(coords) == 5:
                    x1, y1, x2, y2, cls = coords
                    records.append({
                        "filename": fname,
                        "class_id": cls,
                        "class_name": CLASSES[cls] if cls < len(CLASSES) else "Unknown",
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "width": abs(x2 - x1),
                        "height": abs(y2 - y1),
                    })
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def load_all_annotations() -> dict[str, pd.DataFrame]:
    splits = {}
    for split in ("train", "valid", "test"):
        ann_path = REPO_ROOT / split / "_annotations.txt"
        splits[split] = _parse_annotations(ann_path)
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/?size=80&id=1295&format=png", width=64)
    st.title("CV MLOps")
    st.caption("Vehicle Detection · PyTorch · Flask · Docker")
    st.divider()

    st.subheader("Project Links")
    st.markdown(
        "[GitHub Repo](https://github.com/Muhammad-Farooq13/computer-vision-mlops)  \n"
        "[Roboflow Dataset](https://roboflow.com)  \n"
        "[Flask API](http://localhost:5000/docs)"
    )
    st.divider()
    st.caption("**Stack:** PyTorch · ResNet · EfficientNet · VGG · Flask · "
               "Gunicorn · Docker · GitHub Actions")

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs(["🏠 Overview", "📊 Dataset EDA", "🖼️ Predict Demo", "🏗️ Architecture"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.title("🚗 Computer Vision MLOps Pipeline")
    st.markdown(
        """
        An end-to-end MLOps pipeline for **vehicle detection and classification**
        using state-of-the-art deep learning architectures.

        | Layer | Technology |
        |-------|-----------|
        | **Model** | ResNet-18/50, EfficientNet-B0, VGG-16 (PyTorch) |
        | **API** | Flask + Flasgger (Swagger UI) |
        | **Serving** | Gunicorn + Docker |
        | **CI/CD** | GitHub Actions (pytest, flake8, docker build) |
        | **Augmentation** | Albumentations (flip, rotate, brightness/contrast) |
        | **Feature Methods** | Deep CNN features + classical SIFT/ORB |
        """
    )

    st.divider()

    # Annotation summary
    with st.spinner("Loading annotations…"):
        all_ann = load_all_annotations()

    total_boxes = sum(len(df) for df in all_ann.values())
    total_images = sum(
        df["filename"].nunique() for df in all_ann.values() if not df.empty
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Annotations", f"{total_boxes:,}")
    col2.metric("Unique Images", f"{total_images:,}")
    col3.metric("Vehicle Classes", len(CLASSES))
    col4.metric("Splits", "Train / Valid / Test")

    st.divider()

    # Class distribution across all splits
    all_combined = pd.concat(all_ann.values(), ignore_index=True)
    if not all_combined.empty:
        st.subheader("Class Distribution (all splits)")
        class_counts = (
            all_combined["class_name"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "class_name", "count": "count",
                              "class_name": "Vehicle Class", "class_name": "Vehicle Class"})
        )
        # Use pandas value_counts cleanly
        vc = all_combined["class_name"].value_counts()
        chart_df = pd.DataFrame({"Vehicle Class": vc.index, "Count": vc.values})
        st.bar_chart(chart_df.set_index("Vehicle Class"))
    else:
        st.info("Annotation files not found — add train/valid/test split folders "
                "with _annotations.txt to see dataset stats.")

    st.divider()
    st.subheader("Pipeline Architecture")
    st.code(
        textwrap.dedent("""\
        Image Input
            │
            ▼
        data_loader.py       ← reads images + annotations
            │
            ▼
        data_preprocessing.py ← albumentations augmentations
            │
            ▼
        feature_extractor.py  ← ResNet/VGG deep features | SIFT/ORB
            │
            ▼
        train.py              ← PyTorch training loop (early stopping, LR scheduler)
            │
            ▼
        evaluate.py           ← accuracy, precision, recall, F1, confusion matrix
            │
            ▼
        predict.py  ──────────► flask_app.py  ← REST API /predict endpoint
        """),
        language="text",
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATASET EDA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.title("📊 Dataset Exploratory Analysis")

    with st.spinner("Loading annotations…"):
        all_ann = load_all_annotations()

    all_combined = pd.concat(
        [df.assign(split=s) for s, df in all_ann.items() if not df.empty],
        ignore_index=True,
    )

    if all_combined.empty:
        st.warning(
            "No annotation data found in train/, valid/, or test/ directories.  \n"
            "Make sure each split folder contains a `_annotations.txt` file."
        )
    else:
        # Per-split stats
        st.subheader("Per-split Summary")
        summary_rows = []
        for split, df in all_ann.items():
            if df.empty:
                summary_rows.append({"Split": split, "Images": 0, "Boxes": 0,
                                     "Avg boxes/img": 0.0})
            else:
                n_img = df["filename"].nunique()
                n_box = len(df)
                summary_rows.append({
                    "Split": split,
                    "Images": n_img,
                    "Boxes": n_box,
                    "Avg boxes/img": round(n_box / max(n_img, 1), 2),
                })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        st.divider()

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Class Distribution by Split")
            pivot = (
                all_combined.groupby(["split", "class_name"])
                .size()
                .reset_index(name="count")
                .pivot(index="class_name", columns="split", values="count")
                .fillna(0)
                .astype(int)
            )
            st.dataframe(pivot, use_container_width=True)

        with col_right:
            st.subheader("Bounding Box Size Distribution")
            area = all_combined["width"] * all_combined["height"]
            area_df = pd.DataFrame({"area_px": area})
            bins = pd.cut(area_df["area_px"], bins=10)
            bin_counts = bins.value_counts().sort_index()
            st.bar_chart(
                pd.DataFrame({"count": bin_counts.values},
                             index=[str(b) for b in bin_counts.index])
            )

        st.divider()
        st.subheader("Bounding Box Aspect Ratio by Class")
        all_combined["aspect_ratio"] = (
            all_combined["width"] / all_combined["height"].replace(0, np.nan)
        ).fillna(0)
        ar_stats = (
            all_combined.groupby("class_name")["aspect_ratio"]
            .agg(["mean", "std", "min", "max"])
            .round(2)
            .rename(columns={"mean": "Mean AR", "std": "Std AR",
                              "min": "Min AR", "max": "Max AR"})
        )
        st.dataframe(ar_stats, use_container_width=True)

        st.divider()
        st.subheader("Images with Most Annotations (top 10)")
        img_counts = (
            all_combined.groupby("filename").size()
            .reset_index(name="num_boxes")
            .nlargest(10, "num_boxes")
        )
        st.dataframe(img_counts, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDICT DEMO
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.title("🖼️ Predict Demo")
    st.markdown(
        "Upload a vehicle image to see the preprocessing pipeline in action.  \n"
        "Pixel-level feature statistics are extracted with **NumPy + Pillow** "
        "(no GPU required)."
    )

    uploaded = st.file_uploader(
        "Upload an image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        help="Any vehicle photo works — the model pipeline shows preprocessing steps.",
    )

    if uploaded is not None:
        img_pil = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
        orig_w, orig_h = img_pil.size

        st.subheader("Original Image")
        st.image(img_pil, caption=f"Original: {orig_w}×{orig_h} px", use_column_width=True)

        st.divider()
        st.subheader("Preprocessing Pipeline")

        # Step 1: resize
        TARGET = (224, 224)
        img_resized = img_pil.resize(TARGET, Image.BILINEAR)

        # Step 2: auto-contrast
        img_contrast = ImageOps.autocontrast(img_resized)

        # Step 3: sharpened
        img_sharp = img_contrast.filter(ImageFilter.SHARPEN)

        # Step 4: grayscale
        img_gray = img_sharp.convert("L")

        cols = st.columns(4)
        cols[0].image(img_resized, caption="1) Resize 224×224", use_column_width=True)
        cols[1].image(img_contrast, caption="2) Auto-contrast", use_column_width=True)
        cols[2].image(img_sharp, caption="3) Sharpen", use_column_width=True)
        cols[3].image(img_gray, caption="4) Grayscale", use_column_width=True)

        st.divider()
        st.subheader("Pixel-Level Feature Statistics")

        arr = np.array(img_resized, dtype=np.float32) / 255.0  # H×W×3 in [0,1]
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        # ImageNet normalisation (what the model would apply)
        MEAN = np.array([0.485, 0.456, 0.406])
        STD  = np.array([0.229, 0.224, 0.225])
        arr_norm = (arr - MEAN) / STD

        stats = {
            "Channel": ["R", "G", "B"],
            "Raw Mean": [f"{r.mean():.4f}", f"{g.mean():.4f}", f"{b.mean():.4f}"],
            "Raw Std":  [f"{r.std():.4f}",  f"{g.std():.4f}",  f"{b.std():.4f}"],
            "Norm Mean": [f"{arr_norm[:,:,0].mean():.4f}",
                          f"{arr_norm[:,:,1].mean():.4f}",
                          f"{arr_norm[:,:,2].mean():.4f}"],
            "Norm Std":  [f"{arr_norm[:,:,0].std():.4f}",
                          f"{arr_norm[:,:,1].std():.4f}",
                          f"{arr_norm[:,:,2].std():.4f}"],
        }
        st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)

        col_a, col_b = st.columns(2)
        col_a.metric("Brightness (mean luminance)", f"{arr.mean():.3f}")
        col_a.metric("Contrast (global std)", f"{arr.std():.3f}")
        col_b.metric("Min pixel", f"{arr.min():.3f}")
        col_b.metric("Max pixel", f"{arr.max():.3f}")

        st.divider()
        st.subheader("Histogram (R / G / B channels)")

        hist_data = {}
        for ch_idx, ch_name in enumerate(["R", "G", "B"]):
            ch_arr = (arr[:, :, ch_idx].ravel() * 255).astype(np.uint8)
            hist_vals, bin_edges = np.histogram(ch_arr, bins=32, range=(0, 256))
            bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).astype(int)
            hist_data[ch_name] = pd.Series(hist_vals, index=bin_centers)

        hist_df = pd.DataFrame(hist_data)
        st.line_chart(hist_df)

        st.info(
            "**In production:** the preprocessed tensor is forwarded through a "
            "pretrained ResNet/EfficientNet/VGG backbone running on GPU, "
            "producing a classification prediction returned by the Flask API."
        )
    else:
        st.info("👆 Upload an image above to begin.")

        # Show a colour swatch as placeholder
        st.subheader("Vehicle Classes")
        swatch_cols = st.columns(len(CLASSES))
        for col, cls in zip(swatch_cols, CLASSES):
            colour = CLASS_COLORS[cls].lstrip("#")
            r_s = int(colour[0:2], 16)
            g_s = int(colour[2:4], 16)
            b_s = int(colour[4:6], 16)
            swatch = Image.new("RGB", (80, 80), (r_s, g_s, b_s))
            col.image(swatch, caption=cls, use_column_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.title("🏗️ Model Architectures")

    st.subheader("Supported Backbones")
    arch_df = pd.DataFrame({
        "Architecture": ["ResNet-18", "ResNet-50", "EfficientNet-B0", "VGG-16"],
        "Params (M)": [11.7, 25.6, 5.3, 138.4],
        "Feature Dim": [512, 2048, 1280, 4096],
        "Input Size": ["224×224"] * 4,
        "Best For": [
            "Fast prototyping / edge devices",
            "Strong baseline (default)",
            "Mobile / resource-constrained",
            "High-accuracy baseline",
        ],
    })
    st.dataframe(arch_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Feature Extraction Methods")
    feat_df = pd.DataFrame({
        "Method": ["ResNet-50 (deep)", "VGG-16 (deep)", "SIFT (classical)", "ORB (classical)"],
        "Type": ["CNN", "CNN", "Keypoint", "Keypoint"],
        "Output Dim": ["2048", "4096", "variable", "variable"],
        "GPU Required": ["Yes", "Yes", "No", "No"],
        "Description": [
            "Global average-pooled features from final conv block",
            "Penultimate FC layer activations",
            "Scale-invariant keypoints + descriptors",
            "Binary descriptors — fast & rotation-invariant",
        ],
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Training Configuration (config.yaml)")
    cfg_col1, cfg_col2 = st.columns(2)
    with cfg_col1:
        st.markdown("""
| Parameter | Value |
|-----------|-------|
| Default architecture | ResNet-50 |
| Num classes | 10 |
| Dropout | 0.5 |
| Freeze backbone | False |
| Optimizer | Adam |
| Base LR | 0.001 |
        """)
    with cfg_col2:
        st.markdown("""
| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Batch size | 32 |
| Image size | 224×224 |
| Scheduler | ReduceOnPlateau |
| Early stopping | 10 epochs |
| Mixed precision | False |
        """)

    st.divider()
    st.subheader("Data Augmentation (Albumentations)")
    aug_df = pd.DataFrame({
        "Transform": [
            "HorizontalFlip", "VerticalFlip", "Rotate",
            "RandomBrightnessContrast", "ShiftScaleRotate", "Normalize",
        ],
        "Probability": [0.5, 0.2, "±15°", 0.3, 0.5, "always (ImageNet stats)"],
        "Split": [
            "train", "train", "train", "train", "train", "train + valid",
        ],
    })
    st.dataframe(aug_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Flask API Endpoints")
    api_df = pd.DataFrame({
        "Endpoint": ["/predict", "/health", "/docs", "/metrics"],
        "Method": ["POST", "GET", "GET", "GET"],
        "Description": [
            "Upload image → returns class + confidence scores",
            "Liveness probe (Kubernetes / Docker healthcheck)",
            "Swagger UI (Flasgger)",
            "Prometheus metrics",
        ],
        "Auth": ["None (add bearer token for production)", "—", "—", "—"],
    })
    st.dataframe(api_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("CI / CD Pipeline")
    st.code(
        textwrap.dedent("""\
        on: push / pull_request → main

        jobs:
          test:
            matrix: python [3.9, 3.10, 3.11]
            steps:
              pip install requirements-ci.txt   # CPU-safe, no GPU deps
              pip install -e .                   # editable install of src/
              flake8 src/ tests/                 # lint
              pytest tests/ --cov=src/           # 15 pass, 4 skip on CI

          docker:
            build image → health-check /health endpoint
            (only on push to main)
        """),
        language="text",
    )
