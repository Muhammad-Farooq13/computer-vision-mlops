"""
Demo Model Training Script
===========================
Trains a CPU-friendly sklearn RandomForest classifier on color + texture
features extracted with PIL/numpy.  No GPU or PyTorch required.

Feature vector (128-d per image):
  - RGB histogram  : 32 bins × 3 channels = 96 features
  - HSV histogram  : 16 bins (H) + 8 (S) + 8 (V)  = 32 features

Training data:
  If train/ images are present → extracts real features from them.
  Otherwise → generates synthetic feature vectors with per-class color priors
              so the model still produces believable predictions on unseen photos.

Outputs:
  models/demo_model.pkl  — serialised (RandomForestClassifier, LabelEncoder,
                            feature_names, class_names, test_accuracy)
  models/demo_metrics.json — training summary
"""

from __future__ import annotations

import json
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).parent
CLASSES = ["Ambulance", "Bus", "Car", "Motorcycle", "Truck"]
MODEL_DIR = REPO_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "demo_model.pkl"
METRICS_PATH = MODEL_DIR / "demo_metrics.json"

# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(img: Image.Image) -> np.ndarray:
    """Extract a 128-d feature vector from a PIL image."""
    img_rgb = img.convert("RGB").resize((128, 128))
    arr_rgb = np.array(img_rgb, dtype=np.float32) / 255.0  # H×W×3

    # RGB histogram (32 bins each)
    rgb_feats = []
    for ch in range(3):
        hist, _ = np.histogram(arr_rgb[:, :, ch], bins=32, range=(0, 1))
        rgb_feats.append(hist / hist.sum())          # normalise

    # HSV histogram (16+8+8 bins)
    img_hsv = img_rgb.convert("HSV")
    arr_hsv = np.array(img_hsv, dtype=np.float32)
    h_hist, _ = np.histogram(arr_hsv[:, :, 0], bins=16, range=(0, 255))
    s_hist, _ = np.histogram(arr_hsv[:, :, 1], bins=8,  range=(0, 255))
    v_hist, _ = np.histogram(arr_hsv[:, :, 2], bins=8,  range=(0, 255))

    def _norm(h):
        return h / max(h.sum(), 1e-9)

    feats = np.concatenate(
        [rgb_feats[0], rgb_feats[1], rgb_feats[2],
         _norm(h_hist), _norm(s_hist), _norm(v_hist)]
    )
    return feats.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Annotation parser (reuses train/ images when present)
# ─────────────────────────────────────────────────────────────────────────────

def _load_real_data() -> tuple[list[np.ndarray], list[str]]:
    """Try to load real training images via _annotations.txt."""
    X, y = [], []
    for split in ("train", "valid"):
        ann_file = REPO_ROOT / split / "_annotations.txt"
        if not ann_file.exists():
            continue
        for line in ann_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            fname = parts[0]
            img_path = REPO_ROOT / split / fname
            if not img_path.exists():
                continue
            try:
                img = Image.open(img_path)
                # Determine dominant class from first annotation
                cls_id = int(parts[1].split(",")[-1])
                cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else None
                if cls_name:
                    X.append(extract_features(img))
                    y.append(cls_name)
            except Exception:
                continue
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generator (fallback when no images available)
# ─────────────────────────────────────────────────────────────────────────────

# Per-class colour priors (mean R,G,B in [0,1]) — loosely realistic
_CLASS_PRIORS = {
    "Ambulance": {
        "rgb_means": [0.85, 0.82, 0.80],  # mostly white body
        "rgb_stds":  [0.25, 0.20, 0.20],
        "hue_peak":  0,                    # slight red accent
        "sat":       0.20,
        "val":       0.85,
    },
    "Bus": {
        "rgb_means": [0.75, 0.55, 0.10],  # yellow/orange
        "rgb_stds":  [0.20, 0.25, 0.20],
        "hue_peak":  30,
        "sat":       0.70,
        "val":       0.75,
    },
    "Car": {
        "rgb_means": [0.50, 0.50, 0.55],  # diverse (use neutral)
        "rgb_stds":  [0.30, 0.30, 0.30],
        "hue_peak":  120,
        "sat":       0.30,
        "val":       0.55,
    },
    "Motorcycle": {
        "rgb_means": [0.20, 0.20, 0.22],  # dark
        "rgb_stds":  [0.20, 0.20, 0.20],
        "hue_peak":  200,
        "sat":       0.40,
        "val":       0.30,
    },
    "Truck": {
        "rgb_means": [0.35, 0.32, 0.28],  # grey/brown
        "rgb_stds":  [0.22, 0.22, 0.22],
        "hue_peak":  20,
        "sat":       0.25,
        "val":       0.45,
    },
}


def _build_synthetic_image(prior: dict, rng: np.random.Generator) -> Image.Image:
    """Build a 128×128 PIL image with per-class colour statistics."""
    means = np.array(prior["rgb_means"])
    stds  = np.array(prior["rgb_stds"])

    pixels = rng.normal(loc=means, scale=stds, size=(128, 128, 3))
    pixels = np.clip(pixels, 0.0, 1.0)

    # Add a simple structural pattern (edges / gradients) for realism
    # Horizontal gradient
    grad = np.linspace(0, 0.08, 128)[np.newaxis, :, np.newaxis]
    pixels = np.clip(pixels + grad * rng.uniform(-1, 1, (1, 1, 3)), 0, 1)

    img_arr = (pixels * 255).astype(np.uint8)
    return Image.fromarray(img_arr, mode="RGB")


def _generate_synthetic_data(n_per_class: int = 300,
                              seed: int = 42) -> tuple[list[np.ndarray], list[str]]:
    """Generate synthetic training samples."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for cls_name, prior in _CLASS_PRIORS.items():
        for _ in range(n_per_class):
            img = _build_synthetic_image(prior, rng)
            X.append(extract_features(img))
            y.append(cls_name)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Main training flow
# ─────────────────────────────────────────────────────────────────────────────

def train():
    print("=" * 55)
    print("CV MLOps — Demo Model Training")
    print("=" * 55)

    # 1. Try real data first
    X_real, y_real = _load_real_data()
    if X_real:
        print(f"Loaded {len(X_real)} real-image samples from train/ and valid/")
        X_all, y_all = X_real, y_real
    else:
        print("No images found — generating synthetic training data …")
        X_all, y_all = _generate_synthetic_data(n_per_class=300, seed=42)
        print(f"Generated {len(X_all)} synthetic samples "
              f"({len(X_all) // len(CLASSES)} per class)")

    X = np.array(X_all)
    y = np.array(y_all)

    print(f"\nFeature matrix: {X.shape}")
    print("Class distribution:")
    for cls in CLASSES:
        n = (y == cls).sum()
        print(f"  {cls:<12s}: {n}")

    # 2. Encode labels
    le = LabelEncoder()
    le.fit(CLASSES)
    y_enc = le.transform(y)

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
    )

    # 4. Train RandomForest
    print("\nTraining RandomForestClassifier (200 trees, max_depth=20) …")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True,
    )

    print(f"\nTest accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 6. Save model bundle
    bundle = {
        "model": clf,
        "label_encoder": le,
        "classes": CLASSES,
        "feature_dim": X.shape[1],
        "test_accuracy": float(acc),
        "trained_on_real_data": bool(X_real),
        "n_samples": len(X_all),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f, protocol=4)

    # 7. Save metrics JSON
    metrics = {
        "test_accuracy": float(acc),
        "n_samples": len(X_all),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "classes": CLASSES,
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall":    round(report[cls]["recall"],    4),
                "f1-score":  round(report[cls]["f1-score"],  4),
            }
            for cls in CLASSES
        },
        "trained_on_real_data": bool(X_real),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved → {MODEL_PATH}")
    print(f"Metrics saved → {METRICS_PATH}")
    print("=" * 55)
    print("Done.")


if __name__ == "__main__":
    train()
