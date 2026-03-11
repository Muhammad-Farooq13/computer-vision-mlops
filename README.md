# Computer Vision MLOps

[![CI](https://github.com/Muhammad-Farooq13/computer-vision-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/Muhammad-Farooq13/computer-vision-mlops/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-green.svg)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready **vehicle detection and classification** pipeline built with
PyTorch, Flask, Docker, and GitHub Actions CI/CD.  Five classes (Ambulance, Bus,
Car, Motorcycle, Truck) — ResNet / EfficientNet / VGG backbones — one command to
train, one command to serve.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Quick Start](#quick-start)
  - [Local (no GPU)](#local-no-gpu)
  - [Full GPU install](#full-gpu-install)
  - [Docker](#docker)
  - [Streamlit app](#streamlit-app)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Flask API](#flask-api)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

---

## Features

| Category | Details |
|----------|---------|
| **Architectures** | ResNet-18/50, EfficientNet-B0, VGG-16 (PyTorch) |
| **Augmentation** | Albumentations — flip, rotate, brightness/contrast, SSR |
| **Feature extraction** | Deep CNN (ResNet/VGG) + classical SIFT / ORB |
| **Serving** | Flask REST API · Swagger UI · Gunicorn · Docker |
| **CI/CD** | GitHub Actions — lint, pytest matrix, Docker build + health-check |
| **Demo** | Streamlit dashboard (CPU-only, runs on Streamlit Cloud) |

---

## Project Structure

```
computer-vision-mlops/
|-- src/
|   |-- data/
|   |   |-- data_loader.py          # Dataset loading + annotation parser
|   |   `-- data_preprocessing.py   # Albumentations pipeline
|   |-- features/
|   |   `-- feature_extractor.py    # CNN deep features + SIFT/ORB
|   |-- models/
|   |   |-- train.py                # CVModel + ModelTrainer (early stopping)
|   |   |-- evaluate.py             # Accuracy/F1/confusion matrix
|   |   `-- predict.py              # ModelInference + REST helper
|   |-- utils/
|   |   |-- config.py               # YAML config loader
|   |   |-- logger.py               # Rotating file logger
|   |   `-- helpers.py              # Seed, EarlyStopping, count_parameters
|   `-- visualization/
|       `-- visualize.py            # Loss curves, confusion matrix plots
|
|-- tests/                          # pytest suite (15 pass, 4 skip on CI)
|-- api/templates/index.html        # Web upload interface
|-- notebooks/                      # Jupyter exploration notebooks
|-- flask_app.py                    # Entry point for REST API
|-- streamlit_app.py                # Streamlit dashboard entry point
|-- Dockerfile                      # Production image (python:3.11-slim)
|-- docker-compose.yml
|-- requirements.txt                # Full runtime deps (local/GPU)
|-- requirements-ci.txt             # Lean CI deps (CPU-safe, no torch)
|-- pyproject.toml                  # PEP 517 build + linting config
|-- config.yaml                     # Centralised model & data settings
`-- pytest.ini
```

---

## Dataset

| Property | Value |
|----------|-------|
| Task | Object detection (bounding-box annotations) |
| Classes | Ambulance · Bus · Car · Motorcycle · Truck |
| Format | `filename x1,y1,x2,y2,class_id` per line |
| Source | [Roboflow](https://roboflow.com) — see `README.dataset.txt` |
| Splits | `train/`, `valid/`, `test/` — each with `_annotations.txt` |

Images are **not** committed to this repository.  Download the dataset from
Roboflow and place the split folders at the repo root.

---

## Quick Start

### Local (no GPU)

```bash
# Clone
git clone https://github.com/Muhammad-Farooq13/computer-vision-mlops.git
cd computer-vision-mlops

# Create virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install CPU-safe dependencies
pip install -r requirements-ci.txt
pip install -e .
```

### Full GPU install

```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install albumentations opencv-python
pip install -e .
```

### Docker

```bash
# Build
docker build -t cv-mlops:latest .

# Run (exposes port 5000)
docker run -p 5000:5000 cv-mlops:latest

# Or with compose
docker compose up --build
```

Health-check is available at `GET http://localhost:5000/health`.

### Streamlit app

```bash
pip install streamlit>=1.32 pyarrow>=12.0
streamlit run streamlit_app.py
```

Or visit the live demo on **Streamlit Cloud** (badge at the top of this README).

---

## Usage

### Training

```bash
python src/models/train.py
```

Key `config.yaml` options:

| Key | Default | Options |
|-----|---------|---------|
| `model.architecture` | `resnet50` | `resnet18`, `resnet50`, `efficientnet_b0`, `vgg16` |
| `model.num_classes` | `10` | set to number of dataset classes |
| `training.epochs` | `50` | any positive int |
| `training.learning_rate` | `0.001` | float |
| `training.optimizer` | `adam` | `adam`, `sgd`, `adamw` |
| `training.scheduler` | `reduce_on_plateau` | `reduce_on_plateau`, `step`, `cosine` |

Model checkpoints are saved to `models/saved_models/`.

### Evaluation

```bash
python src/models/evaluate.py
```

Reports accuracy, precision, recall, F1, and renders a confusion matrix in
`results/`.

### Flask API

```bash
# Development
python flask_app.py

# Production
gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 flask_app:app
```

| Endpoint | Method | Body | Response |
|----------|--------|------|----------|
| `/predict` | POST | `multipart/form-data` with `file` field | `{"class": "Car", "confidence": 0.92, "scores": {...}}` |
| `/health` | GET | — | `{"status": "ok"}` |
| `/docs` | GET | — | Swagger UI |
| `/metrics` | GET | — | Prometheus metrics |

---

## Testing

```bash
# Run all tests
pytest tests/ -v --cov=src/ --cov-report=term-missing

# Expected result (CPU-only environment)
# 15 passed, 4 skipped
```

Tests skip automatically when GPU packages (`torch`, `torchvision`,
`albumentations`) are not installed — no collection errors in CI.

```
tests/
|-- conftest.py             # Fixtures (sample_model uses importorskip)
|-- test_data_loader.py
|-- test_preprocessing.py
|-- test_model_training.py  # skipped without torchvision
|-- test_inference.py       # skipped without torchvision
|-- test_flask_api.py       # skipped without torchvision
`-- test_utils.py
```

---

## CI/CD

Workflow: `.github/workflows/ci.yml`

```
push / pull_request → main
        |
        +--> test (matrix: Python 3.9, 3.10, 3.11)
        |       pip install requirements-ci.txt
        |       pip install -e .
        |       flake8 src/ tests/
        |       pytest tests/ --cov=src/ --cov-report=xml
        |       Upload coverage → Codecov
        |
        `--> docker (push to main only)
                docker buildx build
                docker run → sleep 30 → curl /health
```

All Actions use **Node 20** runners (`checkout@v4`, `setup-python@v5`,
`cache@v4`, `codecov-action@v5`, `buildx@v3`).

---

## Configuration

All runtime settings live in `config.yaml`.  The config is loaded via
`src/utils/config.py`:

```python
from src.utils.config import load_config
cfg = load_config("config.yaml")
print(cfg["model"]["architecture"])  # resnet50
```

Environment variables (`.env`) are loaded via `python-dotenv` and can override
config values at runtime.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-change`
3. Install dev deps: `pip install -r requirements-ci.txt && pip install -e .`
4. Make your changes and add tests
5. Run `flake8 src/ tests/` and `pytest tests/ -v`
6. Open a Pull Request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) and follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

---

## License

MIT License — see [LICENSE](LICENSE) for details.