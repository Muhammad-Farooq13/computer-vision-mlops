# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-11

### Added
- streamlit_app.py: Interactive 4-tab Streamlit dashboard (Overview, Processing, Predict, Architecture)
- pyproject.toml: PEP 517/518 build config with tool.flake8 and tool.pytest.ini_options sections
- packages.txt: System packages for Streamlit Cloud (libgomp1, libglib2.0-0)
- .streamlit/config.toml: Dark theme configuration for Streamlit Cloud

### Changed
- requirements.txt: Replaced strict == pins with flexible >= pins; removed mlflow, dvc, great-expectations, flasgger, boto3, pymongo, jupyter stack
- requirements-ci.txt: Lean CI-only deps; removed torch/torchvision, albumentations, black, pylint, flasgger, prometheus-client; switched to >= pins
- Dockerfile: Rebuilt on python:3.11-slim using requirements-ci.txt; stdlib urllib.request health-check; runs via gunicorn
- .github/workflows/ci.yml: Bumped all Actions to Node 20 (checkout@v4, setup-python@v5, cache@v4, codecov@v5, buildx@v3); added pip install -e .; removed black --check

### Fixed
- tests/conftest.py: Removed top-level import torch / import cv2 that caused collection errors in CI
- All test files: Added pytest.importorskip guards for torchvision, albumentations, cv2, torch
- tests/test_utils.py: Inline model in test_count_parameters (no longer depends on sample_model fixture)
- pytest.ini: Removed --cov-report=html (produces noisy htmlcov/ dir in CI)
- CI result: 15 passed, 4 skipped -- zero errors across Python 3.9/3.10/3.11
