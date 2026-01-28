# Computer Vision Project

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready computer vision project with MLOps best practices, featuring Flask API deployment, Docker support, comprehensive testing, and automated pipelines.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Development](#model-development)
- [Deployment](#deployment)
- [Testing](#testing)
- [MLOps Pipeline](#mlops-pipeline)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project implements a complete computer vision pipeline for image classification tasks. It follows industry best practices and includes:

- **Data Processing**: Automated data loading, preprocessing, and augmentation
- **Model Training**: Support for multiple CNN architectures (ResNet, EfficientNet, VGG)
- **Evaluation**: Comprehensive metrics and visualization tools
- **Deployment**: Flask REST API with Docker containerization
- **MLOps**: Automated pipelines with MLflow tracking
- **Testing**: Unit and integration tests with pytest

### Objectives

- Develop a robust image classification model
- Implement scalable and maintainable code architecture
- Enable easy deployment and monitoring
- Ensure reproducibility and version control
- Follow MLOps best practices

## ✨ Features

### Core Functionality
- 🔍 **Multi-Model Support**: ResNet, EfficientNet, VGG architectures
- 📊 **Data Augmentation**: Comprehensive augmentation pipeline using Albumentations
- 📈 **Training Monitoring**: Real-time metrics tracking with MLflow
- 🎨 **Visualization**: Built-in plotting and analysis tools
- 🔄 **Preprocessing Pipeline**: Automated image preprocessing and normalization

### Deployment & Production
- 🚀 **Flask REST API**: Production-ready API with health checks
- 🐳 **Docker Support**: Containerized deployment with docker-compose
- 📦 **Model Versioning**: Automated model versioning and registry
- 🔧 **Configuration Management**: YAML-based configuration system
- 📝 **Comprehensive Logging**: Structured logging with rotation

### Quality & Testing
- ✅ **Unit Tests**: Comprehensive test coverage with pytest
- 🧪 **Integration Tests**: API and pipeline testing
- 📊 **Code Quality**: Black, Flake8, isort integration
- 🔄 **CI/CD Ready**: Designed for continuous integration

## 📁 Project Structure

```
OPencv/
├── api/                          # Flask API components
│   ├── static/                   # Static files
│   │   └── uploads/              # Uploaded images
│   └── templates/                # HTML templates
│       └── index.html            # Web interface
│
├── data/                         # Data directory
│   ├── raw/                      # Raw dataset
│   │   ├── train/                # Training images
│   │   ├── test/                 # Test images
│   │   └── valid/                # Validation images
│   └── processed/                # Processed data
│
├── models/                       # Model artifacts
│   └── saved_models/             # Trained model checkpoints
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_data_exploration.ipynb # Data analysis
│   └── 02_model_training.ipynb   # Training experiments
│
├── src/                          # Source code
│   ├── data/                     # Data processing
│   │   ├── data_loader.py        # Data loading utilities
│   │   └── data_preprocessing.py # Preprocessing functions
│   │
│   ├── features/                 # Feature engineering
│   │   └── feature_extractor.py  # Feature extraction
│   │
│   ├── models/                   # Model code
│   │   ├── train.py              # Training logic
│   │   ├── evaluate.py           # Evaluation metrics
│   │   └── predict.py            # Inference engine
│   │
│   ├── utils/                    # Utility functions
│   │   ├── config.py             # Configuration management
│   │   ├── logger.py             # Logging utilities
│   │   └── helpers.py            # Helper functions
│   │
│   └── visualization/            # Visualization tools
│       └── visualize.py          # Plotting functions
│
├── tests/                        # Test suite
│   ├── conftest.py               # Test configuration
│   ├── test_data_loader.py       # Data loading tests
│   ├── test_preprocessing.py     # Preprocessing tests
│   ├── test_model_training.py    # Training tests
│   ├── test_inference.py         # Inference tests
│   ├── test_utils.py             # Utility tests
│   └── test_flask_api.py         # API tests
│
├── logs/                         # Log files
├── results/                      # Results and visualizations
│
├── config.yaml                   # Project configuration
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker image definition
├── docker-compose.yml            # Docker orchestration
├── .dockerignore                 # Docker ignore patterns
├── flask_app.py                  # Flask application
├── mlops_pipeline.py             # MLOps pipeline
├── pytest.ini                    # Pytest configuration
└── README.md                     # This file
```

## 📊 Dataset

### Overview

This project uses an image classification dataset with the following structure:

- **Training Set**: Images for model training
- **Validation Set**: Images for hyperparameter tuning
- **Test Set**: Images for final evaluation

### Dataset Information

- **Format**: Images (JPG, PNG, BMP)
- **Splits**: Train/Valid/Test
- **Annotations**: Class labels and bounding boxes (if applicable)

### Data Organization

```
data/raw/
├── train/
│   ├── _annotations.txt    # Training annotations
│   ├── _classes.txt         # Class names
│   └── *.jpg                # Training images
├── valid/
│   ├── _annotations.txt    # Validation annotations
│   ├── _classes.txt         # Class names
│   └── *.jpg                # Validation images
└── test/
    ├── _annotations.txt    # Test annotations
    ├── _classes.txt         # Class names
    └── *.jpg                # Test images
```

### Preprocessing Steps

1. **Resizing**: All images resized to 224×224 pixels
2. **Normalization**: Pixel values normalized using ImageNet statistics
3. **Augmentation**: Random flips, rotations, brightness/contrast adjustments
4. **Data Validation**: Checking for corrupted images and inconsistencies

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training
- (Optional) Docker for containerized deployment

### Local Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Muhammad-Farooq-13/OPencv.git
cd OPencv
```

2. **Create virtual environment**:
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure the project**:
```bash
# Edit config.yaml with your settings
# Adjust paths, model parameters, training configuration
```

5. **Organize your data**:
```bash
# Move your dataset to data/raw/
# Ensure train/, valid/, test/ folders exist
```

### Docker Installation

1. **Build Docker image**:
```bash
docker build -t cv-model:latest .
```

2. **Run with Docker Compose**:
```bash
docker-compose up -d
```

## 💻 Usage

### 1. Data Exploration

Explore your dataset using the provided notebook:

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 2. Train a Model

#### Using Python Script

```bash
python -m src.models.train
```

#### Using Jupyter Notebook

```bash
jupyter notebook notebooks/02_model_training.ipynb
```

#### Using MLOps Pipeline

```bash
python mlops_pipeline.py --mode full
```

### 3. Evaluate Model

```bash
python mlops_pipeline.py --mode evaluate
```

### 4. Run Flask API

```bash
python flask_app.py
```

The API will be available at `http://localhost:5000`

### 5. Make Predictions

#### Using Python API

```python
from src.models.predict import ModelInference
from src.models.train import CVModel
import cv2

# Load model
model = CVModel(model_name='resnet50', num_classes=10)
inference = ModelInference(model, model_path='models/saved_models/best_model.pth')

# Load image
image = cv2.imread('path/to/image.jpg')

# Predict
result = inference.predict(image, top_k=5)
print(f"Prediction: {result['top_class']} ({result['top_probability']:.2%})")
```

#### Using REST API

```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/predict
```

## 🎓 Model Development

### Supported Architectures

- **ResNet**: ResNet18, ResNet50
- **EfficientNet**: EfficientNet-B0
- **VGG**: VGG16

### Training Configuration

Edit `config.yaml` to customize training:

```yaml
model:
  architecture: "resnet50"
  pretrained: true
  num_classes: 10

training:
  epochs: 50
  learning_rate: 0.001
  batch_size: 32
  optimizer: "adam"
```

### Hyperparameter Tuning

The project supports various hyperparameter tuning approaches:

1. **Manual Tuning**: Modify `config.yaml`
2. **Grid Search**: Implement in `notebooks/02_model_training.ipynb`
3. **MLflow Tracking**: Automatic logging of all experiments

### Model Selection

Compare different models using the evaluation metrics:

```python
from src.models.evaluate import compare_models

# Load results from multiple models
results_list = [results_model1, results_model2, results_model3]
model_names = ['ResNet50', 'EfficientNet', 'VGG16']

compare_models(results_list, model_names)
```

## 🚢 Deployment

### Flask Deployment

#### Local Deployment

```bash
python flask_app.py
```

Access the web interface at `http://localhost:5000`

#### Production Deployment

Use Gunicorn for production:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 flask_app:app
```

### Docker Deployment

#### Build and Run

```bash
# Build image
docker build -t cv-model:latest .

# Run container
docker run -p 5000:5000 -v $(pwd)/models:/app/models cv-model:latest
```

#### Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Environment Variables

Set the following environment variables for deployment:

```bash
export FLASK_ENV=production
export MODEL_PATH=/app/models/saved_models/best_model.pth
export LOG_LEVEL=INFO
```

## ✅ Testing

### Run All Tests

```bash
pytest
```

### Run Specific Test Suites

```bash
# Data tests
pytest tests/test_data_loader.py

# Model tests
pytest tests/test_model_training.py

# API tests
pytest tests/test_flask_api.py
```

### Generate Coverage Report

```bash
pytest --cov=src --cov-report=html
```

View the report at `htmlcov/index.html`

### Test Configuration

Tests are configured in `pytest.ini`:

```ini
[pytest]
testpaths = tests
addopts = -v --strict-markers --cov=src
```

## 🔄 MLOps Pipeline

### Pipeline Components

1. **Data Preparation**: Load and preprocess data
2. **Model Training**: Train with MLflow tracking
3. **Model Evaluation**: Calculate metrics and generate reports
4. **Model Versioning**: Version and register models
5. **Model Monitoring**: Track production performance

### Running the Pipeline

```bash
# Full pipeline
python mlops_pipeline.py --mode full

# Training only
python mlops_pipeline.py --mode train

# Evaluation only
python mlops_pipeline.py --mode evaluate

# With custom config
python mlops_pipeline.py --config custom_config.yaml
```

### MLflow Tracking

View experiments in MLflow UI:

```bash
mlflow ui
```

Access at `http://localhost:5000`

### Model Versioning

Models are automatically versioned with timestamps:

```
models/saved_models/
├── 20260128_120000/
│   ├── model.pth
│   └── metadata.json
├── 20260128_150000/
│   ├── model.pth
│   └── metadata.json
└── best_model.pth
```

## 📚 API Documentation

### Endpoints

#### Health Check

```
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-01-28T12:00:00"
}
```

#### Model Info

```
GET /model/info
```

Response:
```json
{
  "model_loaded": true,
  "architecture": "resnet50",
  "num_classes": 10,
  "image_size": [224, 224]
}
```

#### Single Prediction

```
POST /predict
```

Parameters:
- `file`: Image file (multipart/form-data)
- `top_k`: Number of top predictions (optional, default: 5)

Response:
```json
{
  "success": true,
  "predictions": [
    {
      "class_name": "cat",
      "class_id": 3,
      "probability": 0.95
    }
  ],
  "top_prediction": {
    "class": "cat",
    "probability": 0.95
  }
}
```

#### Batch Prediction

```
POST /predict/batch
```

Parameters:
- `files`: Multiple image files (multipart/form-data)

Response:
```json
{
  "success": true,
  "count": 3,
  "results": [...]
}
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Write unit tests for new features
- Update documentation as needed
- Use meaningful commit messages

### Code Quality

Run code quality checks before committing:

```bash
# Format code
black src/

# Sort imports
isort src/

# Check style
flake8 src/

# Type checking (if using mypy)
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- Flask team for the web framework
- Roboflow for dataset management tools
- The open-source community for various libraries used

## 📞 Contact

**Muhammad Farooq**
- 📧 Email: mfarooqshafee333@gmail.com
- 🐙 GitHub: [@Muhammad-Farooq-13](https://github.com/Muhammad-Farooq-13)

For questions or support, please open an issue on GitHub.

## 🗺️ Roadmap

- [ ] Add support for object detection
- [ ] Implement model quantization for edge deployment
- [ ] Add Kubernetes deployment configuration
- [ ] Implement A/B testing framework
- [ ] Add model interpretability tools (GradCAM, SHAP)
- [ ] Create web-based annotation tool
- [ ] Add support for video inference
- [ ] Implement active learning pipeline

## 📈 Project Status

This project is actively maintained and under continuous development. Current version: 1.0.0

---

Made with ❤️ for the Computer Vision Community
