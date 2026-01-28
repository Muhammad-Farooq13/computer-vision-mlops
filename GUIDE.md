# 🎓 Computer Vision Master Guide

Complete documentation for the Computer Vision project by Muhammad Farooq.

## 📑 Table of Contents

1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Setup](#setup)
4. [Usage](#usage)
5. [API Reference](#api-reference)
6. [Development](#development)
7. [Deployment](#deployment)
8. [FAQ](#faq)

## Introduction

This guide provides comprehensive information about the Computer Vision project structure, usage, and best practices.

### Project Goals

- Build production-ready computer vision models
- Implement MLOps best practices
- Enable easy deployment and scaling
- Maintain high code quality and testing

### Technology Stack

- **Deep Learning**: PyTorch, TorchVision
- **Computer Vision**: OpenCV, Albumentations
- **Web Framework**: Flask
- **MLOps**: MLflow, DVC
- **Testing**: Pytest
- **Deployment**: Docker, Docker Compose
- **CI/CD**: GitHub Actions

## Architecture

### System Architecture

```
┌─────────────────┐
│   Data Source   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Processing │
│  & Augmentation │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Training  │
│   (PyTorch)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Evaluation    │
│   & Metrics     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Registry  │
│    (MLflow)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Flask API     │
│   (REST API)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Client       │
│ (Web/Mobile)    │
└─────────────────┘
```

### Code Structure

```
src/
├── data/           # Data handling
├── features/       # Feature engineering
├── models/         # Model training & inference
├── utils/          # Utilities
└── visualization/  # Plotting tools
```

## Setup

### Prerequisites

```bash
# Check Python version (3.9+)
python --version

# Check pip
pip --version

# Check Git
git --version
```

### Installation

```bash
# Clone repository
git clone https://github.com/Muhammad-Farooq-13/OPencv.git
cd OPencv

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, cv2; print('Setup successful!')"
```

## Usage

### Training a Model

```python
from src.models.train import CVModel, ModelTrainer
from src.utils.config import Config

# Load configuration
config = Config('config.yaml')

# Initialize model
model = CVModel(
    model_name='resnet50',
    num_classes=10,
    pretrained=True
)

# Train
trainer = ModelTrainer(model)
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    lr=0.001
)
```

### Making Predictions

```python
from src.models.predict import ModelInference
import cv2

# Initialize inference
inference = ModelInference(
    model,
    model_path='models/saved_models/best_model.pth'
)

# Load and predict
image = cv2.imread('image.jpg')
result = inference.predict(image, top_k=5)

print(f"Prediction: {result['top_class']}")
print(f"Confidence: {result['top_probability']:.2%}")
```

### Using the API

```bash
# Start server
python flask_app.py

# Make prediction
curl -X POST \
  -F "file=@image.jpg" \
  http://localhost:5000/predict

# Health check
curl http://localhost:5000/health
```

## API Reference

### Endpoints

#### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-01-28T12:00:00"
}
```

#### POST /predict

Single image prediction.

**Parameters:**
- `file`: Image file (multipart/form-data)
- `top_k`: Number of predictions (optional, default: 5)

**Response:**
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

#### GET /model/info

Get model information.

**Response:**
```json
{
  "model_loaded": true,
  "architecture": "resnet50",
  "num_classes": 10,
  "image_size": [224, 224]
}
```

## Development

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_model_training.py

# With coverage
pytest --cov=src --cov-report=html

# Verbose output
pytest -v
```

### Code Quality

```bash
# Format code
black src/

# Sort imports
isort src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

### Adding New Features

1. Create feature branch
2. Implement feature with tests
3. Update documentation
4. Submit pull request

## Deployment

### Docker Deployment

```bash
# Build image
docker build -t cv-model:latest .

# Run container
docker run -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  cv-model:latest

# Using docker-compose
docker-compose up -d
```

### Production Deployment

```bash
# Using Gunicorn
gunicorn -w 4 \
  -b 0.0.0.0:5000 \
  --timeout 120 \
  flask_app:app

# With logging
gunicorn -w 4 \
  -b 0.0.0.0:5000 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log \
  flask_app:app
```

### Environment Variables

```bash
export FLASK_ENV=production
export MODEL_PATH=/app/models/saved_models/best_model.pth
export LOG_LEVEL=INFO
export MAX_CONTENT_LENGTH=16777216
```

## FAQ

### General Questions

**Q: Which model architecture should I use?**
A: ResNet50 is a good starting point. For smaller models, try ResNet18. For better accuracy, try EfficientNet-B0.

**Q: How do I add more classes?**
A: Update `num_classes` in config.yaml and retrain the model with your new dataset.

**Q: Can I use this for object detection?**
A: Currently, this project focuses on classification. Object detection is planned for future releases.

### Training Questions

**Q: How much data do I need?**
A: Minimum 100 images per class. More is better. Consider data augmentation for small datasets.

**Q: Training is too slow. What can I do?**
A: Use a GPU, reduce batch size, or use a smaller model architecture.

**Q: How do I prevent overfitting?**
A: Use data augmentation, dropout, early stopping, and regularization.

### Deployment Questions

**Q: Can I deploy this on cloud platforms?**
A: Yes! The Docker container can be deployed on AWS, Azure, GCP, or any cloud platform.

**Q: How do I scale the API?**
A: Use multiple Gunicorn workers, load balancers, or Kubernetes for horizontal scaling.

**Q: Is the API production-ready?**
A: Yes, but ensure proper security measures (HTTPS, authentication, rate limiting) for production use.

### Troubleshooting

**Q: CUDA out of memory error**
A: Reduce batch size, use gradient accumulation, or use a smaller model.

**Q: Module not found error**
A: Ensure virtual environment is activated and dependencies are installed.

**Q: Port already in use**
A: Change the port in config.yaml or kill the process using the port.

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Docker Documentation](https://docs.docker.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

## Support

For issues and questions:
- 📧 Email: mfarooqshafee333@gmail.com
- 🐛 GitHub Issues: [Create an issue](https://github.com/Muhammad-Farooq-13/OPencv/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/Muhammad-Farooq-13/OPencv/discussions)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

**Maintained by:** Muhammad Farooq  
**Last Updated:** January 28, 2026  
**Version:** 1.0.0
