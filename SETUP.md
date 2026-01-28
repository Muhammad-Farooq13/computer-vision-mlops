# Setup Guide

This guide will help you set up the Computer Vision project on your local machine.

## Prerequisites

- **Python**: 3.9 or higher
- **pip**: Latest version
- **Git**: For cloning the repository
- **(Optional) CUDA**: For GPU acceleration
- **(Optional) Docker**: For containerized deployment

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Muhammad-Farooq-13/OPencv.git
cd OPencv
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

### 5. Configure the Project

Edit `config.yaml` to match your requirements:

```yaml
model:
  architecture: "resnet50"
  num_classes: 10  # Change based on your dataset

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001

data:
  raw_dir: "data/raw"
  image_size: [224, 224]
```

### 6. Prepare Your Dataset

Organize your data in the following structure:

```
data/raw/
├── train/
│   ├── _classes.txt
│   └── *.jpg
├── valid/
│   ├── _classes.txt
│   └── *.jpg
└── test/
    ├── _classes.txt
    └── *.jpg
```

### 7. Test the Setup

Run a quick test:

```bash
pytest tests/test_utils.py -v
```

## GPU Setup (Optional)

If you have an NVIDIA GPU:

1. Install CUDA Toolkit (11.7 or higher)
2. Install cuDNN
3. Verify GPU access:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Docker Setup (Optional)

### Build Image

```bash
docker build -t cv-model:latest .
```

### Run Container

```bash
docker-compose up -d
```

### Verify

```bash
docker-compose ps
docker-compose logs -f
```

Access the API at `http://localhost:5000`

## Troubleshooting

### Common Issues

**1. Import Error: No module named 'torch'**
```bash
pip install torch torchvision
```

**2. OpenCV Import Error**
```bash
pip install opencv-python
```

**3. Permission Denied (Linux/Mac)**
```bash
sudo chmod +x setup.sh
```

**4. CUDA Out of Memory**
- Reduce batch size in `config.yaml`
- Use smaller model architecture

**5. Port Already in Use**
```bash
# Change port in config.yaml
deployment:
  port: 5001
```

## Next Steps

1. **Explore Data**: Open `notebooks/01_data_exploration.ipynb`
2. **Train Model**: Run `python mlops_pipeline.py --mode train`
3. **Start API**: Run `python flask_app.py`
4. **Run Tests**: Run `pytest`

## Getting Help

- 📧 Email: mfarooqshafee333@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/Muhammad-Farooq-13/OPencv/issues)
- 📖 Documentation: See [README.md](README.md)

## Development Setup

For contributors:

```bash
# Install development dependencies
pip install black flake8 isort mypy pytest-cov

# Setup pre-commit hooks
pip install pre-commit
pre-commit install
```

## IDE Setup

### VS Code

Recommended extensions:
- Python (Microsoft)
- Pylance
- Python Docstring Generator
- Docker
- GitLens

### PyCharm

1. Open project in PyCharm
2. Configure Python interpreter (select venv)
3. Enable pytest as test runner
4. Configure code style (PEP 8)

---

For more information, see the [README.md](README.md) file.
