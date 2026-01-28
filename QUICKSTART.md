# Quick Start Guide

Get started with the Computer Vision project in 5 minutes!

## 🚀 Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/Muhammad-Farooq-13/OPencv.git
cd OPencv

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch, cv2; print('Setup complete!')"
```

## 📊 Quick Data Exploration

```bash
# Open Jupyter notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

## 🎯 Train Your First Model

```bash
# Quick training with default settings
python mlops_pipeline.py --mode train
```

## 🌐 Launch Web API

```bash
# Start Flask server
python flask_app.py

# Open browser
# Visit: http://localhost:5000
```

## 🐳 Docker Quick Start

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Access API at http://localhost:5000
```

## 📝 Make a Prediction

### Using Python

```python
from src.models.predict import ModelInference
from src.models.train import CVModel
import cv2

# Load model
model = CVModel(model_name='resnet50', num_classes=10)
inference = ModelInference(model, model_path='models/saved_models/best_model.pth')

# Predict
image = cv2.imread('path/to/image.jpg')
result = inference.predict(image)
print(f"Prediction: {result['top_class']}")
```

### Using API

```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/predict
```

## 🧪 Run Tests

```bash
pytest
```

## 📚 Next Steps

1. **Customize Configuration**: Edit `config.yaml`
2. **Prepare Your Data**: Place in `data/raw/`
3. **Experiment**: Use Jupyter notebooks
4. **Deploy**: Use Docker or Flask

## 🆘 Quick Troubleshooting

**GPU not detected?**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Port already in use?**
```bash
# Change port in config.yaml
deployment:
  port: 5001
```

**Module not found?**
```bash
pip install -r requirements.txt --upgrade
```

## 📞 Need Help?

- 📧 Email: mfarooqshafee333@gmail.com
- 🐛 [Report Issues](https://github.com/Muhammad-Farooq-13/OPencv/issues)
- 📖 [Full Documentation](README.md)

---

**Happy Coding! 🎉**
