"""
Flask Application for Computer Vision Model Deployment
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
import cv2
import numpy as np
import torch
import os
from pathlib import Path
from werkzeug.utils import secure_filename
import logging
from datetime import datetime

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent))

from src.models.predict import ModelInference
from src.models.train import CVModel
from src.utils.config import Config
from src.utils.logger import setup_logger

# Initialize Flask app
app = Flask(__name__, 
           template_folder='api/templates',
           static_folder='api/static')

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'api/static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}

# Setup logger
logger = setup_logger('flask_app', 'logs')

# Load configuration
config = Config()

# Initialize model (will be loaded on first request)
inference_engine = None
model_loaded = False


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_model():
    """Load the trained model"""
    global inference_engine, model_loaded
    
    try:
        # Load model architecture
        num_classes = config.get('model.num_classes', 10)
        model_arch = config.get('model.architecture', 'resnet50')
        
        model = CVModel(model_name=model_arch, num_classes=num_classes, pretrained=False)
        
        # Load model weights
        model_path = Path(config.get('paths.model_save_dir', 'models/saved_models')) / 'best_model.pth'
        
        if not model_path.exists():
            logger.warning(f"Model not found at {model_path}. Using pretrained weights.")
            model = CVModel(model_name=model_arch, num_classes=num_classes, pretrained=True)
            inference_engine = ModelInference(model)
        else:
            inference_engine = ModelInference(model, model_path=str(model_path))
            
        model_loaded = True
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model_loaded = False


@app.before_request
def before_first_request():
    """Initialize model before first request"""
    global model_loaded
    if not model_loaded:
        load_model()


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Predict endpoint for image classification"""
    try:
        # Check if model is loaded
        if not model_loaded or inference_engine is None:
            return jsonify({
                'error': 'Model not loaded',
                'success': False
            }), 500
            
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'success': False
            }), 400
            
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'success': False
            }), 400
            
        # Check if file is allowed
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'File type not allowed. Allowed types: {app.config["ALLOWED_EXTENSIONS"]}',
                'success': False
            }), 400
            
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        
        upload_dir = Path(app.config['UPLOAD_FOLDER'])
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = upload_dir / filename
        file.save(str(filepath))
        
        # Read image
        image = cv2.imread(str(filepath))
        
        if image is None:
            return jsonify({
                'error': 'Could not read image',
                'success': False
            }), 400
            
        # Make prediction
        top_k = request.form.get('top_k', 5, type=int)
        prediction = inference_engine.predict(image, top_k=top_k)
        
        # Create response
        response = {
            'success': True,
            'filename': filename,
            'predictions': prediction['predictions'],
            'top_prediction': {
                'class': prediction['top_class'],
                'probability': prediction['top_probability']
            },
            'image_url': f'/static/uploads/{filename}'
        }
        
        logger.info(f"Prediction made for {filename}: {prediction['top_class']}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    try:
        if not model_loaded or inference_engine is None:
            return jsonify({
                'error': 'Model not loaded',
                'success': False
            }), 500
            
        files = request.files.getlist('files')
        
        if not files:
            return jsonify({
                'error': 'No files provided',
                'success': False
            }), 400
            
        results = []
        
        for file in files:
            if file and allowed_file(file.filename):
                # Save and process file
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{filename}"
                
                upload_dir = Path(app.config['UPLOAD_FOLDER'])
                upload_dir.mkdir(parents=True, exist_ok=True)
                
                filepath = upload_dir / filename
                file.save(str(filepath))
                
                # Read and predict
                image = cv2.imread(str(filepath))
                
                if image is not None:
                    prediction = inference_engine.predict(image)
                    
                    results.append({
                        'filename': filename,
                        'top_prediction': {
                            'class': prediction['top_class'],
                            'probability': prediction['top_probability']
                        }
                    })
                    
        return jsonify({
            'success': True,
            'count': len(results),
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        info = {
            'model_loaded': model_loaded,
            'architecture': config.get('model.architecture', 'unknown'),
            'num_classes': config.get('model.num_classes', 'unknown'),
            'image_size': config.get('data.image_size', [224, 224])
        }
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'error': 'File too large. Maximum size is 16MB',
        'success': False
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'success': False
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'error': 'Internal server error',
        'success': False
    }), 500


if __name__ == '__main__':
    # Create necessary directories
    Path('api/static/uploads').mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(parents=True, exist_ok=True)
    
    # Run app
    host = config.get('deployment.host', '0.0.0.0')
    port = config.get('deployment.port', 5000)
    debug = config.get('deployment.debug', False)
    
    logger.info(f"Starting Flask app on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
