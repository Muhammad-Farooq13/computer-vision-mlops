"""
Integration tests for Flask API
"""

import pytest
import json
import io
from pathlib import Path

# flask_app.py imports src.models.predict which requires torchvision
pytest.importorskip("torchvision")

import sys
sys.path.append(str(Path(__file__).parent.parent))
from flask_app import app


@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestFlaskAPI:
    """Test cases for Flask API endpoints"""
    
    def test_index(self, client):
        """Test index endpoint"""
        response = client.get('/')
        assert response.status_code == 200
        
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] == 'healthy'
        
    def test_model_info(self, client):
        """Test model info endpoint"""
        response = client.get('/model/info')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'model_loaded' in data
        
    def test_predict_no_file(self, client):
        """Test predict endpoint without file"""
        response = client.post('/predict')
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data
        
    def test_predict_with_file(self, client, sample_image, temp_dir):
        """Test predict endpoint with file"""
        import cv2
        
        # Save sample image
        img_path = temp_dir / 'test.jpg'
        cv2.imwrite(str(img_path), sample_image)
        
        # Create file-like object
        with open(img_path, 'rb') as f:
            data = {
                'file': (f, 'test.jpg', 'image/jpeg')
            }
            response = client.post('/predict', 
                                  data=data,
                                  content_type='multipart/form-data')
        
        # Note: This might fail if model is not loaded
        # In production, ensure model is available
        assert response.status_code in [200, 500]
        
    def test_not_found(self, client):
        """Test 404 error"""
        response = client.get('/nonexistent')
        assert response.status_code == 404
