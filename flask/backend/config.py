"""
Configuration file for Flask API
"""
import os

# Flask application settings
DEBUG = os.environ.get('FLASK_DEBUG', True)
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-for-testing-only')
JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'jwt-secret-key-for-testing-only')

# API settings
API_BASE_URL = os.environ.get('API_BASE_URL', 'http://127.0.0.1:5000')
API_VERSION = 'v1'

# CORS settings
CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*')

# Data paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.environ.get('DATA_DIR', os.path.join(PROJECT_ROOT, 'data'))
MODELS_DIR = os.environ.get('MODELS_DIR', os.path.join(PROJECT_ROOT, 'models'))

# Model settings
DEFAULT_MODEL_TYPE = 'collaborative'
DEFAULT_NUM_RECOMMENDATIONS = 5

# Security settings (for production)
PRODUCTION = os.environ.get('FLASK_ENV', '') == 'production'

# Frontend path (for serving React in production)
FRONTEND_BUILD_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend', 'build')
