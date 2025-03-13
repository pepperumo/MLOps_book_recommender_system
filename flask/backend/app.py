"""
Main Flask application file for Book Recommender System
"""
import os
import sys
import logging
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_jwt_extended import JWTManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

# Create Flask app
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Load configuration
app.config.from_pyfile('config.py')

# Configure CORS
CORS(app, resources={r"/*": {"origins": app.config.get('CORS_ORIGINS')}})

# Configure JWT
jwt = JWTManager(app)

# Import routes after app is created to avoid circular imports
from routes.main_routes import main_bp
from routes.api_routes import api_bp

# Register blueprints
app.register_blueprint(main_bp)
app.register_blueprint(api_bp, url_prefix='/api')

# Serve React app in production
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    if app.config.get('PRODUCTION'):
        if path and os.path.exists(os.path.join(app.config.get('FRONTEND_BUILD_PATH'), path)):
            return send_from_directory(app.config.get('FRONTEND_BUILD_PATH'), path)
        return send_from_directory(app.config.get('FRONTEND_BUILD_PATH'), 'index.html')
    else:
        return jsonify({"status": "development", "message": "Flask API is running in development mode"})

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

# Run the app if executed directly
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = app.config.get('DEBUG', False)
    app.run(host='0.0.0.0', port=port, debug=debug)
