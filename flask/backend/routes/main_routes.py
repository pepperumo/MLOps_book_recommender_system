"""
Main routes for Flask web application (non-API)
"""
from flask import Blueprint, render_template, jsonify, request, redirect, url_for
from flask import current_app as app
import os
import logging

logger = logging.getLogger(__name__)

# Create Blueprint
main_bp = Blueprint('main', __name__)

@main_bp.route('/health')
def health_check():
    """
    Health check endpoint for monitoring
    """
    return jsonify({
        "status": "ok",
        "service": "book-recommender-api",
        "version": app.config.get('API_VERSION')
    })

@main_bp.route('/docs')
def api_docs():
    """
    API Documentation page (Swagger UI)
    """
    return render_template('swagger_ui.html')
