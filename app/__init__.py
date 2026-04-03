"""
__init__.py

Application factory for the Explainable AI Maturity Assessment System.

This module:
- loads environment variables
- initializes the Flask app
- loads configuration
- initializes the SQLAlchemy database
- registers API/frontend routes via Blueprint
- creates database tables on startup

The app factory pattern keeps the project modular and scalable.
"""

import os

from dotenv import load_dotenv
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

load_dotenv()

# Global SQLAlchemy instance.
# It is initialized later inside create_app().
db = SQLAlchemy()


def create_app() -> Flask:
    """
    Create and configure the Flask application instance.

    Flow:
    1. create Flask app
    2. load configuration
    3. ensure instance folder exists
    4. initialize SQLAlchemy
    5. register blueprints
    6. create database tables

    Returns:
        Configured Flask app instance.
    """
    app = Flask(__name__, instance_relative_config=True)

    # Load settings from app/config.py
    app.config.from_object("app.config.Config")

    # Ensure the instance folder exists.
    # This is where the SQLite DB file will live.
    os.makedirs(app.instance_path, exist_ok=True)

    # Bind SQLAlchemy to this Flask app.
    db.init_app(app)

    # Register routes / endpoints from routes.py.
    from app.routes import api_bp
    app.register_blueprint(api_bp)

    # Create database tables if they do not yet exist.
    with app.app_context():
        from app import models
        db.create_all()

    return app