"""
config.py

Configuration settings for the Explainable AI Maturity Assessment System.

This module centralizes:
- base directory paths
- SQLite database configuration
- Flask secret key
- OpenAI API key

Keeping configuration in one place makes the application easier to
maintain, debug, and explain in a project presentation.
"""

import os


class Config:
    """
    Main application configuration class.
    """

    # Base project directory.
    BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

    # Instance folder used for runtime data such as SQLite database files.
    INSTANCE_DIR = os.path.join(BASE_DIR, "instance")

    # SQLite database path.
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.join(INSTANCE_DIR, 'app.db')}"

    # Disable SQLAlchemy event system overhead.
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Flask secret key used for sessions / security-related features.
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")

    # OpenAI API key loaded from environment variables.
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")