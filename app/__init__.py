import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app():
    app = Flask(__name__, instance_relative_config=True)#
    app.config.from_object("app.config.Config")

    # Ensure instance folder exists.
    os.makedirs(app.instance_path, exist_ok=True)

    db.init_app(app)

    from app.routes import api_bp
    app.register_blueprint(api_bp)

    with app.app_context():
        from app import models
        db.create_all()

    return app