import os

class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    INSTANCE_DIR = os.path.join(BASE_DIR, "instance")

    SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.join(INSTANCE_DIR, 'app.db')}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")