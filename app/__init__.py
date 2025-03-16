# __init__.py
from flask import Flask
from app.routes import main as main_blueprint
from config.config import LOGGING_LEVEL
import logging

def create_app():
    # Ensure Flask knows where to find templates
    app = Flask(__name__, template_folder="templates")
    # Basic configuration
    app.config["LOGGING_LEVEL"] = LOGGING_LEVEL
    logging.basicConfig(level=LOGGING_LEVEL)
    
    # Import and register blueprints
    app.register_blueprint(main_blueprint)
    
    return app
