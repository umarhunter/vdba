# __init__.py
import os
import logging
#from flask_ngrok import run_with_ngrok 
from flask import Flask
from app.routes import main as main_blueprint
from config.config import LOGGING_LEVEL

def create_app():
    # Ensure Flask knows where to find templates
    app = Flask(__name__, template_folder="templates")
    app.config['DEBUG'] = True
    app.config['PROPAGATE_EXCEPTIONS'] = True
    
    app.secret_key = os.urandom(24)  # or use a fixed string like 'your-secret-key'
    #run_with_ngrok(app) 
    # Basic configuration
    app.config["LOGGING_LEVEL"] = LOGGING_LEVEL
    logging.basicConfig(level=LOGGING_LEVEL)
    
    # Import and register blueprints
    app.register_blueprint(main_blueprint)
    
    return app
