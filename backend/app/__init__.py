import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager
from flask_cors import CORS
from config import config

# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()
jwt = JWTManager()
cors = CORS()

def create_app(config_name="development"):
    """
    Application factory function to create and configure the Flask app
    """
    # Get the absolute path to the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create app with static folder at project root level
    app = Flask(__name__, 
                static_folder=os.path.join(project_root, '..', 'static'),
                static_url_path='/static')
    
    # Load configuration
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)
    
    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)
    cors.init_app(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
    
    # Create directories if not exist
    if not os.path.exists('logs'):
        os.mkdir('logs')
    if not os.path.exists('uploads'):
        os.mkdir('uploads')
    
    # Setup logging
    file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Vehicle Load Management startup')
    
    # Create static folders if they don't exist
    static_folder = app.static_folder
    for folder in ['graphs', 'images', 'documents']:
        folder_path = os.path.join(static_folder, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
            app.logger.info(f'Created static folder: {folder_path}')
    
    # Register error handlers
    from app.utils.error_handlers import register_error_handlers
    register_error_handlers(app)
    
    # Register blueprints
    from app.api.auth import auth_bp
    from app.api.vehicles import vehicles_bp
    from app.api.predictions import predictions_bp
    
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(vehicles_bp, url_prefix='/api/vehicles')
    app.register_blueprint(predictions_bp, url_prefix='/api/predictions')
    
    # Register a simple route for testing
    @app.route('/health')
    def health_check():
        return {'status': 'ok', 'message': 'Server is running'}
    
    # Add a root route for basic API info
    @app.route('/')
    def index():
        return jsonify({
            'status': 'success',
            'message': 'Vehicle Load Management API',
            'version': app.config.get('API_VERSION', 'v1'),
            'endpoints': {
                'auth': '/api/auth',
                'vehicles': '/api/vehicles',
                'predictions': '/api/predictions',
                'health': '/health'
            }
        })
    
    return app 