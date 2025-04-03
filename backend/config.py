import os
from datetime import timedelta

basedir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.dirname(os.path.abspath(__file__))

class Config:
    """Base config."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev_key_replace_in_production')
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'jwt_dev_key_replace_in_production')
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    
    # Use SQLite as the database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///' + os.path.join(basedir, 'app.db'))
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Logging configuration
    LOG_FOLDER = os.path.join(basedir, 'logs')
    
    # Uploads configuration
    UPLOAD_FOLDER = os.path.join(basedir, 'uploads')
    # Static graphs folder - this should match the Flask static_folder config
    STATIC_FOLDER = os.path.join(project_root, 'static')
    GRAPHS_FOLDER = os.path.join(STATIC_FOLDER, 'graphs')
    
    # Maximum content length for requests
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
    
    @staticmethod
    def init_app(app):
        # Create necessary directories
        os.makedirs(app.static_folder, exist_ok=True)
        os.makedirs(os.path.join(app.static_folder, 'graphs'), exist_ok=True)

class DevelopmentConfig(Config):
    """Development config."""
    DEBUG = True
    SQLALCHEMY_ECHO = True

class TestingConfig(Config):
    """Testing config."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

class ProductionConfig(Config):
    """Production config."""
    DEBUG = False
    
    # Override with environment variables in production
    SECRET_KEY = os.environ.get('SECRET_KEY', 'production_key')
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'jwt_production_key')
    
    # Use PostgreSQL in production (if environment variable is set)
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///' + os.path.join(basedir, 'app.db'))

config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
} 