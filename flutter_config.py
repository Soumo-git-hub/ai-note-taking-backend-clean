import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Upload directories
UPLOAD_DIR = BASE_DIR / "uploads"
TEMP_DIR = BASE_DIR / "temp"
NLTK_DATA_DIR = BASE_DIR / "nltk_data"

# Create directories if they don't exist
for directory in [UPLOAD_DIR, TEMP_DIR, NLTK_DATA_DIR]:
    directory.mkdir(exist_ok=True)

# File size limits (in bytes)
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
MAX_PDF_SIZE = 20 * 1024 * 1024     # 20MB
MAX_IMAGE_SIZE = 5 * 1024 * 1024    # 5MB

# Allowed file types
ALLOWED_IMAGE_TYPES = {'.png', '.jpg', '.jpeg'}
ALLOWED_PDF_TYPES = {'.pdf'}

# API configuration
API_CONFIG = {
    'title': 'AI Note Taking App API',
    'description': 'Backend API for AI Note Taking App with Flutter support',
    'version': '2.0.0',
    'docs_url': '/api/docs',
    'redoc_url': '/api/redoc',
    'openapi_url': '/api/openapi.json',
}

# CORS configuration
CORS_ORIGINS = [
    "http://localhost:3000",    # Flutter web development
    "http://localhost:8000",    # Flutter web production
    "http://127.0.0.1:3000",    # Flutter web development
    "http://127.0.0.1:8000",    # Flutter web production
    "http://localhost",         # Android emulator
    "http://10.0.2.2:8000",    # Android emulator localhost
    "capacitor://localhost",    # Capacitor
    "ionic://localhost",        # Ionic
    "http://localhost:8080",    # Alternative port
    "http://localhost:5000",    # Alternative port
    "http://localhost:53330",   # Flutter web debug
    "http://127.0.0.1:53330",   # Flutter web debug
    "*"                         # Allow all origins in development
]

# Database configuration
DATABASE_URL = f"sqlite:///{BASE_DIR}/notes.db"

# AI Service configuration
AI_SERVICE_CONFIG = {
    'model': "mistralai/mistral-7b-instruct:free",
    'api_key': os.getenv("OPENROUTER_API_KEY", "YOUR_API_KEY"),
    'max_retries': 3,
    'retry_delay': 5,
    'timeout': 60,
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': BASE_DIR / 'app.log',
            'formatter': 'default',
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file'],
    },
} 