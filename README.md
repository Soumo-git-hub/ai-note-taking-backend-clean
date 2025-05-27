# AI Note Taking Backend

This is the backend service for the AI Note Taking application, built with FastAPI and Python.

## Local Development Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Initialize the database:
```bash
python init_database.py
```

4. Run the development server:
```bash
uvicorn app:app --reload
```

## Environment Variables

Create a `.env` file with the following variables:
```
SECRET_KEY=your_secret_key
DATABASE_URL=sqlite:///notes.db
UPLOAD_DIR=uploads
TEMP_DIR=temp
```

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Deployment

This application is configured for deployment on Render. The following files are used for deployment:
- `render.yaml`: Render service configuration
- `Procfile`: Process file for Render
- `requirements.txt`: Python dependencies
- `runtime.txt`: Python version specification

## Directory Structure

- `app.py`: Main FastAPI application
- `database.py`: Database models and operations
- `ai_service.py`: AI processing services
- `flutter_config.py`: Flutter-specific configurations
- `uploads/`: Directory for uploaded files
- `temp/`: Directory for temporary files

## Testing

Run the test suite:
```bash
python test_api.py
``` 