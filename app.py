from fastapi import FastAPI, HTTPException, Body, Depends, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
from pathlib import Path
import uvicorn
import logging
from typing import List, Dict, Optional, Any, Union
import json
import requests
import time
from functools import lru_cache
import asyncio
import httpx
from contextlib import asynccontextmanager
from ai_service import get_ai_service, AIService
from PyPDF2 import PdfReader
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract
import logging.config
import jwt
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordBearer
import markdown
import re

# Import Flutter configuration
from flutter_config import (
    API_CONFIG,
    CORS_ORIGINS,
    LOGGING_CONFIG,
    UPLOAD_DIR,
    TEMP_DIR,
    MAX_UPLOAD_SIZE,
    MAX_PDF_SIZE,
    MAX_IMAGE_SIZE,
    ALLOWED_IMAGE_TYPES,
    ALLOWED_PDF_TYPES
)

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Get environment variables
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
PORT = int(os.getenv("PORT", 8000))

# Application startup and shutdown events
# Import the init_db function from database module
from database import (
    init_db,
    get_all_notes,
    get_note_by_id,
    save_note,
    update_note,
    delete_note,
    get_user_by_email,
    create_user,
    verify_user
)

# JWT Configuration
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: Dict[str, Any]) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing database...")
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.critical(f"Database initialization failed: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")

    logger.info("Shutting down application...")

# Create FastAPI app with lifespan
app = FastAPI(
    **API_CONFIG,
    lifespan=lifespan
)

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

# Mount the uploads directory for serving audio files with proper CORS headers
app.mount("/uploads", StaticFiles(directory="uploads", html=True), name="uploads")

# Add CORS middleware with more specific configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Length", "Content-Type", "Content-Disposition", "Accept-Ranges"],
)

# Status endpoint for health checks
@app.get("/api/status")
async def get_status():
    return {"status": "ok", "version": API_CONFIG['version']}

# Add error handling middleware
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Please try again later."}
        )

# Define the path to the frontend directory
frontend_dir = Path(__file__).parent.parent / "frontend"
if not frontend_dir.exists():
    logger.warning(f"Frontend directory not found at {frontend_dir}. Static files may not be served correctly.")

# Mount the static files
try:
    for static_dir in ["css", "js", "img"]:
        static_path = frontend_dir / static_dir
        if static_path.exists():
            app.mount(f"/{static_dir}", StaticFiles(directory=str(static_path)), name=static_dir)
        else:
            logger.warning(f"Static directory {static_dir} not found at {static_path}")
except Exception as e:
    logger.error(f"Failed to mount static directories: {str(e)}")

# Serve the main index.html file at the root
@app.get("/", response_class=FileResponse)
async def read_index():
    index_path = frontend_dir / "index.html"
    if not index_path.exists():
        logger.error(f"Index file not found at {index_path}")
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(str(index_path))

# Define data models
class NoteContent(BaseModel):
    content: str = Field(..., min_length=1, description="The content of the note")
    is_markdown: bool = Field(False, description="Whether the content is in Markdown format")

class NoteCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="The title of the note")
    content: str = Field(..., min_length=1, description="The content of the note")
    is_markdown: bool = Field(False, description="Whether the content is in Markdown format")
    summary: Optional[str] = Field(None, description="The summary of the note")
    quiz: Optional[str] = Field(None, description="Quiz related to the note content")
    mindmap: Optional[str] = Field(None, description="Mind map of the note content")

class NoteUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200, description="The updated title of the note")
    content: Optional[str] = Field(None, min_length=1, description="The updated content of the note")
    is_markdown: Optional[bool] = Field(None, description="Whether the content is in Markdown format")
    summary: Optional[str] = Field(None, description="The updated summary of the note")
    quiz: Optional[str] = Field(None, description="Updated quiz related to the note content")
    mindmap: Optional[str] = Field(None, description="Updated mind map of the note content")

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="The username")
    email: str = Field(..., description="The email address")
    password: str = Field(..., min_length=6, description="The password")

class UserLogin(BaseModel):
    email: str = Field(..., description="The email address")
    password: str = Field(..., description="The password")

class Token(BaseModel):
    access_token: str
    token_type: str

# Add OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

class User(BaseModel):
    email: str
    username: str

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except jwt.InvalidTokenError:
        raise credentials_exception
        
    user = get_user_by_email(email)
    if user is None:
        raise credentials_exception
        
    return User(email=user["email"], username=user["username"])

# API Configuration class
class APIConfig:
    def __init__(self):
        self.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not self.huggingface_api_key:
            raise ValueError("HUGGINGFACE_API_KEY environment variable is not set")
        self.huggingface_api_url = "https://api-inference.huggingface.co/models/"
        self.model = "mistralai/Mistral-7B-Instruct-v0.3"
        self.max_retries = 3
        self.retry_delay = 5

        logger.info(f"API Key present: {bool(self.huggingface_api_key)}")
        logger.info(f"Using model: {self.model}")

# Create API config singleton
@lru_cache()
def get_api_config():
    return APIConfig()

# API endpoints for notes
@app.get("/api/notes", response_model=Dict[str, List[Dict[str, Any]]])
async def api_get_notes(current_user: User = Depends(get_current_user)):
    try:
        notes = get_all_notes()
        return {"notes": notes}
    except Exception as e:
        logger.error(f"Failed to retrieve notes: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve notes")

@app.get("/api/notes/{note_id}", response_model=Dict[str, Any])
async def api_get_note(note_id: int):
    try:
        note = get_note_by_id(note_id)
        if not note:
            raise HTTPException(status_code=404, detail=f"Note with ID {note_id} not found")
        return note
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve note {note_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve note {note_id}")

@app.post("/api/notes", response_model=Dict[str, Union[int, str]])
async def api_create_note(note: NoteCreate):
    try:
        logger.info(f"Creating new note with title: {note.title}")
        note_id = save_note(
            title=note.title,
            content=note.content,
            is_markdown=note.is_markdown,
            summary=note.summary,
            quiz=note.quiz,
            mindmap=note.mindmap
        )
        if note_id == -1:
            logger.error("Failed to create note - database error")
            raise HTTPException(status_code=500, detail="Failed to create note")
        logger.info(f"Note created successfully with ID: {note_id}")
        return {"id": note_id, "message": "Note created successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create note: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create note")

@app.put("/api/notes/{note_id}")
async def api_update_note(
    note_id: int,
    note: NoteUpdate,
    current_user: User = Depends(get_current_user)
):
    try:
        # First check if the note exists
        existing_note = get_note_by_id(note_id)
        if not existing_note:
            raise HTTPException(status_code=404, detail="Note not found")

        # Prepare update data
        update_data = {
            'title': note.title,
            'content': note.content,
            'is_markdown': note.is_markdown,
            'summary': note.summary,
            'quiz': note.quiz,
            'mindmap': note.mindmap,
            'updated_at': datetime.utcnow().isoformat()
        }
        
        # Remove None values
        update_data = {k: v for k, v in update_data.items() if v is not None}
        
        # Update the note
        success = update_note(note_id, update_data)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update note")
            
        # Get the updated note
        updated_note = get_note_by_id(note_id)
        if not updated_note:
            raise HTTPException(status_code=500, detail="Failed to retrieve updated note")
            
        return updated_note
    except Exception as e:
        logger.error(f"Error updating note: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/notes/{note_id}")
async def api_delete_note(
    note_id: int,
    current_user: User = Depends(get_current_user)
):
    try:
        # First check if the note exists
        existing_note = get_note_by_id(note_id)
        if not existing_note:
            raise HTTPException(status_code=404, detail=f"Note with ID {note_id} not found")

        # Attempt to delete the note
        success = delete_note(note_id)
        if not success:
            logger.error(f"Failed to delete note {note_id}")
            raise HTTPException(status_code=500, detail=f"Failed to delete note {note_id}")
            
        logger.info(f"Note {note_id} deleted successfully")
        return {"message": "Note deleted successfully", "note_id": note_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting note {note_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting note: {str(e)}")

# File upload endpoints
@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.lower().endswith(tuple(ALLOWED_PDF_TYPES)):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        # Read file content
        content = await file.read()
        
        # Check file size
        if len(content) > MAX_PDF_SIZE:
            raise HTTPException(status_code=400, detail=f"File size exceeds maximum limit of {MAX_PDF_SIZE/1024/1024}MB")

        # Save file with secure filename
        file_path = UPLOAD_DIR / secure_filename(file.filename)
        with open(file_path, "wb") as f:
            f.write(content)

        try:
            # Extract text from PDF
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"

            # Schedule file deletion
            asyncio.create_task(delete_file_after_delay(file_path))

            return {
                "message": "PDF processed successfully",
                "text": text,
                "pages": len(reader.pages)
            }
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise HTTPException(status_code=500, detail="Error processing PDF file")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error handling PDF upload: {str(e)}")
        raise HTTPException(status_code=500, detail="Error handling PDF upload")

@app.post("/api/handwriting")
async def process_handwriting(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.lower().endswith(tuple(ALLOWED_IMAGE_TYPES)):
            raise HTTPException(status_code=400, detail="Only PNG and JPG files are allowed")

        # Read file content
        content = await file.read()
        
        # Check file size
        if len(content) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=400, detail=f"File size exceeds maximum limit of {MAX_IMAGE_SIZE/1024/1024}MB")

        # Save file with secure filename
        file_path = UPLOAD_DIR / secure_filename(file.filename)
        with open(file_path, "wb") as f:
            f.write(content)
        
        try:
            # Process image with OCR
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            
            # Schedule file deletion
            asyncio.create_task(delete_file_after_delay(file_path))
            
            return {
                "message": "Handwriting processed successfully",
                "text": text
            }
        except Exception as e:
            logger.error(f"Error processing handwriting: {str(e)}")
            raise HTTPException(status_code=500, detail="Error processing handwriting image")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error handling handwriting upload: {str(e)}")
        raise HTTPException(status_code=500, detail="Error handling handwriting upload")

async def delete_file_after_delay(file_path: Path, delay: int = 600):
    await asyncio.sleep(delay)
    try:
        if file_path.exists():
            file_path.unlink()
        logger.info(f"Deleted temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {str(e)}")

# AI feature endpoints
@app.post("/api/summarize", response_model=Dict[str, str])
async def generate_summary(note: NoteContent):
    try:
        ai_service = get_ai_service()
        summary = ai_service.summarize_text(note.content)
        return {"summary": summary}
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating summary")

@app.post("/api/generate-quiz", response_model=Dict[str, Dict[str, List[Dict[str, Any]]]])
async def generate_quiz(note: NoteContent):
    try:
        ai_service = get_ai_service()
        quiz = ai_service.generate_quiz(note.content)
        return {"quiz": quiz}
    except Exception as e:
        logger.error(f"Error generating quiz: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating quiz")

@app.post("/api/mindmap", response_model=Dict[str, Dict[str, Any]])
async def generate_mindmap(note: NoteContent):
    try:
        ai_service = get_ai_service()
        mindmap = ai_service.generate_mindmap(note.content)
        return {"mindmap": mindmap}
    except Exception as e:
        logger.error(f"Error generating mindmap: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating mindmap")

@app.post("/api/text-to-speech", response_model=Dict[str, str])
async def text_to_speech(note: NoteContent):
    try:
        ai_service = get_ai_service()
        audio_url = ai_service.text_to_speech(note.content)
        
        # Verify the audio file exists
        audio_path = os.path.join("uploads", os.path.basename(audio_url))
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found at path: {audio_path}")
            raise HTTPException(status_code=500, detail="Audio file not found after generation")
            
        # Verify the file has content
        if os.path.getsize(audio_path) == 0:
            logger.error(f"Generated audio file is empty: {audio_path}")
            raise HTTPException(status_code=500, detail="Generated audio file is empty")
            
        logger.info(f"Successfully generated and verified audio file: {audio_path}")
        return {"audio_url": audio_url}
        
    except Exception as e:
        logger.error(f"Error converting text to speech: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Authentication endpoints
@app.post("/api/auth/register", response_model=Dict[str, Union[bool, str]])
async def register(user: UserCreate):
    try:
        # Check if user already exists
        existing_user = get_user_by_email(user.email)
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Create new user
        user_id = create_user(
            username=user.username,
            email=user.email,
            password=user.password  # Password will be hashed in create_user
        )
        
        if user_id == -1:
            raise HTTPException(status_code=500, detail="Failed to create user")
        
        return {"success": True, "message": "User registered successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering user: {str(e)}")
        raise HTTPException(status_code=500, detail="Error registering user")

@app.post("/api/auth/login", response_model=Token)
async def login(user: UserLogin):
    try:
        # Verify user credentials
        if not verify_user(user.email, user.password):
            raise HTTPException(
                status_code=401,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        access_token = create_access_token(data={"sub": user.email})
        return {"access_token": access_token, "token_type": "bearer"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        raise HTTPException(status_code=500, detail="Error during login")

# Add a specific endpoint for serving audio files
@app.get("/api/audio/{filename}")
async def get_audio(filename: str):
    try:
        file_path = os.path.join("uploads", filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Audio file not found")
            
        return FileResponse(
            file_path,
            media_type="audio/wav",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Disposition": f'inline; filename="{filename}"',
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    except Exception as e:
        logger.error(f"Error serving audio file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-markdown", response_model=Dict[str, str])
async def process_markdown(note: NoteContent):
    try:
        if not note.is_markdown:
            return {"html": note.content}
            
        # Convert Markdown to HTML
        html = markdown.markdown(
            note.content,
            extensions=[
                'markdown.extensions.fenced_code',
                'markdown.extensions.tables',
                'markdown.extensions.codehilite',
                'markdown.extensions.toc',
                'markdown.extensions.nl2br',
                'markdown.extensions.sane_lists'
            ]
        )
        
        # Add syntax highlighting CSS classes
        html = re.sub(
            r'<pre><code class="language-(\w+)">',
            r'<pre><code class="language-\1 hljs">',
            html
        )
        
        return {"html": html}
    except Exception as e:
        logger.error(f"Error processing Markdown: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing Markdown")

# Start the application
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=True)
