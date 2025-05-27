import sqlite3
import os
import logging
from pathlib import Path
import bcrypt
import jwt
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

# Set up logging
logger = logging.getLogger(__name__)

# Database file path - use absolute path
DB_PATH = Path(__file__).parent.absolute() / "notes.db"

# JWT configuration
SECRET_KEY = "your-very-secret-key"  # Change this in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def init_db():
    """Initialize the database with required tables"""
    try:
        # Ensure the directory exists
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if database file exists
        db_exists = os.path.exists(DB_PATH)
        
        if db_exists:
            logger.info(f"Using existing database file: {DB_PATH}")
            # Just verify connection and structure
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            
            # Check if notes table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='notes'")
            if cursor.fetchone():
                logger.info("Notes table exists in database")
            else:
                logger.warning("Notes table not found in existing database, creating it")
                create_notes_table(cursor)
            # Create users table if not exists
            create_users_table(cursor)
                
            conn.commit()
            conn.close()
        else:
            logger.info(f"Database file not found, creating new one: {DB_PATH}")
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            create_notes_table(cursor)
            create_users_table(cursor)
            conn.commit()
            conn.close()
            logger.info("Database created successfully")
            
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        raise

def create_notes_table(cursor):
    """Create the notes table if it doesn't exist"""
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        is_markdown BOOLEAN DEFAULT 0,
        summary TEXT,
        quiz TEXT,
        mindmap TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

def create_users_table(cursor):
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

def get_all_notes():
    """Retrieve all notes from the database"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM notes ORDER BY updated_at DESC")
        notes = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return notes
    except Exception as e:
        logger.error(f"Error retrieving notes: {str(e)}")
        return []

def get_note_by_id(note_id):
    """Retrieve a specific note by ID"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM notes WHERE id = ?", (note_id,))
        note = cursor.fetchone()
        
        conn.close()
        
        if note:
            return dict(note)
        return None
    except Exception as e:
        logger.error(f"Error retrieving note {note_id}: {str(e)}")
        return None

def save_note(title, content, is_markdown=False, summary=None, quiz=None, mindmap=None):
    """Save a new note to the database"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO notes (title, content, is_markdown, summary, quiz, mindmap) VALUES (?, ?, ?, ?, ?, ?)",
            (title, content, 1 if is_markdown else 0, summary, quiz, mindmap)
        )
        
        note_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Note saved successfully with ID: {note_id}")
        return note_id
    except Exception as e:
        logger.error(f"Error saving note: {str(e)}")
        if conn:
            conn.rollback()
            conn.close()
        return -1

def update_note(note_id, update_data):
    """Update an existing note in the database"""
    conn = None
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        # Always update the updated_at timestamp
        update_data['updated_at'] = datetime.utcnow().isoformat()
        
        # Build the SET clause dynamically based on the update_data
        set_clause = ", ".join([f"{k} = ?" for k in update_data.keys()])
        query = f"UPDATE notes SET {set_clause} WHERE id = ?"
        
        # Convert is_markdown to integer if present
        values = list(update_data.values())
        if 'is_markdown' in update_data:
            is_markdown_index = list(update_data.keys()).index('is_markdown')
            values[is_markdown_index] = 1 if values[is_markdown_index] else 0
        
        values.append(note_id)
        
        cursor.execute(query, values)
        
        # Verify the update was successful
        cursor.execute("SELECT id FROM notes WHERE id = ?", (note_id,))
        if not cursor.fetchone():
            logger.error(f"Note {note_id} not found after update attempt")
            conn.rollback()
            return False
            
        conn.commit()
        logger.info(f"Note {note_id} updated successfully")
        return True
    except Exception as e:
        logger.error(f"Error updating note: {str(e)}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def delete_note(note_id):
    """Delete a note from the database"""
    conn = None
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        # First check if the note exists
        cursor.execute("SELECT id FROM notes WHERE id = ?", (note_id,))
        if not cursor.fetchone():
            logger.warning(f"Note {note_id} not found for deletion")
            return False
        
        # Delete the note
        cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        
        # Verify the deletion
        cursor.execute("SELECT id FROM notes WHERE id = ?", (note_id,))
        if cursor.fetchone():
            logger.error(f"Note {note_id} still exists after deletion attempt")
            conn.rollback()
            return False
            
        # If we get here, the deletion was successful
        conn.commit()
        logger.info(f"Note {note_id} deleted successfully")
        return True
    except Exception as e:
        logger.error(f"Error deleting note {note_id}: {str(e)}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def create_user(username: str, email: str, password: str) -> int:
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        cursor.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (username, email, hashed_password.decode('utf-8'))
        )
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return user_id
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        if conn:
            conn.rollback()
            conn.close()
        return -1

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, email, password FROM users WHERE email = ?", (email,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {"id": row[0], "username": row[1], "email": row[2], "password": row[3]}
        return None
    except Exception as e:
        logger.error(f"Error getting user by email: {str(e)}")
        return None

def verify_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    user = get_user_by_email(email)
    if not user:
        return None
    if bcrypt.checkpw(password.encode('utf-8'), user["password"].encode('utf-8')):
        return user
    return None

def create_access_token(data: Dict[str, Any]) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt