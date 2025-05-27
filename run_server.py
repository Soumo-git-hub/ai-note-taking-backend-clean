import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting server...")
    try:
        uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise 