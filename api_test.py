import os
import requests
import json
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api_test.log")
    ]
)
logger = logging.getLogger("api_test")

# OpenRouter API configuration
API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "mistralai/mistral-7b-instruct:free"

def test_api_key():
    """Test if the API key is present and valid"""
    api_key = API_KEY
    
    logger.info(f"API Key present: {bool(api_key)}")
    logger.info(f"API Key length: {len(api_key) if api_key else 0}")
    
    if not api_key:
        logger.error("No API key found. Please set the OPENROUTER_API_KEY environment variable.")
        return False
    
    return True

def test_model_availability():
    """Test if the OpenRouter API is accessible"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "AI Note Taking App"
    }
    
    logger.info("Testing OpenRouter API availability")
    
    try:
        # Send a minimal request to test the API
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 10
        }
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        
        logger.info(f"API check response status: {response.status_code}")
        
        if response.status_code == 200:
            logger.info("OpenRouter API is available!")
            return True
        elif response.status_code == 401:
            logger.error("Authentication failed. Your API key may be invalid.")
            return False
        else:
            logger.error(f"Unexpected status code: {response.status_code}")
            logger.error(f"Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        logger.error(f"Error checking API availability: {str(e)}")
        return False

def test_api_request(task_type="summarize"):
    """Test a simple API request to the model"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "AI Note Taking App"
    }
    
    logger.info(f"Testing API request for task: {task_type}")
    
    # Create a simple test input
    test_text = "This is a test input to check if the API is working correctly. The quick brown fox jumps over the lazy dog."
    
    # Adjust prompt based on task
    if task_type == "summarize":
        prompt = f"Summarize the following text: {test_text}"
    elif task_type == "quiz":
        prompt = f"Create a quiz based on this text: {test_text}"
    elif task_type == "mindmap":
        prompt = f"Create a mind map for this text: {test_text}"
    else:
        prompt = test_text
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    try:
        logger.info("Sending test request to API...")
        start_time = time.time()
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Request completed in {elapsed_time:.2f} seconds")
        logger.info(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                logger.info("API request successful!")
                logger.info(f"Response type: {type(result)}")
                logger.info(f"Response preview: {str(result)[:200]}...")
                return True
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON response: {response.text[:200]}...")
                return False
        elif response.status_code == 429:
            logger.warning("Rate limit exceeded. The API might be busy or you've made too many requests.")
            return False
        elif response.status_code == 503:
            logger.warning("Service unavailable. The API might be under maintenance.")
            return False
        else:
            logger.error(f"API request failed with status {response.status_code}")
            logger.error(f"Response: {response.text[:200]}")
            return False
            
    except requests.Timeout:
        logger.error("Request timed out. The server might be busy.")
        return False
    except Exception as e:
        logger.error(f"Error making API request: {str(e)}")
        return False

def main():
    """Run all API tests"""
    logger.info("=== Starting API Tests ===")
    
    # Test the API key
    if not test_api_key():
        logger.error("API key test failed. Exiting tests.")
        return
    
    # Test the API availability
    if not test_model_availability():
        logger.error("API availability test failed. Exiting tests.")
        return
    
    # Test different tasks
    for task in ["summarize", "quiz", "mindmap"]:
        logger.info(f"\n--- Testing task: {task} ---")
        test_api_request(task)
    
    logger.info("=== API Tests Completed ===")

if __name__ == "__main__":
    main()