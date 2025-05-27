import os
import json
import logging
import re
import time
import random
import requests
from PIL import Image
import pytesseract
from collections import Counter
from typing import List, Dict, Any, Optional, Union
import pyttsx3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import NLTK resources
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords

    # Initialize NLTK resources - create data directory if needed
    nltk_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)
    
    # Download resources if not already present
    for resource in ['punkt', 'stopwords']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
            logger.info(f"NLTK resource '{resource}' is already downloaded")
        except LookupError:
            logger.info(f"Downloading NLTK resource '{resource}'")
            nltk.download(resource, download_dir=nltk_data_dir, quiet=True)

    # Get stopwords
    stop_words = set(stopwords.words('english'))
    logger.info("NLTK initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize NLTK: {str(e)}")
    # Fallback stopwords if NLTK import fails
    stop_words = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
                  "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
                  "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
                  "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
                  "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
                  "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
                  "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
                  "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
                  "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
                  "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
                  "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
                  "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"}

class AIService:
    def __init__(self):
        # OpenRouter API configuration
        self.api_key = "YOUR_API_KEY"
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "mistralai/mistral-7b-instruct:free"
        self.use_api = bool(self.api_key)

        # Log configuration
        logger.info(f"AI Service initialized with API: {self.use_api}")
        logger.info(f"Using model: {self.model}")

    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        try:
            return sent_tokenize(text)
        except:
            # Fallback if NLTK fails
            return [s.strip() for s in text.split('.') if s.strip()]

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        words = text.lower().split()
        keywords = [w for w in words if len(w) > 3 and w.lower() not in stop_words]
        word_count = Counter(keywords)
        return [word for word, _ in word_count.most_common(10)]

    def _query_model(self, prompt: str, task_type: str = "general") -> Optional[str]:
        """Query the OpenRouter model with retry logic"""
        if not self.api_key:
            logger.warning("No API key available for OpenRouter")
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:8000",  # Required by OpenRouter
            "X-Title": "AI Note Taking App"  # Optional but recommended
        }

        # Format the prompt for Mistral
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.7,
            "top_p": 0.9
        }

        # Add task-specific parameters
        if task_type == "summarize":
            payload["max_tokens"] = 200
        elif task_type == "quiz":
            payload["max_tokens"] = 800
            payload["temperature"] = 0.8
        elif task_type == "mindmap":
            payload["max_tokens"] = 400

        # Retry logic
        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                logger.info(f"Querying model {self.model} for {task_type} (attempt {attempt+1}/{max_retries})")
                logger.debug(f"Sending payload: {payload}")
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)

                if response.status_code == 200:
                    try:
                        result = response.json()
                        logger.debug(f"API response: {result}")

                        # Extract the response from OpenRouter's format
                        if "choices" in result and len(result["choices"]) > 0:
                            return result["choices"][0]["message"]["content"]
                        else:
                            return str(result)
                    except Exception as e:
                        logger.error(f"Failed to parse API response: {str(e)}")
                        return response.text

                elif response.status_code == 429:
                    logger.warning(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue

                elif response.status_code == 503:
                    logger.warning(f"Service unavailable. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue

                else:
                    logger.error(f"API request failed with status {response.status_code}: {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return None

            except requests.Timeout:
                logger.warning(f"Request timed out. Retrying in {retry_delay} seconds...")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return None

            except Exception as e:
                logger.error(f"Error querying model: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return None

        logger.error(f"Failed to get response after {max_retries} attempts")
        return None

    def summarize_text(self, text: str) -> str:
        """Generate a summary of the text"""
        if not text or not text.strip():
            return "No content to summarize."

        try:
            if self.use_api:
                # Limit text to first 1500 characters to avoid token limits
                # For longer texts, we could split and summarize parts separately
                prompt = f"""Summarize the following text in a concise paragraph:

{text[:1500]}

Your summary should capture the main points and key details in a clear, coherent manner.
"""
                result = self._query_model(prompt, "summarize")

                if result:
                    # Clean up the response
                    # Remove any prefixes like "Summary:" or "Here's a summary:"
                    result = re.sub(r'^(Summary:|Here\'s a summary:|Here\'s the summary:)\s*', '', result, flags=re.IGNORECASE)
                    return result.strip()
                else:
                    return "Failed to generate summary."
            else:
                # Fallback to basic summarization if API is not available
                sentences = self._extract_sentences(text)
                if len(sentences) <= 3:
                    return text
                return ". ".join(sentences[:3]) + "."
        except Exception as e:
            logger.error(f"Error in summarize_text: {str(e)}")
            return "Error generating summary."

    def generate_quiz(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Generate a quiz based on the text"""
        if not text.strip():
            return {"mcq": [], "true_false": [], "fill_blank": []}

        if self.use_api:
            prompt = f"""Create a quiz based on the following text. Include:
1. 3 multiple-choice questions with 4 options each
2. 2 true/false questions
3. 2 fill-in-the-blank questions

Format your response as JSON with the following structure:
{{
  "mcq": [
    {{"question": "...", "options": ["...", "...", "...", "..."], "answer": "..."}}
  ],
  "true_false": [
    {{"question": "...", "answer": true/false}}
  ],
  "fill_blank": [
    {{"question": "...", "answer": "..."}}
  ]
}}

Text to create quiz from:
{text[:1500]}
"""
            result = self._query_model(prompt, "quiz")

            if result:
                # Try to extract JSON from the response
                try:
                    # Find JSON-like content in the response
                    json_match = re.search(r'({[\s\S]*})', result)
                    if json_match:
                        json_str = json_match.group(1)
                        quiz_data = json.loads(json_str)

                        # Validate the structure
                        if (isinstance(quiz_data, dict) and
                            "mcq" in quiz_data and
                            "true_false" in quiz_data and
                            "fill_blank" in quiz_data):
                            return quiz_data
                except Exception as e:
                    logger.error(f"Failed to parse quiz JSON: {str(e)}")

        # Local implementation as fallback
        sentences = self._extract_sentences(text)
        keywords = self._extract_keywords(text)

        quiz = {
            "mcq": [],
            "true_false": [],
            "fill_blank": []
        }

        # Generate multiple choice questions
        if len(sentences) >= 3:
            for i in range(min(3, len(sentences))):
                sentence = sentences[i]
                question = f"What is the main point of: '{sentence}'?"

                # Create options with the correct answer and distractors
                correct_answer = "It's the main concept"
                options = [
                    correct_answer,
                    "It's a supporting detail",
                    "It's an example",
                    "It's a conclusion"
                ]

                # Shuffle options
                random.shuffle(options)

                quiz["mcq"].append({
                    "question": question,
                    "options": options,
                    "answer": correct_answer
                })

        # Generate true/false questions
        if len(sentences) >= 2:
            for i in range(min(2, len(sentences))):
                sentence = sentences[i]
                is_true = random.choice([True, False])

                if is_true:
                    question = f"The following statement is important: '{sentence}'"
                else:
                    # Create a false statement by modifying the sentence
                    words = sentence.split()
                    if len(words) > 3:
                        # Replace a word with a keyword
                        idx = random.randint(0, len(words) - 1)
                        original_word = words[idx]
                        words[idx] = random.choice(keywords) if keywords else "different"
                        question = f"The text states: '{' '.join(words)}'"
                    else:
                        question = f"The text mentions: '{sentence} and something unrelated'"

                quiz["true_false"].append({
                    "question": question,
                    "answer": is_true
                })

        # Generate fill in the blank
        if keywords:
            for i in range(min(2, len(keywords))):
                keyword = keywords[i]

                # Find a sentence containing the keyword
                for sentence in sentences:
                    if keyword.lower() in sentence.lower():
                        # Create a fill-in-the-blank question
                        blank_sentence = re.sub(r'\b' + re.escape(keyword) + r'\b', "______", sentence, flags=re.IGNORECASE)

                        quiz["fill_blank"].append({
                            "question": blank_sentence,
                            "answer": keyword
                        })
                        break

        # Ensure there's at least one question in each category
        if not quiz["mcq"]:
            quiz["mcq"].append({
                "question": "What is the main topic of this text?",
                "options": ["Main concept", "Supporting detail", "Example", "Conclusion"],
                "answer": "Main concept"
            })

        if not quiz["true_false"]:
            quiz["true_false"].append({
                "question": "This text contains important information.",
                "answer": True
            })

        if not quiz["fill_blank"]:
            quiz["fill_blank"].append({
                "question": "This text is about ______.",
                "answer": keywords[0] if keywords else "information"
            })

        return quiz

    def generate_mindmap(self, text: str) -> Dict[str, Any]:
        """Generate a mind map from the text"""
        if not text.strip():
            return {"central": "Empty", "branches": []}

        if self.use_api:
            prompt = f"""Create a mind map based on the following text. Format your response as JSON with the following structure:
{{
  "central": "Main topic",
  "branches": [
    {{
      "topic": "Branch topic",
      "subtopics": ["Subtopic 1", "Subtopic 2"]
    }}
  ]
}}

Text to create mind map from:
{text[:1500]}
"""
            result = self._query_model(prompt, "mindmap")

            if result:
                # Try to extract JSON from the response
                try:
                    # Find JSON-like content in the response
                    json_match = re.search(r'({[\s\S]*})', result)
                    if json_match:
                        json_str = json_match.group(1)
                        mindmap_data = json.loads(json_str)

                        # Validate the structure
                        if (isinstance(mindmap_data, dict) and
                            "central" in mindmap_data and
                            "branches" in mindmap_data):
                            return mindmap_data
                except Exception as e:
                    logger.error(f"Failed to parse mindmap JSON: {str(e)}")

        # Local implementation as fallback
        keywords = self._extract_keywords(text)
        sentences = self._extract_sentences(text)

        # Create simple mind map structure
        central_topic = "Main Topic"

        # If we have enough words, use the most common as the central topic
        if keywords and len(keywords) > 0:
            central_topic = keywords[0].capitalize()

        branches = []
        for word in keywords[1:7]:  # Use the next 6 most common words as branches
            # Find sentences containing this keyword
            related_sentences = [s for s in sentences if word.lower() in s.lower()]
            related_words = []

            if related_sentences:
                # Extract related words from sentences containing this keyword
                for sentence in related_sentences:
                    words = [w for w in sentence.split()
                           if len(w) > 3 and w.lower() != word.lower() and w.lower() not in stop_words]
                    related_words.extend(words)

            # Get the most common related words
            related_count = Counter(related_words)
            subtopics = [w.capitalize() for w, _ in related_count.most_common(3)]

            # If we couldn't find related words, use generic subtopics
            if not subtopics:
                subtopics = ["Related concept", "Detail"]

            branches.append({
                "topic": word.capitalize(),
                "subtopics": subtopics
            })

        # Ensure we have at least one branch
        if not branches:
            branches.append({
                "topic": "Subtopic",
                "subtopics": ["Detail 1", "Detail 2"]
            })

        mindmap = {
            "central": central_topic,
            "branches": branches
        }

        return mindmap

    def structure_and_enhance_note(self, text: str) -> Dict[str, Any]:
        """
        Analyzes and enhances a note by extracting structure, key concepts, and reformatting.
        
        Args:
            text (str): The input note text to be structured and enhanced
            
        Returns:
            Dict[str, Any]: A dictionary containing:
                - structured_text: The reformatted note with proper headings and organization
                - key_concepts: List of key concepts and definitions
                - outline: Hierarchical outline of the note's structure
        """
        if not text or not text.strip():
            return {
                "structured_text": "",
                "key_concepts": [],
                "outline": []
            }

        try:
            if self.use_api:
                # First, analyze the structure and key concepts
                analysis_prompt = f"""Analyze the following note and provide:
1. A hierarchical outline of main topics and subtopics
2. Key concepts and their definitions
3. Suggestions for better organization

Note text:
{text[:1500]}

Format your response as JSON with the following structure:
{{
    "outline": [
        {{"topic": "Main Topic 1", "subtopics": ["Subtopic 1.1", "Subtopic 1.2"]}},
        {{"topic": "Main Topic 2", "subtopics": ["Subtopic 2.1", "Subtopic 2.2"]}}
    ],
    "key_concepts": [
        {{"concept": "Concept 1", "definition": "Definition 1"}},
        {{"concept": "Concept 2", "definition": "Definition 2"}}
    ],
    "organization_suggestions": ["Suggestion 1", "Suggestion 2"]
}}"""

                analysis_result = self._query_model(analysis_prompt, "general")
                
                if not analysis_result:
                    return {
                        "structured_text": text,
                        "key_concepts": [],
                        "outline": []
                    }

                try:
                    analysis = json.loads(analysis_result)
                except json.JSONDecodeError:
                    logger.error("Failed to parse analysis result as JSON")
                    return {
                        "structured_text": text,
                        "key_concepts": [],
                        "outline": []
                    }

                # Now, use the analysis to reformat the note
                reformat_prompt = f"""Reformat the following note based on this analysis:
{json.dumps(analysis, indent=2)}

Note text:
{text[:1500]}

Create a well-structured version of the note with:
1. Clear headings and subheadings
2. Highlighted key concepts
3. Better organization of ideas
4. Improved readability

Format your response as JSON with the following structure:
{{
    "structured_text": "The reformatted note text with markdown formatting"
}}"""

                reformat_result = self._query_model(reformat_prompt, "general")
                
                if not reformat_result:
                    return {
                        "structured_text": text,
                        "key_concepts": analysis.get("key_concepts", []),
                        "outline": analysis.get("outline", [])
                    }

                try:
                    reformatted = json.loads(reformat_result)
                    return {
                        "structured_text": reformatted.get("structured_text", text),
                        "key_concepts": analysis.get("key_concepts", []),
                        "outline": analysis.get("outline", [])
                    }
                except json.JSONDecodeError:
                    logger.error("Failed to parse reformat result as JSON")
                    return {
                        "structured_text": text,
                        "key_concepts": analysis.get("key_concepts", []),
                        "outline": analysis.get("outline", [])
                    }

            else:
                logger.warning("API not available for note structuring")
                return {
                    "structured_text": text,
                    "key_concepts": [],
                    "outline": []
                }

        except Exception as e:
            logger.error(f"Error in structure_and_enhance_note: {str(e)}")
            return {
                "structured_text": text,
                "key_concepts": [],
                "outline": []
            }

    def debug_api_connection(self) -> Dict[str, Any]:
        """Test the API connection and return diagnostic information"""
        test_prompt = "Summarize this sentence in a few words: The quick brown fox jumps over the lazy dog."

        try:
            api_url = f"{self.api_url}{self.model}"
            headers = {"Authorization": f"Bearer {self.api_key}"}

            formatted_prompt = f"<s>[INST] {test_prompt} [/INST]"

            payload = {
                "inputs": formatted_prompt,
                "parameters": {
                    "max_new_tokens": 50,
                    "temperature": 0.7,
                    "return_full_text": False
                }
            }

            logger.info(f"Testing API connection to {api_url}")
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)

            result = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content_type": response.headers.get('Content-Type', 'unknown'),
                "api_key_valid": self.api_key is not None and len(self.api_key) > 10,
                "model": self.model
            }

            if response.status_code == 200:
                result["response"] = response.json()
                result["success"] = True
            else:
                result["error"] = response.text
                result["success"] = False

            return result

        except Exception as e:
            logger.error(f"API connection test failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "api_key_valid": self.api_key is not None and len(self.api_key) > 10,
                "model": self.model
            }

    def text_to_speech(self, text: str) -> str:
        """Convert text to speech using pyttsx3."""
        try:
            if not text or not isinstance(text, str):
                raise ValueError("Invalid text input")

            # Initialize the text-to-speech engine
            engine = pyttsx3.init()
            
            # Set properties (optional)
            engine.setProperty('rate', 150)    # Speed of speech
            engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
            
            # Create uploads directory if it doesn't exist
            os.makedirs("uploads", exist_ok=True)
            
            # Use a fixed filename
            filename = "tts_output.wav"
            output_file = f"uploads/{filename}"
            
            try:
                # Save to file
                engine.save_to_file(text, output_file)
                engine.runAndWait()
                
                # Verify the file was created
                if not os.path.exists(output_file):
                    raise Exception("Failed to create audio file")
                
                # Verify the file has content
                if os.path.getsize(output_file) == 0:
                    raise Exception("Generated audio file is empty")
                
                logger.info(f"Successfully generated audio file: {output_file}")
                # Return the URL in the format that works with the API endpoint
                return f"/api/audio/{filename}"
                
            except Exception as e:
                logger.error(f"Error during text-to-speech conversion: {str(e)}")
                if os.path.exists(output_file):
                    try:
                        os.remove(output_file)
                    except:
                        pass
                raise Exception(f"Failed to convert text to speech: {str(e)}")

        except Exception as e:
            logger.error(f"Text-to-speech error: {str(e)}")
            raise

# Create a singleton instance
_ai_service = None

def get_ai_service() -> AIService:
    """Get the AI service singleton instance"""
    global _ai_service
    if _ai_service is None:
        _ai_service = AIService()
    return _ai_service
