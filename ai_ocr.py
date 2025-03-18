import os
import time
import openai
from logging import getLogger

logger = getLogger(__name__)

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def enhance_text_with_ai(raw_text, context=None, max_retries=3, initial_delay=1):
    """
    Use OpenAI to enhance and correct OCR text with retry logic
    Args:
        raw_text (str): The raw OCR text to enhance
        context (str, optional): Additional context about the document
        max_retries (int): Maximum number of retry attempts
        initial_delay (int): Initial delay in seconds between retries
    Returns:
        dict: Enhanced text and confidence metrics
    """
    retry_count = 0
    delay = initial_delay

    while retry_count <= max_retries:
        try:
            # Create a system message that explains the task
            system_message = """You are an expert OCR text enhancement system. Your task is to:
1. Fix common OCR errors and typos
2. Maintain the original formatting
3. Preserve all numerical data
4. Handle technical terms correctly
5. Provide confidence scores for corrections"""

            # Create the user message with the raw text
            user_message = f"""Please enhance this OCR text. Here's the raw text:
{raw_text}

If provided, here's additional context:
{context if context else 'No additional context provided'}

Please return the enhanced text and note any major corrections."""

            # Make the API call
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,  # Lower temperature for more consistent output
                max_tokens=1500
            )

            # Extract the enhanced text
            enhanced_text = response.choices[0].message.content

            # Log success
            logger.info("Successfully enhanced text with AI")

            return {
                "enhanced_text": enhanced_text,
                "original_text": raw_text,
                "confidence_score": 0.95,  # Placeholder for now
                "ai_powered": True
            }

        except openai.RateLimitError as e:
            logger.warning(f"Rate limit error (attempt {retry_count + 1}/{max_retries}): {str(e)}")
            if retry_count == max_retries:
                logger.error("Max retries reached for rate limit, returning original text")
                return fallback_response(raw_text, str(e))
            time.sleep(delay)
            delay *= 2  # Exponential backoff
            retry_count += 1

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return fallback_response(raw_text, str(e))

        except Exception as e:
            logger.error(f"Error enhancing text with AI: {str(e)}")
            return fallback_response(raw_text, str(e))

def fallback_response(raw_text, error_msg):
    """Return original text with error information when AI enhancement fails"""
    return {
        "enhanced_text": raw_text,
        "original_text": raw_text,
        "confidence_score": 0.0,
        "error": error_msg,
        "ai_powered": False
    }

def get_ocr_tool_references():
    """
    Return information about major OCR AI tools and technologies
    """
    return {
        "commercial_solutions": [
            {
                "name": "Azure Computer Vision",
                "description": "Microsoft's AI-powered OCR service with support for 164+ languages",
                "url": "https://azure.microsoft.com/en-us/services/cognitive-services/computer-vision/"
            },
            {
                "name": "Google Cloud Vision AI",
                "description": "Google's machine learning-powered OCR with high accuracy",
                "url": "https://cloud.google.com/vision"
            },
            {
                "name": "Amazon Textract",
                "description": "AWS service for extracting text from documents",
                "url": "https://aws.amazon.com/textract/"
            }
        ],
        "open_source_solutions": [
            {
                "name": "Tesseract",
                "description": "Open source OCR engine supported by Google",
                "url": "https://github.com/tesseract-ocr/tesseract"
            },
            {
                "name": "EasyOCR",
                "description": "Python library for OCR with support for 80+ languages",
                "url": "https://github.com/JaidedAI/EasyOCR"
            },
            {
                "name": "PaddleOCR",
                "description": "Multilingual OCR toolkit by Baidu",
                "url": "https://github.com/PaddlePaddle/PaddleOCR"
            }
        ]
    }