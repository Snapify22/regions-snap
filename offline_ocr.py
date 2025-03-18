import re
import logging
from collections import Counter

logger = logging.getLogger(__name__)

def enhance_text_offline(text):
    """
    Enhance OCR text using local processing techniques when offline
    """
    try:
        # Apply basic text cleaning
        enhanced = clean_text(text)
        
        # Fix common OCR errors
        enhanced = fix_common_errors(enhanced)
        
        # Calculate confidence based on text quality metrics
        confidence = calculate_confidence(enhanced)
        
        return {
            "enhanced_text": enhanced,
            "original_text": text,
            "confidence_score": confidence,
            "ai_powered": False
        }
    except Exception as e:
        logger.error(f"Error in offline text enhancement: {str(e)}")
        return {
            "enhanced_text": text,
            "original_text": text,
            "confidence_score": 0.0,
            "ai_powered": False
        }

def clean_text(text):
    """Remove noise and normalize text"""
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable())
    
    # Fix line breaks
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def fix_common_errors(text):
    """Fix common OCR errors"""
    # Common OCR substitution errors
    substitutions = {
        '|': 'I',
        '0': 'O',
        '1': 'l',
        'rn': 'm',
        'cl': 'd',
        'vv': 'w'
    }
    
    # Apply substitutions
    for error, correction in substitutions.items():
        text = text.replace(error, correction)
    
    # Fix spacing around punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])\s+', r'\1 ', text)
    
    return text

def calculate_confidence(text):
    """
    Calculate confidence score based on text quality metrics
    Returns a score between 0 and 1
    """
    if not text:
        return 0.0
    
    score = 0.0
    total_metrics = 4
    
    # Check for reasonable word lengths
    words = text.split()
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    if 3 <= avg_word_length <= 10:
        score += 0.25
    
    # Check for reasonable character distribution
    char_counts = Counter(text.lower())
    total_chars = sum(char_counts.values())
    vowel_ratio = sum(char_counts[v] for v in 'aeiou') / total_chars if total_chars else 0
    if 0.2 <= vowel_ratio <= 0.5:
        score += 0.25
    
    # Check for reasonable punctuation
    punct_ratio = sum(char_counts[p] for p in '.,!?;:') / total_chars if total_chars else 0
    if punct_ratio <= 0.15:
        score += 0.25
    
    # Check for reasonable capitalization
    cap_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    if 0.1 <= cap_ratio <= 0.3:
        score += 0.25
    
    return score
