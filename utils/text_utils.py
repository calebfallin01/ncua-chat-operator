"""
Utility functions for text preprocessing.
"""

import re
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def expand_abbreviations(text):
    """
    Expand common credit union abbreviations.
    
    Args:
        text (str): The input text.
        
    Returns:
        str: Text with expanded abbreviations.
    """
    # Dictionary of abbreviations and their expansions
    abbreviations = {
        'fcu': 'federal credit union',
        'cu': 'credit union',
        'efcu': 'employees federal credit union',
        'corp': 'corporation',
        'natl': 'national',
        'intl': 'international',
        'coop': 'cooperative',
        'assn': 'association',
        'assoc': 'association',
        'fed': 'federal',
        'fin': 'financial',
        'comm': 'community',
        'svcs': 'services',
        'svs': 'services'
    }
    
    # Expand only if it's a standalone word or at the end
    for abbr, expansion in abbreviations.items():
        # Replace only if it's a whole word
        pattern = r'\b' + re.escape(abbr) + r'\b'
        text = re.sub(pattern, expansion, text)
    
    return text

def preprocess_text(text):
    """
    Preprocess text to normalize credit union names for better matching.
    
    Args:
        text (str): The input text to preprocess.
        
    Returns:
        tuple: (normalized_text, original_text) - both the normalized and original versions.
    """
    if not text:
        return "", ""
    
    # Save the original text
    original_text = text.strip()
    
    # Convert to lowercase
    text = text.lower().strip()
    
    # First, expand any abbreviations in the text
    text = expand_abbreviations(text)
    
    # Log expansion results if different
    if text != original_text.lower():
        logger.info(f"Expanded abbreviations: '{original_text.lower()}' -> '{text}'")
    
    # Save the expanded version for normalization
    expanded_text = text
    
    # Remove "credit union" if it appears at the end
    text = re.sub(r'\bcredit\s+union\b$', '', text).strip()
    
    # Remove "federal credit union" if it appears at the end
    text = re.sub(r'\bfederal\s+credit\s+union\b$', '', text).strip()
    
    # Remove "fcu" if it appears at the end
    text = re.sub(r'\bfcu\b$', '', text).strip()
    
    # Remove "cu" if it appears at the end
    text = re.sub(r'\bcu\b$', '', text).strip()
    
    # Remove "federal" if it appears
    text = re.sub(r'\bfederal\b', '', text).strip()
    
    # Remove "financial" if it appears at the end or anywhere
    text = re.sub(r'\bfinancial\b$', '', text).strip()
    text = re.sub(r'\bfinancial\b', '', text).strip()
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove common special characters
    text = re.sub(r'[^\w\s]', '', text).strip()
    
    # Log if the text was changed
    if text != original_text.lower():
        logger.info(f"Normalized text: '{original_text}' -> '{text}'")
        # Also log if expanded text was different
        if expanded_text != original_text.lower():
            logger.info(f"Abbreviation expansion: '{original_text}' -> '{expanded_text}'")
    
    return text, original_text 