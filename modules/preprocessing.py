"""
Text Preprocessing Module
=========================
Handles text cleaning, normalization, and tokenization for both
resume text and job descriptions.
Uses NLTK for tokenization and stopword removal.
"""

import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Download required NLTK data (only downloads if not already present)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


def clean_text(text: str) -> str:
    """
    Remove unnecessary special characters, extra whitespace, and formatting artifacts.
    
    Args:
        text: Raw text extracted from resume or job description
    
    Returns:
        str: Cleaned text
    """
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Remove email-like patterns temporarily (we'll extract them separately)
    # Keep alphanumeric, common punctuation, and whitespace
    text = re.sub(r'[^\w\s@.+\-/,;:()&]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def normalize_text(text: str) -> str:
    """
    Convert text to lowercase for consistent comparison.
    
    Args:
        text: Cleaned text
    
    Returns:
        str: Normalized (lowercased) text
    """
    return text.lower()


def tokenize_text(text: str) -> list:
    """
    Split text into individual tokens (words).
    
    Args:
        text: Normalized text
    
    Returns:
        list: List of tokens
    """
    try:
        tokens = word_tokenize(text)
    except Exception:
        # Fallback to simple split if NLTK tokenizer fails
        tokens = text.split()
    return tokens


def remove_stopwords(tokens: list) -> list:
    """
    Remove common English stopwords from token list.
    
    Args:
        tokens: List of word tokens
    
    Returns:
        list: Filtered list without stopwords
    """
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    return filtered_tokens


def preprocess_text(text: str) -> dict:
    """
    Complete preprocessing pipeline for text.
    Returns a dictionary with results from each step for display purposes.
    
    Pipeline: Raw Text → Clean → Normalize → Tokenize → Remove Stopwords
    
    Args:
        text: Raw text from resume or job description
    
    Returns:
        dict: Dictionary containing results from each preprocessing step
    """
    # Step 1: Clean text
    cleaned = clean_text(text)
    
    # Step 2: Normalize text
    normalized = normalize_text(cleaned)
    
    # Step 3: Tokenize
    tokens = tokenize_text(normalized)
    
    # Step 4: Remove stopwords
    filtered_tokens = remove_stopwords(tokens)
    
    return {
        'raw_text': text,
        'cleaned_text': cleaned,
        'normalized_text': normalized,
        'tokens': tokens,
        'filtered_tokens': filtered_tokens,
        'num_raw_chars': len(text),
        'num_cleaned_chars': len(cleaned),
        'num_tokens': len(tokens),
        'num_filtered_tokens': len(filtered_tokens),
        'num_stopwords_removed': len(tokens) - len(filtered_tokens)
    }


def get_processed_text(text: str) -> str:
    """
    Get the final processed text (cleaned, normalized, stopwords removed)
    as a single string. Used for TF-IDF vectorization.
    
    Args:
        text: Raw text
    
    Returns:
        str: Processed text as a single string
    """
    result = preprocess_text(text)
    return ' '.join(result['filtered_tokens'])
