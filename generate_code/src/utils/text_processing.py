import re
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """
    Cleans text by removing excessive whitespace and stripping.
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return str(text)
    # Replace multiple spaces/newlines with single space
    return " ".join(text.strip().split())

def format_template(template: str, **kwargs) -> str:
    """
    Safely formats a string template with keyword arguments.
    
    Args:
        template: String with placeholders (e.g., "{text}")
        **kwargs: Values for placeholders
        
    Returns:
        Formatted string
    """
    try:
        return template.format(**kwargs)
    except KeyError as e:
        # If a key is missing, we might want to raise an error or return partially formatted
        # For this project, raising error is safer to catch config issues
        raise ValueError(f"Missing key for prompt template: {e}")

def simple_tokenize(text: str) -> List[str]:
    """
    Simple whitespace tokenizer.
    
    Args:
        text: Input text
        
    Returns:
        List of tokens
    """
    return text.split()

def get_sliding_windows(text: str, window_size: int = 50, step: int = 1) -> List[str]:
    """
    Generates sliding windows of tokens from text.
    Used for Windowed PPL defense.
    
    Args:
        text: Input text
        window_size: Number of tokens per window
        step: Step size for sliding window
        
    Returns:
        List of text segments (windows)
    """
    tokens = simple_tokenize(text)
    
    # If text is shorter than window, return it as is
    if len(tokens) <= window_size:
        return [text]
    
    windows = []
    # Slide window
    for i in range(0, len(tokens) - window_size + 1, step):
        window_tokens = tokens[i : i + window_size]
        windows.append(" ".join(window_tokens))
    
    # Ensure we cover the end if step > 1 (though usually step=1 for PPL)
    # If the loop didn't cover the last segment exactly
    if step > 1 and (len(tokens) - window_size) % step != 0:
         window_tokens = tokens[-window_size:]
         windows.append(" ".join(window_tokens))
         
    return windows

def truncate_text(text: str, max_tokens: int) -> str:
    """
    Truncates text to a maximum number of tokens.
    
    Args:
        text: Input text
        max_tokens: Maximum tokens allowed
        
    Returns:
        Truncated text
    """
    tokens = simple_tokenize(text)
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens])
