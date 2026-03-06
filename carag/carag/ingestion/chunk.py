import tiktoken
from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text, chunk_size=None, overlap=None):
    """
    Split text into chunks based on token count.
    
    Args:
        text: Text to chunk
        chunk_size: Number of tokens per chunk (default: from config)
        overlap: Number of overlapping tokens between chunks (default: from config)
    
    Returns:
        List of text chunks
    """
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if overlap is None:
        overlap = CHUNK_OVERLAP
    
    # Safety check: overlap should be less than chunk_size
    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)
    
    if not text or len(text.strip()) == 0:
        return []
    
    # Use tiktoken for accurate token counting
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    
    if len(tokens) <= chunk_size:
        return [text]
    
    chunks = []
    i = 0
    
    while i < len(tokens):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        i += chunk_size - overlap
        
        # Prevent infinite loop
        if i >= len(tokens):
            break
    
    return chunks

