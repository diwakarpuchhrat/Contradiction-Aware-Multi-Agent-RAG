from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

import torch

# Global GPU/CPU device selection
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model once at module level
_model = None


def get_model():
    """Lazy load the embedding model and move it to the best available device."""
    global _model
    if _model is None:
        model = SentenceTransformer(EMBEDDING_MODEL)
        # Move underlying model to GPU if available for faster inference
        model.to(_device)
        _model = model
    return _model


def embed_texts(texts):
    """
    Generate embeddings for a list of texts, using CUDA when available.
    
    Args:
        texts: List of text strings to embed
    
    Returns:
        Numpy array of embeddings
    """
    if not texts:
        return []
    
    model = get_model()
    # SentenceTransformers will run on the device we pass here
    return model.encode(
        texts,
        device=_device,
        show_progress_bar=False,
        convert_to_numpy=True,
        batch_size=32,
    )
    
    