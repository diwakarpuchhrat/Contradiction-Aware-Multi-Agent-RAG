import os
import dotenv
load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
TOP_K = 5
GROQ_MODEL = "llama-3.1-8b-instant"  # change if needed (alternatives: mixtral-8x7b-32768, gemma-7b-it)

# Groq API key configuration
# Prefer setting it here for project-wide (non-PC-specific) config,
# or leave it empty to fall back to the environment variable GROQ_API_KEY.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_groq_api_key() -> str:
    """
    Return the Groq API key to use.
    Priority:
      1. Explicit value set in GROQ_API_KEY above (project-level)
      2. Environment variable GROQ_API_KEY (machine-level)
    """
    return GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")

# Phase 2.5: Relevance Filtering
RELEVANCE_SIMILARITY_THRESHOLD = 0.3  # Minimum cosine similarity for Stage 1 filter (0.0-1.0)
