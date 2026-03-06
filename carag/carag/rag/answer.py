from groq import Groq
from config import GROQ_MODEL, get_groq_api_key

# Initialize client
_client = None


def get_client():
    """Lazy load the Groq client."""
    global _client
    if _client is None:
        api_key = get_groq_api_key()
        _client = Groq(api_key=api_key) if api_key else Groq()
    return _client


def generate_answer(question, contexts):
    """
    Generate an answer using retrieved contexts.
    
    Args:
        question: User's question
        contexts: List of context dictionaries with 'text' and 'meta' keys
    
    Returns:
        Generated answer string
    """
    if not contexts:
        return "Not found in sources."
    
    # Format contexts with source information
    context_text = "\n\n".join(
        f"Source: {c['meta'].get('source', 'Unknown')}\n{c['text']}"
        for c in contexts
    )
    
    prompt = f"""You are a helpful assistant that answers questions based on the provided sources.

Sources:
{context_text}

Question: {question}

Instructions:
- Answer the question using information from the sources above
- If the sources contain relevant information, provide a comprehensive answer
- If the sources don't contain enough information to fully answer the question, provide a partial answer based on what is available
- Only say "Not found in sources" if the sources contain absolutely no relevant information about the question
- Cite which source(s) you used when possible

Answer:"""
    
    try:
        client = get_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1  # Slightly higher for more natural responses
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {e}"

