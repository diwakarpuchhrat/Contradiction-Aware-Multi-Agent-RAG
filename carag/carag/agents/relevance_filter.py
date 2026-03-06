"""
Relevance Filtering Layer for Phase 2.5 RAG pipeline.

This module implements a two-stage relevance filter that evaluates whether
retrieved chunks are directly relevant to the user's question before passing
them to the Claim Extraction Agent.

This layer is a binary gatekeeper - it does NOT extract claims, summarize,
reason, or decide stance. It only determines if a chunk should be considered.
"""

import json
import numpy as np
from typing import List, Dict, Any, Literal, Tuple, Optional
from groq import Groq
from config import GROQ_MODEL, get_groq_api_key
from embeddings.embed import embed_texts

# Initialize client
_client = None


def get_client():
    """Lazy load the Groq client."""
    global _client
    if _client is None:
        api_key = get_groq_api_key()
        _client = Groq(api_key=api_key) if api_key else Groq()
    return _client


# Relevance confidence levels
ConfidenceLevel = Literal["low", "medium", "high"]


class RelevanceResult:
    """
    Relevance evaluation result for a chunk.
    
    Attributes:
        chunk_id: Unique identifier for the chunk
        is_relevant: Whether the chunk is relevant to the question
        confidence: Confidence level of the relevance decision
    """
    
    def __init__(
        self,
        chunk_id: str,
        is_relevant: bool,
        confidence: ConfidenceLevel = "medium",
        similarity: Optional[float] = None,
        stage: Literal["stage1", "stage2"] = "stage2",
    ):
        self.chunk_id = chunk_id
        self.is_relevant = is_relevant
        self.confidence = confidence
        self.similarity = similarity
        self.stage = stage
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "chunk_id": self.chunk_id,
            "is_relevant": self.is_relevant,
            "confidence": self.confidence,
            "similarity": self.similarity,
            "stage": self.stage,
        }


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Cosine similarity score between -1 and 1
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def stage1_embedding_filter(
    query: str,
    query_embedding: np.ndarray,
    chunks: List[Dict[str, Any]],
    similarity_threshold: float = 0.3
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:
    """
    Stage 1: Fast embedding similarity filter.
    
    Filters out chunks with low cosine similarity to the query.
    This is a cheap pre-filter before the more expensive LLM stage.
    
    Args:
        query: User's question
        query_embedding: Embedding vector of the query
        chunks: List of chunk dictionaries with 'text' and 'meta' keys
        similarity_threshold: Minimum cosine similarity to pass (default: 0.3)
    
    Returns:
        Tuple of (passed_chunks, filtered_chunks, similarity_by_chunk_id)
    """
    if not chunks:
        return [], [], {}
    
    # Embed all chunks
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = embed_texts(chunk_texts)
    
    passed = []
    filtered = []
    similarity_by_chunk_id: Dict[str, float] = {}
    
    for i, chunk in enumerate(chunks):
        chunk_embedding = chunk_embeddings[i]
        similarity = cosine_similarity(query_embedding, chunk_embedding)
        chunk_id = chunk.get("meta", {}).get("chunk_id") or f"chunk_{i}"
        similarity_by_chunk_id[chunk_id] = float(similarity)
        
        if similarity >= similarity_threshold:
            passed.append(chunk)
        else:
            filtered.append(chunk)
    
    return passed, filtered, similarity_by_chunk_id


def stage2_llm_classifier(
    query: str,
    chunk: Dict[str, Any],
    chunk_id: str,
    similarity: Optional[float] = None,
) -> RelevanceResult:
    """
    Stage 2: LLM binary classifier for precision filtering.
    
    Uses an LLM to make a strict binary decision about whether
    the chunk contains information that directly helps answer the question.
    
    Args:
        query: User's question
        chunk: Chunk dictionary with 'text' and 'meta' keys
        chunk_id: Unique identifier for this chunk
    
    Returns:
        RelevanceResult with is_relevant and confidence
    """
    chunk_text = chunk.get("text", "").strip()
    
    if not chunk_text:
        return RelevanceResult(chunk_id, False, "low", similarity=similarity, stage="stage2")
    
    # Strict binary classification prompt
    prompt = f"""You are a relevance classifier. Your task is to determine if the text below contains information that DIRECTLY helps answer the question.

CRITICAL RULES:
1. Answer ONLY true or false
2. Do NOT explain your reasoning
3. Do NOT summarize the text
4. Do NOT add external knowledge
5. Output ONLY valid JSON
6. Mark true ONLY if the text contains information that directly addresses the question
7. Mark false for background, definitions, SEO filler, or tangentially related content
8. Be conservative: when unsure, mark false

Question: {query}

Text to evaluate:
{chunk_text}

Output format (JSON object):
{{
  "is_relevant": true or false,
  "confidence": "low" or "medium" or "high"
}}

Output JSON:"""
    
    try:
        client = get_client()
        
        # Try with response_format first, fallback if not supported
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a binary relevance classifier. Output valid JSON only. Answer true or false."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0,  # Deterministic classification
                response_format={"type": "json_object"}
            )
        except Exception:
            # Fallback if response_format is not supported
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a binary relevance classifier. Output valid JSON only. Answer true or false."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            parsed = json.loads(response_text)
            
            is_relevant = parsed.get("is_relevant", False)
            confidence = parsed.get("confidence", "medium")
            
            # Validate confidence level
            if confidence not in ["low", "medium", "high"]:
                confidence = "medium"
            
            # If LLM output is not boolean, treat as false (conservative)
            if not isinstance(is_relevant, bool):
                is_relevant = False
                confidence = "low"
            
            return RelevanceResult(chunk_id, is_relevant, confidence, similarity=similarity, stage="stage2")
            
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract JSON from response
            import re
            json_match = re.search(r'\{[^{}]*"is_relevant"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    is_relevant = parsed.get("is_relevant", False)
                    confidence = parsed.get("confidence", "medium")
                    if confidence not in ["low", "medium", "high"]:
                        confidence = "medium"
                    if not isinstance(is_relevant, bool):
                        is_relevant = False
                        confidence = "low"
                    return RelevanceResult(chunk_id, is_relevant, confidence, similarity=similarity, stage="stage2")
                except:
                    pass
            
            # If all parsing fails, treat as irrelevant (conservative)
            return RelevanceResult(chunk_id, False, "low", similarity=similarity, stage="stage2")
            
    except Exception as e:
        # On any error, treat as irrelevant (conservative bias)
        print(f"⚠️  Error in LLM relevance classification for chunk {chunk_id}: {e}")
        return RelevanceResult(chunk_id, False, "low", similarity=similarity, stage="stage2")


def filter_chunks_by_relevance(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    similarity_threshold: float = 0.3
) -> Tuple[List[Dict[str, Any]], List[RelevanceResult]]:
    """
    Two-stage relevance filtering pipeline.
    
    Stage 1: Fast embedding similarity filter (removes obviously unrelated chunks)
    Stage 2: LLM binary classifier (precision gate for chunks that pass Stage 1)
    
    Args:
        query: User's question
        retrieved_chunks: List of chunk dictionaries with 'text' and 'meta' keys
        similarity_threshold: Minimum cosine similarity for Stage 1 (default: 0.3)
    
    Returns:
        Tuple of (relevant_chunks, relevance_results)
        - relevant_chunks: Filtered list of chunks that passed both stages
        - relevance_results: List of RelevanceResult for all chunks (for debugging)
    """
    if not retrieved_chunks:
        return [], []
    
    # Generate query embedding once
    query_embedding = embed_texts([query])[0]
    
    # Stage 1: Embedding similarity filter
    stage1_passed, stage1_filtered, similarity_by_chunk_id = stage1_embedding_filter(
        query, query_embedding, retrieved_chunks, similarity_threshold
    )
    
    # Build relevance results for all chunks
    relevance_results = []
    
    # Mark filtered chunks as irrelevant
    for idx, chunk in enumerate(stage1_filtered):
        chunk_id = chunk.get("meta", {}).get("chunk_id") or f"chunk_{idx}"
        relevance_results.append(
            RelevanceResult(
                chunk_id,
                False,
                "low",
                similarity=similarity_by_chunk_id.get(chunk_id),
                stage="stage1",
            )
        )
    
    # Stage 2: LLM binary classifier for chunks that passed Stage 1
    relevant_chunks = []
    
    for idx, chunk in enumerate(stage1_passed):
        chunk_id = chunk.get("meta", {}).get("chunk_id") or f"chunk_{len(stage1_filtered) + idx}"
        
        # LLM classification
        result = stage2_llm_classifier(
            query,
            chunk,
            chunk_id,
            similarity=similarity_by_chunk_id.get(chunk_id),
        )
        relevance_results.append(result)
        
        if result.is_relevant:
            relevant_chunks.append(chunk)
    
    return relevant_chunks, relevance_results
