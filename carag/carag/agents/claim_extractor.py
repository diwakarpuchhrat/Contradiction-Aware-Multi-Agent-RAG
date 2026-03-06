"""
Claim Extraction Agent for Phase-2 RAG pipeline.

This module extracts atomic factual claims from retrieved chunks without
reasoning, summarization, or contradiction detection.
"""

import json
import uuid
from typing import List, Dict, Any, Literal
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


# Claim type definition
ClaimType = Literal["observational", "experimental", "guideline", "opinion", "unknown"]


class Claim:
    """
    Structured claim extracted from a chunk.
    
    Attributes:
        claim_id: Unique identifier for the claim
        claim_text: The factual assertion text
        source_url: URL of the source document
        chunk_id: Identifier for the chunk this claim came from
        claim_type: Type of claim (observational, experimental, guideline, opinion, unknown)
    """
    
    def __init__(
        self,
        claim_text: str,
        source_url: str,
        chunk_id: str,
        claim_type: ClaimType = "unknown",
        claim_id: str = None
    ):
        self.claim_id = claim_id or str(uuid.uuid4())
        self.claim_text = claim_text
        self.source_url = source_url
        self.chunk_id = chunk_id
        self.claim_type = claim_type
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert claim to dictionary format."""
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "source_url": self.source_url,
            "chunk_id": self.chunk_id,
            "claim_type": self.claim_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Claim":
        """Create a Claim from a dictionary."""
        return cls(
            claim_text=data["claim_text"],
            source_url=data["source_url"],
            chunk_id=data["chunk_id"],
            claim_type=data.get("claim_type", "unknown"),
            claim_id=data.get("claim_id")
        )


def extract_claims_from_chunk(
    chunk_text: str,
    source_url: str,
    chunk_id: str
) -> List[Claim]:
    """
    Extract atomic factual claims from a single chunk.
    
    This function uses an LLM to extract claims but enforces strict constraints:
    - No reasoning or inference
    - No summarization
    - No contradiction detection
    - Only explicit factual statements
    
    Args:
        chunk_text: The text content of the chunk
        source_url: URL of the source document
        chunk_id: Unique identifier for this chunk
    
    Returns:
        List of Claim objects (empty list if no claims found)
    """
    if not chunk_text or not chunk_text.strip():
        return []
    
    # Strict extraction prompt
    prompt = f"""Extract all explicit factual claims stated in the text below.

CRITICAL RULES:
1. Extract ONLY explicit factual statements that are directly stated in the text
2. Each claim must be atomic (one factual assertion per claim)
3. Do NOT infer, combine, summarize, or explain
4. Do NOT detect contradictions or analyze stance
5. Do NOT reference external knowledge
6. If no clear factual claims exist, return {{"claims": []}}
7. Each claim must be grounded strictly in the chunk text
8. Output ONLY valid JSON, no additional text

For each claim, classify it as one of:
- "observational": Based on observations or studies
- "experimental": Based on experiments or trials
- "guideline": A recommendation or guideline
- "opinion": An opinion or subjective statement
- "unknown": If classification is uncertain

Output format (JSON object with "claims" array):
{{
  "claims": [
    {{
      "claim_text": "exact factual statement from text",
      "claim_type": "observational|experimental|guideline|opinion|unknown"
    }}
  ]
}}

Text to extract claims from:
{chunk_text}

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
                        "content": "You are a factual claim extractor. Extract only explicit factual statements. Output valid JSON only."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0,  # Deterministic extraction
                response_format={"type": "json_object"}  # Force JSON output
            )
        except Exception:
            # Fallback if response_format is not supported
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a factual claim extractor. Extract only explicit factual statements. Output valid JSON only."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0  # Deterministic extraction
            )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            parsed = json.loads(response_text)
            
            # Handle different response formats
            if isinstance(parsed, dict) and "claims" in parsed:
                claims_data = parsed["claims"]
            elif isinstance(parsed, list):
                # Fallback: if it's a direct array
                claims_data = parsed
            elif isinstance(parsed, dict):
                # Try to find any array in the response
                array_values = [v for v in parsed.values() if isinstance(v, list)]
                claims_data = array_values[0] if array_values else []
            else:
                claims_data = []
            
            # Validate and create Claim objects
            claims = []
            for claim_data in claims_data:
                if not isinstance(claim_data, dict):
                    continue
                
                claim_text = claim_data.get("claim_text", "").strip()
                if not claim_text:
                    continue
                
                claim_type = claim_data.get("claim_type", "unknown")
                if claim_type not in ["observational", "experimental", "guideline", "opinion", "unknown"]:
                    claim_type = "unknown"
                
                claim = Claim(
                    claim_text=claim_text,
                    source_url=source_url,
                    chunk_id=chunk_id,
                    claim_type=claim_type
                )
                claims.append(claim)
            
            return claims
            
        except json.JSONDecodeError as e:
            # If JSON parsing fails, try to extract JSON from the response
            # Sometimes LLMs add explanatory text before/after JSON
            import re
            # Try to find JSON object or array
            json_match = re.search(r'\{[^{}]*"claims"[^{}]*\[.*?\]', response_text, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    if isinstance(parsed, dict) and "claims" in parsed:
                        claims_data = parsed["claims"]
                    elif isinstance(parsed, list):
                        claims_data = parsed
                    else:
                        claims_data = []
                    
                    claims = []
                    for claim_data in claims_data:
                        if isinstance(claim_data, dict):
                            claim_text = claim_data.get("claim_text", "").strip()
                            if claim_text:
                                claim_type = claim_data.get("claim_type", "unknown")
                                if claim_type not in ["observational", "experimental", "guideline", "opinion", "unknown"]:
                                    claim_type = "unknown"
                                claims.append(Claim(
                                    claim_text=claim_text,
                                    source_url=source_url,
                                    chunk_id=chunk_id,
                                    claim_type=claim_type
                                ))
                    return claims
                except:
                    pass
            
            # If all parsing fails, return empty list
            print(f"⚠️  Failed to parse JSON from claim extraction response: {e}")
            return []
            
    except Exception as e:
        print(f"⚠️  Error extracting claims from chunk {chunk_id}: {e}")
        return []


def extract_claims_from_chunks(retrieved_chunks: List[Dict[str, Any]]) -> List[Claim]:
    """
    Extract claims from multiple retrieved chunks.
    
    Args:
        retrieved_chunks: List of chunk dictionaries with 'text' and 'meta' keys
            Each chunk should have:
            - text: The chunk text content
            - meta: Metadata dict with 'source' (URL) and optionally 'doc_id'
    
    Returns:
        List of all extracted Claim objects
    """
    all_claims = []
    
    for idx, chunk in enumerate(retrieved_chunks):
        chunk_text = chunk.get("text", "")
        meta = chunk.get("meta", {})
        source_url = meta.get("source", "unknown")
        
        # Generate chunk_id if not present
        chunk_id = meta.get("chunk_id") or f"chunk_{idx}_{meta.get('doc_id', 'unknown')}"
        
        # Extract claims from this chunk
        claims = extract_claims_from_chunk(chunk_text, source_url, chunk_id)
        all_claims.extend(claims)
    
    return all_claims
