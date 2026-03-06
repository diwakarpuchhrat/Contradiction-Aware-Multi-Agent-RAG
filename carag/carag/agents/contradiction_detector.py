"""
Contradiction Detection Agent (Phase 3).

This module operates strictly over *structured claims* produced by the
Claim Extraction Agent. It builds a claim–claim relationship graph
without deciding which claim is true or generating any final answers.

Responsibilities (and non‑responsibilities):
- ✅ Compare pairs of claims and classify their logical relation as:
    - "entails"
    - "contradicts"
    - "neutral"
- ✅ Produce machine‑readable edge objects with confidence levels.
- ❌ No truth assignment, summarisation, or stance decisions.
- ❌ No merging of claims or explanation of disagreements.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations, islice
from typing import List, Literal, Dict, Any, Optional

from groq import Groq

from config import GROQ_MODEL, get_groq_api_key
from agents.claim_extractor import Claim


RelationLabel = Literal["entails", "contradicts", "neutral"]
ConfidenceLevel = Literal["low", "medium", "high"]


_client: Groq | None = None


def _get_client() -> Groq:
    """
    Lazy-load the Groq client.

    Using the same pattern as other agents keeps configuration
    consistent and makes it easy to swap to a local NLI model later.
    """
    global _client
    if _client is None:
        api_key = get_groq_api_key()
        _client = Groq(api_key=api_key) if api_key else Groq()
    return _client


@dataclass
class ClaimRelation:
    """
    Pairwise relationship between two structured claims.

    This is the *only* output type of the contradiction detector.
    It encodes a labelled edge in the claim graph.
    """

    claim_id_1: str
    claim_id_2: str
    relation: RelationLabel
    confidence: ConfidenceLevel

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id_1": self.claim_id_1,
            "claim_id_2": self.claim_id_2,
            "relation": self.relation,
            "confidence": self.confidence,
        }


def _classify_relation_with_llm(claim_a: Claim, claim_b: Claim) -> ClaimRelation:
    """
    Use an LLM in strict classification mode to label the relation
    between two claims as entails / contradicts / neutral.

    Epistemic discipline:
    - The model is *not* allowed to decide which claim is true.
    - The model must not bring in external knowledge; it reasons only
      about the logical relationship between the two strings.
    """
    system_msg = (
        "You are a careful natural language inference (NLI) classifier. "
        "Given two claims, you must classify ONLY their logical relation, "
        "without deciding which claim is actually true."
    )

    user_prompt = f"""You will compare two claims and classify their logical relationship.

CRITICAL RULES:
1. Consider ONLY the text of the two claims.
2. Do NOT use external knowledge or assumptions.
3. Do NOT decide which claim is correct.
4. Do NOT merge, rephrase, or summarise the claims.
5. Output ONLY valid JSON, no extra commentary.

Possible relations:
- "entails": If Claim A logically implies Claim B, or B implies A
  (they can be paraphrases or strictly more specific/general versions
  that are logically compatible).
- "contradicts": If Claim A and Claim B cannot both be true at the same time.
- "neutral": If Claim A and Claim B can both be true, or are about
  different aspects with no clear entailment or contradiction.

You must also provide a confidence level in your classification:
- "high"
- "medium"
- "low"

Input claims:
- Claim A: "{claim_a.claim_text}"
- Claim B: "{claim_b.claim_text}"

Output format (JSON object):
{{
  "relation": "entails" | "contradicts" | "neutral",
  "confidence": "low" | "medium" | "high"
}}

Output JSON:"""

    try:
        client = _get_client()

        # Prefer structured JSON output if supported
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,  # Deterministic classification
                response_format={"type": "json_object"},
            )
        except Exception:
            # Fallback without explicit response_format
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )

        raw = response.choices[0].message.content.strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Try to rescue embedded JSON if the model wrapped it in text
            import re

            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                # Fall back to neutral / low if we can't parse anything
                return ClaimRelation(
                    claim_id_1=claim_a.claim_id,
                    claim_id_2=claim_b.claim_id,
                    relation="neutral",
                    confidence="low",
                )
            try:
                parsed = json.loads(match.group(0))
            except Exception:
                return ClaimRelation(
                    claim_id_1=claim_a.claim_id,
                    claim_id_2=claim_b.claim_id,
                    relation="neutral",
                    confidence="low",
                )

        relation = parsed.get("relation", "neutral")
        if relation not in ("entails", "contradicts", "neutral"):
            relation = "neutral"

        confidence = parsed.get("confidence", "medium")
        if confidence not in ("low", "medium", "high"):
            confidence = "medium"

        return ClaimRelation(
            claim_id_1=claim_a.claim_id,
            claim_id_2=claim_b.claim_id,
            relation=relation,  # type: ignore[arg-type]
            confidence=confidence,  # type: ignore[arg-type]
        )

    except Exception as e:
        # On any failure, default to neutral / low to avoid fabricating structure.
        print(f"⚠️  Error in contradiction classification for claims "
              f"{claim_a.claim_id} and {claim_b.claim_id}: {e}")
        return ClaimRelation(
            claim_id_1=claim_a.claim_id,
            claim_id_2=claim_b.claim_id,
            relation="neutral",
            confidence="low",
        )


def build_contradiction_graph(
    claims: List[Claim],
    max_pairs: Optional[int] = None,
) -> List[ClaimRelation]:
    """
    Build a pairwise claim relationship graph for Phase 3.

    This function:
    - Iterates over all *unique* unordered pairs of distinct claims
      (no self-comparisons, no duplicate edges).
    - Calls the NLI classifier for each pair.
    - Returns a flat list of JSON-serialisable edge objects.

    The caller is responsible for any downstream interpretation
    (stance clustering, explanations, decisions).
    """
    if not claims or len(claims) < 2:
        return []

    relations: List[ClaimRelation] = []

    # Use combinations to avoid duplicate and self-pairs.
    # Optionally cap the total number of evaluated pairs to avoid
    # pathological O(n^2) behaviour on very large claim sets.
    all_pairs = combinations(claims, 2)
    if max_pairs is not None and max_pairs > 0:
        all_pairs = islice(all_pairs, max_pairs)

    for claim_a, claim_b in all_pairs:
        relation = _classify_relation_with_llm(claim_a, claim_b)
        relations.append(relation)

    return relations


def serialize_relations(relations: List[ClaimRelation]) -> List[Dict[str, Any]]:
    """
    Helper to convert a list of ClaimRelation objects to pure JSON.

    This keeps phases loosely coupled: consumers in later stages
    can depend only on the schema, not Python classes.
    """
    return [r.to_dict() for r in relations]

