"""
Stance Clustering Agent (Phase 4).

This module consumes:
- Structured claims from Phase 2, and
- Pairwise claim relations from Phase 3

and groups claims into stance clusters.

Clustering rules:
- Claims connected via "entails" belong to the same cluster.
- Claims connected via "contradicts" must live in *different* clusters.
- Neutral relations do not force clustering, but we may optionally
  attach neutral claims to the closest existing cluster using
  semantic similarity.

Responsibilities:
- ✅ Produce stance clusters with stable IDs.
- ✅ Provide short, neutral summaries of each stance.
- ❌ Do NOT resolve disagreements or choose a “true” stance.
- ❌ Do NOT generate final user-facing answers or decisions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Any, Literal, Tuple, Set

import numpy as np

from agents.claim_extractor import Claim
from agents.contradiction_detector import ClaimRelation
from embeddings.embed import embed_texts


ConfidenceLevel = Literal["low", "medium", "high"]


@dataclass
class StanceCluster:
    """
    A stance cluster groups claims that support a coherent position.

    Attributes:
        stance_id: Stable identifier for the stance.
        claim_ids: List of claim IDs belonging to this stance.
        summary: Short, neutral description of the stance position.
        source_count: Number of distinct source URLs supporting this stance.
    """

    stance_id: str
    claim_ids: List[str]
    summary: str
    source_count: int

    def to_output_dict(self) -> Dict[str, Any]:
        """
        Convert to the Phase 4 output schema.
        """
        return {
            "stance_id": self.stance_id,
            "claims": self.claim_ids,
            "summary": self.summary,
            "source_count": str(self.source_count),
        }


class _UnionFind:
    """
    Simple Union-Find/Disjoint-Set structure for entailment clustering.
    """

    def __init__(self, elements: List[str]) -> None:
        self.parent: Dict[str, str] = {e: e for e in elements}
        self.rank: Dict[str, int] = {e: 0 for e in elements}

    def find(self, x: str) -> str:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> None:
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1


def _build_entailment_components(
    claims: List[Claim], relations: List[ClaimRelation]
) -> Dict[str, List[str]]:
    """
    Build connected components of claims under the "entails" relation.

    We treat entailment as an undirected relation for clustering:
    if A entails B or B entails A, they belong in the same component.
    """
    claim_ids = [c.claim_id for c in claims]
    uf = _UnionFind(claim_ids)

    for rel in relations:
        if rel.relation == "entails":
            uf.union(rel.claim_id_1, rel.claim_id_2)

    components: Dict[str, List[str]] = {}
    for cid in claim_ids:
        root = uf.find(cid)
        components.setdefault(root, []).append(cid)

    return components


def _cluster_neutral_or_isolated_claims(
    claims: List[Claim],
    components: Dict[str, List[str]],
    similarity_threshold: float = 0.7,
) -> Dict[str, List[str]]:
    """
    Optionally attach neutral / isolated claims to the closest
    semantic cluster using embeddings.

    This step obeys:
    - No stance resolution.
    - No modification of existing entailment-based cores.
    """
    # Map claim_id -> Claim
    claim_by_id: Dict[str, Claim] = {c.claim_id: c for c in claims}

    # Determine which IDs are already in some component
    assigned: Set[str] = set()
    for ids in components.values():
        assigned.update(ids)

    all_ids: Set[str] = set(claim_by_id.keys())
    unassigned = list(all_ids - assigned)

    # If everything is already clustered, we are done
    if not unassigned or not components:
        return components

    # Pre-compute embeddings for all claim texts
    all_claims_sorted = sorted(claim_by_id.values(), key=lambda c: c.claim_id)
    all_texts = [c.claim_text for c in all_claims_sorted]
    all_embeddings = embed_texts(all_texts)
    emb_by_id: Dict[str, np.ndarray] = {
        c.claim_id: np.array(emb) for c, emb in zip(all_claims_sorted, all_embeddings)
    }

    # Compute a centroid embedding for each existing component
    centroid_by_root: Dict[str, np.ndarray] = {}
    for root, ids in components.items():
        vecs = [emb_by_id[cid] for cid in ids if cid in emb_by_id]
        if not vecs:
            continue
        centroid_by_root[root] = np.mean(np.stack(vecs), axis=0)

    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    # Attach each unassigned claim to the closest centroid if similar enough
    for cid in unassigned:
        if cid not in emb_by_id or not centroid_by_root:
            # If we have no embedding or no existing clusters, make its own cluster
            components[cid] = [cid]
            continue

        emb = emb_by_id[cid]
        best_root = None
        best_sim = -1.0
        for root, centroid in centroid_by_root.items():
            sim = _cosine(emb, centroid)
            if sim > best_sim:
                best_sim = sim
                best_root = root

        if best_root is not None and best_sim >= similarity_threshold:
            components[best_root].append(cid)
        else:
            # Keep as its own small stance
            components[cid] = [cid]

    return components


def _summarise_stance(claims_for_stance: List[Claim]) -> str:
    """
    Create a short, neutral summary of a stance cluster.

    To preserve epistemic discipline:
    - We describe the shared position without judging correctness.
    - We avoid overgeneralising beyond the claims’ content.
    - We keep the summary very short (1–2 sentences).
    """
    if not claims_for_stance:
        return ""

    # Simple heuristic summary: use an LLM-free approach for robustness.
    # We join top-N claim texts into a compressed description.
    # If you wish, this can later be swapped for a small LLM call.
    texts = [c.claim_text.strip() for c in claims_for_stance if c.claim_text.strip()]
    if not texts:
        return ""

    # Use the first few distinct claims as a "pseudo-summary".
    # This stays grounded and avoids hallucination.
    unique_texts = []
    seen = set()
    for t in texts:
        if t not in seen:
            unique_texts.append(t)
            seen.add(t)
        if len(unique_texts) >= 3:
            break

    # Very short, neutral description
    if len(unique_texts) == 1:
        return unique_texts[0]
    else:
        return " / ".join(unique_texts)


def cluster_stances(
    claims: List[Claim],
    relations: List[ClaimRelation],
) -> List[StanceCluster]:
    """
    Main Phase 4 entry point.

    Inputs:
        claims: List of structured Claim objects from Phase 2.
        relations: List of ClaimRelation objects from Phase 3.

    Outputs:
        List of StanceCluster objects.

    Notes:
    - Entailment edges define the cores of stance clusters.
    - Contradiction edges are NOT used to merge clusters; they will be
      used by the Explanation Agent to present opposing stances.
    - Neutral edges only influence optional semantic attachment of
      previously isolated claims.
    """
    if not claims:
        return []

    claim_by_id: Dict[str, Claim] = {c.claim_id: c for c in claims}

    # Step 1: entailment-based connected components
    components = _build_entailment_components(claims, relations)

    # Step 2: attach isolated / neutral claims semantically
    components = _cluster_neutral_or_isolated_claims(claims, components)

    # Build stance clusters with summaries and source counts
    stance_clusters: List[StanceCluster] = []
    for idx, (root, cid_list) in enumerate(sorted(components.items()), start=1):
        member_claims = [claim_by_id[cid] for cid in cid_list if cid in claim_by_id]

        # Unique sources supporting this stance
        sources: Set[str] = set()
        for c in member_claims:
            if c.source_url:
                sources.add(c.source_url)

        summary = _summarise_stance(member_claims)
        stance_id = f"stance_{idx}"

        stance_clusters.append(
            StanceCluster(
                stance_id=stance_id,
                claim_ids=[c.claim_id for c in member_claims],
                summary=summary,
                source_count=len(sources),
            )
        )

    return stance_clusters


def serialize_stance_clusters(stances: List[StanceCluster]) -> List[Dict[str, Any]]:
    """
    Helper to expose stance clusters as pure JSON for downstream phases.
    """
    return [s.to_output_dict() for s in stances]

