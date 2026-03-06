"""
Multi-Stance Explanation & Decision Agent (Phase 5).

This module consumes:
- User question
- Stance clusters from Phase 4
- Claim-level relations from Phase 3 (for consensus detection)

and produces a *final structured JSON response* that:
- Preserves multiple stances and explicit disagreement.
- Does NOT hide minority views or fabricate consensus.
- Optionally suggests a recommended position when one stance has
  substantially more supporting evidence or matches a user preference.

The output strictly follows:

{
  "question": "",
  "stances": [
    {
      "stance_summary": "",
      "supporting_sources": "",
      "claim_count": ""
    }
  ],
  "consensus_status": "consensus | disagreement",
  "recommended_position": ""
}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Literal, Set, Tuple

from agents.claim_extractor import Claim
from agents.contradiction_detector import ClaimRelation
from agents.stance_clusterer import StanceCluster


ConsensusStatus = Literal["consensus", "disagreement"]


@dataclass
class _StanceView:
    """
    Internal representation of a stance plus its claims and sources.
    """

    stance: StanceCluster
    claims: List[Claim]
    sources: Set[str]


def _map_claims_to_stances(
    stances: List[StanceCluster],
    claims: List[Claim],
) -> Dict[str, _StanceView]:
    """
    Build a helper mapping from stance_id → _StanceView.
    """
    claim_by_id: Dict[str, Claim] = {c.claim_id: c for c in claims}

    views: Dict[str, _StanceView] = {}
    for stance in stances:
        member_claims = [claim_by_id[cid] for cid in stance.claim_ids if cid in claim_by_id]
        srcs: Set[str] = set()
        for c in member_claims:
            if c.source_url:
                srcs.add(c.source_url)
        views[stance.stance_id] = _StanceView(
            stance=stance,
            claims=member_claims,
            sources=srcs,
        )
    return views


def _detect_disagreement(
    stances: List[StanceCluster],
    relations: List[ClaimRelation],
) -> bool:
    """
    Determine whether there is *genuine* disagreement between stances.

    We only treat the situation as disagreement if:
    - There are at least two stances, AND
    - There exists at least one pair of claims in *different* stances
      that are linked by a "contradicts" relation.
    """
    if len(stances) <= 1 or not relations:
        return False

    # Map each claim to its stance
    stance_by_claim: Dict[str, str] = {}
    for stance in stances:
        for cid in stance.claim_ids:
            stance_by_claim[cid] = stance.stance_id

    for rel in relations:
        if rel.relation != "contradicts":
            continue
        s1 = stance_by_claim.get(rel.claim_id_1)
        s2 = stance_by_claim.get(rel.claim_id_2)
        if s1 is not None and s2 is not None and s1 != s2:
            return True

    return False


def _choose_recommended_stance(
    views: Dict[str, _StanceView],
    consensus_status: ConsensusStatus,
    user_preferred_stance_id: Optional[str] = None,
    dominance_ratio: float = 1.5,
) -> str:
    """
    Decide on a recommended position string.

    Rules:
    - If there is consensus (no cross-stance contradictions):
        - If there is a single non-empty stance, recommend its summary.
        - Otherwise (e.g. multiple compatible sub-stances), recommend a
          generic consensus message.
    - If there is disagreement:
        - If user_preferred_stance_id is provided and valid, recommend
          that stance explicitly.
        - Else, identify the stance with the most supporting sources; if
          it exceeds the next best by `dominance_ratio`, recommend it.
        - Otherwise, return "No clear consensus."
    """
    if not views:
        return "No claim-level evidence found in retrieved sources."

    non_empty = [v for v in views.values() if v.claims]

    if consensus_status == "consensus":
        if len(non_empty) == 1:
            stance_summary = non_empty[0].stance.summary or "Single stance identified from sources."
            return stance_summary
        else:
            return "The available evidence supports a broadly consistent stance without major disagreement."

    # Disagreement case
    if user_preferred_stance_id and user_preferred_stance_id in views:
        pref = views[user_preferred_stance_id]
        base = pref.stance.summary or "User-preferred stance."
        return f"User-preferred stance: {base}"

    # Compute support by source count (primary) and claim count (tie-breaker)
    ranked: List[Tuple[str, int, int]] = []
    for sid, view in views.items():
        ranked.append((sid, len(view.sources), len(view.claims)))

    if not ranked:
        return "No clear consensus."

    # Sort by source_count desc, then claim_count desc
    ranked.sort(key=lambda x: (x[1], x[2]), reverse=True)

    top_sid, top_sources, top_claims = ranked[0]
    if len(ranked) == 1:
        stance_summary = views[top_sid].stance.summary or "Single stance identified from sources."
        return stance_summary

    second_sid, second_sources, second_claims = ranked[1]

    # Check dominance by sources; fall back to claims if sources tie
    dominant = False
    if top_sources >= dominance_ratio * max(1, second_sources):
        dominant = True
    elif top_sources == second_sources and top_claims >= dominance_ratio * max(1, second_claims):
        dominant = True

    if not dominant:
        return "No clear consensus."

    summary = views[top_sid].stance.summary or "Better-supported stance among the available sources."
    return f"Based on the number of distinct supporting sources, the best-supported stance is: {summary}"


def build_multi_stance_answer(
    question: str,
    claims: List[Claim],
    relations: List[ClaimRelation],
    stances: List[StanceCluster],
    user_preferred_stance_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Phase 5 entry point.

    Produces the final JSON object that should be returned to the user.

    This function:
    - Does NOT hide minority stances.
    - Does NOT fabricate agreement where contradictions exist.
    - Only suggests a recommended position when evidence or user
      preference clearly favors one stance.
    """
    # Map stance → claims & sources for internal reasoning
    stance_views = _map_claims_to_stances(stances, claims)

    # Determine whether we have genuine disagreement
    has_disagreement = _detect_disagreement(stances, relations)
    consensus_status: ConsensusStatus = "disagreement" if has_disagreement else "consensus"

    # Build public stance views
    public_stances: List[Dict[str, Any]] = []
    for sid, view in stance_views.items():
        summary = view.stance.summary or "No concise summary available for this stance."
        supporting_sources_str = ", ".join(sorted(view.sources)) if view.sources else "No explicit sources recorded."
        claim_count_str = str(len(view.claims))

        public_stances.append(
            {
                "stance_summary": summary,
                "supporting_sources": supporting_sources_str,
                "claim_count": claim_count_str,
            }
        )

    # If we have no stances at all but some claims, expose this explicitly
    if not public_stances and claims:
        public_stances.append(
            {
                "stance_summary": "Claims were extracted but could not be organised into clear stances.",
                "supporting_sources": "Unknown",
                "claim_count": str(len(claims)),
            }
        )

    recommended_position = _choose_recommended_stance(
        stance_views,
        consensus_status=consensus_status,
        user_preferred_stance_id=user_preferred_stance_id,
    )

    return {
        "question": question,
        "stances": public_stances,
        "consensus_status": consensus_status,
        "recommended_position": recommended_position,
    }

