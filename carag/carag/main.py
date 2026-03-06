import json

from search.duckduckgo import duckduckgo_search
from ingestion.fetch import fetch_page
from ingestion.clean import clean_html
from ingestion.chunk import chunk_text
from embeddings.embed import embed_texts
from vectorstore.store import VectorStore
from agents.claim_extractor import extract_claims_from_chunks
from agents.relevance_filter import filter_chunks_by_relevance
from agents.contradiction_detector import build_contradiction_graph, serialize_relations
from agents.stance_clusterer import cluster_stances, serialize_stance_clusters
from agents.explanation_agent import build_multi_stance_answer
from config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, RELEVANCE_SIMILARITY_THRESHOLD
from rag.answer import generate_answer


def _format_text_answer(final_output: dict) -> str:
    """
    Deterministic human-readable answer derived from the structured output.
    This avoids an extra LLM call in the common case, while still providing
    a plain text response.
    """
    if not final_output:
        return "No output produced."

    question = final_output.get("question", "")
    consensus = final_output.get("consensus_status", "consensus")
    recommended = final_output.get("recommended_position", "")
    stances = final_output.get("stances", []) or []

    lines = []
    if question:
        lines.append(f"Question: {question}")
        lines.append("")

    if recommended:
        lines.append("Answer:")
        lines.append(recommended)
        lines.append("")

    lines.append(f"Consensus status: {consensus}")
    lines.append("")

    if stances:
        lines.append("Stances found:")
        for i, s in enumerate(stances, 1):
            summary = s.get("stance_summary", "")
            sources = s.get("supporting_sources", "")
            claim_count = s.get("claim_count", "")
            lines.append(f"{i}. {summary}")
            if claim_count:
                lines.append(f"   - Claims: {claim_count}")
            if sources:
                lines.append(f"   - Sources: {sources}")
        lines.append("")
    else:
        lines.append("No stance clusters were produced from the retrieved sources.")
        lines.append("")

    return "\n".join(lines).strip()


def run(query, log_fn=print):
    """
    Main RAG pipeline: search, fetch, clean, chunk, embed, retrieve, answer.
    
    Args:
        query: User's question
        log_fn: Logging function for progress updates (defaults to print).
                Streamlit or other UIs can pass a custom logger to capture logs.
    """
    log_fn(f"🔍 Searching for: {query}")
    search_results = duckduckgo_search(query, max_results=10)
    
    if not search_results:
        log_fn("❌ No search results found.")
        return None
    
    log_fn(f"✅ Found {len(search_results)} search results")
    
    # Fetch and clean documents
    log_fn("\n📥 Fetching and cleaning web pages...")
    documents = []
    for i, result in enumerate(search_results, 1):
        log_fn(f"  [{i}/{len(search_results)}] Fetching: {result['url']}")
        html = fetch_page(result["url"])
        if not html:
            log_fn(f"    ⚠️  Failed to fetch")
            continue
        
        text = clean_html(html)
        if not text or len(text.strip()) < 100:
            log_fn(f"    ⚠️  No meaningful content extracted")
            continue
        
        documents.append({
            "id": f"doc_{len(documents) + 1}",
            "source": result["url"],
            "title": result.get("title", ""),
            "text": text
        })
        log_fn(f"    ✅ Extracted {len(text)} characters")
    
    if not documents:
        log_fn("❌ No documents successfully fetched.")
        return None
    
    log_fn(f"\n✅ Successfully processed {len(documents)} documents")
    
    # Chunk documents
    log_fn("\n✂️  Chunking documents...")
    chunks = []
    metadata = []
    
    for doc in documents:
        doc_chunks = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for chunk in doc_chunks:
            chunks.append(chunk)
            metadata.append({
                "doc_id": doc["id"],
                "source": doc["source"],
                "title": doc["title"]
            })
    
    log_fn(f"✅ Created {len(chunks)} chunks")
    
    if not chunks:
        log_fn("❌ No chunks created.")
        return None
    
    # Embed chunks
    log_fn("\n🧠 Generating embeddings...")
    embeddings = embed_texts(chunks)
    log_fn(f"✅ Generated {len(embeddings)} embeddings (dim: {len(embeddings[0])})")
    
    # Create vector store
    log_fn("\n🗄️  Building vector store...")
    store = VectorStore(len(embeddings[0]))
    store.add(embeddings, chunks, metadata)
    log_fn(f"✅ Vector store ready with {store.index.ntotal} vectors")
    
    # Search for relevant chunks
    log_fn(f"\n🔎 Retrieving top {TOP_K} relevant chunks...")
    query_embedding = embed_texts([query])[0]
    retrieved = store.search(query_embedding, TOP_K)
    log_fn(f"✅ Retrieved {len(retrieved)} chunks")
    
    # Phase 2.5: Relevance Filtering Layer
    log_fn("\n🔍 Filtering chunks by relevance...")
    
    # Add chunk IDs to metadata for tracking
    for idx, chunk in enumerate(retrieved):
        if "chunk_id" not in chunk.get("meta", {}):
            chunk["meta"]["chunk_id"] = f"chunk_{idx}_{chunk['meta'].get('doc_id', 'unknown')}"
    
    relevant_chunks, relevance_results = filter_chunks_by_relevance(
        query, retrieved, RELEVANCE_SIMILARITY_THRESHOLD
    )
    
    log_fn(f"✅ Filtered to {len(relevant_chunks)} relevant chunks (removed {len(retrieved) - len(relevant_chunks)})")
    
    # Debug output: Show relevance filtering results
    log_fn("\n" + "=" * 80)
    log_fn("--- Relevance Filtering ---")
    log_fn("=" * 80)
    for result in relevance_results:
        status = "relevant" if result.is_relevant else "irrelevant"
        log_fn(f"Chunk ID: {result.chunk_id} → {status} ({result.confidence} confidence)")
    log_fn("=" * 80)
    
    # Phase-2: Extract claims from RELEVANT chunks only
    log_fn("\n🔬 Extracting claims from relevant chunks...")
    claims = extract_claims_from_chunks(relevant_chunks)
    log_fn(f"✅ Extracted {len(claims)} claims")
    
    # Display extracted claims
    if claims:
        log_fn("\n" + "=" * 80)
        log_fn("--- Extracted Claims ---")
        log_fn("=" * 80)
        for i, claim in enumerate(claims, 1):
            log_fn(f"[{i}] Claim: \"{claim.claim_text}\"")
            log_fn(f"    Type: {claim.claim_type}")
            log_fn(f"    Source: {claim.source_url}")
            log_fn(f"    Chunk ID: {claim.chunk_id}")
            log_fn("")
    else:
        log_fn("\n⚠️  No claims extracted from retrieved chunks.")
    
    # If we have no claims, we cannot build a contradiction-aware view.
    # Return a structured but minimal JSON response.
    if not claims:
        # Fallback: still produce a best-effort text answer from the retrieved contexts.
        # This prevents the app from returning "no relevant output" in many practical cases.
        log_fn("\n🧾 Generating fallback text answer from retrieved contexts...")
        fallback_text = generate_answer(query, retrieved) if retrieved else "Not found in sources."

        final_output = {
            "question": query,
            "stances": [],
            "consensus_status": "consensus",
            "recommended_position": "No claim-level evidence found in retrieved sources.",
            "text_answer": fallback_text,
            "relevance": [r.to_dict() for r in relevance_results],
        }
    else:
        # Phase 3: Contradiction Detection – build claim relationship graph
        log_fn("\n🧭 Building claim relationship graph (contradiction detection)...")

        # Guardrail: avoid quadratic explosion for very large claim sets.
        # Even with many claims, we only sample a small number of pairs
        # for NLI to keep runtime reasonable.
        num_claims = len(claims)
        num_pairs = num_claims * (num_claims - 1) // 2
        max_pairs = None
        if num_pairs > 100:
            max_pairs = 50
            log_fn(
                f"⚠️  Large number of claim pairs ({num_pairs}). "
                f"Sampling {max_pairs} pairs for NLI instead of all pairs."
            )
        
        relations = build_contradiction_graph(claims, max_pairs=max_pairs)
        relations_json = serialize_relations(relations)
        log_fn(f"✅ Computed {len(relations_json)} pairwise claim relations")
        
        # Phase 4: Stance Clustering
        log_fn("\n🧩 Clustering claims into stances...")
        stance_clusters = cluster_stances(claims, relations)
        stance_clusters_json = serialize_stance_clusters(stance_clusters)
        log_fn(f"✅ Formed {len(stance_clusters_json)} stance clusters")
        
        # Phase 5: Multi-Stance Explanation & Decision Layer
        log_fn("\n🧾 Generating multi-stance explanation...")
        final_output = build_multi_stance_answer(
            question=query,
            claims=claims,
            relations=relations,
            stances=stance_clusters,
        )
        # Always provide a plain-text answer alongside the JSON.
        final_output["text_answer"] = _format_text_answer(final_output)
        final_output["relevance"] = [r.to_dict() for r in relevance_results]
    
    log_fn("\n" + "=" * 80)
    log_fn("FINAL STRUCTURED OUTPUT")
    log_fn("=" * 80)
    log_fn(json.dumps(final_output, indent=2))
    log_fn("=" * 80)
    return final_output


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        # query = "Is coffee good or bad for heart health?"
        query = "Is it better to paint stop signs red or yellow?"
    
    run(query)

