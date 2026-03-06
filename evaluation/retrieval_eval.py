import json
import os
from typing import Dict, List, Tuple, Any


# Ensure the existing carag source directory is on sys.path so we can
# reuse the exact same embedding + vector store components as the main pipeline.
import sys

import numpy as np
from sentence_transformers import CrossEncoder

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CARAG_SRC = os.path.join(PROJECT_ROOT, "carag", "carag")
if CARAG_SRC not in sys.path:
    sys.path.insert(0, CARAG_SRC)

from embeddings.embed import embed_texts  # type: ignore  # noqa: E402
from vectorstore.store import VectorStore  # type: ignore  # noqa: E402
from config import TOP_K  # type: ignore  # noqa: E402


# Reranker configuration
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker_model: CrossEncoder | None = None


# FAISS search configuration
FAISS_STAGE1_K = 10  # Top-K from FAISS before reranking
L2_DISTANCE_THRESHOLD = 1.2  # Discard chunks with L2 distance above this threshold


def _get_reranker() -> CrossEncoder:
    """
    Lazy-load the cross-encoder reranker used to score (query, chunk) pairs.
    """
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker_model


def _load_retrieval_dataset() -> Dict:
    """
    Load retrieval evaluation dataset.

    Expected JSON structure:
    {
      "chunks": [
        {"id": "chunk_1", "text": "..."},
        ...
      ],
      "queries": [
        {
          "id": "q1",
          "query": "question?",
          "relevant_chunk_ids": ["chunk_1", "chunk_3"]
        },
        ...
      ]
    }
    """
    dataset_path = os.path.join(PROJECT_ROOT, "evaluation_dataset", "retrieval_queries.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_vector_store(chunks: List[Dict]) -> Tuple[VectorStore, List[str]]:
    """
    Build a FAISS-based vector store from the given chunks.

    Returns:
        (store, chunk_ids)
        - store: VectorStore instance containing all chunk embeddings
        - chunk_ids: List mapping index position -> chunk_id
    """
    texts = [c["text"] for c in chunks]
    chunk_ids = [c["id"] for c in chunks]

    embeddings = embed_texts(texts)
    if len(embeddings) == 0:
        raise RuntimeError("No embeddings were generated for retrieval evaluation.")

    dim = len(embeddings[0])
    store = VectorStore(dim)

    metadata = [{"chunk_id": cid} for cid in chunk_ids]
    store.add(embeddings, texts, metadata)
    return store, chunk_ids


def _faiss_search_with_scores(
    store: VectorStore,
    query_embedding: Any,
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Perform a FAISS search that exposes both distances and metadata.
    This keeps the core VectorStore interface unchanged while allowing
    evaluation code to apply distance-based filtering.
    """
    if store.index.ntotal == 0:
        return []

    query_array = np.array([query_embedding], dtype="float32")
    k = min(top_k, store.index.ntotal)
    distances, indices = store.index.search(query_array, k)

    results: List[Dict[str, Any]] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(store.texts):
            meta = store.meta[idx]
            text = store.texts[idx]
            results.append(
                {
                    "text": text,
                    "meta": meta,
                    "distance": float(dist),
                    "chunk_id": meta.get("chunk_id"),
                }
            )
    return results


def _rerank_chunks(
    query_text: str,
    candidates: List[Dict[str, Any]],
    final_k: int,
) -> List[Dict[str, Any]]:
    """
    Rerank retrieved chunks using a cross-encoder relevance model and
    return the top-k results.
    """
    if not candidates:
        return []

    reranker = _get_reranker()
    pairs = [(query_text, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    # Sort by reranker score (descending) and keep top-k
    candidates_sorted = sorted(
        candidates,
        key=lambda x: x.get("rerank_score", 0.0),
        reverse=True,
    )
    return candidates_sorted[:final_k]


def _precision_recall_at_k(
    relevant_ids: List[str],
    retrieved_ids: List[str],
    k: int,
) -> Tuple[float, float]:
    """
    Compute Precision@K and Recall@K.

    Precision@K = (# relevant retrieved chunks) / K
    Recall@K    = (# relevant retrieved chunks) / (# total relevant chunks)
    """
    if k <= 0:
        return 0.0, 0.0

    relevant_set = set(relevant_ids)
    retrieved_at_k = retrieved_ids[:k]
    retrieved_set = set(retrieved_at_k)

    num_relevant_retrieved = len(relevant_set.intersection(retrieved_set))
    precision = num_relevant_retrieved / float(k)
    recall = num_relevant_retrieved / float(len(relevant_set)) if relevant_set else 0.0
    return precision, recall


def run_retrieval_evaluation(k: int = TOP_K) -> Tuple[float, float]:
    """
    Run retrieval evaluation over all queries in the dataset.

    Returns:
        (precision_at_k, recall_at_k) averaged across queries.
    """
    data = _load_retrieval_dataset()
    chunks = data.get("chunks", [])
    queries = data.get("queries", [])

    if not chunks or not queries:
        raise RuntimeError("Retrieval evaluation dataset is empty or malformed.")

    store, _ = _build_vector_store(chunks)

    precisions: List[float] = []
    recalls: List[float] = []

    # Stage-1 FAISS depth (before reranking) is at least FAISS_STAGE1_K
    stage1_k = max(k * 2, FAISS_STAGE1_K)

    for q in queries:
        query_text = q["query"]
        relevant_ids = q.get("relevant_chunk_ids", [])

        query_embedding = embed_texts([query_text])[0]

        # Stage 1: FAISS retrieval with distances
        faiss_results = _faiss_search_with_scores(store, query_embedding, top_k=stage1_k)

        # Stage 2: distance-based filtering
        filtered = [
            r for r in faiss_results if r.get("distance", float("inf")) <= L2_DISTANCE_THRESHOLD
        ]

        # If everything was filtered out, fall back to unfiltered FAISS results
        if not filtered:
            filtered = faiss_results

        # Stage 3: Cross-encoder reranking, keep final top-k
        reranked = _rerank_chunks(query_text, filtered, final_k=k)

        retrieved_ids = [
            r.get("chunk_id")
            for r in reranked
            if r.get("chunk_id") is not None
        ]

        p_at_k, r_at_k = _precision_recall_at_k(relevant_ids, retrieved_ids, k)
        precisions.append(p_at_k)
        recalls.append(r_at_k)

    # Macro-average across queries
    precision_avg = sum(precisions) / len(precisions) if precisions else 0.0
    recall_avg = sum(recalls) / len(recalls) if recalls else 0.0
    return precision_avg, recall_avg


def print_retrieval_results(k: int = TOP_K) -> None:
    """Helper to run and pretty-print retrieval evaluation results."""
    precision, recall = run_retrieval_evaluation(k=k)
    print("Retrieval Evaluation")
    print("--------------------")
    print(f"Precision@{k}: {precision:.2f}")
    print(f"Recall@{k}: {recall:.2f}")


if __name__ == "__main__":
    print_retrieval_results()

