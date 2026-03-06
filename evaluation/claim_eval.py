import json
import os
import re
from typing import Dict, List, Tuple, Any


import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CARAG_SRC = os.path.join(PROJECT_ROOT, "carag", "carag")
if CARAG_SRC not in sys.path:
    sys.path.insert(0, CARAG_SRC)

from agents.claim_extractor import extract_claims_from_chunk  # type: ignore  # noqa: E402
from embeddings.embed import embed_texts  # type: ignore  # noqa: E402
import numpy as np


# Similarity threshold above which a predicted claim is considered
# a correct match for a gold claim.
CLAIM_SIMILARITY_THRESHOLD: float = 0.80


def _load_claim_dataset() -> List[Dict]:
    """
    Load claim extraction evaluation dataset.

    Expected JSON structure:
    [
      {
        "id": "example_1",
        "text_chunk": "...",
        "true_claims": ["claim a", "claim b"]
      },
      ...
    ]
    """
    path = os.path.join(PROJECT_ROOT, "evaluation_dataset", "claim_dataset.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalise_claim(text: str) -> str:
    """
    Normalise claim text with lightweight text cleaning:

    - Lowercasing
    - Removing punctuation
    - Collapsing repeated whitespace
    """
    # Lowercase and strip
    text = text.lower().strip()
    # Remove punctuation characters
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple spaces
    return " ".join(text.split())


def _compute_precision_recall_f1(
    correct: int,
    predicted: int,
    gold: int,
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 for claim extraction.

    Precision = correct_extracted_claims / total_extracted_claims
    Recall    = correct_extracted_claims / total_true_claims
    F1        = 2 * (precision * recall) / (precision + recall)
    """
    precision = correct / predicted if predicted > 0 else 0.0
    recall = correct / gold if gold > 0 else 0.0
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def _cosine_sim_matrix(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity matrix between two 2D arrays.

    a: shape (n_pred, dim)
    b: shape (n_gold, dim)
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=float)

    # Normalise rows
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.matmul(a_norm, b_norm.T)


def _greedy_semantic_match(
    predicted_texts: List[str],
    gold_texts: List[str],
    similarity_threshold: float = CLAIM_SIMILARITY_THRESHOLD,
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Greedy one-to-one semantic matching between predicted and gold claims.

    Returns:
        (num_correct_matches, match_details)

        match_details: list of dicts with keys:
            - pred_text
            - gold_text
            - similarity
    """
    if not predicted_texts or not gold_texts:
        return 0, []

    # Embed all claims in one batch for efficiency
    all_texts = [*predicted_texts, *gold_texts]
    embeddings = embed_texts(all_texts)
    if len(embeddings) == 0:
        return 0, []

    embeddings_pred = np.array(embeddings[: len(predicted_texts)], dtype="float32")
    embeddings_gold = np.array(embeddings[len(predicted_texts) :], dtype="float32")

    sim_matrix = _cosine_sim_matrix(embeddings_pred, embeddings_gold)

    num_pred, num_gold = sim_matrix.shape
    unmatched_pred = set(range(num_pred))
    unmatched_gold = set(range(num_gold))

    matches: List[Dict[str, Any]] = []
    correct = 0

    # Greedy: repeatedly pick the highest remaining similarity above threshold
    while unmatched_pred and unmatched_gold:
        # Restrict to unmatched indices
        submatrix = sim_matrix[list(unmatched_pred)][:, list(unmatched_gold)]
        if submatrix.size == 0:
            break

        max_idx = np.unravel_index(np.argmax(submatrix), submatrix.shape)
        max_sim = float(submatrix[max_idx])
        if max_sim < similarity_threshold:
            break

        pred_global_idx = list(unmatched_pred)[max_idx[0]]
        gold_global_idx = list(unmatched_gold)[max_idx[1]]

        unmatched_pred.remove(pred_global_idx)
        unmatched_gold.remove(gold_global_idx)

        correct += 1
        matches.append(
            {
                "pred_text": predicted_texts[pred_global_idx],
                "gold_text": gold_texts[gold_global_idx],
                "similarity": max_sim,
            }
        )

    return correct, matches


def run_claim_extraction_evaluation(
    return_debug: bool = False,
) -> Tuple[float, float, float, List[Dict[str, Any]], List[str], List[str]]:
    """
    Evaluate the claim extraction agent against the dataset.

    Returns:
        (precision, recall, f1) aggregated across all examples.
    """
    dataset = _load_claim_dataset()

    total_correct = 0
    total_predicted = 0
    total_gold = 0

    # For detailed debug output
    matched_details: List[Dict[str, Any]] = []
    unmatched_predicted_all: List[str] = []
    missing_gold_all: List[str] = []

    for idx, example in enumerate(dataset):
        chunk_text = example["text_chunk"]
        true_claims = example.get("true_claims", [])

        # Use a deterministic dummy source / chunk id; evaluation only cares about text.
        predicted_claims = extract_claims_from_chunk(
            chunk_text=chunk_text,
            source_url="evaluation_dataset",
            chunk_id=f"eval_chunk_{idx}",
        )

        predicted_texts_raw = [c.claim_text for c in predicted_claims]
        gold_texts_raw = [str(c) for c in true_claims]

        # Apply normalisation (lowercasing, punctuation removal, whitespace)
        predicted_texts = [_normalise_claim(t) for t in predicted_texts_raw]
        gold_texts = [_normalise_claim(t) for t in gold_texts_raw]

        # Semantic greedy matching
        correct, matches = _greedy_semantic_match(predicted_texts, gold_texts)

        total_correct += correct
        total_predicted += len(predicted_texts)
        total_gold += len(gold_texts)

        if return_debug:
            matched_details.extend(matches)

            matched_pred_texts = {m["pred_text"] for m in matches}
            matched_gold_texts = {m["gold_text"] for m in matches}

            unmatched_predicted = [
                t for t in predicted_texts if t not in matched_pred_texts
            ]
            missing_gold = [t for t in gold_texts if t not in matched_gold_texts]

            unmatched_predicted_all.extend(unmatched_predicted)
            missing_gold_all.extend(missing_gold)

    precision, recall, f1 = _compute_precision_recall_f1(
        total_correct,
        total_predicted,
        total_gold,
    )

    if return_debug:
        return precision, recall, f1, matched_details, unmatched_predicted_all, missing_gold_all

    # For backward compatibility if called without expecting debug outputs.
    return precision, recall, f1, [], [], []


def print_claim_extraction_results() -> None:
    """Helper to run and pretty-print claim extraction evaluation results."""
    (
        precision,
        recall,
        f1,
        matched_details,
        unmatched_predicted,
        missing_gold,
    ) = run_claim_extraction_evaluation(return_debug=True)

    print("Claim Extraction Evaluation")
    print("---------------------------")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print()
    print("Correctly matched claims")
    print("------------------------")
    if not matched_details:
        print("(none)")
    else:
        for m in matched_details:
            sim = m.get("similarity", 0.0)
            print(f"Predicted: {m['pred_text']}")
            print(f"Gold:      {m['gold_text']}")
            print(f"Similarity: {sim:.3f}")
            print()

    print("Unmatched predicted claims")
    print("--------------------------")
    if not unmatched_predicted:
        print("(none)")
    else:
        for t in unmatched_predicted:
            print(f"- {t}")

    print()
    print("Missing ground truth claims")
    print("---------------------------")
    if not missing_gold:
        print("(none)")
    else:
        for t in missing_gold:
            print(f"- {t}")


if __name__ == "__main__":
    print_claim_extraction_results()

