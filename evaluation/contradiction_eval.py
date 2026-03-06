import json
import os
from typing import Dict, List, Tuple


import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CARAG_SRC = os.path.join(PROJECT_ROOT, "carag", "carag")
if CARAG_SRC not in sys.path:
    sys.path.insert(0, CARAG_SRC)

from agents.claim_extractor import Claim  # type: ignore  # noqa: E402
from agents.contradiction_detector import (  # type: ignore  # noqa: E402
    _classify_relation_with_llm,
    RelationLabel,
)


LABELS: List[RelationLabel] = ["entails", "contradicts", "neutral"]


def _load_contradiction_dataset() -> List[Dict]:
    """
    Load contradiction/NLI evaluation dataset.

    Expected JSON structure:
    [
      {
        "id": "pair_1",
        "claim_1": "...",
        "claim_2": "...",
        "true_label": "entails" | "contradicts" | "neutral"
      },
      ...
    ]
    """
    path = os.path.join(PROJECT_ROOT, "evaluation_dataset", "contradiction_pairs.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _compute_multiclass_metrics(
    y_true: List[RelationLabel],
    y_pred: List[RelationLabel],
) -> Tuple[float, float, float, float]:
    """
    Compute accuracy, macro-averaged precision, recall, and F1
    for a three-way NLI classifier.
    """
    assert len(y_true) == len(y_pred)
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Overall accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / float(n)

    # Per-class counts
    tp: Dict[RelationLabel, int] = {l: 0 for l in LABELS}
    fp: Dict[RelationLabel, int] = {l: 0 for l in LABELS}
    fn: Dict[RelationLabel, int] = {l: 0 for l in LABELS}

    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    def safe_div(a: float, b: float) -> float:
        return a / b if b > 0 else 0.0

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for label in LABELS:
        precision_l = safe_div(tp[label], tp[label] + fp[label])
        recall_l = safe_div(tp[label], tp[label] + fn[label])
        if precision_l + recall_l == 0.0:
            f1_l = 0.0
        else:
            f1_l = 2.0 * (precision_l * recall_l) / (precision_l + recall_l)
        precisions.append(precision_l)
        recalls.append(recall_l)
        f1s.append(f1_l)

    precision_macro = sum(precisions) / len(precisions)
    recall_macro = sum(recalls) / len(recalls)
    f1_macro = sum(f1s) / len(f1s)
    return accuracy, precision_macro, recall_macro, f1_macro


def run_contradiction_evaluation() -> Tuple[float, float, float, float]:
    """
    Evaluate the NLI / contradiction detector using labelled claim pairs.

    Returns:
        (accuracy, precision_macro, recall_macro, f1_macro)
    """
    dataset = _load_contradiction_dataset()

    y_true: List[RelationLabel] = []
    y_pred: List[RelationLabel] = []

    for example in dataset:
        claim1_text = example["claim_1"]
        claim2_text = example["claim_2"]
        true_label: RelationLabel = example["true_label"]  # type: ignore[assignment]

        claim_a = Claim(
            claim_text=claim1_text,
            source_url="evaluation_dataset",
            chunk_id="eval_chunk_a",
        )
        claim_b = Claim(
            claim_text=claim2_text,
            source_url="evaluation_dataset",
            chunk_id="eval_chunk_b",
        )

        relation = _classify_relation_with_llm(claim_a, claim_b)

        y_true.append(true_label)
        y_pred.append(relation.relation)

    return _compute_multiclass_metrics(y_true, y_pred)


def print_contradiction_results() -> None:
    """Helper to run and pretty-print contradiction detection evaluation."""
    accuracy, precision, recall, f1 = run_contradiction_evaluation()
    print("Contradiction Detection Evaluation")
    print("----------------------------------")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")


if __name__ == "__main__":
    print_contradiction_results()

