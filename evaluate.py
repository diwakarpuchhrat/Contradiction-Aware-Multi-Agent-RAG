"""
Master evaluation script for the contradiction-aware multi-agent RAG system.

Run:
    python evaluate.py

This will:
- Evaluate retrieval (Precision@K, Recall@K)
- Evaluate claim extraction (Precision, Recall, F1)
- Evaluate contradiction detection (Accuracy, Precision, Recall, F1)
"""

import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Ensure the existing carag source tree is available for all evaluation modules.
CARAG_SRC = os.path.join(PROJECT_ROOT, "carag", "carag")
if CARAG_SRC not in sys.path:
    sys.path.insert(0, CARAG_SRC)

# Add evaluation package path (for explicitness when running from other CWDs)
EVAL_SRC = os.path.join(PROJECT_ROOT, "evaluation")
if EVAL_SRC not in sys.path:
    sys.path.insert(0, EVAL_SRC)


from evaluation.retrieval_eval import print_retrieval_results  # type: ignore  # noqa: E402
from evaluation.claim_eval import print_claim_extraction_results  # type: ignore  # noqa: E402
from evaluation.contradiction_eval import print_contradiction_results  # type: ignore  # noqa: E402


def main() -> None:
    print("=================================")
    print("SYSTEM EVALUATION RESULTS")
    print("=================================")
    print()

    # Retrieval
    print_retrieval_results()
    print()

    # Claim extraction
    print_claim_extraction_results()
    print()

    # Contradiction detection
    print_contradiction_results()


if __name__ == "__main__":
    main()

