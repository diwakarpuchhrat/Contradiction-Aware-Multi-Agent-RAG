## Contradiction-Aware Multi-Agent RAG – Phases 1–5

### Overview

This project implements a modular, contradiction-aware RAG pipeline with explicit claim- and stance-level reasoning. The system is organised into clearly separated phases and agents so that each step is auditable and suitable for academic evaluation.

### Implemented Phases

- **Phase 1 – Web-based RAG**
  - DuckDuckGo search
  - Web page fetching and HTML cleaning
  - Text chunking
  - Embedding generation (Sentence-Transformers)
  - FAISS vector store

- **Phase 2 – Claim Extraction Agent**
  - Extracts atomic factual claims from relevant chunks
  - Outputs structured claim objects with IDs, text, type, source URL, and chunk ID

- **Phase 2.5 – Relevance Filtering Layer**
  - Two-stage relevance gate:
    - Fast embedding similarity filter
    - LLM-based binary classifier
  - Ensures only directly relevant chunks are passed to the Claim Extraction Agent

- **Phase 3 – Contradiction Detection Agent**
  - Compares claims pairwise (no self-comparisons, no duplicates)
  - Classifies relations as `"entails"`, `"contradicts"`, or `"neutral"`
  - Uses Groq LLM in strict JSON classification mode
  - Builds a claim relationship graph with:
    - `claim_id_1`
    - `claim_id_2`
    - `relation`
    - `confidence`

- **Phase 4 – Stance Clustering Agent**
  - Groups claims into stance clusters based on logical relations:
    - Entailment edges define cluster cores
    - Contradiction edges ensure opposing clusters remain separate
    - Neutral/isolated claims may attach to the closest semantic cluster or form their own stance
  - Outputs:
    - `stance_id`
    - `claims` (list of claim IDs)
    - `summary` (short, neutral stance description)
    - `source_count` (number of distinct supporting sources)

- **Phase 5 – Multi-Stance Explanation & Decision Agent**
  - Consumes:
    - User question
    - Stance clusters
    - Claim-level relations
  - Produces final structured JSON:
    - `question`
    - `stances`: list of objects with:
      - `stance_summary`
      - `supporting_sources`
      - `claim_count`
    - `consensus_status`: `"consensus"` or `"disagreement"`
    - `recommended_position`
  - Behaviour:
    - Explicitly states when there is no single consensus
    - Never hides minority stances
    - Suggests a recommended position only when one stance is clearly better supported (or user-preferred)

### Installation

```bash
pip install -r requirements.txt
```

Set up your Groq API key either in `config.py` or via the environment:

```bash
export GROQ_API_KEY="your_api_key_here"
```

### Running an End-to-End Example

From the `carag/` directory:

```bash
python main.py "Is coffee good or bad for heart health?"
```

The system executes the full pipeline:

1. **Web Retrieval (Phase 1)** – DuckDuckGo → fetch → clean → chunk → embed → FAISS retrieval  
2. **Relevance Filter (Phase 2.5)** – Two-stage relevance gating  
3. **Claim Extraction (Phase 2)** – Atomic, source-grounded claims with IDs  
4. **Contradiction Detection (Phase 3)** – Pairwise claim relations (entails/contradicts/neutral)  
5. **Stance Clustering (Phase 4)** – Stance clusters with summaries and source counts  
6. **Explanation & Decision (Phase 5)** – Final JSON with consensus status and recommended position  

Example (truncated) final output shape:

```json
{
  "question": "Is coffee good or bad for heart health?",
  "stances": [
    {
      "stance_summary": "Several sources state that moderate coffee consumption is associated with a reduced risk of heart disease.",
      "supporting_sources": "https://example.com/article1, https://example.com/article2",
      "claim_count": "5"
    },
    {
      "stance_summary": "Some sources warn that high coffee intake can increase blood pressure and may be risky for certain heart patients.",
      "supporting_sources": "https://example.com/article3",
      "claim_count": "3"
    }
  ],
  "consensus_status": "disagreement",
  "recommended_position": "Based on the number of distinct supporting sources, the best-supported stance is: Several sources state that moderate coffee consumption is associated with a reduced risk of heart disease."
}
```

All intermediate agents (relevance filter, claim extractor, contradiction detector, stance clusterer, explanation agent) are implemented as separate, modular Python modules under `agents/`, ensuring strict separation of concerns and source traceability.

