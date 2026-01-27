# NirnAI - RAG-Powered Legal Document Review

<div align="center">

**Intelligent Title Review System using Retrieval-Augmented Generation**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## Overview

NirnAI is an AI-powered legal document review system designed for Indian property title verification. It uses a **two-stage LLM pipeline (Generator → Critic)** combined with **RAG (Retrieval-Augmented Generation)** to analyze merged case files and produce structured review outputs.

The system retrieves similar historical cases (precedents) to calibrate issue severity, identify acceptable variations, and ensure consistent decision-making across reviews.

---

## Features

- **Two-Stage LLM Review Pipeline**
  - **Generator**: High-recall issue detection from source documents
  - **Critic**: High-precision refinement, de-duplication, and severity calibration

- **RAG-Powered Precedent Learning**
  - Semantic search across historical cases
  - State-specific filtering for relevant precedents
  - Learns from past review notes, exceptions, and accepted variations

- **Evidence-First Approach**
  - Every flagged issue requires exact snippets from source documents
  - No generic or unsubstantiated feedback

- **Flexible LLM Backend**
  - Supports OpenAI (GPT-4) and Anthropic (Claude)
  - Pluggable architecture for easy provider switching

- **Local Vector Store**
  - ChromaDB for persistent embeddings
  - Sentence-Transformers for local embeddings (no external API required)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MERGED_CASE_JSON                              │
│  (attachments + encumbranceDetails + reportJson)                        │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    CURRENT_CASE_EXTRACT                                 │
│  - Case fingerprint (state, district, SRO, survey, parties)             │
│  - Normalized key fields                                                │
│  - EC summary                                                           │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
┌───────────────────────────┐   ┌─────────────────────────────────────────┐
│     ChromaDB (RAG)        │   │          LLM Pipeline                   │
│  ┌─────────────────────┐  │   │  ┌─────────────────────────────────┐    │
│  │  Precedent Chunks   │  │──▶│  │  Stage 1: GENERATOR             │    │
│  │  - fingerprint      │  │   │  │  (High-recall issue detection)  │    │
│  │  - review_notes     │  │   │  └───────────────┬─────────────────┘    │
│  │  - exceptions       │  │   │                  │                      │
│  │  - flow_of_title    │  │   │                  ▼                      │
│  └─────────────────────┘  │   │  ┌─────────────────────────────────┐    │
└───────────────────────────┘   │  │  Stage 2: CRITIC                │    │
                                │  │  (Precision, de-dup, severity)  │    │
                                │  └───────────────┬─────────────────┘    │
                                └──────────────────┼──────────────────────┘
                                                   │
                                                   ▼
                              ┌─────────────────────────────────────────────┐
                              │              REVIEW_OBJECT                  │
                              │  {                                          │
                              │    "case_id": "...",                        │
                              │    "issues": [...],                         │
                              │    "summary": "...",                        │
                              │    "recommendation": "APPROVE|REJECT|HOLD"  │
                              │  }                                          │
                              └─────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.11+
- Conda (recommended) or pip

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/mehardeep-wysbryx/NirnAI-Review-RAG-Model.git
   cd NirnAI
   ```

2. **Create a conda environment**
   ```bash
   conda create -n nirnai python=3.11 -y
   conda activate nirnai
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Key**
   
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=sk-your-openai-api-key-here
   ```
---

## Usage

### Quick Test Run

```bash
python test_run.py
```

This will:
1. Check for API key configuration
2. Initialize the vector store with precedents
3. Retrieve similar historical cases
4. Run the two-stage review pipeline
5. Output results to `outputs/` directory

### Programmatic Usage

```python
from src.review import ReviewPipeline

# Initialize the pipeline
pipeline = ReviewPipeline()

# Load your merged case
with open("examples/example_merged_case.json") as f:
    merged_case = json.load(f)

# Run the review
result = pipeline.run_review(merged_case, verbose=True)

print(json.dumps(result, indent=2))
```

### Adding New Precedents

Place precedent JSON files in `data/precedents/`. The system will automatically ingest them on the next run.

```python
from src.ingest import PrecedentStore

store = PrecedentStore()
store.ingest_precedents("data/precedents/")
```

---

## Project Structure

```
NirnAI/
├── data/
│   └── precedents/           # Historical case JSONs for RAG
│       ├── ts_gift_deed_001.json
│       ├── ap_gift_settlement_boundary_mismatch_001.json
│       ├── ap_gift_settlement_house_no_discrepancy_001.json
│       ├── ap_sale_deed_with_link_doc_001.json
│       └── ap_partition_active_mortgage_001.json
├── examples/
│   ├── example_merged_case.json    # Sample input format
│   └── example_merged_case2.json
├── outputs/                  # Generated review outputs
├── src/
│   ├── __init__.py
│   ├── embeddings.py         # Embedding providers (SentenceTransformers, OpenAI)
│   ├── ingest.py             # Precedent ingestion and RAG retrieval
│   ├── llm.py                # LLM abstraction (OpenAI, Anthropic)
│   ├── prompts.py            # Generator and Critic prompt templates
│   ├── review.py             # Main review pipeline orchestration
│   └── utils.py              # Data extraction and normalization utilities
├── chroma_db/                # Persistent vector store (auto-generated)
├── .env                      # API keys (create this file)
├── requirements.txt
├── test_run.py               # Quick test script
├── evaluate.py               # Evaluation harness
└── README.md
```

---

## Data Formats

### Input: Merged Case JSON

```json
{
  "attachments": [
    {
      "fileName": "Sale_Deed.pdf",
      "s3Key": "...",
      "ocrOutput": "Document No: 1234/2024\nParties: JOHN DOE..."
    }
  ],
  "encumbranceDetails": {
    "encumbrance": [
      {
        "documentDetails": {
          "documentNo": "1234",
          "year": "2024",
          "deedType": "Sale Deed"
        },
        "executant": [{"name": "SELLER NAME"}],
        "claimant": [{"name": "BUYER NAME"}]
      }
    ]
  },
  "reportJson": {
    "property_address": "Survey No. 123, Village XYZ",
    "sections": [
      {
        "name": "Property Details",
        "content": "The property is located at..."
      }
    ]
  }
}
```

### Output: Review Object

```json
{
  "case_id": "review_test_case_20260123_143022",
  "timestamp": "2026-01-23T14:30:22.123456",
  "issues": [
    {
      "id": "ISS-001",
      "category": "BOUNDARY_MISMATCH",
      "severity": "MEDIUM",
      "description": "East boundary differs between EC and source deed",
      "evidence": {
        "source_doc": "Deed shows: 'East: Road'",
        "report": "EC shows: 'East: Survey 456'"
      },
      "precedent_reference": "Similar issue in AP-2025-002 was accepted"
    }
  ],
  "summary": "Review identified 2 issues requiring attention...",
  "recommendation": "HOLD"
}
```

### Precedent JSON Format

```json
{
  "case_id": "ap_sale_deed_001",
  "meta": {
    "state": "Andhra Pradesh",
    "district": "Kakinada",
    "sro": "Kakinada",
    "survey_numbers": ["123/4"],
    "deed_types": ["Sale Deed"]
  },
  "key_fields": {
    "parties": ["JOHN DOE", "JANE DOE"],
    "extent": "500 sq yards",
    "boundaries": {...}
  },
  "ec_transactions": [...],
  "review_notes": "Maker noted minor name variation...",
  "exceptions": ["Name spelling variation accepted"],
  "flow_of_title": "Property flows from A → B → C..."
}
```

---

## Configuration

### RAG Parameters

In `src/review.py`:

```python
precedents = self.precedent_store.retrieve_precedents(
    query=fingerprint,
    k=8,    # Number of chunks to retrieve
    n=5,    # Number of unique precedents to return
    filter_state=state_filter,  # State-specific filtering
)
```

### LLM Selection

The system auto-selects based on available API keys:
- If `OPENAI_API_KEY` is set → Uses GPT-4
- If `ANTHROPIC_API_KEY` is set → Uses Claude

---

## How RAG Works

1. **Ingestion**: Precedent JSONs are chunked into semantic segments:
   - `fingerprint`: Location/property identifiers
   - `key_fields`: Parties, extent, boundaries
   - `ec_summary`: Transaction history
   - `review_notes`: Past reviewer observations
   - `exceptions`: Accepted variations
   - `flow_of_title`: Ownership chain

2. **Embedding**: Each chunk is embedded using Sentence-Transformers (`all-MiniLM-L6-v2`)

3. **Retrieval**: For a new case:
   - Extract case fingerprint
   - Query ChromaDB for top-k similar chunks
   - Consolidate into top-n unique precedents
   - Inject into LLM prompts for context

4. **Learning**: The LLM uses precedent context to:
   - Calibrate issue severity
   - Identify acceptable variations
   - Apply consistent decision patterns

---

## Evaluation

Run the evaluation harness:

```bash
python evaluate.py
```

This measures:
- Issue detection precision/recall
- Severity calibration accuracy
- Consistency with human reviewers

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Built with [ChromaDB](https://www.trychroma.com/) for vector storage
- [Sentence-Transformers](https://www.sbert.net/) for local embeddings
- OpenAI GPT-4 / Anthropic Claude for LLM inference

---

<div align="center">
  <sub>Built with ❤️ by Wysbryx</sub>
</div>
