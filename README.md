# NirnAI - RAG-Powered Legal Document Review Function

<div align="center">

**Intelligent Title Review System using Retrieval-Augmented Generation**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 🎯 Overview

NirnAI is an AI-powered legal document review system designed for Indian property title verification. It uses a **two-stage LLM pipeline (Generator → Critic)** combined with **RAG (Retrieval-Augmented Generation)** to analyze merged case files and produce structured review outputs.

The system retrieves similar historical cases (precedents) to calibrate issue severity, identify acceptable variations, and ensure consistent decision-making across reviews.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔄 **Two-Stage LLM Pipeline** | Generator (high-recall) → Critic (high-precision) |
| 📚 **RAG-Powered Learning** | Semantic search across historical precedents |
| 📍 **Evidence-First** | Every issue backed by exact document snippets |
| 🔌 **Pluggable LLMs** | Supports OpenAI GPT-4 & Anthropic Claude |
| 💾 **Local Vector Store** | ChromaDB with Sentence-Transformers embeddings |

---

## 🏗️ System Architecture

```mermaid
flowchart TB
    subgraph Input["📄 INPUT"]
        A[("Merged Case JSON")]
        A1["attachments<br/>(OCR text)"]
        A2["encumbranceDetails<br/>(EC records)"]
        A3["reportJson<br/>(Maker's report)"]
        A --> A1 & A2 & A3
    end

    subgraph Extract["🔍 EXTRACTION"]
        B["Case Fingerprint<br/>State | District | SRO | Survey"]
        C["Key Fields<br/>Parties | Extent | Boundaries"]
        D["EC Summary<br/>Transaction History"]
    end

    subgraph RAG["🧠 RAG ENGINE"]
        E[("ChromaDB<br/>Vector Store")]
        F["Semantic Search<br/>Top-K Chunks"]
        G["Consolidate<br/>Top-N Precedents"]
        E --> F --> G
    end

    subgraph LLM["🤖 TWO-STAGE LLM"]
        H["Stage 1: GENERATOR<br/>High-Recall Issue Detection"]
        I["Stage 2: CRITIC<br/>Precision + Severity Calibration"]
        H --> I
    end

    subgraph Output["📋 OUTPUT"]
        J[("Review Object")]
        J1["Issues List"]
        J2["Evidence"]
        J3["Recommendation"]
        J --> J1 & J2 & J3
    end

    A1 & A2 & A3 --> B & C & D
    B --> F
    G --> H
    C & D --> H
    I --> J

    style Input fill:#e1f5fe
    style Extract fill:#fff3e0
    style RAG fill:#f3e5f5
    style LLM fill:#e8f5e9
    style Output fill:#fce4ec
```

---

## 🔄 Two-Stage Review Pipeline

The system uses a **Generator → Critic** architecture to balance recall and precision:

```mermaid
flowchart LR
    subgraph Stage1["🎯 STAGE 1: GENERATOR"]
        direction TB
        G1["Input: Case Extract + Precedents"]
        G2["Task: Find ALL potential issues"]
        G3["Output: High-recall candidate list"]
        G1 --> G2 --> G3
    end

    subgraph Stage2["✅ STAGE 2: CRITIC"]
        direction TB
        C1["Input: Generator issues + Evidence"]
        C2["Tasks:<br/>• Validate evidence<br/>• Remove duplicates<br/>• Calibrate severity<br/>• Check precedents"]
        C3["Output: Final reviewed issues"]
        C1 --> C2 --> C3
    end

    Stage1 -->|"Candidate Issues"| Stage2

    style Stage1 fill:#fff9c4
    style Stage2 fill:#c8e6c9
```

### Why Two Stages?

| Stage | Goal | Trade-off |
|-------|------|-----------|
| **Generator** | Don't miss any issues | May include false positives |
| **Critic** | Only valid, evidence-backed issues | Filters noise, calibrates severity |

---

## 📚 RAG Retrieval Process

```mermaid
flowchart TB
    subgraph Ingestion["📥 PRECEDENT INGESTION"]
        P1["Historical Case JSON"]
        P2["Chunk into segments:<br/>• fingerprint<br/>• key_fields<br/>• ec_summary<br/>• review_notes<br/>• exceptions<br/>• flow_of_title"]
        P3["Embed with<br/>Sentence-Transformers"]
        P4[("Store in<br/>ChromaDB")]
        P1 --> P2 --> P3 --> P4
    end

    subgraph Retrieval["🔍 RETRIEVAL (at review time)"]
        R1["New Case Fingerprint"]
        R2["Vector Similarity Search<br/>(k=8 chunks)"]
        R3["Group by case_id"]
        R4["Return Top-N Precedents<br/>(n=5 cases)"]
        R1 --> R2 --> R3 --> R4
    end

    P4 -.->|"Query"| R2

    style Ingestion fill:#e3f2fd
    style Retrieval fill:#fce4ec
```

### Chunking Strategy

Each precedent is split into **6 semantic chunks** for granular retrieval:

```mermaid
graph LR
    subgraph Precedent["📁 Precedent JSON"]
        A["fingerprint<br/><small>Location identifiers</small>"]
        B["key_fields<br/><small>Parties, extent</small>"]
        C["ec_summary<br/><small>Transaction history</small>"]
        D["review_notes<br/><small>Reviewer observations</small>"]
        E["exceptions<br/><small>Accepted variations</small>"]
        F["flow_of_title<br/><small>Ownership chain</small>"]
    end

    style A fill:#bbdefb
    style B fill:#c8e6c9
    style C fill:#fff9c4
    style D fill:#ffccbc
    style E fill:#e1bee7
    style F fill:#b2dfdb
```

---

## 📊 Data Flow

```mermaid
sequenceDiagram
    participant User
    participant Pipeline as ReviewPipeline
    participant Utils as utils.py
    participant RAG as ChromaDB
    participant Gen as Generator LLM
    participant Crit as Critic LLM

    User->>Pipeline: merged_case.json
    Pipeline->>Utils: Extract fingerprint & key fields
    Utils-->>Pipeline: CURRENT_CASE_EXTRACT
    
    Pipeline->>RAG: Query similar precedents
    RAG-->>Pipeline: Top 5 precedents
    
    Pipeline->>Gen: Case extract + Precedents + SOP
    Gen-->>Pipeline: Candidate issues (high recall)
    
    Pipeline->>Crit: Candidates + Evidence + Precedents
    Crit-->>Pipeline: Validated issues (high precision)
    
    Pipeline-->>User: REVIEW_OBJECT.json
```

---

## 🚀 Installation

### Prerequisites

- Python 3.11+
- Conda (recommended) or pip

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/mehardeep-wysbryx/NirnAI-Review-RAG-Model.git
   cd NirnAI-Review-RAG-Model
   ```

2. **Create conda environment**
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
   # For OpenAI
   OPENAI_API_KEY=sk-your-openai-api-key-here
   
   # OR for Anthropic
   ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
   ```

---

## 💻 Usage

### Quick Test

```bash
python test_run.py
```

### Programmatic Usage

```python
from src.review import ReviewPipeline
import json

# Initialize pipeline
pipeline = ReviewPipeline()

# Load case
with open("examples/example_merged_case.json") as f:
    merged_case = json.load(f)

# Run review
result = pipeline.run_review(merged_case, verbose=True)
print(json.dumps(result, indent=2))
```

---

## 📁 Project Structure

```
NirnAI/
├── 📂 data/
│   └── 📂 precedents/          # Historical case JSONs for RAG
│       ├── ts_gift_deed_001.json
│       ├── ap_gift_settlement_boundary_mismatch_001.json
│       └── ...
├── 📂 examples/                 # Sample input files
├── 📂 outputs/                  # Generated review outputs
├── 📂 src/
│   ├── embeddings.py           # Embedding providers
│   ├── ingest.py               # Precedent ingestion & RAG
│   ├── llm.py                  # LLM abstraction layer
│   ├── prompts.py              # Generator & Critic prompts
│   ├── review.py               # Main pipeline orchestration
│   └── utils.py                # Data extraction utilities
├── 📂 chroma_db/               # Vector store (auto-generated)
├── .env                         # API keys (create this)
├── requirements.txt
├── test_run.py                  # Quick test script
└── evaluate.py                  # Evaluation harness
```

---

## 📋 Data Formats

### Input: Merged Case JSON

```mermaid
graph TB
    subgraph MergedCase["merged_case.json"]
        A["attachments[]<br/><small>OCR output from documents</small>"]
        B["encumbranceDetails{}<br/><small>EC transaction records</small>"]
        C["reportJson{}<br/><small>Maker's structured report</small>"]
    end
    
    style A fill:#e3f2fd
    style B fill:#fff3e0
    style C fill:#e8f5e9
```

### Output: Review Object

```json
{
  "case_id": "review_test_case_20260123",
  "timestamp": "2026-01-23T14:30:22",
  "issues": [
    {
      "id": "ISS-001",
      "category": "BOUNDARY_MISMATCH",
      "severity": "MEDIUM",
      "description": "East boundary differs between EC and deed",
      "evidence": {
        "source_doc": "Deed shows: 'East: Road'",
        "report": "EC shows: 'East: Survey 456'"
      },
      "precedent_reference": "Similar in AP-2025-002"
    }
  ],
  "summary": "Review identified 2 issues...",
  "recommendation": "HOLD"
}
```

---

## ⚙️ Configuration

### RAG Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 8 | Number of chunks to retrieve |
| `n` | 5 | Number of unique precedents to return |
| `filter_state` | auto | Filter by state (e.g., "Andhra Pradesh") |

### LLM Selection

```mermaid
flowchart LR
    A{API Key?} -->|OPENAI_API_KEY| B["GPT-4"]
    A -->|ANTHROPIC_API_KEY| C["Claude"]
    
    style B fill:#74b9ff
    style C fill:#fd79a8
```

---

## 📈 How It Learns

```mermaid
mindmap
  root((NirnAI<br/>Learning))
    Precedent Memory
      Past review notes
      Accepted exceptions
      Severity calibrations
    State-Specific
      AP patterns
      TS patterns
      KA patterns
    Issue Types
      Boundary mismatches
      Name variations
      Document chains
      Mortgage encumbrances
```

---

## 🧪 Evaluation

```bash
python evaluate.py
```

Metrics measured:
- Issue detection precision/recall
- Severity calibration accuracy
- Consistency with human reviewers

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

| Technology | Use |
|------------|-----|
| [ChromaDB](https://www.trychroma.com/) | Vector storage |
| [Sentence-Transformers](https://www.sbert.net/) | Local embeddings |
| OpenAI / Anthropic | LLM inference |

---

<div align="center">
  
**Built with ❤️ by Wysbryx**

</div>
