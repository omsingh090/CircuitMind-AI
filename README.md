# CircuitMind AI - Agentic AI Electronics Engineering Copilot

CircuitMind AI is an agentic AI copilot designed to assist electronics hardware engineers with requirement analysis, component datasheet search, technical specification extraction, electrical compatibility verification, circuit connection planning, and engineering report generation with source citations.

---

## Architecture Principles

1. **Modular Codebase:** Separation of concerns across API, agents, graph orchestration, RAG, tools, services, and frontend.
2. **Controlled Orchestration:** State-driven agent orchestration powered by LangGraph.
3. **Deterministic Electrical Validation:** Direct Python calculation and deterministic logic for electrical checks (voltage tolerances, current limits, pin compatibility); never relying on probabilistic LLM guesses for safety-critical checks.
4. **Zero Specification Hallucination:** Strict sourcing requirement with page citations for all extracted component specs.
5. **Structured Outputs:** Pydantic schemas enforce type safety and strict schema validation throughout the pipeline.
6. **Engineering Integrity:** Preliminary plans and calculations require physical engineering verification; designs are never marked fabrication-ready automatically.

---

## Directory Structure

```text
CircuitMind-AI/
├── app/                        # Application source code
│   ├── api/                    # FastAPI routes, routers, and request/response models
│   ├── agents/                 # Specialized domain AI agents (search, spec extractor, comparator, validator)
│   ├── graph/                  # LangGraph workflows, state graph schemas, and execution nodes
│   ├── rag/                    # Datasheet ingestion, chunking, embeddings, and vector retrieval
│   ├── tools/                  # Deterministic electrical validation and engineering tools
│   └── services/               # Core application services, report generation, and orchestration
├── data/                       # Data storage
│   ├── raw/                    # Raw electronic component datasheets (PDFs)
│   └── processed/              # Processed chunks, metadata caches, and vector databases (ChromaDB)
├── evaluation/                 # Benchmark datasets, evaluation harnesses, and metrics
│   └── results/                # Evaluation benchmark runs and test outputs
├── tests/                      # Unit, integration, and end-to-end test suites
├── frontend/                   # Streamlit web application interface
├── scripts/                    # Utility scripts (datasheet downloaders, indexing scripts, db migration)
├── .env.example                # Environment variable template
├── .gitignore                  # Git ignore rules for virtualenvs, caches, and secrets
├── PROJECT_RULES.md            # Project rules, engineering guidelines, and architecture standards
├── README.md                   # Project overview, directory guide, and quickstart documentation
└── requirements.txt            # Python dependencies
```

---

## Technology Stack

- **Runtime:** Python 3.10+
- **API Framework:** FastAPI, Uvicorn
- **Agent Orchestration:** LangChain, LangGraph
- **Vector Search & RAG:** ChromaDB, PyMuPDF (`fitz`), Hugging Face Embeddings (`all-MiniLM-L6-v2`)
- **LLM Integrations:** OpenAI API
- **Data Validation:** Pydantic v2
- **Frontend UI:** Streamlit
- **Testing:** Pytest, pytest-asyncio, HTTPX
- **Deployment:** Docker, Docker Compose

---

## Development Milestones

- [ ] **Milestone 1:** Datasheet ingestion and retrieval pipeline (PyMuPDF parser, chunker, embeddings, ChromaDB vector store).
- [ ] **Milestone 2:** RAG-based technical question answering with source citations.
- [ ] **Milestone 3:** Specialized AI agents (Specification Extractor, Component Comparator, Electrical Validator).
- [ ] **Milestone 4:** LangGraph orchestration for end-to-end engineering workflows.
- [ ] **Milestone 5:** FastAPI backend endpoints and Streamlit interactive dashboard.
- [ ] **Milestone 6:** Evaluation harness, Docker containerization, and production deployment.

---

## Quickstart Setup

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. Environment Setup
```bash
# Clone the repository
git clone <repo-url>
cd CircuitMind-AI

# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
```bash
cp .env.example .env
# Edit .env with your Gemini API key and local configurations
```

### 4. Running the Backend API
```bash
# Start FastAPI backend server on port 8000
./venv/Scripts/python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --reload
```

API Documentation:
- Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`
- Circuit analysis endpoint: `POST http://127.0.0.1:8000/api/v1/analyze`
- Passage retrieval endpoint: `POST http://127.0.0.1:8000/api/v1/retrieve`

### 5. Running the Streamlit Frontend
```bash
# Start Streamlit application
./venv/Scripts/python.exe -m streamlit run frontend/app.py
```
Access the application at `http://localhost:8501`.

The Streamlit interface acts as a Hybrid Engineering Copilot featuring:
- **Engineering Workspace:** Interactive prompt console with example engineering query presets and analysis history.
- **Agent Monitor:** Live execution telemetry for the 5 LangGraph nodes with millisecond timings.
- **Datasheet Explorer:** Evidence viewer for retrieved datasheet chunks with PDF source citations and RRF scores, plus interactive direct retrieval search.
- **Validation Center:** Visual electrical compatibility dashboard for `VALID`, `INVALID`, and `INCONCLUSIVE` statuses with deterministic physics calculations and warnings.
- **Engineering Reports:** Structured markdown report viewer with datasheet-grounded specifications, source references table, and export capability.

### 6. Running the Test Suite
```bash
# Run all unit and integration tests
./venv/Scripts/python.exe -m pytest tests/ -v
```


