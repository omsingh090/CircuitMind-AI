# CircuitMind AI - Project Rules and Guidelines

## Project Objective
Build an Agentic AI Electronics Engineering Copilot to help electronics engineers:
- Understand engineering requirements.
- Search electronic component datasheets.
- Extract component specifications.
- Compare electronic components.
- Validate electrical compatibility.
- Generate preliminary circuit connection plans.
- Generate engineering reports with source citations.

## Technology Stack
- **Language & Runtime:** Python 3.10+
- **API Framework:** FastAPI, Uvicorn
- **Agent Orchestration:** LangChain, LangGraph
- **LLM & Embeddings:** Google Gemini API (`google-genai`), Hugging Face embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- **Vector Database:** ChromaDB
- **Document Ingestion:** PyMuPDF (`fitz`)
- **Data Validation & Typing:** Pydantic v2
- **Frontend UI:** Streamlit
- **Deployment & Containerization:** Docker, Docker Compose
- **Testing:** pytest, pytest-asyncio

## Architecture Principles
1. **Modular Codebase:** Maintain clean separation between API, agents, graph, RAG, tools, services, and frontend.
2. **Controlled Orchestration:** Use LangGraph for deterministic, stateful agent workflows.
3. **Deterministic Electrical Validation:** Never use LLM approximations for electrical checks (voltage levels, current draw, pin assignments); implement deterministic Python validation functions.
4. **Zero Specification Hallucination:** Never invent technical specifications. Every extracted specification must include datasheet source and page references.
5. **Structured Outputs:** Use Pydantic schemas for data validation, agent input/output, and API responses.
6. **Graceful Error Handling:** Handle missing information, corrupted PDFs, and out-of-spec queries predictably.
7. **Rigorous Testing:** Write unit and integration tests for all critical functions (parsers, validators, tools, pipelines).
8. **Secure Configuration:** Keep API keys and environment-specific settings in `.env` files; never commit credentials.
9. **Engineering Integrity:** Never claim an electrical design is fabrication-ready without proper engineering verification and physical review.

## Development Rules
- Do not generate the entire application at once.
- Implement one milestone at a time.
- Explain files before creating them.
- Do not overwrite existing code without checking it.
- Run tests after implementation.
- Avoid unnecessary dependencies.
- Follow Python best practices (PEP 8, type hints, docstrings).
- Never claim an electrical design is fabrication-ready without proper engineering verification.

## Development Milestones
- **Milestone 1:** Build a datasheet ingestion and retrieval pipeline.
- **Milestone 2:** Build RAG-based technical question answering.
- **Milestone 3:** Build specialized AI agents.
- **Milestone 4:** Build LangGraph orchestration.
- **Milestone 5:** Build FastAPI and Streamlit interfaces.
- **Milestone 6:** Evaluation, Docker and deployment.

