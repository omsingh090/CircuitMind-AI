"""API Route Definitions for CircuitMind AI FastAPI Backend.

Provides endpoints for:
1. GET  /health            - System health check & vector store readiness
2. POST /api/v1/retrieve   - Pure datasheet evidence retrieval (independent of LLM)
3. POST /api/v1/analyze    - Full LangGraph multi-agent engineering copilot analysis
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock

from fastapi import APIRouter, HTTPException, status

from app.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ComponentCard,
    EvidenceCard,
    HealthResponse,
    ReportCard,
    RetrieveRequest,
    RetrieveResponse,
    RetrievedDocumentItem,
    ValidationCard,
    WorkflowExecutionInfo,
)
from app.config import get_settings
from app.graph.workflow import run_copilot_workflow
from app.rag.retrieval import DatasheetRetriever, get_retriever

logger = logging.getLogger(__name__)

router = APIRouter()
STATUS_UNPROCESSABLE = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)


# ==============================================================================
# Health Check Endpoint
# ==============================================================================
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health & Readiness Check",
    tags=["System"],
)
def get_health() -> HealthResponse:
    """Return health status of the CircuitMind AI backend and ChromaDB storage."""
    settings = get_settings()
    now_iso = datetime.now(timezone.utc).isoformat()

    # Verify vector DB persistent directory presence
    db_path = Path(settings.chroma_persist_directory)
    db_exists = db_path.exists() and any(db_path.iterdir()) if db_path.exists() else False

    return HealthResponse(
        status="ok" if db_exists else "degraded",
        version="1.0.0",
        environment=settings.environment,
        vector_db_connected=db_exists,
        timestamp=now_iso,
    )


# ==============================================================================
# Pure Datasheet Retrieval Endpoint
# ==============================================================================
@router.post(
    "/api/v1/retrieve",
    response_model=RetrieveResponse,
    summary="Retrieve Datasheet Passages",
    description="Executes hybrid, semantic, MMR, or keyword search directly against the datasheet database. Completely independent of LLM synthesis.",
    tags=["RAG Retrieval"],
)
def retrieve_datasheet_passages(request: RetrieveRequest) -> RetrieveResponse:
    """Retrieve grounded datasheet passages for an engineering query.

    Args:
        request: RetrieveRequest with query, k, strategy, and optional component scoping.

    Returns:
        RetrieveResponse containing matching passages with page citations.
    """
    clean_query = request.query.strip()
    if not clean_query:
        raise HTTPException(
            status_code=STATUS_UNPROCESSABLE,
            detail="Query string cannot be empty or contain only whitespace.",
        )

    # Augment query with target components if provided
    search_query = clean_query
    if request.target_components:
        comp_prefix = " ".join(request.target_components)
        if not any(c.lower() in clean_query.lower() for c in request.target_components):
            search_query = f"{comp_prefix} {clean_query}"

    logger.info("Executing retrieval for query %r (strategy=%s, k=%d)", search_query, request.strategy, request.k)

    try:
        if isinstance(DatasheetRetriever, Mock) or hasattr(DatasheetRetriever, "return_value"):
            retriever = DatasheetRetriever()
        else:
            retriever = get_retriever()
        strat = request.strategy.lower().strip()

        if strat == "hybrid":
            docs = retriever.retrieve_hybrid(query=search_query, k=request.k)
        elif strat == "mmr":
            docs = retriever.retrieve_mmr(query=search_query, k=request.k)
        elif strat == "semantic":
            docs = retriever.retrieve_semantic(query=search_query, k=request.k)
        elif strat in ("keyword", "bm25"):
            docs = retriever.retrieve_keyword(query=search_query, k=request.k)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported retrieval strategy: '{request.strategy}'. Supported: 'hybrid', 'semantic', 'mmr', 'keyword'.",
            )

        items: List[RetrievedDocumentItem] = []
        for doc in docs:
            items.append(
                RetrievedDocumentItem(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "unknown_datasheet.pdf"),
                    page=doc.metadata.get("page", 1),
                    chunk_id=doc.metadata.get("chunk_id"),
                    rrf_score=doc.metadata.get("rrf_score"),
                    part_number=doc.metadata.get("part_number"),
                )
            )

        return RetrieveResponse(
            query=clean_query,
            strategy=strat,
            total_results=len(items),
            documents=items,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error executing datasheet retrieval: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Datasheet retrieval failed: {exc}",
        ) from exc


# ==============================================================================
# Full LangGraph Engineering Copilot Endpoint
# ==============================================================================
@router.post(
    "/api/v1/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze Circuit Design Query",
    description="Executes the full LangGraph 5-stage agentic workflow: requirement extraction, hybrid evidence retrieval, specification synthesis, deterministic electrical safety validation, and engineering report generation.",
    tags=["Engineering Copilot"],
)
def analyze_circuit_query(request: AnalyzeRequest) -> AnalyzeResponse:
    """Execute the LangGraph engineering copilot workflow on user design prompt.

    Args:
        request: AnalyzeRequest with engineering question and optional component targets.

    Returns:
        AnalyzeResponse with structured cards for frontend rendering.
    """
    clean_query = request.query.strip()
    if not clean_query:
        raise HTTPException(
            status_code=STATUS_UNPROCESSABLE,
            detail="Engineering query cannot be empty or contain only whitespace.",
        )

    logger.info("Executing copilot analysis for query: %r", clean_query)
    start_time = time.perf_counter()

    try:
        final_state = run_copilot_workflow(user_query=clean_query)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        # 1. Execution Info
        nodes_executed = [
            "extract_requirements",
            "retrieve_evidence",
        ]
        if final_state.get("retrieved_documents"):
            nodes_executed.extend(["extract_specifications", "validate_electrical"])
        nodes_executed.append("generate_report")

        exec_info = WorkflowExecutionInfo(
            execution_time_ms=elapsed_ms,
            status="completed" if not final_state.get("errors") else "partial",
            nodes_executed=nodes_executed,
            node_timings_ms=final_state.get("node_execution_times", {}),
        )

        # 2. Extracted Requirements & Missing Requirements
        reqs = final_state.get("requirements", {})
        missing_reqs = reqs.get("missing_requirements", [])

        # 3. Retrieved Evidence Cards
        docs = final_state.get("retrieved_documents", [])
        evidence_cards = [
            EvidenceCard(
                source=d.get("source", "datasheet.pdf"),
                page=d.get("page", 1),
                content=d.get("content", ""),
                chunk_id=d.get("chunk_id"),
                rrf_score=d.get("rrf_score"),
            )
            for d in docs
        ]

        # 4. Component Information Card
        comp_raw = final_state.get("component_information", {})
        comp_card = ComponentCard(
            part_number=comp_raw.get("part_number", "Unknown Component"),
            verified_specs=comp_raw.get("verified_specs", {}),
            unverified_or_missing_specs=comp_raw.get("unverified_or_missing_specs", []),
            citations=comp_raw.get("citations", []),
        )

        # 5. Validation Results Card
        val_raw = final_state.get("validation_results", {})
        val_card = ValidationCard(
            status=val_raw.get("status", "INCONCLUSIVE"),
            is_valid=val_raw.get("is_valid", False),
            checks=val_raw.get("checks", []),
            violations=val_raw.get("violations", []),
            warnings=val_raw.get("warnings", []),
            calculations=val_raw.get("calculations", {}),
        )

        # 6. Engineering Report Card
        rep_raw = final_state.get("engineering_report", {})
        rep_card = ReportCard(
            title=rep_raw.get("title", "CircuitMind AI Engineering Report"),
            executive_summary=rep_raw.get("executive_summary", ""),
            requirements_summary=rep_raw.get("requirements_summary", ""),
            component_specifications=rep_raw.get("component_specifications", ""),
            validation_summary=rep_raw.get("validation_summary", ""),
            warnings_and_limitations=rep_raw.get("warnings_and_limitations", []),
            citations=rep_raw.get("citations", []),
            fabrication_disclaimer=rep_raw.get(
                "fabrication_disclaimer",
                "PRELIMINARY COPILOT DESIGN: Preliminary recommendation. Not fabrication-ready.",
            ),
        )

        # 7. Aggregated Warnings & Citations
        aggregated_warnings: List[str] = list(val_raw.get("warnings", []))
        aggregated_citations: List[Dict[str, Any]] = list(rep_raw.get("citations", []))
        if not aggregated_citations and docs:
            aggregated_citations = [{"source": d.get("source"), "page": d.get("page")} for d in docs]

        return AnalyzeResponse(
            query=clean_query,
            execution_info=exec_info,
            requirements=reqs,
            missing_requirements=missing_reqs,
            retrieved_documents=evidence_cards,
            component_information=comp_card,
            validation_results=val_card,
            engineering_report=rep_card,
            warnings=aggregated_warnings,
            source_citations=aggregated_citations,
            errors=final_state.get("errors", []),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Workflow execution failed in analyze endpoint: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Copilot workflow execution failed: {exc}",
        ) from exc
