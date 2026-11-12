"""Pydantic Request and Response Schemas for CircuitMind AI FastAPI Backend.

Defines the frontend-ready data contract supporting both direct API consumers
and the future Streamlit UI (chat messages, live agent execution monitors,
evidence explorer cards, specification cards, and engineering reports).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ==============================================================================
# Health Check Schemas
# ==============================================================================
class HealthResponse(BaseModel):
    """System health and operational status response."""

    status: str = Field(default="ok", description="Overall health status ('ok' or 'degraded')")
    version: str = Field(default="1.0.0", description="CircuitMind AI API version")
    environment: str = Field(default="development", description="Operational environment")
    vector_db_connected: bool = Field(default=True, description="Whether ChromaDB persistent store is available")
    timestamp: str = Field(description="ISO 8601 UTC timestamp of the health check")


# ==============================================================================
# Retrieval Schemas (/api/v1/retrieve)
# ==============================================================================
class RetrieveRequest(BaseModel):
    """Request payload for pure datasheet evidence retrieval."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Technical query string (e.g. 'LM35 supply voltage range')",
        examples=["What is the operating supply voltage range of the LM35?"],
    )
    k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of passages to retrieve",
    )
    strategy: str = Field(
        default="hybrid",
        description="Retrieval strategy: 'hybrid' (RRF), 'semantic', 'mmr', or 'keyword'",
    )
    target_components: Optional[List[str]] = Field(
        default=None,
        description="Optional list of component part numbers to scope search",
        examples=[["LM35"]],
    )


class RetrievedDocumentItem(BaseModel):
    """A single retrieved datasheet chunk with source citations and scoring."""

    content: str = Field(description="Datasheet passage text excerpt")
    source: str = Field(description="Source PDF filename (e.g. 'lm35.pdf')")
    page: Any = Field(description="Original PDF page number where passage is found")
    chunk_id: Optional[str] = Field(default=None, description="Deterministic hash chunk identifier")
    rrf_score: Optional[float] = Field(default=None, description="Reciprocal Rank Fusion or similarity score")
    part_number: Optional[str] = Field(default=None, description="Associated component part number")


class RetrieveResponse(BaseModel):
    """Response payload containing retrieved datasheet evidence."""

    query: str = Field(description="Original search query")
    strategy: str = Field(description="Retrieval strategy employed")
    total_results: int = Field(description="Number of documents returned")
    documents: List[RetrievedDocumentItem] = Field(default_factory=list, description="Retrieved passages")


# ==============================================================================
# Analysis & Copilot Schemas (/api/v1/analyze)
# ==============================================================================
class AnalyzeRequest(BaseModel):
    """Request payload for full LangGraph engineering copilot analysis."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Engineering design query or circuit prompt",
        examples=["Can I operate an LM35 temperature sensor at 5V supply with 10mA current budget?"],
    )
    target_components: Optional[List[str]] = Field(
        default=None,
        description="Optional explicit component part numbers",
        examples=[["LM35"]],
    )


class WorkflowExecutionInfo(BaseModel):
    """Execution telemetry of the LangGraph multi-agent workflow."""

    execution_time_ms: float = Field(description="Total workflow execution time in milliseconds")
    status: str = Field(default="completed", description="Execution status: 'completed', 'partial', 'error'")
    nodes_executed: List[str] = Field(
        default_factory=list,
        description="Chronological sequence of agent nodes executed",
    )
    node_timings_ms: Dict[str, float] = Field(
        default_factory=dict,
        description="Execution duration in milliseconds per workflow node",
    )


class EvidenceCard(BaseModel):
    """Frontend-ready card representing a retrieved datasheet passage."""

    source: str = Field(description="Datasheet PDF filename")
    page: Any = Field(description="Page number in datasheet")
    content: str = Field(description="Passage text content")
    chunk_id: Optional[str] = Field(default=None, description="Unique chunk ID")
    rrf_score: Optional[float] = Field(default=None, description="RRF relevance score")


class ComponentCard(BaseModel):
    """Frontend-ready card for verified component specifications and unverified parameters."""

    part_number: str = Field(description="Component identifier")
    verified_specs: Dict[str, Any] = Field(default_factory=dict, description="Verified specs with citations")
    unverified_or_missing_specs: List[str] = Field(
        default_factory=list,
        description="Parameters not found in retrieved datasheet evidence",
    )
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="Source citations")


class ValidationCard(BaseModel):
    """Frontend-ready card for deterministic electrical validation results."""

    status: str = Field(
        default="INCONCLUSIVE",
        description="Validation status: 'VALID', 'INVALID', or 'INCONCLUSIVE'",
    )
    is_valid: bool = Field(default=False, description="Whether the circuit design passed all deterministic safety checks")
    checks: List[Dict[str, Any]] = Field(default_factory=list, description="List of individual checks (PASS/FAIL)")
    violations: List[str] = Field(default_factory=list, description="Critical electrical safety violations")
    warnings: List[str] = Field(default_factory=list, description="Engineering warnings and derating notices")
    calculations: Dict[str, Any] = Field(
        default_factory=dict,
        description="Deterministic calculations (e.g. quiescent power dissipation)",
    )


class ReportCard(BaseModel):
    """Frontend-ready card for the structured engineering report."""

    title: str = Field(description="Report title")
    executive_summary: str = Field(description="High-level engineering evaluation summary")
    requirements_summary: str = Field(description="Stated and missing requirements summary")
    component_specifications: str = Field(description="Verified component specifications with page citations")
    validation_summary: str = Field(description="Deterministic electrical validation summary")
    warnings_and_limitations: List[str] = Field(default_factory=list, description="Design warnings and limitations")
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="Datasheet citations")
    fabrication_disclaimer: str = Field(description="Mandatory engineering integrity fabrication disclaimer")


class AnalyzeResponse(BaseModel):
    """Complete structured response from the CircuitMind AI engineering copilot."""

    query: str = Field(description="The user's original circuit inquiry")
    execution_info: WorkflowExecutionInfo = Field(description="Workflow execution timing and telemetry")
    requirements: Dict[str, Any] = Field(default_factory=dict, description="Extracted quantitative constraints")
    missing_requirements: List[str] = Field(
        default_factory=list,
        description="Explicitly identified unstated engineering parameters",
    )
    retrieved_documents: List[EvidenceCard] = Field(
        default_factory=list,
        description="Retrieved datasheet passages with citations",
    )
    component_information: ComponentCard = Field(description="Verified component specifications and citations")
    validation_results: ValidationCard = Field(description="Deterministic electrical safety validation results")
    engineering_report: ReportCard = Field(description="Final synthesized structured engineering report")
    warnings: List[str] = Field(default_factory=list, description="Engineering warnings and design caveats")
    source_citations: List[Dict[str, Any]] = Field(default_factory=list, description="Aggregated source citations")
    errors: List[str] = Field(default_factory=list, description="System notices or non-fatal execution errors")

