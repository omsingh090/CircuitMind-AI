"""LangGraph Workflow Orchestration for CircuitMind AI Copilot.

Coordinates specialized engineering agents into a deterministic, stateful workflow:
1. extract_requirements (RequirementAgent)
2. retrieve_evidence (ResearchAgent / Hybrid RAG)
3. extract_specifications (ComponentAgent)
4. validate_electrical (ValidationAgent / Deterministic Safety Tool)
5. generate_report (ReportAgent)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.component_agent import component_agent_node
from app.agents.report_agent import report_agent_node
from app.agents.requirement_agent import requirement_agent_node
from app.agents.research_agent import research_agent_node
from app.agents.validation_agent import validation_agent_node
from app.graph.state import CircuitMindState

logger = logging.getLogger(__name__)


def profile_node(node_name: str, node_func: Callable[[CircuitMindState], Dict[str, Any]]) -> Callable[[CircuitMindState], Dict[str, Any]]:
    """Wrap a node function to measure execution duration and persist in node_execution_times."""
    def wrapper(state: CircuitMindState) -> Dict[str, Any]:
        start = time.perf_counter()
        result = node_func(state)
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
        timings = dict(state.get("node_execution_times") or {})
        timings[node_name] = elapsed_ms
        result["node_execution_times"] = timings
        return result

    return wrapper


def should_continue_analysis(state: CircuitMindState) -> str:
    """Conditional edge router: checks if datasheet evidence was found.

    If no documents were retrieved, routes directly to report generation to explain
    lack of evidence rather than attempting component specification extraction.
    """
    docs = state.get("retrieved_documents", [])
    if not docs:
        logger.warning("No datasheet evidence retrieved. Routing directly to report agent.")
        return "generate_report"
    return "extract_specifications"


def create_copilot_workflow() -> CompiledStateGraph:
    """Construct and compile the CircuitMind AI LangGraph StateGraph.

    Returns:
        CompiledStateGraph instance ready for invocation.
    """
    workflow = StateGraph(CircuitMindState)

    # Register Nodes with Execution Profiling
    workflow.add_node("extract_requirements", profile_node("extract_requirements", requirement_agent_node))
    workflow.add_node("retrieve_evidence", profile_node("retrieve_evidence", research_agent_node))
    workflow.add_node("extract_specifications", profile_node("extract_specifications", component_agent_node))
    workflow.add_node("validate_electrical", profile_node("validate_electrical", validation_agent_node))
    workflow.add_node("generate_report", profile_node("generate_report", report_agent_node))

    # Define Edges
    workflow.add_edge(START, "extract_requirements")
    workflow.add_edge("extract_requirements", "retrieve_evidence")

    # Conditional routing after research
    workflow.add_conditional_edges(
        "retrieve_evidence",
        should_continue_analysis,
        {
            "extract_specifications": "extract_specifications",
            "generate_report": "generate_report",
        },
    )

    workflow.add_edge("extract_specifications", "validate_electrical")
    workflow.add_edge("validate_electrical", "generate_report")
    workflow.add_edge("generate_report", END)

    # Compile Graph
    app = workflow.compile()
    logger.info("CircuitMind AI StateGraph successfully compiled.")
    return app


def run_copilot_workflow(
    user_query: str,
    workflow: Optional[CompiledStateGraph] = None,
) -> CircuitMindState:
    """Execute the complete CircuitMind AI LangGraph engineering copilot workflow.

    Args:
        user_query: The user's circuit engineering question or design prompt.
        workflow: Optional pre-compiled StateGraph instance.

    Returns:
        The final CircuitMindState dictionary after execution through all nodes.
    """
    if not user_query or not user_query.strip():
        raise ValueError("User query cannot be empty.")

    app = workflow or create_copilot_workflow()

    initial_state: CircuitMindState = {
        "user_query": user_query.strip(),
        "requirements": {},
        "retrieved_documents": [],
        "component_information": {},
        "validation_results": {},
        "engineering_report": {},
        "node_execution_times": {},
        "errors": [],
    }

    logger.info("Executing CircuitMind AI workflow for query: %r", user_query)
    final_state: CircuitMindState = app.invoke(initial_state)
    logger.info("Workflow execution completed successfully.")
    return final_state

