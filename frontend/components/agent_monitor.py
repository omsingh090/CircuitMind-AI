"""Agent Execution Monitor Component for CircuitMind AI Streamlit UI.

Visualizes the 5-stage LangGraph multi-agent execution pipeline, per-node
durations (ms), and execution telemetry returned by the FastAPI backend.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st
from frontend.styles import status_badge

# Metadata for each canonical stage in the 5-stage workflow
STAGE_METADATA = [
    {
        "id": "extract_requirements",
        "agent": "RequirementAgent",
        "role": "Requirement Parser",
        "desc": "Extracts quantitative design constraints and catalogs missing parameters.",
    },
    {
        "id": "retrieve_evidence",
        "agent": "ResearchAgent",
        "role": "Datasheet Retrieval",
        "desc": "Executes hybrid dense vector and BM25 search over verified datasheets.",
    },
    {
        "id": "extract_specifications",
        "agent": "ComponentAgent",
        "role": "Specification Synthesizer",
        "desc": "Extracts verified operating ratings, tolerances, and pin limits with citations.",
    },
    {
        "id": "validate_electrical",
        "agent": "ValidationAgent",
        "role": "Deterministic Validator",
        "desc": "Performs pure Python physics calculations and electrical rating checks (Zero LLM).",
    },
    {
        "id": "generate_report",
        "agent": "ReportAgent",
        "role": "Report Synthesizer",
        "desc": "Assembles structured engineering report with citations and mandatory disclaimer.",
    },
]


def render_agent_monitor(analysis_data: Optional[Dict[str, Any]] = None) -> None:
    """Render the 5-stage LangGraph agent execution flow and telemetry.

    Args:
        analysis_data: Optional full AnalyzeResponse dictionary from FastAPI.
                       If None, falls back to st.session_state["latest_analysis"].
    """
    if analysis_data is None:
        analysis_data = st.session_state.get("latest_analysis")
    st.markdown(
        """
        <div style="margin-bottom: 12px;">
            <h3 style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC; margin: 0;">
                ⚡ LANGGRAPH AGENT EXECUTION MONITOR
            </h3>
            <p style="font-size: 0.82rem; color: #94A3B8; margin: 2px 0 0 0;">
                Live inspection of multi-agent state orchestration and per-node wall-clock execution telemetry.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not analysis_data:
        st.info("No active workflow telemetry. Submit a query in the Engineering Workspace to inspect execution.")
        return

    exec_info = analysis_data.get("execution_info", {})
    total_time_ms = exec_info.get("execution_time_ms", 0.0)
    exec_status = exec_info.get("status", "completed")
    nodes_executed: List[str] = exec_info.get("nodes_executed", [])
    node_timings: Dict[str, float] = exec_info.get("node_timings_ms", {})

    # Top Telemetry Metrics Bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Total Execution Time</div>
                <div class="metric-value">{total_time_ms:,.1f} ms</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Pipeline Status</div>
                <div class="metric-value" style="color: {'#34D399' if exec_status == 'completed' else '#FBBF24'};">
                    {exec_status.upper()}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Nodes Executed</div>
                <div class="metric-value">{len(nodes_executed)} / 5</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        avg_time = total_time_ms / len(nodes_executed) if nodes_executed else 0.0
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Avg Node Latency</div>
                <div class="metric-value">{avg_time:.1f} ms</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 5-Stage Agent Node Flow Cards
    st.markdown("#### Agentic Execution Pipeline Flow")
    for idx, stage in enumerate(STAGE_METADATA, 1):
        stage_id = stage["id"]
        is_executed = stage_id in nodes_executed
        timing_ms = node_timings.get(stage_id)

        border_color = "#38BDF8" if is_executed else "#1E293B"
        badge_html = (
            f'<span class="badge badge-valid">✓ COMPLETED ({timing_ms:.1f} ms)</span>'
            if timing_ms is not None
            else ('<span class="badge badge-valid">✓ EXECUTED</span>' if is_executed else '<span class="badge badge-info">SKIPPED / PENDING</span>')
        )

        st.markdown(
            f"""
            <div class="tech-card" style="border-left: 4px solid {border_color};">
                <div class="tech-card-header">
                    <div>
                        <span style="font-size: 0.75rem; color: #64748B; font-weight: 700; margin-right: 6px;">
                            STAGE {idx}
                        </span>
                        <strong style="font-size: 0.95rem; color: #F1F5F9;">{stage['agent']}</strong>
                        <span style="font-size: 0.8rem; color: #94A3B8; margin-left: 8px;">({stage['role']})</span>
                    </div>
                    <div>
                        {badge_html}
                    </div>
                </div>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-bottom: 6px;">
                    {stage['desc']}
                </div>
                <div style="font-size: 0.72rem; color: #64748B; font-family: monospace;">
                    State transition: state["{stage_id}"] ➔ LangGraph compiled StateGraph
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

