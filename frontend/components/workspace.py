"""Engineering Workspace Component for CircuitMind AI Streamlit UI.

Provides the primary conversational engineering workspace, query submission,
conversation history, and instant executive overview.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import streamlit as st
from frontend.api_client import CircuitMindAPIClient
from frontend.styles import status_badge


def render_workspace(
    api_client: CircuitMindAPIClient,
    target_components: Optional[List[str]] = None,
    on_analysis_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    """Render the primary Engineering Workspace query interface and conversation history.

    Args:
        api_client: CircuitMindAPIClient instance for executing analysis.
        target_components: Optional component scoping.
        on_analysis_complete: Optional callback invoked when analysis succeeds.
    """
    st.markdown(
        """
        <div style="margin-bottom: 12px;">
            <h3 style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC; margin: 0;">
                💬 CONVERSATIONAL ENGINEERING WORKSPACE
            </h3>
            <p style="font-size: 0.82rem; color: #94A3B8; margin: 2px 0 0 0;">
                Submit hardware design queries, supply constraints, temperature ranges, or interface inquiries.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Pre-fill query if preset selected
    default_query = st.session_state.get(
        "session_width_query",
        "Design a basic temperature monitoring system using the LM35 temperature sensor. "
        "Requirements: Measure temperatures from 0°C to 100°C. Use a 5V supply. "
        "Calculate the expected sensor output at 25°C. Identify the sensor's supply voltage requirements.",
    )

    query_input = st.text_area(
        "Hardware Design Query",
        value=default_query,
        height=100,
        placeholder="e.g. Design a 5V circuit with LM35 sensor drawing max 50mA with 5V ADC reference and 10k load...",
        label_visibility="collapsed",
    )

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        submit = st.button("🚀 Analyze Circuit Design", type="primary", use_container_width=True)
    with col_info:
        st.caption("Executes 5 LangGraph agent nodes with deterministic electrical safety validation.")

    # Handle Analysis Execution
    if submit:
        if not query_input.strip():
            st.warning("Please enter a hardware design prompt.")
        else:
            with st.spinner("Connecting to FastAPI backend & executing LangGraph multi-agent pipeline..."):
                try:
                    result = api_client.analyze_circuit(
                        query=query_input,
                        target_components=target_components,
                    )
                    st.session_state["latest_analysis"] = result
                    if "history" not in st.session_state:
                        st.session_state["history"] = []
                    st.session_state["history"].append(result)

                    if on_analysis_complete:
                        on_analysis_complete(result)

                    st.success("Analysis completed successfully. Inspect tabs for detailed engineering telemetry.")
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")

    # Display Latest Analysis Quick Overview if Available
    latest = st.session_state.get("latest_analysis")
    if latest:
        st.markdown("<hr style='border: none; border-top: 1px solid #1E293B; margin: 20px 0 16px 0;' />", unsafe_allow_html=True)
        st.markdown("#### Latest Analysis Overview")

        val_res = latest.get("validation_results", {})
        comp_info = latest.get("component_information", {})
        exec_info = latest.get("execution_info", {})
        rep_card = latest.get("engineering_report", {})

        status = val_res.get("status", "INCONCLUSIVE")
        time_ms = exec_info.get("execution_time_ms", 0.0)
        part = comp_info.get("part_number", "Component")

        col_q1, col_q2, col_q3 = st.columns([2, 1, 1])
        with col_q1:
            st.markdown(
                f"""
                <div class="tech-card">
                    <div style="font-size: 0.75rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Inquiry</div>
                    <div style="font-size: 0.88rem; color: #F1F5F9; margin-top: 4px; font-weight: 500;">
                        {latest.get('query', '')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_q2:
            st.markdown(
                f"""
                <div class="tech-card" style="text-align: center;">
                    <div style="font-size: 0.75rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Validation</div>
                    <div style="margin-top: 6px;">{status_badge(status)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_q3:
            st.markdown(
                f"""
                <div class="tech-card" style="text-align: center;">
                    <div style="font-size: 0.75rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Latency</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #38BDF8; font-family: monospace; margin-top: 4px;">
                        {time_ms:,.1f} ms
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Quick Executive Summary Box
        st.markdown(
            f"""
            <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 6px; padding: 12px 16px; margin-bottom: 12px;">
                <div style="font-size: 0.75rem; color: #64748B; font-weight: 700; text-transform: uppercase; margin-bottom: 4px;">
                    Executive Evaluation
                </div>
                <div style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.5; white-space: pre-line;">
                    {rep_card.get('executive_summary', '')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Conversation History Accordion
    history: List[Dict[str, Any]] = st.session_state.get("history", [])
    if history:
        with st.expander(f"📚 Analysis History ({len(history)} queries executed)"):
            for idx, item in enumerate(reversed(history), 1):
                h_val = item.get("validation_results", {}).get("status", "INCONCLUSIVE")
                h_time = item.get("execution_info", {}).get("execution_time_ms", 0.0)
                st.markdown(
                    f"**#{idx}**: `{item.get('query')[:90]}...` — {status_badge(h_val)} (`{h_time:,.1f} ms`)",
                    unsafe_allow_html=True,
                )

