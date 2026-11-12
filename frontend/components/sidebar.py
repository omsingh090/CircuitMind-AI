"""Sidebar Component for CircuitMind AI Streamlit UI.

Provides backend server configuration, preset selection, target component scoping,
and retrieval strategy controls.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import streamlit as st
from frontend.api_client import CircuitMindAPIClient


PRESETS = [
    {
        "label": "🌡️ LM35 0°C–100°C 5V Monitoring",
        "query": (
            "Design a basic temperature monitoring system using the LM35 temperature sensor. "
            "Requirements: Measure temperatures from 0°C to 100°C. Use a 5V supply. "
            "Calculate expected sensor output at 25°C. Identify supply voltage requirements."
        ),
        "target": "LM35",
    },
    {
        "label": "⚡ 36V Overvoltage Stress Test",
        "query": (
            "Can I operate the LM35 precision temperature sensor with a 36V DC industrial supply rail? "
            "Evaluate voltage safety compatibility and absolute maximum ratings."
        ),
        "target": "LM35",
    },
    {
        "label": "🔋 3.3V Low-Voltage Undervoltage Test",
        "query": (
            "Can I power the LM35 temperature sensor directly from a 3.3V microcontroller supply rail? "
            "Validate minimum operating voltage compatibility."
        ),
        "target": "LM35",
    },
    {
        "label": "🎯 Fully Specified Design (Complete)",
        "query": (
            "Design a 5V circuit with LM35 sensor drawing max 50mA with 5V ADC reference and 10k load. "
            "Verify all electrical compatibility checks and output transfer voltages."
        ),
        "target": "LM35",
    },
]


def render_sidebar(
    api_client: CircuitMindAPIClient,
    on_preset_select: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    """Render sidebar controls and return active configuration dictionary.

    Args:
        api_client: CircuitMindAPIClient instance.
        on_preset_select: Optional callback when a preset button is clicked.

    Returns:
        Dictionary containing selected options (base_url, target_component, strategy).
    """
    with st.sidebar:
        st.markdown(
            """
            <div style="margin-bottom: 16px;">
                <h3 style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin: 0;">
                    ⚙️ COPILOT CONTROLS
                </h3>
                <span style="font-size: 0.75rem; color: #64748B;">Workspace Configuration</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 1. Backend Server Configuration
        st.markdown("**FastAPI Backend Endpoint**")
        current_url = st.text_input(
            "API Base URL",
            value=api_client.base_url,
            label_visibility="collapsed",
            help="HTTP address of the CircuitMind AI FastAPI service.",
        )

        col_recon, _ = st.columns([1, 1])
        with col_recon:
            if st.button("🔄 Test Link", use_container_width=True):
                api_client.base_url = current_url.rstrip("/")
                try:
                    health = api_client.check_health()
                    st.success(f"Connected (v{health.get('version', '1.0.0')})")
                except Exception as exc:
                    st.error(f"Failed: {exc}")

        st.markdown("<hr style='border: none; border-top: 1px solid #1E293B; margin: 12px 0;' />", unsafe_allow_html=True)

        # 2. Target Component Selection
        st.markdown("**Component Scope**")
        comp_choice = st.selectbox(
            "Target Component",
            options=["LM35", "Auto-Detect from Query"],
            index=0,
            label_visibility="collapsed",
            help="Scope datasheet retrieval to specific electronic component part numbers.",
        )
        target_component = ["LM35"] if comp_choice == "LM35" else None

        # 3. Retrieval Strategy Selection
        st.markdown("**Retrieval Strategy**")
        strategy = st.selectbox(
            "RAG Strategy",
            options=["hybrid", "semantic", "mmr", "keyword"],
            format_func=lambda s: {
                "hybrid": "Hybrid RRF (Dense + BM25)",
                "semantic": "Dense Semantic (ChromaDB)",
                "mmr": "Maximal Marginal Relevance",
                "keyword": "BM25 Lexical Keyword",
            }.get(s, s),
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border: none; border-top: 1px solid #1E293B; margin: 12px 0;' />", unsafe_allow_html=True)

        # 4. Engineering Query Presets
        st.markdown("**Engineering Presets**")
        st.caption("Click to load standard benchmark hardware queries:")
        for preset in PRESETS:
            if st.button(preset["label"], use_container_width=True):
                st.session_state["session_width_query"] = preset["query"]
                if on_preset_select:
                    on_preset_select(preset["query"], preset["target"])
                st.rerun()

        st.markdown("<hr style='border: none; border-top: 1px solid #1E293B; margin: 12px 0;' />", unsafe_allow_html=True)

        # 5. Architecture Summary
        st.markdown(
            """
            <div style="font-size: 0.72rem; color: #64748B; line-height: 1.4;">
                <strong style="color: #94A3B8;">Architecture Contract:</strong><br>
                Streamlit UI ➔ FastAPI REST API ➔ LangGraph Multi-Agent Workflow ➔ ChromaDB RAG + Deterministic Physics Tools.
            </div>
            """,
            unsafe_allow_html=True,
        )

        return {
            "base_url": current_url.rstrip("/"),
            "target_component": target_component,
            "strategy": strategy,
        }

