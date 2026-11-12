"""Main Streamlit Application for CircuitMind AI.

Entry point for the Hybrid Engineering Copilot interface: combines conversational
engineering assistant with live multi-agent execution telemetry, datasheet explorer,
deterministic validation dashboard, and structured engineering report viewer.
"""

from __future__ import annotations

import streamlit as st
from frontend.api_client import CircuitMindAPIClient
from frontend.components.agent_monitor import render_agent_monitor
from frontend.components.datasheet_explorer import render_datasheet_explorer
from frontend.components.header import render_header
from frontend.components.report_viewer import render_report_viewer
from frontend.components.sidebar import render_sidebar
from frontend.components.validation_center import render_validation_center
from frontend.components.workspace import render_workspace
from frontend.styles import apply_theme

# Page Configuration
st.set_page_config(
    page_title="CircuitMind AI — Electronics Engineering Copilot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Initialize and run the CircuitMind AI Streamlit web application."""
    # Apply dark navy engineering stylesheet
    apply_theme()

    # Initialize session state objects
    if "api_client" not in st.session_state:
        st.session_state["api_client"] = CircuitMindAPIClient()
    if "latest_analysis" not in st.session_state:
        st.session_state["latest_analysis"] = None
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "direct_retrieval_results" not in st.session_state:
        st.session_state["direct_retrieval_results"] = None

    api_client: CircuitMindAPIClient = st.session_state["api_client"]

    # Callback when a preset button is clicked in the sidebar
    def handle_preset_select(query_text: str, target: str) -> None:
        st.session_state["session_width_query"] = query_text

    # Render Sidebar & retrieve active controls
    sidebar_opts = render_sidebar(api_client, on_preset_select=handle_preset_select)

    # If user changed base URL in sidebar, update client instance
    if sidebar_opts["base_url"] != api_client.base_url:
        api_client.base_url = sidebar_opts["base_url"]

    # Render Top Branding & Telemetry Header
    render_header(api_client)

    # Render Primary 5 Workspace Tabs
    tab_workspace, tab_monitor, tab_explorer, tab_validator, tab_reports = st.tabs([
        "💬 Engineering Workspace",
        "⚡ Agent Monitor",
        "🔍 Datasheet Explorer",
        "🛡️ Validation Center",
        "📄 Engineering Reports",
    ])

    with tab_workspace:
        render_workspace(
            api_client=api_client,
            target_components=sidebar_opts["target_component"],
        )

    # Dynamically read current analysis from session state after workspace execution
    current_analysis = st.session_state.get("latest_analysis")

    with tab_monitor:
        render_agent_monitor(analysis_data=current_analysis)

    with tab_explorer:
        render_datasheet_explorer(
            analysis_data=current_analysis,
            api_client=api_client,
            target_components=sidebar_opts["target_component"],
        )

    with tab_validator:
        render_validation_center(analysis_data=current_analysis)

    with tab_reports:
        render_report_viewer(analysis_data=current_analysis)


if __name__ == "__main__":
    main()

