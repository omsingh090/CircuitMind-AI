"""Design System and Custom Styling for CircuitMind AI Streamlit Frontend.

Applies a professional dark navy engineering workstation theme with cyan and
blue accents, compact technical cards, and clean typography.
"""

from __future__ import annotations

import streamlit as st


DARK_NAVY_CSS = """
<style>
/* ==============================================================================
   CircuitMind AI Design System - Dark Navy Engineering Theme
   ============================================================================== */

/* Base container styling */
.stApp {
    background-color: #0B0F19;
    color: #F1F5F9;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0F172A;
    border-right: 1px solid #1E293B;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #94A3B8;
}

/* Top header bar accent */
header[data-testid="stHeader"] {
    background-color: #0B0F19;
    border-bottom: 1px solid #1E293B;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #0F172A;
    padding: 6px 12px;
    border-radius: 8px;
    border: 1px solid #1E293B;
}

.stTabs [data-baseweb="tab"] {
    color: #94A3B8;
    background-color: transparent;
    border-radius: 6px;
    padding: 6px 16px;
    font-weight: 500;
    font-size: 0.9rem;
    border: none;
    transition: all 0.15s ease-in-out;
}

.stTabs [aria-selected="true"] {
    background-color: #1E293B !important;
    color: #38BDF8 !important;
    font-weight: 600;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
}

/* Technical Card Component */
.tech-card {
    background-color: #111827;
    border: 1px solid #1E293B;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
    transition: border-color 0.15s ease;
}

.tech-card:hover {
    border-color: #334155;
}

.tech-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #1E293B;
    padding-bottom: 8px;
    margin-bottom: 12px;
}

.tech-card-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #38BDF8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 0;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.badge-valid {
    background-color: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.4);
}

.badge-invalid {
    background-color: rgba(239, 68, 68, 0.15);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.4);
}

.badge-inconclusive {
    background-color: rgba(245, 158, 11, 0.15);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.4);
}

.badge-citation {
    background-color: rgba(56, 189, 248, 0.1);
    color: #38BDF8;
    border: 1px solid rgba(56, 189, 248, 0.3);
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

.badge-info {
    background-color: rgba(99, 102, 241, 0.15);
    color: #A5B4FC;
    border: 1px solid rgba(99, 102, 241, 0.3);
}

/* Metric blocks */
.metric-box {
    background-color: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 6px;
    padding: 12px;
    text-align: center;
}

.metric-label {
    font-size: 0.75rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}

.metric-value {
    font-size: 1.25rem;
    font-weight: 700;
    color: #F8FAFC;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

/* Stage pipeline container */
.stage-pipeline {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 12px 0;
}

.stage-pill {
    display: flex;
    align-items: center;
    gap: 6px;
    background-color: #0F172A;
    border: 1px solid #1E293B;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

.stage-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #34D399;
}

/* Engineering report callouts */
.disclaimer-box {
    background-color: rgba(245, 158, 11, 0.08);
    border-left: 3px solid #F59E0B;
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    margin-top: 16px;
    font-size: 0.85rem;
    color: #FCD34D;
}

/* Code & preformatted text */
code {
    color: #38BDF8 !important;
    background-color: #0F172A !important;
    border: 1px solid #1E293B !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-size: 0.85em !important;
}
</style>
"""


def apply_theme() -> None:
    """Inject custom dark navy CSS theme into Streamlit application."""
    st.markdown(DARK_NAVY_CSS, unsafe_allow_html=True)


def status_badge(status: str) -> str:
    """Render a styled HTML badge for validation and check states.

    Args:
        status: Status string (VALID, INVALID, INCONCLUSIVE, PASS, FAIL, UNVERIFIED).

    Returns:
        HTML string with appropriate badge classes and styling.
    """
    clean = str(status).upper().strip()
    if clean in ("VALID", "PASS", "COMPATIBLE", "OK"):
        return f'<span class="badge badge-valid">✓ {clean}</span>'
    elif clean in ("INVALID", "FAIL", "VIOLATION", "ERROR"):
        return f'<span class="badge badge-invalid">✗ {clean}</span>'
    elif clean in ("INCONCLUSIVE", "UNVERIFIED", "WARNING"):
        return f'<span class="badge badge-inconclusive">⚠ {clean}</span>'
    else:
        return f'<span class="badge badge-info">{clean}</span>'


def citation_badge(source: str, page: Any) -> str:
    """Render a compact technical citation badge.

    Args:
        source: Datasheet PDF filename (e.g. 'lm35.pdf').
        page: Page number in datasheet.

    Returns:
        HTML string representing formatted citation tag.
    """
    return f'<span class="badge badge-citation">[{source}, p. {page}]</span>'

