"""Application Header Component for CircuitMind AI Streamlit UI.

Displays branding, project identity, and live backend connection telemetry.
"""

from __future__ import annotations

import time
from typing import Optional

import streamlit as st
from frontend.api_client import CircuitMindAPIClient


def render_header(api_client: CircuitMindAPIClient) -> None:
    """Render top application banner with live backend health status.

    Args:
        api_client: CircuitMindAPIClient instance for status probing.
    """
    col_brand, col_status = st.columns([3, 1])

    with col_brand:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 4px;">
                <span style="font-size: 1.8rem;">⚡</span>
                <div>
                    <h1 style="font-size: 1.5rem; font-weight: 700; color: #F8FAFC; margin: 0; letter-spacing: -0.02em;">
                        CIRCUITMIND AI <span style="font-size: 0.9rem; color: #38BDF8; font-weight: 500; margin-left: 8px;">COPILOT v1.0</span>
                    </h1>
                    <p style="font-size: 0.85rem; color: #94A3B8; margin: 0;">
                        Agentic Multi-Agent Electronics Engineering & Deterministic Safety Copilot
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_status:
        # Check backend health
        is_online = False
        latency_ms: Optional[float] = None
        error_msg: Optional[str] = None

        t0 = time.perf_counter()
        try:
            health = api_client.check_health()
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            is_online = health.get("status") in ("ok", "degraded")
        except Exception as exc:
            error_msg = str(exc)

        if is_online:
            st.markdown(
                f"""
                <div style="text-align: right; margin-top: 8px;">
                    <span class="badge badge-valid" style="font-size: 0.8rem; padding: 4px 10px;">
                        ● API Online ({latency_ms}ms)
                    </span>
                    <div style="font-size: 0.7rem; color: #64748B; margin-top: 2px; font-family: monospace;">
                        {api_client.base_url}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="text-align: right; margin-top: 8px;">
                    <span class="badge badge-invalid" style="font-size: 0.8rem; padding: 4px 10px;">
                        ● API Offline
                    </span>
                    <div style="font-size: 0.7rem; color: #EF4444; margin-top: 2px; font-family: monospace;">
                        Server unreachable
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<hr style='border: none; border-top: 1px solid #1E293B; margin: 8px 0 16px 0;' />", unsafe_allow_html=True)

