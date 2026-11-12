"""Engineering Report Viewer Component for CircuitMind AI Streamlit UI.

Renders executive summaries, verified component specifications, citations,
fabrication disclaimers, and provides technical report export capabilities.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import streamlit as st
from frontend.styles import citation_badge, status_badge


def render_report_viewer(analysis_data: Optional[Dict[str, Any]] = None) -> None:
    """Render structured engineering report and technical specifications.

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
                📄 STRUCTURED ENGINEERING REPORT VIEWER
            </h3>
            <p style="font-size: 0.82rem; color: #94A3B8; margin: 2px 0 0 0;">
                Synthesized engineering findings, verified component ratings, datasheet page citations, and safety boundaries.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not analysis_data:
        st.info("No report generated yet. Submit an engineering inquiry to generate an analysis report.")
        return

    rep = analysis_data.get("engineering_report", {})
    comp = analysis_data.get("component_information", {})
    val = analysis_data.get("validation_results", {})

    title = rep.get("title", "CircuitMind AI Engineering Report")
    exec_summary = rep.get("executive_summary", "")
    citations: List[Dict[str, Any]] = rep.get("citations", [])
    disclaimer = rep.get("fabrication_disclaimer", "")

    # Report Header Card
    st.markdown(
        f"""
        <div class="tech-card" style="border-left: 4px solid #38BDF8;">
            <div class="tech-card-header">
                <span class="tech-card-title">{title}</span>
                {status_badge(val.get('status', 'INCONCLUSIVE'))}
            </div>
            <div style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.6; white-space: pre-line;">
                {exec_summary}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_specs, col_reqs = st.columns([3, 2])

    with col_specs:
        st.markdown(f"#### Verified {comp.get('part_number', 'Component')} Specifications")
        verified_specs: Dict[str, Any] = comp.get("verified_specs", {})

        if not verified_specs:
            st.warning("No verified specifications extracted from retrieved evidence.")
        else:
            for key, spec in verified_specs.items():
                label = key.replace("_", " ").title()
                raw = spec.get("raw", str(spec))
                src = spec.get("source", "datasheet.pdf")
                pg = spec.get("page", 1)
                status_tag = spec.get("status", "").upper()
                conds = spec.get("conditions", "")
                variant = spec.get("variant", "")

                status_html = f'<span class="badge badge-info" style="margin-left: 6px;">{status_tag}</span>' if status_tag else ""
                variant_html = f'<span style="font-size: 0.72rem; color: #64748B;"> | Variant: {variant}</span>' if variant else ""

                st.markdown(
                    f"""
                    <div style="background: #111827; border: 1px solid #1E293B; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                            <div>
                                <strong style="color: #F8FAFC; font-size: 0.88rem;">{label}</strong>
                                {status_html}
                            </div>
                            {citation_badge(src, pg)}
                        </div>
                        <div style="font-size: 0.85rem; color: #38BDF8; font-family: monospace; font-weight: 600;">
                            {raw}
                        </div>
                        <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 4px;">
                            Conditions: {conds or 'Nominal operating range'}{variant_html}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        unverified = comp.get("unverified_or_missing_specs", [])
        if unverified:
            st.markdown("##### Unverified Parameters in Retrieved Context")
            for u in unverified:
                st.markdown(f"- <span style='font-size: 0.8rem; color: #94A3B8;'>❓ {u}</span>", unsafe_allow_html=True)

    with col_reqs:
        st.markdown("#### Design Requirements Analysis")
        reqs = analysis_data.get("requirements", {})
        missing_reqs = analysis_data.get("missing_requirements", [])

        st.markdown(
            f"""
            <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 6px; padding: 12px; margin-bottom: 12px;">
                <div style="font-size: 0.75rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Stated Constraints</div>
                <div style="font-size: 0.82rem; color: #CBD5E1; margin-top: 6px; line-height: 1.5;">
                    • Target Component: <code>{comp.get('part_number', 'N/A')}</code><br>
                    • Supply Voltage: <code>{reqs.get('target_supply_voltage_v', 'Unstated')} V</code><br>
                    • Current Target: <code>{reqs.get('target_max_current_ma', 'Unstated')} mA</code><br>
                    • Temp Range: <code>{reqs.get('min_temp_c', 'Unstated')}°C to {reqs.get('max_temp_c', 'Unstated')}°C</code><br>
                    • ADC Ref: <code>{reqs.get('adc_reference_voltage_v', 'Unstated')} V</code><br>
                    • Load: <code>{reqs.get('load_impedance_ohms', 'Unstated')} Ω</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="font-size: 0.75rem; color: #64748B; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;">
                Missing Engineering Constraints
            </div>
            """,
            unsafe_allow_html=True,
        )
        if missing_reqs:
            for m in missing_reqs:
                st.markdown(f"- <span style='font-size: 0.8rem; color: #FBBF24;'>⚠️ {m}</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='font-size: 0.8rem; color: #34D399;'>✓ All primary constraints specified.</span>", unsafe_allow_html=True)

    # Citations Table
    st.markdown("#### Grounded Datasheet Citations Table")
    if citations:
        cite_cols = st.columns(min(len(citations), 4))
        for idx, cite in enumerate(citations):
            col = cite_cols[idx % len(cite_cols)]
            with col:
                st.markdown(
                    f"""
                    <div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 6px; padding: 8px 12px; margin-bottom: 8px;">
                        <div style="font-size: 0.72rem; color: #64748B;">Citation #{idx + 1}</div>
                        <div style="font-size: 0.85rem; color: #38BDF8; font-weight: 600; font-family: monospace;">
                            {cite.get('source', 'datasheet.pdf')}
                        </div>
                        <div style="font-size: 0.75rem; color: #94A3B8;">Page {cite.get('page', 1)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.caption("No explicit citations cataloged.")

    # Mandatory Fabrication Disclaimer
    st.markdown(
        f"""
        <div class="disclaimer-box">
            <strong>⚠️ ENGINEERING INTEGRITY NOTICE</strong><br>
            {disclaimer}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Technical JSON Payload Inspector & Download
    col_dl, col_insp = st.columns([1, 1])
    with col_dl:
        report_md = f"# {title}\n\n{exec_summary}\n\n## Mandatory Disclaimer\n{disclaimer}"
        st.download_button(
            "💾 Export Report Markdown",
            data=report_md,
            file_name="circuitmind_engineering_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col_insp:
        with st.expander("🔬 Inspect Raw FastAPI JSON Payload"):
            st.json(analysis_data)

