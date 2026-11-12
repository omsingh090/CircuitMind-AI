"""Electrical Safety Validation Center Component for CircuitMind AI Streamlit UI.

Displays deterministic electrical safety checks, 3-state overall validation status
(VALID, INVALID, INCONCLUSIVE), physics calculations, and safety warnings.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st
from frontend.styles import status_badge


def render_validation_center(analysis_data: Optional[Dict[str, Any]] = None) -> None:
    """Render deterministic electrical validation results and physics calculations.

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
                🛡️ ELECTRICAL SAFETY & VALIDATION CENTER
            </h3>
            <p style="font-size: 0.82rem; color: #94A3B8; margin: 2px 0 0 0;">
                Deterministic Python validation engine. Strictly zero LLM arithmetic or probabilistic safety approximations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not analysis_data:
        st.info("No active validation data. Submit an engineering design prompt to evaluate electrical safety.")
        return

    val_res = analysis_data.get("validation_results", {})
    val_status = str(val_res.get("status", "INCONCLUSIVE")).upper()
    is_valid = val_res.get("is_valid", False)
    checks: List[Dict[str, Any]] = val_res.get("checks", [])
    violations: List[str] = val_res.get("violations", [])
    warnings: List[str] = val_res.get("warnings", [])
    calcs: Dict[str, Any] = val_res.get("calculations", {})

    # 1. Primary Overall Validation Status Banner
    if val_status == "VALID":
        st.markdown(
            """
            <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #10B981; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 1.4rem;">✅</span>
                        <div>
                            <strong style="color: #34D399; font-size: 1.05rem;">VALID: DESIGN PASSED DETERMINISTIC CHECKS</strong>
                            <div style="font-size: 0.82rem; color: #A7F3D0;">All verified operating ratings and interface constraints were proven within safe datasheet limits.</div>
                        </div>
                    </div>
                    <span class="badge badge-valid" style="font-size: 0.9rem; padding: 4px 12px;">VALID</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif val_status == "INVALID" or violations:
        st.markdown(
            f"""
            <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid #EF4444; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 1.4rem;">🚨</span>
                        <div>
                            <strong style="color: #F87171; font-size: 1.05rem;">INVALID: ELECTRICAL SAFETY VIOLATIONS DETECTED</strong>
                            <div style="font-size: 0.82rem; color: #FECACA;">Circuit design violates verified component ratings or operating limits. Hardware damage risk.</div>
                        </div>
                    </div>
                    <span class="badge badge-invalid" style="font-size: 0.9rem; padding: 4px 12px;">INVALID</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:  # INCONCLUSIVE
        st.markdown(
            """
            <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid #F59E0B; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 1.4rem;">⚠️</span>
                        <div>
                            <strong style="color: #FBBF24; font-size: 1.05rem;">INCONCLUSIVE: INSUFFICIENT EVIDENCE OR MISSING CONSTRAINTS</strong>
                            <div style="font-size: 0.82rem; color: #FDE68A;">
                                Supply compatibility passed, but full circuit validation cannot be confirmed because interface constraints (ADC or load impedance) remain unverified.
                            </div>
                        </div>
                    </div>
                    <span class="badge badge-inconclusive" style="font-size: 0.9rem; padding: 4px 12px;">INCONCLUSIVE</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Critical Violations Section
    if violations:
        st.markdown("#### 🚨 Safety Violations")
        for v in violations:
            st.error(f"**VIOLATION:** {v}")

    # 3. Individual Deterministic Checks List
    st.markdown("#### Automated Deterministic Safety Checks")
    if not checks:
        st.info("No deterministic checks could be executed (insufficient component specifications or query parameters).")
    else:
        for c in checks:
            name = c.get("check_name", "Check")
            chk_status = c.get("status", "FAIL")
            details = c.get("details", {})

            badge = status_badge(chk_status)

            detail_lines = []
            if "headroom_v" in details:
                detail_lines.append(f"Operating Headroom: <code>{details.get('headroom_v')} V</code> to maximum limit")
            if "power_mw" in details:
                detail_lines.append(f"Calculated Dissipation: <code>{details.get('power_mw')} mW</code>")
            if "max_sensor_output_v" in details and "adc_reference_v" in details:
                detail_lines.append(f"Sensor Output: <code>{details.get('max_sensor_output_v')} V</code> vs ADC Ref: <code>{details.get('adc_reference_v')} V</code>")
            if "load_current_ma" in details:
                detail_lines.append(f"Load Current: <code>{details.get('load_current_ma')} mA</code> (Max: <code>{details.get('max_rated_load_ma', 10.0)} mA</code>)")
            if "reason" in details:
                detail_lines.append(f"<span style='color: #FBBF24;'>{details.get('reason')}</span>")

            details_html = "<br>".join(detail_lines) if detail_lines else "Completed within nominal specifications."

            st.markdown(
                f"""
                <div class="tech-card">
                    <div class="tech-card-header">
                        <span style="font-weight: 600; color: #F1F5F9; font-size: 0.92rem;">{name}</span>
                        <div>{badge}</div>
                    </div>
                    <div style="font-size: 0.82rem; color: #94A3B8; line-height: 1.5;">
                        {details_html}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 4. Deterministic Physics Calculations Grid
    st.markdown("#### Deterministic Physics Calculations (Zero LLM Approximation)")
    if not calcs:
        st.caption("No mathematical calculations performed for this query.")
    else:
        calc_cols = st.columns(min(len(calcs), 3))
        col_idx = 0
        for key, val in calcs.items():
            col = calc_cols[col_idx % len(calc_cols)]
            label = key.replace("_", " ").title()

            if isinstance(val, dict):
                formula = val.get("formula", "N/A")
                if "output_voltage_mv" in val:
                    mv = val.get("output_voltage_mv")
                    v_val = val.get("output_voltage_v")
                    t_c = val.get("temperature_c")
                    with col:
                        st.markdown(
                            f"""
                            <div class="metric-box">
                                <div class="metric-label">{label} ({t_c}°C)</div>
                                <div class="metric-value">{mv} mV <span style="font-size: 0.8rem; color: #38BDF8;">({v_val} V)</span></div>
                                <div style="font-size: 0.7rem; color: #64748B; margin-top: 4px; font-family: monospace;">{formula}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                elif "power_mw" in val:
                    mw = val.get("power_mw")
                    with col:
                        st.markdown(
                            f"""
                            <div class="metric-box">
                                <div class="metric-label">{label}</div>
                                <div class="metric-value">{mw} mW</div>
                                <div style="font-size: 0.7rem; color: #64748B; margin-top: 4px; font-family: monospace;">{formula}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    with col:
                        st.markdown(
                            f"""
                            <div class="metric-box">
                                <div class="metric-label">{label}</div>
                                <div class="metric-value">{val}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            col_idx += 1

    # 5. Engineering Warnings & Derating Notices
    if warnings:
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown("#### ⚠️ Engineering Warnings & Derating Notices")
        for w in warnings:
            st.warning(w)

