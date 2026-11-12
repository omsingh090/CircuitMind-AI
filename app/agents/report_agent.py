"""Report Agent for CircuitMind AI.

Generates a structured, engineering-grade evaluation report combining:
- Design Requirements and missing parameters
- Verified component specifications with datasheet page citations
- Deterministic electrical calculations and compatibility results
- Design warnings, violations, and limitations
- Mandatory engineering integrity fabrication disclaimer
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from google import genai

from app.config import get_settings
from app.graph.state import CircuitMindState, EngineeringReport

logger = logging.getLogger(__name__)

MANDATORY_DISCLAIMER = (
    "PRELIMINARY COPILOT DESIGN: This analysis is an engineering copilot recommendation. "
    "It is NOT fabrication-ready or certified for hardware deployment without formal schematic capture, "
    "physical prototype testing, thermal assessment, and senior engineering review."
)


class ReportAgent:
    """Agent that compiles structured engineering reports from workflow state."""

    def __init__(self, gemini_client: Optional[genai.Client] = None) -> None:
        """Initialize ReportAgent with optional Gemini client."""
        self.settings = get_settings()
        self.client = gemini_client

    def generate_report(self, state: CircuitMindState) -> EngineeringReport:
        """Compile a complete engineering report from current workflow state.

        Args:
            state: The current LangGraph workflow state dictionary.

        Returns:
            EngineeringReport structured object.
        """
        user_query = state.get("user_query", "Circuit Design Inquiry")
        reqs = state.get("requirements", {})
        comp_info = state.get("component_information", {})
        validation = state.get("validation_results", {})
        docs = state.get("retrieved_documents", [])
        errors = state.get("errors", [])

        part_name = comp_info.get("part_number") or (
            reqs.get("target_components", ["Electronic Component"])[0]
            if reqs.get("target_components")
            else "Electronic Component"
        )

        # 1. Executive Summary - synchronized with validation status
        val_status = validation.get(
            "status",
            "INCONCLUSIVE" if not validation.get("checks") else ("VALID" if validation.get("is_valid") else "INVALID")
        )
        is_valid = validation.get("is_valid", False)
        violations = validation.get("violations", [])
        warnings = validation.get("warnings", [])

        has_passed_checks = any(c.get("status") == "PASS" for c in validation.get("checks", []))
        if val_status == "INVALID" or violations:
            status_text = f"VALIDATION FAILED: Design contains {len(violations)} electrical violation(s). Circuit status is INVALID."
        elif val_status == "INCONCLUSIVE":
            if has_passed_checks:
                status_text = "INCONCLUSIVE: Preliminary checks (e.g. supply voltage compatibility) passed, but complete circuit validation is INCONCLUSIVE because required ADC interface or load constraints remain unverified/unspecified."
            else:
                status_text = "INCONCLUSIVE: Essential electrical parameters or verified evidence missing. Circuit compatibility cannot be confirmed."
        else:
            status_text = "PASSED DETERMINISTIC VALIDATION: PRELIMINARY COMPATIBLE. All verified electrical checks passed within datasheet operating boundaries."

        exec_summary = (
            f"Engineering evaluation for '{user_query}'.\n"
            f"Target Component: {part_name}.\n"
            f"Validation Status: {status_text}\n"
            f"Evidence Base: {len(docs)} datasheet passages retrieved from project database."
        )

        # 2. Requirements Summary
        identified = reqs.get("identified_requirements", {})
        missing = reqs.get("missing_requirements", [])

        req_lines = ["### Stated Design Constraints:"]
        if identified:
            for k, v in identified.items():
                req_lines.append(f"- **{k.replace('_', ' ').title()}**: {v}")
        else:
            req_lines.append("- *No explicit quantitative design constraints specified.*")

        req_lines.append("\n### Unspecified / Missing Parameters:")
        if missing:
            for m in missing:
                req_lines.append(f"- ⚠️ {m}")
        else:
            req_lines.append("- *All basic operational parameters were provided.*")
        reqs_summary = "\n".join(req_lines)

        # 3. Component Specifications
        verified = comp_info.get("verified_specs", {})
        unverified = comp_info.get("unverified_or_missing_specs", [])

        spec_lines = [f"### Verified {part_name} Datasheet Specifications:"]
        if verified:
            for k, val in verified.items():
                citation = val.get("citation", "[datasheet]")
                raw = val.get("raw", str(val))
                status_tag = f" [{val.get('status').title()}]" if val.get("status") else ""
                conds = f" (Conditions: {val.get('conditions')})" if val.get("conditions") else ""
                spec_lines.append(f"- **{k.replace('_', ' ').title()}**{status_tag}: {raw}{conds} {citation}")
        else:
            spec_lines.append("- *No verified specifications could be extracted from retrieved context.*")

        if unverified:
            spec_lines.append("\n### Unverified / Not Found in Retrieved Passages:")
            for u in unverified:
                spec_lines.append(f"- ❓ {u}")
        specs_summary = "\n".join(spec_lines)

        # 4. Electrical Validation Summary
        checks = validation.get("checks", [])
        calcs = validation.get("calculations", {})

        val_lines = ["### Deterministic Safety Checks:"]
        if checks:
            for c in checks:
                status_icon = "✅" if c.get("status") == "PASS" else ("⚠️" if c.get("status") == "UNVERIFIED" else "❌")
                val_lines.append(f"- {status_icon} **{c.get('check_name')}**: {c.get('status')}")
        else:
            val_lines.append("- *No automated checks performed (insufficient data).*")

        if calcs:
            val_lines.append("\n### Deterministic Calculations (Zero LLM Approximation):")
            for k, v in calcs.items():
                label = k.replace("_", " ").title()
                if isinstance(v, dict):
                    formula = v.get("formula", "N/A")
                    if "output_voltage_mv" in v:
                        mv = v.get("output_voltage_mv")
                        volts = v.get("output_voltage_v")
                        val_lines.append(f"- **{label}**: {mv} mV ({volts} V) [Formula: `{formula}`]")
                    elif "power_mw" in v:
                        mw = v.get("power_mw")
                        val_lines.append(f"- **{label}**: {mw} mW [Formula: `{formula}`]")
                    else:
                        val_lines.append(f"- **{label}**: {v}")
                else:
                    val_lines.append(f"- **{label}**: {v}")
        val_summary = "\n".join(val_lines)

        # 5. Warnings and Limitations
        all_warnings: List[str] = []
        for viol in violations:
            all_warnings.append(f"VIOLATION: {viol}")
        for warn in warnings:
            all_warnings.append(f"WARNING: {warn}")
        for err in errors:
            all_warnings.append(f"SYSTEM NOTICE: {err}")

        # 6. Citations
        citations = comp_info.get("citations", [])
        if not citations and docs:
            citations = [{"source": d.get("source"), "page": d.get("page")} for d in docs]

        return EngineeringReport(
            title=f"CircuitMind AI Engineering Report: {part_name} Design Analysis",
            executive_summary=exec_summary,
            requirements_summary=reqs_summary,
            component_specifications=specs_summary,
            validation_summary=val_summary,
            warnings_and_limitations=all_warnings,
            citations=citations,
            fabrication_disclaimer=MANDATORY_DISCLAIMER,
        )


def report_agent_node(state: CircuitMindState) -> Dict[str, Any]:
    """LangGraph node compiling the final engineering report."""
    agent = ReportAgent()
    try:
        report = agent.generate_report(state)
        return {"engineering_report": report.model_dump()}
    except Exception as exc:
        logger.error("Error in report_agent_node: %s", exc)
        fallback_report = EngineeringReport(
            title="CircuitMind AI Engineering Report - Synthesis Failed",
            executive_summary=f"Report compilation failed: {exc}",
            requirements_summary="Unavailable",
            component_specifications="Unavailable",
            validation_summary="Unavailable",
            warnings_and_limitations=[f"Error during report compilation: {exc}"],
            citations=[],
        )
        return {
            "engineering_report": fallback_report.model_dump(),
            "errors": [f"ReportAgent error: {exc}"],
        }
