"""CLI Runner for CircuitMind AI LangGraph Engineering Workflow.

Executes the compiled multi-agent stategraph:
1. RequirementAgent: Parses constraints and identifies missing parameters without guessing.
2. ResearchAgent: Retrieves grounded datasheet evidence using Hybrid RAG (ChromaDB + BM25 + RRF).
3. ComponentAgent: Synthesizes verified electrical specifications with source and page citations.
4. ValidationAgent: Runs deterministic safety calculations (Ohm's Law, power dissipation, voltage limits).
5. ReportAgent: Compiles the final structured engineering report with citations and fabrication disclaimer.

Usage:
    python scripts/run_workflow.py "Can I operate an LM35 sensor at 5V supply with 10mA current budget?"
    python scripts/run_workflow.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.graph.state import CircuitMindState
from app.graph.workflow import create_copilot_workflow

# Configure terminal logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEPARATOR_DOUBLE = "=" * 80
SEPARATOR_SINGLE = "-" * 80
SEPARATOR_DOTS = "·" * 80


def print_banner(query: str) -> None:
    """Display CLI startup banner."""
    print(SEPARATOR_DOUBLE)
    print(" ⚡ CIRCUITMIND AI: AGENTIC ELECTRONICS ENGINEERING COPILOT")
    print(" LangGraph Multi-Agent Orchestration & Deterministic Safety Engine")
    print(SEPARATOR_DOUBLE)
    print(f" INPUT QUERY: \"{query}\"")
    print(SEPARATOR_DOUBLE)


def print_stage_header(step_num: int, name: str, description: str) -> None:
    """Display stage separator and title."""
    print(f"\n[STAGE {step_num}/5] >>> {name.upper()}")
    print(f"Role: {description}")
    print(SEPARATOR_SINGLE)


def display_requirements(requirements: Dict[str, Any]) -> None:
    """Format and display extracted engineering requirements."""
    targets = requirements.get("target_components", [])
    v_target = requirements.get("target_supply_voltage_v")
    i_target = requirements.get("target_max_current_ma")
    t_min = requirements.get("min_temp_c")
    t_max = requirements.get("max_temp_c")
    identified = requirements.get("identified_requirements", {})
    missing = requirements.get("missing_requirements", [])

    print("• Target Components Identified:")
    if targets:
        for t in targets:
            print(f"    - {t}")
    else:
        print("    - None explicitly specified")

    print("\n• Stated Electrical & Operational Constraints:")
    if v_target is not None:
        print(f"    - Supply Voltage: {v_target} V")
    if i_target is not None:
        print(f"    - Target Current Limit: {i_target} mA")
    if t_min is not None or t_max is not None:
        print(f"    - Operating Temperature: {t_min}°C to {t_max}°C")

    if not identified:
        print("    - (No quantitative operational limits were stated in the query)")

    print("\n• Explicitly Unstated / Missing Requirements (Zero-Hallucination Policy):")
    if missing:
        for m in missing:
            print(f"    ⚠️  {m}")
    else:
        print("    - All required baseline operational constraints were provided.")


def display_retrieved_evidence(documents: List[Dict[str, Any]], verbose: bool = False) -> None:
    """Format and display retrieved datasheet chunks with citations."""
    if not documents:
        print("⚠️  No relevant datasheet passages were retrieved from the database.")
        return

    print(f"Total Passages Retrieved: {len(documents)} (Hybrid RRF Semantic + BM25 Search)\n")
    for idx, doc in enumerate(documents, start=1):
        source = doc.get("source", "datasheet.pdf")
        page = doc.get("page", "?")
        score = doc.get("rrf_score")
        score_str = f" | RRF Score: {score:.5f}" if score is not None else ""
        print(f"[{idx}] Citation: [{source}, p. {page}]{score_str}")

        content = doc.get("content", "").strip()
        if not verbose and len(content) > 220:
            content_preview = content[:220].replace("\n", " ") + "..."
        else:
            content_preview = content.replace("\n", " ")
        print(f"    Excerpt: \"{content_preview}\"")
        if idx < len(documents):
            print(SEPARATOR_DOTS)


def display_component_specifications(comp_info: Dict[str, Any]) -> None:
    """Format and display verified component specifications."""
    part = comp_info.get("part_number", "Component")
    verified = comp_info.get("verified_specs", {})
    unverified = comp_info.get("unverified_or_missing_specs", [])

    print(f"Component: {part}")
    print("\n• Verified Datasheet Specifications (with Mandatory Citations):")
    if verified:
        for key, val in verified.items():
            citation = val.get("citation", "[datasheet]")
            raw = val.get("raw", str(val))
            label = key.replace("_", " ").title()
            print(f"    ✅ {label}: {raw}  -->  {citation}")
    else:
        print("    - No verified parameters could be extracted from retrieved evidence.")

    print("\n• Specifications Not Found / Unverified in Retrieved Passages:")
    if unverified:
        for u in unverified:
            print(f"    ❓ {u}")
    else:
        print("    - All primary component parameters were verified.")


def display_electrical_validation(validation: Dict[str, Any]) -> None:
    """Format and display deterministic electrical safety checks and calculations."""
    is_valid = validation.get("is_valid", False)
    val_status = validation.get("status", "VALID" if is_valid else "INVALID")
    checks = validation.get("checks", [])
    violations = validation.get("violations", [])
    warnings = validation.get("warnings", [])
    calculations = validation.get("calculations", {})

    if val_status == "VALID":
        status_tag = "✅ ALL COMPATIBILITY CHECKS PASSED (VALID)"
    elif val_status == "INCONCLUSIVE":
        status_tag = "⚠️ INCONCLUSIVE (Insufficient Parameters / Evidence)"
    else:
        status_tag = "❌ SAFETY VIOLATIONS DETECTED (INVALID)"
    print(f"Overall Electrical Safety Status: {status_tag}")

    print("\n• Automated Deterministic Checks (ElectricalCalculator):")
    if checks:
        for c in checks:
            status_val = c.get("status", "FAIL")
            tag = "✅ PASS" if status_val == "PASS" else ("⚠️ UNVERIFIED" if status_val == "UNVERIFIED" else "❌ FAIL")
            print(f"    [{tag}] {c.get('check_name')}")
            details = c.get("details", {})
            if "headroom_v" in details:
                print(f"           Headroom: {details.get('headroom_v')} V to maximum limit")
            if "power_mw" in details:
                print(f"           Calculated Dissipation: {details.get('power_mw')} mW")
            if "reason" in details:
                print(f"           Notice: {details.get('reason')}")
    else:
        print("    - No automated checks executed (insufficient parameter data).")

    if calculations:
        print("\n• Deterministic Physics Calculations (Zero LLM Approximation):")
        for k, v in calculations.items():
            label = k.replace("_", " ").title()
            if isinstance(v, dict):
                formula = v.get("formula", "N/A")
                if "output_voltage_mv" in v:
                    mv = v.get("output_voltage_mv")
                    volts = v.get("output_voltage_v")
                    t_c = v.get("temperature_c")
                    print(f"    - {label}: {mv} mV ({volts} V) at {t_c}°C (Formula: {formula})")
                elif "power_mw" in v:
                    mw = v.get("power_mw")
                    print(f"    - {label}: {mw} mW (Formula: {formula})")
                else:
                    print(f"    - {label}: {v} (Formula: {formula})")
            else:
                print(f"    - {label}: {v}")

    if violations:
        print("\n• Critical Electrical Violations:")
        for v in violations:
            print(f"    🚨 {v}")

    if warnings:
        print("\n• Engineering Warnings / Derating Margins:")
        for w in warnings:
            print(f"    ⚠️  {w}")


def display_engineering_report(report: Dict[str, Any]) -> None:
    """Format and display the final synthesized engineering report."""
    title = report.get("title", "CircuitMind AI Engineering Report")
    exec_summary = report.get("executive_summary", "")
    citations = report.get("citations", [])
    disclaimer = report.get("fabrication_disclaimer", "")

    print(SEPARATOR_DOUBLE)
    print(f" 📄 {title.upper()}")
    print(SEPARATOR_DOUBLE)

    print("\n[EXECUTIVE SUMMARY]")
    print(exec_summary.strip())

    print("\n[CITATIONS TABLE]")
    if citations:
        for idx, cite in enumerate(citations, start=1):
            src = cite.get("source", "datasheet.pdf")
            pg = cite.get("page", "?")
            print(f"  ({idx}) Source: {src} | Page: {pg}")
    else:
        print("  - No citations recorded.")

    print("\n[ENGINEERING INTEGRITY DISCLAIMER]")
    print(f"  ⚠️  {disclaimer}")
    print(SEPARATOR_DOUBLE)


def execute_workflow(
    query: str,
    verbose: bool = False,
    output_json: bool = False,
) -> CircuitMindState:
    """Invoke the LangGraph workflow and render output for each stage.

    Args:
        query: Engineering question or design inquiry.
        verbose: Whether to display full chunk excerpts.
        output_json: If True, outputs raw JSON instead of human-readable UI.

    Returns:
        The final CircuitMindState dictionary.
    """
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("Engineering query cannot be empty.")

    if not output_json:
        print_banner(clean_query)

    app = create_copilot_workflow()

    initial_state: CircuitMindState = {
        "user_query": clean_query,
        "requirements": {},
        "retrieved_documents": [],
        "component_information": {},
        "validation_results": {},
        "engineering_report": {},
        "errors": [],
    }

    # Stream workflow execution node-by-node
    accumulated_state: CircuitMindState = dict(initial_state)

    for step in app.stream(initial_state):
        for node_name, node_update in step.items():
            accumulated_state.update(node_update)

            if output_json:
                continue

            if node_name == "extract_requirements":
                print_stage_header(
                    1,
                    "Requirement Extraction",
                    "RequirementAgent (identifies stated constraints and missing parameters)",
                )
                display_requirements(accumulated_state.get("requirements", {}))

            elif node_name == "retrieve_evidence":
                print_stage_header(
                    2,
                    "Datasheet Evidence Retrieval",
                    "ResearchAgent (Hybrid RAG: ChromaDB Dense Vectors + BM25 Keyword Search)",
                )
                display_retrieved_evidence(
                    accumulated_state.get("retrieved_documents", []),
                    verbose=verbose,
                )

            elif node_name == "extract_specifications":
                print_stage_header(
                    3,
                    "Component Specification Extraction",
                    "ComponentAgent (extracts verified parameters with source & page citations)",
                )
                display_component_specifications(
                    accumulated_state.get("component_information", {})
                )

            elif node_name == "validate_electrical":
                print_stage_header(
                    4,
                    "Deterministic Electrical Validation",
                    "ValidationAgent & ElectricalCalculator (zero LLM safety approval policy)",
                )
                display_electrical_validation(
                    accumulated_state.get("validation_results", {})
                )

            elif node_name == "generate_report":
                print_stage_header(
                    5,
                    "Structured Engineering Report Synthesis",
                    "ReportAgent (assembles findings, calculations, citations, and disclaimer)",
                )
                display_engineering_report(
                    accumulated_state.get("engineering_report", {})
                )

    if output_json:
        print(json.dumps(accumulated_state, indent=2, default=str))

    return accumulated_state


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description="CircuitMind AI — LangGraph Multi-Agent Electronics Engineering Copilot Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="Can I operate an LM35 sensor at 5V supply with 10mA max current consumption?",
        help="Engineering circuit query or design prompt (e.g. 'Can I run LM35 at 5V supply?')",
    )
    parser.add_argument(
        "-q",
        "--question",
        dest="query_opt",
        help="Alternative flag to specify engineering query.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Display full datasheet passage excerpts.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw state dictionary as JSON.",
    )

    args = parser.parse_args(argv)
    query = (args.query_opt or args.query).strip()

    if not query:
        print("Error: Engineering query cannot be empty.", file=sys.stderr)
        return 1

    try:
        execute_workflow(query=query, verbose=args.verbose, output_json=args.json)
        return 0
    except KeyboardInterrupt:
        print("\n[Operation cancelled by user]", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nExecution Error: {exc}", file=sys.stderr)
        logger.exception("Workflow execution crashed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

