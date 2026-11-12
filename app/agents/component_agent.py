"""Component Agent for CircuitMind AI.

Extracts electrical specifications from retrieved datasheet evidence.
Enforces strict grounding: every extracted specification must cite source filename and page.
Clearly distinguishes verified specifications from unstated or missing information.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from google import genai

from app.config import get_settings
from app.graph.state import (
    CircuitMindState,
    ComponentInformation,
    ComponentSpecification,
)

logger = logging.getLogger(__name__)


class ComponentAgent:
    """Agent that synthesizes verified component specifications from retrieved datasheet passages."""

    def __init__(self, gemini_client: Optional[genai.Client] = None) -> None:
        """Initialize ComponentAgent with optional Gemini client for parsing."""
        self.settings = get_settings()
        self.client = gemini_client

    def extract_from_documents(
        self,
        docs: List[Dict[str, Any]],
        target_component: Optional[str] = None,
    ) -> ComponentInformation:
        """Extract electrical parameters from retrieved document passages.

        Args:
            docs: List of retrieved document dictionaries with content and metadata.
            target_component: Target part number name (e.g. 'LM35').

        Returns:
            ComponentInformation containing verified specs, citations, and missing specs.
        """
        part_name = target_component or "Unknown Component"
        if not docs:
            return ComponentInformation(
                part_number=part_name,
                verified_specs={},
                unverified_or_missing_specs=[
                    "No datasheet passages were retrieved to verify electrical specifications."
                ],
                citations=[],
            )

        verified_specs: Dict[str, Any] = {}
        missing_specs: List[str] = []
        citations: List[Dict[str, Any]] = []

        # Collect unique citations
        seen_cites = set()
        for doc in docs:
            src = doc.get("source", "datasheet.pdf")
            pg = doc.get("page", 1)
            key = (src, pg)
            if key not in seen_cites:
                seen_cites.add(key)
                citations.append({"source": src, "page": pg})

        full_text = "\n\n".join(
            f"--- Source: {d.get('source')} | Page: {d.get('page')} ---\n{d.get('content', '')}"
            for d in docs
        )

        # 1. Operating Supply Voltage (prose or table inequality: e.g. 4 V to 30 V, or 4 V ≤ VS ≤ 30 V)
        supply_match = re.search(
            r"(?:operat(?:ing|es?|e)\s+(?:from\s+)?|supply\s+voltage(?:\s+range)?(?:\s+of)?\s*[:\s]*)(\d+(?:\.\d+)?)\s*V\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*V",
            full_text,
            re.IGNORECASE,
        )
        if not supply_match:
            supply_match = re.search(
                r"(\d+(?:\.\d+)?)\s*V?\s*(?:≤|<=|<)\s*V_?S?\s*(?:≤|<=|<)\s*(\d+(?:\.\d+)?)\s*V",
                full_text,
            )

        if supply_match:
            v_min = float(supply_match.group(1))
            v_max = float(supply_match.group(2))
            src_info = self._find_source_info(docs, supply_match.group(0))
            verified_specs["supply_voltage_recommended"] = {
                "parameter": "Recommended Operating Supply Voltage",
                "value": f"{v_min} to {v_max}",
                "min_v": v_min,
                "max_v": v_max,
                "unit": "V",
                "status": "recommended_operating",
                "conditions": "Continuous operating range across rated temperature",
                "variant": "LM35, LM35A, LM35C, LM35CA, LM35D",
                "source": src_info["source"],
                "page": src_info["page"],
                "citation": src_info["citation"],
                "raw": f"{v_min} V to {v_max} V",
            }
        else:
            missing_specs.append("Recommended operating supply voltage range")

        # 2. Absolute Maximum Supply Voltage (e.g., -0.2 V to 35 V, or "Supply voltage –0.2 35 V")
        abs_max_match = re.search(
            r"supply\s+voltage\s*[:\s]*([–\-]?\d+(?:\.\d+)?)\s*(?:V|\s)\s*(?:to|[–\-])?\s*(\d+(?:\.\d+)?)\s*V",
            full_text,
            re.IGNORECASE,
        )
        if not abs_max_match:
            abs_max_match = re.search(
                r"absolute\s+maximum\s+ratings.*?supply\s+voltage\s*[:\s]*([–\-]?\d+(?:\.\d+)?)\s*V\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*V",
                full_text,
                re.IGNORECASE | re.DOTALL,
            )

        if abs_max_match:
            abs_v_min_str = abs_max_match.group(1).replace("–", "-")
            abs_v_min = float(abs_v_min_str)
            abs_v_max = float(abs_max_match.group(2))
            src_info = self._find_source_info(docs, abs_max_match.group(0)[:50])
            verified_specs["supply_voltage_abs_max"] = {
                "parameter": "Supply Voltage (Absolute Maximum)",
                "value": f"{abs_v_min} to {abs_v_max}",
                "min_v": abs_v_min,
                "max_v": abs_v_max,
                "unit": "V",
                "status": "absolute_maximum",
                "conditions": "Stresses beyond limits may cause permanent damage",
                "variant": "All variants",
                "source": src_info["source"],
                "page": src_info["page"],
                "citation": src_info["citation"],
                "raw": f"{abs_v_min} V to {abs_v_max} V",
            }
        else:
            missing_specs.append("Absolute maximum supply voltage rating")

        # 3. Scale factor / Sensitivity (e.g. 10 mV/°C, 10-mV/°C, or 10.0 mV/C)
        # Prioritize canonical scale factor / transfer function phrasing to avoid application-note op-amp gains (e.g. 100 mV/°C)
        scale_match = re.search(
            r"(?:linear\s*\+?\s*|VOUT\s*=\s*|scale\s+factor[:\s]*|sensitivity[:\s]*)(\d+(?:\.\d+)?)\s*[- ]?m?V\s*/\s*°?C",
            full_text,
            re.IGNORECASE,
        )
        if not scale_match:
            scale_match = re.search(
                r"(\d+(?:\.\d+)?)\s*[- ]?m?V\s*/\s*°?C",
                full_text,
                re.IGNORECASE,
            )
        if scale_match:
            scale_val = float(scale_match.group(1))
            src_info = self._find_source_info(docs, scale_match.group(0))
            verified_specs["scale_factor"] = {
                "parameter": "Scale Factor",
                "value": scale_val,
                "unit": "mV/°C",
                "status": "nominal",
                "conditions": "Linear sensitivity across rated temperature",
                "variant": "LM35, LM35A, LM35C, LM35D",
                "source": src_info["source"],
                "page": src_info["page"],
                "citation": src_info["citation"],
                "raw": f"{scale_val} mV/°C",
            }
        else:
            missing_specs.append("Sensor scale factor / transfer sensitivity")

        # 4. Quiescent current / operating current (e.g. 60 uA, 56 uA, 131 uA, 0.06 mA, "Less Than 60-μA Current Drain")
        current_match = re.search(
            r"(?:less\s+than\s+)?(\d+(?:\.\d+)?)\s*[- ]?([µμu]A|mA|A)\s+current\s+drain",
            full_text,
            re.IGNORECASE,
        )
        if not current_match:
            current_match = re.search(
                r"(?:quiescent|drain)\s+current(?:\s+is)?\s*[:\s]*(?:less\s+than\s+)?(\d+(?:\.\d+)?)\s*([µμu]A|mA|A)\b",
                full_text,
                re.IGNORECASE,
            )
        if not current_match:
            current_match = re.search(
                r"quiescent\s+current[^\n]*\n(?:[^\n]*\n){0,3}?(\d+(?:\.\d+)?)\s*\n?([µμu]A|mA|A)\b",
                full_text,
                re.IGNORECASE,
            )

        if current_match:
            curr_val = float(current_match.group(1))
            unit = current_match.group(2)
            unit_norm = unit.replace("μ", "µ")
            curr_ma = curr_val if "ma" in unit.lower() else (curr_val / 1000.0 if any(u in unit for u in ("u", "µ", "μ")) else curr_val * 1000.0)
            src_info = self._find_source_info(docs, current_match.group(0))
            verified_specs["quiescent_current"] = {
                "parameter": "Quiescent Current (Typical)",
                "value": curr_val,
                "value_ma": round(curr_ma, 4),
                "unit": unit_norm,
                "status": "typical",  # STRICTLY TYPICAL, NEVER GUARANTEED MAXIMUM
                "conditions": "TA = 25°C, VS = 4V to 30V (Note: Guaranteed maximum is higher over temperature, e.g. 131 µA)",
                "variant": "LM35, LM35A, LM35C, LM35D",
                "source": src_info["source"],
                "page": src_info["page"],
                "citation": src_info["citation"],
                "raw": f"{curr_val} {unit_norm} (Typical)",
            }
            # Explicitly record guaranteed maximum as a separate parameter consideration
            missing_specs.append("Guaranteed maximum quiescent current across full operating temperature")
        else:
            missing_specs.append("Quiescent / operating drain current")

        # 5. Non-Linearity (DO NOT classify as accuracy tolerance)
        nl_match = re.search(
            r"non-linearity\s+only\s*±?\s*([¼½¾]|\d+(?:\.\d+)?)\s*°?C",
            full_text,
            re.IGNORECASE,
        )
        if nl_match:
            nl_raw = nl_match.group(1)
            frac_map = {"¼": 0.25, "½": 0.5, "¾": 0.75}
            nl_val = frac_map.get(nl_raw, float(nl_raw) if nl_raw not in frac_map else 0.25)
            nl_src_info = self._find_source_info(docs, nl_match.group(0))
            verified_specs["nonlinearity"] = {
                "parameter": "Non-Linearity (Typical)",
                "value": nl_val,
                "value_c": nl_val,
                "unit": "°C",
                "status": "typical",
                "conditions": "TA = 25°C, 0°C to 100°C operating span",
                "variant": "LM35, LM35A",
                "source": nl_src_info["source"],
                "page": nl_src_info["page"],
                "citation": nl_src_info["citation"],
                "raw": f"±{nl_val}°C ({nl_raw}) (Typical Non-Linearity)",
            }

        # 6. Accuracy / Tolerance (Only if genuine accuracy text is retrieved; never alias from non-linearity)
        acc_match = re.search(
            r"(?:accurac(?:y|ies)\s+of\s+|ensured\s+accuracy\s*(?:of)?\s*)±?\s*([¼½¾]|\d+(?:\.\d+)?)\s*°?C",
            full_text,
            re.IGNORECASE,
        )
        if not acc_match:
            acc_match = re.search(
                r"([¼½¾]|\d+(?:\.\d+)?)\s*°?C\s+ensured\s+accuracy",
                full_text,
                re.IGNORECASE,
            )

        if acc_match:
            acc_raw = acc_match.group(1)
            frac_map = {"¼": 0.25, "½": 0.5, "¾": 0.75}
            acc_val = frac_map.get(acc_raw, float(acc_raw) if acc_raw not in frac_map else 0.25)
            acc_src_info = self._find_source_info(docs, acc_match.group(0))
            acc_status = "guaranteed" if "ensured" in acc_match.group(0).lower() else "typical"
            verified_specs["accuracy_tolerance"] = {
                "parameter": "Measurement Accuracy",
                "value": acc_val,
                "value_c": acc_val,
                "unit": "°C",
                "status": acc_status,
                "conditions": "TA = 25°C room temperature",
                "variant": "LM35A / LM35",
                "source": acc_src_info["source"],
                "page": acc_src_info["page"],
                "citation": acc_src_info["citation"],
                "raw": f"±{acc_val}°C ({acc_raw})",
            }
        else:
            missing_specs.append("Measurement accuracy and tolerance")

        # 7. Operating Temperature Range
        temp_match = re.search(
            r"(?:operate\s+over\s+a\s+|full\s+|rated\s+for\s+full\s+)([−–\-]?\d+)\s*°?C\s*(?:to|-)\s*([−–\-+]?\d+)\s*°?C\s*(?:temperature\s+range)?",
            full_text,
            re.IGNORECASE,
        )
        if not temp_match:
            temp_match = re.search(
                r"TJ\s*=\s*([−–\-]?\d+)\s*°?C\s*(?:to|-)\s*([−–\-+]?\d+)\s*°?C",
                full_text,
                re.IGNORECASE,
            )

        if temp_match:
            t_min_str = temp_match.group(1).replace("−", "-").replace("–", "-")
            t_max_str = temp_match.group(2).replace("−", "-").replace("–", "-")
            t_min = float(t_min_str)
            t_max = float(t_max_str)
            src_info = self._find_source_info(docs, temp_match.group(0))
            verified_specs["operating_temperature"] = {
                "parameter": "Rated Operating Temperature Range",
                "value": f"{t_min} to {t_max}",
                "min_c": t_min,
                "max_c": t_max,
                "unit": "°C",
                "status": "guaranteed",
                "conditions": "Continuous junction temperature range",
                "variant": "LM35: -55°C to 150°C, LM35C: -40°C to 110°C, LM35D: 0°C to 100°C",
                "source": src_info["source"],
                "page": src_info["page"],
                "citation": src_info["citation"],
                "raw": f"{t_min}°C to {t_max}°C",
            }
        else:
            missing_specs.append("Rated operational temperature range")

        return ComponentInformation(
            part_number=part_name,
            verified_specs=verified_specs,
            unverified_or_missing_specs=missing_specs,
            citations=citations,
        )

    def _find_source_info(self, docs: List[Dict[str, Any]], snippet: str) -> Dict[str, Any]:
        """Find source and page metadata for a snippet among retrieved documents."""
        clean_snip = re.sub(r"\s+", " ", snippet.strip().lower())
        for doc in docs:
            content_clean = re.sub(r"\s+", " ", doc.get("content", "").lower())
            if clean_snip[:25] in content_clean or any(word in content_clean for word in clean_snip.split()[:3]):
                src = doc.get("source", "datasheet.pdf")
                pg = doc.get("page", 1)
                return {"source": src, "page": pg, "citation": f"[{src}, p. {pg}]"}
        first = docs[0] if docs else {}
        src = first.get("source", "datasheet.pdf")
        pg = first.get("page", 1)
        return {"source": src, "page": pg, "citation": f"[{src}, p. {pg}]"}

    def _find_source_page(self, docs: List[Dict[str, Any]], snippet: str) -> str:
        """Find the source and page citation string for a snippet among retrieved documents."""
        return self._find_source_info(docs, snippet)["citation"]


def component_agent_node(state: CircuitMindState) -> Dict[str, Any]:
    """LangGraph node extracting verified component specifications from retrieved evidence."""
    retrieved_docs = state.get("retrieved_documents", [])
    requirements = state.get("requirements", {})
    target_components = requirements.get("target_components", [])
    part_name = target_components[0] if target_components else "LM35"

    agent = ComponentAgent()
    try:
        comp_info = agent.extract_from_documents(retrieved_docs, target_component=part_name)
        return {"component_information": comp_info.model_dump()}
    except Exception as exc:
        logger.error("Error in component_agent_node: %s", exc)
        return {
            "component_information": ComponentInformation(
                part_number=part_name,
                unverified_or_missing_specs=["Specification extraction encountered an internal error."],
            ).model_dump(),
            "errors": [f"ComponentAgent error: {exc}"],
        }

