"""Requirement Agent for CircuitMind AI.

Analyzes raw user engineering queries to extract design requirements, operating constraints,
and component targets.
Strictly identifies unstated or missing parameters without inventing or assuming values.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import errors, types

from app.config import get_settings
from app.graph.state import CircuitMindState, ExtractedRequirements

logger = logging.getLogger(__name__)

REQUIREMENT_SYSTEM_PROMPT = """You are CircuitMind AI's Engineering Requirements Analyst.
Extract technical parameters from the user's circuit query into a JSON object.

CRITICAL RULES:
1. ONLY extract parameters explicitly stated in the query. NEVER assume or invent missing values.
2. Identify missing engineering parameters explicitly (e.g., supply voltage, max current, operating temperature, accuracy).
3. Return only valid JSON with keys:
   - "target_components": list of component part numbers (e.g., ["LM35"])
   - "target_supply_voltage_v": float or null
   - "target_max_current_ma": float or null
   - "min_temp_c": float or null
   - "max_temp_c": float or null
   - "identified_requirements": object of key-value pairs
   - "missing_requirements": list of unstated parameters
"""


class RequirementAgent:
    """Agent responsible for parsing user queries into structured engineering requirements."""

    def __init__(self, gemini_client: Optional[genai.Client] = None) -> None:
        """Initialize RequirementAgent with optional Gemini client for complex language parsing."""
        self.settings = get_settings()
        self.client = gemini_client

    def extract_deterministic(self, query: str) -> ExtractedRequirements:
        """Deterministically extract electrical parameters using robust regex pattern matching.

        Ensures that parameter extraction functions independently of LLM availability.
        """
        identified: Dict[str, Any] = {}
        missing: List[str] = []

        # 1. Component Part Numbers (e.g., LM35, LM358, NE555, STM32, ESP32, etc.)
        part_pattern = re.compile(r"\b([A-Z]{1,4}\d{2,5}[A-Z\d]*)\b", re.IGNORECASE)
        found_parts = [p.upper() for p in part_pattern.findall(query) if len(p) >= 3]
        target_components = list(dict.fromkeys(found_parts))  # preserve order & unique

        if target_components:
            identified["target_components"] = target_components
        else:
            missing.append("Specific component part number not specified")

        # 2. ADC interface constraints (e.g., "ADC reference: 0.8V", "0V to 0.8V ADC range", "Vref = 1.2V", "0.8V ADC reference")
        # Explicit anchors prevent matching sensor supply voltage as ADC reference
        p_adc_prefix = re.compile(
            r"(?:adc\s+(?:reference|ref|input(?:\s+range)?|full-?scale)?|vref|reference\s+voltage)\s*[:=]?\s*(?:(?:0\s*v?\s*(?:to|-)\s*)?(\d+(?:\.\d+)?)\s*v|\b(\d+(?:\.\d+)?)\s*v\b)",
            re.IGNORECASE,
        )
        p_adc_suffix = re.compile(
            r"\b(\d+(?:\.\d+)?)\s*v\s*(?:adc\s+(?:reference|ref|input)?|vref)(?![:=])\b",
            re.IGNORECASE,
        )
        adc_match = p_adc_prefix.search(query) or p_adc_suffix.search(query)
        adc_ref_v: Optional[float] = None
        adc_span = (-1, -1)
        if adc_match:
            try:
                if adc_match.re == p_adc_prefix:
                    adc_ref_v = float(adc_match.group(1) or adc_match.group(2))
                else:
                    adc_ref_v = float(adc_match.group(1))
                identified["adc_reference_voltage_v"] = adc_ref_v
                adc_span = adc_match.span()
            except (ValueError, TypeError):
                pass
        else:
            missing.append("ADC reference voltage / input voltage range not specified")

        # 3. Supply voltage extraction (e.g., "Sensor supply: 5V", "5V supply", "Vcc = 5V", "5V circuit")
        # First check explicit supply keywords to prevent colliding with ADC reference voltages
        p_supply_prefix = re.compile(
            r"(?:(?:sensor\s+)?supply(?:\s+voltage)?|vcc|vdd|v_s|vs|power\s+supply|powered\s+(?:by|from)|operating\s+voltage)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:v(?:olts?)?)\b",
            re.IGNORECASE,
        )
        p_supply_suffix = re.compile(
            r"\b(\d+(?:\.\d+)?)\s*(?:v(?:olts?)?)\s+(?:sensor\s+)?(?:supply|circuit|power)\b",
            re.IGNORECASE,
        )
        supply_match = p_supply_prefix.search(query) or p_supply_suffix.search(query)
        target_voltage: Optional[float] = None

        if supply_match:
            try:
                target_voltage = float(supply_match.group(1))
                identified["target_supply_voltage_v"] = target_voltage
            except ValueError:
                pass
        else:
            # Fallback to general voltage only if outside the ADC reference span
            p_any_v = re.compile(r"(\d+(?:\.\d+)?)\s*(?:v(?:olts?)?)\b", re.IGNORECASE)
            for m in p_any_v.finditer(query):
                start, end = m.span()
                if not (adc_span[0] <= start and end <= adc_span[1]):
                    try:
                        target_voltage = float(m.group(1))
                        identified["target_supply_voltage_v"] = target_voltage
                        break
                    except ValueError:
                        pass

        if target_voltage is None:
            missing.append("Supply voltage requirement not specified")

        # 3. Current extraction (e.g., 50mA, 0.5A, 100 uA, 10 Amperes)
        current_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(ma|a|ua|amperes?)\b", re.IGNORECASE)
        curr_match = current_pattern.search(query)
        target_current_ma: Optional[float] = None
        if curr_match:
            try:
                val = float(curr_match.group(1))
                unit = curr_match.group(2).lower()
                if "ma" in unit:
                    target_current_ma = val
                elif "ua" in unit:
                    target_current_ma = val / 1000.0
                else:  # Amperes
                    target_current_ma = val * 1000.0
                identified["target_max_current_ma"] = target_current_ma
            except ValueError:
                pass
        else:
            missing.append("Operating current / load consumption not specified")

        # 4. Temperature range extraction (e.g. -40 to 125 C, 0C to 70C)
        temp_pattern = re.compile(
            r"(-?\d+(?:\.\d+)?)\s*(?:°?c)?\s*(?:to|-)\s*(-?\d+(?:\.\d+)?)\s*°?c",
            re.IGNORECASE,
        )
        temp_match = temp_pattern.search(query)
        min_temp: Optional[float] = None
        max_temp: Optional[float] = None
        if temp_match:
            try:
                min_temp = float(temp_match.group(1))
                max_temp = float(temp_match.group(2))
                identified["min_temp_c"] = min_temp
                identified["max_temp_c"] = max_temp
            except ValueError:
                pass
        else:
            missing.append("Operating temperature range not specified")

        # 6. Interface / Accuracy
        if "accuracy" not in query.lower() and "tolerance" not in query.lower():
            missing.append("Required accuracy or sensor tolerance not specified")

        # 7. Output load impedance constraints
        load_pattern = re.compile(
            r"(?:(\d+(?:\.\d+)?)\s*(k?ohm|k?Ω|k)\s*(?:load|resistor|impedance)|(?:load|resistor|impedance)\s*(?:of|is|:|=)?\s*(\d+(?:\.\d+)?)\s*(k?ohm|k?Ω|k)?\b)",
            re.IGNORECASE,
        )
        load_match = load_pattern.search(query)
        load_ohms: Optional[float] = None
        if load_match:
            try:
                val_str = load_match.group(1) or load_match.group(3)
                unit_str = (load_match.group(2) or load_match.group(4) or "").lower()
                if val_str:
                    val = float(val_str)
                    load_ohms = val * 1000.0 if "k" in unit_str else val
                    identified["load_impedance_ohms"] = load_ohms
            except ValueError:
                pass
        if load_ohms is None:
            missing.append("Load impedance / output termination constraint not specified")

        return ExtractedRequirements(
            target_components=target_components,
            target_supply_voltage_v=target_voltage,
            target_max_current_ma=target_current_ma,
            min_temp_c=min_temp,
            max_temp_c=max_temp,
            adc_reference_voltage_v=adc_ref_v,
            load_impedance_ohms=load_ohms,
            identified_requirements=identified,
            missing_requirements=missing,
        )

    def extract(self, query: str) -> ExtractedRequirements:
        """Extract requirements from query using deterministic analysis."""
        if not query or not query.strip():
            return ExtractedRequirements(
                missing_requirements=["User engineering query was empty."]
            )

        # Baseline deterministic extraction
        reqs = self.extract_deterministic(query)
        return reqs


def requirement_agent_node(state: CircuitMindState) -> Dict[str, Any]:
    """LangGraph node function executing requirements extraction on user query."""
    user_query = state.get("user_query", "")
    agent = RequirementAgent()
    try:
        extracted = agent.extract(user_query)
        return {"requirements": extracted.model_dump()}
    except Exception as exc:
        logger.error("Error in requirement_agent_node: %s", exc)
        return {
            "requirements": ExtractedRequirements(
                missing_requirements=["Failed to parse requirements."]
            ).model_dump(),
            "errors": [f"RequirementAgent error: {exc}"],
        }

