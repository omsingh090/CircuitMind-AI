"""Unit tests for CircuitMind AI Specialized Agent Nodes.

Tests:
1. RequirementAgent parameter extraction and missing requirement identification.
2. ResearchAgent hybrid retrieval execution and metadata preservation.
3. ComponentAgent specification extraction and citation mapping.
4. ValidationAgent deterministic electrical checks (pass and violation scenarios).
5. ReportAgent structured engineering report synthesis and mandatory disclaimer.
"""

from unittest.mock import MagicMock
import pytest
from langchain_core.documents import Document

from app.agents.component_agent import ComponentAgent, component_agent_node
from app.agents.report_agent import MANDATORY_DISCLAIMER, ReportAgent, report_agent_node
from app.agents.requirement_agent import RequirementAgent, requirement_agent_node
from app.agents.research_agent import ResearchAgent, research_agent_node
from app.agents.validation_agent import ValidationAgent, validation_agent_node
from app.graph.state import CircuitMindState


class TestRequirementAgent:
    """Tests for RequirementAgent."""

    def test_extract_explicit_requirements(self) -> None:
        """Verify extraction of component, voltage, current, and temperature constraints."""
        agent = RequirementAgent()
        query = "I want to design a 5V circuit with LM35 drawing 50mA from -40C to 125C."
        reqs = agent.extract(query)

        assert "LM35" in reqs.target_components
        assert reqs.target_supply_voltage_v == 5.0
        assert reqs.target_max_current_ma == 50.0
        assert reqs.min_temp_c == -40.0
        assert reqs.max_temp_c == 125.0
        assert len(reqs.missing_requirements) > 0  # e.g., accuracy unstated

    def test_identify_missing_requirements(self) -> None:
        """Verify that unspecified parameters are explicitly identified without guessing."""
        agent = RequirementAgent()
        query = "Can I use LM35 for temperature monitoring?"
        reqs = agent.extract(query)

        assert "LM35" in reqs.target_components
        assert reqs.target_supply_voltage_v is None
        assert reqs.target_max_current_ma is None
        assert any("Supply voltage" in m for m in reqs.missing_requirements)
        assert any("Operating current" in m for m in reqs.missing_requirements)

    def test_extract_adc_reference_disambiguated_from_sensor_supply(self) -> None:
        """Verify ADC reference voltage (0.8V) is not conflated with sensor supply voltage (5V)."""
        agent = RequirementAgent()
        query = (
            "Design an LM35 system:\n"
            "- Sensor supply: 5V\n"
            "- ADC reference: 0.8V\n"
            "- ADC input range: 0V to 0.8V\n"
            "- Temperature: 0°C to 100°C"
        )
        reqs = agent.extract(query)

        assert reqs.target_supply_voltage_v == 5.0
        assert reqs.adc_reference_voltage_v == 0.8
        assert reqs.min_temp_c == 0.0
        assert reqs.max_temp_c == 100.0

    def test_extract_adc_reference_when_stated_before_supply(self) -> None:
        """Verify supply voltage and ADC reference are correctly parsed regardless of order."""
        agent = RequirementAgent()
        query = "ADC reference: 0.8V, Sensor supply: 5V, 10k load"
        reqs = agent.extract(query)

        assert reqs.target_supply_voltage_v == 5.0
        assert reqs.adc_reference_voltage_v == 0.8
        assert reqs.load_impedance_ohms == 10000.0

    def test_requirement_agent_node(self) -> None:
        """Verify requirement_agent_node integration into state dictionary."""
        state: CircuitMindState = {
            "user_query": "5V LM35 sensor circuit",
            "requirements": {},
            "retrieved_documents": [],
            "component_information": {},
            "validation_results": {},
            "engineering_report": {},
            "errors": [],
        }
        res = requirement_agent_node(state)
        assert "requirements" in res
        assert "LM35" in res["requirements"]["target_components"]
        assert res["requirements"]["target_supply_voltage_v"] == 5.0


class TestResearchAgent:
    """Tests for ResearchAgent."""

    def test_research_agent_retrieval_and_metadata_preservation(self) -> None:
        """Verify ResearchAgent calls hybrid retriever and preserves document metadata."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve_hybrid.return_value = [
            Document(
                page_content="LM35 operates from 4 V to 30 V.",
                metadata={"source": "lm35.pdf", "page": 19, "chunk_id": "lm35_p19_001", "rrf_score": 0.033},
            )
        ]

        agent = ResearchAgent(retriever=mock_retriever)
        docs = agent.retrieve("LM35 supply voltage", target_components=["LM35"], k=2)

        assert len(docs) == 1
        assert docs[0]["source"] == "lm35.pdf"
        assert docs[0]["page"] == 19
        assert docs[0]["chunk_id"] == "lm35_p19_001"
        assert docs[0]["rrf_score"] == 0.033
        assert "4 V to 30 V" in docs[0]["content"]

    def test_research_agent_empty_query(self) -> None:
        """Verify empty query returns empty document list without error."""
        agent = ResearchAgent()
        assert agent.retrieve("") == []
        assert agent.retrieve("   ") == []


class TestComponentAgent:
    """Tests for ComponentAgent."""

    def test_component_agent_extract_specifications_with_citations(self) -> None:
        """Verify extraction of operating voltage, absolute maximums, and sensitivity with citations."""
        docs = [
            {
                "content": "The LM35 device has a recommended supply voltage of 4 V to 30 V.",
                "source": "lm35.pdf",
                "page": 19,
            },
            {
                "content": "Absolute Maximum Ratings: Supply voltage -0.2 V to 35 V.",
                "source": "lm35.pdf",
                "page": 4,
            },
            {
                "content": "The scale factor is 10 mV/°C. The quiescent current is 60 uA at 25°C.",
                "source": "lm35.pdf",
                "page": 1,
            },
        ]

        agent = ComponentAgent()
        info = agent.extract_from_documents(docs, target_component="LM35")

        assert info.part_number == "LM35"
        assert "supply_voltage_recommended" in info.verified_specs
        assert info.verified_specs["supply_voltage_recommended"]["min_v"] == 4.0
        assert info.verified_specs["supply_voltage_recommended"]["max_v"] == 30.0
        assert "[lm35.pdf, p. 19]" in info.verified_specs["supply_voltage_recommended"]["citation"]

        assert "supply_voltage_abs_max" in info.verified_specs
        assert info.verified_specs["supply_voltage_abs_max"]["max_v"] == 35.0

        assert "scale_factor" in info.verified_specs
        assert info.verified_specs["scale_factor"]["value"] == 10.0

        assert "quiescent_current" in info.verified_specs
        assert info.verified_specs["quiescent_current"]["value_ma"] == 0.06

        assert len(info.citations) >= 3

    def test_component_agent_empty_documents(self) -> None:
        """Verify empty document list registers missing specifications."""
        agent = ComponentAgent()
        info = agent.extract_from_documents([], target_component="LM35")
        assert len(info.verified_specs) == 0
        assert len(info.unverified_or_missing_specs) > 0


class TestValidationAgent:
    """Tests for ValidationAgent."""

    def test_validation_agent_compatible_circuit(self) -> None:
        """Verify 5V operation on LM35 passes deterministic validation with power calculation."""
        reqs = {
            "target_components": ["LM35"],
            "target_supply_voltage_v": 5.0,
            "target_max_current_ma": 10.0,
            "adc_reference_voltage_v": 5.0,
            "load_impedance_ohms": 10000.0,
        }
        comp_info = {
            "part_number": "LM35",
            "verified_specs": {
                "supply_voltage_recommended": {"min_v": 4.0, "max_v": 30.0},
                "supply_voltage_abs_max": {"max_v": 35.0},
                "quiescent_current": {"value_ma": 0.06},
            },
        }

        agent = ValidationAgent()
        val = agent.validate(requirements=reqs, component_info=comp_info)

        assert val.is_valid is True
        assert len(val.violations) == 0
        assert "quiescent_power" in val.calculations
        assert val.calculations["quiescent_power"]["power_mw"] == pytest.approx(0.3, rel=1e-2)  # 5V * 0.06mA = 0.3mW

    def test_validation_agent_undervoltage_violation(self) -> None:
        """Verify 3.3V operating voltage fails validation when minimum is 4.0V."""
        reqs = {
            "target_components": ["LM35"],
            "target_supply_voltage_v": 3.3,
        }
        comp_info = {
            "part_number": "LM35",
            "verified_specs": {
                "supply_voltage_recommended": {"min_v": 4.0, "max_v": 30.0},
            },
        }

        agent = ValidationAgent()
        val = agent.validate(requirements=reqs, component_info=comp_info)

        assert val.is_valid is False
        assert any("UNDER-VOLTAGE" in v for v in val.violations)


class TestReportAgent:
    """Tests for ReportAgent."""

    def test_report_agent_complete_report_generation(self) -> None:
        """Verify ReportAgent compiles executive summary, specifications, calculations, and disclaimer."""
        state: CircuitMindState = {
            "user_query": "Design a 5V LM35 circuit",
            "requirements": {
                "target_components": ["LM35"],
                "target_supply_voltage_v": 5.0,
                "identified_requirements": {"target_supply_voltage_v": 5.0},
                "missing_requirements": ["Load impedance not specified"],
            },
            "retrieved_documents": [
                {"source": "lm35.pdf", "page": 19, "content": "Operating supply voltage: 4V to 30V."}
            ],
            "component_information": {
                "part_number": "LM35",
                "verified_specs": {
                    "supply_voltage": {"raw": "4V to 30V", "citation": "[lm35.pdf, p. 19]"}
                },
                "unverified_or_missing_specs": ["Thermal resistance"],
                "citations": [{"source": "lm35.pdf", "page": 19}],
            },
            "validation_results": {
                "is_valid": True,
                "checks": [{"check_name": "Voltage Check", "status": "PASS"}],
                "violations": [],
                "warnings": [],
                "calculations": {"quiescent_power": {"power_mw": 0.3, "formula": "P = V * I"}},
            },
            "engineering_report": {},
            "errors": [],
        }

        agent = ReportAgent()
        report = agent.generate_report(state)

        assert "LM35" in report.title
        assert "LM35" in report.executive_summary
        assert "5.0" in report.requirements_summary
        assert "Load impedance not specified" in report.requirements_summary
        assert "[lm35.pdf, p. 19]" in report.component_specifications
        assert "0.3 mW" in report.validation_summary
        assert report.fabrication_disclaimer == MANDATORY_DISCLAIMER
        assert len(report.citations) == 1

