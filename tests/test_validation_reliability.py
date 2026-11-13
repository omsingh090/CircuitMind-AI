"""Tests for CircuitMind AI validation reliability and deterministic calculations.

Covers:
1. Empty checks list returns INCONCLUSIVE (never VALID).
2. Failed checks or violations return INVALID (is_valid=False).
3. Missing evidence returns INCONCLUSIVE.
4. Deterministic calculation for LM35 output at 25°C = 250 mV.
5. Deterministic calculation for LM35 output at 100°C = 1000 mV (1.0 V).
6. Every extracted specification has a source and page citation.
7. Retrieval duplicate suppression and page diversity limits.
8. Engineering report status agrees with validation status (VALID/INVALID/INCONCLUSIVE).
"""

from typing import List
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from app.agents.component_agent import ComponentAgent
from app.agents.report_agent import ReportAgent
from app.agents.requirement_agent import RequirementAgent
from app.agents.validation_agent import ValidationAgent
from app.graph.state import CircuitMindState
from app.rag.retrieval import DatasheetRetriever
from app.tools.electrical_calculator import sensor_transfer_output


class TestValidationReliability:
    """Test suite for safety-critical electrical validation logic."""

    def test_validation_no_checks_returns_inconclusive(self) -> None:
        """Requirement 1: Empty checks list must return status='INCONCLUSIVE' and is_valid=False."""
        agent = ValidationAgent()
        # No supply voltage requirement or component specs provided -> no checks can be performed
        reqs = {"target_components": ["LM35"]}
        comp_info = {"part_number": "LM35", "verified_specs": {}}

        result = agent.validate(requirements=reqs, component_info=comp_info)

        assert result.status == "INCONCLUSIVE"
        assert result.is_valid is False
        assert len(result.checks) == 0
        assert any("INCONCLUSIVE" in w or "missing parameters" in w.lower() for w in result.warnings)

    def test_validation_failed_checks_returns_invalid(self) -> None:
        """Requirement 2: Overvoltage or failed checks must return status='INVALID' and is_valid=False."""
        agent = ValidationAgent()
        # LM35 max recommended is 30V; absolute max 35V. Apply 36V.
        reqs = {
            "target_components": ["LM35"],
            "target_supply_voltage_v": 36.0,
        }
        comp_info = {
            "part_number": "LM35",
            "verified_specs": {
                "supply_voltage_recommended": {"min_v": 4.0, "max_v": 30.0},
                "supply_voltage_abs_max": {"max_v": 35.0},
            },
        }

        result = agent.validate(requirements=reqs, component_info=comp_info)

        assert result.status == "INVALID"
        assert result.is_valid is False
        assert len(result.violations) > 0
        assert any("absolute maximum" in v.lower() or "critical violation" in v.lower() for v in result.violations)

    def test_validation_missing_evidence_returns_inconclusive(self) -> None:
        """Requirement 3: Missing supply spec prevents full validation and returns INCONCLUSIVE."""
        agent = ValidationAgent()
        # User specified 5V supply, but datasheet evidence has no supply voltage range
        reqs = {
            "target_components": ["LM35"],
            "target_supply_voltage_v": 5.0,
        }
        comp_info = {
            "part_number": "LM35",
            "verified_specs": {
                "scale_factor": {"value": 10.0, "unit": "mV/°C"},
            },
        }

        result = agent.validate(requirements=reqs, component_info=comp_info)

        assert result.status == "INCONCLUSIVE"
        assert result.is_valid is False
        assert len(result.checks) == 0
        assert any("verified" in w.lower() or "inconclusive" in w.lower() for w in result.warnings)

    def test_deterministic_calculation_25c(self) -> None:
        """Requirement 4: Deterministic calculation for 25°C gives exactly 250 mV (0.25 V)."""
        calc = sensor_transfer_output(
            temperature_c=25.0,
            scale_factor_mv_per_c=10.0,
            offset_v=0.0,
        )

        assert calc["temperature_c"] == 25.0
        assert calc["scale_factor_mv_per_c"] == 10.0
        assert calc["output_voltage_mv"] == pytest.approx(250.0, rel=1e-5)
        assert calc["output_voltage_v"] == pytest.approx(0.25, rel=1e-5)
        assert "scale_factor" in calc["formula"]

    def test_deterministic_calculation_100c(self) -> None:
        """Requirement 5: Deterministic calculation for 100°C gives exactly 1000 mV (1.0 V)."""
        calc = sensor_transfer_output(
            temperature_c=100.0,
            scale_factor_mv_per_c=10.0,
            offset_v=0.0,
        )

        assert calc["temperature_c"] == 100.0
        assert calc["output_voltage_mv"] == pytest.approx(1000.0, rel=1e-5)
        assert calc["output_voltage_v"] == pytest.approx(1.0, rel=1e-5)

    def test_every_extracted_specification_has_citation(self) -> None:
        """Requirement 6: Every verified spec has valid source and page citations."""
        docs = [
            {
                "content": "The LM35 operates from 4 V to 30 V. Less than 60-µA current drain.",
                "source": "lm35.pdf",
                "page": 1,
            },
            {
                "content": "Absolute Maximum Ratings: Supply voltage –0.2 to 35 V. Rated operating temp −55°C to 150°C.",
                "source": "lm35.pdf",
                "page": 4,
            },
            {
                "content": "Linearly + 10-mV/°C Scale Factor. Typical accuracy of ±0.25°C.",
                "source": "lm35.pdf",
                "page": 1,
            },
        ]

        agent = ComponentAgent()
        comp_info = agent.extract_from_documents(docs, target_component="LM35")

        assert len(comp_info.verified_specs) >= 4
        for spec_name, spec_data in comp_info.verified_specs.items():
            assert "citation" in spec_data, f"Spec {spec_name} missing citation field"
            assert "source" in spec_data, f"Spec {spec_name} missing source field"
            assert "page" in spec_data, f"Spec {spec_name} missing page field"
            assert spec_data["source"] == "lm35.pdf"
            assert spec_data["page"] in (1, 4)
            assert spec_data["citation"].startswith("[lm35.pdf, p.")

    def test_duplicate_retrieval_suppression(self) -> None:
        """Requirement 7: Near-duplicate chunks and single-page clustering are constrained."""
        retriever = DatasheetRetriever()

        # Mock ChromaDB semantic search and BM25 search with heavily skewed duplicates on page 15
        retriever.vector_store = MagicMock()
        retriever.bm25_retriever = MagicMock()

        # Page 15 has 4 near-duplicate chunks, page 1 has 1, page 4 has 1
        mock_docs = [
            Document(page_content="LM35 layout and thermal monitoring system considerations.", metadata={"source": "lm35.pdf", "page": 15, "chunk_id": "c1"}),
            Document(page_content="LM35 layout and thermal monitoring system wiring guidelines.", metadata={"source": "lm35.pdf", "page": 15, "chunk_id": "c2"}),
            Document(page_content="LM35 layout and thermal monitoring sensor decoupling.", metadata={"source": "lm35.pdf", "page": 15, "chunk_id": "c3"}),
            Document(page_content="LM35 layout and thermal monitoring system notes.", metadata={"source": "lm35.pdf", "page": 15, "chunk_id": "c4"}),
            Document(page_content="Features: Operates from 4 V to 30 V. 10 mV/C scale factor.", metadata={"source": "lm35.pdf", "page": 1, "chunk_id": "c5"}),
            Document(page_content="Absolute Maximum Ratings: Supply voltage 35 V.", metadata={"source": "lm35.pdf", "page": 4, "chunk_id": "c6"}),
        ]

        retriever.vector_store.similarity_search_with_score.return_value = [(d, 0.2) for d in mock_docs]
        retriever.bm25_retriever.invoke.return_value = mock_docs

        results = retriever.retrieve_hybrid("LM35 temperature sensor 5V", k=5, max_per_page=2)

        # Max per page constraint must cap page 15 at 2 documents
        p15_count = sum(1 for d in results if d.metadata.get("page") == 15)
        assert p15_count <= 2

        # Pages 1 and 4 must now be included rather than starved
        pages = [d.metadata.get("page") for d in results]
        assert 1 in pages or 4 in pages

    def test_report_status_agrees_with_validation_status(self) -> None:
        """Requirement 8: Report executive summary strictly reflects VALID, INVALID, or INCONCLUSIVE."""
        agent = ReportAgent()

        # Case A: VALID
        state_valid: CircuitMindState = {
            "user_query": "LM35 5V",
            "requirements": {"target_components": ["LM35"], "identified_requirements": {}, "missing_requirements": []},
            "retrieved_documents": [{"source": "lm35.pdf", "page": 1, "content": "4V to 30V"}],
            "component_information": {"part_number": "LM35", "verified_specs": {}, "unverified_or_missing_specs": [], "citations": []},
            "validation_results": {
                "status": "VALID",
                "is_valid": True,
                "checks": [{"check_name": "Operating Voltage", "status": "PASS"}],
                "violations": [],
                "warnings": [],
                "calculations": {},
            },
            "engineering_report": {},
            "errors": [],
        }
        report_valid = agent.generate_report(state_valid)
        assert "PASSED DETERMINISTIC VALIDATION" in report_valid.executive_summary

        # Case B: INVALID
        state_invalid: CircuitMindState = {
            "user_query": "LM35 40V",
            "requirements": {"target_components": ["LM35"], "identified_requirements": {}, "missing_requirements": []},
            "retrieved_documents": [{"source": "lm35.pdf", "page": 1, "content": "4V to 30V"}],
            "component_information": {"part_number": "LM35", "verified_specs": {}, "unverified_or_missing_specs": [], "citations": []},
            "validation_results": {
                "status": "INVALID",
                "is_valid": False,
                "checks": [{"check_name": "Operating Voltage", "status": "FAIL"}],
                "violations": ["EXCEEDS 35V ABSOLUTE MAX"],
                "warnings": [],
                "calculations": {},
            },
            "engineering_report": {},
            "errors": [],
        }
        report_invalid = agent.generate_report(state_invalid)
        assert "VALIDATION FAILED" in report_invalid.executive_summary

        # Case C: INCONCLUSIVE
        state_inconclusive: CircuitMindState = {
            "user_query": "LM35 unknown",
            "requirements": {"target_components": ["LM35"], "identified_requirements": {}, "missing_requirements": []},
            "retrieved_documents": [],
            "component_information": {"part_number": "LM35", "verified_specs": {}, "unverified_or_missing_specs": [], "citations": []},
            "validation_results": {
                "status": "INCONCLUSIVE",
                "is_valid": False,
                "checks": [],
                "violations": [],
                "warnings": ["Insufficient datasheet parameters"],
                "calculations": {},
            },
            "engineering_report": {},
            "errors": [],
        }
        report_inconclusive = agent.generate_report(state_inconclusive)
        assert "INCONCLUSIVE" in report_inconclusive.executive_summary
        assert "PASSED DETERMINISTIC VALIDATION" not in report_inconclusive.executive_summary

    def test_typical_quiescent_current_not_labeled_guaranteed_maximum(self) -> None:
        """Requirement: 60µA quiescent current must be labeled 'typical' and not guaranteed maximum."""
        docs = [
            {
                "content": "Features: Operates from 4 V to 30 V. Less Than 60-μA Current Drain. Low Self-Heating.",
                "source": "lm35.pdf",
                "page": 1,
            }
        ]
        agent = ComponentAgent()
        info = agent.extract_from_documents(docs, target_component="LM35")

        assert "quiescent_current" in info.verified_specs
        iq_spec = info.verified_specs["quiescent_current"]
        assert iq_spec["status"] == "typical"
        assert iq_spec["status"] != "guaranteed_maximum"
        assert iq_spec["status"] != "guaranteed"
        assert iq_spec["value_ma"] == 0.06
        assert "25°C" in iq_spec["conditions"]
        # Guaranteed maximum over full temperature range must be listed in unverified/missing
        assert any("Guaranteed maximum quiescent current" in m for m in info.unverified_or_missing_specs)

    def test_nonlinearity_not_confused_with_accuracy(self) -> None:
        """Requirement: ±0.25°C typical nonlinearity must NOT be classified as accuracy tolerance."""
        docs = [
            {
                "content": "• Non-Linearity Only ±¼°C Typical • Low-Cost Due to Wafer-Level Trimming",
                "source": "lm35.pdf",
                "page": 1,
            }
        ]
        agent = ComponentAgent()
        info = agent.extract_from_documents(docs, target_component="LM35")

        # Nonlinearity must be extracted as its own parameter
        assert "nonlinearity" in info.verified_specs
        nl = info.verified_specs["nonlinearity"]
        assert nl["value_c"] == 0.25
        assert nl["status"] == "typical"
        assert "Non-Linearity" in nl["parameter"]

        # Must NOT be classified as accuracy tolerance
        assert "accuracy_tolerance" not in info.verified_specs
        assert any("accuracy" in m.lower() for m in info.unverified_or_missing_specs)

    def test_supply_compatibility_passes_at_5v_against_4v_to_30v(self) -> None:
        """Requirement: 5V supply check validates against verified recommended 4V to 30V range."""
        agent = ValidationAgent()
        reqs = {
            "target_components": ["LM35"],
            "target_supply_voltage_v": 5.0,
            "adc_reference_voltage_v": 5.0,
            "load_impedance_ohms": 10000.0,
        }
        comp_info = {
            "part_number": "LM35",
            "verified_specs": {
                "supply_voltage_recommended": {
                    "min_v": 4.0,
                    "max_v": 30.0,
                    "status": "recommended_operating",
                    "citation": "[lm35.pdf, p. 1]",
                },
                "supply_voltage_abs_max": {"max_v": 35.0},
            },
        }

        result = agent.validate(requirements=reqs, component_info=comp_info)

        # Supply check must be explicitly PASS with 25V headroom
        supply_check = next(c for c in result.checks if c["check_name"] == "Operating Supply Voltage Compatibility")
        assert supply_check["status"] == "PASS"
        assert supply_check["details"]["headroom_v"] == 25.0
        assert supply_check["details"]["min_recommended_v"] == 4.0
        assert supply_check["details"]["max_recommended_v"] == 30.0

    def test_missing_adc_load_constraints_produces_inconclusive_overall_status(self) -> None:
        """Requirement: Supply PASS can coexist with overall INCONCLUSIVE when ADC/load constraints are missing."""
        agent = ValidationAgent()
        # Design specifies 5V supply and LM35, but NO ADC reference or load constraints
        reqs = {
            "target_components": ["LM35"],
            "target_supply_voltage_v": 5.0,
        }
        comp_info = {
            "part_number": "LM35",
            "verified_specs": {
                "supply_voltage_recommended": {"min_v": 4.0, "max_v": 30.0},
                "supply_voltage_abs_max": {"max_v": 35.0},
                "quiescent_current": {"value_ma": 0.06, "status": "typical"},
                "scale_factor": {"value": 10.0},
            },
        }

        result = agent.validate(requirements=reqs, component_info=comp_info)

        # 1. Supply check explicitly PASSED
        supply_check = next(c for c in result.checks if c["check_name"] == "Operating Supply Voltage Compatibility")
        assert supply_check["status"] == "PASS"

        # 2. ADC & load interface checks are UNVERIFIED
        adc_check = next(c for c in result.checks if c["check_name"] == "ADC Interface & Signal Headroom")
        assert adc_check["status"] == "UNVERIFIED"

        load_check = next(c for c in result.checks if c["check_name"] == "Output Load Impedance Feasibility")
        assert load_check["status"] == "UNVERIFIED"

        # 3. Overall validation status MUST be INCONCLUSIVE, not VALID
        assert result.status == "INCONCLUSIVE"
        assert result.is_valid is False
        assert any("complete circuit validation cannot be confirmed" in w or "Supply voltage compatibility check passed" in w for w in result.warnings)

    def test_specifications_preserve_metadata_conditions_and_variant(self) -> None:
        """Requirement: Verified specifications preserve unit, status, conditions, variant, source, and citation."""
        docs = [
            {
                "content": "• Operates From 4 V to 30 V • Less Than 60-μA Current Drain • 10-mV/°C Scale Factor",
                "source": "lm35.pdf",
                "page": 1,
            }
        ]
        agent = ComponentAgent()
        info = agent.extract_from_documents(docs, target_component="LM35")

        for key in ("supply_voltage_recommended", "quiescent_current", "scale_factor"):
            assert key in info.verified_specs, f"Expected {key} in verified_specs"
            spec = info.verified_specs[key]
            assert "parameter" in spec
            assert "value" in spec
            assert "unit" in spec
            assert "status" in spec
            assert "conditions" in spec
            assert "variant" in spec
            assert "source" in spec
            assert "page" in spec
            assert "citation" in spec
            assert spec["source"] == "lm35.pdf"
            assert spec["page"] == 1

    def test_adc_reference_0_8v_saturation_at_100c_fails_and_invalidates_design(self) -> None:
        """Verify ADC reference of 0.8V causes ADC Interface check to FAIL when sensor outputs 1.0V at 100°C.

        Checks:
        1. ADC reference 0.8V remains 0.8V.
        2. Sensor supply 5V remains 5V.
        3. At 100°C, LM35 output is 1.0V.
        4. ADC compatibility returns FAIL when permitted input range is 0V to 0.8V.
        5. Overall validation becomes INVALID.
        """
        verified_specs = {
            "scale_factor": {
                "parameter": "Scale Factor",
                "value": 10.0,
                "unit": "mV/°C",
                "status": "nominal",
                "source": "lm35.pdf",
                "page": 1,
            },
            "supply_voltage_recommended": {
                "parameter": "Supply Voltage (Recommended)",
                "value": "4 to 30",
                "min_v": 4.0,
                "max_v": 30.0,
                "unit": "V",
                "status": "recommended_operating",
                "source": "lm35.pdf",
                "page": 1,
            },
            "quiescent_current": {
                "parameter": "Quiescent Current",
                "value": 60.0,
                "value_ma": 0.06,
                "unit": "µA",
                "status": "typical",
                "source": "lm35.pdf",
                "page": 1,
            },
        }

        # Query parsing check: ensure 0.8V ADC ref and 5V supply extraction
        req_agent = RequirementAgent()
        query = (
            "Design an LM35 temperature monitoring system.\n"
            "- Sensor supply: 5V\n"
            "- ADC reference: 0.8V\n"
            "- ADC input range: 0V to 0.8V\n"
            "- Temperature: 0°C to 100°C\n"
            "- Load impedance: 10k ohm"
        )
        extracted = req_agent.extract(query)
        assert extracted.target_supply_voltage_v == 5.0
        assert extracted.adc_reference_voltage_v == 0.8
        assert extracted.min_temp_c == 0.0
        assert extracted.max_temp_c == 100.0

        # Electrical validation execution
        val_agent = ValidationAgent()
        val_res = val_agent.validate(
            requirements=extracted.model_dump(),
            component_info={"part_number": "LM35", "verified_specs": verified_specs},
        )

        # 1. Output voltage at 100°C must be 1.0V (1000 mV)
        assert "output_at_100c" in val_res.calculations
        assert val_res.calculations["output_at_100c"]["output_voltage_v"] == 1.0
        assert val_res.calculations["output_at_100c"]["output_voltage_mv"] == 1000.0

        # 2. Output voltage at 25°C must be 0.25V (250 mV)
        assert "output_at_25c" in val_res.calculations
        assert val_res.calculations["output_at_25c"]["output_voltage_v"] == 0.25
        assert val_res.calculations["output_at_25c"]["output_voltage_mv"] == 250.0

        # 3. ADC check must FAIL because 1.0V exceeds 0.8V
        adc_checks = [c for c in val_res.checks if c["check_name"] == "ADC Interface & Signal Headroom"]
        assert len(adc_checks) == 1
        assert adc_checks[0]["status"] == "FAIL"
        assert adc_checks[0]["details"]["max_sensor_output_v"] == 1.0
        assert adc_checks[0]["details"]["adc_reference_v"] == 0.8

        # 4. Violation must be explicitly recorded
        assert any("ADC SATURATION VIOLATION" in v for v in val_res.violations)

        # 5. Overall validation must be strictly INVALID
        assert val_res.status == "INVALID"
        assert val_res.is_valid is False

    def test_specification_categories_and_evidence_preservation(self) -> None:
        """Verify ComponentAgent populates standardized categories and evidence text."""
        docs = [
            {
                "source": "lm35.pdf",
                "page": 4,
                "content": "Recommended Operating Conditions: Operating supply voltage range: 4 V to 30 V. Specified operating temperature: LM35D 0°C to 100°C.",
            },
            {
                "source": "lm35.pdf",
                "page": 4,
                "content": "Absolute Maximum Ratings: Supply voltage: -0.2 V to 35 V. Output voltage: -1 V to 6 V.",
            },
            {
                "source": "lm35.pdf",
                "page": 7,
                "content": "Quiescent current VS = 5V, TA = 25°C is 56 µA typical, tested limit 67 µA.",
            },
            {
                "source": "lm35.pdf",
                "page": 1,
                "content": "Scale factor: 10 mV/°C linear sensitivity. Non-linearity only ±¼°C typical.",
            },
        ]
        agent = ComponentAgent()
        comp_info = agent.extract_from_documents(docs, target_component="LM35D")

        specs = comp_info.verified_specs
        assert "supply_voltage_recommended" in specs
        assert specs["supply_voltage_recommended"]["category"] == "Recommended Operating Conditions"
        assert specs["supply_voltage_recommended"]["evidence_text"] is not None
        assert specs["supply_voltage_recommended"]["page"] == 4

        assert "supply_voltage_abs_max" in specs
        assert specs["supply_voltage_abs_max"]["category"] == "Absolute Maximum Ratings"
        assert specs["supply_voltage_abs_max"]["evidence_text"] is not None

        assert "quiescent_current" in specs
        assert specs["quiescent_current"]["category"] == "Typical Characteristics"
        assert specs["quiescent_current"]["status"] == "typical"
        assert specs["quiescent_current"]["evidence_text"] is not None

        assert "nonlinearity" in specs
        assert specs["nonlinearity"]["category"] == "Typical Characteristics"

        assert comp_info.specific_variant == "LM35D"

    def test_lm35_variant_validation_temperature_violation(self) -> None:
        """Verify operating temperature violation on LM35D produces FAIL check and INVALID status."""
        reqs = {
            "target_components": ["LM35D"],
            "component_variant": "LM35D",
            "target_supply_voltage_v": 5.0,
            "min_temp_c": 0.0,
            "max_temp_c": 125.0,  # LM35D is only rated to 100°C!
            "adc_reference_voltage_v": 3.3,
            "load_impedance_ohms": 10000.0,
        }
        verified_specs = {
            "supply_voltage_recommended": {"min_v": 4.0, "max_v": 30.0, "page": 4},
            "quiescent_current": {"value_ma": 0.056, "page": 7},
            "scale_factor": {"value": 10.0, "page": 1},
        }
        val_agent = ValidationAgent()
        res = val_agent.validate(
            requirements=reqs,
            component_info={"part_number": "LM35D", "specific_variant": "LM35D", "verified_specs": verified_specs},
        )

        assert res.status == "INVALID"
        assert res.is_valid is False
        temp_checks = [c for c in res.checks if "Operating Temperature Range" in c["check_name"]]
        assert len(temp_checks) == 1
        assert temp_checks[0]["status"] == "FAIL"
        assert any("TEMPERATURE OVERFLOW" in v for v in res.violations)

    def test_adc_compatibility_3v3_12bit_passes_with_quantization_and_caveats(self) -> None:
        """Verify compliant ADC configuration produces PASS, 12-bit quantization, and engineering caveats."""
        reqs = {
            "target_components": ["LM35"],
            "target_supply_voltage_v": 5.0,
            "min_temp_c": 0.0,
            "max_temp_c": 100.0,
            "adc_reference_voltage_v": 3.3,
            "adc_resolution_bits": 12,
            "load_impedance_ohms": 10000.0,
        }
        verified_specs = {
            "supply_voltage_recommended": {"min_v": 4.0, "max_v": 30.0, "page": 4},
            "quiescent_current": {"value_ma": 0.056, "page": 7},
            "scale_factor": {"value": 10.0, "page": 1},
        }
        val_agent = ValidationAgent()
        res = val_agent.validate(
            requirements=reqs,
            component_info={"part_number": "LM35", "verified_specs": verified_specs},
        )

        assert res.status == "VALID"
        assert res.is_valid is True
        assert len(res.violations) == 0

        adc_checks = [c for c in res.checks if c["check_name"] == "ADC Interface & Signal Headroom"]
        assert len(adc_checks) == 1
        assert adc_checks[0]["status"] == "PASS"

        details = adc_checks[0]["details"]
        assert details["margin_v"] == 2.3
        assert "quantization" in details
        q = details["quantization"]
        assert q["total_codes"] == 4096
        assert abs(q["temp_resolution_c_per_count"] - 0.0806) < 0.001
        assert len(details["interface_caveats"]) >= 4

    def test_missing_adc_reference_yields_unverified_and_inconclusive(self) -> None:
        """Verify that omitting ADC reference voltage yields UNVERIFIED check and INCONCLUSIVE overall status."""
        reqs = {
            "target_components": ["LM35"],
            "target_supply_voltage_v": 5.0,
            "min_temp_c": 0.0,
            "max_temp_c": 100.0,
            "load_impedance_ohms": 10000.0,
            # Note: adc_reference_voltage_v is intentionally omitted
        }
        verified_specs = {
            "supply_voltage_recommended": {"min_v": 4.0, "max_v": 30.0, "page": 4},
            "quiescent_current": {"value_ma": 0.056, "page": 7},
            "scale_factor": {"value": 10.0, "page": 1},
        }
        val_agent = ValidationAgent()
        res = val_agent.validate(
            requirements=reqs,
            component_info={"part_number": "LM35", "verified_specs": verified_specs},
        )

        assert res.status == "INCONCLUSIVE"
        assert res.is_valid is False
        adc_checks = [c for c in res.checks if c["check_name"] == "ADC Interface & Signal Headroom"]
        assert len(adc_checks) == 1
        assert adc_checks[0]["status"] == "UNVERIFIED"

    def test_lm35d_0_to_125c_with_unicode_dash_produces_fail_and_invalid(self) -> None:
        """Verify LM35D with 0–125°C (using en-dash) fails temperature range and marks design INVALID.

        Ensures:
        1. Temperature range 0–125°C with Unicode en-dash is extracted as min=0.0, max=125.0.
        2. LM35D is identified as component_variant.
        3. Operating Temperature Range check status is FAIL.
        4. Evidence card contains page, user constraint, specified limit, and comparison.
        5. Overall status is INVALID despite unverified load impedance.
        """
        query = "LM35D, 5V, 0–125°C, ADC 3.3V"
        req_agent = RequirementAgent()
        extracted = req_agent.extract(query)

        assert extracted.component_variant == "LM35D"
        assert extracted.min_temp_c == 0.0
        assert extracted.max_temp_c == 125.0
        assert extracted.adc_reference_voltage_v == 3.3

        verified_specs = {
            "supply_voltage_recommended": {"min_v": 4.0, "max_v": 30.0, "page": 4},
            "scale_factor": {"value": 10.0, "page": 1},
            "quiescent_current": {"value_ma": 0.056, "page": 7},
        }
        val_agent = ValidationAgent()
        res = val_agent.validate(
            requirements=extracted.model_dump(),
            component_info={"part_number": "LM35D", "specific_variant": "LM35D", "verified_specs": verified_specs},
        )

        assert res.status == "INVALID"
        assert res.is_valid is False
        assert any("TEMPERATURE OVERFLOW" in v for v in res.violations)
        assert any("exceeds rated maximum limit (100.0°C) for variant LM35D" in v for v in res.violations)

        temp_checks = [c for c in res.checks if "Operating Temperature Range" in c["check_name"]]
        assert len(temp_checks) == 1
        t_chk = temp_checks[0]
        assert t_chk["status"] == "FAIL"

        details = t_chk["details"]
        assert details["datasheet_page"] == 4
        assert "Rated Operating Range for LM35D: 0.0°C to 100.0°C" in details["extracted_specification"]
        assert "0.0°C to 125.0°C" in details["user_constraint"]
        assert "Operating Range Margin: Low 0.0°C, High -25.0°C" in details["calculation"]
        assert details["comparison_result"] == "FAIL"

    def test_lm35d_0_to_100c_has_no_temperature_violation(self) -> None:
        """Verify LM35D operated at 0–100°C has no temperature range violation."""
        query = "LM35D, 5V, 0–100°C, ADC 3.3V, load 10k"
        req_agent = RequirementAgent()
        extracted = req_agent.extract(query)

        assert extracted.component_variant == "LM35D"
        assert extracted.min_temp_c == 0.0
        assert extracted.max_temp_c == 100.0

        verified_specs = {
            "supply_voltage_recommended": {"min_v": 4.0, "max_v": 30.0, "page": 4},
            "scale_factor": {"value": 10.0, "page": 1},
            "quiescent_current": {"value_ma": 0.056, "page": 7},
        }
        val_agent = ValidationAgent()
        res = val_agent.validate(
            requirements=extracted.model_dump(),
            component_info={"part_number": "LM35D", "specific_variant": "LM35D", "verified_specs": verified_specs},
        )

        temp_checks = [c for c in res.checks if "Operating Temperature Range" in c["check_name"]]
        assert len(temp_checks) == 1
        assert temp_checks[0]["status"] == "PASS"
        assert not any("TEMPERATURE" in v for v in res.violations)

    def test_lm35c_within_range_has_no_temperature_violation(self) -> None:
        """Verify LM35C operated within its -40°C to 110°C rated range passes without violation."""
        query = "LM35C, 5V, 0–100°C, ADC 3.3V, load 10k"
        req_agent = RequirementAgent()
        extracted = req_agent.extract(query)

        assert extracted.component_variant == "LM35C"
        assert extracted.min_temp_c == 0.0
        assert extracted.max_temp_c == 100.0

        verified_specs = {
            "supply_voltage_recommended": {"min_v": 4.0, "max_v": 30.0, "page": 4},
            "scale_factor": {"value": 10.0, "page": 1},
            "quiescent_current": {"value_ma": 0.056, "page": 7},
        }
        val_agent = ValidationAgent()
        res = val_agent.validate(
            requirements=extracted.model_dump(),
            component_info={"part_number": "LM35C", "specific_variant": "LM35C", "verified_specs": verified_specs},
        )

        temp_checks = [c for c in res.checks if "Operating Temperature Range" in c["check_name"]]
        assert len(temp_checks) == 1
        assert temp_checks[0]["status"] == "PASS"
        assert not any("TEMPERATURE" in v for v in res.violations)
        assert res.status == "VALID"

    def test_lm35_unspecified_variant_handles_uncertainty_explicitly(self) -> None:
        """Verify generic LM35 at 0–125°C does not fail (rated -55 to 150°C) but warns of LM35D commercial limit."""
        query = "LM35, 5V, 0–125°C, ADC 3.3V"
        req_agent = RequirementAgent()
        extracted = req_agent.extract(query)

        assert extracted.component_variant == "LM35"
        assert extracted.min_temp_c == 0.0
        assert extracted.max_temp_c == 125.0

        verified_specs = {
            "supply_voltage_recommended": {"min_v": 4.0, "max_v": 30.0, "page": 4},
            "scale_factor": {"value": 10.0, "page": 1},
            "quiescent_current": {"value_ma": 0.056, "page": 7},
        }
        val_agent = ValidationAgent()
        res = val_agent.validate(
            requirements=extracted.model_dump(),
            component_info={"part_number": "LM35", "verified_specs": verified_specs},
        )

        temp_checks = [c for c in res.checks if "Operating Temperature Range" in c["check_name"]]
        assert len(temp_checks) == 1
        # LM35 rated max is 150°C, so 125°C is PASS for generic LM35
        assert temp_checks[0]["status"] == "PASS"
        assert not any("TEMPERATURE OVERFLOW" in v for v in res.violations)
        # But generic uncertainty warning must be present
        assert any("GENERIC PART NUMBER UNCERTAINTY" in w for w in res.warnings)

    def test_definite_fail_never_overridden_by_unverified_check(self) -> None:
        """Verify a definite FAIL check guarantees INVALID status even if all other checks are UNVERIFIED."""
        # Query with only supply voltage (fails voltage check) and no other parameters
        reqs = {
            "target_components": ["LM35"],
            "target_supply_voltage_v": 38.0,  # Fails absolute maximum of 35V
        }
        verified_specs = {
            "supply_voltage_recommended": {"min_v": 4.0, "max_v": 30.0, "page": 4},
            "supply_voltage_abs_max": {"max_v": 35.0, "page": 4},
        }
        val_agent = ValidationAgent()
        res = val_agent.validate(
            requirements=reqs,
            component_info={"part_number": "LM35", "verified_specs": verified_specs},
        )

        # Voltage check is FAIL; ADC & Load checks are UNVERIFIED
        has_fail = any(c["status"] == "FAIL" for c in res.checks)
        has_unverified = any(c["status"] == "UNVERIFIED" for c in res.checks)
        assert has_fail is True
        assert has_unverified is True
        # Must be strictly INVALID, never INCONCLUSIVE
        assert res.status == "INVALID"
        assert res.is_valid is False



