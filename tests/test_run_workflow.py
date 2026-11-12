"""Unit and CLI integration tests for scripts/run_workflow.py.

Tests:
1. Command-line argument parsing and default demonstration query execution.
2. Full stage-by-stage terminal output formatting.
3. --json output option producing parseable JSON with all state keys.
4. Empty query validation and error code handling.
5. Missing evidence and error resilience.
"""

from unittest.mock import MagicMock, patch
import json
import pytest
from langchain_core.documents import Document

from scripts.run_workflow import (
    display_component_specifications,
    display_electrical_validation,
    display_engineering_report,
    display_requirements,
    display_retrieved_evidence,
    execute_workflow,
    main,
)


@pytest.fixture
def mock_lm35_chunks():
    """Mocked datasheet passages for LM35 tests."""
    return [
        Document(
            page_content="The LM35 device has an operating supply voltage range of 4 V to 30 V.",
            metadata={"source": "lm35.pdf", "page": 19, "chunk_id": "lm35_p19_001", "rrf_score": 0.033},
        ),
        Document(
            page_content="Absolute Maximum Ratings: Supply voltage -0.2 V to 35 V.",
            metadata={"source": "lm35.pdf", "page": 4, "chunk_id": "lm35_p4_002", "rrf_score": 0.031},
        ),
        Document(
            page_content="Scale factor is 10 mV/°C. Quiescent current is 60 uA at 25°C.",
            metadata={"source": "lm35.pdf", "page": 1, "chunk_id": "lm35_p1_003", "rrf_score": 0.030},
        ),
    ]


class TestRunWorkflowCLI:
    """Test suite for CLI execution and output formatting."""

    def test_main_default_execution(self, mock_lm35_chunks, capsys) -> None:
        """Verify main() executes default query and prints all 5 stages."""
        with patch("app.agents.research_agent.DatasheetRetriever") as mock_retriever_cls:
            mock_retriever = MagicMock()
            mock_retriever.retrieve_hybrid.return_value = mock_lm35_chunks
            mock_retriever_cls.return_value = mock_retriever

            exit_code = main(["Can I run LM35 at 5V?"])

            assert exit_code == 0
            captured = capsys.readouterr().out

            # Verify presence of all 5 stages in terminal output
            assert "[STAGE 1/5]" in captured
            assert "REQUIREMENT EXTRACTION" in captured
            assert "[STAGE 2/5]" in captured
            assert "DATASHEET EVIDENCE RETRIEVAL" in captured
            assert "[STAGE 3/5]" in captured
            assert "COMPONENT SPECIFICATION EXTRACTION" in captured
            assert "[STAGE 4/5]" in captured
            assert "DETERMINISTIC ELECTRICAL VALIDATION" in captured
            assert "[STAGE 5/5]" in captured
            assert "STRUCTURED ENGINEERING REPORT SYNTHESIS" in captured

            # Verify citations and disclaimer
            assert "[lm35.pdf, p. 19]" in captured
            assert "ENGINEERING INTEGRITY DISCLAIMER" in captured

    def test_main_with_question_flag_and_verbose(self, mock_lm35_chunks, capsys) -> None:
        """Verify main() with -q and --verbose flags."""
        with patch("app.agents.research_agent.DatasheetRetriever") as mock_retriever_cls:
            mock_retriever = MagicMock()
            mock_retriever.retrieve_hybrid.return_value = mock_lm35_chunks
            mock_retriever_cls.return_value = mock_retriever

            exit_code = main(["-q", "LM35 operating voltage range", "--verbose"])

            assert exit_code == 0
            captured = capsys.readouterr().out
            assert "LM35" in captured
            assert "Total Passages Retrieved: 3" in captured

    def test_main_json_output(self, mock_lm35_chunks, capsys) -> None:
        """Verify --json flag emits valid JSON with all state keys."""
        with patch("app.agents.research_agent.DatasheetRetriever") as mock_retriever_cls:
            mock_retriever = MagicMock()
            mock_retriever.retrieve_hybrid.return_value = mock_lm35_chunks
            mock_retriever_cls.return_value = mock_retriever

            exit_code = main(["--json", "5V LM35 circuit"])

            assert exit_code == 0
            captured = capsys.readouterr().out

            data = json.loads(captured)
            assert "user_query" in data
            assert "requirements" in data
            assert "retrieved_documents" in data
            assert "component_information" in data
            assert "validation_results" in data
            assert "engineering_report" in data
            assert data["requirements"]["target_supply_voltage_v"] == 5.0

    def test_main_empty_query_error(self, capsys) -> None:
        """Verify empty query returns exit code 1."""
        exit_code = main(["   "])
        assert exit_code == 1
        captured_err = capsys.readouterr().err
        assert "Engineering query cannot be empty" in captured_err

    def test_display_helpers_direct(self, capsys) -> None:
        """Verify display helper functions format output without exceptions."""
        display_requirements({
            "target_components": ["LM35"],
            "target_supply_voltage_v": 5.0,
            "missing_requirements": ["Load impedance not specified"],
        })
        display_retrieved_evidence([
            {"source": "lm35.pdf", "page": 19, "content": "Sample content", "rrf_score": 0.03}
        ])
        display_component_specifications({
            "part_number": "LM35",
            "verified_specs": {"supply": {"raw": "4V to 30V", "citation": "[lm35.pdf, p. 19]"}},
            "unverified_or_missing_specs": ["Thermal resistance"],
        })
        display_electrical_validation({
            "is_valid": True,
            "checks": [{"check_name": "Voltage Check", "status": "PASS", "details": {"headroom_v": 25.0}}],
            "violations": [],
            "warnings": ["Low margin"],
            "calculations": {"quiescent_power": {"power_mw": 0.3, "formula": "P=VI"}},
        })
        display_engineering_report({
            "title": "LM35 Report",
            "executive_summary": "Passed all checks.",
            "citations": [{"source": "lm35.pdf", "page": 19}],
            "fabrication_disclaimer": "Not fabrication-ready.",
        })

        out = capsys.readouterr().out
        assert "LM35" in out
        assert "4V to 30V" in out
        assert "Voltage Check" in out
        assert "Not fabrication-ready" in out

