"""Integration tests for CircuitMind AI LangGraph Workflow Execution.

Tests:
1. End-to-end StateGraph execution with mocked retrieval passages.
2. Conditional edge routing when zero datasheet documents are found.
3. State structure integrity across all nodes.
4. Input query validation.
"""

from unittest.mock import MagicMock, patch
import pytest
from langchain_core.documents import Document

from app.graph.workflow import create_copilot_workflow, run_copilot_workflow


@pytest.fixture
def mock_lm35_chunks():
    """Sample mocked datasheet chunks for LM35."""
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


class TestWorkflowExecution:
    """Integration test suite for the compiled StateGraph workflow."""

    def test_workflow_end_to_end_success(self, mock_lm35_chunks) -> None:
        """Verify complete StateGraph execution through all nodes producing an engineering report."""
        with patch("app.agents.research_agent.DatasheetRetriever") as mock_retriever_cls:
            mock_retriever = MagicMock()
            mock_retriever.retrieve_hybrid.return_value = mock_lm35_chunks
            mock_retriever_cls.return_value = mock_retriever

            final_state = run_copilot_workflow("Design a 5V circuit with LM35 sensor drawing max 50mA with 5V ADC reference and 10k load")

            # 1. State integrity
            assert "requirements" in final_state
            assert "retrieved_documents" in final_state
            assert "component_information" in final_state
            assert "validation_results" in final_state
            assert "engineering_report" in final_state

            # 2. Requirement extraction
            reqs = final_state["requirements"]
            assert "LM35" in reqs["target_components"]
            assert reqs["target_supply_voltage_v"] == 5.0
            assert reqs["target_max_current_ma"] == 50.0
            assert reqs["adc_reference_voltage_v"] == 5.0
            assert reqs["load_impedance_ohms"] == 10000.0

            # 3. Evidence retrieval
            docs = final_state["retrieved_documents"]
            assert len(docs) == 3
            assert docs[0]["source"] == "lm35.pdf"

            # 4. Component extraction
            comp = final_state["component_information"]
            assert comp["part_number"] == "LM35"
            assert "supply_voltage_recommended" in comp["verified_specs"]

            # 5. Electrical validation
            val = final_state["validation_results"]
            assert val["is_valid"] is True
            assert val["status"] == "VALID"
            assert len(val["violations"]) == 0
            assert "quiescent_power" in val["calculations"]

            # 6. Final Report
            report = final_state["engineering_report"]
            assert "LM35" in report["title"]
            assert "PRELIMINARY COPILOT DESIGN" in report["fabrication_disclaimer"]
            assert len(report["citations"]) > 0

    def test_workflow_conditional_routing_no_evidence(self) -> None:
        """Verify conditional edge routes directly to report generation when 0 chunks are found."""
        with patch("app.agents.research_agent.DatasheetRetriever") as mock_retriever_cls:
            mock_retriever = MagicMock()
            mock_retriever.retrieve_hybrid.return_value = []  # No passages found
            mock_retriever_cls.return_value = mock_retriever

            final_state = run_copilot_workflow("Design an unknown sensor circuit with UNKNOWN99")

            # Documents should be empty
            assert final_state["retrieved_documents"] == []

            # Report generated acknowledging lack of evidence
            report = final_state["engineering_report"]
            assert "UNKNOWN99" in report["title"] or "Electronic Component" in report["title"]
            assert report["fabrication_disclaimer"] is not None

    def test_workflow_empty_query_raises_error(self) -> None:
        """Verify empty query is rejected immediately before graph invocation."""
        with pytest.raises(ValueError, match="User query cannot be empty"):
            run_copilot_workflow("")

        with pytest.raises(ValueError, match="User query cannot be empty"):
            run_copilot_workflow("   \t\n")

