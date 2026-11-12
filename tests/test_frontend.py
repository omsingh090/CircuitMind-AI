"""Tests for CircuitMind AI Streamlit Frontend and API Client.

Validates the HTTP API client communication layer, custom environment variable
overrides, error handling (connection failures, timeouts, status errors),
and badge styling functions.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

from frontend.api_client import (
    APIClientError,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    CircuitMindAPIClient,
)
from frontend.styles import citation_badge, status_badge
from frontend.components.agent_monitor import render_agent_monitor
from frontend.components.datasheet_explorer import render_datasheet_explorer
from frontend.components.report_viewer import render_report_viewer
from frontend.components.validation_center import render_validation_center
from frontend.components.workspace import render_workspace
import streamlit as st


class TestCircuitMindAPIClient:
    """Test suite for CircuitMindAPIClient."""

    def test_api_client_default_url(self) -> None:
        """Verify default URL fallback to 127.0.0.1:8000 when environment variable is unset."""
        with patch.dict(os.environ, {}, clear=True):
            client = CircuitMindAPIClient()
            assert client.base_url == "http://127.0.0.1:8000"

    def test_api_client_custom_url_from_env(self) -> None:
        """Verify CIRCUITMIND_API_URL environment variable overrides default URL."""
        with patch.dict(os.environ, {"CIRCUITMIND_API_URL": "http://api.production.internal:8000/"}):
            client = CircuitMindAPIClient()
            assert client.base_url == "http://api.production.internal:8000"

    def test_api_client_custom_url_argument(self) -> None:
        """Verify explicitly provided base_url takes precedence."""
        client = CircuitMindAPIClient(base_url="http://custom-host:9000/")
        assert client.base_url == "http://custom-host:9000"

    def test_api_client_check_health_success(self) -> None:
        """Verify check_health parses successful 200 response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "ok",
            "version": "1.0.0",
            "vector_db_connected": True,
        }

        with patch("httpx.Client.get", return_value=mock_response):
            client = CircuitMindAPIClient()
            health = client.check_health()
            assert health["status"] == "ok"
            assert health["version"] == "1.0.0"

    def test_api_client_check_health_connection_error(self) -> None:
        """Verify check_health raises APIConnectionError when backend is unreachable."""
        with patch("httpx.Client.get", side_effect=httpx.ConnectError("Connection refused")):
            client = CircuitMindAPIClient()
            with pytest.raises(APIConnectionError) as exc_info:
                client.check_health()
            assert "Cannot connect to CircuitMind AI backend" in str(exc_info.value)

    def test_api_client_check_health_timeout(self) -> None:
        """Verify check_health raises APITimeoutError on request timeout."""
        with patch("httpx.Client.get", side_effect=httpx.TimeoutException("Timed out")):
            client = CircuitMindAPIClient()
            with pytest.raises(APITimeoutError) as exc_info:
                client.check_health()
            assert "timed out" in str(exc_info.value).lower()

    def test_api_client_analyze_circuit_success(self) -> None:
        """Verify analyze_circuit posts payload and parses structured response."""
        mock_payload = {
            "query": "Design an LM35 5V circuit",
            "execution_info": {
                "execution_time_ms": 120.5,
                "status": "completed",
                "nodes_executed": ["extract_requirements", "retrieve_evidence"],
                "node_timings_ms": {"extract_requirements": 40.2},
            },
            "validation_results": {
                "status": "INCONCLUSIVE",
                "is_valid": False,
                "checks": [],
            },
            "engineering_report": {
                "title": "CircuitMind AI Engineering Report: LM35",
                "executive_summary": "Preliminary evaluation.",
            },
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_payload

        with patch("httpx.Client.post", return_value=mock_response) as mock_post:
            client = CircuitMindAPIClient()
            res = client.analyze_circuit("Design an LM35 5V circuit", target_components=["LM35"])

            assert res["query"] == "Design an LM35 5V circuit"
            assert res["execution_info"]["execution_time_ms"] == 120.5
            assert res["validation_results"]["status"] == "INCONCLUSIVE"
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["json"]["target_components"] == ["LM35"]

    def test_api_client_analyze_circuit_empty_query_raises_value_error(self) -> None:
        """Verify empty query raises ValueError before making any network request."""
        client = CircuitMindAPIClient()
        with pytest.raises(ValueError) as exc:
            client.analyze_circuit("")
        assert "cannot be empty" in str(exc.value)

    def test_api_client_analyze_circuit_status_error(self) -> None:
        """Verify HTTP 422 or 500 raises APIStatusError with response detail."""
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.json.return_value = {"detail": "Validation error: invalid voltage"}
        mock_response.text = '{"detail": "Validation error: invalid voltage"}'

        with patch("httpx.Client.post", return_value=mock_response):
            client = CircuitMindAPIClient()
            with pytest.raises(APIStatusError) as exc_info:
                client.analyze_circuit("Faulty prompt")
            assert exc_info.value.status_code == 422
            assert "Validation error" in str(exc_info.value)

    def test_api_client_analyze_circuit_timeout(self) -> None:
        """Verify request timeout raises APITimeoutError."""
        with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Workflow timed out")):
            client = CircuitMindAPIClient()
            with pytest.raises(APITimeoutError) as exc_info:
                client.analyze_circuit("Long query")
            assert "timed out" in str(exc_info.value).lower()

    def test_api_client_retrieve_passages_success(self) -> None:
        """Verify retrieve_passages calls /api/v1/retrieve and returns passages."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "query": "LM35 supply",
            "strategy": "hybrid",
            "total_results": 1,
            "documents": [
                {"source": "lm35.pdf", "page": 1, "content": "4V to 30V", "rrf_score": 0.05}
            ],
        }

        with patch("httpx.Client.post", return_value=mock_response):
            client = CircuitMindAPIClient()
            res = client.retrieve_passages("LM35 supply", k=3, strategy="hybrid")
            assert res["total_results"] == 1
            assert res["documents"][0]["source"] == "lm35.pdf"


class TestFrontendStyles:
    """Test suite for style formatting helpers."""

    def test_status_badge_valid(self) -> None:
        """Verify VALID status renders badge-valid class."""
        html = status_badge("VALID")
        assert "badge-valid" in html
        assert "VALID" in html

    def test_status_badge_invalid(self) -> None:
        """Verify INVALID status renders badge-invalid class."""
        html = status_badge("INVALID")
        assert "badge-invalid" in html
        assert "INVALID" in html

    def test_status_badge_inconclusive(self) -> None:
        """Verify INCONCLUSIVE status renders badge-inconclusive class."""
        html = status_badge("INCONCLUSIVE")
        assert "badge-inconclusive" in html
        assert "INCONCLUSIVE" in html

    def test_status_badge_unverified(self) -> None:
        """Verify UNVERIFIED check status renders badge-inconclusive class."""
        html = status_badge("UNVERIFIED")
        assert "badge-inconclusive" in html
        assert "UNVERIFIED" in html

    def test_citation_badge_formatting(self) -> None:
        """Verify citation badge formats source and page correctly."""
        html = citation_badge("lm35.pdf", 1)
        assert "badge-citation" in html
        assert "[lm35.pdf, p. 1]" in html


class TestSessionStateConsistency:
    """Test suite for Streamlit session-state consistency across workspace tabs."""

    @pytest.fixture
    def sample_analysis_payload(self) -> dict:
        """Standard mock AnalyzeResponse payload."""
        return {
            "query": "Design an LM35 5V monitoring circuit",
            "execution_info": {
                "execution_time_ms": 154.2,
                "status": "completed",
                "nodes_executed": [
                    "extract_requirements",
                    "retrieve_evidence",
                    "extract_specifications",
                    "validate_electrical",
                    "generate_report",
                ],
                "node_timings_ms": {
                    "extract_requirements": 25.1,
                    "retrieve_evidence": 32.4,
                    "extract_specifications": 28.3,
                    "validate_electrical": 12.0,
                    "generate_report": 56.4,
                },
            },
            "retrieved_documents": [
                {
                    "source": "lm35.pdf",
                    "page": 1,
                    "chunk_id": "chunk_001",
                    "rrf_score": 0.033,
                    "content": "Operates from 4 V to 30 V with 60 uA drain current.",
                }
            ],
            "component_information": {
                "part_number": "LM35",
                "verified_specs": {
                    "supply_voltage_recommended": {
                        "raw": "4 V to 30 V",
                        "status": "verified",
                        "source": "lm35.pdf",
                        "page": 1,
                    }
                },
                "unverified_or_missing_specs": [],
                "citations": [{"source": "lm35.pdf", "page": 1}],
            },
            "validation_results": {
                "status": "INCONCLUSIVE",
                "is_valid": False,
                "checks": [
                    {
                        "check_name": "Supply Voltage Compatibility",
                        "status": "PASS",
                        "details": {"headroom_v": 25.0},
                    }
                ],
                "violations": [],
                "warnings": ["Load impedance not specified"],
                "calculations": {
                    "output_voltage_25c": {"voltage_mv": 250.0},
                },
            },
            "engineering_report": {
                "title": "CircuitMind AI Engineering Report: LM35",
                "executive_summary": "Passed preliminary supply check.",
                "citations": [{"source": "lm35.pdf", "page": 1}],
                "fabrication_disclaimer": "PRELIMINARY COPILOT DESIGN: Not fabrication-ready.",
            },
        }

    def _mock_columns(self, spec):
        """Helper to create dynamic number of mock columns matching requested spec."""
        count = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
        return [MagicMock() for _ in range(count)]

    def test_components_fallback_to_session_state_latest_analysis(
        self, sample_analysis_payload: dict
    ) -> None:
        """Verify tabs read from st.session_state['latest_analysis'] when analysis_data is None."""
        mock_state = {
            "latest_analysis": sample_analysis_payload,
            "direct_retrieval_results": None,
        }

        with patch.object(st, "session_state", mock_state), \
             patch("streamlit.markdown") as mock_markdown, \
             patch("streamlit.columns", side_effect=self._mock_columns), \
             patch("streamlit.container", return_value=MagicMock()), \
             patch("streamlit.download_button", return_value=False), \
             patch("streamlit.expander", return_value=MagicMock()):

            # 1. Agent Monitor should read from session state
            render_agent_monitor(analysis_data=None)
            # Verify execution time from session state is formatted in markdown
            rendered_texts = [call[0][0] for call in mock_markdown.call_args_list if call[0]]
            assert any("154.2 ms" in str(txt) for txt in rendered_texts)

            # 2. Validation Center should read from session state
            mock_markdown.reset_mock()
            render_validation_center(analysis_data=None)
            val_texts = [call[0][0] for call in mock_markdown.call_args_list if call[0]]
            assert any("INCONCLUSIVE" in str(txt) for txt in val_texts)

            # 3. Report Viewer should read from session state
            mock_markdown.reset_mock()
            render_report_viewer(analysis_data=None)
            rep_texts = [call[0][0] for call in mock_markdown.call_args_list if call[0]]
            assert any("CircuitMind AI Engineering Report: LM35" in str(txt) for txt in rep_texts)

            # 4. Datasheet Explorer should read from session state
            mock_markdown.reset_mock()
            render_datasheet_explorer(analysis_data=None, api_client=MagicMock())
            exp_texts = [call[0][0] for call in mock_markdown.call_args_list if call[0]]
            assert any("chunk_001" in str(txt) for txt in exp_texts)

    def test_components_render_empty_states_when_no_data_in_session(self) -> None:
        """Verify components show appropriate informative empty states when no analysis exists."""
        mock_state = {
            "latest_analysis": None,
            "direct_retrieval_results": None,
        }

        with patch.object(st, "session_state", mock_state), \
             patch("streamlit.info") as mock_info, \
             patch("streamlit.markdown"), \
             patch("streamlit.columns", side_effect=self._mock_columns):

            # Agent monitor empty state
            render_agent_monitor(analysis_data=None)
            mock_info.assert_called_with("No active workflow telemetry. Submit a query in the Engineering Workspace to inspect execution.")

            # Validation center empty state
            mock_info.reset_mock()
            render_validation_center(analysis_data=None)
            mock_info.assert_called_with("No active validation data. Submit an engineering design prompt to evaluate electrical safety.")

            # Report viewer empty state
            mock_info.reset_mock()
            render_report_viewer(analysis_data=None)
            mock_info.assert_called_with("No report generated yet. Submit an engineering inquiry to generate an analysis report.")

    def test_workspace_preserves_latest_analysis_on_failed_request(
        self, sample_analysis_payload: dict
    ) -> None:
        """Verify previous successful analysis is preserved when a new analysis request fails."""
        mock_state = {
            "latest_analysis": sample_analysis_payload,
            "history": [sample_analysis_payload],
        }

        mock_client = MagicMock()
        mock_client.analyze_circuit.side_effect = APIStatusError(500, "Backend service unavailable")

        with patch.object(st, "session_state", mock_state), \
             patch("streamlit.text_area", return_value="Failing design prompt"), \
             patch("streamlit.button", return_value=True), \
             patch("streamlit.spinner", return_value=MagicMock()), \
             patch("streamlit.error") as mock_error, \
             patch("streamlit.columns", side_effect=self._mock_columns), \
             patch("streamlit.expander", return_value=MagicMock()), \
             patch("streamlit.markdown"):

            render_workspace(api_client=mock_client)

            # Error should be displayed to the user
            mock_error.assert_called_once()
            assert "Analysis failed" in mock_error.call_args[0][0]

            # Crucial check: latest_analysis must NOT be cleared or overwritten
            assert mock_state["latest_analysis"] == sample_analysis_payload
            # History must NOT be corrupted
            assert len(mock_state["history"]) == 1

    def test_workspace_updates_history_and_latest_analysis_on_success(self) -> None:
        """Verify successful analyses update latest_analysis and append to history."""
        mock_state = {
            "latest_analysis": None,
            "history": [],
        }

        res1 = {
            "query": "Query 1",
            "validation_results": {"status": "VALID"},
            "execution_info": {"execution_time_ms": 100.0},
            "engineering_report": {"executive_summary": "Summary 1"},
        }
        res2 = {
            "query": "Query 2",
            "validation_results": {"status": "INCONCLUSIVE"},
            "execution_info": {"execution_time_ms": 120.0},
            "engineering_report": {"executive_summary": "Summary 2"},
        }

        mock_client = MagicMock()
        mock_client.analyze_circuit.side_effect = [res1, res2]

        with patch.object(st, "session_state", mock_state), \
             patch("streamlit.text_area", side_effect=["Query 1", "Query 2"]), \
             patch("streamlit.button", return_value=True), \
             patch("streamlit.spinner", return_value=MagicMock()), \
             patch("streamlit.columns", side_effect=self._mock_columns), \
             patch("streamlit.expander", return_value=MagicMock()), \
             patch("streamlit.markdown"), \
             patch("streamlit.success"):

            # Run 1
            render_workspace(api_client=mock_client)
            assert mock_state["latest_analysis"] == res1
            assert len(mock_state["history"]) == 1

            # Run 2
            render_workspace(api_client=mock_client)
            assert mock_state["latest_analysis"] == res2
            assert len(mock_state["history"]) == 2
            assert mock_state["history"][0]["query"] == "Query 1"
            assert mock_state["history"][1]["query"] == "Query 2"

    def test_direct_retrieval_isolated_from_latest_analysis(
        self, sample_analysis_payload: dict
    ) -> None:
        """Verify direct retrieval stores in direct_retrieval_results without modifying latest_analysis."""
        mock_state = {
            "latest_analysis": sample_analysis_payload,
            "direct_retrieval_results": None,
        }

        mock_client = MagicMock()
        mock_client.retrieve_passages.return_value = {
            "query": "LM35 quiescent current",
            "strategy": "hybrid",
            "retrieval_time_ms": 42.1,
            "documents": [
                {
                    "source": "lm35.pdf",
                    "page": 2,
                    "content": "Quiescent current is typically 60 uA.",
                    "chunk_id": "chunk_002",
                }
            ],
        }

        with patch.object(st, "session_state", mock_state), \
             patch("streamlit.text_input", return_value="LM35 quiescent current"), \
             patch("streamlit.number_input", return_value=3), \
             patch("streamlit.selectbox", return_value="hybrid"), \
             patch("streamlit.button", return_value=True), \
             patch("streamlit.spinner", return_value=MagicMock()), \
             patch("streamlit.columns", side_effect=self._mock_columns), \
             patch("streamlit.markdown"):

            render_datasheet_explorer(analysis_data=None, api_client=mock_client)

            # Direct retrieval results must be populated
            assert mock_state["direct_retrieval_results"] is not None
            assert mock_state["direct_retrieval_results"]["query"] == "LM35 quiescent current"
            assert len(mock_state["direct_retrieval_results"]["documents"]) == 1

            # Crucial check: latest_analysis was NOT overwritten or mutated by direct retrieval
            assert mock_state["latest_analysis"] == sample_analysis_payload


