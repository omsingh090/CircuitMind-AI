"""FastAPI Backend Integration and Unit Tests for CircuitMind AI.

Tests:
1. Root endpoint (GET /) service info.
2. Health check endpoint (GET /health).
3. Retrieval endpoint (POST /api/v1/retrieve) with mocked hybrid retriever.
4. Retrieval input validation (empty query, invalid strategy).
5. Copilot analysis endpoint (POST /api/v1/analyze) with mocked StateGraph workflow.
6. Copilot analysis input validation.
7. CORS middleware configuration and response headers.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.api.main import create_app
from app.graph.state import CircuitMindState


@pytest.fixture
def client() -> TestClient:
    """Fixture providing a test client for the FastAPI application."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def mock_lm35_documents():
    """Sample mocked datasheet chunks."""
    return [
        Document(
            page_content="The LM35 operates from 4 V to 30 V with low 60 uA drain current.",
            metadata={"source": "lm35.pdf", "page": 1, "chunk_id": "chunk_001", "rrf_score": 0.033},
        ),
        Document(
            page_content="Absolute Maximum Ratings: Supply voltage -0.2 V to 35 V.",
            metadata={"source": "lm35.pdf", "page": 4, "chunk_id": "chunk_002", "rrf_score": 0.031},
        ),
    ]


@pytest.fixture
def mock_workflow_state():
    """Sample full CircuitMindState output from LangGraph copilot."""
    return {
        "user_query": "Can I operate an LM35 sensor at 5V supply with 10mA current budget?",
        "requirements": {
            "target_components": ["LM35"],
            "target_supply_voltage_v": 5.0,
            "target_max_current_ma": 10.0,
            "identified_requirements": {"target_supply_voltage_v": 5.0, "target_max_current_ma": 10.0},
            "missing_requirements": ["Operating temperature range not specified"],
        },
        "retrieved_documents": [
            {"source": "lm35.pdf", "page": 1, "content": "Operates from 4V to 30V.", "chunk_id": "c1", "rrf_score": 0.033},
            {"source": "lm35.pdf", "page": 4, "content": "Supply voltage -0.2V to 35V.", "chunk_id": "c2", "rrf_score": 0.031},
        ],
        "component_information": {
            "part_number": "LM35",
            "verified_specs": {
                "supply_voltage_recommended": {"raw": "4 V to 30 V", "citation": "[lm35.pdf, p. 1]"},
                "scale_factor": {"raw": "10.0 mV/°C", "citation": "[lm35.pdf, p. 1]"},
            },
            "unverified_or_missing_specs": ["Thermal resistance"],
            "citations": [{"source": "lm35.pdf", "page": 1}, {"source": "lm35.pdf", "page": 4}],
        },
        "validation_results": {
            "is_valid": True,
            "checks": [{"check_name": "Supply Voltage Compatibility", "status": "PASS", "details": {"headroom_v": 25.0}}],
            "violations": [],
            "warnings": [],
            "calculations": {"quiescent_power": {"power_mw": 0.3, "formula": "P = |V * I|"}},
        },
        "engineering_report": {
            "title": "CircuitMind AI Engineering Report: LM35 Design Analysis",
            "executive_summary": "Passed all preliminary checks.",
            "requirements_summary": "5V supply, 10mA current budget.",
            "component_specifications": "4V to 30V verified.",
            "validation_summary": "All checks passed.",
            "warnings_and_limitations": [],
            "citations": [{"source": "lm35.pdf", "page": 1}],
            "fabrication_disclaimer": "PRELIMINARY COPILOT DESIGN: Preliminary recommendation. Not fabrication-ready.",
        },
        "errors": [],
    }


# ==============================================================================
# 1. Root & Health Endpoint Tests
# ==============================================================================
def test_root_endpoint(client: TestClient) -> None:
    """Verify GET / returns API service overview and documentation links."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["version"] == "1.0.0"
    assert "documentation" in data


def test_health_endpoint(client: TestClient) -> None:
    """Verify GET /health returns operational status and vector database connection flag."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "degraded")
    assert data["version"] == "1.0.0"
    assert "timestamp" in data
    assert "vector_db_connected" in data


# ==============================================================================
# 2. Retrieval Endpoint Tests (/api/v1/retrieve)
# ==============================================================================
def test_retrieve_endpoint_success(client: TestClient, mock_lm35_documents) -> None:
    """Verify POST /api/v1/retrieve returns matching datasheet passages with citations."""
    with patch("app.api.routes.DatasheetRetriever") as mock_retriever_cls:
        mock_retriever = MagicMock()
        mock_retriever.retrieve_hybrid.return_value = mock_lm35_documents
        mock_retriever_cls.return_value = mock_retriever

        payload = {
            "query": "What is the recommended supply voltage range of LM35?",
            "k": 2,
            "strategy": "hybrid",
            "target_components": ["LM35"],
        }
        response = client.post("/api/v1/retrieve", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What is the recommended supply voltage range of LM35?"
        assert data["strategy"] == "hybrid"
        assert data["total_results"] == 2
        assert len(data["documents"]) == 2

        doc0 = data["documents"][0]
        assert doc0["source"] == "lm35.pdf"
        assert doc0["page"] == 1
        assert "4 V to 30 V" in doc0["content"]
        assert doc0["rrf_score"] == 0.033


def test_retrieve_endpoint_empty_query_validation(client: TestClient) -> None:
    """Verify 422 Unprocessable Entity when query is empty or whitespace."""
    response = client.post("/api/v1/retrieve", json={"query": "   "})
    assert response.status_code == 422


def test_retrieve_endpoint_unsupported_strategy(client: TestClient) -> None:
    """Verify 400 Bad Request when unsupported strategy is provided."""
    with patch("app.api.routes.DatasheetRetriever") as mock_retriever_cls:
        mock_retriever_cls.return_value = MagicMock()
        payload = {"query": "LM35 voltage", "strategy": "unsupported_strategy"}
        response = client.post("/api/v1/retrieve", json=payload)
        assert response.status_code == 400
        assert "Unsupported retrieval strategy" in response.json()["detail"]


# ==============================================================================
# 3. Analyze Endpoint Tests (/api/v1/analyze)
# ==============================================================================
def test_analyze_endpoint_success(client: TestClient, mock_workflow_state) -> None:
    """Verify POST /api/v1/analyze invokes LangGraph workflow and returns full structured cards."""
    with patch("app.api.routes.run_copilot_workflow") as mock_run_workflow:
        mock_run_workflow.return_value = mock_workflow_state

        payload = {
            "query": "Can I operate an LM35 sensor at 5V supply with 10mA current budget?",
            "target_components": ["LM35"],
        }
        response = client.post("/api/v1/analyze", json=payload)

        assert response.status_code == 200
        data = response.json()

        # Telemetry
        assert "execution_info" in data
        assert data["execution_info"]["status"] == "completed"
        assert data["execution_info"]["execution_time_ms"] >= 0.0
        assert "extract_requirements" in data["execution_info"]["nodes_executed"]

        # Requirements Card
        assert data["requirements"]["target_supply_voltage_v"] == 5.0
        assert "Operating temperature range not specified" in data["missing_requirements"]

        # Evidence Cards
        assert len(data["retrieved_documents"]) == 2
        assert data["retrieved_documents"][0]["source"] == "lm35.pdf"

        # Component Card
        assert data["component_information"]["part_number"] == "LM35"
        assert "supply_voltage_recommended" in data["component_information"]["verified_specs"]

        # Validation Card
        assert data["validation_results"]["is_valid"] is True
        assert len(data["validation_results"]["violations"]) == 0
        assert "quiescent_power" in data["validation_results"]["calculations"]

        # Engineering Report Card
        assert "LM35" in data["engineering_report"]["title"]
        assert "PRELIMINARY COPILOT DESIGN" in data["engineering_report"]["fabrication_disclaimer"]


def test_analyze_endpoint_empty_query_validation(client: TestClient) -> None:
    """Verify 422 Unprocessable Entity when query string is empty."""
    response = client.post("/api/v1/analyze", json={"query": ""})
    assert response.status_code == 422


def test_analyze_workflow_error_sanitization(client: TestClient) -> None:
    """Verify that when copilot workflow raises an exception with secret keys, secrets are redacted in API response."""
    fake_key = "AQ." + ("x" * 40)
    with patch("app.api.routes.run_copilot_workflow", side_effect=RuntimeError(f"Backend failed for key={fake_key}")):
        response = client.post("/api/v1/analyze", json={"query": "Design an LM35 sensor circuit"})

        assert response.status_code == 500
        error_detail = response.json().get("detail", "")
        assert fake_key not in error_detail
        assert "REDACTED" in error_detail
        assert "Copilot workflow execution failed" in error_detail


# ==============================================================================
# 4. CORS Middleware Tests
# ==============================================================================
def test_cors_middleware_headers(client: TestClient) -> None:
    """Verify CORS middleware sets Access-Control-Allow-Origin header."""
    headers = {"Origin": "http://localhost:8501", "Access-Control-Request-Method": "POST"}
    response = client.options("/api/v1/analyze", headers=headers)
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers

