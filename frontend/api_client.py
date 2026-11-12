"""HTTP API Client for CircuitMind AI Streamlit Frontend.

Provides a robust, typed interface for communicating with the CircuitMind AI
FastAPI backend over HTTP using HTTPX.

Architectural Rule:
The Streamlit frontend NEVER imports or executes LangGraph or Gemini directly.
All requests are routed exclusively through the FastAPI REST API.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("circuitmind.frontend.api")


class APIClientError(Exception):
    """Base exception for all CircuitMind AI API client errors."""


class APIConnectionError(APIClientError):
    """Raised when the backend API server is unreachable."""


class APITimeoutError(APIClientError):
    """Raised when an API request times out."""


class APIStatusError(APIClientError):
    """Raised when the API returns an unexpected HTTP error code."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"API Error {status_code}: {message}")
        self.status_code = status_code
        self.message = message


class CircuitMindAPIClient:
    """Synchronous HTTP client for CircuitMind AI FastAPI endpoints."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        default_timeout: float = 60.0,
        health_timeout: float = 4.0,
    ) -> None:
        """Initialize the CircuitMind API client.

        Args:
            base_url: Base URL of the FastAPI backend. If None, reads from
                      CIRCUITMIND_API_URL or API_BASE_URL environment variables,
                      falling back to 'http://127.0.0.1:8000'.
            default_timeout: Timeout in seconds for analysis and retrieval requests.
            health_timeout: Fast timeout in seconds for health status checks.
        """
        env_url = os.getenv("CIRCUITMIND_API_URL") or os.getenv("API_BASE_URL")
        self.base_url = (base_url or env_url or "http://127.0.0.1:8000").rstrip("/")
        self.default_timeout = default_timeout
        self.health_timeout = health_timeout
        logger.info("CircuitMind API client initialized targeting %s", self.base_url)

    def check_health(self) -> Dict[str, Any]:
        """Check the health and readiness of the FastAPI backend.

        Returns:
            Dictionary with health status, version, and database connection state.

        Raises:
            APIConnectionError: If the server cannot be reached.
            APITimeoutError: If the health check times out.
        """
        url = f"{self.base_url}/health"
        try:
            with httpx.Client(timeout=self.health_timeout) as client:
                response = client.get(url)
                if response.status_code == 200:
                    return response.json()
                raise APIStatusError(response.status_code, response.text)
        except httpx.ConnectError as exc:
            logger.warning("Failed to connect to CircuitMind backend at %s: %s", url, exc)
            raise APIConnectionError(
                f"Cannot connect to CircuitMind AI backend at {self.base_url}. "
                "Ensure FastAPI is running (uvicorn app.api.main:app --port 8000)."
            ) from exc
        except httpx.TimeoutException as exc:
            logger.warning("Health check timed out at %s: %s", url, exc)
            raise APITimeoutError(
                f"Health check timed out after {self.health_timeout}s contacting {self.base_url}."
            ) from exc
        except Exception as exc:
            if isinstance(exc, APIClientError):
                raise
            raise APIClientError(f"Unexpected error during health check: {exc}") from exc

    def analyze_circuit(
        self,
        query: str,
        target_components: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute full multi-agent LangGraph workflow via POST /api/v1/analyze.

        Args:
            query: The engineering design or circuit inquiry.
            target_components: Optional explicit component part numbers.

        Returns:
            Structured AnalyzeResponse dictionary containing cards for rendering.

        Raises:
            APIConnectionError: If backend is unreachable.
            APITimeoutError: If workflow analysis times out.
            APIStatusError: If server returns an HTTP error code (e.g. 422, 500).
        """
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query string cannot be empty.")

        url = f"{self.base_url}/api/v1/analyze"
        payload: Dict[str, Any] = {"query": clean_query}
        if target_components:
            payload["target_components"] = target_components

        logger.info("Posting analysis request to %s: %r", url, clean_query[:60])
        try:
            with httpx.Client(timeout=self.default_timeout) as client:
                response = client.post(url, json=payload)
                if response.status_code == 200:
                    return response.json()

                # Extract error detail if available
                detail = "Request failed"
                try:
                    err_json = response.json()
                    detail = err_json.get("detail", response.text)
                except Exception:
                    detail = response.text

                raise APIStatusError(response.status_code, str(detail))

        except httpx.ConnectError as exc:
            raise APIConnectionError(
                f"Could not connect to CircuitMind backend at {self.base_url}. "
                "Verify the server is running."
            ) from exc
        except httpx.TimeoutException as exc:
            raise APITimeoutError(
                f"Analysis request timed out after {self.default_timeout}s. "
                "The agentic workflow took longer than expected."
            ) from exc
        except Exception as exc:
            if isinstance(exc, APIClientError):
                raise
            raise APIClientError(f"Analysis request error: {exc}") from exc

    def retrieve_passages(
        self,
        query: str,
        k: int = 5,
        strategy: str = "hybrid",
        target_components: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Retrieve pure datasheet evidence passages via POST /api/v1/retrieve.

        Args:
            query: The technical search query.
            k: Number of passages to return (1-20).
            strategy: 'hybrid', 'semantic', 'mmr', or 'keyword'.
            target_components: Optional explicit component scoping.

        Returns:
            RetrieveResponse dictionary with retrieved passages and citations.
        """
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query string cannot be empty.")

        url = f"{self.base_url}/api/v1/retrieve"
        payload: Dict[str, Any] = {
            "query": clean_query,
            "k": k,
            "strategy": strategy,
        }
        if target_components:
            payload["target_components"] = target_components

        logger.info("Posting retrieval request to %s (strategy=%s, k=%d)", url, strategy, k)
        try:
            with httpx.Client(timeout=self.default_timeout) as client:
                response = client.post(url, json=payload)
                if response.status_code == 200:
                    return response.json()

                detail = response.text
                try:
                    detail = response.json().get("detail", detail)
                except Exception:
                    pass
                raise APIStatusError(response.status_code, str(detail))

        except httpx.ConnectError as exc:
            raise APIConnectionError(f"Backend unreachable at {self.base_url}") from exc
        except httpx.TimeoutException as exc:
            raise APITimeoutError(f"Retrieval timed out after {self.default_timeout}s") from exc
        except Exception as exc:
            if isinstance(exc, APIClientError):
                raise
            raise APIClientError(f"Retrieval request error: {exc}") from exc

