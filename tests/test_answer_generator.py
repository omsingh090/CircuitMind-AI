"""Unit tests for CircuitMind AI Grounded RAG Answer Generation Engine using Google Gemini.

Tests:
1. Successful grounded answer generation with mocked Gemini responses.
2. Missing Gemini API key handling.
3. Empty question validation.
4. Empty retrieved context fallback handling (zero API calls).
5. Context formatting and citation deduplication.
6. Insufficient context detection flag.
7. Gemini API error handling (APIError).
8. Strategy propagation to retriever.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch
import httpx
import pytest
from google.genai import errors
from langchain_core.documents import Document

from app.rag.answer_generator import (
    GroundedAnswerGenerator,
    GroundedAnswerResult,
    SourceCitation,
    format_context_for_prompt,
    generate_grounded_answer,
)


@pytest.fixture
def mock_retriever() -> MagicMock:
    """Fixture returning a mock DatasheetRetriever with sample LM35 chunks."""
    retriever = MagicMock()
    retriever.retrieve_hybrid.return_value = [
        Document(
            page_content="The LM35 device has a very wide 4-V to 30-V power supply voltage range.",
            metadata={"source": "lm35.pdf", "page": 19, "chunk_id": "lm35_p19_001", "part_number": "LM35"},
        ),
        Document(
            page_content="Absolute Maximum Ratings: Supply voltage -0.2 V to 35 V.",
            metadata={"source": "lm35.pdf", "page": 4, "chunk_id": "lm35_p4_002", "part_number": "LM35"},
        ),
    ]
    return retriever


@pytest.fixture
def mock_gemini_client() -> MagicMock:
    """Fixture returning a mock Google GenAI client returning a grounded completion."""
    client = MagicMock()
    response_mock = MagicMock()
    response_mock.text = (
        "- **Datasheet Specification**: The recommended operating supply voltage range for the LM35 is "
        "4 V to 30 V [lm35.pdf, p. 19].\n"
        "- **Absolute Maximum Rating**: The supply voltage absolute maximum is -0.2 V to 35 V [lm35.pdf, p. 4]."
    )
    client.models.generate_content.return_value = response_mock
    return client


# ==============================================================================
# 1. Successful Grounded Answer Generation
# ==============================================================================
def test_successful_grounded_answer_generation(
    mock_retriever: MagicMock,
    mock_gemini_client: MagicMock,
) -> None:
    """Verify that generate_answer retrieves context, formats prompt, calls Gemini, and returns GroundedAnswerResult."""
    generator = GroundedAnswerGenerator(
        retriever=mock_retriever,
        gemini_client=mock_gemini_client,
        model_name="gemini-2.5-flash",
        temperature=0.0,
    )

    result = generator.generate_answer(
        question="What is the operating voltage range of LM35?",
        k=2,
        strategy="hybrid",
    )

    assert isinstance(result, GroundedAnswerResult)
    assert result.question == "What is the operating voltage range of LM35?"
    assert "4 V to 30 V [lm35.pdf, p. 19]" in result.answer
    assert result.model == "gemini-2.5-flash"
    assert result.has_sufficient_context is True
    assert len(result.sources) == 2
    assert result.sources[0].source == "lm35.pdf"
    assert result.sources[0].page in (4, 19)

    # Verify retrieval call
    mock_retriever.retrieve_hybrid.assert_called_once_with(
        query="What is the operating voltage range of LM35?",
        k=2,
        dense_weight=0.5,
        sparse_weight=0.5,
    )

    # Verify Gemini client call
    mock_gemini_client.models.generate_content.assert_called_once()
    call_kwargs = mock_gemini_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-flash"
    assert "What is the operating voltage range of LM35?" in call_kwargs["contents"]
    assert "CRITICAL GROUNDING RULES" in call_kwargs["config"].system_instruction
    assert call_kwargs["config"].temperature == 0.0


# ==============================================================================
# 2. Missing Gemini API Key Handling
# ==============================================================================
def test_missing_api_key_handling(mock_retriever: MagicMock) -> None:
    """Verify ValueError is raised if Gemini client has no API key and none was provided."""
    with patch.dict("os.environ", {}, clear=True), \
         patch("app.rag.answer_generator.get_settings") as mock_get_settings:

        mock_settings = MagicMock()
        mock_settings.gemini_api_key = None
        mock_settings.gemini_model = "gemini-2.5-flash"
        mock_settings.gemini_temperature = 0.0
        mock_get_settings.return_value = mock_settings

        generator = GroundedAnswerGenerator(
            retriever=mock_retriever,
            gemini_client=None,
            api_key=None,
        )

        with pytest.raises(ValueError, match="Gemini API key is missing"):
            generator.generate_answer("What is LM35?")


# ==============================================================================
# 3. Empty Question Validation
# ==============================================================================
@pytest.mark.parametrize("invalid_q", ["", "   ", "\t\n  "])
def test_empty_question_validation(
    mock_retriever: MagicMock,
    mock_gemini_client: MagicMock,
    invalid_q: str,
) -> None:
    """Verify that empty or whitespace questions raise ValueError without calling retrieval or Gemini."""
    generator = GroundedAnswerGenerator(
        retriever=mock_retriever,
        gemini_client=mock_gemini_client,
    )

    with pytest.raises(ValueError, match="Question cannot be empty"):
        generator.generate_answer(invalid_q)

    mock_retriever.retrieve_hybrid.assert_not_called()
    mock_gemini_client.models.generate_content.assert_not_called()


# ==============================================================================
# 4. Empty Retrieved Context Fallback (Zero LLM Calls)
# ==============================================================================
def test_empty_retrieved_context_fallback(mock_gemini_client: MagicMock) -> None:
    """Verify that when 0 chunks are retrieved, a fallback is returned without calling Gemini."""
    empty_retriever = MagicMock()
    empty_retriever.retrieve_hybrid.return_value = []

    generator = GroundedAnswerGenerator(
        retriever=empty_retriever,
        gemini_client=mock_gemini_client,
    )

    result = generator.generate_answer("What is STM32F4 microcontroller?")

    assert result.has_sufficient_context is False
    assert "does not contain sufficient information" in result.answer
    assert result.sources == []
    assert result.retrieved_chunks == []

    # Verify Gemini was NEVER called when no context exists
    mock_gemini_client.models.generate_content.assert_not_called()


# ==============================================================================
# 5. Context Formatting and Citation Deduplication
# ==============================================================================
def test_context_formatting_and_citation_deduplication() -> None:
    """Verify format_context_for_prompt produces numbered passages and deduplicates source pages."""
    docs = [
        Document(page_content="Snippet 1", metadata={"source": "lm35.pdf", "page": 4}),
        Document(page_content="Snippet 2", metadata={"source": "lm35.pdf", "page": 4}),  # Same page
        Document(page_content="Snippet 3", metadata={"source": "lm35.pdf", "page": 19, "part_number": "LM35"}),
    ]

    context_str, citations = format_context_for_prompt(docs)

    assert "--- [Datasheet Passage 1] Source: lm35.pdf | Page: 4 ---" in context_str
    assert "--- [Datasheet Passage 2] Source: lm35.pdf | Page: 4 ---" in context_str
    assert "--- [Datasheet Passage 3] Source: lm35.pdf | Page: 19 | Part: LM35 ---" in context_str

    # Exactly 2 unique citations (page 4 and page 19)
    assert len(citations) == 2
    assert citations[0].source == "lm35.pdf"
    assert citations[0].page == 4
    assert citations[1].source == "lm35.pdf"
    assert citations[1].page == 19
    assert citations[1].part_number == "LM35"


# ==============================================================================
# 6. Insufficient Context Detection Flag
# ==============================================================================
def test_insufficient_context_detection(
    mock_retriever: MagicMock,
    mock_gemini_client: MagicMock,
) -> None:
    """Verify has_sufficient_context is False when Gemini explicitly states lack of information."""
    insufficient_response = MagicMock()
    insufficient_response.text = (
        "The provided datasheet context does not contain sufficient information to answer "
        "the exact sleep current in deep sleep mode."
    )
    mock_gemini_client.models.generate_content.return_value = insufficient_response

    generator = GroundedAnswerGenerator(
        retriever=mock_retriever,
        gemini_client=mock_gemini_client,
    )

    result = generator.generate_answer("What is the deep sleep current?")
    assert result.has_sufficient_context is False
    assert "does not contain sufficient information" in result.answer


# ==============================================================================
# 7. Gemini API Retry, Backoff, and Error Handling
# ==============================================================================
def test_successful_first_attempt(
    mock_retriever: MagicMock,
    mock_gemini_client: MagicMock,
) -> None:
    """Verify that successful response on attempt 1 succeeds immediately with zero sleep calls."""
    generator = GroundedAnswerGenerator(
        retriever=mock_retriever,
        gemini_client=mock_gemini_client,
        max_retries=3,
    )

    with patch("time.sleep") as mock_sleep:
        result = generator.generate_answer("What is the supply voltage?")
        assert result.answer != ""
        assert mock_gemini_client.models.generate_content.call_count == 1
        assert mock_sleep.call_count == 0


def test_503_followed_by_successful_retry(
    mock_retriever: MagicMock,
) -> None:
    """Verify that an HTTP 503 transient error triggers exponential backoff with jitter and succeeds on retry."""
    client = MagicMock()
    success_resp = MagicMock()
    success_resp.text = "LM35 operates from 4 V to 30 V [lm35.pdf, p. 19]."

    client.models.generate_content.side_effect = [
        errors.APIError(
            code=503,
            response_json={"error": {"message": "Service Unavailable - Temporary high load"}},
        ),
        success_resp,
    ]

    generator = GroundedAnswerGenerator(
        retriever=mock_retriever,
        gemini_client=client,
        max_retries=3,
        initial_backoff=1.0,
    )

    with patch("time.sleep") as mock_sleep:
        result = generator.generate_answer("What is the operating voltage?")

        assert "4 V to 30 V [lm35.pdf, p. 19]" in result.answer
        assert client.models.generate_content.call_count == 2
        assert mock_sleep.call_count == 1

        # Exponential backoff base 1.0 * (2^0) = 1.0 + jitter in [0.1, 0.5] -> 1.1 <= sleep_time <= 1.5
        sleep_arg = mock_sleep.call_args[0][0]
        assert 1.0 <= sleep_arg <= 1.6


def test_429_rate_limit_retry_success(
    mock_retriever: MagicMock,
) -> None:
    """Verify that an HTTP 429 rate limit error is retried and succeeds on the second attempt."""
    client = MagicMock()
    success_resp = MagicMock()
    success_resp.text = "LM35 scale factor is 10 mV/°C [lm35.pdf, p. 1]."

    client.models.generate_content.side_effect = [
        errors.APIError(
            code=429,
            response_json={"error": {"message": "Resource Exhausted: Quota exceeded"}},
        ),
        success_resp,
    ]

    generator = GroundedAnswerGenerator(
        retriever=mock_retriever,
        gemini_client=client,
        max_retries=3,
        initial_backoff=1.0,
    )

    with patch("time.sleep") as mock_sleep:
        result = generator.generate_answer("What is the scale factor?")
        assert "10 mV/°C" in result.answer
        assert client.models.generate_content.call_count == 2
        assert mock_sleep.call_count == 1


def test_network_timeout_retry_success(
    mock_retriever: MagicMock,
) -> None:
    """Verify that network timeout exceptions are treated as transient and retried successfully."""
    client = MagicMock()
    success_resp = MagicMock()
    success_resp.text = "LM35 accuracy is 0.5°C at 25°C [lm35.pdf, p. 1]."

    client.models.generate_content.side_effect = [
        httpx.TimeoutException("Connection timed out waiting for headers"),
        success_resp,
    ]

    generator = GroundedAnswerGenerator(
        retriever=mock_retriever,
        gemini_client=client,
        max_retries=3,
    )

    with patch("time.sleep") as mock_sleep:
        result = generator.generate_answer("What is the accuracy?")
        assert "0.5°C" in result.answer
        assert client.models.generate_content.call_count == 2
        assert mock_sleep.call_count == 1


def test_all_retries_exhausted(
    mock_retriever: MagicMock,
) -> None:
    """Verify that when all 3 attempts encounter retryable errors, a user-friendly error is raised."""
    client = MagicMock()
    client.models.generate_content.side_effect = errors.APIError(
        code=503,
        response_json={"error": {"message": "Service Unavailable"}},
    )

    generator = GroundedAnswerGenerator(
        retriever=mock_retriever,
        gemini_client=client,
        max_retries=3,
        initial_backoff=1.0,
    )

    with patch("time.sleep") as mock_sleep:
        with pytest.raises(RuntimeError, match="Service is currently unavailable after 3 attempts"):
            generator.generate_answer("What is the supply voltage?")

        # 3 API attempts made
        assert client.models.generate_content.call_count == 3
        # 2 backoff sleeps between 3 attempts
        assert mock_sleep.call_count == 2


def test_authentication_failure_without_retry(
    mock_retriever: MagicMock,
) -> None:
    """Verify that HTTP 403 (invalid API key / forbidden) fails immediately with zero retries."""
    client = MagicMock()
    client.models.generate_content.side_effect = errors.APIError(
        code=403,
        response_json={"error": {"message": "The caller does not have permission / Invalid API key"}},
    )

    generator = GroundedAnswerGenerator(
        retriever=mock_retriever,
        gemini_client=client,
        max_retries=3,
    )

    with patch("time.sleep") as mock_sleep:
        with pytest.raises(RuntimeError, match="Gemini API call failed \\(HTTP 403\\)"):
            generator.generate_answer("What is the pinout?")

        # Strictly 1 attempt and 0 sleep calls
        assert client.models.generate_content.call_count == 1
        assert mock_sleep.call_count == 0


def test_invalid_model_without_retry(
    mock_retriever: MagicMock,
) -> None:
    """Verify that HTTP 404 (model not found) fails immediately without retry."""
    client = MagicMock()
    client.models.generate_content.side_effect = errors.APIError(
        code=404,
        response_json={"error": {"message": "models/unknown-model is not found"}},
    )

    generator = GroundedAnswerGenerator(
        retriever=mock_retriever,
        gemini_client=client,
        max_retries=3,
    )

    with patch("time.sleep") as mock_sleep:
        with pytest.raises(RuntimeError, match="Gemini API call failed \\(HTTP 404\\)"):
            generator.generate_answer("What is the pinout?")

        assert client.models.generate_content.call_count == 1
        assert mock_sleep.call_count == 0


def test_unexpected_non_retryable_error(
    mock_retriever: MagicMock,
) -> None:
    """Verify that unexpected non-retryable programming errors raise RuntimeError immediately without retry."""
    client = MagicMock()
    client.models.generate_content.side_effect = TypeError("Unexpected data type received")

    generator = GroundedAnswerGenerator(
        retriever=mock_retriever,
        gemini_client=client,
        max_retries=3,
    )

    with patch("time.sleep") as mock_sleep:
        with pytest.raises(RuntimeError, match="Answer generation error"):
            generator.generate_answer("What is the pinout?")

        assert client.models.generate_content.call_count == 1
        assert mock_sleep.call_count == 0


# ==============================================================================
# 8. Strategy Routing
# ==============================================================================
def test_strategy_routing(
    mock_retriever: MagicMock,
    mock_gemini_client: MagicMock,
) -> None:
    """Verify generate_answer routes to the specified retrieval strategy."""
    mock_retriever.retrieve_mmr.return_value = [
        Document(page_content="MMR chunk", metadata={"source": "lm35.pdf", "page": 1})
    ]
    mock_retriever.retrieve_semantic.return_value = [
        Document(page_content="Semantic chunk", metadata={"source": "lm35.pdf", "page": 1})
    ]

    generator = GroundedAnswerGenerator(
        retriever=mock_retriever,
        gemini_client=mock_gemini_client,
    )

    generator.generate_answer("Test MMR", strategy="mmr")
    mock_retriever.retrieve_mmr.assert_called_once()

    generator.generate_answer("Test Semantic", strategy="semantic")
    mock_retriever.retrieve_semantic.assert_called_once()
