"""Grounded RAG Answer Generation Engine for CircuitMind AI.

Orchestrates hybrid datasheet retrieval, context formatting, and deterministic
Google Gemini answer synthesis using the official google-genai SDK.
Enforces strict zero-hallucination grounding rules, mandatory source page
citations, and explicit distinction between datasheet specifications and
engineering assumptions.
"""

from __future__ import annotations

import logging
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field
from langchain_core.documents import Document

from app.config import get_settings
from app.rag.retrieval import DatasheetRetriever

logger = logging.getLogger(__name__)


# ==============================================================================
# Error Classification & Classification Constants
# ==============================================================================
class GeminiErrorKind:
    """Standardized classification categories for Google Gemini API errors."""

    RATE_LIMIT = "rate_limit"              # HTTP 429 / RESOURCE_EXHAUSTED
    TRANSIENT_SERVER = "transient_server"  # HTTP 500, 502, 503, 504 / UNAVAILABLE / INTERNAL
    INVALID_REQUEST = "invalid_request"    # HTTP 400 / INVALID_ARGUMENT
    AUTH_FAILURE = "auth_failure"          # HTTP 401, 403 / UNAUTHENTICATED / PERMISSION_DENIED
    NOT_FOUND = "not_found"                # HTTP 404 / NOT_FOUND
    TIMEOUT = "timeout"                    # Network or request timeout
    NETWORK_ERROR = "network_error"        # Connection, DNS, or socket error
    UNKNOWN = "unknown"                    # Unclassified exception


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404}
RETRYABLE_NETWORK_EXCEPTIONS = (
    TimeoutError,
    ConnectionError,
    httpx.TimeoutException,
    httpx.NetworkError,
)


def sanitize_error_message(msg: str) -> str:
    """Remove potential API keys, tokens, or credential substrings from error text.

    Ensures no secrets or sensitive environment variables leak into logs, exceptions,
    or API responses.
    """
    if not msg:
        return ""
    sanitized = str(msg)

    # 1. Redact explicit secrets from environment or settings if present
    try:
        settings = get_settings()
        configured_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if configured_key and len(configured_key) > 6 and configured_key in sanitized:
            sanitized = sanitized.replace(configured_key, "[REDACTED_API_KEY]")
    except Exception:
        pass

    # 2. Redact patterns like key=..., api_key=..., Bearer ..., AIza..., AQ....
    sanitized = re.sub(
        r"(?:api_?key|key|token|auth)=([a-zA-Z0-9_\-\.]{8,})",
        r"key=[REDACTED]",
        sanitized,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(
        r"Bearer\s+([a-zA-Z0-9_\-\.]{8,})",
        "Bearer [REDACTED]",
        sanitized,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(r"\bAIza[0-9A-Za-z-_]{35}\b", "[REDACTED_API_KEY]", sanitized)
    sanitized = re.sub(r"\bAQ\.[0-9A-Za-z-_]{40,}\b", "[REDACTED_API_KEY]", sanitized)

    return sanitized


def classify_gemini_error(exc: Exception) -> Tuple[str, Optional[int], str]:
    """Classify an exception from Gemini or network stack into standardized category.

    Args:
        exc: Exception raised during generation.

    Returns:
        Tuple of (error_kind, status_code, actionable_message).
    """
    code: Optional[int] = None
    status_str: str = ""
    raw_msg: str = str(exc)

    if isinstance(exc, errors.APIError):
        code = getattr(exc, "code", None)
        status_str = str(getattr(exc, "status", "")).upper()
        raw_msg = getattr(exc, "message", None) or str(exc)

    sanitized_msg = sanitize_error_message(raw_msg)

    # 1. Timeouts
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)) or "timeout" in sanitized_msg.lower():
        return (
            GeminiErrorKind.TIMEOUT,
            code or 504,
            "Request to Google Gemini timed out. The model took too long to generate a response.",
        )

    # 2. Connection / Network errors
    if isinstance(exc, (ConnectionError, httpx.NetworkError)):
        return (
            GeminiErrorKind.NETWORK_ERROR,
            code or 503,
            "Network connectivity error while reaching Google Gemini API. Please check your internet connection.",
        )

    # 3. HTTP 429: Rate limit / Resource exhausted
    if code == 429 or "RESOURCE_EXHAUSTED" in status_str or "quota" in sanitized_msg.lower():
        return (
            GeminiErrorKind.RATE_LIMIT,
            429,
            "Gemini API rate limit or quota exceeded (HTTP 429). Please wait before submitting another request.",
        )

    # 4. HTTP 500, 502, 503, 504: Transient server errors
    if code in (500, 502, 503, 504) or any(s in status_str for s in ("UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED")):
        return (
            GeminiErrorKind.TRANSIENT_SERVER,
            code or 503,
            f"Google Gemini service is temporarily unavailable (HTTP {code or 503}). Spikes in demand are temporary.",
        )

    # 5. HTTP 401, 403: Authentication or permission failure
    if code in (401, 403) or any(s in status_str for s in ("UNAUTHENTICATED", "PERMISSION_DENIED")):
        return (
            GeminiErrorKind.AUTH_FAILURE,
            code or 403,
            f"Authentication failed (HTTP {code or 403}). Please verify that GEMINI_API_KEY is valid and authorized.",
        )

    # 6. HTTP 404: Model or resource not found
    if code == 404 or "NOT_FOUND" in status_str:
        return (
            GeminiErrorKind.NOT_FOUND,
            404,
            "Gemini model or resource not found (HTTP 404). Please verify the model identifier.",
        )

    # 7. HTTP 400: Invalid request
    if code == 400 or "INVALID_ARGUMENT" in status_str:
        return (
            GeminiErrorKind.INVALID_REQUEST,
            400,
            "Invalid request sent to Gemini API (HTTP 400). Parameter or prompt syntax was rejected.",
        )

    return (GeminiErrorKind.UNKNOWN, code, sanitized_msg)


def extract_retry_after_seconds(exc: Exception) -> Optional[float]:
    """Extract Retry-After delay in seconds from HTTP headers or APIError response payload."""
    # 1. Check HTTP response headers if available on APIError
    response_obj = getattr(exc, "response", None)
    if response_obj is not None:
        headers = getattr(response_obj, "headers", None)
        if headers and hasattr(headers, "get"):
            header_val = headers.get("retry-after") or headers.get("Retry-After")
            if header_val:
                try:
                    return max(0.0, float(header_val))
                except (ValueError, TypeError):
                    pass

    # 2. Check response_json details for RetryInfo (standard Google RPC detail)
    response_json = getattr(exc, "response_json", None)
    if isinstance(response_json, dict):
        error_dict = response_json.get("error", {})
        if isinstance(error_dict, dict):
            details = error_dict.get("details", [])
            if isinstance(details, list):
                for item in details:
                    if isinstance(item, dict) and "retryDelay" in item:
                        delay_str = str(item["retryDelay"]).rstrip("s")
                        try:
                            return max(0.0, float(delay_str))
                        except (ValueError, TypeError):
                            pass

    return None


def is_retryable_error(exc: Exception) -> bool:
    """Determine whether an error from the Gemini API or network stack is transient and retryable."""
    kind, _, _ = classify_gemini_error(exc)
    return kind in (
        GeminiErrorKind.RATE_LIMIT,
        GeminiErrorKind.TRANSIENT_SERVER,
        GeminiErrorKind.TIMEOUT,
        GeminiErrorKind.NETWORK_ERROR,
    )


def is_auth_or_client_error(exc: Exception) -> bool:
    """Determine whether an error is a permanent client or authentication error that should not be retried."""
    kind, _, _ = classify_gemini_error(exc)
    return kind in (
        GeminiErrorKind.AUTH_FAILURE,
        GeminiErrorKind.NOT_FOUND,
        GeminiErrorKind.INVALID_REQUEST,
    )


class GeminiGenerationError(RuntimeError):
    """Exception raised when Google Gemini answer generation fails, preserving retrieved evidence."""

    def __init__(
        self,
        message: str,
        *,
        error_kind: str = GeminiErrorKind.UNKNOWN,
        status_code: Optional[int] = None,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
        sources: Optional[List[SourceCitation]] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.error_kind = error_kind
        self.status_code = status_code
        self.retrieved_chunks = retrieved_chunks or []
        self.sources = sources or []
        self.original_error = original_error


GROUNDED_SYSTEM_PROMPT = """You are CircuitMind AI, an expert agentic electronics engineering copilot.
Your objective is to provide concise, rigorous, and strictly grounded answers to electronics hardware engineering questions based ONLY on the provided datasheet context passages.

CRITICAL GROUNDING RULES:
1. STRICT FACTUAL GROUNDING: Base your answer ONLY on the technical specifications, tables, pinouts, and characteristics present in the provided context. Never invent, fabricate, or hallucinate technical parameters.
2. MANDATORY SOURCE CITATIONS: For EVERY technical claim, specification, rating, or number, you MUST cite the source datasheet filename and page number in brackets, for example: [lm35.pdf, p. 4].
3. INSUFFICIENT CONTEXT: If the context does not contain enough information to answer the question, state explicitly: "The provided datasheet context does not contain sufficient information to answer [specific detail]." Do NOT guess.
4. SPECIFICATIONS VS. CALCULATIONS: Clearly distinguish exact values ("Datasheet Specification") from any derived calculations or engineering assumptions ("Engineering Calculation / Assumption").
5. CONCISE & ENGINEERING-FOCUSED: Present information clearly, using bullet points for multi-parameter answers. Avoid conversational filler or boilerplate text.
"""


class SourceCitation(BaseModel):
    """Source datasheet and page reference citation."""

    source: str = Field(description="Datasheet PDF filename")
    page: Any = Field(description="Page number where specification originates")
    part_number: Optional[str] = Field(default=None, description="Component part number if available")


class GroundedAnswerResult(BaseModel):
    """Structured response model for grounded RAG answer generation."""

    question: str = Field(description="The original user engineering question")
    answer: str = Field(description="Concise, grounded answer with page citations or clear error description")
    sources: List[SourceCitation] = Field(default_factory=list, description="Unique source citations referenced")
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list, description="Metadata of retrieved passages")
    model: str = Field(description="Gemini model used for synthesis")
    strategy: str = Field(default="hybrid", description="Retrieval strategy employed")
    has_sufficient_context: bool = Field(default=True, description="Whether context contained adequate information")
    generation_success: bool = Field(default=True, description="Whether Gemini model generation succeeded")
    error: Optional[str] = Field(default=None, description="Sanitized, actionable error message if generation failed")
    error_kind: Optional[str] = Field(default=None, description="Classification of error if generation failed")


def format_context_for_prompt(docs: List[Document]) -> Tuple[str, List[SourceCitation]]:
    """Format retrieved document chunks into a numbered context block and extract unique citations.

    Args:
        docs: List of retrieved LangChain Document chunks.

    Returns:
        Tuple of (formatted_context_string, list_of_unique_source_citations).
    """
    if not docs:
        return "No relevant datasheet passages were retrieved.", []

    formatted_passages: List[str] = []
    unique_citations: List[SourceCitation] = []
    seen_sources: set[Tuple[str, Any]] = set()

    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown_datasheet.pdf")
        page = doc.metadata.get("page", "unknown")
        part = doc.metadata.get("part_number")

        key = (source, page)
        if key not in seen_sources:
            seen_sources.add(key)
            unique_citations.append(
                SourceCitation(source=source, page=page, part_number=part)
            )

        part_tag = f" | Part: {part}" if part else ""
        header = f"--- [Datasheet Passage {idx}] Source: {source} | Page: {page}{part_tag} ---"
        formatted_passages.append(f"{header}\n{doc.page_content.strip()}")

    context_str = "\n\n".join(formatted_passages)
    return context_str, unique_citations


class GroundedAnswerGenerator:
    """Generates strictly grounded technical answers from electronic component datasheets using Google Gemini."""

    def __init__(
        self,
        retriever: Optional[DatasheetRetriever] = None,
        gemini_client: Optional[genai.Client] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        max_backoff: float = 10.0,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the answer generator with retrieval engine and Google GenAI client.

        Args:
            retriever: Configured DatasheetRetriever instance.
            gemini_client: Pre-configured Gemini client (useful for dependency injection and mocking).
            api_key: Explicit Gemini API key. If omitted, reads from settings / environment.
            model_name: Gemini model identifier (e.g. 'gemini-3.8-flash').
            temperature: LLM sampling temperature (defaults to 0.0 for determinism).
            max_retries: Maximum number of attempts for transient API errors (default: 3).
            initial_backoff: Base backoff delay in seconds for exponential backoff (default: 1.0).
            max_backoff: Maximum backoff ceiling in seconds (default: 10.0).
            timeout: Request timeout in seconds for Gemini API calls (default: 30.0).
            **kwargs: Backward-compatible support for legacy client kwargs.
        """
        settings = get_settings()

        self.retriever = retriever or DatasheetRetriever()
        self.model_name = model_name or settings.gemini_model
        self.temperature = temperature if temperature is not None else settings.gemini_temperature
        self.max_retries = max(1, max_retries)
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.timeout = timeout if timeout is not None else getattr(settings, "gemini_timeout_seconds", 30.0)

        # Setup Gemini client
        injected_client = gemini_client or kwargs.get("openai_client")
        if injected_client is not None:
            self.client = injected_client
        else:
            effective_key = (
                api_key
                or settings.gemini_api_key
                or os.getenv("GEMINI_API_KEY")
            )
            if not effective_key:
                logger.warning("GEMINI_API_KEY is not configured.")
                self.client = None
            else:
                # Never log raw credentials
                logger.info("Initializing Google GenAI client with %.1fs timeout.", self.timeout)
                http_options = types.HttpOptions(timeout=int(self.timeout * 1000))
                self.client = genai.Client(api_key=effective_key, http_options=http_options)

    def generate_answer(
        self,
        question: str,
        k: int = 5,
        strategy: str = "hybrid",
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        fetch_k: int = 15,
        lambda_mult: float = 0.7,
        raise_on_error: bool = True,
    ) -> GroundedAnswerResult:
        """Retrieve datasheet context and generate a grounded, cited engineering answer using Google Gemini.

        Args:
            question: User engineering query.
            k: Number of retrieved chunks to provide as context.
            strategy: Retrieval strategy ('hybrid', 'mmr', 'semantic', 'keyword').
            dense_weight: Weight for dense search in hybrid mode.
            sparse_weight: Weight for sparse search in hybrid mode.
            fetch_k: Candidates pool for MMR mode.
            lambda_mult: Diversity parameter for MMR mode.
            raise_on_error: Whether to raise GeminiGenerationError on failure (default: True).
                            If False, returns a GroundedAnswerResult with generation_success=False
                            and preserved retrieved evidence.

        Returns:
            GroundedAnswerResult object containing answer text, citations, and metadata.

        Raises:
            ValueError: If the question is empty or the Gemini client cannot be initialized.
            GeminiGenerationError / RuntimeError: If Gemini API request fails and raise_on_error is True.
        """
        # Validate question
        if not question or not question.strip():
            raise ValueError("Question cannot be empty or contain only whitespace.")

        clean_question = question.strip()

        # Check Gemini client readiness
        if self.client is None:
            raise ValueError(
                "Gemini API key is missing. Please set the GEMINI_API_KEY environment variable "
                "or configure it in your .env file."
            )

        # Step 1: Execute retrieval
        logger.info("Retrieving context for query %r using '%s' strategy...", clean_question, strategy)
        strat = strategy.lower().strip()
        if strat == "hybrid":
            retrieved_docs = self.retriever.retrieve_hybrid(
                query=clean_question,
                k=k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )
        elif strat == "mmr":
            retrieved_docs = self.retriever.retrieve_mmr(
                query=clean_question,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
            )
        elif strat == "semantic":
            retrieved_docs = self.retriever.retrieve_semantic(
                query=clean_question,
                k=k,
            )
        elif strat in ("keyword", "bm25"):
            retrieved_docs = self.retriever.retrieve_keyword(
                query=clean_question,
                k=k,
            )
        else:
            raise ValueError(f"Unsupported retrieval strategy: '{strategy}'")

        # Step 2: Handle empty retrieval results
        if not retrieved_docs:
            logger.warning("No datasheet context retrieved for query: %r", clean_question)
            return GroundedAnswerResult(
                question=clean_question,
                answer=(
                    "The provided datasheet context does not contain sufficient information to answer "
                    "this question. No matching component specifications were found in the database."
                ),
                sources=[],
                retrieved_chunks=[],
                model=self.model_name,
                strategy=strat,
                has_sufficient_context=False,
                generation_success=True,
            )

        # Step 3: Format context and citations
        context_text, citations = format_context_for_prompt(retrieved_docs)
        chunks_metadata = [
            {
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "chunk_id": d.metadata.get("chunk_id"),
                "rrf_score": d.metadata.get("rrf_score"),
                "part_number": d.metadata.get("part_number"),
                "content": d.page_content,
            }
            for d in retrieved_docs
        ]

        # Step 4: Construct user message
        user_message_content = (
            f"ENGINEERING QUESTION:\n{clean_question}\n\n"
            f"RETRIEVED DATASHEET CONTEXT:\n{context_text}\n\n"
            "Provide a concise, technically rigorous answer based strictly on the context above. "
            "Cite the source datasheet and page number for every factual statement using [source, p. X] format."
        )

        # Step 5: Execute Google Gemini generation with exponential backoff, jitter & Retry-After support
        logger.info("Calling Gemini model '%s' (temperature=%.1f)...", self.model_name, self.temperature)
        timeout_ms = int(self.timeout * 1000)
        config = types.GenerateContentConfig(
            system_instruction=GROUNDED_SYSTEM_PROMPT,
            temperature=self.temperature,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )

        last_error: Optional[Exception] = None
        raw_answer: Optional[str] = None
        last_kind: str = GeminiErrorKind.UNKNOWN
        last_code: Optional[int] = None
        last_actionable_msg: str = ""

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_message_content,
                    config=config,
                )
                raw_answer = response.text or ""
                break  # Successful generation, exit retry loop
            except Exception as err:
                last_error = err
                kind, code, actionable_msg = classify_gemini_error(err)
                last_kind = kind
                last_code = code
                last_actionable_msg = actionable_msg

                # Case 1: Non-retryable client, auth, or model errors (400, 401, 403, 404)
                if is_auth_or_client_error(err):
                    sanitized_code = code or "UNKNOWN"
                    logger.error(
                        "Gemini non-retryable error (HTTP %s): %s. Aborting immediately without retry.",
                        sanitized_code,
                        actionable_msg,
                    )
                    error_msg = (
                        f"Gemini API call failed (HTTP {sanitized_code}): "
                        f"Authentication or request error. Please verify credentials or model name."
                    )
                    if raise_on_error:
                        raise GeminiGenerationError(
                            error_msg,
                            error_kind=kind,
                            status_code=code,
                            retrieved_chunks=chunks_metadata,
                            sources=citations,
                            original_error=err,
                        ) from err
                    return GroundedAnswerResult(
                        question=clean_question,
                        answer=(
                            f"Answer generation failed ({error_msg}). "
                            f"{len(retrieved_docs)} datasheet passage(s) were retrieved successfully "
                            "and are preserved below for review."
                        ),
                        sources=citations,
                        retrieved_chunks=chunks_metadata,
                        model=self.model_name,
                        strategy=strat,
                        has_sufficient_context=False,
                        generation_success=False,
                        error=error_msg,
                        error_kind=kind,
                    )

                # Case 2: Transient, retryable error (429, 500, 502, 503, 504, network timeouts)
                if is_retryable_error(err) and attempt < self.max_retries:
                    retry_after = extract_retry_after_seconds(err)
                    jitter = random.uniform(0.1, 0.5)
                    if retry_after is not None and retry_after > 0:
                        sleep_time = min(retry_after + jitter, self.max_backoff)
                        logger.info("Respecting Retry-After header: sleeping for %.2fs", sleep_time)
                    else:
                        backoff = min(
                            self.initial_backoff * (2 ** (attempt - 1)),
                            self.max_backoff,
                        )
                        sleep_time = backoff + jitter

                    err_code = code or type(err).__name__
                    logger.warning(
                        "Temporary Gemini API error (attempt %d/%d, code=%s): %s. Retrying in %.2fs...",
                        attempt,
                        self.max_retries,
                        err_code,
                        actionable_msg,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                    continue

                # Case 3: Other unexpected non-retryable errors
                if not is_retryable_error(err):
                    sanitized_err_str = sanitize_error_message(str(err))
                    logger.error("Non-retryable unexpected error during Gemini generation: %s", sanitized_err_str)
                    error_msg = f"Answer generation error: {sanitized_err_str}"
                    if raise_on_error:
                        raise GeminiGenerationError(
                            error_msg,
                            error_kind=kind,
                            status_code=code,
                            retrieved_chunks=chunks_metadata,
                            sources=citations,
                            original_error=err,
                        ) from err
                    return GroundedAnswerResult(
                        question=clean_question,
                        answer=(
                            f"Answer generation error: {sanitized_err_str}. "
                            f"{len(retrieved_docs)} datasheet passage(s) were retrieved successfully "
                            "and are preserved below for review."
                        ),
                        sources=citations,
                        retrieved_chunks=chunks_metadata,
                        model=self.model_name,
                        strategy=strat,
                        has_sufficient_context=False,
                        generation_success=False,
                        error=error_msg,
                        error_kind=kind,
                    )

        if raw_answer is None:
            # All retries exhausted
            logger.error("All %d attempts to call Gemini API failed.", self.max_retries)
            sanitized_last = sanitize_error_message(str(last_error))
            exhausted_msg = (
                f"Gemini API call failed: Service is currently unavailable after {self.max_retries} attempts "
                f"({sanitized_last})."
            )
            if raise_on_error:
                raise GeminiGenerationError(
                    exhausted_msg,
                    error_kind=last_kind,
                    status_code=last_code,
                    retrieved_chunks=chunks_metadata,
                    sources=citations,
                    original_error=last_error,
                ) from last_error
            return GroundedAnswerResult(
                question=clean_question,
                answer=(
                    f"Answer generation failed: Service remained unavailable after {self.max_retries} attempts. "
                    f"{len(retrieved_docs)} datasheet passage(s) were retrieved successfully and are preserved below."
                ),
                sources=citations,
                retrieved_chunks=chunks_metadata,
                model=self.model_name,
                strategy=strat,
                has_sufficient_context=False,
                generation_success=False,
                error=exhausted_msg,
                error_kind=last_kind,
            )

        # Assess whether answer indicates insufficient context
        has_sufficient = "does not contain sufficient information" not in raw_answer.lower()

        return GroundedAnswerResult(
            question=clean_question,
            answer=raw_answer.strip(),
            sources=citations,
            retrieved_chunks=chunks_metadata,
            model=self.model_name,
            strategy=strat,
            has_sufficient_context=has_sufficient,
            generation_success=True,
        )


def generate_grounded_answer(
    question: str,
    k: int = 5,
    strategy: str = "hybrid",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    max_retries: int = 3,
    timeout: Optional[float] = None,
    raise_on_error: bool = True,
) -> GroundedAnswerResult:
    """Convenience helper to generate a grounded answer for a single question using Google Gemini.

    Args:
        question: User engineering question.
        k: Number of chunks to retrieve for context.
        strategy: Retrieval strategy ('hybrid', 'mmr', 'semantic', 'keyword').
        api_key: Optional Gemini API key override.
        model_name: Optional Gemini model name override.
        max_retries: Maximum number of retry attempts for transient API errors (default: 3).
        timeout: Optional request timeout in seconds.
        raise_on_error: Whether to raise on failure or return result with preserved evidence.

    Returns:
        GroundedAnswerResult containing synthesized answer and citations.
    """
    generator = GroundedAnswerGenerator(
        api_key=api_key,
        model_name=model_name,
        max_retries=max_retries,
        timeout=timeout,
    )
    return generator.generate_answer(
        question=question,
        k=k,
        strategy=strategy,
        raise_on_error=raise_on_error,
    )
