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

# Constants for retry and error handling
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404}
RETRYABLE_NETWORK_EXCEPTIONS = (
    TimeoutError,
    ConnectionError,
    httpx.TimeoutException,
    httpx.NetworkError,
)


def is_retryable_error(exc: Exception) -> bool:
    """Determine whether an error from the Gemini API or network stack is transient and retryable."""
    if isinstance(exc, errors.APIError):
        return exc.code in RETRYABLE_STATUS_CODES
    return isinstance(exc, RETRYABLE_NETWORK_EXCEPTIONS)


def is_auth_or_client_error(exc: Exception) -> bool:
    """Determine whether an error is a permanent client or authentication error that should not be retried."""
    if isinstance(exc, errors.APIError):
        return exc.code in NON_RETRYABLE_STATUS_CODES
    return False

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
    answer: str = Field(description="Concise, grounded answer with page citations")
    sources: List[SourceCitation] = Field(default_factory=list, description="Unique source citations referenced")
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list, description="Metadata of retrieved passages")
    model: str = Field(description="Gemini model used for synthesis")
    strategy: str = Field(default="hybrid", description="Retrieval strategy employed")
    has_sufficient_context: bool = Field(default=True, description="Whether context contained adequate information")


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
        **kwargs: Any,
    ) -> None:
        """Initialize the answer generator with retrieval engine and Google GenAI client.

        Args:
            retriever: Configured DatasheetRetriever instance.
            gemini_client: Pre-configured Gemini client (useful for dependency injection and mocking).
            api_key: Explicit Gemini API key. If omitted, reads from settings / environment.
            model_name: Gemini model identifier (e.g. 'gemini-2.5-flash').
            temperature: LLM sampling temperature (defaults to 0.0 for determinism).
            max_retries: Maximum number of attempts for transient API errors (default: 3).
            initial_backoff: Base backoff delay in seconds for exponential backoff (default: 1.0).
            max_backoff: Maximum backoff ceiling in seconds (default: 10.0).
            **kwargs: Backward-compatible support for legacy client kwargs.
        """
        settings = get_settings()

        self.retriever = retriever or DatasheetRetriever()
        self.model_name = model_name or settings.gemini_model
        self.temperature = temperature if temperature is not None else settings.gemini_temperature
        self.max_retries = max(1, max_retries)
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff

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
                logger.info("Initializing Google GenAI client.")
                self.client = genai.Client(api_key=effective_key)

    def generate_answer(
        self,
        question: str,
        k: int = 5,
        strategy: str = "hybrid",
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        fetch_k: int = 15,
        lambda_mult: float = 0.7,
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

        Returns:
            GroundedAnswerResult object containing answer text, citations, and metadata.

        Raises:
            ValueError: If the question is empty or the Gemini client cannot be initialized.
            RuntimeError: If the Gemini API request fails.
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
            )

        # Step 3: Format context and citations
        context_text, citations = format_context_for_prompt(retrieved_docs)
        chunks_metadata = [
            {
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "chunk_id": d.metadata.get("chunk_id"),
                "rrf_score": d.metadata.get("rrf_score"),
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

        # Step 5: Execute Google Gemini generation with exponential backoff & jitter
        logger.info("Calling Gemini model '%s' (temperature=%.1f)...", self.model_name, self.temperature)
        config = types.GenerateContentConfig(
            system_instruction=GROUNDED_SYSTEM_PROMPT,
            temperature=self.temperature,
        )

        last_error: Optional[Exception] = None
        raw_answer: Optional[str] = None

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

                # Case 1: Non-retryable authentication or client configuration error (400, 401, 403, 404)
                if is_auth_or_client_error(err):
                    code = getattr(err, "code", "UNKNOWN")
                    msg = getattr(err, "message", None) or str(err)
                    # Never log raw API credentials or tokens
                    logger.error(
                        "Gemini non-retryable error (HTTP %s): %s. Aborting without retry.",
                        code,
                        msg,
                    )
                    raise RuntimeError(
                        f"Gemini API call failed (HTTP {code}): Authentication or request error. Please verify credentials or model name."
                    ) from err

                # Case 2: Transient, retryable error (429, 500, 503, network timeouts)
                if is_retryable_error(err) and attempt < self.max_retries:
                    backoff = min(
                        self.initial_backoff * (2 ** (attempt - 1)),
                        self.max_backoff,
                    )
                    jitter = random.uniform(0.1, 0.5)
                    sleep_time = backoff + jitter
                    err_code = getattr(err, "code", type(err).__name__)
                    err_msg = getattr(err, "message", None) or str(err)
                    logger.warning(
                        "Temporary Gemini API error (attempt %d/%d, code=%s): %s. Retrying in %.2fs...",
                        attempt,
                        self.max_retries,
                        err_code,
                        err_msg,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                    continue

                # Case 3: Other unexpected non-retryable errors
                if not is_retryable_error(err):
                    logger.error("Non-retryable unexpected error during Gemini generation: %s", str(err))
                    raise RuntimeError(f"Answer generation error: {err}") from err

        if raw_answer is None:
            # All retries exhausted
            logger.error("All %d attempts to call Gemini API failed.", self.max_retries)
            raise RuntimeError(
                f"Gemini API call failed: Service is currently unavailable after {self.max_retries} attempts ({last_error})."
            ) from last_error

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
        )


def generate_grounded_answer(
    question: str,
    k: int = 5,
    strategy: str = "hybrid",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    max_retries: int = 3,
) -> GroundedAnswerResult:
    """Convenience helper to generate a grounded answer for a single question using Google Gemini.

    Args:
        question: User engineering question.
        k: Number of chunks to retrieve for context.
        strategy: Retrieval strategy ('hybrid', 'mmr', 'semantic', 'keyword').
        api_key: Optional Gemini API key override.
        model_name: Optional Gemini model name override.
        max_retries: Maximum number of retry attempts for transient API errors (default: 3).

    Returns:
        GroundedAnswerResult containing synthesized answer and citations.
    """
    generator = GroundedAnswerGenerator(
        api_key=api_key,
        model_name=model_name,
        max_retries=max_retries,
    )
    return generator.generate_answer(question=question, k=k, strategy=strategy)
