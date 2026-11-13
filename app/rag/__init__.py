"""RAG layer: Datasheet parsing, chunking, embeddings, vector storage, and retrieval."""

from app.rag.ingestion import DatasheetIngestor, generate_chunk_id
from app.rag.retrieval import DatasheetRetriever, get_retriever, reset_retriever_cache
from app.rag.answer_generator import (
    GeminiErrorKind,
    GeminiGenerationError,
    GroundedAnswerGenerator,
    GroundedAnswerResult,
    SourceCitation,
    classify_gemini_error,
    generate_grounded_answer,
    sanitize_error_message,
)

__all__ = [
    "DatasheetIngestor",
    "DatasheetRetriever",
    "GeminiErrorKind",
    "GeminiGenerationError",
    "GroundedAnswerGenerator",
    "GroundedAnswerResult",
    "SourceCitation",
    "classify_gemini_error",
    "generate_chunk_id",
    "generate_grounded_answer",
    "get_retriever",
    "reset_retriever_cache",
    "sanitize_error_message",
]
