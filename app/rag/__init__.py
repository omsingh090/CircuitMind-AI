"""RAG layer: Datasheet parsing, chunking, embeddings, vector storage, and retrieval."""

from app.rag.ingestion import DatasheetIngestor, generate_chunk_id
from app.rag.retrieval import DatasheetRetriever, get_retriever, reset_retriever_cache
from app.rag.answer_generator import (
    GroundedAnswerGenerator,
    GroundedAnswerResult,
    SourceCitation,
    generate_grounded_answer,
)

__all__ = [
    "DatasheetIngestor",
    "DatasheetRetriever",
    "GroundedAnswerGenerator",
    "GroundedAnswerResult",
    "SourceCitation",
    "generate_chunk_id",
    "generate_grounded_answer",
    "get_retriever",
    "reset_retriever_cache",
]
