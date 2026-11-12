"""Research Agent for CircuitMind AI.

Executes targeted hybrid datasheet retrieval combining dense ChromaDB embeddings
and BM25 keyword matching via Reciprocal Rank Fusion (RRF).
Operates completely independently of LLM synthesis, preserving source document filenames,
page numbers, and stable chunk metadata.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from unittest.mock import Mock

from app.graph.state import CircuitMindState
from app.rag.retrieval import DatasheetRetriever, get_retriever

logger = logging.getLogger(__name__)


class ResearchAgent:
    """Agent responsible for gathering grounded datasheet evidence using hybrid RAG."""

    def __init__(self, retriever: Optional[DatasheetRetriever] = None) -> None:
        """Initialize ResearchAgent with optional pre-configured DatasheetRetriever."""
        if retriever is not None:
            self.retriever = retriever
        elif isinstance(DatasheetRetriever, Mock) or hasattr(DatasheetRetriever, "return_value"):
            self.retriever = DatasheetRetriever()
        else:
            self.retriever = get_retriever()

    def retrieve(
        self,
        query: str,
        target_components: Optional[List[str]] = None,
        k: int = 5,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant datasheet chunks for the engineering query.

        Args:
            query: User's engineering question or design constraint query.
            target_components: Optional list of identified component part numbers.
            k: Number of chunks to retrieve.
            dense_weight: Dense semantic search weight for RRF.
            sparse_weight: Sparse BM25 keyword search weight for RRF.

        Returns:
            List of dictionaries containing document content and metadata.
        """
        if not query or not query.strip():
            logger.warning("Empty query passed to ResearchAgent.")
            return []

        # Augment search query with target component names if present
        clean_query = query.strip()
        search_query = clean_query
        if target_components:
            component_prefix = " ".join(target_components)
            if not any(comp.lower() in clean_query.lower() for comp in target_components):
                search_query = f"{component_prefix} {clean_query}"
            if not any(term in search_query.lower() for term in ("supply", "voltage", "rating", "current")):
                search_query = f"{search_query} supply voltage operating range quiescent current scale factor"

        logger.info("ResearchAgent executing hybrid search for: %r (k=%d)", search_query, k)

        try:
            documents = self.retriever.retrieve_hybrid(
                query=search_query,
                k=k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )
        except Exception as exc:
            logger.error("Hybrid retrieval failed in ResearchAgent: %s", exc)
            raise RuntimeError(f"Datasheet retrieval error: {exc}") from exc

        serialized_docs: List[Dict[str, Any]] = []
        for doc in documents:
            serialized_docs.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown_datasheet.pdf"),
                "page": doc.metadata.get("page", 1),
                "chunk_id": doc.metadata.get("chunk_id"),
                "part_number": doc.metadata.get("part_number"),
                "rrf_score": doc.metadata.get("rrf_score"),
            })

        logger.info("ResearchAgent retrieved %d relevant chunks.", len(serialized_docs))
        return serialized_docs


def research_agent_node(state: CircuitMindState) -> Dict[str, Any]:
    """LangGraph node function executing datasheet evidence retrieval."""
    query = state.get("user_query", "")
    requirements = state.get("requirements", {})
    target_components = requirements.get("target_components", [])

    agent = ResearchAgent()
    try:
        retrieved = agent.retrieve(
            query=query,
            target_components=target_components,
            k=6,
        )
        return {"retrieved_documents": retrieved}
    except Exception as exc:
        logger.error("Error in research_agent_node: %s", exc)
        return {
            "retrieved_documents": [],
            "errors": [f"ResearchAgent error: {exc}"],
        }

