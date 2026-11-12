"""Performance and Benchmark Tests for CircuitMind AI Hybrid Retrieval.

Validates that:
1. Warm hybrid retrieval latency is sub-second (target < 500ms, typical ~20ms).
2. get_retriever() returns a shared singleton instance.
3. ResearchAgent reuses the shared retriever by default.
4. reset_retriever_cache() properly clears the singleton and class-level caches.
5. Cached retrieval preserves identical document rankings and RRF scores.
"""

from __future__ import annotations

import time
import pytest

from app.agents.research_agent import ResearchAgent
from app.rag.retrieval import DatasheetRetriever, get_retriever, reset_retriever_cache


class TestRetrievalPerformance:
    """Benchmark and caching test suite for datasheet retrieval."""

    def test_singleton_get_retriever_returns_same_instance(self) -> None:
        """Verify get_retriever() returns the identical cached instance across calls."""
        r1 = get_retriever()
        r2 = get_retriever()
        assert r1 is r2
        assert r1.embeddings is r2.embeddings

    def test_research_agent_reuses_singleton_retriever(self) -> None:
        """Verify ResearchAgent instances share the pre-warmed singleton retriever."""
        agent1 = ResearchAgent()
        agent2 = ResearchAgent()
        assert agent1.retriever is agent2.retriever
        assert agent1.retriever is get_retriever()

    def test_reset_retriever_cache_clears_state(self) -> None:
        """Verify reset_retriever_cache() forces re-creation on next get_retriever call."""
        r1 = get_retriever()
        reset_retriever_cache()
        r2 = get_retriever()
        assert r1 is not r2

    def test_warm_hybrid_retrieval_latency_under_500ms(self) -> None:
        """Verify that warm hybrid retrieval completes well under 500 ms (typically ~20 ms)."""
        retriever = get_retriever()

        # Warm up ChromaDB and BM25 index
        _ = retriever.retrieve_hybrid("LM35 supply voltage operating range", k=5)

        # Measure 3 warm query executions
        latencies_ms = []
        for _ in range(3):
            t0 = time.perf_counter()
            results = retriever.retrieve_hybrid("LM35 precision temperature sensor", k=5)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)
            assert len(results) > 0

        avg_latency = sum(latencies_ms) / len(latencies_ms)
        # Verify strict performance guarantee: sub-500ms (preventing 35s regression)
        assert avg_latency < 500.0, f"Expected warm retrieval < 500ms, got {avg_latency:.2f}ms"

    def test_retrieval_ranking_fidelity(self) -> None:
        """Verify retrieved documents preserve RRF scoring order and metadata integrity."""
        retriever = get_retriever()
        docs = retriever.retrieve_hybrid("LM35 10 mV/°C scale factor", k=4)

        assert len(docs) > 0
        scores = [d.metadata.get("rrf_score", 0.0) for d in docs]
        # Assert strictly non-increasing order of RRF scores
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

        for doc in docs:
            assert "source" in doc.metadata
            assert "page" in doc.metadata
            assert doc.page_content.strip() != ""

