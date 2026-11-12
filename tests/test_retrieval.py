"""Comprehensive Unit Tests for CircuitMind AI Retrieval Engines and Evaluation Framework.

Tests:
1. Successful semantic retrieval.
2. Successful MMR retrieval.
3. Empty query handling.
4. Empty retrieval results.
5. Source metadata preservation.
6. Page number preservation.
7. Duplicate result handling.
8. Missing ChromaDB collection.
9. Retrieval evaluation metric calculations.
10. Hybrid retrieval result merging.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import chromadb
import pytest
from langchain_core.documents import Document

from app.rag.retrieval import DatasheetRetriever, compute_doc_identifier
from evaluation.evaluate_retrieval import RelevanceMatcher, evaluate_query


# ==============================================================================
# 1. Successful Semantic Retrieval
# ==============================================================================
def test_successful_semantic_retrieval() -> None:
    """Verify that semantic retrieval returns the requested number of Document chunks."""
    retriever = DatasheetRetriever()

    mock_docs = [
        Document(page_content="LM35 features linear 10-mV/°C scale factor.", metadata={"source": "lm35.pdf", "page": 1}),
        Document(page_content="Operating voltage range is 4 V to 30 V.", metadata={"source": "lm35.pdf", "page": 4}),
        Document(page_content="Absolute maximum supply voltage is 35 V.", metadata={"source": "lm35.pdf", "page": 4}),
    ]

    mock_store = MagicMock()
    mock_store.similarity_search.return_value = mock_docs

    with patch.object(retriever, "get_vector_store", return_value=mock_store):
        results = retriever.retrieve_semantic(query="What is the supply voltage range?", k=3)

        assert len(results) == 3
        assert results[0].page_content == "LM35 features linear 10-mV/°C scale factor."
        mock_store.similarity_search.assert_called_once_with(
            query="What is the supply voltage range?",
            k=3,
            filter=None,
        )


# ==============================================================================
# 2. Successful MMR Retrieval
# ==============================================================================
def test_successful_mmr_retrieval() -> None:
    """Verify that MMR retrieval invokes vector_store.max_marginal_relevance_search with proper parameters."""
    retriever = DatasheetRetriever()

    mock_docs = [
        Document(page_content="Page 1: General Features", metadata={"source": "lm35.pdf", "page": 1}),
        Document(page_content="Page 15: Circuit Diagram", metadata={"source": "lm35.pdf", "page": 15}),
    ]

    mock_store = MagicMock()
    mock_store.max_marginal_relevance_search.return_value = mock_docs

    with patch.object(retriever, "get_vector_store", return_value=mock_store):
        results = retriever.retrieve_mmr(
            query="temperature sensor circuit",
            k=2,
            fetch_k=8,
            lambda_mult=0.65,
        )

        assert len(results) == 2
        assert results[0].metadata["page"] == 1
        assert results[1].metadata["page"] == 15
        mock_store.max_marginal_relevance_search.assert_called_once_with(
            query="temperature sensor circuit",
            k=2,
            fetch_k=8,
            lambda_mult=0.65,
            filter=None,
        )


# ==============================================================================
# 3. Empty Query Handling
# ==============================================================================
@pytest.mark.parametrize("empty_query", ["", "   ", "\t", "\n\n  \t "])
def test_empty_query_handling(empty_query: str) -> None:
    """Verify that all retrieval methods raise ValueError with a helpful message on empty or whitespace queries."""
    retriever = DatasheetRetriever()

    with pytest.raises(ValueError, match="empty or contain only whitespace"):
        retriever.retrieve_semantic(empty_query)

    with pytest.raises(ValueError, match="empty or contain only whitespace"):
        retriever.retrieve_mmr(empty_query)

    with pytest.raises(ValueError, match="empty or contain only whitespace"):
        retriever.retrieve_keyword(empty_query)

    with pytest.raises(ValueError, match="empty or contain only whitespace"):
        retriever.retrieve_hybrid(empty_query)


# ==============================================================================
# 4. Empty Retrieval Results
# ==============================================================================
def test_empty_retrieval_results() -> None:
    """Verify that retrieval methods gracefully return an empty list when no documents match."""
    retriever = DatasheetRetriever()

    mock_store = MagicMock()
    mock_store.similarity_search.return_value = []
    mock_store.max_marginal_relevance_search.return_value = []

    mock_bm25 = MagicMock()
    mock_bm25.invoke.return_value = []

    with patch.object(retriever, "get_vector_store", return_value=mock_store), \
         patch.object(retriever, "get_keyword_retriever", return_value=mock_bm25):

        sem_results = retriever.retrieve_semantic("out-of-domain query", k=5)
        assert sem_results == []

        mmr_results = retriever.retrieve_mmr("out-of-domain query", k=5)
        assert mmr_results == []

        kw_results = retriever.retrieve_keyword("out-of-domain query", k=5)
        assert kw_results == []

        hybrid_results = retriever.retrieve_hybrid("out-of-domain query", k=5)
        assert hybrid_results == []


# ==============================================================================
# 5. Source Metadata Preservation
# ==============================================================================
def test_source_metadata_preservation() -> None:
    """Verify that all retrieved chunks preserve the source filename in metadata."""
    retriever = DatasheetRetriever()

    doc_sample = Document(
        page_content="LM35 Precision Centigrade Temperature Sensor",
        metadata={"source": "lm35.pdf", "page": 1, "chunk_id": "lm35_p1_001"},
    )

    with patch.object(retriever, "retrieve_semantic", return_value=[doc_sample]), \
         patch.object(retriever, "retrieve_keyword", return_value=[doc_sample]):

        fused = retriever.retrieve_hybrid("accuracy", k=1)
        assert len(fused) == 1
        assert "source" in fused[0].metadata
        assert fused[0].metadata["source"] == "lm35.pdf"


# ==============================================================================
# 6. Page Number Preservation
# ==============================================================================
def test_page_number_preservation() -> None:
    """Verify that the original PDF page number metadata is preserved as an integer."""
    retriever = DatasheetRetriever()

    doc_p4 = Document(
        page_content="Supply Voltage: -0.2V to 35V",
        metadata={"source": "lm35.pdf", "page": 4, "chunk_id": "lm35_p4_002"},
    )
    doc_p19 = Document(
        page_content="Power supply recommendations: 4V to 30V",
        metadata={"source": "lm35.pdf", "page": 19, "chunk_id": "lm35_p19_003"},
    )

    with patch.object(retriever, "retrieve_semantic", return_value=[doc_p4, doc_p19]), \
         patch.object(retriever, "retrieve_keyword", return_value=[doc_p4]):

        results = retriever.retrieve_hybrid("supply voltage", k=2)
        assert len(results) == 2
        assert "page" in results[0].metadata
        assert "page" in results[1].metadata
        assert isinstance(results[0].metadata["page"], int)
        assert isinstance(results[1].metadata["page"], int)
        assert results[0].metadata["page"] in (4, 19)


# ==============================================================================
# 7. Duplicate Result Handling
# ==============================================================================
def test_duplicate_result_handling() -> None:
    """Verify that identical chunks retrieved by both dense and sparse engines are deduplicated."""
    retriever = DatasheetRetriever()

    shared_chunk = Document(
        page_content="Identical technical specification chunk.",
        metadata={"source": "lm35.pdf", "page": 4, "chunk_id": "lm35_p4_shared"},
    )

    unique_sparse_chunk = Document(
        page_content="Unique sparse chunk.",
        metadata={"source": "lm35.pdf", "page": 10, "chunk_id": "lm35_p10_sparse"},
    )

    # Dense returns [shared_chunk], Sparse returns [shared_chunk, unique_sparse_chunk]
    with patch.object(retriever, "retrieve_semantic", return_value=[shared_chunk]), \
         patch.object(retriever, "retrieve_keyword", return_value=[shared_chunk, unique_sparse_chunk]):

        fused = retriever.retrieve_hybrid("test query", k=5)

        # Total unique chunks must be 2, not 3
        assert len(fused) == 2
        chunk_ids = [d.metadata["chunk_id"] for d in fused]
        assert len(chunk_ids) == len(set(chunk_ids)), "Result chunks must not contain duplicates."
        assert "lm35_p4_shared" in chunk_ids
        assert "lm35_p10_sparse" in chunk_ids


# ==============================================================================
# 8. Missing ChromaDB Collection Handling
# ==============================================================================
def test_missing_chromadb_collection(tmp_path: Path) -> None:
    """Verify that querying a non-existent ChromaDB collection raises ValueError with available collections."""
    db_dir = tmp_path / "test_chroma_db"
    db_dir.mkdir()
    client = chromadb.PersistentClient(path=str(db_dir))
    # Create an unrelated collection
    client.create_collection("existing_other_collection")

    retriever = DatasheetRetriever(
        persist_directory=str(db_dir),
        collection_name="missing_datasheet_collection",
    )

    with pytest.raises(ValueError) as exc_info:
        retriever.get_vector_store()

    error_msg = str(exc_info.value)
    assert "missing_datasheet_collection" in error_msg
    assert "existing_other_collection" in error_msg


# ==============================================================================
# 9. Retrieval Evaluation Metric Calculations
# ==============================================================================
def test_retrieval_evaluation_metric_calculations() -> None:
    """Verify the mathematical correctness of Precision@K, Recall@K, and MRR calculations."""
    mock_retriever = MagicMock()
    # Mock retrieved top-5 chunks: 3 are relevant (ranks 1, 2, 4), 2 are irrelevant (ranks 3, 5)
    mock_retriever.retrieve_mmr.return_value = [
        Document(page_content="Relevant chunk 1 with keyword", metadata={"source": "lm35.pdf", "page": 1}),
        Document(page_content="Relevant chunk 2 with keyword", metadata={"source": "lm35.pdf", "page": 13}),
        Document(page_content="Irrelevant background fluff", metadata={"source": "lm35.pdf", "page": 30}),
        Document(page_content="Relevant chunk 4 with keyword", metadata={"source": "lm35.pdf", "page": 1}),
        Document(page_content="Irrelevant ordering note", metadata={"source": "lm35.pdf", "page": 35}),
    ]

    matcher = RelevanceMatcher(mode="page_only")
    query_item = {
        "id": "TEST_Q1",
        "question": "What is the scale factor?",
        "expected_pages": [1, 13, 15],  # 3 expected pages total; pages 1 and 13 are retrieved
        "expected_keywords": ["scale factor"],
    }

    result = evaluate_query(
        query_item=query_item,
        retriever=mock_retriever,
        matcher=matcher,
        strategy="mmr",
        k=5,
    )

    # 1. Precision@5: 3 relevant chunks out of 5 = 3 / 5 = 0.60
    assert result["precision_at_k"] == 0.60

    # 2. Recall@5: 2 expected pages covered ({1, 13}) out of 3 expected pages = 2 / 3 = 0.6667
    assert result["recall_at_k"] == 0.6667

    # 3. Reciprocal Rank (RR): first relevant chunk appears at Rank 1 => 1.0 / 1 = 1.0
    assert result["reciprocal_rank"] == 1.0
    assert result["first_relevant_rank"] == 1
    assert result["hit"] is True

    # Test boundary case: 0 relevant chunks retrieved
    mock_retriever.retrieve_mmr.return_value = [
        Document(page_content="Irrelevant", metadata={"source": "lm35.pdf", "page": 99}),
    ]
    zero_result = evaluate_query(
        query_item=query_item,
        retriever=mock_retriever,
        matcher=matcher,
        strategy="mmr",
        k=1,
    )
    assert zero_result["precision_at_k"] == 0.0
    assert zero_result["recall_at_k"] == 0.0
    assert zero_result["reciprocal_rank"] == 0.0
    assert zero_result["first_relevant_rank"] is None
    assert zero_result["hit"] is False


# ==============================================================================
# 10. Hybrid Retrieval Result Merging (Reciprocal Rank Fusion)
# ==============================================================================
def test_hybrid_retrieval_result_merging() -> None:
    """Verify Reciprocal Rank Fusion (RRF) mathematical score merging, ranking, and weights."""
    retriever = DatasheetRetriever()

    # Doc A: Rank 1 in Dense, Rank 2 in Sparse
    doc_a = Document(page_content="Doc A content", metadata={"source": "test.pdf", "page": 1, "chunk_id": "doc_a"})
    # Doc B: Rank 2 in Dense, not in Sparse
    doc_b = Document(page_content="Doc B content", metadata={"source": "test.pdf", "page": 2, "chunk_id": "doc_b"})
    # Doc C: Rank 1 in Sparse, not in Dense
    doc_c = Document(page_content="Doc C content", metadata={"source": "test.pdf", "page": 3, "chunk_id": "doc_c"})

    # Dense candidate list: [Doc A, Doc B]
    # Sparse candidate list: [Doc C, Doc A]
    with patch.object(retriever, "retrieve_semantic", return_value=[doc_a, doc_b]), \
         patch.object(retriever, "retrieve_keyword", return_value=[doc_c, doc_a]):

        # Parameters: dense_weight=0.6, sparse_weight=0.4, rrf_c=60
        fused = retriever.retrieve_hybrid(
            query="test merging",
            k=3,
            dense_weight=0.6,
            sparse_weight=0.4,
            rrf_c=60,
        )

        assert len(fused) == 3

        # Theoretical RRF calculation:
        # Score(Doc A) = 0.6 / (60 + 1) + 0.4 / (60 + 2) = 0.6 / 61 + 0.4 / 62 = 0.009836 + 0.006452 = 0.016288
        # Score(Doc C) = 0.4 / (60 + 1) = 0.4 / 61 = 0.006557
        # Score(Doc B) = 0.6 / (60 + 2) = 0.6 / 62 = 0.009677
        #
        # Ordering must be: Doc A (0.016288) > Doc B (0.009677) > Doc C (0.006557)
        assert fused[0].metadata["chunk_id"] == "doc_a"
        assert fused[1].metadata["chunk_id"] == "doc_b"
        assert fused[2].metadata["chunk_id"] == "doc_c"

        assert fused[0].metadata["hybrid_rank"] == 1
        assert fused[1].metadata["hybrid_rank"] == 2
        assert fused[2].metadata["hybrid_rank"] == 3

        # Check precision of computed RRF scores
        assert pytest.approx(fused[0].metadata["rrf_score"], abs=1e-4) == 0.016288
        assert pytest.approx(fused[1].metadata["rrf_score"], abs=1e-4) == 0.009677
        assert pytest.approx(fused[2].metadata["rrf_score"], abs=1e-4) == 0.006557
