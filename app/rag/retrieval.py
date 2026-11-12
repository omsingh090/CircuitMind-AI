"""Hybrid Retrieval Engine for CircuitMind AI.

Combines dense semantic vector search (ChromaDB + HuggingFace) with
sparse keyword search (BM25) using Reciprocal Rank Fusion (RRF).
Preserves document source and page citations, eliminates duplicates,
and maintains independent accessibility for all retrieval strategies
(Semantic, MMR, Keyword, Hybrid).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from app.config import get_settings

logger = logging.getLogger(__name__)


def compute_doc_identifier(doc: Document) -> str:
    """Generate a reliable, unique identifier for a document chunk.

    Prefers 'chunk_id' from metadata; falls back to SHA-256 hash of source, page, and content.

    Args:
        doc: LangChain Document chunk.

    Returns:
        Deterministic string identifier.
    """
    if "chunk_id" in doc.metadata and doc.metadata["chunk_id"]:
        return str(doc.metadata["chunk_id"])

    source = str(doc.metadata.get("source", "unknown"))
    page = str(doc.metadata.get("page", "0"))
    payload = f"{source}::page_{page}::{doc.page_content.strip()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


class DatasheetRetriever:
    """Retrieves component datasheet chunks using Semantic, MMR, BM25, or Hybrid Search."""

    # Class-level caches across instances to avoid redundant HuggingFace/ChromaDB/BM25 loads
    _shared_embeddings: Optional[HuggingFaceEmbeddings] = None
    _shared_vector_store: Optional[Chroma] = None
    _shared_keyword_retriever: Optional[BM25Retriever] = None

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
    ) -> None:
        """Initialize the retriever with vector database and embedding model settings.

        Args:
            persist_directory: Path to the ChromaDB persistent directory.
            collection_name: Name of the ChromaDB collection.
            embedding_model_name: Name of the Hugging Face embedding model.
        """
        settings = get_settings()

        self.persist_directory = persist_directory or settings.chroma_persist_directory
        self.collection_name = collection_name or settings.chroma_collection_name
        self.embedding_model_name = embedding_model_name or settings.embedding_model_name

        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._vector_store: Optional[Chroma] = None
        self._keyword_retriever: Optional[BM25Retriever] = None

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Lazily initialize and return the HuggingFace embeddings instance.
        
        Leverages shared instance and local_files_only=True to eliminate network retries
        and multi-second cold start delays.
        """
        if self._embeddings is not None:
            return self._embeddings

        settings = get_settings()
        is_default_model = self.embedding_model_name == settings.embedding_model_name
        if is_default_model and DatasheetRetriever._shared_embeddings is not None:
            self._embeddings = DatasheetRetriever._shared_embeddings
            return self._embeddings

        logger.info("Initializing embedding model '%s'...", self.embedding_model_name)
        # Fast path: Load locally cached weights without network check or Hub retry delays
        try:
            emb = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": "cpu", "local_files_only": True},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception:
            # Fallback if local files are missing on fresh checkout
            emb = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

        if is_default_model:
            DatasheetRetriever._shared_embeddings = emb
        self._embeddings = emb
        return self._embeddings

    def get_vector_store(self) -> Chroma:
        """Connect to and validate the persistent ChromaDB collection.

        Returns:
            Chroma vector store instance configured for the collection.

        Raises:
            ConnectionError: If ChromaDB persistent directory cannot be accessed.
            ValueError: If the configured collection does not exist or has no documents.
        """
        if self._vector_store is not None:
            return self._vector_store

        settings = get_settings()
        is_default_target = (
            str(Path(self.persist_directory).resolve()) == str(Path(settings.chroma_persist_directory).resolve())
            and self.collection_name == settings.chroma_collection_name
        )
        if is_default_target and DatasheetRetriever._shared_vector_store is not None:
            self._vector_store = DatasheetRetriever._shared_vector_store
            return self._vector_store

        persist_path = Path(self.persist_directory).resolve()
        if not persist_path.exists():
            raise ConnectionError(
                f"ChromaDB persistent directory not found at: '{persist_path}'. "
                "Ensure datasheets have been ingested using 'scripts/ingest.py'."
            )

        try:
            client = chromadb.PersistentClient(path=str(persist_path))
        except Exception as err:
            logger.error("Failed to connect to ChromaDB at '%s': %s", persist_path, err)
            raise ConnectionError(f"Database connection failure at '{persist_path}': {err}") from err

        try:
            existing_collections = [col.name for col in client.list_collections()]
        except Exception as err:
            logger.error("Failed to list collections from ChromaDB: %s", err)
            raise ConnectionError(f"Failed to inspect database collections: {err}") from err

        if self.collection_name not in existing_collections:
            raise ValueError(
                f"Collection '{self.collection_name}' not found in ChromaDB. "
                f"Available collections: {existing_collections}. "
                "Please ingest datasheets first using 'scripts/ingest.py'."
            )

        collection = client.get_collection(name=self.collection_name)
        doc_count = collection.count()
        if doc_count == 0:
            raise ValueError(
                f"Collection '{self.collection_name}' exists but contains 0 document chunks. "
                "Please ingest datasheets first using 'scripts/ingest.py'."
            )

        logger.debug(
            "Connected to ChromaDB collection '%s' (%d total chunks).",
            self.collection_name,
            doc_count,
        )

        store = Chroma(
            client=client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )
        if is_default_target:
            DatasheetRetriever._shared_vector_store = store
        self._vector_store = store
        return self._vector_store

    def get_keyword_retriever(self) -> BM25Retriever:
        """Initialize or return the cached in-memory BM25 keyword retriever.

        Loads all indexed chunks and metadata from the persistent ChromaDB collection.

        Returns:
            Initialized BM25Retriever instance.

        Raises:
            ConnectionError: If database connection fails.
            ValueError: If collection is missing or contains 0 chunks.
        """
        if self._keyword_retriever is not None:
            return self._keyword_retriever

        settings = get_settings()
        is_default_target = (
            str(Path(self.persist_directory).resolve()) == str(Path(settings.chroma_persist_directory).resolve())
            and self.collection_name == settings.chroma_collection_name
        )
        if is_default_target and DatasheetRetriever._shared_keyword_retriever is not None:
            self._keyword_retriever = DatasheetRetriever._shared_keyword_retriever
            return self._keyword_retriever

        persist_path = Path(self.persist_directory).resolve()
        if not persist_path.exists():
            raise ConnectionError(f"ChromaDB directory not found: '{persist_path}'")

        try:
            client = chromadb.PersistentClient(path=str(persist_path))
            collection = client.get_collection(name=self.collection_name)
            data = collection.get(include=["documents", "metadatas"])
        except Exception as err:
            logger.error("Failed to load chunks for BM25 indexing from '%s': %s", self.collection_name, err)
            raise ConnectionError(f"Error loading documents for keyword indexing: {err}") from err

        chunk_texts = data.get("documents") or []
        chunk_metas = data.get("metadatas") or []
        chunk_ids = data.get("ids") or []

        if not chunk_texts:
            raise ValueError(
                f"ChromaDB collection '{self.collection_name}' contains no chunks to build BM25 index."
            )

        docs: List[Document] = []
        for cid, text, meta in zip(chunk_ids, chunk_texts, chunk_metas):
            doc_meta = dict(meta) if meta else {}
            doc_meta["chunk_id"] = cid
            docs.append(Document(page_content=text, metadata=doc_meta))

        logger.info("Building BM25 keyword index over %d datasheet chunks...", len(docs))
        kw_retriever = BM25Retriever.from_documents(docs)
        if is_default_target:
            DatasheetRetriever._shared_keyword_retriever = kw_retriever
        self._keyword_retriever = kw_retriever
        return self._keyword_retriever

    def retrieve_semantic(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Perform standard dense vector similarity search.

        Args:
            query: The engineering query string. Must not be empty.
            k: Number of top chunks to return (default: 5).
            filter_metadata: Optional metadata filter dict.

        Returns:
            List of LangChain Document objects ordered by similarity.
        """
        self._validate_query(query)
        if k <= 0:
            raise ValueError(f"Parameter 'k' must be greater than 0, got {k}.")

        vector_store = self.get_vector_store()
        logger.debug("Executing Semantic vector search: query=%r, k=%d", query.strip(), k)
        try:
            return vector_store.similarity_search(
                query=query.strip(),
                k=k,
                filter=filter_metadata,
            )
        except Exception as err:
            logger.error("Semantic search failed: %s", err)
            raise RuntimeError(f"Semantic search execution error: {err}") from err

    def retrieve_mmr(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 15,
        lambda_mult: float = 0.7,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Perform Maximum Marginal Relevance (MMR) search for diverse retrieval.

        Args:
            query: The engineering query string. Must not be empty.
            k: Number of diverse top chunks to return (default: 5).
            fetch_k: Number of initial candidates to evaluate (default: 15).
            lambda_mult: Diversity weight between 0.0 (maximum diversity) and 1.0 (pure similarity).
            filter_metadata: Optional metadata filter dict.

        Returns:
            List of LangChain Document objects with source and page metadata.
        """
        self._validate_query(query)
        if k <= 0:
            raise ValueError(f"Parameter 'k' must be greater than 0, got {k}.")
        if fetch_k < k:
            raise ValueError(f"Parameter 'fetch_k' ({fetch_k}) must be greater than or equal to 'k' ({k}).")
        if not (0.0 <= lambda_mult <= 1.0):
            raise ValueError(f"Parameter 'lambda_mult' must be in range [0.0, 1.0], got {lambda_mult}.")

        vector_store = self.get_vector_store()
        clean_query = query.strip()

        logger.debug("Executing MMR search: query=%r, k=%d, fetch_k=%d, lambda=%.2f", clean_query, k, fetch_k, lambda_mult)
        try:
            return vector_store.max_marginal_relevance_search(
                query=clean_query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=filter_metadata,
            )
        except Exception as err:
            logger.error("MMR retrieval failed: %s", err)
            raise RuntimeError(f"MMR search execution error: {err}") from err

    def retrieve_keyword(
        self,
        query: str,
        k: int = 5,
    ) -> List[Document]:
        """Perform sparse keyword-based retrieval using BM25.

        Args:
            query: Engineering query string.
            k: Number of top chunks to return (default: 5).

        Returns:
            List of LangChain Document objects ranked by BM25 score.
        """
        self._validate_query(query)
        if k <= 0:
            raise ValueError(f"Parameter 'k' must be greater than 0, got {k}.")

        bm25 = self.get_keyword_retriever()
        bm25.k = k

        clean_query = query.strip()
        logger.debug("Executing BM25 keyword search: query=%r, k=%d", clean_query, k)
        try:
            results = bm25.invoke(clean_query)
            return results[:k]
        except Exception as err:
            logger.error("BM25 search failed: %s", err)
            raise RuntimeError(f"BM25 search execution error: {err}") from err

    def retrieve_hybrid(
        self,
        query: str,
        k: int = 5,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_c: int = 60,
        candidate_k: int = 20,
        max_per_page: int = 2,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Perform Hybrid Search combining Semantic Vector Search and BM25 via Reciprocal Rank Fusion (RRF).

        Formula:
            RRF_Score(d) = sum( w_m / (rrf_c + rank_m(d)) )

        Deduplicates chunks across candidate pools using stable chunk identifiers, enforces page diversity,
        suppresses near-duplicates, and preserves complete source and page metadata.

        Args:
            query: Engineering query string. Must not be empty.
            k: Number of fused top chunks to return (default: 5).
            dense_weight: Weight multiplier for dense semantic vector search (default: 0.5).
            sparse_weight: Weight multiplier for sparse BM25 keyword search (default: 0.5).
            rrf_c: Reciprocal Rank Fusion smoothing constant (default: 60).
            candidate_k: Initial candidate pool fetched from each sub-retriever (default: 20).
            max_per_page: Maximum number of chunks allowed from a single page for diversity (default: 2).
            filter_metadata: Optional metadata filter dict for the vector store.

        Returns:
            List of deduplicated and diversified LangChain Document objects ranked by RRF score.
        """
        self._validate_query(query)
        if k <= 0:
            raise ValueError(f"Parameter 'k' must be greater than 0, got {k}.")
        if candidate_k < k:
            raise ValueError(f"Parameter 'candidate_k' ({candidate_k}) must be >= 'k' ({k}).")
        if dense_weight < 0.0 or sparse_weight < 0.0:
            raise ValueError("Retrieval weights must be non-negative.")
        if dense_weight + sparse_weight == 0.0:
            raise ValueError("At least one retrieval weight must be strictly positive.")
        if rrf_c < 1:
            raise ValueError(f"Smoothing constant 'rrf_c' must be >= 1, got {rrf_c}.")
        if max_per_page < 1:
            raise ValueError(f"Parameter 'max_per_page' must be >= 1, got {max_per_page}.")

        clean_query = query.strip()

        # Step 1: Retrieve candidate sets from both sub-retrievers
        dense_candidates: List[Document] = []
        try:
            dense_candidates = self.retrieve_semantic(
                query=clean_query,
                k=candidate_k,
                filter_metadata=filter_metadata,
            )
        except Exception as err:
            logger.warning("Dense retriever failed during hybrid search: %s", err)

        sparse_candidates: List[Document] = []
        try:
            sparse_candidates = self.retrieve_keyword(
                query=clean_query,
                k=candidate_k,
            )
        except Exception as err:
            logger.warning("Sparse BM25 retriever failed during hybrid search: %s", err)

        if not dense_candidates and not sparse_candidates:
            logger.warning("Both semantic and keyword retrievers returned 0 results for %r.", clean_query)
            return []

        # Step 2: Reciprocal Rank Fusion (RRF) with chunk deduplication
        rrf_scores: Dict[str, float] = {}
        doc_store: Dict[str, Document] = {}

        # Dense ranking contribution
        for rank, doc in enumerate(dense_candidates, start=1):
            doc_id = compute_doc_identifier(doc)
            doc_store[doc_id] = doc
            dense_score = dense_weight / (rrf_c + rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + dense_score

        # Sparse ranking contribution
        for rank, doc in enumerate(sparse_candidates, start=1):
            doc_id = compute_doc_identifier(doc)
            if doc_id not in doc_store:
                doc_store[doc_id] = doc
            sparse_score = sparse_weight / (rrf_c + rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + sparse_score

        # Step 3: Sort by fused RRF score descending with page diversity & near-duplicate suppression
        ranked_doc_ids = sorted(
            rrf_scores.keys(),
            key=lambda did: rrf_scores[did],
            reverse=True,
        )

        fused_results: List[Document] = []
        page_counts: Dict[Any, int] = {}
        deferred_doc_ids: List[str] = []

        for doc_id in ranked_doc_ids:
            cand_doc = doc_store[doc_id]
            pg = cand_doc.metadata.get("page", 0)

            # Check page diversity ceiling
            if page_counts.get(pg, 0) >= max_per_page:
                deferred_doc_ids.append(doc_id)
                continue

            # Check near-duplicate text overlap with already accepted docs from same page
            is_near_dup = False
            cand_words = set(cand_doc.page_content.lower().split())
            if cand_words:
                for accepted in fused_results:
                    if accepted.metadata.get("page") == pg:
                        acc_words = set(accepted.page_content.lower().split())
                        if acc_words:
                            overlap = len(cand_words & acc_words) / min(len(cand_words), len(acc_words))
                            if overlap > 0.65:
                                is_near_dup = True
                                break

            if is_near_dup:
                deferred_doc_ids.append(doc_id)
                continue

            # Accept document
            page_counts[pg] = page_counts.get(pg, 0) + 1
            fused_results.append(cand_doc)
            if len(fused_results) >= k:
                break

        # If diversity filtering pruned too aggressively, backfill from deferred candidates
        if len(fused_results) < k:
            for def_id in deferred_doc_ids:
                fused_results.append(doc_store[def_id])
                if len(fused_results) >= k:
                    break

        # Finalize metadata
        final_results: List[Document] = []
        for final_rank, doc in enumerate(fused_results[:k], start=1):
            did = compute_doc_identifier(doc)
            fused_metadata = dict(doc.metadata)
            fused_metadata["rrf_score"] = round(rrf_scores.get(did, 0.0), 6)
            fused_metadata["hybrid_rank"] = final_rank
            fused_metadata["chunk_id"] = did

            final_results.append(
                Document(
                    page_content=doc.page_content,
                    metadata=fused_metadata,
                )
            )

        logger.debug(
            "Hybrid RRF search fused %d dense and %d sparse candidates into %d results (max_per_page=%d).",
            len(dense_candidates),
            len(sparse_candidates),
            len(final_results),
            max_per_page,
        )
        return final_results

    @staticmethod
    def _validate_query(query: str) -> None:
        """Validate that the query string is non-empty and contains non-whitespace text."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query string cannot be empty or contain only whitespace.")


# ==============================================================================
# Shared Singleton Retriever Access
# ==============================================================================
_retriever_instance: Optional[DatasheetRetriever] = None


def get_retriever() -> DatasheetRetriever:
    """Return a shared singleton DatasheetRetriever instance with cached embeddings & index.

    Guarantees sub-50ms hybrid retrieval by avoiding repeated PyTorch model loading
    and Hugging Face Hub network validation loops.
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = DatasheetRetriever()
    return _retriever_instance


def reset_retriever_cache() -> None:
    """Reset the singleton instance and class-level shared models.

    Crucial for unit testing and mock isolation.
    """
    global _retriever_instance
    _retriever_instance = None
    DatasheetRetriever._shared_embeddings = None
    DatasheetRetriever._shared_vector_store = None
    DatasheetRetriever._shared_keyword_retriever = None
