"""Verification script for CircuitMind AI Retrieval Engines (Hybrid RRF, MMR, Semantic, BM25).

Queries the persistent ChromaDB collection and displays retrieved chunks with their rank,
source datasheet, page number, RRF score, and content.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to sys.path so script can be invoked from any directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from langchain_chroma import Chroma

from app.config import get_settings
from app.rag.retrieval import DatasheetRetriever

# Default configurations aligned with app settings and ingestion pipeline
settings = get_settings()
DB_PATH = settings.chroma_persist_directory
COLLECTION_NAME = settings.chroma_collection_name
MODEL_NAME = settings.embedding_model_name

# Configurable defaults
DEFAULT_K = 5
DEFAULT_FETCH_K = 15
DEFAULT_LAMBDA_MULT = 0.7
DEFAULT_DENSE_WEIGHT = 0.5
DEFAULT_SPARSE_WEIGHT = 0.5
DEFAULT_RRF_C = 60

_retriever_instance: Optional[DatasheetRetriever] = None


def get_retriever() -> DatasheetRetriever:
    """Return a singleton instance of DatasheetRetriever."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = DatasheetRetriever(
            persist_directory=DB_PATH,
            collection_name=COLLECTION_NAME,
            embedding_model_name=MODEL_NAME,
        )
    return _retriever_instance


def get_vector_store() -> Chroma:
    """Connect to and return the Chroma vector store.

    Returns:
        Configured Chroma vector store instance.

    Raises:
        ConnectionError: If database connection fails.
        ValueError: If collection does not exist or has no documents.
    """
    retriever = get_retriever()
    return retriever.get_vector_store()


def search_datasheet(
    query: str,
    k: int = DEFAULT_K,
    strategy: str = "hybrid",
    fetch_k: int = DEFAULT_FETCH_K,
    lambda_mult: float = DEFAULT_LAMBDA_MULT,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
    sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
    rrf_c: int = DEFAULT_RRF_C,
) -> List[Document]:
    """Execute search for a technical query using the selected strategy.

    Args:
        query: User engineering query.
        k: Number of diverse/top chunks to return (default: 5).
        strategy: Retrieval mode: 'hybrid' (RRF), 'mmr', 'semantic', or 'keyword'.
        fetch_k: Number of initial candidates to evaluate for MMR (default: 15).
        lambda_mult: MMR diversity factor (default: 0.7).
        dense_weight: Dense weight for hybrid search (default: 0.5).
        sparse_weight: Sparse weight for hybrid search (default: 0.5).
        rrf_c: RRF smoothing constant (default: 60).

    Returns:
        List of LangChain Document chunks matching query with source citations.

    Raises:
        ValueError: If query is empty or strategy is unknown.
        ConnectionError: If ChromaDB persistent directory cannot be found or read.
    """
    if not query or not query.strip():
        raise ValueError("Query string cannot be empty or contain only whitespace.")

    retriever = get_retriever()
    strategy_clean = strategy.lower().strip()

    if strategy_clean == "hybrid":
        return retriever.retrieve_hybrid(
            query=query.strip(),
            k=k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            rrf_c=rrf_c,
        )
    elif strategy_clean == "mmr":
        return retriever.retrieve_mmr(
            query=query.strip(),
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )
    elif strategy_clean == "semantic":
        return retriever.retrieve_semantic(
            query=query.strip(),
            k=k,
        )
    elif strategy_clean in ("keyword", "bm25"):
        return retriever.retrieve_keyword(
            query=query.strip(),
            k=k,
        )
    else:
        raise ValueError(
            f"Unknown retrieval strategy '{strategy}'. Supported: 'hybrid', 'mmr', 'semantic', 'keyword'."
        )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CircuitMind AI - Datasheet Retrieval Verification CLI (Hybrid RRF / MMR / Semantic / BM25)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        default=None,
        help="Engineering query string (if omitted, prompts interactively).",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        type=str,
        default="hybrid",
        choices=["hybrid", "mmr", "semantic", "keyword"],
        help="Retrieval strategy to evaluate.",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=DEFAULT_K,
        help="Number of chunks to return.",
    )
    parser.add_argument(
        "--fetch-k",
        type=int,
        default=DEFAULT_FETCH_K,
        help="Number of initial candidate chunks for MMR.",
    )
    parser.add_argument(
        "--lambda-mult",
        type=float,
        default=DEFAULT_LAMBDA_MULT,
        help="MMR diversity trade-off (0.0 = maximal diversity, 1.0 = pure similarity).",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=DEFAULT_DENSE_WEIGHT,
        help="Hybrid RRF weight for semantic vector search.",
    )
    parser.add_argument(
        "--sparse-weight",
        type=float,
        default=DEFAULT_SPARSE_WEIGHT,
        help="Hybrid RRF weight for BM25 keyword search.",
    )
    parser.add_argument(
        "--rrf-c",
        type=int,
        default=DEFAULT_RRF_C,
        help="RRF smoothing constant.",
    )
    return parser.parse_args()


def display_results(query: str, results: List[Document], strategy: str) -> None:
    """Format and print retrieved chunks with rank, source, page, RRF score, and content.

    Args:
        query: The executed query string.
        results: Retrieved LangChain Document objects.
        strategy: Name of the strategy used.
    """
    print("\n" + "=" * 70)
    print(f"CIRCUITMIND AI - RETRIEVAL RESULTS ({strategy.upper()} STRATEGY)")
    print("=" * 70)
    print(f"Query: \"{query}\"")
    print(f"Total Chunks Retrieved: {len(results)}")
    print("=" * 70)

    for index, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", "Unknown Page")
        part_number = doc.metadata.get("part_number")
        rrf_score = doc.metadata.get("rrf_score")

        meta_parts = [f"Source: {source}", f"Page: {page}"]
        if part_number:
            meta_parts.append(f"Part: {part_number}")
        if rrf_score is not None:
            meta_parts.append(f"RRF Score: {rrf_score:.6f}")

        print(f"\n[Rank {index}] {' | '.join(meta_parts)}")
        print("-" * 70)
        print(doc.page_content.strip())
        print("-" * 70)


def main() -> int:
    """Main CLI execution routine."""
    args = parse_args()

    # Query acquisition: CLI argument or interactive prompt
    query = args.query
    if not query:
        try:
            query = input(f"\nAsk CircuitMind [{args.strategy.upper()}]: ")
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            return 0

    # Input validation
    if not query or not query.strip():
        print("\n[Error] Query cannot be empty or only whitespace. Please provide a valid engineering question.")
        return 1

    try:
        results = search_datasheet(
            query=query,
            k=args.k,
            strategy=args.strategy,
            fetch_k=args.fetch_k,
            lambda_mult=args.lambda_mult,
            dense_weight=args.dense_weight,
            sparse_weight=args.sparse_weight,
            rrf_c=args.rrf_c,
        )
    except ConnectionError as err:
        print(f"\n[Database Connection Error] {err}")
        return 1
    except ValueError as err:
        print(f"\n[Validation Error] {err}")
        return 1
    except Exception as err:
        print(f"\n[Retrieval Failure] An unexpected error occurred: {err}")
        return 1

    if not results:
        print("\nNo relevant datasheet passages found for your query.")
        return 0

    display_results(query.strip(), results, strategy=args.strategy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
