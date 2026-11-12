"""Retrieval Evaluation Framework for CircuitMind AI.

Benchmarking tool that evaluates and compares retrieval strategies
(Semantic Vector Search, Maximum Marginal Relevance, BM25 Keyword Search,
and Hybrid Reciprocal Rank Fusion) against verified ground-truth datasheet benchmarks.
Computes Precision@K, Recall@K, and Mean Reciprocal Rank (MRR).
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add repository root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from app.rag.retrieval import DatasheetRetriever

logger = logging.getLogger(__name__)


class RelevanceMatcher:
    """Configurable evaluator determining whether a retrieved chunk is relevant to a query."""

    def __init__(
        self,
        mode: str = "page_or_keyword",
        min_keywords: int = 1,
    ) -> None:
        """Initialize relevance matcher.

        Args:
            mode: Matching logic: 'page_or_keyword', 'page_and_keyword',
                  'page_only', or 'keyword_only'.
            min_keywords: Minimum number of keyword occurrences required.
        """
        valid_modes = {"page_or_keyword", "page_and_keyword", "page_only", "keyword_only"}
        if mode not in valid_modes:
            raise ValueError(f"Invalid matching mode '{mode}'. Choose from: {valid_modes}")

        self.mode = mode
        self.min_keywords = max(1, min_keywords)

    def is_relevant(
        self,
        doc: Document,
        expected_pages: List[int],
        expected_keywords: List[str],
    ) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate if a single document chunk meets the ground-truth relevance criteria.

        Args:
            doc: Retrieved LangChain Document chunk.
            expected_pages: Verified page numbers containing the target information.
            expected_keywords: Specific technical terms expected in the chunk text.

        Returns:
            Tuple of (is_relevant: bool, debug_details: dict).
        """
        doc_page = doc.metadata.get("page")
        page_matched = doc_page in expected_pages if doc_page is not None else False

        content_lower = doc.page_content.lower()
        matched_keywords = [
            kw for kw in expected_keywords if kw.lower() in content_lower
        ]
        keyword_matched = len(matched_keywords) >= self.min_keywords

        if self.mode == "page_and_keyword":
            relevant = page_matched and keyword_matched
        elif self.mode == "page_or_keyword":
            relevant = page_matched or keyword_matched
        elif self.mode == "page_only":
            relevant = page_matched
        elif self.mode == "keyword_only":
            relevant = keyword_matched
        else:
            relevant = False

        details = {
            "page": doc_page,
            "page_matched": page_matched,
            "matched_keywords": matched_keywords,
            "keyword_matched": keyword_matched,
            "mode": self.mode,
        }
        return relevant, details


def evaluate_query(
    query_item: Dict[str, Any],
    retriever: DatasheetRetriever,
    matcher: RelevanceMatcher,
    strategy: str = "hybrid",
    k: int = 5,
    fetch_k: int = 15,
    lambda_mult: float = 0.7,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    rrf_c: int = 60,
) -> Dict[str, Any]:
    """Execute retrieval and compute ranking and relevance metrics for a single question.

    Args:
        query_item: Dictionary containing 'id', 'question', 'expected_pages', 'expected_keywords'.
        retriever: Configured DatasheetRetriever instance.
        matcher: Configured RelevanceMatcher instance.
        strategy: 'hybrid', 'mmr', 'semantic', or 'keyword'.
        k: Top-k evaluation threshold (default: 5).
        fetch_k: Initial candidates pool for MMR.
        lambda_mult: MMR diversity factor.
        dense_weight: Dense weight for hybrid search.
        sparse_weight: Sparse weight for hybrid search.
        rrf_c: RRF smoothing constant.

    Returns:
        Evaluation record dictionary with metrics and per-rank details.
    """
    qid = query_item.get("id", "UNKNOWN")
    question = query_item["question"]
    expected_pages = query_item.get("expected_pages", [])
    expected_keywords = query_item.get("expected_keywords", [])

    # Route query to the selected strategy
    strat = strategy.lower().strip()
    if strat == "hybrid":
        retrieved_docs = retriever.retrieve_hybrid(
            query=question,
            k=k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            rrf_c=rrf_c,
        )
    elif strat == "mmr":
        retrieved_docs = retriever.retrieve_mmr(
            query=question,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )
    elif strat == "semantic":
        retrieved_docs = retriever.retrieve_semantic(
            query=question,
            k=k,
        )
    elif strat in ("keyword", "bm25"):
        retrieved_docs = retriever.retrieve_keyword(
            query=question,
            k=k,
        )
    else:
        raise ValueError(f"Unknown strategy: '{strategy}'")

    relevant_count = 0
    first_relevant_rank: Optional[int] = None
    retrieved_ranks: List[Dict[str, Any]] = []
    retrieved_pages_set = set()

    for idx, doc in enumerate(retrieved_docs, start=1):
        is_rel, details = matcher.is_relevant(doc, expected_pages, expected_keywords)
        doc_page = doc.metadata.get("page")

        if is_rel:
            relevant_count += 1
            if first_relevant_rank is None:
                first_relevant_rank = idx
            if doc_page is not None:
                retrieved_pages_set.add(doc_page)

        retrieved_ranks.append(
            {
                "rank": idx,
                "source": doc.metadata.get("source"),
                "page": doc_page,
                "is_relevant": is_rel,
                "details": details,
                "snippet": doc.page_content[:140].replace("\n", " "),
            }
        )

    # Calculate Precision@K
    precision_at_k = relevant_count / k if k > 0 else 0.0

    # Calculate Recall@K (Coverage of expected pages retrieved)
    total_expected = len(expected_pages)
    if total_expected > 0:
        covered_pages = len(retrieved_pages_set.intersection(expected_pages))
        recall_at_k = covered_pages / total_expected
    else:
        recall_at_k = 1.0 if relevant_count > 0 else 0.0

    # Calculate Reciprocal Rank (RR)
    reciprocal_rank = (1.0 / first_relevant_rank) if first_relevant_rank is not None else 0.0

    return {
        "id": qid,
        "question": question,
        "category": query_item.get("category", "general"),
        "strategy": strat,
        "expected_pages": expected_pages,
        "expected_keywords": expected_keywords,
        "retrieved_count": len(retrieved_docs),
        "relevant_count": relevant_count,
        "first_relevant_rank": first_relevant_rank,
        "precision_at_k": round(precision_at_k, 4),
        "recall_at_k": round(recall_at_k, 4),
        "reciprocal_rank": round(reciprocal_rank, 4),
        "hit": first_relevant_rank is not None,
        "retrieved_ranks": retrieved_ranks,
    }


def run_benchmark(
    dataset_path: Path,
    output_report_path: Optional[Path] = None,
    strategy: str = "hybrid",
    k: int = 5,
    fetch_k: int = 15,
    lambda_mult: float = 0.7,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    rrf_c: int = 60,
    match_mode: str = "page_or_keyword",
    min_keywords: int = 1,
    verbose: bool = False,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Run retrieval evaluation benchmark for a specified strategy.

    Args:
        dataset_path: JSON file containing benchmark questions.
        output_report_path: Optional destination JSON file for benchmark results.
        strategy: 'hybrid', 'mmr', 'semantic', or 'keyword'.
        k: Top-K evaluation cut-off.
        fetch_k: Initial MMR candidate pool.
        lambda_mult: MMR diversity factor.
        dense_weight: Dense weight for hybrid search.
        sparse_weight: Sparse weight for hybrid search.
        rrf_c: RRF smoothing constant.
        match_mode: Relevance matching logic mode.
        min_keywords: Minimum technical keywords required for a match.
        verbose: If True, prints chunk previews.
        quiet: If True, suppresses console prints.

    Returns:
        Complete evaluation report dictionary.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        questions: List[Dict[str, Any]] = json.load(f)

    if not questions:
        raise ValueError("Evaluation dataset is empty.")

    retriever = DatasheetRetriever()
    matcher = RelevanceMatcher(mode=match_mode, min_keywords=min_keywords)

    if not quiet:
        print("\n" + "=" * 80)
        print(f"CIRCUITMIND AI - RETRIEVAL EVALUATION ({strategy.upper()} STRATEGY)")
        print("=" * 80)
        print(f"Dataset:            {dataset_path.name} ({len(questions)} questions)")
        print(f"Strategy:           {strategy.upper()}")
        print(f"Settings:           k={k}, fetch_k={fetch_k}, lambda={lambda_mult}, weights=({dense_weight}, {sparse_weight})")
        print(f"Relevance Matching: mode='{match_mode}', min_keywords={min_keywords}")
        print("=" * 80)

    per_query_results: List[Dict[str, Any]] = []

    for item in questions:
        res = evaluate_query(
            query_item=item,
            retriever=retriever,
            matcher=matcher,
            strategy=strategy,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            rrf_c=rrf_c,
        )
        per_query_results.append(res)

        if not quiet:
            status_tag = "[HIT] " if res["hit"] else "[MISS]"
            rr_str = f"RR: {res['reciprocal_rank']:.2f}"
            first_str = f"First: Rank {res['first_relevant_rank']}" if res["first_relevant_rank"] else "First: None"
            p_str = f"P@{k}: {res['precision_at_k']:.2f}"
            r_str = f"R@{k}: {res['recall_at_k']:.2f}"

            print(f"\n{status_tag} {res['id']}: \"{res['question']}\"")
            print(f"       Expected Pages: {res['expected_pages']} | {first_str} | {rr_str} | {p_str} | {r_str}")

            if verbose:
                for rank_item in res["retrieved_ranks"]:
                    tag = "+" if rank_item["is_relevant"] else "-"
                    print(f"         [{tag}] Rank {rank_item['rank']}: Page {rank_item['page']} | {rank_item['snippet']}...")

    # Calculate aggregate summary metrics
    total_queries = len(per_query_results)
    mean_precision = sum(r["precision_at_k"] for r in per_query_results) / total_queries
    mean_recall = sum(r["recall_at_k"] for r in per_query_results) / total_queries
    mrr = sum(r["reciprocal_rank"] for r in per_query_results) / total_queries
    hit_count = sum(1 for r in per_query_results if r["hit"])
    hit_rate = (hit_count / total_queries) * 100.0

    summary_metrics = {
        "strategy": strategy,
        "total_queries_evaluated": total_queries,
        "k": k,
        "fetch_k": fetch_k,
        "lambda_mult": lambda_mult,
        "dense_weight": dense_weight,
        "sparse_weight": sparse_weight,
        "rrf_c": rrf_c,
        "relevance_mode": match_mode,
        "min_keywords": min_keywords,
        "mean_precision_at_k": round(mean_precision, 4),
        "mean_recall_at_k": round(mean_recall, 4),
        "mean_reciprocal_rank_mrr": round(mrr, 4),
        "hit_rate_at_k_percentage": round(hit_rate, 2),
    }

    if not quiet:
        print("\n" + "=" * 80)
        print(f"OVERALL EVALUATION SUMMARY ({strategy.upper()})")
        print("=" * 80)
        print(f"Total Questions Evaluated: {total_queries}")
        print(f"Precision@{k}:              {mean_precision * 100:.2f}%  (Average fraction of retrieved chunks that are relevant)")
        print(f"Recall@{k}:                 {mean_recall * 100:.2f}%  (Average fraction of expected pages covered in top {k})")
        print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}  (Average reciprocal rank of first relevant chunk)")
        print(f"Hit Rate@{k}:               {hit_rate:.2f}%  ({hit_count}/{total_queries} queries retrieved at least 1 relevant chunk)")
        print("=" * 80)

    report = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "summary": summary_metrics,
        "queries": per_query_results,
    }

    if output_report_path:
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        if not quiet:
            print(f"\n[Saved] Evaluation report written to: {output_report_path}\n")

    return report


def run_strategy_comparison(
    dataset_path: Path,
    output_report_path: Path,
    k: int = 5,
    match_mode: str = "page_or_keyword",
    min_keywords: int = 1,
) -> Dict[str, Any]:
    """Execute side-by-side benchmark comparison across Semantic, MMR, and Hybrid (RRF) search.

    Args:
        dataset_path: Path to benchmark questions file.
        output_report_path: Path to output comparison JSON report.
        k: Top-K evaluation cut-off.
        match_mode: Relevance matching logic mode.
        min_keywords: Minimum technical keywords required for a match.

    Returns:
        Dictionary containing comparative metrics across all three strategies.
    """
    print("\n" + "=" * 85)
    print("CIRCUITMIND AI - RETRIEVAL STRATEGY BENCHMARK COMPARISON")
    print("=" * 85)
    print(f"Evaluating 3 Strategies on: {dataset_path.name}")
    print("1. Semantic-Only (Dense Vector Similarity)")
    print("2. Maximum Marginal Relevance (MMR)")
    print("3. Hybrid Search (Semantic + BM25 with Reciprocal Rank Fusion)")
    print("=" * 85)

    strategies = ["semantic", "mmr", "hybrid"]
    comparison_results: Dict[str, Any] = {}

    for strat in strategies:
        print(f"\n>>> Running evaluation for '{strat.upper()}' strategy...")
        report = run_benchmark(
            dataset_path=dataset_path,
            strategy=strat,
            k=k,
            match_mode=match_mode,
            min_keywords=min_keywords,
            quiet=True,
        )
        comparison_results[strat] = report["summary"]

    # Display comparison table
    print("\n" + "=" * 85)
    print("STRATEGY COMPARISON RESULTS (Top-5 Chunks)")
    print("=" * 85)
    header = f"{'Strategy':<18} | {'Precision@5':<14} | {'Recall@5':<14} | {'MRR':<12} | {'Hit Rate@5':<12}"
    print(header)
    print("-" * 85)

    for strat in strategies:
        s = comparison_results[strat]
        p_str = f"{s['mean_precision_at_k'] * 100:.2f}%"
        r_str = f"{s['mean_recall_at_k'] * 100:.2f}%"
        mrr_str = f"{s['mean_reciprocal_rank_mrr']:.4f}"
        hit_str = f"{s['hit_rate_at_k_percentage']:.2f}%"
        name = strat.upper()
        if strat == "hybrid":
            name = "HYBRID (RRF)"
        print(f"{name:<18} | {p_str:<14} | {r_str:<14} | {mrr_str:<12} | {hit_str:<12}")

    print("=" * 85)

    full_comparison = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "k": k,
        "match_mode": match_mode,
        "comparison": comparison_results,
    }

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(full_comparison, f, indent=2)

    print(f"\n[Saved] Comparison report written to: {output_report_path}\n")
    return full_comparison


def main() -> int:
    """CLI entry point for retrieval evaluation."""
    parser = argparse.ArgumentParser(
        description="CircuitMind AI - Retrieval Evaluation Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(PROJECT_ROOT / "evaluation" / "questions.json"),
        help="Path to the JSON file containing evaluation questions.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "evaluation" / "results" / "retrieval_report.json"),
        help="Output destination path for the JSON evaluation report.",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        type=str,
        default="hybrid",
        choices=["hybrid", "mmr", "semantic", "keyword", "compare"],
        help="Strategy to evaluate, or 'compare' to benchmark Semantic, MMR, and Hybrid side-by-side.",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=5,
        help="Evaluation cut-off rank (Precision@K, Recall@K).",
    )
    parser.add_argument(
        "--fetch-k",
        type=int,
        default=15,
        help="MMR initial candidate pool size.",
    )
    parser.add_argument(
        "--lambda-mult",
        type=float,
        default=0.7,
        help="MMR lambda trade-off factor (0.0 to 1.0).",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=0.5,
        help="Hybrid RRF weight for semantic search.",
    )
    parser.add_argument(
        "--sparse-weight",
        type=float,
        default=0.5,
        help="Hybrid RRF weight for BM25 keyword search.",
    )
    parser.add_argument(
        "--rrf-c",
        type=int,
        default=60,
        help="RRF smoothing constant.",
    )
    parser.add_argument(
        "--match-mode",
        type=str,
        default="page_or_keyword",
        choices=["page_or_keyword", "page_and_keyword", "page_only", "keyword_only"],
        help="Relevance matching criterion.",
    )
    parser.add_argument(
        "--min-keywords",
        type=int,
        default=1,
        help="Minimum required keyword occurrences in chunk.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Display detailed chunk snippets for each rank.",
    )

    args = parser.parse_args()

    try:
        if args.strategy == "compare":
            comparison_output = PROJECT_ROOT / "evaluation" / "results" / "retrieval_comparison_report.json"
            run_strategy_comparison(
                dataset_path=Path(args.dataset),
                output_report_path=comparison_output,
                k=args.k,
                match_mode=args.match_mode,
                min_keywords=args.min_keywords,
            )
        else:
            run_benchmark(
                dataset_path=Path(args.dataset),
                output_report_path=Path(args.output),
                strategy=args.strategy,
                k=args.k,
                fetch_k=args.fetch_k,
                lambda_mult=args.lambda_mult,
                dense_weight=args.dense_weight,
                sparse_weight=args.sparse_weight,
                rrf_c=args.rrf_c,
                match_mode=args.match_mode,
                min_keywords=args.min_keywords,
                verbose=args.verbose,
            )
        return 0
    except Exception as err:
        print(f"\n[Error] Evaluation run failed: {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
