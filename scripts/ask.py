"""CLI tool for querying CircuitMind AI and receiving grounded datasheet answers with citations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so script can run from any working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.rag.answer_generator import GroundedAnswerGenerator


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="CircuitMind AI - Grounded Datasheet Question Answering CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-q",
        "--question",
        type=str,
        default=None,
        help="Engineering question about component datasheets.",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        type=str,
        default="hybrid",
        choices=["hybrid", "mmr", "semantic", "keyword"],
        help="Retrieval strategy to fetch context.",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=5,
        help="Number of retrieved datasheet chunks to provide as context.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=settings.gemini_model,
        help="Google Gemini model name (e.g. 'gemini-2.5-flash').",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Display the raw retrieved datasheet passages alongside the answer.",
    )
    return parser.parse_args()


def main() -> int:
    """Main CLI execution routine."""
    args = parse_args()

    # Query acquisition
    question = args.question
    if not question:
        try:
            question = input("\nAsk CircuitMind Copilot: ")
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            return 0

    if not question or not question.strip():
        print("\n[Error] Question cannot be empty or only whitespace.")
        return 1

    try:
        generator = GroundedAnswerGenerator(model_name=args.model)
    except Exception as err:
        print(f"\n[Initialization Error] {err}")
        return 1

    print(f"\nSearching datasheets [{args.strategy.upper()}] & synthesizing grounded answer...\n")
    try:
        result = generator.generate_answer(
            question=question,
            k=args.k,
            strategy=args.strategy,
        )
    except ValueError as err:
        print(f"[Configuration Error] {err}")
        return 1
    except RuntimeError as err:
        print(f"[API Error] {err}")
        return 1
    except Exception as err:
        print(f"[Unexpected Error] {err}")
        return 1

    print("=" * 70)
    print("CIRCUITMIND AI - GROUNDED ENGINEERING ANSWER")
    print("=" * 70)
    print(f"Question: \"{result.question}\"\n")
    print(result.answer)
    print("\n" + "-" * 70)
    print("SOURCE CITATIONS:")
    if result.sources:
        for citation in result.sources:
            part_str = f" (Part: {citation.part_number})" if citation.part_number else ""
            print(f" • {citation.source} | Page: {citation.page}{part_str}")
    else:
        print(" • None (Insufficient context)")
    print("=" * 70)

    if args.show_context and result.retrieved_chunks:
        print("\nRETRIEVED CONTEXT CHUNKS:")
        for idx, chunk in enumerate(result.retrieved_chunks, start=1):
            print(f" [{idx}] Source: {chunk.get('source')} | Page: {chunk.get('page')} | ID: {chunk.get('chunk_id')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

