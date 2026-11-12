"""CLI entry point for ingesting electronic component datasheets into CircuitMind AI."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so script can run from any working directory
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.config import get_settings
from app.rag.ingestion import DatasheetIngestor


def setup_logging(verbose: bool = False) -> None:
    """Configure console logging level and format."""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    logging.basicConfig(level=level, format=format_str, datefmt="%H:%M:%S")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="CircuitMind AI - Electronic Component Datasheet Ingestion CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path to a single PDF datasheet to ingest.",
    )
    group.add_argument(
        "-d",
        "--dir",
        type=str,
        default=str(settings.data_raw_dir),
        help="Directory containing PDF datasheets to ingest in batch.",
    )

    parser.add_argument(
        "-p",
        "--part-number",
        type=str,
        default=None,
        help="Optional electronic component part number (e.g. 'ESP32-WROOM-32', 'LM7805').",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion and overwrite existing chunks in ChromaDB.",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=settings.chroma_persist_directory,
        help="Custom ChromaDB storage directory.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=settings.chroma_collection_name,
        help="Custom ChromaDB collection name.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable detailed DEBUG logging.",
    )

    return parser.parse_args()


def main() -> int:
    """Main CLI execution routine."""
    args = parse_args()
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger("ingest-cli")

    logger.info("Initializing CircuitMind AI Datasheet Ingestor...")
    try:
        ingestor = DatasheetIngestor(
            persist_directory=args.persist_dir,
            collection_name=args.collection,
        )
    except Exception as err:
        logger.error("Failed to initialize ingestor: %s", err)
        return 1

    if args.file:
        file_path = Path(args.file)
        logger.info("Processing single file: %s", file_path)
        try:
            result = ingestor.ingest_pdf(
                pdf_path=file_path,
                part_number=args.part_number,
                force=args.force,
            )
            print("\n" + "=" * 60)
            print("INGESTION SUMMARY")
            print("=" * 60)
            print(f"File:            {result.get('source')}")
            print(f"Status:          {result.get('status').upper()}")
            print(f"Pages Extracted: {result.get('pages_extracted', 0)}")
            print(f"Chunks Stored:   {result.get('total_chunks', 0)}")
            if result.get("part_number"):
                print(f"Part Number:     {result.get('part_number')}")
            if result.get("message"):
                print(f"Note:            {result.get('message')}")
            print("=" * 60 + "\n")
            return 0
        except Exception as err:
            logger.error("Ingestion failed: %s", err)
            return 1
    else:
        dir_path = Path(args.dir)
        logger.info("Processing directory: %s", dir_path)
        try:
            results = ingestor.ingest_directory(directory_path=dir_path, force=args.force)
            if not results:
                logger.warning("No PDF files were processed.")
                return 0

            print("\n" + "=" * 60)
            print("BATCH INGESTION SUMMARY")
            print("=" * 60)
            for res in results:
                source = res.get("source", "Unknown")
                status = res.get("status", "unknown").upper()
                chunks = res.get("total_chunks", 0)
                note = res.get("message") or res.get("error") or ""
                print(f"• {source:<30} | {status:<10} | {chunks:>3} chunks | {note}")
            print("=" * 60 + "\n")
            return 0
        except Exception as err:
            logger.error("Batch ingestion failed: %s", err)
            return 1


if __name__ == "__main__":
    sys.exit(main())

