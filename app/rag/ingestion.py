"""PDF Datasheet Ingestion Pipeline for CircuitMind AI.

Extracts text page-by-page from technical PDF datasheets, splits content
into retrieval-optimized chunks with stable IDs, generates Hugging Face
vector embeddings, and persists them into ChromaDB while preventing duplicates.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import pymupdf  # modern PyMuPDF import
except ImportError:
    import fitz as pymupdf  # fallback for older versions

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter


try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from app.config import get_settings

logger = logging.getLogger(__name__)


def generate_chunk_id(source: str, page: int, chunk_content: str) -> str:
    """Generate a stable, deterministic chunk ID from source, page, and content hash.

    Args:
        source: Name or relative path of the source PDF.
        page: 1-indexed page number where the chunk originates.
        chunk_content: Extracted text content of the chunk.

    Returns:
        A deterministic string ID combining stem name, page number, and hash.
    """
    clean_source = Path(source).stem.replace(" ", "_").replace("-", "_")
    payload = f"{source}::page_{page}::{chunk_content.strip()}".encode("utf-8")
    hash_digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"{clean_source}_p{page}_{hash_digest}"


class DatasheetIngestor:
    """Ingestion pipeline for electronic component datasheets."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        """Initialize the datasheet ingestor with settings and vector store.

        Args:
            persist_directory: Directory for ChromaDB persistent storage.
            collection_name: ChromaDB collection name.
            embedding_model_name: Hugging Face embedding model name.
            chunk_size: Maximum character count per chunk.
            chunk_overlap: Overlapping character count between consecutive chunks.
        """
        settings = get_settings()

        self.persist_directory = persist_directory or settings.chroma_persist_directory
        self.collection_name = collection_name or settings.chroma_collection_name
        self.embedding_model_name = embedding_model_name or settings.embedding_model_name
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        # Ensure persist directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        logger.info(
            "Initializing DatasheetIngestor (ChromaDB: %s, Collection: %s, Model: %s)",
            self.persist_directory,
            self.collection_name,
            self.embedding_model_name,
        )

        # Initialize Text Splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
        )

        # Initialize Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Initialize ChromaDB persistent client and collection
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def extract_pages(self, pdf_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Validate PDF and extract text from each page independently.

        Skips pages with no extractable text.

        Args:
            pdf_path: Path to the target PDF file.

        Returns:
            List of dictionaries containing page number, extracted text, and source filename.

        Raises:
            FileNotFoundError: If the specified PDF file does not exist.
            ValueError: If the file is not a PDF or contains no extractable text.
        """
        resolved_path = Path(pdf_path).resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {resolved_path}")

        if resolved_path.suffix.lower() != ".pdf":
            raise ValueError(f"Target file is not a PDF: {resolved_path.name}")

        filename = resolved_path.name
        pages_data: List[Dict[str, Any]] = []

        try:
            doc = pymupdf.open(str(resolved_path))
        except Exception as err:
            logger.error("Failed to open PDF '%s': %s", filename, err)
            raise ValueError(f"Corrupted or unreadable PDF '{filename}': {err}") from err

        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            raise ValueError(f"PDF file '{filename}' has zero pages.")

        for page_idx in range(total_pages):
            page_number = page_idx + 1
            page = doc[page_idx]
            text = page.get_text("text").strip()

            if not text:
                logger.info(
                    "Page %d of '%s' contains no extractable text. Skipping.",
                    page_number,
                    filename,
                )
                continue

            pages_data.append(
                {
                    "page": page_number,
                    "text": text,
                    "source": filename,
                    "file_path": str(resolved_path),
                }
            )

        doc.close()

        if not pages_data:
            raise ValueError(
                f"PDF file '{filename}' contains no extractable text across all {total_pages} page(s)."
            )

        logger.info(
            "Extracted text from %d/%d pages in '%s'.",
            len(pages_data),
            total_pages,
            filename,
        )
        return pages_data

    def chunk_page(
        self,
        page_data: Dict[str, Any],
        part_number: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Split a single page's text into retrieval chunks with stable IDs and metadata.

        Args:
            page_data: Dict containing 'page', 'text', 'source', and 'file_path'.
            part_number: Optional electronic component part number (e.g., 'ESP32-WROOM-32').

        Returns:
            List of chunk records with 'chunk_id', 'content', and 'metadata'.
        """
        source = page_data["source"]
        page_number = page_data["page"]
        text = page_data["text"]

        splits = self.text_splitter.split_text(text)
        chunks: List[Dict[str, Any]] = []

        for idx, chunk_text in enumerate(splits):
            chunk_id = generate_chunk_id(source, page_number, chunk_text)

            metadata: Dict[str, Any] = {
                "source": source,
                "page": int(page_number),
                "chunk_index": int(idx),
                "total_chunks_on_page": int(len(splits)),
                "char_length": int(len(chunk_text)),
            }

            if part_number:
                metadata["part_number"] = part_number

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "content": chunk_text,
                    "metadata": metadata,
                }
            )

        return chunks

    def is_already_ingested(self, source_filename: str) -> bool:
        """Check if any chunks from the given source PDF already exist in the collection.

        Args:
            source_filename: Name of the PDF file to query.

        Returns:
            True if chunks with the matching source filename are found, False otherwise.
        """
        try:
            results = self.collection.get(
                where={"source": source_filename},
                limit=1,
            )
            return bool(results and results.get("ids") and len(results["ids"]) > 0)
        except Exception as err:
            logger.warning("Error querying collection for existing source '%s': %s", source_filename, err)
            return False

    def ingest_pdf(
        self,
        pdf_path: Union[str, Path],
        part_number: Optional[str] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Ingest a technical PDF datasheet into ChromaDB.

        Performs validation, page extraction, chunking, embedding generation,
        and vector storage. Prevents duplicate ingestion unless force=True.

        Args:
            pdf_path: Path to the target PDF file.
            part_number: Optional component part number for metadata tagging.
            force: If True, deletes existing chunks for this source and re-ingests.

        Returns:
            Dictionary summarizing ingestion status, chunk count, and metadata.
        """
        path = Path(pdf_path).resolve()
        source_name = path.name

        # Check for duplicate ingestion
        if self.is_already_ingested(source_name):
            if not force:
                logger.info(
                    "Datasheet '%s' has already been ingested. Skipping (use force=True to re-ingest).",
                    source_name,
                )
                return {
                    "source": source_name,
                    "status": "skipped",
                    "reason": "already_ingested",
                    "total_chunks": 0,
                    "message": f"'{source_name}' already indexed. Pass force=True to re-index.",
                }
            else:
                logger.info("Force re-ingesting '%s'. Removing existing chunks from ChromaDB.", source_name)
                self.collection.delete(where={"source": source_name})

        # Step 1: Extract pages
        pages_data = self.extract_pages(path)

        # Step 2: Chunk each page independently
        all_chunks: List[Dict[str, Any]] = []
        for page_data in pages_data:
            page_chunks = self.chunk_page(page_data, part_number=part_number)
            all_chunks.extend(page_chunks)

        if not all_chunks:
            raise ValueError(f"No valid chunks could be generated for '{source_name}'.")

        # Step 3: Prepare vectors and metadata for ChromaDB
        chunk_ids = [c["chunk_id"] for c in all_chunks]
        chunk_texts = [c["content"] for c in all_chunks]
        chunk_metadatas = [c["metadata"] for c in all_chunks]

        logger.info(
            "Generating embeddings for %d chunks from '%s'...",
            len(chunk_texts),
            source_name,
        )
        embeddings = self.embeddings.embed_documents(chunk_texts)

        # Step 4: Persist into ChromaDB
        logger.info("Storing %d chunks in ChromaDB collection '%s'...", len(chunk_ids), self.collection_name)
        self.collection.add(
            ids=chunk_ids,
            documents=chunk_texts,
            embeddings=embeddings,
            metadatas=chunk_metadatas,
        )

        logger.info("Successfully ingested '%s' (%d chunks).", source_name, len(chunk_ids))
        return {
            "source": source_name,
            "status": "success",
            "pages_extracted": len(pages_data),
            "total_chunks": len(chunk_ids),
            "chunk_ids": chunk_ids,
            "part_number": part_number,
        }

    def ingest_directory(
        self,
        directory_path: Union[str, Path],
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        """Ingest all PDF files located within a directory.

        Args:
            directory_path: Path to the directory containing PDF datasheets.
            force: If True, re-indexes existing files.

        Returns:
            List of ingestion summary dictionaries for all processed files.
        """
        dir_path = Path(directory_path).resolve()
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        pdf_files = sorted(dir_path.glob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDF files found in directory: %s", dir_path)
            return []

        logger.info("Found %d PDF file(s) in '%s'. Beginning batch ingestion.", len(pdf_files), dir_path)
        results: List[Dict[str, Any]] = []

        for pdf_file in pdf_files:
            try:
                res = self.ingest_pdf(pdf_file, force=force)
                results.append(res)
            except Exception as err:
                logger.error("Failed to ingest '%s': %s", pdf_file.name, err)
                results.append(
                    {
                        "source": pdf_file.name,
                        "status": "failed",
                        "error": str(err),
                        "total_chunks": 0,
                    }
                )

        return results

