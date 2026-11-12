"""Unit and integration tests for Milestone 1A: PDF Datasheet Ingestion."""

import tempfile
from pathlib import Path
import pytest
import pymupdf

from app.rag.ingestion import DatasheetIngestor, generate_chunk_id


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> Path:
    """Create a temporary multi-page electronic datasheet PDF for testing."""
    pdf_file = tmp_path / "LM7805_Voltage_Regulator.pdf"
    doc = pymupdf.open()

    # Page 1: Features and Description
    page1 = doc.new_page()
    page1.insert_text(
        (50, 72),
        "LM7805 3-Terminal Positive Voltage Regulator\n\n"
        "Features:\n"
        "- Output Current up to 1.5A\n"
        "- Output Voltages of 5V\n"
        "- Thermal Overload Protection\n"
        "- Short Circuit Protection\n"
        "- Output Transistor Safe Operating Area Protection\n\n"
        "Description:\n"
        "The LM7805 series of fixed-voltage integrated-circuit voltage regulators is "
        "designed for a wide range of applications. These include on-card regulation for "
        "elimination of noise and distribution problems associated with single-point regulation.\n",
        fontsize=11,
    )

    # Page 2: Blank page (should be skipped by parser)
    doc.new_page()

    # Page 3: Electrical Characteristics
    page3 = doc.new_page()
    page3.insert_text(
        (50, 72),
        "LM7805 Electrical Characteristics:\n\n"
        "Absolute Maximum Ratings:\n"
        "- Input Voltage (for VO = 5V): 35V\n"
        "- Operating Junction Temperature: 0 to 125 deg C\n"
        "- Storage Temperature Range: -65 to 150 deg C\n\n"
        "Electrical Parameters (TJ = 25C, IO = 500mA, VI = 10V):\n"
        "- Output Voltage (VO): Min 4.8V, Typ 5.0V, Max 5.2V\n"
        "- Line Regulation (7V <= VI <= 25V): Typ 3.0mV, Max 50mV\n"
        "- Load Regulation (5mA <= IO <= 1.5A): Typ 15mV, Max 50mV\n"
        "- Quiescent Current: Typ 4.2mA, Max 8.0mA\n"
        "- Dropout Voltage (IO = 1A): Typ 2.0V\n",
        fontsize=11,
    )

    doc.save(str(pdf_file))
    doc.close()
    return pdf_file


def test_generate_chunk_id_stability() -> None:
    """Verify that chunk IDs are deterministic and depend on source, page, and content."""
    id1 = generate_chunk_id("datasheet.pdf", 1, "Output voltage: 5V")
    id2 = generate_chunk_id("datasheet.pdf", 1, "Output voltage: 5V")
    id3 = generate_chunk_id("datasheet.pdf", 2, "Output voltage: 5V")
    id4 = generate_chunk_id("datasheet.pdf", 1, "Output voltage: 3.3V")

    assert id1 == id2, "Identical content and metadata must produce identical chunk IDs."
    assert id1 != id3, "Different page numbers must produce different chunk IDs."
    assert id1 != id4, "Different text content must produce different chunk IDs."
    assert "p1" in id1


def test_extract_pages_validation(tmp_path: Path) -> None:
    """Verify that missing files and non-PDF files raise descriptive errors."""
    ingestor = DatasheetIngestor(persist_directory=str(tmp_path / "chromadb"))

    with pytest.raises(FileNotFoundError):
        ingestor.extract_pages(tmp_path / "nonexistent.pdf")

    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Not a PDF file.")
    with pytest.raises(ValueError, match="not a PDF"):
        ingestor.extract_pages(txt_file)


def test_extract_pages_skips_empty_pages(sample_pdf_path: Path, tmp_path: Path) -> None:
    """Verify that pages without extractable text are skipped."""
    ingestor = DatasheetIngestor(persist_directory=str(tmp_path / "chromadb"))
    pages_data = ingestor.extract_pages(sample_pdf_path)

    # 3 total pages, page 2 is blank, so exactly 2 pages should be returned (Page 1 and Page 3)
    assert len(pages_data) == 2
    assert pages_data[0]["page"] == 1
    assert "Output Current up to 1.5A" in pages_data[0]["text"]
    assert pages_data[1]["page"] == 3
    assert "Dropout Voltage" in pages_data[1]["text"]


def test_ingest_pdf_and_duplicate_prevention(sample_pdf_path: Path, tmp_path: Path) -> None:
    """Verify full ingestion pipeline, vector persistence, and duplicate prevention."""
    chroma_dir = str(tmp_path / "chromadb_test")
    ingestor = DatasheetIngestor(
        persist_directory=chroma_dir,
        collection_name="test_datasheets",
        chunk_size=800,
        chunk_overlap=150,
    )

    # First ingestion
    result = ingestor.ingest_pdf(sample_pdf_path, part_number="LM7805", force=False)
    assert result["status"] == "success"
    assert result["pages_extracted"] == 2
    assert result["total_chunks"] >= 2
    assert result["part_number"] == "LM7805"

    # Verify chunks in ChromaDB
    collection = ingestor.collection
    stored = collection.get(where={"source": sample_pdf_path.name})
    assert len(stored["ids"]) == result["total_chunks"]
    assert stored["metadatas"][0]["part_number"] == "LM7805"
    assert stored["metadatas"][0]["page"] in (1, 3)

    # Second ingestion without force must be skipped
    second_result = ingestor.ingest_pdf(sample_pdf_path, part_number="LM7805", force=False)
    assert second_result["status"] == "skipped"
    assert second_result["reason"] == "already_ingested"

    # Ingestion with force=True should re-index without duplication
    force_result = ingestor.ingest_pdf(sample_pdf_path, part_number="LM7805", force=True)
    assert force_result["status"] == "success"
    post_force = collection.get(where={"source": sample_pdf_path.name})
    assert len(post_force["ids"]) == result["total_chunks"]

