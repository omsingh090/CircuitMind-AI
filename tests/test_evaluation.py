"""Unit tests for CircuitMind AI Retrieval Evaluation Framework."""

import json
from pathlib import Path
import pytest
from langchain_core.documents import Document

from evaluation.evaluate_retrieval import RelevanceMatcher


def test_questions_dataset_schema() -> None:
    """Verify that evaluation/questions.json exists, contains >=15 items with valid schema."""
    dataset_path = Path("evaluation/questions.json")
    assert dataset_path.exists(), "evaluation/questions.json must exist."

    with open(dataset_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    assert len(questions) >= 15, "Dataset must have at least 15 technical questions."

    for item in questions:
        assert "id" in item
        assert "question" in item and len(item["question"].strip()) > 0
        assert "expected_pages" in item and isinstance(item["expected_pages"], list)
        assert len(item["expected_pages"]) > 0, f"Question {item['id']} must have expected pages."
        assert "expected_keywords" in item and isinstance(item["expected_keywords"], list)
        assert len(item["expected_keywords"]) > 0, f"Question {item['id']} must have expected keywords."


def test_relevance_matcher_modes() -> None:
    """Verify different relevance matching modes in RelevanceMatcher."""
    doc = Document(
        page_content="Linear + 10-mV/°C scale factor with 0.5°C ensured accuracy at 25°C.",
        metadata={"source": "lm35.pdf", "page": 1},
    )

    # 1. page_only mode
    matcher_page = RelevanceMatcher(mode="page_only")
    rel, _ = matcher_page.is_relevant(doc, expected_pages=[1, 2], expected_keywords=["unrelated"])
    assert rel is True

    rel_fail, _ = matcher_page.is_relevant(doc, expected_pages=[4], expected_keywords=["scale factor"])
    assert rel_fail is False

    # 2. keyword_only mode
    matcher_kw = RelevanceMatcher(mode="keyword_only")
    rel_kw, _ = matcher_kw.is_relevant(doc, expected_pages=[99], expected_keywords=["scale factor"])
    assert rel_kw is True

    rel_kw_fail, _ = matcher_kw.is_relevant(doc, expected_pages=[1], expected_keywords=["nonexistent keyword"])
    assert rel_kw_fail is False

    # 3. page_and_keyword mode
    matcher_and = RelevanceMatcher(mode="page_and_keyword")
    rel_and, _ = matcher_and.is_relevant(doc, expected_pages=[1], expected_keywords=["10-mV/°C"])
    assert rel_and is True

    rel_and_fail, _ = matcher_and.is_relevant(doc, expected_pages=[99], expected_keywords=["10-mV/°C"])
    assert rel_and_fail is False

    # 4. page_or_keyword mode
    matcher_or = RelevanceMatcher(mode="page_or_keyword")
    rel_or1, _ = matcher_or.is_relevant(doc, expected_pages=[1], expected_keywords=["nonexistent"])
    assert rel_or1 is True
    rel_or2, _ = matcher_or.is_relevant(doc, expected_pages=[99], expected_keywords=["10-mV/°C"])
    assert rel_or2 is True

