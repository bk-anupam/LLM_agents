import os
import pytest
from langchain_core.documents import Document
from RAG_BOT.pdf_processor import load_pdf

# Define the path to the test data directory relative to this test file
# Assuming the test file is in RAG_BOT/tests/integration/
# and data is in RAG_BOT/tests/data/hindi/
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'hindi')

# Helper function to check if a path is a valid file
def check_pdf_exists(pdf_name):
    pdf_path = os.path.join(TEST_DATA_DIR, pdf_name)
    if not os.path.isfile(pdf_path):
        pytest.skip(f"Test PDF not found: {pdf_path}")
    return pdf_path

@pytest.fixture
def sakar_pdf_path():
    return check_pdf_exists("01.03.14-h.pdf")

@pytest.fixture
def avyakt_pdf_path():
    return check_pdf_exists("03. AV-H-07.01.1980.pdf")

@pytest.fixture
def multi_date_header_pdf_path():
    return check_pdf_exists("FHM - 17-11-2013 (AM Revised - 31-12-1996).pdf")

def test_load_pdf_sakar_murli(sakar_pdf_path):
    """
    Tests loading a standard Sakar Murli PDF.
    Checks if the correct date is extracted and is_avyakt is not set.
    """
    documents = load_pdf(sakar_pdf_path)

    assert isinstance(documents, list)
    assert len(documents) > 0
    assert all(isinstance(doc, Document) for doc in documents)

    expected_date = "2014-03-01"
    for doc in documents:
        assert "date" in doc.metadata
        assert doc.metadata["date"] == expected_date
        assert "is_avyakt" not in doc.metadata

def test_load_pdf_avyakt_murli(avyakt_pdf_path):
    """
    Tests loading a standard Avyakt Murli PDF.
    Checks if the correct date is extracted and is_avyakt is set to True.
    """
    documents = load_pdf(avyakt_pdf_path)

    assert isinstance(documents, list)
    assert len(documents) > 0
    assert all(isinstance(doc, Document) for doc in documents)

    expected_date = "1980-01-07"
    for doc in documents:
        assert "date" in doc.metadata
        assert doc.metadata["date"] == expected_date
        assert "is_avyakt" in doc.metadata
        assert doc.metadata["is_avyakt"] is True

def test_load_pdf_multiple_header_dates(multi_date_header_pdf_path):
    """
    Tests loading a PDF with multiple dates in the header (original/revised).
    Checks if the *first* date found is extracted and applied consistently.
    """
    documents = load_pdf(multi_date_header_pdf_path)

    assert isinstance(documents, list)
    assert len(documents) > 0
    assert all(isinstance(doc, Document) for doc in documents)

    # The current logic should pick the first date it finds in the header
    expected_date = "2013-11-17"
    for doc in documents:
        assert "date" in doc.metadata
        assert doc.metadata["date"] == expected_date
        # Assuming this is Sakar based on filename pattern, but code checks content.
        # If the content check finds 'avyakt', this assertion might need adjustment.
        # For now, testing based on the primary date extraction logic.
        assert "is_avyakt" not in doc.metadata
