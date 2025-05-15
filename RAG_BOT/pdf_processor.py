import os
import sys
from langchain_community.document_loaders import PyMuPDFLoader # Keep only PyMuPDFLoader if PyPDFLoader is unused
from langchain_core.documents import Document
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from RAG_BOT.logger import logger
from RAG_BOT.document_processor import DocumentProcessor # Import the base class


class PdfProcessor(DocumentProcessor):
    """
    Processes PDF files to extract text, metadata (including dates and type),
    inheriting common functionality from DocumentProcessor.
    """
    def _load_pdf_multi_murli(self, pdf_path, header_check_chars=300):
        """
        Loads a PDF, detects dates at the start of pages (indicating new Murlis),
        and applies the correct date metadata to all pages of that Murli.
        This is a private helper function.

        Args:
            pdf_path (str): Path to the PDF file.
            header_check_chars (int): Number of characters at the start of a page to check for a date.

        Returns:
            list[Document]: A list of Document objects with corrected date metadata.
        """
        #loader = PyPDFLoader(pdf_path)
        loader = PyMuPDFLoader(pdf_path) # Use PyMuPDFLoader for better performance
        try:
            pages = loader.load() # Use load() instead of load_and_split() initially
        except Exception as e:
            logger.error(f"Failed to load PDF: {pdf_path}. Error: {e}")
            return []

        all_documents = []
        current_date = None
        current_is_avyakt = None # Track avyakt status similarly
        # Extract filename from the path for the source metadata
        filename = os.path.basename(pdf_path)
        logger.info(f"Processing {len(pages)} pages from {pdf_path}...")

        for i, page in enumerate(pages):
            page_text = page.page_content
            metadata = page.metadata.copy() # Work on a copy

            # Check the beginning of the page for a new date/type
            header_text = page_text[:header_check_chars]
            header_len = len(header_text)
            header_preview = repr(header_text[:100]) # Use repr() to show whitespace/special chars clearly
            logger.debug(f"Page {metadata.get('page', i)}: Header Text Length={header_len}. Preview (first 100 chars): {header_preview}")
            potential_new_date = self.extract_date_from_text(header_text) # Use self.method
            potential_is_avyakt = self.get_murli_type(header_text) # Use self.method

            # If a date is found in the header, assume it's the start of a new Murli
            if potential_new_date:
                if potential_new_date != current_date:
                    logger.debug(f"Found new date '{potential_new_date}' on page {metadata.get('page', i)}.")
                    current_date = potential_new_date
                # Update avyakt status whenever a new date is found
                if potential_is_avyakt != current_is_avyakt:
                    logger.debug(f"Murli type set to Avyakt={potential_is_avyakt} starting page {metadata.get('page', i)}.")
                    current_is_avyakt = potential_is_avyakt

            # Apply the current date and type (if found) to the page's metadata
            if current_date:
                metadata["date"] = current_date
            else:
                # If no date has been found yet (e.g., first few pages have no date)
                if "date" in metadata:
                    del metadata["date"] # Remove potentially incorrect date from loader

            # Only add is_avyakt to metadata if it's an Avyakt Murli
            if current_is_avyakt is True:
                metadata["is_avyakt"] = True
            elif "is_avyakt" in metadata:
                del metadata["is_avyakt"]

            metadata["source"] = filename 
            # Create a new Document object with updated metadata
            # Using page_text ensures we have the content, metadata has page number etc.
            processed_doc = Document(page_content=page_text, metadata=metadata)
            all_documents.append(processed_doc)

        if current_date is None:
            logger.warning(f"No date could be extracted from the headers in {pdf_path}.")

        logger.info(f"Finished processing {len(all_documents)} documents metadata date: {current_date}, is_avyakt: {current_is_avyakt}.")
        return all_documents


    def load_pdf(self, pdf_path):
        """
        Loads a PDF and processes it to extract content and metadata,
        handling multiple Murlis within a single PDF.        
        Args:
            pdf_path (str): Path to the PDF file.
        Returns:
            list[Document]: A list of Document objects with extracted content and metadata.
        """
        # Call the private helper function with a default header_check_chars
        return self._load_pdf_multi_murli(pdf_path, header_check_chars=300)


if __name__ == "__main__":
    # Example usage with the new class structure
    TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'tests', 'data', 'hindi')
    pdf_name = "03. AV-H-07.01.1980.pdf"
    pdf_path = os.path.join(TEST_DATA_DIR, pdf_name)

    # Instantiate the processor
    pdf_processor = PdfProcessor()

    # Load documents using the instance method
    documents_with_correct_dates = pdf_processor.load_pdf(pdf_path)

    if documents_with_correct_dates:
        # Optional: Print metadata of first few docs to verify
        for i in range(min(5, len(documents_with_correct_dates))):
             print(f"Doc {i} Metadata: {documents_with_correct_dates[i].metadata}")

        # Proceed with splitting using inherited methods
        # chunks = pdf_processor.split_text(documents_with_correct_dates)
        chunks = pdf_processor.semantic_chunking(documents_with_correct_dates) # Example using semantic
        # ... further processing (e.g., indexing chunks) ...
        print(f"\nTotal chunks created: {len(chunks)}")
        if chunks:
            print(f"First chunk metadata: {chunks[0].metadata}")
            print(f"Last chunk metadata: {chunks[-1].metadata}")
    else:
        print(f"Could not process PDF: {pdf_path}")
