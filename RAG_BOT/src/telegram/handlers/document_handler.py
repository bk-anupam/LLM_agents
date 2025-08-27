import os
from telebot.types import Message
from RAG_BOT.src.logger import logger
from RAG_BOT.src.utils import detect_document_language
from .base_handler import BaseHandler

class DocumentHandler(BaseHandler):
    """Handles incoming document messages for indexing."""

    def handle(self, message: Message):
        user_id = message.from_user.id
        if not message.document:
            self.bot.reply_to(message, "No document provided.")
            return

        file_id = message.document.file_id
        mime_type = message.document.mime_type
        logger.info(f"Received document from user mime_type: {mime_type} (file_id: {file_id})")
        file_path = None
        processed_successfully = False
        file_name = "unknown_file" # Default
        try:
            file_ext, default_doc_name, processing_mime_type = self._process_document_metadata(message)
            file_name = self._determine_file_name(message, file_ext, default_doc_name)
            logger.info(f"User {user_id} uploaded {mime_type} (processed as {processing_mime_type}): {file_name}")
            
            upload_dir = os.path.join(self.project_root_dir, "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, file_name)
            
            file_info = self.bot.get_file(file_id)
            downloaded_file = self.bot.download_file(file_info.file_path)
            with open(file_path, 'wb') as new_file:
                new_file.write(downloaded_file)
            logger.info(f"Document saved to: {file_path}")

            documents = []
            if processing_mime_type == 'application/pdf':
                documents = self.pdf_processor.load_pdf(file_path)
            elif processing_mime_type in ['text/html', 'application/xhtml+xml']:
                doc = self.htm_processor.load_htm(file_path)
                if doc:
                    documents.append(doc)

            if not documents:
                logger.warning(f"No documents loaded from: {file_path}. Skipping indexing.")
                self.bot.reply_to(message, f"Could not load content from '{file_name}'.")
                return

            language = detect_document_language(documents, file_name_for_logging=file_name)
            if language not in ['en', 'hi']:
                logger.warning(f"Unsupported language detected: {language}. Aborting document indexing.")
                self.bot.reply_to(message, f"Unsupported language '{language}' detected in '{file_name}'. Indexing aborted.")
                return
            
            for doc in documents:
                doc.metadata['language'] = language
            
            was_indexed = self.vector_store_instance.index_document(documents, semantic_chunk=self.config.SEMANTIC_CHUNKING)
            if was_indexed:
                self.bot.reply_to(message, f"Document '{file_name}' uploaded and indexed successfully.")
                processed_successfully = True
            else:
                self.bot.reply_to(message, f"Document '{file_name}' was not indexed (possibly already exists or an error occurred).")

        except ValueError as ve:
             logger.warning(f"Unsupported file type for user {user_id}: {ve}")
             self.bot.reply_to(message, str(ve))
             return
        except Exception as e:
            logger.error(f"Error handling document upload from user {user_id} for {file_name}: {str(e)}", exc_info=True)
            self.bot.reply_to(message, "Sorry, I encountered an error processing your document.")
        finally:
            if file_path:
                self._cleanup_uploaded_file(file_path, processed_successfully)

    def _cleanup_uploaded_file(self, file_path, processed_successfully):
        if processed_successfully and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Successfully processed and removed '{file_path}' from uploads directory.")
            except OSError as e:
                logger.error(f"Error removing processed file '{file_path}' from uploads: {e}")
        elif not processed_successfully and os.path.exists(file_path):
            logger.info(f"File '{file_path}' was not successfully processed/indexed and will remain in the uploads directory.")

    def _determine_file_name(self, message, file_ext, default_doc_name):
        original_file_name = message.document.file_name
        file_name = original_file_name or default_doc_name
        if not file_name.lower().endswith(file_ext) and original_file_name is None:
            file_name = os.path.splitext(file_name)[0] + file_ext
        return file_name

    def _process_document_metadata(self, message: Message):
        mime_type = message.document.mime_type
        file_id = message.document.file_id
        original_file_name = message.document.file_name
        file_ext = None
        processing_mime_type = mime_type

        if mime_type == 'application/pdf':
            file_ext, default_doc_name = ".pdf", f"doc_{file_id}.pdf"
        elif mime_type in ['text/html', 'application/xhtml+xml']:
            file_ext, default_doc_name = ".htm", f"doc_{file_id}.htm"
        elif mime_type == 'application/octet-stream' and original_file_name:
            name, ext = os.path.splitext(original_file_name)
            if ext.lower() in ['.htm', '.html']:
                file_ext, default_doc_name, processing_mime_type = ".htm", original_file_name, 'text/html'
            elif ext.lower() == '.pdf':
                file_ext, default_doc_name, processing_mime_type = ".pdf", original_file_name, 'application/pdf'
        
        if file_ext is None:
            raise ValueError(f"Unsupported file type ({mime_type}) or unable to determine from name.")
        return file_ext, default_doc_name, processing_mime_type
