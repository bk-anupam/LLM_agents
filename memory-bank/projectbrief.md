# Project Brief

This document serves as the foundational brief for the project, outlining its core requirements, goals, and scope.

## Project Goals

- Provide a Telegram bot that answers spiritual questions using Retrieval-Augmented Generation (RAG) techniques.
- Enable users to query a knowledge base of spiritual documents and receive accurate, context-grounded answers.
- Support document upload and automatic indexing for PDFs and HTM files.
- Deliver multi-language support (English and Hindi) and user-selectable language preference.
- Ensure robust, production-ready deployment with logging, configuration, and integration tests.

## Scope

- Telegram interface for user interaction.
- RAG agent workflow with retrieval, reranking, evaluation, query reframing, and answer generation.
- ChromaDB vector store for document embeddings and retrieval.
- Integration with Google Gemini LLM for query analysis and answer generation.
- Document upload and indexing pipeline.
- Multi-language support and language detection.
- Centralized configuration and logging.
- Integration and unit tests for core functionalities.

## Non-Goals

- Not intended as a general-purpose chatbot outside the spiritual document domain.
- Does not provide real-time updates or notifications beyond Telegram's capabilities.
- No support for non-Telegram messaging platforms.
- Does not include advanced user management or authentication features.
