# Project Brief

This document serves as the foundational brief for the project, outlining its core requirements, goals, and scope.

## Project Goals

- Provide a Telegram bot that answers spiritual questions using Retrieval-Augmented Generation (RAG) techniques.
- Maintain conversational context, allowing the bot to answer questions about the chat history.
- Enable users to query a knowledge base of spiritual documents and receive accurate, context-grounded answers.
- Support document upload and automatic indexing for PDFs and HTM files.
- Deliver multi-language support (English and Hindi) and user-selectable language preference.
- Ensure robust, production-ready deployment with logging, configuration, and integration tests.

## Scope

- Telegram interface for user interaction.
- A stateful, conversational agent with a robust workflow including:
  - Intelligent routing to distinguish between knowledge queries and conversational chat.
  - Hybrid search (semantic and lexical) for comprehensive document retrieval.
  - Web search fallback (using Tavily) when local documents are insufficient.
  - Context reranking, relevance evaluation, and query reframing.
  - Grounded answer generation based on retrieved context.
  - Conversational memory to handle multi-turn dialogues.
- ChromaDB vector store for document embeddings and retrieval.
- Integration with Google Gemini LLM for query analysis and answer generation.
- Document upload and indexing pipeline.
- Multi-language support and language detection.
- Centralized configuration and logging.
- Integration and unit tests for core functionalities.

## Non-Goals

- The agent's primary purpose is answering questions from the spiritual document domain; it is not an open-domain general-purpose chatbot.
- Does not provide real-time updates or notifications beyond Telegram's capabilities.
- No support for non-Telegram messaging platforms.
- Does not include advanced user management or authentication features.
