# System Patterns

This document details the system architecture, key technical decisions, design patterns in use, component relationships, and critical implementation paths.

## Architecture

The system is a Telegram bot application built around a Retrieval-Augmented Generation (RAG) agent using Langchain and LangGraph. The architecture includes:
- Telegram interface for user interaction.
- Flask/Gunicorn webhook server for Telegram API integration.
- A stateful, conversational agent with core logic managed by a MessageHandler and centralized configuration.
- The agent workflow is defined as a graph with nodes for:
  - **Routing:** Classifying queries as RAG or conversational.
  - **Conversational Handling:** Answering questions about the chat history.
  - **Hybrid Retrieval:** Combining semantic and lexical (BM25) search.
  - **Web Search:** Falling back to Tavily for external information.
  - **Reranking, Evaluation, Query Reframing, and Answer Generation.**
- ChromaDB vector store for document embeddings and retrieval.
- Integration with Google Gemini LLM for query analysis, context evaluation, and answer generation.
- Integration with **Tavily** for web search capabilities.
- Document upload and indexing pipeline for PDFs and HTM files.
- Conversational memory with summarization to manage long-running chats.

## Key Technical Decisions

- Use of LangGraph to model the agent workflow as a dynamic graph, enabling self-correction and flexible execution paths.
- Use of a **Router** node to classify user intent and direct the workflow accordingly.
- **Hybrid search** combining semantic (ChromaDB) and lexical (BM25) retrieval for improved recall.
- **Web search fallback** using Tavily to augment the local knowledge base when necessary.
- **Sentence window retrieval** to provide more coherent and complete context to the LLM.
- Reranking of retrieved context using a CrossEncoder model for improved answer relevance.
- Centralized configuration and logging for maintainability.
- Use of ChromaDB for scalable vector storage and retrieval.
- Support for multi-language queries and document indexing.

## Design Patterns

- Graph-based workflow orchestration (LangGraph).
- **Router Pattern** for conditional workflow execution based on query type (RAG vs. conversational).
- **ToolNode Pattern** for both local context retrieval and external web search (Tavily).
- **Self-correction loop** that includes query reframing and a **Fallback Pattern** to web search.
- Modular separation of concerns: message handling, agent orchestration, document indexing, and storage.

## Component Relationships

- User interacts with the Telegram bot, which communicates with the webhook server.
- The webhook server passes messages to the MessageHandler.
- The MessageHandler invokes the RAG agent orchestrator, which manages the workflow nodes.
- The agent's **Router** node first classifies the query.
- For RAG queries, the **ContextRetrieverTool** interacts with the VectorStoreManager (for hybrid search) and ChromaDB.
- If local context is insufficient, a **Tavily tool** is called for web search.
- For conversational queries, a dedicated **Conversational Handler** node uses chat history to generate a response.
- External models (Gemini LLM, Sentence Transformers, CrossEncoder) are called at various workflow stages.
- Document uploads are indexed and stored for retrieval.

## Critical Implementation Paths

- The agent's conditional workflow graph:
  - **Routing** → (if RAG) **Hybrid Retrieval** → **Reranking** → **Evaluation** → (optional: **Web Search** or **Reframing**) → **Answer Generation**.
  - **Routing** → (if conversational) → **Conversational Handler**.
- Document upload and automatic indexing pipeline.
- Multi-language support and language detection for both queries and documents.
- Robust error handling and graceful fallback when no relevant context is found.
