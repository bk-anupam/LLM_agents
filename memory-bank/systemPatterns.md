# System Patterns

This document details the system architecture, key technical decisions, design patterns in use, component relationships, and critical implementation paths.

## Architecture

The system is a Telegram bot application built around a Retrieval-Augmented Generation (RAG) agent using Langchain and LangGraph. The architecture includes:
- Telegram interface for user interaction.
- Flask/Gunicorn webhook server for Telegram API integration.
- Core logic managed by a MessageHandler and centralized configuration.
- RAG agent workflow defined as a graph with nodes for retrieval, reranking, evaluation, query reframing, and answer generation.
- ChromaDB vector store for document embeddings and retrieval.
- Integration with Google Gemini LLM for query analysis, context evaluation, and answer generation.
- Document upload and indexing pipeline for PDFs and HTM files.

## Key Technical Decisions

- Use of LangGraph to model the agent workflow as a dynamic graph, enabling self-correction and flexible execution paths.
- Reranking of retrieved context using a CrossEncoder model for improved answer relevance.
- Centralized configuration and logging for maintainability.
- Use of ChromaDB for scalable vector storage and retrieval.
- Support for multi-language queries and document indexing.

## Design Patterns

- Graph-based workflow orchestration (LangGraph).
- ToolNode pattern for context retrieval.
- Self-correction loop for query reframing and retry.
- Modular separation of concerns: message handling, agent orchestration, document indexing, and storage.

## Component Relationships

- User interacts with the Telegram bot, which communicates with the webhook server.
- The webhook server passes messages to the MessageHandler.
- The MessageHandler invokes the RAG agent orchestrator, which manages the workflow nodes.
- ContextRetrieverTool interacts with the VectorStoreManager and ChromaDB.
- External models (Gemini LLM, Sentence Transformers, CrossEncoder) are called at various workflow stages.
- Document uploads are indexed and stored for retrieval.

## Critical Implementation Paths

- The agent's workflow graph: retrieval → reranking → evaluation → (optional reframing) → answer generation.
- Document upload and automatic indexing pipeline.
- Multi-language support and language detection for both queries and documents.
- Robust error handling and graceful fallback when no relevant context is found.
