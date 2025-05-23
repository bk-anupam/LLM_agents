# Product Context

This document describes the product context, including why this project exists, the problems it solves, how it should work, and user experience goals.

## Problem Solved

- Spiritual seekers interested in Brahmakumaris philosophy and Rajyoga meditation need accurate, context-grounded answers to their questions.
- Manual searching through spiritual texts (murlis) is inefficient and not tailored to the seeker’s context or intent.
- Existing chatbots do not provide answers grounded in authoritative Brahmakumaris documents.

## How it Works

- Users interact with a Telegram bot designed specifically for Brahmakumaris teachings.
- The bot uses a Retrieval-Augmented Generation (RAG) agent to:
  - Retrieve and rerank relevant context from indexed spiritual documents (primarily murlis) stored in ChromaDB.
  - Evaluate and, if needed, reframe queries to ensure the best possible answer.
  - Generate answers using Google Gemini LLM, always grounding responses in the retrieved context.
- Users can upload PDF/HTM murlis, filter queries by date, select language (English/Hindi), and receive answers tailored to Brahmakumaris philosophy and Rajyoga meditation.

## User Experience Goals

- Fast, accurate, and context-grounded answers from the Brahmakumaris perspective.
- Seamless Telegram interface with support for document upload and multi-language queries.
- Clear feedback when context is insufficient or no answer is found.
- Privacy: Uploaded murli documents are strictly used by the chatbot for answering spiritual queries and are not meant for public distribution.
- Robust error handling and graceful fallback in all user interactions.
