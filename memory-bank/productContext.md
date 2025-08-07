# Product Context

This document describes the product context, including why this project exists, the problems it solves, how it should work, and user experience goals.

## Problem Solved

- Spiritual seekers interested in Brahmakumaris philosophy and Rajyoga meditation need accurate, context-grounded answers to their questions.
- Manual searching through spiritual texts (murlis) is inefficient and not tailored to the seeker’s context or intent.
- Existing chatbots do not provide answers grounded in authoritative Brahmakumaris documents.

## How it Works

- Users interact with a Telegram bot designed specifically for Brahmakumaris teachings.
- The bot uses a stateful, conversational Retrieval-Augmented Generation (RAG) agent that can:
  - **Intelligently Route Queries:** First, it determines if a user's message is a question for the knowledge base or a conversational query about the chat history (e.g., "summarize our chat").
  - **Retrieve and Rerank Context:** For knowledge questions, it performs a hybrid search (semantic and lexical) on indexed spiritual documents (primarily murlis) stored in ChromaDB. To ensure comprehensive context, it retrieves not just the best matching snippet but also the surrounding sentences. The results are then reranked for relevance.
  - **Evaluate and Self-Correct:** The agent evaluates if the retrieved context is sufficient. If not, it can reframe the user's query or fall back to searching the web using Tavily to find an answer.
  - **Generate Grounded Answers:** It generates answers using Google Gemini LLM, always grounding responses in the retrieved context (from local documents or the web).
  - **Maintain Memory:** The agent remembers the conversation, allowing for follow-up questions and meta-discussion about the chat.
- Users can upload PDF/HTM murlis, filter queries by date, select language (English/Hindi), and receive answers tailored to Brahmakumaris philosophy and Rajyoga meditation.

## User Experience Goals

- Fast, accurate, and context-grounded answers from the Brahmakumaris perspective.
- Seamless Telegram interface with support for document upload and multi-language queries.
- A stateful and helpful conversational partner that remembers the context of the discussion.
- Clear feedback when context is insufficient or no answer is found.
- Privacy: Uploaded murli documents are strictly used by the chatbot for answering spiritual queries and are not meant for public distribution.
- Robust error handling and graceful fallback in all user interactions.
