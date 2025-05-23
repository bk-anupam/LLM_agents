# Technical Context

This document provides context on the technologies used, development setup, technical constraints, dependencies, and tool usage patterns.

## Technologies Used

- Python
- Langchain & LangGraph
- Google Generative AI (Gemini)
- ChromaDB
- Sentence Transformers (via langchain-huggingface and sentence-transformers)
- pyTelegramBotAPI
- Flask
- Gunicorn

## Development Setup

- Clone the repository and navigate to the project directory.
- Create a virtual environment and activate it.
- Install dependencies from `requirements.txt`.
- Configure environment variables in a `.env` file (Telegram, Gemini, paths, model config).
- Start the bot using Docker, Gunicorn, or Flask for development.
- The bot will index documents from the data directory on startup.

## Technical Constraints

- Requires a publicly accessible HTTPS endpoint for Telegram webhooks.
- Designed for deployment with Docker or Gunicorn for production.
- Relies on Google Gemini API and ChromaDB for core functionality.

## Dependencies

- Python
- langchain
- langgraph
- google-generativeai
- chromadb
- sentence-transformers
- pyTelegramBotAPI
- Flask
- Gunicorn

## Tool Usage Patterns

- Development and orchestration use Python, Langchain, and LangGraph.
- Document embeddings and reranking use Sentence Transformers and CrossEncoder.
- Telegram bot interactions handled via pyTelegramBotAPI.
- Webhooks managed by Flask and Gunicorn.
- Vector storage and retrieval via ChromaDB.
- Configuration and logging are centralized for maintainability.
