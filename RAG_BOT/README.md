# RAG_BOT: Telegram RAG Agent to answer spiritual questions

This project implements a Telegram bot powered by a Retrieval-Augmented Generation (RAG) agent built with Langchain and LangGraph. The agent is designed to answer questions based on a collection of spiritual documents stored in a ChromaDB vector store.

## Features

*   **Telegram Interface:** Interact with the RAG agent directly through Telegram.
*   **RAG Pipeline:** Utilizes a sophisticated LangGraph agent for:
    *   **Hybrid Search:** Combines semantic and lexical search for comprehensive context retrieval from local documents.
    *   **Web Search Fallback:** Automatically uses Tavily tools to search the web if local retrieval yields insufficient context.
    *   **Reranking** retrieved context using a CrossEncoder model for improved relevance.
    *   Evaluating the relevance of retrieved context.
    *   Reframing the user's query if the initial context is insufficient.
    *   Generating answers (in a structured JSON format) grounded in the retrieved documents (local or web) using Google's Gemini models.
*   **Vector Store:** Uses ChromaDB to store and query document embeddings.
*   **Document Indexing:**
    *   Upload PDF and HTM documents directly to the Telegram bot for automatic indexing.
    *   Automatic indexing of PDF and HTM documents from a specified data directory on startup.
    *   Language detection for uploaded documents.
*   **Date Filtering:** Supports filtering queries by date using the format `date:YYYY-MM-DD` within the `/query` command or general messages.
*   **Multi-language Support:** Initial support for English and Hindi, with user-selectable language preference via `/language` command and language detection for uploaded documents.
*   **Session Management:** Basic in-memory session handling for conversation context (via `MessageHandler`).
*   **Webhook Deployment:** Designed for deployment using Flask, Gunicorn, and Telegram webhooks.
*   **Configuration:** Centralized configuration management (`config.py`).
*   **Logging:** Structured logging for monitoring and debugging (`logger.py`).
*   **Integration Tests:** Includes tests to verify core functionalities like indexing, retrieval, and agent logic.

## Technology Stack

*   **Python:** Core programming language.
*   **Langchain & LangGraph:** Framework for building the RAG agent and defining the workflow.
*   **Google Generative AI (Gemini):** LLM used for understanding queries, evaluating context, reframing questions, and generating answers.
*   **ChromaDB:** Vector database for storing and retrieving document embeddings.
*   **Sentence Transformers:** (via `langchain-huggingface` and `sentence-transformers`) For generating document embeddings and for reranking (CrossEncoder).
*   **pyTelegramBotAPI:** Library for interacting with the Telegram Bot API.
*   **Flask:** Web framework for handling Telegram webhooks.
*   **Gunicorn:** WSGI HTTP server for running the Flask application.
*   **Tavily MCP:** Used for web search capabilities when local retrieval is insufficient.

## How It Works

Think of the bot as an intelligent assistant that uses a specific process to answer your questions, especially those about the indexed spiritual documents. The agent maintains an internal **AgentState** (a conversational state) that tracks the original query, current query, retrieved documents, evaluation results, and whether web search has been attempted, allowing it to make informed decisions throughout the workflow. Here's the flow:

1.  **Query Analysis:** First, the agent analyzes your question. Does it need to consult the documents, or is it a general question it already knows?
2.  **Smart Retrieval (Hybrid Search):** If documents are needed, the agent performs a hybrid search, combining **semantic search** (using embeddings in ChromaDB) and **lexical search** (using BM25) on the local knowledge base to find the most relevant snippets based on your query.
3.  **Relevance Check:** The agent doesn't just blindly use what it finds. It evaluates if the retrieved information *actually* helps answer your original question.
4.  **Context Reranking:** The initially retrieved documents are then reranked using a more sophisticated model (CrossEncoder) to further refine the context and bring the most relevant passages to the top.
5.  **Relevance Evaluation (Post-Reranking):** The agent evaluates if the *reranked* context is sufficient to answer the original question.
6.  **Self-Correction Loop & Web Search Fallback:**
    *   If the reranked context from the local vector store is still not good enough, the agent smartly rephrases the query and attempts to retrieve information from the web using **Tavily tools**. This acts as a built-in retry mechanism for better results.
    *   This web search is specifically triggered when local retrieval fails to provide sufficient context, ensuring the agent tries external sources before giving up.
7.  **Grounded Generation:** Finally, using the validated and reranked context (from either local documents or web search), the agent generates a clear answer in a structured JSON format. If no relevant context is found even after all attempts, it informs the user gracefully (also in JSON format).

This graph-based approach allows the agent to dynamically decide its path, evaluate its own findings, and even self-correct, leading to more accurate and relevant answers.

### Workflow Diagram

The following diagram visualizes the agent's workflow :

![Agent Workflow](rag_agent_graph.png)
## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd RAG_BOT
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    Create a `.env` file in the project root directory and add the following variables:
    ```dotenv
    # Telegram
    TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN"
    WEBHOOK_URL="YOUR_PUBLIC_HTTPS_URL_FOR_WEBHOOK" # e.g., from ngrok or your deployment

    # Google Gemini
    GEMINI_API_KEY="YOUR_GOOGLE_API_KEY"

    # Tavily (for web search fallback)
    TAVILY_API_KEY="YOUR_TAVILY_API_KEY"

    # Paths (adjust if needed)
    VECTOR_STORE_PATH="./chroma_db"       # Default path for ChromaDB
    DATA_PATH="./data"                    # Directory for documents to be indexed on startup
    # INDEXED_DATA_PATH="./indexed_data"  # (Optional) Path to move indexed files, if implemented

    # Agent/Model Config (adjust defaults in config.py or override here)
    # LLM_MODEL_NAME="gemini-1.5-flash-latest" # Or another compatible Gemini model
    # JUDGE_LLM_MODEL_NAME="gemini-1.5-flash-latest" # LLM for evaluating context/responses
    # EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2"
    # RERANKER_MODEL_NAME="cross-encoder/ms-marco-MiniLM-L-6-v2" # Or other CrossEncoder model
    # TEMPERATURE=0.1
    # INITIAL_RETRIEVAL_K=20 # Number of documents to initially retrieve before reranking (semantic)
    # BM25_TOP_K=10          # Number of documents to retrieve using BM25 (lexical)
    # BM25_MAX_CORPUS_SIZE=10000 # Max corpus size for BM25 indexing
    # RERANK_TOP_N=5         # Number of documents to keep after reranking
    # SEARCH_TYPE="similarity" # Or "mmr" (for ChromaDB semantic search)
    # SEMANTIC_CHUNKING=True # Or False
    # LANGUAGE="en" # Default language for the bot (e.g., 'en', 'hi')
    # LOG_LEVEL="INFO"
    # MAX_CONVERSATION_HISTORY=10
    # MAX_RECON_MURLIS=5 # Max Murlis to reconstruct from chunks
    # MAX_CHUNKS_PER_MURLI_RECON=20 # Max chunks to fetch for a single Murli reconstruction
    # MAX_CHUNKS_FOR_DATE_FILTER=40 # Max chunks to consider for date filtering
    # ASYNC_OPERATION_TIMEOUT=60 # Timeout for async operations in seconds
    # PORT=5000 # Port for Flask app
    ```
    *   Replace placeholders with your actual credentials and desired settings.
    *   Ensure the `WEBHOOK_URL` is a publicly accessible HTTPS URL pointing to where your Flask app will run. Tools like `ngrok` can be useful for local development.

## Usage

1.  **Start the Bot:**
    Ensure your `.env` file is configured. If using Docker (recommended for Gunicorn):
    ```bash
    docker build -t rag-bot .
    docker run -p 5000:5000 --env-file .env rag-bot
    ```
    Alternatively, to run directly with Gunicorn (if installed locally):
    ```bash
    gunicorn -b 0.0.0.0:5000 bot:app
    ```
    Or for development with Flask's built-in server:
    ```bash
    python bot.py
    ```
    This will start the application server and set up the Telegram webhook. The bot will also attempt to index any documents found in the `DATA_PATH` directory.

2.  **Interact with the Bot on Telegram:**
    *   Find your bot on Telegram.
    *   Send `/start` to initiate interaction.
    *   Send `/help` to see available commands.
    *   **Set Language:** Use `/language hindi` or `/language english` to set your preferred language for bot responses.
    *   **Upload Documents:** Send PDF or HTM documents directly to the chat to have them indexed. The bot will attempt to detect the document's language and index it.
    *   **Query Documents:**
        *   Use the `/query` command: `/query What is the essence of the Murli?`
        *   Query with a date filter: `/query Summarize the main points date:1969-01-18`
        *   Send a general message: `What were the main points about soul consciousness on 1969-01-23?` (The agent will attempt retrieval).
    *   **General Questions:** Ask general knowledge questions (e.g., "What is the capital of France?"). The agent should answer directly without using the retrieval tool.

## Running Tests

Navigate to the project root directory and run the integration tests using `unittest`:

```bash
python -m unittest discover -s RAG_BOT/tests/integration -p 'test_*.py'
```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain) for orchestration and LLM integration.
- [ChromaDB](https://www.trychroma.com/) for vector database support.
- [LangGraph](https://github.com/langgraph/langgraph) for graph-based workflow management.

[def]: image.png