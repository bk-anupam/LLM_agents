# Stage 1: Builder stage for Python dependencies and ML models
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install build-time dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up a non-root user for security
RUN useradd -m -s /bin/bash -u 1000 user
# Switch to the "user" user
USER user

ENV HOME=/home/user
WORKDIR $HOME/app

# Set user-specific environment variables
ENV HF_HOME=$HOME/.cache/huggingface
ENV PATH=$HOME/.local/bin:$PATH
RUN mkdir -p $HF_HOME

# Install Python dependencies as the non-root user
COPY --chown=user:user requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# Pre-download and cache the embedding and reranker models during the build process.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
print('Caching embedding model: paraphrase-multilingual-mpnet-base-v2'); \
SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2'); \
print('Caching reranker model: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'); \
CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')"
 
 
# Stage 2: Final runtime image
FROM python:3.11-slim
 
# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_PORT=5000
EXPOSE ${APP_PORT}
 
# Install runtime dependencies: gsutil and Node.js
# 1. Install gsutil by copying from the official slim image
# Use the canonical path to avoid issues with symlinks in the source image.
COPY --from=gcr.io/google.com/cloudsdktool/cloud-sdk:slim /usr/lib/google-cloud-sdk/ /google/google-cloud-sdk/
ENV PATH=/google/google-cloud-sdk/bin:$PATH
 
# 2. Install Node.js (for npx) using NodeSource repository for a lightweight install
RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*
 
# Set up the same non-root user
RUN useradd -m -s /bin/bash -u 1000 user
 
# Set user-specific environment variables
ENV HOME=/home/user
ENV HF_HOME=$HOME/.cache/huggingface
ENV PATH=$HOME/.local/bin:$PATH
 
# Switch to the "user" user
USER user
WORKDIR $HOME/app
 
# Copy installed Python packages and cached models from the builder stage
COPY --chown=user:user --from=builder /home/user/.local /home/user/.local
COPY --chown=user:user --from=builder /home/user/.cache/huggingface $HF_HOME
 
# Verify gsutil is available after copying
RUN gsutil version || echo "Warning: gsutil not found"
# Copy the startup script and application code
COPY --chown=user:user startup.sh ./
RUN chmod +x ./startup.sh
COPY --chown=user:user RAG_BOT/ ./RAG_BOT/
 
# The startup script will download the DB and then execute the CMD
ENTRYPOINT ["./startup.sh"]
CMD ["python", "-m", "RAG_BOT.bot"]
