# ---- Builder Stage ----
# Use a full image for building to have access to build tools like compilers.
FROM python:3.10 as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV NVM_DIR="/opt/nvm"
ENV NODE_VERSION="22.14.0"
ENV PATH="$NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH"

# Install build-time dependencies (compilers, curl for nvm)
RUN apt-get update && apt-get install -y --no-install-recommends \
     curl \
     gnupg \
     && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
     && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
     && apt-get update && apt-get install -y gcsfuse \
     && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install nvm and Node.js ONCE in the builder stage.
RUN mkdir -p "$NVM_DIR" && \
    # The pipe (|) takes the output from curl and feeds it directly into bash.
    # bash executes the script as if it were run directly from the terminal.
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash && \
    # The line . "$NVM_DIR/nvm.sh" is used to source the nvm.sh script, which sets up the nvm environment in the current shell session.
    # . is a command that sources a file, meaning it reads and executes the commands in the file within the current shell session.
    # When you run a script normally (e.g., bash script.sh), it executes in a subshell, and any environment changes or variables set 
    # within the script are lost when the script finishes running.
    # By sourcing nvm.sh, the script makes the nvm command and its functions available in the current shell session. This is necessary 
    # because the nvm installation script only sets up the environment for the shell session in which it's run. By sourcing nvm.sh, 
    # the script ensures that nvm is available for the subsequent commands, such as nvm install, nvm use, etc.
    . "$NVM_DIR/nvm.sh" && \
    nvm install "$NODE_VERSION" && \
    nvm alias default "$NODE_VERSION" && \
    nvm cache clear && \
    # This find command cleans up nvm to reduce image size
    find "$NVM_DIR" -name "test" -o -name "benchmark" -o -name "docs" -o -name "examples" | xargs rm -rf

# Set up working directory, copy requirements, and install Python packages
WORKDIR /app
COPY requirements.txt ./
# Upgrades the pip package installer to its latest version within the system's Python environment. 
# --no-cache-dir prevents pip from storing downloaded packages in a cache, helping to keep the image size smaller.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire application source code into the builder
COPY . .

# ---- Final Stage ----
# Use a slim image for the final application to reduce size and attack surface.
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_PORT=5000
EXPOSE ${APP_PORT}
ENV NVM_DIR="/opt/nvm"
ENV NODE_VERSION="22.14.0"
ENV PATH="/home/user/app/.venv/bin:$NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH"
ENV HF_HOME="/home/user/.cache/huggingface"

# Install only essential runtime OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set up non-root user for security
RUN useradd -m -s /bin/bash -u 1000 user
WORKDIR /home/user/app

# Copy the installed NVM/Node and Python packages from the builder stage
COPY --from=builder --chown=user:user ${NVM_DIR} ${NVM_DIR}
COPY --from=builder --chown=user:user /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copy your application code from the builder stage
COPY --from=builder --chown=user:user /app .

# Create the HF_HOME directory and set permissions
RUN mkdir -p ${HF_HOME} && chown -R user:user /home/user

# Switch to the non-root user
USER user

# Command to run the application. Your README mentions Gunicorn, which is a great choice for production.
# To use this, your bot.py needs to expose a Flask 'app' object.
# CMD ["gunicorn", "--bind", "0.0.0.0:${APP_PORT}", "RAG_BOT.bot:app"]
CMD ["python", "-m", "RAG_BOT.bot"]

# This command mounts BOTH the data and the chroma_db directories using bind mount
# docker run -p 5000:5000 --env-file ./RAG_BOT/.env \
# 	 -e HF_HOME=/home/user/.cache/huggingface \
#    -v "$(pwd)/RAG_BOT/data:/home/user/app/RAG_BOT/data:ro" \
#    -v "$(pwd)/RAG_BOT/chroma_db:/home/user/app/RAG_BOT/chroma_db" \
#    -v "$HOME/.cache/huggingface:/home/user/.cache/huggingface" \
#    -t rag_bot_image:latest