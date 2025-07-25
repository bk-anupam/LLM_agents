FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
# EXPOSE informs Docker which port the application inside the container will listen on.
ENV APP_PORT=5000
EXPOSE ${APP_PORT}

RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    # Install Google Cloud SDK for gsutil
    apt-transport-https \
    ca-certificates \
    gnupg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
    && apt-get update && apt-get install -y google-cloud-sdk \
    && rm -rf /var/lib/apt/lists/*

# Upgrades the pip package installer to its latest version within the system's Python environment. 
# --no-cache-dir prevents pip from storing downloaded packages in a cache, helping to keep the image size smaller.
RUN pip install --no-cache-dir --upgrade pip

ENV NVM_DIR="/opt/nvm"
ENV NODE_VERSION="22.14.0" 

# Install nvm, Node.js, and npm
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
    nvm install $NODE_VERSION && \
    nvm use $NODE_VERSION && \
    nvm alias default $NODE_VERSION && \
    nvm cache clear

# Add Node to PATH for all users and sessions
# NVM stores node versions with a 'v' prefix in the directory, e.g., v22.14.0
# This makes node and npm, npx commands globally accessible within the container.
ENV PATH="$NVM_DIR/versions/node/v${NODE_VERSION}/bin:$PATH"

# Set up a new user named "user" with user ID 1000
RUN useradd -m -s /bin/bash -u 1000 user

# Set user-specific environment variables
ENV HOME=/home/user
# Define Hugging Face cache directory for the user
ENV HF_HOME=$HOME/.cache/huggingface 
# Prepend user's local bin to PATH for user-installed packages
ENV PATH=$HOME/.local/bin:$PATH 

# Switch to the "user" user
USER user

# Set the working directory to the user's home directory app folder
WORKDIR $HOME/app    

# Create the HF_HOME directory as the user to ensure correct permissions
RUN mkdir -p $HF_HOME

# Always specify the `--chown=user` with `ADD` and `COPY` to ensure the new files are owned by your user.
COPY --chown=user:user requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Pre-download and cache the embedding and reranker models during the build process.
# This avoids slow, rate-limited downloads on container startup and makes them part of the image.
# The model names here MUST match the ones set in your Cloud Run environment variables.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
print('Caching embedding model: paraphrase-multilingual-mpnet-base-v2'); \
SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2'); \
print('Caching reranker model: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'); \
CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')"

# Copy the startup script and make it executable
COPY --chown=user:user startup.sh ./
RUN chmod +x ./startup.sh

# Copy your application code
COPY --chown=user:user RAG_BOT/ ./RAG_BOT/

# The startup script will download the DB and then execute the CMD.
ENTRYPOINT ["./startup.sh"]
CMD ["python", "-m", "RAG_BOT.bot"]