FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
# Hugging Face Spaces will set the PORT environment variable (typically 7860).
# Your application (bot.py) reads this PORT variable.
# EXPOSE informs Docker which port the application inside the container will listen on.
ENV APP_PORT=8080
EXPOSE ${APP_PORT}

RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \    
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

# In HF docker space The container runs with user ID 1000. To avoid permission issues you should create a user 
# and set its WORKDIR before any COPY or download (https://huggingface.co/docs/hub/en/spaces-sdks-docker)
# Set up a new user named "user" with user ID 1000
RUN useradd -m -s /bin/bash -u 1000 user

# Set user-specific environment variables
ENV HOME=/home/user
# Define Hugging Face cache directory for the user
ENV HF_HOME=$HOME/.cache 
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

# Copy your application code
COPY --chown=user:user RAG_BOT/ ./RAG_BOT/
# If other application files/directories are at the root of your build context, copy them too:
# COPY --chown=user:user other_app_dir/ ./other_app_dir/
# COPY --chown=user:user app_file.py ./

# Command to run the application
# Use Gunicorn as indicated by your Procfile.
# This requires your RAG_BOT/bot.py to have a module-level Flask app instance named 'app'.
# Example: In RAG_BOT/bot.py, after your main_setup_and_run(), add:
# application = main_setup_and_run(create_app_only=True) # if main_setup_and_run can return the app
# Or, ensure a global 'app' variable is assigned the Flask instance.
# CMD ["gunicorn", "--bind", "0.0.0.0:${APP_PORT}", "RAG_BOT.bot:app"]
CMD ["python", "-m", "RAG_BOT.bot"]