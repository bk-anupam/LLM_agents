#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

# Function to run on exit
cleanup() {
    echo "SIGTERM received, running final GCS sync..."
    # Ensure PYTHONPATH is set up correctly if your modules are in a src directory
    export PYTHONPATH=/home/user/app/RAG_BOT/src
    python -m RAG_BOT.src.services.gcs_uploader --final-sync
    echo "Final sync complete."
}

# Trap SIGTERM and run the cleanup function
trap cleanup TERM

# The local path inside the container where the vector store will be copied to.
LOCAL_VECTOR_STORE_PATH="/home/user/app/RAG_BOT/chroma_db"

# Check if the GCS_VECTOR_STORE_PATH environment variable is set.
if [ -n "$GCS_VECTOR_STORE_PATH" ]; then
  echo "GCS_VECTOR_STORE_PATH is set to $GCS_VECTOR_STORE_PATH"

  # Create the target directory.
  mkdir -p "$LOCAL_VECTOR_STORE_PATH"

  # Check if the source GCS directory has any files before attempting to copy.
  # `gsutil -q stat ...` returns a non-zero exit code if no files match the pattern.
  if gsutil -q stat "${GCS_VECTOR_STORE_PATH}/**" >/dev/null 2>&1; then
    echo "Downloading vector store from GCS to $LOCAL_VECTOR_STORE_PATH..."
    # Use gsutil to recursively copy the database contents. The -q flag suppresses progress output for cleaner logs.
    gsutil -q -m cp -r "${GCS_VECTOR_STORE_PATH}/"* "$LOCAL_VECTOR_STORE_PATH/"
    echo "Download process finished."
  else
    echo "Source GCS path is empty or does not exist. Skipping download."
  fi
else
  echo "GCS_VECTOR_STORE_PATH is not set. The application will use or create a local vector store."
  # Ensure the directory exists even if not downloading.
  mkdir -p "$LOCAL_VECTOR_STORE_PATH"
fi

# Export the path for the Python application to use. Your Config class will pick this up.
export VECTOR_STORE_PATH="$LOCAL_VECTOR_STORE_PATH"

# Execute the command passed to this script (from the Dockerfile's CMD).
# The `&` runs the command in the background (in a child process), and `wait` waits for it.
# This is crucial for the trap to work correctly.
exec "$@" &

# Wait for the process to exit. $! holds the PID of the last background command.
# The wait command pauses the script's execution until the process with that specific PID has finished.
# When Cloud Run needs to terminate a container instance, it sends a SIGTERM signal to the main process.
# which is the startup shell script in this case. The trap will catch this signal and run the cleanup function.
wait $!
