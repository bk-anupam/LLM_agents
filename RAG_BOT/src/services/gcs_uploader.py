
import asyncio
import subprocess
import argparse
from RAG_BOT.src.config.config import Config
from RAG_BOT.src.logger import logger

class GCSUploaderService:
    _lock = asyncio.Lock()

    def __init__(self, cfg: Config):
        self.config = cfg
        self.vector_store_path = self.config.VECTOR_STORE_PATH
        self.gcs_bucket_path = self.config.GCS_VECTOR_STORE_PATH

    async def sync_vector_store(self):
        """
        Asynchronously syncs the local vector store to the GCS bucket.
        Uses a lock to prevent concurrent sync operations.
        """
        if not self.gcs_bucket_path:
            logger.warning("GCS_VECTOR_STORE_PATH is not configured. Skipping sync.")
            return

        async with self._lock:
            logger.info(f"Starting GCS sync from {self.vector_store_path} to {self.gcs_bucket_path}...")
            try:
                command = [
                    "gsutil",
                    "-m",
                    "rsync",
                    "-r",
                    self.vector_store_path,
                    self.gcs_bucket_path,
                ]
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                stdout, stderr = await process.communicate()

                if process.returncode == 0:
                    logger.info("GCS sync completed successfully.")
                    if stdout:
                        logger.debug(f"gsutil stdout:\n{stdout.decode()}")
                else:
                    logger.error(f"GCS sync failed with return code {process.returncode}.")
                    if stderr:
                        logger.error(f"gsutil stderr:\n{stderr.decode()}")

            except FileNotFoundError:
                logger.error("gsutil command not found. Please ensure the Google Cloud SDK is installed and in the system's PATH.")
            except Exception as e:
                logger.error(f"An unexpected error occurred during GCS sync: {e}")

    def final_sync(self):
        """
        Performs a final, blocking sync to GCS. Intended for use in shutdown hooks.
        """
        if not self.gcs_bucket_path:
            logger.warning("GCS_VECTOR_STORE_PATH is not configured. Skipping final sync.")
            return

        logger.info(f"Starting FINAL GCS sync from {self.vector_store_path} to {self.gcs_bucket_path}...")
        try:
            subprocess.run(
                ["gsutil", "-m", "rsync", "-r", self.vector_store_path, self.gcs_bucket_path],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("Final GCS sync completed successfully.")
        except FileNotFoundError:
            logger.error("gsutil command not found for final sync.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Final GCS sync failed with return code {e.returncode}.")
            logger.error(f"stderr: {e.stderr}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during final GCS sync: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCS Vector Store Uploader")
    parser.add_argument("--final-sync", action="store_true", help="Run a final, blocking sync.")
    args = parser.parse_args()

    if args.final_sync:
        from RAG_BOT.src.config.config import Config
        app_config = Config()
        uploader = GCSUploaderService(app_config)
        uploader.final_sync()
