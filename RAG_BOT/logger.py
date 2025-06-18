# /home/bk_anupam/code/LLM_agents/RAG_BOT/logger.py
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get log level from environment variable, default to INFO
log_level_name = os.getenv('LOG_LEVEL', 'INFO').upper()

# Map string level name to logging level constant
log_level = logging.getLevelName(log_level_name)

# Validate if the level name was valid, otherwise default to INFO
if not isinstance(log_level, int):
    print(f"Warning: Invalid LOG_LEVEL '{log_level_name}'. Defaulting to INFO.")
    log_level = logging.INFO

# Example Breakdown:
# %(asctime)s: Time of log creation
# %(name)s: Name of the logger (often the module name if using getLogger(__name__))
# %(module)s: Module filename (without extension)
# %(funcName)s: Name of the function/method containing the logging call
# %(lineno)d: Line number where the logging call occurs
# %(levelname)s: Text logging level ('INFO', 'WARNING', etc.)
# %(message)s: The logged message itself
log_format = '%(asctime)s-%(name)s-[%(module)s.%(funcName)s:%(lineno)d]-%(levelname)s - %(message)s'
# # Configure logging
# logging.basicConfig(
#     level=log_level, # Use the determined level
#     # Add %(module)s, %(funcName)s, and %(lineno)d to the format string
#     format=log_format,
#     # Example Breakdown:
#     # %(asctime)s: Time of log creation
#     # %(name)s: Name of the logger (often the module name if using getLogger(__name__))
#     # %(module)s: Module filename (without extension)
#     # %(funcName)s: Name of the function/method containing the logging call
#     # %(lineno)d: Line number where the logging call occurs
#     # %(levelname)s: Text logging level ('INFO', 'WARNING', etc.)
#     # %(message)s: The logged message itself
# )
logger = logging.getLogger(__name__)
logger.setLevel(log_level)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(console_handler)

# File handler
log_file = os.getenv('LOG_FILE', 'rag_bot.log')
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(file_handler)

# Optional: Log the effective level being used
logger.info(f"Logging configured with level: {logging.getLevelName(log_level)} (console and file: {log_file})")

