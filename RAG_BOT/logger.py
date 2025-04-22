import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    # Add %(module)s, %(funcName)s, and %(lineno)d to the format string
    format='%(asctime)s - %(name)s - [%(module)s.%(funcName)s:%(lineno)d] - %(levelname)s - %(message)s'
    # Example Breakdown:
    # %(asctime)s: Time of log creation
    # %(name)s: Name of the logger (often the module name if using getLogger(__name__))
    # %(module)s: Module filename (without extension)
    # %(funcName)s: Name of the function/method containing the logging call
    # %(lineno)d: Line number where the logging call occurs
    # %(levelname)s: Text logging level ('INFO', 'WARNING', etc.)
    # %(message)s: The logged message itself
)
logger = logging.getLogger(__name__)