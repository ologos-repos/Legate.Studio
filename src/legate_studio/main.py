"""
Legate Studio - Package entry point

This module provides the main() entry point for the legate-studio CLI script.
It creates and runs the Flask application.
"""
from legate_studio.core import create_app
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Entry point for the legate-studio CLI script."""
    load_dotenv()
    app = create_app()

    port = int(os.getenv('PORT', 8000))
    debug = os.getenv('FLASK_ENV') == 'development'

    logger.info(f"Starting Legate Studio on port {port} (debug={debug})")
    app.run(host='0.0.0.0', port=port, debug=debug)


# Create app at module level so gunicorn can use it:  gunicorn legate_studio.main:app
app = None  # Initialized lazily via main() or directly


if __name__ == '__main__':
    main()
