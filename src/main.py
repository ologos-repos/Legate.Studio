"""
Legate Studio - Dashboard and Transcript Dropbox
"""
from legato_pit.core import create_app
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create application
app = create_app()

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    debug = os.getenv('FLASK_ENV') == 'development'

    logger.info(f"Starting Legate Studio on port {port} (debug={debug})")
    app.run(host='0.0.0.0', port=port, debug=debug)
