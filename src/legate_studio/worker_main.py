"""
Worker Process Entry Point for Fly.io

This module is run as a separate process in Fly.io deployments:
    python -m legate_studio.worker_main

It creates a Flask app context and runs the MotifWorker to process
jobs from the queue.

Configuration via environment:
- FLY_PROCESS_GROUP: Set to 'worker' by Fly.io
- WORKER_POLL_INTERVAL: Seconds between polls (default: 2)
- WORKER_LOCK_DURATION: Lock timeout in seconds (default: 300)
"""

import os
import signal
import sys
import time
import logging

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Load environment
from dotenv import load_dotenv
load_dotenv()


def main():
    """Main entry point for the worker process."""
    logger.info("=" * 60)
    logger.info("Legato.Pit Motif Processing Worker")
    logger.info("=" * 60)

    # Import app and worker
    from legate_studio.core import create_app
    from legate_studio.worker import MotifWorker

    # Create Flask app for context
    app = create_app()

    # Create worker
    worker = MotifWorker(app)

    # Graceful shutdown handling
    shutdown_requested = False

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            logger.warning("Force shutdown requested")
            sys.exit(1)
        shutdown_requested = True
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        worker.stop()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Start worker
    logger.info("Starting worker...")
    worker.start()

    # Keep running until shutdown
    try:
        while not shutdown_requested:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        shutdown_requested = True

    # Stop worker
    logger.info("Stopping worker...")
    worker.stop()

    logger.info("Worker shutdown complete")


if __name__ == '__main__':
    main()
