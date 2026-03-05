"""
Background Worker for Motif Processing Jobs

Polls the processing_jobs table and processes motifs through:
1. Parse - Split transcript into threads
2. Classify - Categorize via Claude API
3. Correlate - Check for similar existing entries
4. Extract - Generate markdown artifacts
5. Write - Commit to user's Library repo

Can run as:
- Daemon thread within the Flask app (for single-process deployments)
- Separate process via worker_main.py (for Fly.io multi-process)
"""

import logging
import secrets
import threading
import time
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Worker configuration
POLL_INTERVAL = 2  # seconds between polls
LOCK_DURATION = 300  # 5 minutes lock timeout
MAX_RETRIES = 3


class MotifWorker:
    """Background worker that processes motif jobs."""

    def __init__(self, app):
        """Initialize the worker.

        Args:
            app: Flask application instance (for app context)
        """
        self.app = app
        self.worker_id = f"worker-{secrets.token_hex(4)}"
        self._running = False
        self._thread = None

    def start(self):
        """Start the worker in a background thread."""
        if self._running:
            logger.warning(f"Worker {self.worker_id} already running")
            return

        # Clean up stale jobs before starting
        self._cleanup_stale_jobs()

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"Motif worker {self.worker_id} started")

    def _cleanup_stale_jobs(self):
        """Mark very old processing jobs as failed to prevent infinite retries."""
        from .rag.database import init_db

        try:
            db = init_db()

            # Mark jobs stuck in 'processing' for more than 1 hour as failed
            # These are likely from crashed workers or database lock issues
            one_hour_ago = (datetime.utcnow() - timedelta(hours=1)).isoformat()

            cursor = db.execute("""
                UPDATE processing_jobs
                SET status = 'failed',
                    error_message = 'Job timed out - stuck in processing state',
                    updated_at = CURRENT_TIMESTAMP
                WHERE status = 'processing'
                  AND updated_at < ?
            """, (one_hour_ago,))
            db.commit()

            if cursor.rowcount > 0:
                logger.info(f"Cleaned up {cursor.rowcount} stale processing jobs")
        except Exception as e:
            logger.warning(f"Failed to cleanup stale jobs: {e}")

    def stop(self):
        """Signal the worker to stop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
            logger.info(f"Motif worker {self.worker_id} stopped")

    def _run_loop(self):
        """Main worker loop."""
        consecutive_errors = 0

        while self._running:
            try:
                with self.app.app_context():
                    job = self._claim_job()
                    if job:
                        consecutive_errors = 0
                        self._process_job(job)
                    else:
                        time.sleep(POLL_INTERVAL)

            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Worker error: {e}")

                # Back off on repeated errors
                backoff = min(POLL_INTERVAL * (2 ** consecutive_errors), 60)
                time.sleep(backoff)

    def _claim_job(self) -> Optional[dict]:
        """Attempt to claim an available job using optimistic locking.

        Uses an atomic UPDATE with subquery to prevent race conditions
        when multiple workers are running.

        Returns:
            Job dict if claimed, None if no jobs available
        """
        from .rag.database import init_db

        db = init_db()  # Shared DB for job queue
        now = datetime.utcnow()
        lock_until = now + timedelta(seconds=LOCK_DURATION)

        # Find and claim a job atomically
        # Jobs are claimable if:
        # - status is 'pending', or
        # - status is 'processing' but lock has expired (crashed worker)
        cursor = db.execute("""
            UPDATE processing_jobs
            SET worker_id = ?,
                locked_until = ?,
                status = 'processing',
                started_at = COALESCE(started_at, ?),
                updated_at = ?
            WHERE job_id = (
                SELECT job_id FROM processing_jobs
                WHERE (status = 'pending')
                   OR (status = 'processing' AND (locked_until IS NULL OR locked_until < ?))
                ORDER BY created_at ASC
                LIMIT 1
            )
        """, (self.worker_id, lock_until, now, now, now))

        db.commit()

        if cursor.rowcount == 0:
            return None

        # Fetch the claimed job
        row = db.execute("""
            SELECT * FROM processing_jobs
            WHERE worker_id = ? AND status = 'processing'
            ORDER BY updated_at DESC
            LIMIT 1
        """, (self.worker_id,)).fetchone()

        if row:
            logger.info(f"Worker {self.worker_id} claimed job {row['job_id']} for user {row['user_id']}")
            return dict(row)

        return None

    def _process_job(self, job: dict):
        """Process a single job through all stages.

        Args:
            job: Job dict from database
        """
        from .motif_processor import MotifProcessor

        job_id = job['job_id']
        user_id = job['user_id']

        if not user_id:
            logger.error(f"Job {job_id} has no user_id, marking as failed")
            self._mark_job_failed(job_id, "Job has no user_id")
            return

        try:
            logger.info(f"Processing job {job_id} for user {user_id}")

            # Set up lock renewal thread
            stop_renewal = threading.Event()
            renewal_thread = threading.Thread(
                target=self._lock_renewal_loop,
                args=(job_id, stop_renewal),
                daemon=True
            )
            renewal_thread.start()

            try:
                processor = MotifProcessor(job_id, user_id, self.app)
                processor.process(
                    job['input_content'],
                    job.get('source_id')
                )
            finally:
                # Stop lock renewal
                stop_renewal.set()
                renewal_thread.join(timeout=5)

            logger.info(f"Job {job_id} completed successfully")

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")

            # Check if we should retry
            retry_count = job.get('retry_count', 0)
            if retry_count < MAX_RETRIES and self._is_retryable(e):
                self._mark_job_for_retry(job_id, retry_count + 1, str(e))
            else:
                self._mark_job_failed(job_id, str(e))

    def _lock_renewal_loop(self, job_id: str, stop_event: threading.Event):
        """Periodically renew the job lock to prevent timeout.

        Args:
            job_id: The job ID to renew lock for
            stop_event: Event to signal when to stop
        """
        renewal_interval = LOCK_DURATION // 2  # Renew at half the lock duration

        while not stop_event.wait(renewal_interval):
            try:
                self._renew_lock(job_id)
            except Exception as e:
                logger.warning(f"Failed to renew lock for {job_id}: {e}")

    def _renew_lock(self, job_id: str):
        """Renew the job lock to prevent timeout.

        Args:
            job_id: The job ID to renew lock for
        """
        from .rag.database import init_db

        db = init_db()
        lock_until = datetime.utcnow() + timedelta(seconds=LOCK_DURATION)

        db.execute("""
            UPDATE processing_jobs
            SET locked_until = ?, updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ? AND worker_id = ?
        """, (lock_until, job_id, self.worker_id))
        db.commit()

    def _is_retryable(self, error: Exception) -> bool:
        """Determine if an error is retryable.

        Args:
            error: The exception that occurred

        Returns:
            True if the error is transient and should be retried
        """
        error_str = str(error).lower()

        # These errors are NOT retryable - they indicate permanent failures
        permanent_errors = [
            'unique constraint',
            'database is locked',  # SQLite contention - backing off won't help if stuck
        ]
        if any(pattern in error_str for pattern in permanent_errors):
            return False

        retryable_patterns = [
            'rate_limit',
            'rate limit',
            'timeout',
            'timed out',
            'connection',
            'temporary',
            'overloaded',
            '529',  # Anthropic overloaded
            '503',  # Service unavailable
        ]
        return any(pattern in error_str for pattern in retryable_patterns)

    def _mark_job_for_retry(self, job_id: str, retry_count: int, error: str):
        """Mark a job for retry.

        Args:
            job_id: The job ID
            retry_count: New retry count
            error: Error message from this attempt
        """
        from .rag.database import init_db

        db = init_db()
        db.execute("""
            UPDATE processing_jobs
            SET status = 'pending',
                retry_count = ?,
                error_message = ?,
                worker_id = NULL,
                locked_until = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE job_id = ?
        """, (retry_count, f"Retry {retry_count}: {error}", job_id))
        db.commit()

        logger.info(f"Job {job_id} marked for retry (attempt {retry_count})")

    def _mark_job_failed(self, job_id: str, error: str):
        """Mark a job as failed.

        Args:
            job_id: The job ID
            error: Error message
        """
        from .rag.database import init_db

        db = init_db()
        db.execute("""
            UPDATE processing_jobs
            SET status = 'failed',
                error_message = ?,
                updated_at = CURRENT_TIMESTAMP,
                completed_at = CURRENT_TIMESTAMP
            WHERE job_id = ?
        """, (error, job_id))
        db.commit()


def start_worker(app) -> MotifWorker:
    """Start a background worker.

    Args:
        app: Flask application instance

    Returns:
        The started MotifWorker instance
    """
    worker = MotifWorker(app)
    worker.start()
    return worker


def start_worker_if_needed(app):
    """Start the worker if in single-process mode.

    In multi-process deployments (Fly.io with separate worker process),
    this does nothing. The worker is started via worker_main.py instead.

    Args:
        app: Flask application instance
    """
    import os

    # Check if we're the web process or a dedicated worker
    # In Fly.io, FLY_PROCESS_GROUP indicates which process we are
    process_group = os.environ.get('FLY_PROCESS_GROUP', 'app')

    if process_group == 'worker':
        # We're the dedicated worker process, worker_main.py handles this
        return None

    # Check if we should run an in-process worker
    # This is useful for local development or single-process deployments
    run_inline = os.environ.get('MOTIF_WORKER_INLINE', '').lower() in ('1', 'true', 'yes')

    if run_inline:
        logger.info("Starting inline motif worker")
        return start_worker(app)

    return None
