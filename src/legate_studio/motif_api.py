"""
Motif Processing API

REST endpoints for submitting and tracking motif processing jobs.

Endpoints:
- POST /motif/api/submit - Submit a transcript for processing
- GET /motif/api/jobs/<job_id> - Get job status and progress
- GET /motif/api/jobs - List user's jobs
- DELETE /motif/api/jobs/<job_id> - Cancel a pending job
"""

import secrets
import logging
from datetime import datetime

from flask import Blueprint, request, jsonify, session, current_app

from .core import login_required, library_required, paid_required

logger = logging.getLogger(__name__)

motif_api_bp = Blueprint('motif_api', __name__, url_prefix='/motif')

# Configuration
MAX_TRANSCRIPT_SIZE = 500 * 1024  # 500KB
SYNC_THRESHOLD = 5000  # Process synchronously if under this size


@motif_api_bp.route('/api/submit', methods=['POST'])
@library_required
@paid_required
def submit_job():
    """Submit a transcript for motif processing.

    Request body (JSON):
    {
        "transcript": "The transcript text...",
        "source_id": "optional-source-identifier",
        "sync": false  // Optional: force sync processing
    }

    Response:
    {
        "job_id": "job-abc123",
        "status": "pending" | "processing" | "completed",
        "message": "Job queued for processing",
        "entry_ids": ["..."]  // Only if sync=true and completed
    }
    """
    from .rag.database import init_db
    from .core import get_api_key_for_user

    user = session.get('user', {})
    user_id = user.get('user_id')

    if not user_id:
        return jsonify({'error': 'User not authenticated'}), 401

    # Check if user has API key configured (platform key for managed tier, or BYOK)
    api_key = get_api_key_for_user(user_id, 'anthropic')
    if not api_key:
        return jsonify({
            'error': 'No Anthropic API key configured. Please add your API key in Settings.',
            'missing_key': True
        }), 400

    data = request.get_json()
    if not data or not data.get('transcript'):
        return jsonify({'error': 'transcript field required'}), 400

    transcript = data['transcript'].strip()
    source_id = data.get('source_id', f"pit-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}")
    force_sync = data.get('sync', False)

    # Validate size
    if len(transcript.encode('utf-8')) > MAX_TRANSCRIPT_SIZE:
        return jsonify({'error': f'Transcript exceeds {MAX_TRANSCRIPT_SIZE // 1024}KB limit'}), 400

    # Generate job ID
    job_id = f"job-{secrets.token_hex(8)}"

    # Decide sync vs async
    is_short = len(transcript) < SYNC_THRESHOLD
    process_sync = force_sync or is_short

    if process_sync:
        # Process synchronously for short content
        from .motif_processor import process_motif_sync
        from .auth import StaleInstallationError

        logger.info(f"Processing job {job_id} synchronously for user {user_id}")

        try:
            result = process_motif_sync(transcript, user_id, source_id)

            return jsonify({
                'job_id': job_id,
                'status': result.get('status', 'failed'),
                'message': 'Processed synchronously',
                'entry_ids': result.get('entry_ids', []),
                'error': result.get('error'),
            })
        except StaleInstallationError:
            logger.warning(f"Stale installation for user {user_id}, needs re-auth")
            return jsonify({
                'error': 'GitHub authorization expired. Please re-authenticate.',
                'needs_reauth': True,
                'reauth_url': '/auth/github-app-login'
            }), 401

    else:
        # Queue for async processing
        db = init_db()  # Shared DB
        db.execute("""
            INSERT INTO processing_jobs (job_id, user_id, input_content, source_id, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """, (job_id, user_id, transcript, source_id))
        db.commit()

        logger.info(f"Job {job_id} queued for user {user_id}")

        return jsonify({
            'job_id': job_id,
            'status': 'pending',
            'message': 'Job queued for processing'
        })


@motif_api_bp.route('/api/jobs/<job_id>', methods=['GET'])
@login_required
@paid_required
def get_job_status(job_id: str):
    """Get job status and progress.

    Response:
    {
        "job_id": "job-abc123",
        "status": "processing",
        "current_stage": "classifying",
        "progress_pct": 45,
        "threads_total": 5,
        "threads_completed": 2,
        "threads_failed": 0,
        "threads": [
            {"thread_id": "thread-001", "status": "extracted", "title": "..."},
            ...
        ],
        "created_at": "...",
        "started_at": "...",
        "result_entry_ids": ["library.concept.xyz", ...]  // when completed
        "error": "..."  // when failed
    }
    """
    from .rag.database import init_db

    user = session.get('user', {})
    user_id = user.get('user_id')

    db = init_db()

    # Get job (verify ownership)
    job = db.execute("""
        SELECT * FROM processing_jobs WHERE job_id = ? AND user_id = ?
    """, (job_id, user_id)).fetchone()

    if not job:
        return jsonify({'error': 'Job not found'}), 404

    job_dict = dict(job)

    # Get threads
    threads = db.execute("""
        SELECT thread_id, status, category, title, correlation_action, entry_id, error_message
        FROM processing_threads
        WHERE job_id = ?
        ORDER BY thread_index
    """, (job_id,)).fetchall()

    job_dict['threads'] = [dict(t) for t in threads]

    # Parse result_entry_ids if present
    if job_dict.get('result_entry_ids'):
        job_dict['result_entry_ids'] = [
            eid.strip() for eid in job_dict['result_entry_ids'].split(',')
            if eid.strip()
        ]
    else:
        job_dict['result_entry_ids'] = []

    # Clean up internal fields
    job_dict.pop('input_content', None)  # Don't send full content back
    job_dict.pop('worker_id', None)
    job_dict.pop('locked_until', None)

    return jsonify(job_dict)


@motif_api_bp.route('/api/jobs', methods=['GET'])
@login_required
@paid_required
def list_jobs():
    """List user's recent jobs.

    Query params:
    - limit: Max jobs to return (default 20, max 100)
    - status: Filter by status (optional)

    Response:
    {
        "jobs": [
            {"job_id": "...", "status": "...", "created_at": "...", ...}
        ]
    }
    """
    from .rag.database import init_db

    user = session.get('user', {})
    user_id = user.get('user_id')

    limit = min(int(request.args.get('limit', 20)), 100)
    status = request.args.get('status')

    db = init_db()

    if status:
        rows = db.execute("""
            SELECT job_id, status, current_stage, progress_pct,
                   threads_total, threads_completed, threads_failed,
                   source_id, created_at, started_at, completed_at, error_message
            FROM processing_jobs
            WHERE user_id = ? AND status = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (user_id, status, limit)).fetchall()
    else:
        rows = db.execute("""
            SELECT job_id, status, current_stage, progress_pct,
                   threads_total, threads_completed, threads_failed,
                   source_id, created_at, started_at, completed_at, error_message
            FROM processing_jobs
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (user_id, limit)).fetchall()

    return jsonify({'jobs': [dict(r) for r in rows]})


@motif_api_bp.route('/api/jobs/<job_id>', methods=['DELETE'])
@login_required
@paid_required
def cancel_job(job_id: str):
    """Cancel a pending job.

    Only works for jobs with status 'pending'.

    Response:
    {
        "success": true,
        "message": "Job cancelled"
    }
    """
    from .rag.database import init_db

    user = session.get('user', {})
    user_id = user.get('user_id')

    db = init_db()

    # Only cancel pending jobs
    cursor = db.execute("""
        UPDATE processing_jobs
        SET status = 'cancelled', updated_at = CURRENT_TIMESTAMP
        WHERE job_id = ? AND user_id = ? AND status = 'pending'
    """, (job_id, user_id))
    db.commit()

    if cursor.rowcount == 0:
        return jsonify({'error': 'Job not found or cannot be cancelled'}), 404

    return jsonify({'success': True, 'message': 'Job cancelled'})


# ============ UI Routes ============

@motif_api_bp.route('/')
@library_required
@paid_required
def index():
    """Motif processing page with job submission and status tracking."""
    from flask import render_template
    from .core import get_api_key_for_user

    user = session.get('user', {})
    user_id = user.get('user_id')

    # Check if user has API key (platform key for managed tier, or BYOK)
    has_api_key = get_api_key_for_user(user_id, 'anthropic') is not None

    return render_template(
        'motif.html',
        title='Motif',
        has_api_key=has_api_key
    )
