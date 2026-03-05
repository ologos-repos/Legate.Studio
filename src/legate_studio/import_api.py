"""
Markdown Import API

Endpoints for importing, classifying, and reviewing markdown files from a ZIP.

Flow:
1. POST /import/upload - Upload ZIP file, get job_id
2. POST /import/jobs/<job_id>/classify - Run classification
3. GET /import/jobs/<job_id> - Review classified files
4. PUT /import/jobs/<job_id>/files/<idx> - Edit a file's classification (optional)
5. POST /import/jobs/<job_id>/confirm - Confirm and write to Library
"""

import logging
from dataclasses import asdict

from flask import Blueprint, request, jsonify, session, render_template

from .core import login_required, library_required, paid_required
from .markdown_importer import (
    MarkdownImporter,
    store_job,
    get_job,
    delete_job,
    list_user_jobs,
    ImportedFile,
)

logger = logging.getLogger(__name__)

import_api_bp = Blueprint('import_api', __name__, url_prefix='/import')

# Configuration
MAX_ZIP_SIZE = 10 * 1024 * 1024  # 10MB


def file_to_dict(f: ImportedFile) -> dict:
    """Convert ImportedFile to JSON-serializable dict."""
    return {
        'original_path': f.original_path,
        'category': f.category,
        'title': f.title,
        'description': f.description,
        'domain_tags': f.domain_tags,
        'key_phrases': f.key_phrases,
        'needs_chord': f.needs_chord,
        'chord_name': f.chord_name,
        'chord_scope': f.chord_scope,
        'entry_id': f.entry_id,
        'target_path': f.target_path,
        'status': f.status,
        'error': f.error,
        # Include content preview for review
        'content_preview': f.existing_body[:500] + ('...' if len(f.existing_body) > 500 else ''),
        'has_existing_frontmatter': bool(f.existing_frontmatter),
    }


def job_to_dict(job, include_files=True) -> dict:
    """Convert ImportJob to JSON-serializable dict."""
    result = {
        'job_id': job.job_id,
        'user_id': job.user_id,
        'created_at': job.created_at,
        'status': job.status,
        'error': job.error,
        'file_count': len(job.files),
        'classified_count': sum(1 for f in job.files if f.status == 'classified'),
        'written_count': sum(1 for f in job.files if f.status == 'written'),
        'error_count': sum(1 for f in job.files if f.status == 'error'),
    }
    if include_files:
        result['files'] = [file_to_dict(f) for f in job.files]
    return result


# ============ UI Routes ============

@import_api_bp.route('/')
@library_required
@paid_required
def index():
    """Import page with upload form and job list."""
    user = session.get('user', {})
    user_id = user.get('user_id')

    jobs = list_user_jobs(user_id)
    jobs_data = [job_to_dict(j, include_files=False) for j in jobs]

    return render_template(
        'import.html',
        title='Import Markdown',
        jobs=jobs_data,
    )


# ============ API Routes ============

@import_api_bp.route('/api/upload', methods=['POST'])
@library_required
@paid_required
def upload_zip():
    """Upload a ZIP file of markdown files.

    Request: multipart/form-data with 'file' field containing ZIP
    Optional form fields:
    - category_mode: 'auto' (classify into existing) or 'new' (use new category)
    - new_category: name for new category (when category_mode='new')

    Response:
    {
        "job_id": "import-abc123",
        "status": "pending",
        "file_count": 15,
        "message": "Upload successful. Run /classify to process."
    }
    """
    user = session.get('user', {})
    user_id = user.get('user_id')

    if not user_id:
        return jsonify({'error': 'User not authenticated'}), 401

    # Check for file
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'No file selected'}), 400

    # Validate file type
    if not file.filename.lower().endswith('.zip'):
        return jsonify({'error': 'File must be a ZIP archive'}), 400

    # Read and validate size
    zip_content = file.read()
    if len(zip_content) > MAX_ZIP_SIZE:
        return jsonify({
            'error': f'File too large. Maximum size is {MAX_ZIP_SIZE // (1024*1024)}MB.'
        }), 400

    if len(zip_content) == 0:
        return jsonify({'error': 'Empty file'}), 400

    # Get category mode options
    category_mode = request.form.get('category_mode', 'auto')  # 'auto' or 'new'
    new_category = request.form.get('new_category', '').strip()

    # Validate new category if specified
    if category_mode == 'new' and not new_category:
        return jsonify({'error': 'New category name required when using new category mode'}), 400

    try:
        importer = MarkdownImporter(user_id)
        job = importer.create_job(zip_content)

        # Store category options on the job for use during classification
        job.category_mode = category_mode
        job.new_category = new_category

        store_job(job)

        if len(job.files) == 0:
            return jsonify({
                'error': 'No markdown files found in ZIP'
            }), 400

        logger.info(f"User {user_id} uploaded ZIP with {len(job.files)} files, job {job.job_id}, category_mode={category_mode}")

        return jsonify({
            'job_id': job.job_id,
            'status': job.status,
            'file_count': len(job.files),
            'category_mode': category_mode,
            'new_category': new_category if category_mode == 'new' else None,
            'message': 'Upload successful. Call /classify to process files.',
        })

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@import_api_bp.route('/api/jobs', methods=['GET'])
@login_required
@paid_required
def list_jobs():
    """List user's import jobs.

    Response:
    {
        "jobs": [
            {"job_id": "...", "status": "...", "file_count": 15, ...}
        ]
    }
    """
    user = session.get('user', {})
    user_id = user.get('user_id')

    jobs = list_user_jobs(user_id)
    return jsonify({
        'jobs': [job_to_dict(j, include_files=False) for j in jobs]
    })


@import_api_bp.route('/api/jobs/<job_id>', methods=['GET'])
@login_required
@paid_required
def get_job_details(job_id: str):
    """Get job details including all files.

    Response:
    {
        "job_id": "import-abc123",
        "status": "classified",
        "files": [
            {
                "original_path": "notes/idea.md",
                "category": "concept",
                "title": "My Idea",
                ...
            }
        ]
    }
    """
    user = session.get('user', {})
    user_id = user.get('user_id')

    job = get_job(job_id, user_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    return jsonify(job_to_dict(job))


@import_api_bp.route('/api/jobs/<job_id>/classify', methods=['POST'])
@library_required
@paid_required
def classify_job(job_id: str):
    """Run classification on all files in the job.

    Uses the category_mode set during upload:
    - 'auto': Claude classifies into existing categories
    - 'new': All files assigned to the new_category

    Response:
    {
        "job_id": "import-abc123",
        "status": "classified",
        "classified_count": 14,
        "error_count": 1,
        "files": [...]
    }
    """
    from .core import get_api_key_for_user

    user = session.get('user', {})
    user_id = user.get('user_id')

    job = get_job(job_id, user_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    if job.status not in ('pending', 'classified'):
        return jsonify({'error': f'Job cannot be classified in {job.status} state'}), 400

    # Only need API key for auto mode
    category_mode = getattr(job, 'category_mode', 'auto')
    if category_mode == 'auto':
        api_key = get_api_key_for_user(user_id, 'anthropic')
        if not api_key:
            return jsonify({
                'error': 'No Anthropic API key configured. Please add your API key in Settings.',
                'missing_key': True
            }), 400

    try:
        importer = MarkdownImporter(user_id)
        importer.classify_job(job)
        store_job(job)  # Update stored job

        logger.info(f"Job {job_id} classified: {job.status}")

        return jsonify(job_to_dict(job))

    except Exception as e:
        logger.error(f"Classification failed for {job_id}: {e}")
        return jsonify({'error': f'Classification failed: {str(e)}'}), 500


@import_api_bp.route('/api/jobs/<job_id>/files/<int:file_idx>', methods=['PUT'])
@login_required
@paid_required
def update_file_classification(job_id: str, file_idx: int):
    """Update a file's classification before confirming.

    Request body (JSON):
    {
        "category": "reflection",
        "title": "Updated Title",
        "domain_tags": ["tag1", "tag2"],
        "needs_chord": false
    }

    Only provided fields will be updated.
    """
    user = session.get('user', {})
    user_id = user.get('user_id')

    job = get_job(job_id, user_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    if file_idx < 0 or file_idx >= len(job.files):
        return jsonify({'error': 'File index out of range'}), 404

    file = job.files[file_idx]

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    # Update allowed fields
    if 'category' in data:
        file.category = data['category'].lower()
    if 'title' in data:
        file.title = data['title']
    if 'description' in data:
        file.description = data['description']
    if 'domain_tags' in data:
        file.domain_tags = data['domain_tags']
    if 'key_phrases' in data:
        file.key_phrases = data['key_phrases']
    if 'needs_chord' in data:
        file.needs_chord = data['needs_chord']
    if 'chord_name' in data:
        file.chord_name = data['chord_name']
    if 'chord_scope' in data:
        file.chord_scope = data['chord_scope']

    # Regenerate derived fields
    from .markdown_importer import generate_entry_id, generate_target_path
    file.entry_id = generate_entry_id(file.category, file.title)
    file.target_path = generate_target_path(file.category, file.title)

    store_job(job)

    return jsonify({
        'success': True,
        'file': file_to_dict(file)
    })


@import_api_bp.route('/api/jobs/<job_id>/confirm', methods=['POST'])
@library_required
@paid_required
def confirm_import(job_id: str):
    """Confirm the import and write all classified files to Library.

    Request body (optional JSON):
    {
        "source_id": "my-import-2024"  // Optional custom source identifier
    }

    Response:
    {
        "success": true,
        "entry_ids": ["library.concept.xyz", ...],
        "written_count": 14,
        "error_count": 1
    }
    """
    user = session.get('user', {})
    user_id = user.get('user_id')

    job = get_job(job_id, user_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    if job.status != 'classified':
        return jsonify({
            'error': f'Job must be classified before confirming. Current status: {job.status}'
        }), 400

    # Get optional source_id
    data = request.get_json() or {}
    source_id = data.get('source_id')

    try:
        importer = MarkdownImporter(user_id)
        entry_ids = importer.write_files(job, source_id)
        store_job(job)  # Update stored job

        written = sum(1 for f in job.files if f.status == 'written')
        errors = sum(1 for f in job.files if f.status == 'error')

        logger.info(f"Job {job_id} completed: {written} written, {errors} errors")

        return jsonify({
            'success': True,
            'entry_ids': entry_ids,
            'written_count': written,
            'error_count': errors,
            'job': job_to_dict(job)
        })

    except Exception as e:
        logger.error(f"Import confirm failed for {job_id}: {e}")
        return jsonify({'error': f'Import failed: {str(e)}'}), 500


@import_api_bp.route('/api/jobs/<job_id>', methods=['DELETE'])
@login_required
@paid_required
def delete_job_endpoint(job_id: str):
    """Delete/cancel an import job.

    Response:
    {
        "success": true,
        "message": "Job deleted"
    }
    """
    user = session.get('user', {})
    user_id = user.get('user_id')

    job = get_job(job_id, user_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    delete_job(job_id)

    return jsonify({
        'success': True,
        'message': 'Job deleted'
    })


# ============ Preview Page ============

@import_api_bp.route('/jobs/<job_id>')
@library_required
@paid_required
def preview_job(job_id: str):
    """Preview page for reviewing classified files before import."""
    user = session.get('user', {})
    user_id = user.get('user_id')

    job = get_job(job_id, user_id)
    if not job:
        return render_template(
            'error.html',
            title='Not Found',
            message='Import job not found'
        ), 404

    return render_template(
        'import_preview.html',
        title='Review Import',
        job=job_to_dict(job),
    )
