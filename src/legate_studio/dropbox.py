"""
Transcript Dropbox for Legato.Pit

Secure transcript upload endpoint that triggers LEGATO processing.
Designed for mobile-first experience.

Security measures:
- Authentication required
- Rate limiting on uploads
- File type validation
- Size limits
- CSRF protection via session
"""
import os
import re
import logging
from datetime import datetime

import requests
from flask import (
    Blueprint, render_template, request, current_app,
    flash, redirect, url_for, jsonify, session
)

from .core import login_required, library_required

logger = logging.getLogger(__name__)

dropbox_bp = Blueprint('dropbox', __name__, url_prefix='/dropbox')

# Configuration
MAX_TRANSCRIPT_SIZE = 500 * 1024  # 500KB
MAX_AUDIO_SIZE = 25 * 1024 * 1024  # 25MB (Whisper API limit)
ALLOWED_EXTENSIONS = {'txt', 'md', 'text'}
ALLOWED_AUDIO_EXTENSIONS = {'webm', 'mp3', 'mp4', 'm4a', 'wav', 'ogg', 'flac'}


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def allowed_audio_file(filename):
    """Check if audio file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS


def sanitize_source_id(source_id):
    """Sanitize source identifier."""
    if not source_id:
        return f"dropbox-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    # Remove any characters that aren't alphanumeric, dash, or underscore
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '-', source_id)
    return sanitized[:100]  # Limit length


def get_category_definitions():
    """Get user category definitions for classifier."""
    from flask import g, session
    from .rag.database import get_user_legato_db, get_user_categories

    db = get_user_legato_db()

    # Get user_id from session or default
    user = session.get('user', {})
    user_id = user.get('user_id', 'default')

    categories = get_user_categories(db, user_id)

    # Format for classifier: list of {name, display_name, description, folder_name}
    return [
        {
            'name': cat['name'],
            'display_name': cat['display_name'],
            'description': cat.get('description', ''),
            'folder_name': cat['folder_name'],
        }
        for cat in categories
    ]


def dispatch_transcript(transcript_text, source_id, user_id=None):
    """
    Dispatch transcript to Legato.Conduct via repository_dispatch.

    Args:
        transcript_text: The transcript content
        source_id: Source identifier
        user_id: User ID for multi-tenant mode

    Returns:
        Tuple of (success: bool, message: str)
    """
    from flask import session
    from .auth import get_user_installation_token

    # In multi-tenant mode, use user's installation token
    user = session.get('user', {})
    user_id = user_id or user.get('user_id')
    org = user.get('username')  # User's GitHub username

    token = get_user_installation_token(user_id, 'library') if user_id else None
    if not token:
        logger.error("No GitHub token available for transcript dispatch")
        return False, "GitHub authorization required for transcript dispatch"

    if not org:
        logger.error("No organization available for transcript dispatch")
        return False, "User organization not found"

    repo = current_app.config['CONDUCT_REPO']

    # Get user-defined categories for dynamic classification
    category_definitions = get_category_definitions()
    logger.info(f"Sending {len(category_definitions)} category definitions to classifier")
    # Debug: Log category names being sent
    category_names = [c['name'] for c in category_definitions]
    logger.info(f"Category names being sent to Conduct: {category_names}")

    # Prepare dispatch payload
    # Include both 'transcript' and 'text' fields for compatibility with Conduct
    # Conduct's classifier may expect either field name depending on version
    payload = {
        'event_type': 'transcript-received',
        'client_payload': {
            'transcript': transcript_text,
            'text': transcript_text,  # Alias for compatibility
            'raw_text': transcript_text,  # For routing.json compatibility
            'source': source_id,
            'category_definitions': category_definitions,  # User-defined categories for classifier
        }
    }

    # Log dispatch details (truncate content for log readability)
    preview = transcript_text[:100] + '...' if len(transcript_text) > 100 else transcript_text
    logger.info(f"Dispatching transcript to Conduct: source={source_id}, length={len(transcript_text)} chars")
    logger.debug(f"Transcript preview: {preview!r}")

    try:
        response = requests.post(
            f'https://api.github.com/repos/{org}/{repo}/dispatches',
            json=payload,
            headers={
                'Authorization': f'Bearer {token}',
                'Accept': 'application/vnd.github+json',
                'X-GitHub-Api-Version': '2022-11-28'
            },
            timeout=15
        )

        if response.status_code == 204:
            logger.info(f"Transcript dispatched successfully: {source_id}")
            return True, "Transcript submitted for processing"
        else:
            logger.error(f"Dispatch failed: {response.status_code} - {response.text}")
            return False, f"Failed to dispatch transcript: {response.status_code}"

    except requests.RequestException as e:
        logger.error(f"Dispatch request failed: {e}")
        return False, "Network error while submitting transcript"


@dropbox_bp.route('/')
@library_required
def index():
    """Transcript upload form."""
    return render_template('dropbox.html', title='Motif Processing')


@dropbox_bp.route('/upload', methods=['POST'])
@login_required
def upload():
    """Handle transcript upload using Pit-native motif processing."""
    from .motif_processor import process_motif_sync
    from .auth import get_user_api_key

    user = session.get('user', {})
    user_id = user.get('user_id')

    # Check if user has API key configured (required for motif processing)
    api_key = get_user_api_key(user_id, 'anthropic') if user_id else None
    if not api_key:
        flash('Motif processing requires an Anthropic API key. Please add your API key in Settings.', 'warning')
        return redirect(url_for('dropbox.index'))

    # Get source identifier
    source_id = sanitize_source_id(request.form.get('source_id', ''))

    transcript_text = None

    # Check for text input first (more common use case)
    text_input = request.form.get('transcript', '').strip()
    if text_input:
        transcript_text = text_input

        if len(transcript_text.encode('utf-8')) > MAX_TRANSCRIPT_SIZE:
            flash(f'Transcript too long. Maximum size is {MAX_TRANSCRIPT_SIZE // 1024}KB.', 'error')
            return redirect(url_for('dropbox.index'))

    # Check for file upload if no text
    elif 'file' in request.files:
        file = request.files['file']

        # Only process if a file was actually selected
        if file.filename:
            if not allowed_file(file.filename):
                flash('Invalid file type. Please upload .txt or .md files.', 'error')
                return redirect(url_for('dropbox.index'))

            # Read file content
            content = file.read()

            if len(content) > MAX_TRANSCRIPT_SIZE:
                flash(f'File too large. Maximum size is {MAX_TRANSCRIPT_SIZE // 1024}KB.', 'error')
                return redirect(url_for('dropbox.index'))

            try:
                transcript_text = content.decode('utf-8')
            except UnicodeDecodeError:
                flash('Could not read file. Please ensure it is UTF-8 encoded text.', 'error')
                return redirect(url_for('dropbox.index'))

            # Use filename as source if not provided
            if not source_id or source_id.startswith('dropbox-'):
                source_id = sanitize_source_id(file.filename.rsplit('.', 1)[0])

    # No content provided
    if not transcript_text:
        flash('Please enter transcript text or upload a file.', 'error')
        return redirect(url_for('dropbox.index'))

    # Process using Pit-native motif processor
    try:
        result = process_motif_sync(transcript_text, user_id, source_id)

        if result.get('success'):
            entry_count = len(result.get('entry_ids', []))
            flash(f'Processed {entry_count} note(s) from transcript. (Source: {source_id})', 'success')
        else:
            flash(f"Processing failed: {result.get('error', 'Unknown error')}", 'error')

    except Exception as e:
        logger.error(f"Motif processing failed: {e}")
        flash(f'Processing failed: {str(e)}', 'error')

    return redirect(url_for('dropbox.index'))


@dropbox_bp.route('/api/debug-categories', methods=['GET'])
@login_required
def api_debug_categories():
    """
    Debug endpoint: Show exactly what category definitions would be sent to Conduct.

    This helps diagnose classification issues by showing:
    - All categories that will be passed to the classifier
    - Whether your new category is included
    - The description (which the classifier uses for context)
    """
    category_definitions = get_category_definitions()

    return jsonify({
        'count': len(category_definitions),
        'category_names': [c['name'] for c in category_definitions],
        'categories': category_definitions,
        'note': 'These are the exact category definitions that will be sent to Conduct for classification'
    })


@dropbox_bp.route('/api/upload', methods=['POST'])
@login_required
def api_upload():
    """
    API endpoint for transcript upload.
    Accepts JSON: {"transcript": "...", "source_id": "..."}
    """
    # Disabled in multi-tenant mode until Pit-native processing is implemented
    if current_app.config.get('LEGATO_MODE') == 'multi-tenant':
        return jsonify({
            'error': 'Motif processing is temporarily disabled while we build secure multi-tenant support.',
            'disabled': True
        }), 503

    data = request.get_json()

    if not data or not data.get('transcript'):
        return jsonify({'error': 'Missing transcript field'}), 400

    transcript_text = data['transcript'].strip()
    source_id = sanitize_source_id(data.get('source_id', ''))

    if len(transcript_text.encode('utf-8')) > MAX_TRANSCRIPT_SIZE:
        return jsonify({'error': f'Transcript exceeds maximum size of {MAX_TRANSCRIPT_SIZE // 1024}KB'}), 400

    success, message = dispatch_transcript(transcript_text, source_id)

    if success:
        return jsonify({
            'success': True,
            'message': message,
            'source_id': source_id
        })
    else:
        return jsonify({'error': message}), 500


@dropbox_bp.route('/api/transcribe', methods=['POST'])
@login_required
def api_transcribe():
    """
    Transcribe audio using OpenAI Whisper.

    Accepts audio file upload via multipart/form-data.
    Returns: {"success": true, "transcript": "..."} or {"error": "..."}
    """
    from .rag.whisper_service import get_whisper_service

    # Check if Whisper service is available
    whisper = get_whisper_service()
    if not whisper:
        return jsonify({
            'error': 'Voice transcription not available. OPENAI_API_KEY not configured.'
        }), 503

    # Check for audio file
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    if not audio_file.filename:
        return jsonify({'error': 'No audio file selected'}), 400

    # Validate file type
    filename = audio_file.filename
    if not allowed_audio_file(filename):
        return jsonify({
            'error': f'Invalid audio format. Supported: {", ".join(ALLOWED_AUDIO_EXTENSIONS)}'
        }), 400

    # Read audio data
    audio_data = audio_file.read()

    if len(audio_data) > MAX_AUDIO_SIZE:
        return jsonify({
            'error': f'Audio file too large. Maximum size is {MAX_AUDIO_SIZE // (1024*1024)}MB.'
        }), 400

    if len(audio_data) == 0:
        return jsonify({'error': 'Empty audio file'}), 400

    logger.info(f"Received audio for transcription: {filename}, {len(audio_data)} bytes")

    # Transcribe
    success, result = whisper.transcribe(audio_data, filename)

    if success:
        return jsonify({
            'success': True,
            'transcript': result
        })
    else:
        return jsonify({'error': result}), 500


@dropbox_bp.route('/api/transcribe/status', methods=['GET'])
@login_required
def api_transcribe_status():
    """
    Check if voice transcription is available.

    Returns: {"available": true/false, "model": "whisper-1"}
    """
    from .rag.whisper_service import get_whisper_service

    whisper = get_whisper_service()
    if whisper:
        return jsonify({
            'available': True,
            'model': whisper.model,
            'max_size_mb': MAX_AUDIO_SIZE // (1024 * 1024)
        })
    else:
        return jsonify({
            'available': False,
            'reason': 'OPENAI_API_KEY not configured'
        })
