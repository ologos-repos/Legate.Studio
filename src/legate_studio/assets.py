"""
Assets Blueprint - Library Asset Management

Provides API for uploading, listing, and managing assets (images, files)
that are stored in per-category assets folders in the Legate.Library repo.

Asset folder structure:
  category/
    assets/
      image-abc123.png
      diagram-def456.jpg
    note.md  (can reference: ![alt](assets/image-abc123.png))
"""

import os
import re
import secrets
import logging
import mimetypes
from datetime import datetime
from typing import Optional

from flask import Blueprint, request, jsonify, g, send_file, Response
from io import BytesIO

from .core import login_required, library_required, get_user_library_repo
from .auth import get_user_installation_token

logger = logging.getLogger(__name__)

assets_bp = Blueprint('assets', __name__, url_prefix='/library/assets')

# Maximum file size for uploads (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

# Allowed MIME types for uploads
ALLOWED_MIME_TYPES = {
    # Images
    'image/png',
    'image/jpeg',
    'image/gif',
    'image/webp',
    'image/svg+xml',
    # Documents
    'application/pdf',
    # Data
    'text/csv',
    'application/json',
}


def get_db():
    """Get legato database connection for current user."""
    from .rag.database import get_user_legato_db
    return get_user_legato_db()


def get_user_id():
    """Get current user ID."""
    from flask import session
    user = session.get('user')
    if user and user.get('user_id'):
        return user['user_id']
    return None


def generate_asset_id() -> str:
    """Generate a unique asset ID."""
    return f"asset-{secrets.token_hex(6)}"


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename for safe storage.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename with only safe characters
    """
    # Remove path components
    filename = os.path.basename(filename)

    # Replace spaces with hyphens
    filename = filename.replace(' ', '-')

    # Remove unsafe characters, keep alphanumeric, hyphen, underscore, dot
    filename = re.sub(r'[^a-zA-Z0-9._-]', '', filename)

    # Limit length
    name, ext = os.path.splitext(filename)
    if len(name) > 50:
        name = name[:50]

    return f"{name}{ext}" if ext else name


def get_file_extension(mime_type: str) -> str:
    """Get file extension for a MIME type.

    Args:
        mime_type: MIME type string

    Returns:
        File extension (e.g., '.png')
    """
    ext = mimetypes.guess_extension(mime_type)
    if ext:
        return ext
    # Fallbacks for common types
    mime_to_ext = {
        'image/png': '.png',
        'image/jpeg': '.jpg',
        'image/gif': '.gif',
        'image/webp': '.webp',
        'image/svg+xml': '.svg',
        'application/pdf': '.pdf',
        'text/csv': '.csv',
        'application/json': '.json',
    }
    return mime_to_ext.get(mime_type, '')


# ============ API Endpoints ============

@assets_bp.route('/upload', methods=['POST'])
@library_required
def upload_asset():
    """Upload an asset to a category's assets folder.

    Request:
        Form data with:
        - file: The file to upload
        - category: Target category (required)
        - alt_text: Alt text for images (optional)
        - description: Description of the asset (optional)

    Response:
    {
        "success": true,
        "asset_id": "asset-abc123",
        "file_path": "concept/assets/image-abc123.png",
        "markdown_ref": "![alt text](assets/image-abc123.png)"
    }
    """
    from .rag.github_service import create_binary_file, file_exists, create_file

    # Validate request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'No filename'}), 400

    category = request.form.get('category', '').strip().lower()
    if not category:
        return jsonify({'error': 'category is required'}), 400

    alt_text = request.form.get('alt_text', '').strip()
    description = request.form.get('description', '').strip()

    # Read file content
    content = file.read()
    if len(content) > MAX_FILE_SIZE:
        return jsonify({'error': f'File too large. Maximum size is {MAX_FILE_SIZE // 1024 // 1024}MB'}), 400

    # Validate MIME type
    mime_type = file.content_type or mimetypes.guess_type(file.filename)[0]
    if mime_type not in ALLOWED_MIME_TYPES:
        return jsonify({
            'error': f'File type not allowed: {mime_type}',
            'allowed_types': list(ALLOWED_MIME_TYPES)
        }), 400

    # Get user credentials
    user_id = get_user_id()
    if not user_id:
        return jsonify({'error': 'Authentication required'}), 401

    token = get_user_installation_token(user_id, 'library')
    if not token:
        return jsonify({'error': 'GitHub authorization required'}), 401

    library_repo = get_user_library_repo()
    if not library_repo:
        return jsonify({'error': 'Library repo not configured'}), 400

    try:
        db = get_db()

        # Generate asset ID and filename
        asset_id = generate_asset_id()
        original_name = sanitize_filename(file.filename)
        ext = get_file_extension(mime_type) or os.path.splitext(original_name)[1]

        # Create filename: original-name-assetid.ext
        name_part = os.path.splitext(original_name)[0][:30]
        filename = f"{name_part}-{asset_id[-6:]}{ext}"

        # Build file path: category/assets/filename
        file_path = f"{category}/assets/{filename}"

        # Ensure assets folder exists (create .gitkeep if needed)
        assets_folder = f"{category}/assets"
        gitkeep_path = f"{assets_folder}/.gitkeep"
        if not file_exists(library_repo, gitkeep_path, token):
            try:
                create_file(
                    repo=library_repo,
                    path=gitkeep_path,
                    content="# Assets folder for images and files\n",
                    message=f"Create assets folder for {category}",
                    token=token
                )
                logger.info(f"Created assets folder: {assets_folder}")
            except Exception as e:
                # Folder might already exist, continue
                logger.debug(f"Assets folder may already exist: {e}")

        # Upload the file to GitHub
        result = create_binary_file(
            repo=library_repo,
            path=file_path,
            content=content,
            message=f"Add asset: {filename}",
            token=token
        )

        # Store in database
        db.execute("""
            INSERT INTO library_assets
            (asset_id, category, filename, file_path, mime_type, file_size, alt_text, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (asset_id, category, filename, file_path, mime_type, len(content), alt_text, description))
        db.commit()

        logger.info(f"Uploaded asset: {asset_id} -> {file_path}")

        # Generate markdown reference
        markdown_ref = f"![{alt_text or filename}](assets/{filename})"

        return jsonify({
            'success': True,
            'asset_id': asset_id,
            'filename': filename,
            'file_path': file_path,
            'mime_type': mime_type,
            'file_size': len(content),
            'markdown_ref': markdown_ref,
            'commit_sha': result.get('commit', {}).get('sha', '')[:7],
        })

    except Exception as e:
        logger.error(f"Failed to upload asset: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@assets_bp.route('/list', methods=['GET'])
@library_required
def list_assets():
    """List assets, optionally filtered by category.

    Query params:
        - category: Filter by category (optional)
        - limit: Maximum results (default: 50)

    Response:
    {
        "assets": [
            {
                "asset_id": "asset-abc123",
                "category": "concept",
                "filename": "diagram-abc123.png",
                "file_path": "concept/assets/diagram-abc123.png",
                "mime_type": "image/png",
                "file_size": 12345,
                "alt_text": "Architecture diagram",
                "markdown_ref": "![Architecture diagram](assets/diagram-abc123.png)",
                "created_at": "2026-01-15T10:30:00Z"
            },
            ...
        ]
    }
    """
    category = request.args.get('category', '').strip().lower()
    limit = min(int(request.args.get('limit', 50)), 100)

    try:
        db = get_db()

        if category:
            rows = db.execute("""
                SELECT asset_id, category, filename, file_path, mime_type, file_size,
                       alt_text, description, created_at
                FROM library_assets
                WHERE category = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (category, limit)).fetchall()
        else:
            rows = db.execute("""
                SELECT asset_id, category, filename, file_path, mime_type, file_size,
                       alt_text, description, created_at
                FROM library_assets
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,)).fetchall()

        assets = []
        for row in rows:
            asset = dict(row)
            # Add markdown reference
            asset['markdown_ref'] = f"![{asset['alt_text'] or asset['filename']}](assets/{asset['filename']})"
            assets.append(asset)

        return jsonify({'assets': assets, 'count': len(assets)})

    except Exception as e:
        logger.error(f"Failed to list assets: {e}")
        return jsonify({'error': str(e)}), 500


@assets_bp.route('/<asset_id>', methods=['GET'])
@library_required
def get_asset(asset_id: str):
    """Get asset metadata.

    Response:
    {
        "asset_id": "asset-abc123",
        "category": "concept",
        "filename": "diagram-abc123.png",
        "file_path": "concept/assets/diagram-abc123.png",
        "mime_type": "image/png",
        "file_size": 12345,
        "alt_text": "Architecture diagram",
        "markdown_ref": "![Architecture diagram](assets/diagram-abc123.png)",
        "created_at": "2026-01-15T10:30:00Z"
    }
    """
    try:
        db = get_db()

        row = db.execute("""
            SELECT asset_id, category, filename, file_path, mime_type, file_size,
                   alt_text, description, created_at
            FROM library_assets
            WHERE asset_id = ?
        """, (asset_id,)).fetchone()

        if not row:
            return jsonify({'error': 'Asset not found'}), 404

        asset = dict(row)
        asset['markdown_ref'] = f"![{asset['alt_text'] or asset['filename']}](assets/{asset['filename']})"

        return jsonify(asset)

    except Exception as e:
        logger.error(f"Failed to get asset: {e}")
        return jsonify({'error': str(e)}), 500


@assets_bp.route('/<asset_id>/raw', methods=['GET'])
@library_required
def get_asset_raw(asset_id: str):
    """Get the raw binary content of an asset.

    Returns the file content with appropriate MIME type.
    """
    from .rag.github_service import get_binary_file

    try:
        db = get_db()

        row = db.execute("""
            SELECT file_path, mime_type, filename
            FROM library_assets
            WHERE asset_id = ?
        """, (asset_id,)).fetchone()

        if not row:
            return jsonify({'error': 'Asset not found'}), 404

        # Get user credentials
        user_id = get_user_id()
        if not user_id:
            return jsonify({'error': 'Authentication required'}), 401

        token = get_user_installation_token(user_id, 'library')
        if not token:
            return jsonify({'error': 'GitHub authorization required'}), 401

        library_repo = get_user_library_repo()
        if not library_repo:
            return jsonify({'error': 'Library repo not configured'}), 400

        # Fetch from GitHub
        content = get_binary_file(library_repo, row['file_path'], token)
        if content is None:
            return jsonify({'error': 'Asset file not found in repository'}), 404

        return Response(
            content,
            mimetype=row['mime_type'],
            headers={
                'Content-Disposition': f'inline; filename="{row["filename"]}"'
            }
        )

    except Exception as e:
        logger.error(f"Failed to get raw asset: {e}")
        return jsonify({'error': str(e)}), 500


@assets_bp.route('/<asset_id>', methods=['DELETE'])
@library_required
def delete_asset(asset_id: str):
    """Delete an asset from the library.

    Removes the file from GitHub and the database record.

    Response:
    {
        "success": true,
        "deleted": "asset-abc123"
    }
    """
    from .rag.github_service import delete_file

    try:
        db = get_db()

        row = db.execute("""
            SELECT file_path, filename
            FROM library_assets
            WHERE asset_id = ?
        """, (asset_id,)).fetchone()

        if not row:
            return jsonify({'error': 'Asset not found'}), 404

        # Get user credentials
        user_id = get_user_id()
        if not user_id:
            return jsonify({'error': 'Authentication required'}), 401

        token = get_user_installation_token(user_id, 'library')
        if not token:
            return jsonify({'error': 'GitHub authorization required'}), 401

        library_repo = get_user_library_repo()
        if not library_repo:
            return jsonify({'error': 'Library repo not configured'}), 400

        # Delete from GitHub
        try:
            delete_file(
                repo=library_repo,
                path=row['file_path'],
                message=f"Delete asset: {row['filename']}",
                token=token
            )
        except Exception as e:
            if '404' not in str(e):
                raise
            # File already deleted from GitHub, continue to clean up database

        # Delete from database
        db.execute("DELETE FROM library_assets WHERE asset_id = ?", (asset_id,))
        db.commit()

        logger.info(f"Deleted asset: {asset_id}")

        return jsonify({'success': True, 'deleted': asset_id})

    except Exception as e:
        logger.error(f"Failed to delete asset: {e}")
        return jsonify({'error': str(e)}), 500


@assets_bp.route('/<asset_id>', methods=['PATCH'])
@library_required
def update_asset(asset_id: str):
    """Update asset metadata (alt_text, description).

    Request body:
    {
        "alt_text": "New alt text",
        "description": "New description"
    }

    Response:
    {
        "success": true,
        "asset_id": "asset-abc123"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON body required'}), 400

    try:
        db = get_db()

        # Check asset exists
        existing = db.execute(
            "SELECT asset_id FROM library_assets WHERE asset_id = ?",
            (asset_id,)
        ).fetchone()

        if not existing:
            return jsonify({'error': 'Asset not found'}), 404

        # Build update
        updates = []
        params = []

        if 'alt_text' in data:
            updates.append('alt_text = ?')
            params.append(data['alt_text'])

        if 'description' in data:
            updates.append('description = ?')
            params.append(data['description'])

        if not updates:
            return jsonify({'error': 'No fields to update'}), 400

        updates.append('updated_at = CURRENT_TIMESTAMP')
        params.append(asset_id)

        db.execute(
            f"UPDATE library_assets SET {', '.join(updates)} WHERE asset_id = ?",
            params
        )
        db.commit()

        return jsonify({'success': True, 'asset_id': asset_id})

    except Exception as e:
        logger.error(f"Failed to update asset: {e}")
        return jsonify({'error': str(e)}), 500


# ============ Utility Functions ============

def get_assets_for_category(category: str) -> list[dict]:
    """Get all assets for a category.

    Args:
        category: Category name

    Returns:
        List of asset dictionaries
    """
    db = get_db()
    rows = db.execute("""
        SELECT asset_id, filename, file_path, mime_type, alt_text
        FROM library_assets
        WHERE category = ?
        ORDER BY created_at DESC
    """, (category,)).fetchall()

    return [dict(row) for row in rows]


def get_markdown_ref(asset_id: str) -> Optional[str]:
    """Get the markdown reference for an asset.

    Args:
        asset_id: Asset ID

    Returns:
        Markdown reference string like ![alt](assets/filename.png),
        or None if asset not found
    """
    db = get_db()
    row = db.execute("""
        SELECT filename, alt_text
        FROM library_assets
        WHERE asset_id = ?
    """, (asset_id,)).fetchone()

    if not row:
        return None

    alt = row['alt_text'] or row['filename']
    return f"![{alt}](assets/{row['filename']})"
