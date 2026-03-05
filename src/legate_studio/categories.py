"""
Categories Blueprint - User-defined category management.

Provides API and UI for managing knowledge categories.
Categories are stored per-user and can be customized from defaults.
"""

import os
import re
import logging
from flask import Blueprint, request, jsonify, render_template, g

from .core import login_required, library_required

logger = logging.getLogger(__name__)


def create_category_folder(folder_name: str, token: str = None, user_id: str = None) -> dict:
    """Create a folder in the Library repo with .gitkeep.

    Args:
        folder_name: The folder to create (e.g., 'work-queues')
        token: GitHub token (required)
        user_id: User ID for getting library repo

    Returns:
        Dict with status and any error info
    """
    from .rag.github_service import commit_file
    from .core import get_user_library_repo

    if not token:
        return {'created': False, 'error': 'GitHub authorization required'}

    library_repo = get_user_library_repo(user_id)

    try:
        result = commit_file(
            repo=library_repo,
            path=f"{folder_name}/.gitkeep",
            content="# This folder contains knowledge entries\n",
            message=f"Create {folder_name} category folder",
            token=token
        )
        return {'created': True, 'commit': result.get('commit', {}).get('sha', '')[:7]}
    except Exception as e:
        # File might already exist, which is fine
        if 'sha' in str(e).lower() or '422' in str(e):
            return {'created': False, 'exists': True}
        logger.error(f"Failed to create category folder {folder_name}: {e}")
        return {'created': False, 'error': str(e)}

categories_bp = Blueprint('categories', __name__, url_prefix='/categories')


def get_db():
    """Get legato database connection for current user."""
    from .rag.database import get_user_legato_db
    return get_user_legato_db()


def get_user_id():
    """Get current user ID (or 'default' for single-user mode)."""
    from flask import session
    user = session.get('user')
    if user and user.get('user_id'):
        return user['user_id']
    return 'default'


# ============ API Endpoints ============

@categories_bp.route('/api/list', methods=['GET'])
@login_required
def api_list_categories():
    """List all active categories for the current user.

    Response:
    {
        "categories": [
            {
                "id": 1,
                "name": "epiphany",
                "display_name": "Epiphany",
                "description": "Major breakthrough...",
                "folder_name": "epiphanys",
                "sort_order": 1
            },
            ...
        ]
    }
    """
    from .rag.database import get_user_categories

    db = get_db()
    user_id = get_user_id()
    categories = get_user_categories(db, user_id)

    return jsonify({'categories': categories})


@categories_bp.route('/api/create', methods=['POST'])
@login_required
def api_create_category():
    """Create a new category.

    Request body:
    {
        "name": "research",           -- slug (required)
        "display_name": "Research",   -- human readable (required)
        "description": "...",         -- optional
        "folder_name": "research",    -- optional, defaults to {name}s
        "color": "#6366f1"            -- optional, defaults to indigo
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON body required'}), 400

    name = data.get('name', '').lower().strip()
    display_name = data.get('display_name', '').strip()
    description = data.get('description', '').strip()
    folder_name = data.get('folder_name', '').strip()
    color = data.get('color', '#6366f1').strip()

    if not name or not display_name:
        return jsonify({'error': 'name and display_name are required'}), 400

    # Validate name (slug format)
    if not re.match(r'^[a-z][a-z0-9-]*$', name):
        return jsonify({'error': 'name must start with a letter and contain only lowercase letters, numbers, and hyphens'}), 400

    if len(name) > 30:
        return jsonify({'error': 'name must be 30 characters or less'}), 400

    # Validate color (hex format)
    if color and not re.match(r'^#[0-9a-fA-F]{6}$', color):
        return jsonify({'error': 'color must be a valid hex color (e.g., #6366f1)'}), 400

    # Default folder_name - use category name directly (singular, matching DB defaults)
    if not folder_name:
        folder_name = name

    from flask import session
    from .auth import get_user_installation_token

    try:
        db = get_db()
        user_id = get_user_id()

        # Get user token for GitHub operations
        token = get_user_installation_token(user_id, 'library') if user_id != 'default' else None

        # Get next sort_order
        max_order = db.execute(
            "SELECT MAX(sort_order) FROM user_categories WHERE user_id = ?",
            (user_id,)
        ).fetchone()[0] or 0

        cursor = db.execute("""
            INSERT INTO user_categories (user_id, name, display_name, description, folder_name, sort_order, color)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user_id, name, display_name, description, folder_name, max_order + 1, color))

        db.commit()
        category_id = cursor.lastrowid

        logger.info(f"Created category: {name} (id={category_id})")

        # Create folder in Library repo (only if we have a token)
        folder_result = create_category_folder(folder_name, token=token, user_id=user_id) if token else {'created': False, 'error': 'No token'}
        if folder_result.get('created'):
            logger.info(f"Created folder {folder_name} in Library repo")
        elif folder_result.get('exists'):
            logger.info(f"Folder {folder_name} already exists in Library repo")
        elif folder_result.get('error'):
            logger.warning(f"Could not create folder {folder_name}: {folder_result['error']}")

        return jsonify({
            'success': True,
            'id': category_id,
            'name': name,
            'folder_created': folder_result.get('created', False),
        })

    except Exception as e:
        if 'UNIQUE constraint' in str(e):
            return jsonify({'error': f'Category "{name}" already exists'}), 409
        logger.error(f"Failed to create category: {e}")
        return jsonify({'error': str(e)}), 500


@categories_bp.route('/api/adopt', methods=['POST'])
@login_required
def api_adopt_category():
    """Adopt an orphaned category by recreating its database record.

    This is used when a category exists in knowledge_entries but not in
    user_categories (e.g., after database loss).

    Request body:
    {
        "name": "work-queue",           -- the orphaned category slug (required)
        "display_name": "Work Queue",   -- human readable (optional, will guess from name)
        "description": "...",           -- optional
        "folder_name": "work-queues",   -- optional, will guess from name
        "color": "#6366f1"              -- optional
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON body required'}), 400

    name = data.get('name', '').lower().strip()
    if not name:
        return jsonify({'error': 'name is required'}), 400

    # Check that this category actually exists in entries (is truly orphaned)
    db = get_db()
    entry_count = db.execute(
        "SELECT COUNT(*) FROM knowledge_entries WHERE category = ?",
        (name,)
    ).fetchone()[0]

    if entry_count == 0:
        return jsonify({'error': f'No entries found with category "{name}"'}), 404

    # Use provided values or generate reasonable defaults
    display_name = data.get('display_name', '').strip()
    if not display_name:
        display_name = name.replace('-', ' ').title()

    folder_name = data.get('folder_name', '').strip()
    if not folder_name:
        folder_name = name  # Use category name directly (singular, matching DB defaults)

    description = data.get('description', '').strip()
    color = data.get('color', '#6366f1').strip()

    try:
        user_id = get_user_id()

        # Check if category exists but is inactive (soft-deleted)
        existing = db.execute(
            "SELECT id, is_active FROM user_categories WHERE user_id = ? AND name = ?",
            (user_id, name)
        ).fetchone()

        if existing:
            if existing['is_active'] == 1:
                # Already active - this shouldn't happen but handle it
                return jsonify({'error': f'Category "{name}" is already active'}), 409

            # Reactivate the inactive category
            db.execute("""
                UPDATE user_categories
                SET is_active = 1, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (existing['id'],))
            db.commit()

            logger.info(f"Reactivated category: {name} (id={existing['id']}, {entry_count} entries)")

            return jsonify({
                'success': True,
                'id': existing['id'],
                'name': name,
                'reactivated': True,
                'entry_count': entry_count,
            })

        # Category doesn't exist at all - create it
        max_order = db.execute(
            "SELECT MAX(sort_order) FROM user_categories WHERE user_id = ?",
            (user_id,)
        ).fetchone()[0] or 0

        cursor = db.execute("""
            INSERT INTO user_categories (user_id, name, display_name, description, folder_name, sort_order, color)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user_id, name, display_name, description, folder_name, max_order + 1, color))

        db.commit()
        category_id = cursor.lastrowid

        logger.info(f"Adopted orphaned category: {name} (id={category_id}, {entry_count} entries)")

        return jsonify({
            'success': True,
            'id': category_id,
            'name': name,
            'display_name': display_name,
            'entry_count': entry_count,
        })

    except Exception as e:
        logger.error(f"Failed to adopt category: {e}")
        return jsonify({'error': str(e)}), 500


@categories_bp.route('/api/<int:category_id>', methods=['PUT'])
@login_required
def api_update_category(category_id: int):
    """Update an existing category.

    Request body (all optional):
    {
        "display_name": "New Display Name",
        "description": "New description",
        "folder_name": "new-folder",
        "sort_order": 5,
        "color": "#ff5500"
    }

    If folder_name changes, all notes in the old folder will be moved to the new folder.
    """
    from flask import session
    from .auth import get_user_installation_token

    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON body required'}), 400

    try:
        db = get_db()
        user_id = get_user_id()

        # Get user token for GitHub operations
        token = get_user_installation_token(user_id, 'library') if user_id != 'default' else None

        # Get current category info
        current = db.execute(
            "SELECT name, folder_name FROM user_categories WHERE id = ? AND user_id = ?",
            (category_id, user_id)
        ).fetchone()

        if not current:
            return jsonify({'error': 'Category not found'}), 404

        old_folder = current['folder_name']
        new_folder = data.get('folder_name', '').strip() if 'folder_name' in data else old_folder
        folder_changed = new_folder and new_folder != old_folder

        # Validate color if provided
        if 'color' in data:
            color = data['color'].strip()
            if color and not re.match(r'^#[0-9a-fA-F]{6}$', color):
                return jsonify({'error': 'color must be a valid hex color (e.g., #6366f1)'}), 400

        # Build update query dynamically
        updates = []
        params = []

        if 'display_name' in data:
            updates.append('display_name = ?')
            params.append(data['display_name'].strip())

        if 'description' in data:
            updates.append('description = ?')
            params.append(data['description'].strip())

        if 'folder_name' in data:
            updates.append('folder_name = ?')
            params.append(new_folder)

        if 'sort_order' in data:
            updates.append('sort_order = ?')
            params.append(int(data['sort_order']))

        if 'color' in data:
            updates.append('color = ?')
            params.append(data['color'].strip())

        if not updates:
            return jsonify({'error': 'No fields to update'}), 400

        # If folder is changing, move files in GitHub first
        files_moved = 0
        move_errors = []

        if folder_changed:
            from .rag.github_service import list_folder, move_file, create_file
            from .core import get_user_library_repo

            # Use the user token already retrieved above (line 329)
            library_repo = get_user_library_repo()

            if token:
                try:
                    # Create new folder with .gitkeep first
                    try:
                        create_file(
                            repo=library_repo,
                            path=f"{new_folder}/.gitkeep",
                            content="# This folder contains knowledge entries\n",
                            message=f"Create {new_folder} folder for category rename",
                            token=token
                        )
                    except Exception:
                        pass  # Folder might already exist

                    # List files in old folder
                    files = list_folder(library_repo, old_folder, token)

                    for file_info in files:
                        if file_info.get('type') != 'file':
                            continue
                        if file_info.get('name') == '.gitkeep':
                            continue

                        old_path = file_info['path']
                        new_path = f"{new_folder}/{file_info['name']}"

                        try:
                            move_file(
                                repo=library_repo,
                                old_path=old_path,
                                new_path=new_path,
                                message=f"Move {file_info['name']} from {old_folder} to {new_folder}",
                                token=token
                            )
                            files_moved += 1

                            # Update file_path in database
                            db.execute(
                                "UPDATE knowledge_entries SET file_path = ? WHERE file_path = ?",
                                (new_path, old_path)
                            )

                        except Exception as e:
                            move_errors.append(f"{file_info['name']}: {str(e)}")
                            logger.error(f"Failed to move {old_path}: {e}")

                except Exception as e:
                    logger.error(f"Failed to move files during folder rename: {e}")
                    return jsonify({
                        'error': f'Failed to move files: {str(e)}',
                        'files_moved': files_moved
                    }), 500
            else:
                logger.warning("No GitHub token available, cannot move files during folder rename")

        # Update database
        updates.append('updated_at = CURRENT_TIMESTAMP')
        params.extend([category_id, user_id])

        db.execute(
            f"UPDATE user_categories SET {', '.join(updates)} WHERE id = ? AND user_id = ?",
            params
        )
        db.commit()

        logger.info(f"Updated category id={category_id}" +
                   (f", moved {files_moved} files from {old_folder} to {new_folder}" if folder_changed else ""))

        response = {'success': True}
        if folder_changed:
            response['files_moved'] = files_moved
            if move_errors:
                response['move_errors'] = move_errors

        return jsonify(response)

    except Exception as e:
        logger.error(f"Failed to update category: {e}")
        return jsonify({'error': str(e)}), 500


@categories_bp.route('/api/<int:category_id>/stats', methods=['GET'])
@login_required
def api_category_stats(category_id: int):
    """Get stats for a category (note count, etc).

    Used to warn before deletion.
    """
    try:
        db = get_db()
        user_id = get_user_id()

        # Get category name
        cat = db.execute(
            "SELECT name, display_name FROM user_categories WHERE id = ? AND user_id = ?",
            (category_id, user_id)
        ).fetchone()

        if not cat:
            return jsonify({'error': 'Category not found'}), 404

        # Count notes using this category
        result = db.execute(
            "SELECT COUNT(*) FROM knowledge_entries WHERE category = ?",
            (cat['name'],)
        ).fetchone()
        note_count = result[0] if result else 0

        return jsonify({
            'name': cat['name'],
            'display_name': cat['display_name'],
            'note_count': note_count,
        })

    except Exception as e:
        logger.error(f"Failed to get category stats: {e}")
        return jsonify({'error': str(e)}), 500


@categories_bp.route('/api/<int:category_id>', methods=['DELETE'])
@login_required
def api_delete_category(category_id: int):
    """Delete (soft-delete) a category.

    Note: This sets is_active=0, not a hard delete.
    Entries using this category will keep their current category.
    Requires confirm=true in request body to actually delete.
    """
    data = request.get_json() or {}
    confirm = data.get('confirm', False)

    try:
        db = get_db()
        user_id = get_user_id()

        # Get category name for logging
        cat = db.execute(
            "SELECT name FROM user_categories WHERE id = ? AND user_id = ?",
            (category_id, user_id)
        ).fetchone()

        if not cat:
            return jsonify({'error': 'Category not found'}), 404

        # Count affected notes
        result = db.execute(
            "SELECT COUNT(*) FROM knowledge_entries WHERE category = ?",
            (cat['name'],)
        ).fetchone()
        note_count = result[0] if result else 0

        if not confirm and note_count > 0:
            return jsonify({
                'error': 'Confirmation required',
                'note_count': note_count,
                'requires_confirm': True,
            }), 400

        db.execute(
            "UPDATE user_categories SET is_active = 0, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (category_id,)
        )
        db.commit()

        logger.info(f"Deleted category: {cat['name']} (id={category_id}), orphaned {note_count} notes")

        return jsonify({'success': True, 'name': cat['name'], 'orphaned_notes': note_count})

    except Exception as e:
        logger.error(f"Failed to delete category: {e}")
        return jsonify({'error': str(e)}), 500


@categories_bp.route('/api/reorder', methods=['POST'])
@login_required
def api_reorder_categories():
    """Reorder categories.

    Request body:
    {
        "order": [3, 1, 2, 5, 4]  -- category IDs in new order
    }
    """
    data = request.get_json()
    if not data or 'order' not in data:
        return jsonify({'error': 'order array required'}), 400

    try:
        db = get_db()
        user_id = get_user_id()

        for idx, category_id in enumerate(data['order']):
            db.execute(
                "UPDATE user_categories SET sort_order = ? WHERE id = ? AND user_id = ?",
                (idx + 1, category_id, user_id)
            )

        db.commit()

        return jsonify({'success': True})

    except Exception as e:
        logger.error(f"Failed to reorder categories: {e}")
        return jsonify({'error': str(e)}), 500


# ============ UI Route ============

@categories_bp.route('/')
@library_required
def index():
    """Category management page."""
    from .rag.database import get_user_categories

    db = get_db()
    user_id = get_user_id()
    categories = get_user_categories(db, user_id)

    # Also get inactive categories so users can reactivate them
    inactive_rows = db.execute("""
        SELECT id, name, display_name, description, folder_name, color, sort_order
        FROM user_categories
        WHERE user_id = ? AND is_active = 0
        ORDER BY name
    """, (user_id,)).fetchall()
    inactive_categories = [dict(row) for row in inactive_rows]

    return render_template('categories.html', categories=categories, inactive_categories=inactive_categories)


@categories_bp.route('/api/debug', methods=['GET'])
@login_required
def api_debug_categories():
    """Debug endpoint to check database state.

    Returns all categories (including inactive) and database path info.
    """
    from .rag.database import get_db_path

    db = get_db()
    user_id = get_user_id()

    # Get ALL categories (including inactive)
    all_categories = db.execute("""
        SELECT id, user_id, name, display_name, folder_name, is_active, color, created_at
        FROM user_categories
        WHERE user_id = ?
        ORDER BY sort_order
    """, (user_id,)).fetchall()

    # Get categories used in entries
    entry_categories = db.execute("""
        SELECT category, COUNT(*) as count
        FROM knowledge_entries
        WHERE category IS NOT NULL AND category != ''
        GROUP BY category
    """).fetchall()

    # Find mismatches
    db_category_names = {row['name'] for row in all_categories}
    entry_category_names = {row['category'] for row in entry_categories}
    orphaned = entry_category_names - db_category_names
    unused = db_category_names - entry_category_names

    return jsonify({
        'db_path': str(get_db_path('legato.db')),
        'user_id': user_id,
        'total_in_db': len(all_categories),
        'total_in_entries': len(entry_categories),
        'categories_in_db': [dict(row) for row in all_categories],
        'categories_in_entries': [{'name': row['category'], 'count': row['count']} for row in entry_categories],
        'orphaned_in_entries': list(orphaned),
        'unused_in_db': list(unused),
    })
