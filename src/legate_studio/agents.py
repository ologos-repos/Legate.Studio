"""
Agent Queue Blueprint

Handles queuing, approval, and spawning of Lab project agents.
Provides an approval gateway before spawning new repositories.

Project spawning is now handled directly by Pit using the chord_executor
module, replacing the previous Conduct workflow dispatch.
"""

import os
import json
import secrets
import logging
from datetime import datetime

import requests
from flask import Blueprint, request, jsonify, session, current_app, g, render_template

from .core import login_required, library_required, copilot_required, paid_required

logger = logging.getLogger(__name__)

agents_bp = Blueprint('agents', __name__, url_prefix='/agents')


def get_db():
    """Get agents database connection."""
    if 'agents_db_conn' not in g:
        from .rag.database import init_agents_db
        g.agents_db_conn = init_agents_db()
    return g.agents_db_conn


def get_legato_db():
    """Get legato database connection for current user."""
    from .rag.database import get_user_legato_db
    return get_user_legato_db()


def generate_queue_id() -> str:
    """Generate a unique queue ID."""
    return f"aq-{secrets.token_hex(6)}"


def verify_system_token(req) -> bool:
    """Verify the request has a valid system token.

    NOTE: This is for machine-to-machine auth (Conduct -> Pit).
    In multi-tenant mode, external API endpoints using this may need redesign
    to include user context in the request.
    """
    auth_header = req.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        token = auth_header[7:]
        # TODO: In multi-tenant mode, consider per-user API tokens
        system_pat = current_app.config.get('SYSTEM_PAT')
        return token == system_pat
    return False


# ============ Page Routes ============

@agents_bp.route('/')
@library_required
@paid_required
@copilot_required
def index():
    """Agents queue management page."""
    from flask import session

    db = get_db()
    user_id = session.get('user', {}).get('user_id')

    # Get pending agents for THIS USER only
    pending_rows = db.execute(
        """
        SELECT queue_id, project_name, project_type, title, description,
               source_transcript, related_entry_id, comments, created_at
        FROM agent_queue
        WHERE status = 'pending' AND user_id = ?
        ORDER BY created_at DESC
        """,
        (user_id,)
    ).fetchall()
    pending_agents = [dict(row) for row in pending_rows]

    # Look up note titles for related_entry_ids and parse comments
    # Only access legato_db if we have pending agents with related entries
    legato_db = None
    for agent in pending_agents:
        if agent.get('related_entry_id'):
            # Lazy load legato_db only when needed
            if legato_db is None:
                try:
                    legato_db = get_legato_db()
                except ValueError:
                    # No user_id available - skip note title lookups
                    legato_db = False

            if legato_db:
                # Handle comma-separated entry IDs (take first one for display)
                entry_id = agent['related_entry_id'].split(',')[0].strip()
                note = legato_db.execute(
                    "SELECT title FROM knowledge_entries WHERE entry_id = ?",
                    (entry_id,)
                ).fetchone()
                agent['related_note_title'] = note['title'] if note else None

        # Parse comments JSON array
        if agent.get('comments'):
            try:
                agent['comments_list'] = json.loads(agent['comments'])
            except (json.JSONDecodeError, TypeError):
                agent['comments_list'] = []
        else:
            agent['comments_list'] = []

    # Get failed spawns (can be retried) for THIS USER only
    failed_rows = db.execute(
        """
        SELECT queue_id, project_name, project_type, title, description,
               spawn_result, created_at, updated_at
        FROM agent_queue
        WHERE status = 'spawn_failed' AND user_id = ?
        ORDER BY updated_at DESC
        """,
        (user_id,)
    ).fetchall()
    failed_agents = [dict(row) for row in failed_rows]

    # Parse spawn_result to get error message
    for agent in failed_agents:
        if agent.get('spawn_result'):
            try:
                result = json.loads(agent['spawn_result'])
                agent['error'] = result.get('error', 'Unknown error')
            except (json.JSONDecodeError, TypeError):
                agent['error'] = 'Unknown error'

    # Get recent processed agents (last 20) for THIS USER only
    recent_rows = db.execute(
        """
        SELECT queue_id, project_name, project_type, title, status,
               approved_by, approved_at
        FROM agent_queue
        WHERE status NOT IN ('pending', 'spawn_failed') AND user_id = ?
        ORDER BY updated_at DESC
        LIMIT 20
        """,
        (user_id,)
    ).fetchall()
    recent_agents = [dict(row) for row in recent_rows]

    return render_template(
        'agents.html',
        pending_agents=pending_agents,
        failed_agents=failed_agents,
        recent_agents=recent_agents,
    )


# ============ API Endpoints (called by Pit UI) ============

@agents_bp.route('/api/create', methods=['POST'])
@login_required
@paid_required
@copilot_required
def api_create_agent():
    """Create a new agent from selected notes.

    Request body:
    {
        "note_ids": ["kb-abc123", "kb-def456"],
        "project_name": "optional-slug",
        "project_type": "note" or "chord",
        "initial_comment": "Optional comment"
    }

    Response:
    {
        "success": true,
        "queue_id": "aq-xxx",
        "project_name": "..."
    }
    """
    import re

    try:
        db = get_db()
        legato_db = get_legato_db()
        data = request.get_json() or {}

        note_ids = data.get('note_ids', [])
        project_name = data.get('project_name', '').strip()
        project_type = data.get('project_type', 'note').lower()
        initial_comment = data.get('initial_comment', '').strip()

        user = session.get('user', {})
        username = user.get('username', 'unknown')

        # Validate
        if not note_ids:
            return jsonify({'error': 'At least one note is required'}), 400
        if len(note_ids) > 5:
            return jsonify({'error': 'Maximum 5 notes allowed'}), 400
        if project_type not in ('note', 'chord'):
            project_type = 'note'

        # Look up all notes
        notes = []
        for nid in note_ids:
            entry = legato_db.execute(
                "SELECT entry_id, title, category, content, domain_tags FROM knowledge_entries WHERE entry_id = ?",
                (nid.strip(),)
            ).fetchone()
            if entry:
                notes.append(dict(entry))
            else:
                return jsonify({'error': f'Note not found: {nid}'}), 404

        if not notes:
            return jsonify({'error': 'No valid notes found'}), 400

        # Use first note as primary
        primary = notes[0]

        # Generate project name if not provided
        if not project_name:
            slug = re.sub(r'[^a-z0-9]+', '_', primary['title'].lower()).strip('_')
            project_name = slug[:50]

        # Generate queue_id
        queue_id = generate_queue_id()

        # Build initial comments
        initial_comments = []
        if initial_comment:
            initial_comments.append({
                "text": initial_comment,
                "author": username,
                "timestamp": datetime.now().isoformat() + "Z"
            })

        # Build signal JSON
        repo_suffix = "Chord" if project_type == "chord" else "Note"
        signal_json = {
            "title": primary['title'],
            "intent": primary['content'][:500] if primary['content'] else "",
            "domain_tags": primary.get('domain_tags', '').split(',') if primary.get('domain_tags') else [],
            "source_notes": [n['entry_id'] for n in notes],
            "path": f"{project_name}.{repo_suffix}",
        }

        # Build tasker body
        notes_section = "\n".join([f"- **{n['title']}** (`{n['entry_id']}`)" for n in notes])
        tasker_body = f"""## Tasker: {primary['title']}

### Linked Notes
{notes_section}

### Context
{primary['content'][:1000] if primary['content'] else 'No content'}

---
*Queued via Pit UI by {username} | {len(notes)} note(s) linked*
"""

        # Build description
        if len(notes) > 1:
            description = f"Multi-note chord linking {len(notes)} notes"
        else:
            description = primary['content'][:200] if primary['content'] else primary['title']

        # Insert into agent_queue
        db.execute(
            """
            INSERT INTO agent_queue
            (queue_id, project_name, project_type, title, description,
             signal_json, tasker_body, source_transcript, related_entry_id, comments, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            """,
            (
                queue_id,
                project_name,
                project_type,
                primary['title'],
                description,
                json.dumps(signal_json),
                tasker_body,
                f'pit-ui:{username}',
                ','.join(n['entry_id'] for n in notes),
                json.dumps(initial_comments),
            )
        )
        db.commit()

        logger.info(f"Created agent via UI: {queue_id} - {project_name} by {username}")

        return jsonify({
            'success': True,
            'queue_id': queue_id,
            'project_name': project_name,
            'project_type': project_type,
            'notes_linked': len(notes),
        })

    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        return jsonify({'error': str(e)}), 500


@agents_bp.route('/api/queue-chord', methods=['POST'])
@login_required
@paid_required
@copilot_required
def api_queue_chord():
    """Mark a library entry as needing a chord (lightweight flagging).

    This just sets needs_chord=1 on the entry. The actual agent creation
    happens later via /api/from-entry when the user provides a project name.

    Request body:
    {
        "entry_id": "kb-abc123"
    }

    Response:
    {
        "status": "success",
        "entry_id": "kb-abc123",
        "needs_chord": true
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    entry_id = data.get('entry_id')
    if not entry_id:
        return jsonify({'error': 'Missing entry_id'}), 400

    try:
        legato_db = get_legato_db()

        # Check entry exists
        entry = legato_db.execute(
            "SELECT id, title, needs_chord FROM knowledge_entries WHERE entry_id = ?",
            (entry_id,)
        ).fetchone()

        if not entry:
            return jsonify({'error': 'Entry not found'}), 404

        if entry['needs_chord']:
            return jsonify({
                'status': 'already_flagged',
                'entry_id': entry_id,
                'message': 'Entry is already flagged for chord'
            })

        # Flag entry as needing chord
        legato_db.execute("""
            UPDATE knowledge_entries
            SET needs_chord = 1, updated_at = CURRENT_TIMESTAMP
            WHERE entry_id = ?
        """, (entry_id,))
        legato_db.commit()

        logger.info(f"Flagged entry for chord: {entry_id}")

        return jsonify({
            'status': 'success',
            'entry_id': entry_id,
            'needs_chord': True
        })

    except Exception as e:
        logger.error(f"Failed to flag entry for chord: {e}")
        return jsonify({'error': str(e)}), 500


@agents_bp.route('/api/from-entry', methods=['POST'])
@login_required
@paid_required
@copilot_required
def api_queue_from_entry():
    """Queue an agent to create a Chord (Lab repo) from a Note (library entry).

    Request body:
    {
        "entry_id": "kb-abc123"
    }

    Response:
    {
        "status": "queued",
        "queue_id": "aq-abc123def456"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    entry_id = data.get('entry_id')
    project_name = data.get('project_name')

    if not entry_id:
        return jsonify({'error': 'Missing entry_id'}), 400

    if not project_name:
        return jsonify({'error': 'Missing project_name'}), 400

    # Validate project name
    import re
    project_name = re.sub(r'[^a-z0-9_]', '', project_name.lower().replace(' ', '_'))[:30]
    if len(project_name) < 2:
        return jsonify({'error': 'Project name must be at least 2 characters'}), 400

    try:
        legato_db = get_legato_db()
        agents_db = get_db()

        # Get the library entry from legato.db
        entry = legato_db.execute(
            "SELECT * FROM knowledge_entries WHERE entry_id = ?",
            (entry_id,)
        ).fetchone()

        if not entry:
            return jsonify({'error': 'Entry not found'}), 404

        entry = dict(entry)

        # Build tasker body from entry content
        content_preview = entry['content'][:500] if entry['content'] else ''
        tasker_body = f"""## Tasker: {entry['title']}

### Context
From knowledge entry `{entry_id}`:
"{content_preview}"

### Objective
Implement the project as described in the knowledge entry.

### Acceptance Criteria
- [ ] Core functionality implemented
- [ ] Documentation updated
- [ ] Tests written

### Constraints
- Follow patterns in `copilot-instructions.md`
- Reference `SIGNAL.md` for project intent
- Keep PRs focused and reviewable

### References
- Source entry: `{entry_id}`
- Category: {entry.get('category', 'general')}

---
*Generated from Pit library entry | Source: {entry_id}*
"""

        # Build signal JSON - always creates a Chord (repo) from a Note (entry)
        signal_json = {
            "id": f"lab.chord.{project_name}",
            "type": "project",
            "source": "pit-library",
            "category": "chord",
            "title": entry['title'],
            "domain_tags": [],
            "intent": entry.get('content', '')[:200],
            "key_phrases": [],
            "path": f"{project_name}.Chord",
        }

        queue_id = generate_queue_id()

        agents_db.execute(
            """
            INSERT INTO agent_queue
            (queue_id, project_name, project_type, title, description,
             signal_json, tasker_body, source_transcript, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            """,
            (
                queue_id,
                project_name,
                'chord',  # Always chord - we're creating a repo from a note
                entry['title'],
                entry.get('content', '')[:500],
                json.dumps(signal_json),
                tasker_body,
                f"library:{entry_id}",
            )
        )
        agents_db.commit()

        logger.info(f"Queued agent from entry: {queue_id} - {project_name}")

        return jsonify({
            'status': 'queued',
            'queue_id': queue_id,
            'project_name': project_name,
        })

    except Exception as e:
        logger.error(f"Failed to queue from entry: {e}")
        return jsonify({'error': str(e)}), 500


# ============ API Endpoints (called by Conduct) ============

@agents_bp.route('/api/queue', methods=['POST'])
def api_queue_agent():
    """Queue a new agent for approval.

    Called by Conduct when a PROJECT thread is classified.
    Requires system token authentication (machine-to-machine).

    Request body:
    {
        "project_name": "MyProject",
        "project_type": "note" or "chord",
        "title": "Project Title",
        "description": "Project description",
        "signal_json": { ... },
        "tasker_body": "Issue body markdown",
        "source_transcript": "transcript-id"
    }

    Response:
    {
        "status": "queued",
        "queue_id": "aq-abc123def456"
    }
    """
    if not verify_system_token(request):
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    required_fields = ['project_name', 'project_type', 'title', 'signal_json', 'tasker_body']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400

    try:
        db = get_db()
        queue_id = generate_queue_id()

        # Serialize signal_json if it's a dict
        signal_json = data['signal_json']
        if isinstance(signal_json, dict):
            signal_json = json.dumps(signal_json)

        db.execute(
            """
            INSERT INTO agent_queue
            (queue_id, project_name, project_type, title, description,
             signal_json, tasker_body, source_transcript, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            """,
            (
                queue_id,
                data['project_name'],
                data['project_type'],
                data['title'],
                data.get('description', ''),
                signal_json,
                data['tasker_body'],
                data.get('source_transcript'),
            )
        )
        db.commit()

        logger.info(f"Queued agent: {queue_id} - {data['project_name']}")

        return jsonify({
            'status': 'queued',
            'queue_id': queue_id,
        })

    except Exception as e:
        logger.error(f"Failed to queue agent: {e}")
        return jsonify({'error': str(e)}), 500


@agents_bp.route('/api/pending', methods=['GET'])
@login_required
@paid_required
def api_list_pending():
    """List all pending agents.

    Response:
    {
        "agents": [
            {
                "queue_id": "aq-abc123",
                "project_name": "MyProject",
                "project_type": "note",
                "title": "Project Title",
                "description": "...",
                "source_transcript": "...",
                "created_at": "2026-01-09T..."
            }
        ],
        "count": 1
    }
    """
    from flask import session

    try:
        db = get_db()
        user_id = session.get('user', {}).get('user_id')

        rows = db.execute(
            """
            SELECT queue_id, project_name, project_type, title, description,
                   source_transcript, created_at
            FROM agent_queue
            WHERE status = 'pending' AND user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,)
        ).fetchall()

        agents = [dict(row) for row in rows]

        return jsonify({
            'agents': agents,
            'count': len(agents),
        })

    except Exception as e:
        logger.error(f"Failed to list pending agents: {e}")
        return jsonify({'error': str(e)}), 500


@agents_bp.route('/api/pending-count', methods=['GET'])
@login_required
@paid_required
def api_pending_count():
    """Get count of pending agents (lightweight for nav badge).

    Response:
    {
        "count": 3
    }
    """
    from flask import session

    try:
        db = get_db()
        user_id = session.get('user', {}).get('user_id')

        result = db.execute(
            "SELECT COUNT(*) FROM agent_queue WHERE status = 'pending' AND user_id = ?",
            (user_id,)
        ).fetchone()
        count = result[0] if result else 0
        return jsonify({'count': count})
    except Exception as e:
        logger.error(f"Failed to get pending count: {e}")
        return jsonify({'count': 0})


@agents_bp.route('/api/debug', methods=['GET'])
@login_required
@paid_required
def api_debug_agents():
    """Debug endpoint to check agent queue database state.

    Returns database path and queue summary for the current user.
    """
    from flask import session
    from .rag.database import get_db_path

    agents_db = get_db()
    user_id = session.get('user', {}).get('user_id')

    # Get status counts for THIS USER only
    status_counts = agents_db.execute("""
        SELECT status, COUNT(*) as count
        FROM agent_queue
        WHERE user_id = ?
        GROUP BY status
    """, (user_id,)).fetchall()

    # Get recent queue IDs for THIS USER only
    recent = agents_db.execute("""
        SELECT queue_id, status, project_name, created_at
        FROM agent_queue
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT 10
    """, (user_id,)).fetchall()

    return jsonify({
        'db_path': str(get_db_path('agents.db')),
        'user_id': user_id,
        'status_counts': {row['status']: row['count'] for row in status_counts},
        'total_agents': sum(row['count'] for row in status_counts),
        'recent_agents': [dict(row) for row in recent]
    })


@agents_bp.route('/api/<queue_id>/approve', methods=['POST'])
@login_required
@paid_required
@copilot_required
def api_approve_agent(queue_id: str):
    """Approve an agent and trigger spawn.

    This triggers the Conduct spawn-project workflow via repository_dispatch.
    Also updates the linked Library entry's chord_status to 'active'.

    Request body (optional):
    {
        "additional_comments": "Extra context or instructions for the agent..."
    }

    Response:
    {
        "status": "approved",
        "queue_id": "aq-abc123",
        "dispatch_sent": true
    }
    """
    try:
        # Check OAuth token is available (chord spawning requires OAuth)
        user = session.get('user', {})
        user_id = user.get('user_id')

        logger.info(f"api_approve_agent: user_id={user_id}, session_keys={list(session.keys())}")

        if user_id:
            from .auth import _get_user_oauth_token
            oauth_token = _get_user_oauth_token(user_id)
            logger.info(f"api_approve_agent: oauth_token present={bool(oauth_token)}, len={len(oauth_token) if oauth_token else 0}")

            if not oauth_token:
                return jsonify({
                    'error': 'GitHub authorization expired. Please re-authenticate to approve agents.',
                    'needs_reauth': True,
                    'reauth_url': '/auth/github-app-login'
                }), 401

        agents_db = get_db()
        data = request.get_json() or {}
        additional_comments = data.get('additional_comments', '').strip()

        username = user.get('username', 'unknown')
        org = user.get('username')  # Use user's org, not hardcoded

        # Get the queued agent - MUST belong to current user
        row = agents_db.execute(
            "SELECT * FROM agent_queue WHERE queue_id = ? AND status = 'pending' AND user_id = ?",
            (queue_id, user_id)
        ).fetchone()

        if not row:
            return jsonify({'error': 'Agent not found or already processed'}), 404

        agent = dict(row)

        # Append additional comments to tasker_body if provided
        if additional_comments:
            agent['tasker_body'] = agent['tasker_body'] + f"""

---

## Additional Comments from Approver

{additional_comments}

*— {username}*
"""

        # Get user_id for multi-tenant mode
        user_id = user.get('user_id') if user.get('auth_mode') == 'github_app' else None

        # Spawn the project directly (replaces Conduct dispatch)
        dispatch_result = trigger_spawn_workflow(agent, user_id=user_id)

        # Set status based on spawn result
        spawn_success = dispatch_result.get('success', False)
        spawn_error = dispatch_result.get('error', '')
        new_status = 'spawned' if spawn_success else 'spawn_failed'

        # Check if failure was due to token issues - prompt reauth
        if not spawn_success and ('re-authenticate' in spawn_error.lower() or '401' in spawn_error):
            # Clear the invalid session token
            session.pop('github_token', None)
            return jsonify({
                'error': 'GitHub authorization expired. Please re-authenticate to approve agents.',
                'needs_reauth': True,
                'reauth_url': '/auth/github-app-login'
            }), 401

        # Update agent queue status
        agents_db.execute(
            """
            UPDATE agent_queue
            SET status = ?,
                approved_by = ?,
                approved_at = CURRENT_TIMESTAMP,
                spawn_result = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE queue_id = ?
            """,
            (new_status, username, json.dumps(dispatch_result), queue_id)
        )
        agents_db.commit()

        # Only update Library entries if spawn succeeded
        # related_entry_id may be a single ID or comma-separated list for multi-note chords
        related_entry_id = agent.get('related_entry_id')
        logger.info(f"Chord linking check: spawn_success={spawn_success}, related_entry_id={related_entry_id}")
        if spawn_success and related_entry_id:
            try:
                import re
                from .rag.github_service import get_file_content, commit_file

                legato_db = get_legato_db()
                chord_repo_name = f"{agent['project_name']}.Chord"
                chord_repo_full = f"{org}/{chord_repo_name}"

                # Handle single or multiple entry IDs
                entry_ids = [eid.strip() for eid in related_entry_id.split(',') if eid.strip()]

                for entry_id in entry_ids:
                    # Update local DB
                    logger.info(f"Updating DB for entry_id={entry_id} with chord_repo={chord_repo_full}")
                    result = legato_db.execute(
                        """
                        UPDATE knowledge_entries
                        SET chord_status = 'active',
                            chord_repo = ?,
                            needs_chord = 0,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE entry_id = ?
                        """,
                        (chord_repo_full, entry_id)
                    )
                    logger.info(f"DB update affected {result.rowcount} rows for entry_id={entry_id}")

                    # Update GitHub frontmatter
                    try:
                        entry = legato_db.execute(
                            "SELECT file_path FROM knowledge_entries WHERE entry_id = ?",
                            (entry_id,)
                        ).fetchone()

                        if entry and entry['file_path']:
                            file_path = entry['file_path']
                            from .core import get_user_library_repo
                            from .auth import get_user_installation_token
                            library_repo = get_user_library_repo()
                            token = get_user_installation_token(user_id, 'library') if user_id else None

                            logger.info(f"Got installation token for frontmatter update: {bool(token)}")
                            if not token:
                                logger.warning(f"No token available to update frontmatter for {entry_id}")
                                continue

                            content = get_file_content(library_repo, file_path, token)
                            if content and content.startswith('---'):
                                parts = content.split('---', 2)
                                if len(parts) >= 3:
                                    frontmatter = parts[1]
                                    body = parts[2]

                                    # Remove old chord fields if present
                                    new_frontmatter = re.sub(r'^needs_chord:.*\n?', '', frontmatter, flags=re.MULTILINE)
                                    new_frontmatter = re.sub(r'^chord_status:.*\n?', '', new_frontmatter, flags=re.MULTILINE)
                                    new_frontmatter = re.sub(r'^chord_repo:.*\n?', '', new_frontmatter, flags=re.MULTILINE)
                                    new_frontmatter = re.sub(r'^chord_id:.*\n?', '', new_frontmatter, flags=re.MULTILINE)

                                    # Add new chord fields (needs_chord: false since it's now active)
                                    new_frontmatter = new_frontmatter.rstrip() + f"\nneeds_chord: false\nchord_status: active\nchord_repo: {chord_repo_full}\n"

                                    new_content = f'---{new_frontmatter}---{body}'
                                    commit_file(
                                        repo=library_repo,
                                        path=file_path,
                                        content=new_content,
                                        message=f'Link to chord: {chord_repo_name}',
                                        token=token
                                    )
                                    logger.info(f"Updated frontmatter for {entry_id} with chord link")
                    except Exception as e:
                        logger.warning(f"Could not update frontmatter for {entry_id}: {e}")

                legato_db.commit()
                logger.info(f"Updated {len(entry_ids)} Library entries with chord_status=active, chord_repo={chord_repo_full}")
            except Exception as e:
                logger.warning(f"Failed to update Library entries {related_entry_id}: {e}")

        logger.info(f"Approved agent: {queue_id} by {username}")

        return jsonify({
            'status': 'approved',
            'queue_id': queue_id,
            'dispatch_sent': dispatch_result.get('success', False),
        })

    except Exception as e:
        logger.error(f"Failed to approve agent: {e}")
        return jsonify({'error': str(e)}), 500


@agents_bp.route('/api/<queue_id>/retry', methods=['POST'])
@login_required
@paid_required
@copilot_required
def api_retry_spawn(queue_id: str):
    """Retry spawning a failed chord.

    Only works for agents with status='spawn_failed'.

    Response:
    {
        "status": "spawned" | "spawn_failed",
        "queue_id": "aq-abc123",
        "error": "..." (if failed)
    }
    """
    try:
        user = session.get('user', {})
        user_id = user.get('user_id')

        # Check OAuth token is available (chord spawning requires OAuth)
        if user_id:
            from .auth import _get_user_oauth_token
            oauth_token = _get_user_oauth_token(user_id)

            if not oauth_token:
                return jsonify({
                    'error': 'GitHub authorization expired. Please re-authenticate to retry spawning.',
                    'needs_reauth': True,
                    'reauth_url': '/auth/github-app-login'
                }), 401

        agents_db = get_db()
        username = user.get('username', 'unknown')
        org = user.get('username')

        # Get the failed agent - MUST belong to current user
        row = agents_db.execute(
            "SELECT * FROM agent_queue WHERE queue_id = ? AND status = 'spawn_failed' AND user_id = ?",
            (queue_id, user_id)
        ).fetchone()

        if not row:
            return jsonify({'error': 'Agent not found or not in failed state'}), 404

        agent = dict(row)

        # Retry spawn
        dispatch_result = trigger_spawn_workflow(agent, user_id=user_id)

        spawn_success = dispatch_result.get('success', False)
        spawn_error = dispatch_result.get('error', '')
        new_status = 'spawned' if spawn_success else 'spawn_failed'

        # Check if failure was due to token issues - prompt reauth
        if not spawn_success and ('re-authenticate' in spawn_error.lower() or '401' in spawn_error):
            session.pop('github_token', None)
            return jsonify({
                'error': 'GitHub authorization expired. Please re-authenticate to retry spawning.',
                'needs_reauth': True,
                'reauth_url': '/auth/github-app-login'
            }), 401

        # Update status
        agents_db.execute(
            """
            UPDATE agent_queue
            SET status = ?,
                spawn_result = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE queue_id = ?
            """,
            (new_status, json.dumps(dispatch_result), queue_id)
        )
        agents_db.commit()

        # If successful, update Library entries
        if spawn_success:
            related_entry_id = agent.get('related_entry_id')
            if related_entry_id:
                try:
                    legato_db = get_legato_db()
                    chord_repo_name = f"{agent['project_name']}.Chord"
                    chord_repo_full = f"{org}/{chord_repo_name}"

                    entry_ids = [eid.strip() for eid in related_entry_id.split(',') if eid.strip()]
                    for entry_id in entry_ids:
                        legato_db.execute(
                            """
                            UPDATE knowledge_entries
                            SET chord_status = 'active',
                                chord_repo = ?,
                                needs_chord = 0,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE entry_id = ?
                            """,
                            (chord_repo_full, entry_id)
                        )
                    legato_db.commit()
                    logger.info(f"Updated {len(entry_ids)} Library entries on retry spawn")
                except Exception as e:
                    logger.error(f"Failed to update Library entries on retry: {e}")

        logger.info(f"Retry spawn for {queue_id}: {new_status}")

        response = {
            'status': new_status,
            'queue_id': queue_id,
        }
        if not spawn_success:
            response['error'] = dispatch_result.get('error', 'Unknown error')
            response['needs_reauth'] = 'Bad credentials' in str(dispatch_result.get('error', ''))

        return jsonify(response)

    except Exception as e:
        logger.error(f"Failed to retry spawn: {e}")
        return jsonify({'error': str(e)}), 500


@agents_bp.route('/api/reject-all', methods=['POST'])
@login_required
@paid_required
def api_reject_all():
    """Reject all pending agents (mark as rejected, not delete).

    This keeps the rejection records so agents won't be re-queued on next sync.

    Response:
    {
        "status": "cleared",
        "count": 5
    }
    """
    import re
    from .rag.github_service import get_file_content, commit_file
    from .auth import get_user_installation_token

    try:
        db = get_db()
        user = session.get('user', {})
        user_id = user.get('user_id')
        username = user.get('username', 'unknown')

        # Get user token for GitHub operations
        token = get_user_installation_token(user_id, 'library') if user_id else None
        if not token:
            return jsonify({'error': 'GitHub authorization required'}), 401

        from .core import get_user_library_repo
        library_repo = get_user_library_repo()

        # Get all pending agents for THIS USER with their linked entries
        pending = db.execute(
            "SELECT queue_id, related_entry_id FROM agent_queue WHERE status = 'pending' AND user_id = ?",
            (user_id,)
        ).fetchall()

        count = len(pending)

        # Collect all entry IDs to reset
        all_entry_ids = []
        for agent in pending:
            if agent['related_entry_id']:
                entry_ids = [eid.strip() for eid in agent['related_entry_id'].split(',') if eid.strip()]
                all_entry_ids.extend(entry_ids)

        # Mark all as rejected for THIS USER (NOT delete - keeps record to prevent re-queueing)
        db.execute(
            """
            UPDATE agent_queue
            SET status = 'rejected',
                approved_by = ?,
                approved_at = CURRENT_TIMESTAMP,
                spawn_result = '{"rejected": true, "reason": "bulk reject"}',
                updated_at = CURRENT_TIMESTAMP
            WHERE status = 'pending' AND user_id = ?
            """,
            (username, user_id)
        )
        db.commit()

        # Reset chord_status AND needs_chord on all linked notes
        if all_entry_ids:
            legato_db = get_legato_db()
            for entry_id in all_entry_ids:
                # Update local DB
                legato_db.execute(
                    """
                    UPDATE knowledge_entries
                    SET chord_status = 'rejected', needs_chord = 0
                    WHERE entry_id = ?
                    """,
                    (entry_id,)
                )

                # Update GitHub frontmatter to remove needs_chord
                if token:
                    try:
                        entry = legato_db.execute(
                            "SELECT file_path FROM knowledge_entries WHERE entry_id = ?",
                            (entry_id,)
                        ).fetchone()

                        if entry and entry['file_path']:
                            file_path = entry['file_path']
                            content = get_file_content(library_repo, file_path, token)

                            if content and content.startswith('---'):
                                parts = content.split('---', 2)
                                if len(parts) >= 3:
                                    frontmatter = parts[1]
                                    body = parts[2]

                                    # Remove chord-related fields
                                    new_frontmatter = re.sub(r'^needs_chord:.*\n?', '', frontmatter, flags=re.MULTILINE)
                                    new_frontmatter = re.sub(r'^chord_name:.*\n?', '', new_frontmatter, flags=re.MULTILINE)
                                    new_frontmatter = re.sub(r'^chord_scope:.*\n?', '', new_frontmatter, flags=re.MULTILINE)
                                    new_frontmatter = re.sub(r'^chord_status:.*\n?', '', new_frontmatter, flags=re.MULTILINE)
                                    new_frontmatter = re.sub(r'^chord_repo:.*\n?', '', new_frontmatter, flags=re.MULTILINE)
                                    new_frontmatter = re.sub(r'^chord_id:.*\n?', '', new_frontmatter, flags=re.MULTILINE)

                                    if new_frontmatter != frontmatter:
                                        new_content = f'---{new_frontmatter}---{body}'
                                        commit_file(
                                            repo=library_repo,
                                            path=file_path,
                                            content=new_content,
                                            message=f'Reset chord fields: bulk rejection',
                                            token=token
                                        )
                                        logger.info(f"Updated frontmatter for {entry_id}: removed chord fields")
                    except Exception as e:
                        logger.warning(f"Could not update frontmatter for {entry_id}: {e}")

            legato_db.commit()
            logger.info(f"Reset chord_status and needs_chord for {len(all_entry_ids)} entries")

        logger.info(f"Rejected {count} pending agents by {username}")

        return jsonify({
            'status': 'cleared',
            'count': count,
        })

    except Exception as e:
        logger.error(f"Failed to reject pending agents: {e}")
        return jsonify({'error': str(e)}), 500


@agents_bp.route('/api/<queue_id>/comments', methods=['POST'])
@login_required
def api_add_comment(queue_id: str):
    """Add a comment to an agent (max 5 total).

    Request body:
    {
        "text": "Comment text here"
    }

    Response:
    {
        "status": "added",
        "comments": [...]
    }
    """
    try:
        db = get_db()
        data = request.get_json() or {}
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'Comment text is required'}), 400

        user = session.get('user', {})
        user_id = user.get('user_id')
        username = user.get('username', 'unknown')

        # Get current comments - MUST belong to current user
        agent = db.execute(
            "SELECT comments FROM agent_queue WHERE queue_id = ? AND status = 'pending' AND user_id = ?",
            (queue_id, user_id)
        ).fetchone()

        if not agent:
            return jsonify({'error': 'Agent not found or already processed'}), 404

        # Parse existing comments
        try:
            comments = json.loads(agent['comments']) if agent['comments'] else []
        except (json.JSONDecodeError, TypeError):
            comments = []

        # Check limit
        if len(comments) >= 5:
            return jsonify({'error': 'Maximum 5 comments allowed'}), 400

        # Add new comment
        comments.append({
            "text": text,
            "author": username,
            "timestamp": datetime.now().isoformat() + "Z"
        })

        # Save - include user_id check for safety
        db.execute(
            "UPDATE agent_queue SET comments = ?, updated_at = CURRENT_TIMESTAMP WHERE queue_id = ? AND user_id = ?",
            (json.dumps(comments), queue_id, user_id)
        )
        db.commit()

        return jsonify({
            'status': 'added',
            'comments': comments
        })

    except Exception as e:
        logger.error(f"Failed to add comment: {e}")
        return jsonify({'error': str(e)}), 500


@agents_bp.route('/api/<queue_id>/reject', methods=['POST'])
@login_required
def api_reject_agent(queue_id: str):
    """Reject an agent (won't spawn).

    Request body (optional):
    {
        "reason": "Not needed"
    }

    Response:
    {
        "status": "rejected",
        "queue_id": "aq-abc123"
    }
    """
    from .auth import get_user_installation_token

    try:
        db = get_db()
        data = request.get_json() or {}
        reason = data.get('reason', '')

        user = session.get('user', {})
        user_id = user.get('user_id')
        username = user.get('username', 'unknown')

        # Get the agent's related_entry_id before rejecting - MUST belong to current user
        agent = db.execute(
            "SELECT related_entry_id FROM agent_queue WHERE queue_id = ? AND user_id = ?",
            (queue_id, user_id)
        ).fetchone()

        if not agent:
            return jsonify({'error': 'Agent not found or not authorized'}), 404

        related_entry_id = agent['related_entry_id']

        # Update status to rejected - include user_id check for safety
        cursor = db.execute(
            """
            UPDATE agent_queue
            SET status = 'rejected',
                approved_by = ?,
                approved_at = CURRENT_TIMESTAMP,
                spawn_result = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE queue_id = ? AND status = 'pending' AND user_id = ?
            """,
            (username, json.dumps({'rejected': True, 'reason': reason}), queue_id, user_id)
        )
        db.commit()

        if cursor.rowcount == 0:
            return jsonify({'error': 'Agent not found or already processed'}), 404

        # Reset chord_status to 'rejected' AND needs_chord=0 on linked notes to prevent re-queueing
        if related_entry_id:
            import re
            from .rag.github_service import get_file_content, commit_file
            from .core import get_user_library_repo

            legato_db = get_legato_db()
            token = get_user_installation_token(user_id, 'library') if user_id else None
            library_repo = get_user_library_repo()
            entry_ids = [eid.strip() for eid in related_entry_id.split(',') if eid.strip()]

            for entry_id in entry_ids:
                # Update local DB - set chord_status='rejected' to prevent re-queueing
                legato_db.execute(
                    """
                    UPDATE knowledge_entries
                    SET chord_status = 'rejected', needs_chord = 0
                    WHERE entry_id = ?
                    """,
                    (entry_id,)
                )

                # Update GitHub frontmatter to remove needs_chord
                try:
                    entry = legato_db.execute(
                        "SELECT file_path FROM knowledge_entries WHERE entry_id = ?",
                        (entry_id,)
                    ).fetchone()

                    if entry and entry['file_path'] and token:
                        file_path = entry['file_path']
                        content = get_file_content(library_repo, file_path, token)

                        if content and content.startswith('---'):
                            parts = content.split('---', 2)
                            if len(parts) >= 3:
                                frontmatter = parts[1]
                                body = parts[2]

                                # Remove chord-related fields from frontmatter
                                new_frontmatter = re.sub(r'^needs_chord:.*\n?', '', frontmatter, flags=re.MULTILINE)
                                new_frontmatter = re.sub(r'^chord_name:.*\n?', '', new_frontmatter, flags=re.MULTILINE)
                                new_frontmatter = re.sub(r'^chord_scope:.*\n?', '', new_frontmatter, flags=re.MULTILINE)
                                new_frontmatter = re.sub(r'^chord_status:.*\n?', '', new_frontmatter, flags=re.MULTILINE)
                                new_frontmatter = re.sub(r'^chord_repo:.*\n?', '', new_frontmatter, flags=re.MULTILINE)
                                new_frontmatter = re.sub(r'^chord_id:.*\n?', '', new_frontmatter, flags=re.MULTILINE)

                                if new_frontmatter != frontmatter:
                                    new_content = f'---{new_frontmatter}---{body}'
                                    commit_file(
                                        repo=library_repo,
                                        path=file_path,
                                        content=new_content,
                                        message=f'Reset chord fields: agent rejected',
                                        token=token
                                    )
                                    logger.info(f"Updated frontmatter for {entry_id}: removed chord fields")
                except Exception as e:
                    logger.warning(f"Could not update frontmatter for {entry_id}: {e}")

            legato_db.commit()
            logger.info(f"Reset chord_status and needs_chord for {len(entry_ids)} entries after rejection")

        logger.info(f"Rejected agent: {queue_id} by {username}")

        return jsonify({
            'status': 'rejected',
            'queue_id': queue_id,
        })

    except Exception as e:
        logger.error(f"Failed to reject agent: {e}")
        return jsonify({'error': str(e)}), 500


@agents_bp.route('/api/<queue_id>', methods=['GET'])
@login_required
def api_get_agent(queue_id: str):
    """Get details of a specific queued agent."""
    try:
        db = get_db()
        user_id = session.get('user', {}).get('user_id')

        # Get agent - MUST belong to current user
        row = db.execute(
            "SELECT * FROM agent_queue WHERE queue_id = ? AND user_id = ?",
            (queue_id, user_id)
        ).fetchone()

        if not row:
            return jsonify({'error': 'Agent not found'}), 404

        agent = dict(row)
        # Parse JSON fields
        if agent.get('signal_json'):
            try:
                agent['signal_json'] = json.loads(agent['signal_json'])
            except json.JSONDecodeError:
                pass
        if agent.get('spawn_result'):
            try:
                agent['spawn_result'] = json.loads(agent['spawn_result'])
            except json.JSONDecodeError:
                pass

        return jsonify(agent)

    except Exception as e:
        logger.error(f"Failed to get agent: {e}")
        return jsonify({'error': str(e)}), 500


# ============ GitHub Artifact Sync ============

def fetch_conduct_workflow_runs(token: str, org: str, repo: str, limit: int = 10) -> list:
    """Fetch recent process-transcript workflow runs from Conduct.

    Args:
        token: GitHub PAT
        org: GitHub org
        repo: Conduct repo name
        limit: Max runs to fetch

    Returns:
        List of workflow run dicts
    """
    try:
        response = requests.get(
            f'https://api.github.com/repos/{org}/{repo}/actions/workflows/process-transcript.yml/runs',
            params={'per_page': limit, 'status': 'completed'},
            headers={
                'Authorization': f'Bearer {token}',
                'Accept': 'application/vnd.github+json',
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        return data.get('workflow_runs', [])
    except requests.RequestException as e:
        logger.error(f"Failed to fetch workflow runs: {e}")
        return []


def fetch_routing_artifact(token: str, org: str, repo: str, run_id: int) -> dict | None:
    """Download and parse routing-decisions artifact from a workflow run.

    Args:
        token: GitHub PAT
        org: GitHub org
        repo: Conduct repo name
        run_id: Workflow run ID

    Returns:
        Parsed routing.json dict, or None if not found
    """
    import zipfile
    import io

    try:
        # List artifacts for the run
        response = requests.get(
            f'https://api.github.com/repos/{org}/{repo}/actions/runs/{run_id}/artifacts',
            headers={
                'Authorization': f'Bearer {token}',
                'Accept': 'application/vnd.github+json',
            },
            timeout=15,
        )
        response.raise_for_status()
        artifacts = response.json().get('artifacts', [])

        # Find routing-decisions artifact
        routing_artifact = None
        for artifact in artifacts:
            if artifact['name'] == 'routing-decisions':
                routing_artifact = artifact
                break

        if not routing_artifact:
            return None

        # Download the artifact (it's a zip file)
        download_url = routing_artifact['archive_download_url']
        response = requests.get(
            download_url,
            headers={
                'Authorization': f'Bearer {token}',
                'Accept': 'application/vnd.github+json',
            },
            timeout=30,
        )
        response.raise_for_status()

        # Extract routing.json from zip
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            with zf.open('routing.json') as f:
                return json.load(f)

    except Exception as e:
        logger.error(f"Failed to fetch routing artifact for run {run_id}: {e}")
        return None


def import_projects_from_routing(routing: list, run_id: int, db) -> dict:
    """Import CHORD projects from routing data into agent_queue.

    Only imports items with project_scope='chord'. Notes go to the Library,
    not the agent queue. Chords are queued for user approval before spawning.

    Uses sync_history table to track processed items - this persists even when
    the agent queue is cleared, preventing duplicate imports.

    Args:
        routing: Parsed routing.json list
        run_id: Workflow run ID (for source tracking)
        db: Database connection

    Returns:
        Dict with import stats
    """
    stats = {'found': 0, 'imported': 0, 'skipped': 0, 'skipped_notes': 0, 'errors': 0}

    for item in routing:
        if item.get('type') != 'PROJECT':
            continue

        item_id = item.get('id', 'unknown')

        # Check sync_history first - this persists even when queue is cleared
        already_processed = db.execute(
            "SELECT id FROM sync_history WHERE run_id = ? AND item_id = ?",
            (run_id, item_id)
        ).fetchone()

        if already_processed:
            stats['skipped'] += 1
            continue

        # Only queue chords - notes go to Library, not agent queue
        # A note can be escalated to a chord later via the from-entry API
        project_scope = item.get('project_scope', '').lower()
        if project_scope != 'chord':
            # Record in sync_history so we don't check this item again
            db.execute(
                "INSERT OR IGNORE INTO sync_history (run_id, item_id) VALUES (?, ?)",
                (run_id, item_id)
            )
            stats['skipped_notes'] += 1
            continue

        stats['found'] += 1

        project_name = item.get('project_name', 'unnamed')
        source_id = f"conduct-run:{run_id}:{item_id}"

        try:
            # Build tasker body
            description = item.get('project_description') or item.get('description') or ''
            raw_text = item.get('raw_text', '')[:500]

            tasker_body = f"""## Tasker: {item.get('knowledge_title') or item.get('title', 'Untitled')}

### Context
From voice transcript:
"{raw_text}"

### Objective
{description or 'Implement the project as described.'}

### Acceptance Criteria
{chr(10).join(f"- [ ] {kp}" for kp in item.get('key_phrases', [])[:5]) or '- [ ] Core functionality implemented'}

### Constraints
- Follow patterns in `copilot-instructions.md`
- Reference `SIGNAL.md` for project intent
- Keep PRs focused and reviewable

### References
- Source: Conduct workflow run {run_id}
- Thread: {item.get('id', 'unknown')}

---
*Generated from Conduct pipeline*
"""

            project_scope = item.get('project_scope', 'chord')
            repo_suffix = "Chord" if project_scope == "chord" else "Note"
            signal_json = {
                "id": f"lab.{project_scope}.{project_name}",
                "type": "project",
                "source": "conduct",
                "category": project_scope,
                "title": item.get('knowledge_title') or item.get('title', 'Untitled'),
                "domain_tags": item.get('domain_tags', []),
                "key_phrases": item.get('key_phrases', []),
                "path": f"{project_name}.{repo_suffix}",
            }

            queue_id = generate_queue_id()
            related_knowledge_id = item.get('related_knowledge_id')

            db.execute(
                """
                INSERT INTO agent_queue
                (queue_id, project_name, project_type, title, description,
                 signal_json, tasker_body, source_transcript, related_entry_id, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                """,
                (
                    queue_id,
                    project_name,
                    item.get('project_scope', 'chord'),
                    item.get('knowledge_title') or item.get('title', 'Untitled'),
                    description[:500],
                    json.dumps(signal_json),
                    tasker_body,
                    source_id,
                    related_knowledge_id,
                )
            )

            # Record in sync_history so we don't re-import if queue is cleared
            db.execute(
                "INSERT OR IGNORE INTO sync_history (run_id, item_id) VALUES (?, ?)",
                (run_id, item_id)
            )

            stats['imported'] += 1
            logger.info(f"Imported project from Conduct: {queue_id} - {project_name}")

        except Exception as e:
            logger.error(f"Failed to import project {project_name}: {e}")
            stats['errors'] += 1

    db.commit()
    return stats


def import_chords_from_library(legato_db, agents_db, user_id: str = None) -> dict:
    """Import chord escalations from Library entries into agent_queue.

    Finds Library entries with needs_chord=1 and chord_status IS NULL,
    and queues them for user approval. After queueing, updates the
    Library entry's chord_status to 'pending'.

    Args:
        legato_db: Legato database connection (knowledge entries)
        agents_db: Agents database connection (agent queue)
        user_id: The user's ID (required for multi-tenant isolation)

    Returns:
        Dict with import stats
    """
    if not user_id:
        logger.error("import_chords_from_library called without user_id - skipping")
        return {'found': 0, 'queued': 0, 'already_queued': 0, 'errors': 0, 'multi_note_chords': 0}
    stats = {'found': 0, 'queued': 0, 'already_queued': 0, 'errors': 0, 'multi_note_chords': 0}

    # Find entries that need chord escalation
    # Only include entries where:
    # - needs_chord = 1 (flagged for chord)
    # - chord_status is NULL or empty (not yet processed, not rejected, not active)
    entries = legato_db.execute(
        """
        SELECT entry_id, title, category, content, chord_name, chord_scope, file_path, source_transcript
        FROM knowledge_entries
        WHERE needs_chord = 1
        AND (chord_status IS NULL OR chord_status = '')
        """
    ).fetchall()

    stats['found'] = len(entries)

    # Group entries by source_transcript for multi-note chord support
    # If multiple notes from the same transcript all need chords, they're related
    from collections import defaultdict
    transcript_groups = defaultdict(list)
    for entry in entries:
        entry = dict(entry)
        # Group by source_transcript (notes from same transcript are related)
        source = entry.get('source_transcript') or 'unknown'
        transcript_groups[source].append(entry)

    # Process each transcript group
    chord_groups = {}
    for source, group_entries in transcript_groups.items():
        if len(group_entries) == 1:
            # Single entry - use its chord_name
            entry = group_entries[0]
            chord_name = entry['chord_name'] or entry['entry_id'].split('-')[-1][:20]
        else:
            # Multiple entries from same transcript - create combined chord name
            # Use the most specific chord_name or create a combined one
            chord_names = [e.get('chord_name') for e in group_entries if e.get('chord_name')]
            if chord_names:
                chord_name = f"{chord_names[0]}-multi"
            else:
                # Derive from source transcript
                chord_name = f"{source.replace('dropbox-', '')[:15]}-multi"

        chord_groups[chord_name] = group_entries

    for chord_name, group_entries in chord_groups.items():
        # Collect all entry IDs in this group
        entry_ids = [e['entry_id'] for e in group_entries]
        related_entry_id = ','.join(entry_ids)  # Comma-separated for multi-note

        # Check if any entry is already queued FOR THIS USER
        already_queued = False
        for entry_id in entry_ids:
            existing = agents_db.execute(
                "SELECT queue_id, status FROM agent_queue WHERE related_entry_id LIKE ? AND user_id = ?",
                (f'%{entry_id}%', user_id)
            ).fetchone()
            if existing:
                already_queued = True
                stats['already_queued'] += 1
                if existing['status'] == 'rejected':
                    legato_db.execute(
                        "UPDATE knowledge_entries SET needs_chord = 0 WHERE entry_id = ?",
                        (entry_id,)
                    )
                    legato_db.commit()

        if already_queued:
            continue

        # Mark ALL entries as pending
        for entry_id in entry_ids:
            legato_db.execute(
                """
                UPDATE knowledge_entries
                SET chord_status = 'pending', chord_id = ?
                WHERE entry_id = ? AND (chord_status IS NULL OR chord_status = '')
                """,
                (f"lab.chord.{chord_name}", entry_id)
            )
        legato_db.commit()

        # Track multi-note chords
        if len(group_entries) > 1:
            stats['multi_note_chords'] += 1

        try:
            # Build combined tasker body from all entries
            primary_entry = group_entries[0]

            if len(group_entries) == 1:
                # Single note chord
                content_preview = primary_entry['content'][:500] if primary_entry['content'] else ''
                tasker_body = f"""## Tasker: {primary_entry['title']}

### Context
From Library entry `{primary_entry['entry_id']}`:
"{content_preview}"

### Objective
Implement the project as described in the knowledge entry.

### Acceptance Criteria
- [ ] Core functionality implemented
- [ ] Documentation updated
- [ ] Tests written

### Constraints
- Follow patterns in `copilot-instructions.md`
- Reference `SIGNAL.md` for project intent
- Keep PRs focused and reviewable

### References
- Source entry: `{primary_entry['entry_id']}`
- Category: {primary_entry.get('category', 'general')}

---
*Generated from Library entry | needs_chord escalation*
"""
            else:
                # Multi-note chord - combine all entries
                titles = [e['title'] for e in group_entries]
                combined_title = f"Multi-note: {titles[0]} (+{len(titles)-1} related)"

                context_sections = []
                for e in group_entries:
                    preview = e['content'][:300] if e['content'] else ''
                    context_sections.append(f"**{e['title']}** (`{e['entry_id']}`):\n\"{preview}\"")

                tasker_body = f"""## Tasker: {combined_title}

### Context
This chord combines {len(group_entries)} related notes:

{chr(10).join(context_sections)}

### Objective
Implement a unified solution addressing all related notes above.

### Acceptance Criteria
- [ ] Core functionality addresses all {len(group_entries)} notes
- [ ] Documentation updated
- [ ] Tests written

### Constraints
- Follow patterns in `copilot-instructions.md`
- Reference `SIGNAL.md` for project intent
- Keep PRs focused and reviewable

### References
- Source entries: {', '.join(f'`{e["entry_id"]}`' for e in group_entries)}
- Categories: {', '.join(set(e.get('category', 'general') for e in group_entries))}

---
*Generated from {len(group_entries)} Library entries | multi-note chord*
"""

            chord_scope = primary_entry.get('chord_scope', 'chord')
            repo_suffix = "Chord" if chord_scope == "chord" else "Note"
            signal_json = {
                "id": f"lab.{chord_scope}.{chord_name}",
                "type": "project",
                "source": "library-escalation",
                "category": chord_scope,
                "title": primary_entry['title'] if len(group_entries) == 1 else f"Multi-note: {primary_entry['title']}",
                "domain_tags": [],
                "intent": primary_entry['content'][:200] if primary_entry['content'] else '',
                "key_phrases": [],
                "entry_count": len(group_entries),
                "path": f"{chord_name}.{repo_suffix}",
            }

            queue_id = generate_queue_id()

            agents_db.execute(
                """
                INSERT INTO agent_queue
                (queue_id, project_name, project_type, title, description,
                 signal_json, tasker_body, source_transcript, related_entry_id, status, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    queue_id,
                    chord_name,
                    primary_entry.get('chord_scope', 'chord'),
                    signal_json['title'],
                    primary_entry['content'][:500] if primary_entry['content'] else '',
                    json.dumps(signal_json),
                    tasker_body,
                    f"library:{primary_entry['entry_id']}",
                    related_entry_id,  # Comma-separated for multi-note
                    user_id,  # Multi-tenant: isolate by user
                )
            )
            agents_db.commit()

            stats['queued'] += 1
            if len(group_entries) > 1:
                logger.info(f"Queued multi-note chord: {queue_id} - {chord_name} ({len(group_entries)} notes)")
            else:
                logger.info(f"Queued chord from Library: {queue_id} - {chord_name} (from {primary_entry['entry_id']})")

        except Exception as e:
            logger.error(f"Failed to queue chord for {chord_name} ({len(group_entries)} notes): {e}")
            stats['errors'] += 1

    return stats


@agents_bp.route('/api/sync', methods=['POST'])
@login_required
def api_sync_from_library():
    """Sync pending chord escalations from Library.

    Checks Library entries for needs_chord=true and chord_status=null,
    and queues them for user approval.

    Response:
    {
        "status": "synced",
        "chords_found": 2,
        "chords_queued": 1,
        "already_queued": 1
    }
    """
    try:
        user_id = session.get('user', {}).get('user_id')
        legato_db = get_legato_db()
        agents_db = get_db()

        stats = import_chords_from_library(legato_db, agents_db, user_id)

        logger.info(f"Library chord sync complete: {stats}")

        return jsonify({
            'status': 'synced',
            'chords_found': stats['found'],
            'chords_queued': stats['queued'],
            'already_queued': stats['already_queued'],
            'errors': stats['errors'],
        })

    except Exception as e:
        logger.error(f"Library chord sync failed: {e}")
        return jsonify({'error': str(e)}), 500


def trigger_spawn_workflow(agent: dict, user_id: str = None) -> dict:
    """Spawn a project directly using the chord executor.

    This replaces the old Conduct dispatch workflow. Projects are now
    created directly by Pit using embedded templates.

    Args:
        agent: Agent dict from database
        user_id: User ID for multi-tenant mode (optional)

    Returns:
        Dict with success status and details
    """
    from .chord_executor import spawn_chord

    # Parse signal_json if it's a string
    signal_json = agent.get('signal_json', '{}')
    if isinstance(signal_json, str):
        try:
            signal_json = json.loads(signal_json)
        except json.JSONDecodeError:
            signal_json = {}

    # Fetch source note content if we have a related entry
    # This provides the chord with direct context from the Library note
    source_note_content = None
    source_note_title = None
    related_entry_id = agent.get('related_entry_id')
    if related_entry_id:
        try:
            legato_db = get_legato_db()
            # Handle comma-separated entry IDs (take first one for source note)
            entry_id = related_entry_id.split(',')[0].strip()
            entry = legato_db.execute(
                "SELECT title, content FROM knowledge_entries WHERE entry_id = ?",
                (entry_id,)
            ).fetchone()
            if entry:
                source_note_title = entry['title']
                source_note_content = entry['content']
                logger.info(f"Fetched source note for chord: {source_note_title}")
        except Exception as e:
            logger.warning(f"Could not fetch source note for chord: {e}")

    try:
        result = spawn_chord(
            name=agent['project_name'],
            project_type=agent.get('project_type', 'chord'),
            title=agent.get('title', agent['project_name']),
            description=agent.get('description', ''),
            domain_tags=signal_json.get('domain_tags', []),
            key_phrases=signal_json.get('key_phrases', []),
            source_entry_id=agent.get('related_entry_id'),
            tasker_body=agent.get('tasker_body'),
            user_id=user_id,
            assign_copilot=True,
            source_note_content=source_note_content,
            source_note_title=source_note_title,
        )

        if result.get('success'):
            logger.info(f"Spawned project for {agent['queue_id']}: {result.get('repo_url')}")
            return {
                'success': True,
                'repo_url': result.get('repo_url'),
                'issue_url': result.get('issue_url'),
                'assigned_copilot': result.get('assigned_copilot', False),
            }
        else:
            logger.error(f"Spawn failed for {agent['queue_id']}: {result.get('error')}")
            return {'success': False, 'error': result.get('error', 'Unknown error')}

    except Exception as e:
        logger.error(f"Spawn failed for {agent['queue_id']}: {e}")
        return {'success': False, 'error': str(e)}
