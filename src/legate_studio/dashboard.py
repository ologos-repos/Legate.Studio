"""
Dashboard for Legato.Pit

Displays LEGATO system status, recent notes, and motif processing jobs.
Uses server-side GitHub API calls with user installation tokens.
"""
import logging
from datetime import datetime

import requests
from flask import Blueprint, render_template, current_app, jsonify

from .core import login_required, library_required

logger = logging.getLogger(__name__)

dashboard_bp = Blueprint('dashboard', __name__, url_prefix='/dashboard')

# Repository names
REPOS = {
    'conduct': 'Legato.Conduct',
    'library': 'Legate.Library',
    'listen': 'Legato.Listen'
}


def github_api(endpoint, token=None):
    """Make authenticated GitHub API request.

    In multi-tenant mode, token must be provided - no SYSTEM_PAT fallback.
    """
    if not token:
        logger.warning("No GitHub token available for API request")
        return None

    try:
        response = requests.get(
            f'https://api.github.com{endpoint}',
            headers={
                'Authorization': f'Bearer {token}',
                'Accept': 'application/vnd.github+json'
            },
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"GitHub API error for {endpoint}: {e}")
        return None


def get_system_status():
    """Get status of all LEGATO repositories."""
    # In multi-tenant mode, skip Conduct/Listen status checks
    if current_app.config.get('LEGATO_MODE') == 'multi-tenant':
        return []

    org = current_app.config['LEGATO_ORG']
    statuses = []

    for key, repo_name in REPOS.items():
        try:
            runs = github_api(f'/repos/{org}/{repo_name}/actions/runs?per_page=1')
            if runs and runs.get('workflow_runs'):
                run = runs['workflow_runs'][0]
                if run['status'] in ('in_progress', 'queued'):
                    status = 'running'
                    text = 'Running'
                elif run['conclusion'] == 'success':
                    status = 'success'
                    text = 'Operational'
                elif run['conclusion'] == 'failure':
                    status = 'error'
                    text = 'Failed'
                else:
                    status = 'warning'
                    text = run['conclusion'] or 'Unknown'
            else:
                status = 'success'
                text = 'Operational'
        except Exception as e:
            logger.error(f"Error fetching status for {repo_name}: {e}")
            status = 'error'
            text = 'Unavailable'

        statuses.append({
            'name': repo_name,
            'status': status,
            'text': text
        })

    return statuses


def get_recent_jobs(limit=5):
    """Get recent transcript processing jobs."""
    # In multi-tenant mode, Conduct is not used
    if current_app.config.get('LEGATO_MODE') == 'multi-tenant':
        return []

    org = current_app.config['LEGATO_ORG']
    conduct = current_app.config['CONDUCT_REPO']

    runs = github_api(f'/repos/{org}/{conduct}/actions/workflows/process-transcript.yml/runs?per_page={limit}')

    if not runs or not runs.get('workflow_runs'):
        return []

    jobs = []
    for run in runs['workflow_runs'][:limit]:
        # Determine status
        if run['status'] in ('in_progress', 'queued'):
            status = 'running'
        elif run['conclusion'] == 'success':
            status = 'success'
        elif run['conclusion'] == 'failure':
            status = 'error'
        else:
            status = 'pending'

        # Try to get source/transcript name from various places
        source_name = None

        # Check display_title first (often contains useful info)
        display_title = run.get('display_title', '')

        # For repository_dispatch, check the head_commit message or event payload
        if run.get('event') == 'repository_dispatch':
            # The source is often in the run name for dispatched events
            source_name = run.get('name', '')
            if 'dropbox' in display_title.lower():
                source_name = display_title

        # For workflow_dispatch, display_title often has the input
        elif run.get('event') == 'workflow_dispatch':
            source_name = display_title

        # Fallback: try to extract from head_commit
        if not source_name and run.get('head_commit'):
            commit_msg = run['head_commit'].get('message', '')
            if commit_msg:
                source_name = commit_msg[:50]

        # Format the title with source info
        if source_name and source_name != 'Process Transcript':
            # Truncate long sources
            if len(source_name) > 40:
                source_name = source_name[:37] + '...'
            title = source_name
        else:
            # Use timestamp as fallback identifier
            created = run['created_at'][:16].replace('T', ' ')
            title = f"Transcript @ {created}"

        jobs.append({
            'id': run['id'],
            'title': title,
            'status': status,
            'status_text': run['conclusion'] or run['status'],
            'created_at': run['created_at'],
            'url': run['html_url'],
            'event_type': run.get('event', 'unknown')
        })

    return jobs


def get_recent_motif_jobs(limit=5):
    """Get recent motif processing jobs for the current user.

    Args:
        limit: Maximum number of jobs to return

    Returns:
        List of job dicts with job_id, status, notes_created, created_at, error
    """
    from flask import session
    from .rag.database import init_db

    user = session.get('user')
    if not user or not user.get('user_id'):
        return []

    user_id = user['user_id']
    db = init_db()  # Shared DB has processing_jobs

    rows = db.execute("""
        SELECT job_id, status, result_entry_ids, error_message, created_at, completed_at
        FROM processing_jobs
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    """, (user_id, limit)).fetchall()

    jobs = []
    for row in rows:
        # Count notes created from result_entry_ids (comma-separated)
        result_ids = row['result_entry_ids'] or ''
        notes_created = len([x for x in result_ids.split(',') if x.strip()]) if result_ids else 0

        jobs.append({
            'job_id': row['job_id'][:8],  # Short ID for display
            'status': row['status'],
            'notes_created': notes_created,
            'created_at': row['created_at'],
            'completed_at': row['completed_at'],
            'error': row['error_message'][:50] if row['error_message'] else None
        })

    return jobs


def get_recent_notes(limit: int = 5) -> list:
    """Get most recent notes for the current user.

    Args:
        limit: Maximum number of notes to return

    Returns:
        List of note dicts with entry_id, title, category, created_at
    """
    from .rag.database import get_user_legato_db

    try:
        db = get_user_legato_db()
        rows = db.execute("""
            SELECT entry_id, title, category, created_at
            FROM knowledge_entries
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error fetching recent notes: {e}")
        return []


def get_calendar_preview() -> dict:
    """Get current month calendar data with note counts per day.

    Returns:
        Dict with year, month, month_name, and days (dict of date -> count)
    """
    from .rag.database import get_user_legato_db
    import calendar

    try:
        db = get_user_legato_db()
        today = datetime.now()
        year_month = today.strftime('%Y-%m')

        rows = db.execute("""
            SELECT DATE(created_at) as date, COUNT(*) as count
            FROM knowledge_entries
            WHERE strftime('%Y-%m', created_at) = ?
            GROUP BY DATE(created_at)
        """, (year_month,)).fetchall()

        # Build calendar data
        cal = calendar.Calendar(firstweekday=6)  # Sunday first
        weeks = cal.monthdays2calendar(today.year, today.month)

        return {
            'year': today.year,
            'month': today.month,
            'month_name': today.strftime('%B'),
            'days': {row['date']: row['count'] for row in rows},
            'weeks': weeks,  # List of weeks, each week is list of (day, weekday) tuples
            'today': today.strftime('%Y-%m-%d')
        }
    except Exception as e:
        logger.error(f"Error fetching calendar preview: {e}")
        return {
            'year': datetime.now().year,
            'month': datetime.now().month,
            'month_name': datetime.now().strftime('%B'),
            'days': {},
            'weeks': [],
            'today': datetime.now().strftime('%Y-%m-%d')
        }


def get_stats():
    """Get system statistics from local database and GitHub.

    Returns:
        Dict with transcripts (unique), notes (total), chords (from GitHub)
    """
    from flask import g
    from .rag.database import get_user_legato_db

    stats = {
        'motifs': 0,
        'notes': 0,
        'chords': 0
    }

    try:
        db = get_user_legato_db()

        # Count unique transcripts (deduplicated by source_transcript)
        result = db.execute("""
            SELECT COUNT(DISTINCT source_transcript)
            FROM knowledge_entries
            WHERE source_transcript IS NOT NULL AND source_transcript != ''
        """).fetchone()
        stats['motifs'] = result[0] if result else 0

        # Count total notes
        result = db.execute("SELECT COUNT(*) FROM knowledge_entries").fetchone()
        stats['notes'] = result[0] if result else 0

        # Count chords from GitHub (source of truth)
        # Uses repos with legato-chord topic
        try:
            from flask import session
            from .auth import get_user_installation_token

            user = session.get('user', {})
            user_id = user.get('user_id')
            org = user.get('username')  # User's GitHub username

            token = get_user_installation_token(user_id, 'library') if user_id else None
            if token and org:
                from .chords import fetch_chord_repos
                repos = fetch_chord_repos(token, org)
                stats['chords'] = len(repos)
        except Exception as e:
            logger.warning(f"Could not fetch chord count from GitHub: {e}")
            # Fallback to local DB count
            result = db.execute("""
                SELECT COUNT(DISTINCT chord_repo)
                FROM knowledge_entries
                WHERE chord_status = 'active' AND chord_repo IS NOT NULL
            """).fetchone()
            stats['chords'] = result[0] if result else 0

    except Exception as e:
        logger.error(f"Error fetching stats: {e}")

    return stats


def get_pending_agents():
    """Get pending agents from the queue for current user."""
    from flask import g, session
    from .rag.database import init_agents_db

    try:
        if 'agents_db_conn' not in g:
            g.agents_db_conn = init_agents_db()
        db = g.agents_db_conn

        # Filter by user_id in multi-tenant mode
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

        return [dict(row) for row in rows]

    except Exception as e:
        logger.error(f"Error fetching pending agents: {e}")
        return []


def get_recent_chord_spawns(limit=5):
    """Get recently approved chord spawns from the agent queue for current user.

    Returns chord approvals from local database, which is faster and
    more reliable than querying GitHub workflows.

    Args:
        limit: Maximum number of spawns to return

    Returns:
        List of chord spawn dicts with status info
    """
    import json
    from flask import g, session
    from .rag.database import init_agents_db

    try:
        if 'agents_db_conn' not in g:
            g.agents_db_conn = init_agents_db()
        db = g.agents_db_conn

        # Filter by user_id in multi-tenant mode
        user = session.get('user', {})
        user_id = user.get('user_id')

        rows = db.execute(
            """
            SELECT queue_id, project_name, project_type, title,
                   status, approved_by, approved_at, spawn_result,
                   created_at
            FROM agent_queue
            WHERE status IN ('approved', 'rejected') AND user_id = ?
            ORDER BY approved_at DESC
            LIMIT ?
            """,
            (user_id, limit)
        ).fetchall()

        spawns = []
        org = user.get('username') or current_app.config.get('LEGATO_ORG', 'bobbyhiddn')

        for row in rows:
            row = dict(row)

            # Parse spawn_result for success/failure info
            spawn_success = False
            if row.get('spawn_result'):
                try:
                    result = json.loads(row['spawn_result'])
                    spawn_success = result.get('success', False)
                except (json.JSONDecodeError, TypeError):
                    pass

            # Determine display status
            if row['status'] == 'rejected':
                status = 'rejected'
                status_text = 'Rejected'
            elif spawn_success:
                status = 'success'
                status_text = 'Spawned'
            else:
                status = 'error'
                status_text = 'Failed'

            # Build repo URL for approved chords
            repo_url = None
            if row['status'] == 'approved' and row['project_type'] == 'chord':
                repo_url = f"https://github.com/{org}/{row['project_name']}.Chord"

            spawns.append({
                'id': row['queue_id'],
                'title': row['title'],
                'project_name': row['project_name'],
                'project_type': row['project_type'],
                'status': status,
                'status_text': status_text,
                'approved_by': row['approved_by'],
                'approved_at': row['approved_at'],
                'created_at': row['created_at'],
                'url': repo_url,
            })

        return spawns

    except Exception as e:
        logger.error(f"Error fetching recent chord spawns: {e}")
        return []


@dashboard_bp.route('/')
@library_required
def index():
    """Main dashboard view."""
    return render_template(
        'dashboard.html',
        title='Dashboard',
        recent_notes=get_recent_notes(),
        calendar_preview=get_calendar_preview(),
        recent_chord_spawns=get_recent_chord_spawns(),
        recent_motif_jobs=get_recent_motif_jobs(),
        stats=get_stats(),
        pending_agents=get_pending_agents()
    )


@dashboard_bp.route('/graph3d')
@library_required
def graph3d():
    """3D graph visualization with force/dendrite/radial layouts."""
    return render_template(
        'dashboard_graph3d.html',
        title='3D Knowledge Graph',
        stats=get_stats()
    )


@dashboard_bp.route('/api/status')
@login_required
def api_status():
    """API endpoint for dashboard data (for live updates)."""
    return jsonify({
        'recent_notes': get_recent_notes(),
        'calendar_preview': get_calendar_preview(),
        'recent_chord_spawns': get_recent_chord_spawns(),
        'recent_motif_jobs': get_recent_motif_jobs(),
        'stats': get_stats(),
        'updated_at': datetime.now().isoformat()
    })
