"""
Chords Blueprint for Legato.Pit

Tracks Chord repositories spawned by agents from Notes.
Fetches repos with 'legato-chord' topic from GitHub.
"""

import logging

import requests
from flask import Blueprint, render_template, jsonify, current_app, g

from .core import login_required, library_required, copilot_required

logger = logging.getLogger(__name__)

chords_bp = Blueprint('chords', __name__, url_prefix='/chords')


def get_legato_db():
    """Get legato database connection for current user."""
    from .rag.database import get_user_legato_db
    return get_user_legato_db()


def fetch_chord_repos(token: str, org: str) -> list[dict]:
    """Fetch all Chord repos from GitHub with legato-chord topic.

    Args:
        token: GitHub PAT
        org: GitHub organization

    Returns:
        List of repo data dicts
    """
    repos = []

    # Search for repos with legato-chord topic in the org
    search_url = "https://api.github.com/search/repositories"
    params = {
        "q": f"org:{org} topic:legato-chord",
        "sort": "created",
        "order": "desc",
        "per_page": 50,
    }

    try:
        response = requests.get(
            search_url,
            params=params,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        # Get linked notes from database
        from .rag.database import get_user_legato_db
        try:
            legato_db = get_user_legato_db()
        except Exception:
            legato_db = None

        for repo in data.get("items", []):
            repo_data = {
                "name": repo["name"],
                "full_name": repo["full_name"],
                "description": repo["description"],
                "html_url": repo["html_url"],
                "created_at": repo["created_at"],
                "updated_at": repo["updated_at"],
                "open_issues_count": repo["open_issues_count"],
                "topics": repo.get("topics", []),
                "default_branch": repo.get("default_branch", "main"),
                "linked_notes": [],
            }

            # Look up linked notes from knowledge_entries
            if legato_db:
                try:
                    linked = legato_db.execute(
                        """
                        SELECT entry_id, title, category
                        FROM knowledge_entries
                        WHERE chord_repo = ? OR chord_repo LIKE ?
                        """,
                        (repo["full_name"], f"%/{repo['name']}"),
                    ).fetchall()
                    repo_data["linked_notes"] = [dict(n) for n in linked]
                except Exception as e:
                    logger.warning(f"Failed to fetch linked notes for {repo['full_name']}: {e}")

            repos.append(repo_data)

    except requests.RequestException as e:
        logger.error(f"Failed to fetch chord repos: {e}")

    return repos


def fetch_repo_details(token: str, repo_full_name: str) -> dict:
    """Fetch detailed info for a specific repo including issues and PRs.

    Args:
        token: GitHub PAT
        repo_full_name: Full repo name (org/repo)

    Returns:
        Dict with repo details, issues, PRs
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }

    details = {
        "issues": [],
        "pull_requests": [],
        "recent_commits": [],
    }

    try:
        # Fetch open issues (not PRs)
        issues_resp = requests.get(
            f"https://api.github.com/repos/{repo_full_name}/issues",
            params={"state": "open", "per_page": 10},
            headers=headers,
            timeout=10,
        )
        if issues_resp.ok:
            for issue in issues_resp.json():
                if "pull_request" not in issue:
                    details["issues"].append({
                        "number": issue["number"],
                        "title": issue["title"],
                        "state": issue["state"],
                        "html_url": issue["html_url"],
                        "created_at": issue["created_at"],
                        "labels": [l["name"] for l in issue.get("labels", [])],
                        "assignee": issue["assignee"]["login"] if issue.get("assignee") else None,
                    })

        # Fetch open PRs
        prs_resp = requests.get(
            f"https://api.github.com/repos/{repo_full_name}/pulls",
            params={"state": "open", "per_page": 10},
            headers=headers,
            timeout=10,
        )
        if prs_resp.ok:
            for pr in prs_resp.json():
                details["pull_requests"].append({
                    "number": pr["number"],
                    "title": pr["title"],
                    "state": pr["state"],
                    "html_url": pr["html_url"],
                    "created_at": pr["created_at"],
                    "user": pr["user"]["login"],
                    "draft": pr.get("draft", False),
                })

        # Fetch recent commits
        commits_resp = requests.get(
            f"https://api.github.com/repos/{repo_full_name}/commits",
            params={"per_page": 5},
            headers=headers,
            timeout=10,
        )
        if commits_resp.ok:
            for commit in commits_resp.json():
                details["recent_commits"].append({
                    "sha": commit["sha"][:7],
                    "message": commit["commit"]["message"].split("\n")[0][:80],
                    "author": commit["commit"]["author"]["name"],
                    "date": commit["commit"]["author"]["date"],
                    "html_url": commit["html_url"],
                })

    except requests.RequestException as e:
        logger.error(f"Failed to fetch repo details for {repo_full_name}: {e}")

    return details


@chords_bp.route('/')
@library_required
@copilot_required
def index():
    """Chords overview - list all Chord repos."""
    from flask import session
    from .auth import get_user_installation_token

    user = session.get('user', {})
    user_id = user.get('user_id')
    org = user.get('username')  # User's GitHub username

    token = get_user_installation_token(user_id, 'library') if user_id else None

    # In multi-tenant mode, require user token - don't fall back to SYSTEM_PAT
    # which would show the wrong user's chords
    repos = []
    if token and org:
        repos = fetch_chord_repos(token, org)

    return render_template('chords.html', repos=repos)


@chords_bp.route('/api/repos')
@login_required
@copilot_required
def api_list_repos():
    """API endpoint to list Chord repos."""
    from flask import session
    from .auth import get_user_installation_token

    user = session.get('user', {})
    user_id = user.get('user_id')
    org = user.get('username')

    token = get_user_installation_token(user_id, 'library') if user_id else None

    # Don't fall back to SYSTEM_PAT - would show wrong user's chords
    if not token:
        return jsonify({'error': 'GitHub authorization required'}), 401

    repos = fetch_chord_repos(token, org) if org else []

    return jsonify({
        'repos': repos,
        'count': len(repos),
    })


@chords_bp.route('/api/repo/<path:repo_name>')
@login_required
@copilot_required
def api_repo_details(repo_name: str):
    """API endpoint to get details for a specific repo."""
    from flask import session
    from .auth import get_user_installation_token

    user = session.get('user', {})
    user_id = user.get('user_id')

    token = get_user_installation_token(user_id, 'library') if user_id else None
    # In multi-tenant mode, require user token - don't fall back to SYSTEM_PAT

    if not token:
        return jsonify({'error': 'GitHub authorization required'}), 401

    details = fetch_repo_details(token, repo_name)

    return jsonify(details)


@chords_bp.route('/api/repo/<path:repo_name>', methods=['DELETE'])
@login_required
@copilot_required
def api_delete_repo(repo_name: str):
    """Delete a Chord repository.

    This permanently deletes the GitHub repository and resets any linked
    Library entries to NOT need a chord (needs_chord = false).

    Args:
        repo_name: Full repo name (org/repo)

    Response:
    {
        "success": true,
        "repo": "org/repo-name",
        "notes_reset": 1
    }
    """
    from flask import session
    from .auth import get_user_installation_token

    user = session.get('user', {})
    user_id = user.get('user_id')

    token = get_user_installation_token(user_id, 'library') if user_id else None
    # In multi-tenant mode, require user token - don't fall back to SYSTEM_PAT

    if not token:
        return jsonify({'error': 'GitHub authorization required'}), 401

    try:
        # First, delete the GitHub repository
        response = requests.delete(
            f'https://api.github.com/repos/{repo_name}',
            headers={
                'Authorization': f'Bearer {token}',
                'Accept': 'application/vnd.github+json',
                'X-GitHub-Api-Version': '2022-11-28',
            },
            timeout=15,
        )

        if response.status_code == 204:
            logger.info(f"Deleted chord repository: {repo_name}")
        elif response.status_code == 404:
            logger.warning(f"Repository not found (may already be deleted): {repo_name}")
        elif response.status_code == 403:
            return jsonify({
                'error': 'Insufficient permissions to delete repository',
                'detail': 'Your GitHub installation needs delete_repo scope'
            }), 403
        else:
            return jsonify({
                'error': f'Failed to delete repository: HTTP {response.status_code}',
                'detail': response.text
            }), response.status_code

        # Get linked Library entries before resetting
        db = get_legato_db()
        linked_entries = db.execute(
            """
            SELECT entry_id, file_path, title FROM knowledge_entries
            WHERE chord_repo = ?
            """,
            (repo_name,)
        ).fetchall()

        # Reset linked entries in local DB - set needs_chord = 0 so they don't re-queue
        result = db.execute(
            """
            UPDATE knowledge_entries
            SET chord_status = NULL,
                chord_repo = NULL,
                chord_id = NULL,
                needs_chord = 0,
                updated_at = CURRENT_TIMESTAMP
            WHERE chord_repo = ?
            """,
            (repo_name,)
        )
        notes_reset = result.rowcount
        db.commit()

        # Update GitHub frontmatter for each linked entry
        from .core import get_user_library_repo
        library_repo = get_user_library_repo()
        frontmatter_updated = 0

        for entry in linked_entries:
            file_path = entry['file_path']
            if not file_path:
                continue

            try:
                from .rag.github_service import get_file_content, commit_file
                import re

                content = get_file_content(library_repo, file_path, token)
                if content and content.startswith('---'):
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        frontmatter = parts[1]
                        body = parts[2]

                        # Remove all chord-related fields from frontmatter
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
                                message=f'Reset chord fields: chord deleted',
                                token=token
                            )
                            frontmatter_updated += 1

            except Exception as e:
                logger.warning(f"Could not update frontmatter for {entry['entry_id']}: {e}")

        if notes_reset > 0:
            logger.info(f"Reset chord status on {notes_reset} library entries, {frontmatter_updated} frontmatter updated")

        return jsonify({
            'success': True,
            'repo': repo_name,
            'notes_reset': notes_reset,
        })

    except requests.RequestException as e:
        logger.error(f"Failed to delete repository {repo_name}: {e}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.error(f"Error deleting chord {repo_name}: {e}")
        return jsonify({'error': str(e)}), 500


@chords_bp.route('/api/linked-notes/<path:repo_name>')
@login_required
@copilot_required
def api_linked_notes(repo_name: str):
    """Get Library notes linked to a Chord repository."""
    try:
        db = get_legato_db()
        rows = db.execute(
            """
            SELECT entry_id, title, category, chord_status
            FROM knowledge_entries
            WHERE chord_repo = ?
            """,
            (repo_name,)
        ).fetchall()

        return jsonify({
            'notes': [dict(row) for row in rows],
            'count': len(rows),
        })

    except Exception as e:
        logger.error(f"Failed to get linked notes for {repo_name}: {e}")
        return jsonify({'error': str(e)}), 500


@chords_bp.route('/api/repo/<path:repo_name>/incident', methods=['POST'])
@login_required
@copilot_required
def api_create_incident(repo_name: str):
    """Dispatch an incident to Conduct for an existing Chord repository.

    This allows shooting new tasks at existing chords for Copilot to work.
    The incident is dispatched to Conduct which creates the issue and assigns Copilot.

    Request body:
    {
        "title": "Add feature X",
        "description": "Detailed description of the incident",
        "note_ids": ["kb-abc123"]  // Optional: link existing notes
    }

    Response:
    {
        "success": true,
        "queue_id": "incident-abc123",
        "dispatched": true
    }
    """
    from flask import request, session
    from .auth import get_user_installation_token
    import secrets

    user = session.get('user', {})
    user_id = user.get('user_id')
    org = user.get('username')

    token = get_user_installation_token(user_id, 'library') if user_id else None

    # Don't fall back to SYSTEM_PAT - would use wrong org for incident dispatch
    if not token:
        return jsonify({'error': 'GitHub authorization required'}), 401

    conduct_repo = current_app.config.get('CONDUCT_REPO', 'Legato.Conduct')

    data = request.get_json()

    if not data:
        return jsonify({'error': 'JSON body required'}), 400

    title = data.get('title', '').strip()
    description = data.get('description', '').strip()
    note_ids = data.get('note_ids', [])

    if not title:
        return jsonify({'error': 'title is required'}), 400

    # Look up linked notes if provided
    notes_section = ""
    if note_ids:
        db = get_legato_db()
        notes = []
        for nid in note_ids:
            entry = db.execute(
                "SELECT entry_id, title FROM knowledge_entries WHERE entry_id = ?",
                (nid.strip(),)
            ).fetchone()
            if entry:
                notes.append(dict(entry))

        if notes:
            notes_section = "\n### Linked Notes\n" + "\n".join(
                [f"- **{n['title']}** (`{n['entry_id']}`)" for n in notes]
            )

    # Build tasker body for Conduct
    tasker_body = f"""## Incident: {title}

{description}
{notes_section}

---
*Incident dispatched via Legato Pit UI*
"""

    # Generate queue_id for tracking
    queue_id = f"incident-{secrets.token_hex(6)}"

    # Dispatch to Conduct with target_repo
    payload = {
        'event_type': 'spawn-agent',
        'client_payload': {
            'queue_id': queue_id,
            'target_repo': repo_name,
            'issue_title': title,
            'tasker_body': tasker_body,
        }
    }

    try:
        response = requests.post(
            f'https://api.github.com/repos/{org}/{conduct_repo}/dispatches',
            json=payload,
            headers={
                'Authorization': f'Bearer {token}',
                'Accept': 'application/vnd.github+json',
                'X-GitHub-Api-Version': '2022-11-28',
            },
            timeout=15,
        )

        # 204 No Content = success for repository_dispatch
        if response.status_code == 204:
            logger.info(f"Dispatched incident to Conduct for {repo_name}: {queue_id}")

            return jsonify({
                'success': True,
                'queue_id': queue_id,
                'dispatched': True,
                'message': f'Incident dispatched to Conduct. Copilot will create and work the issue.',
            })
        else:
            logger.error(f"Dispatch failed: {response.status_code} - {response.text}")
            return jsonify({
                'error': f'Failed to dispatch incident: HTTP {response.status_code}',
                'detail': response.text,
            }), response.status_code

    except requests.RequestException as e:
        logger.error(f"Failed to dispatch incident for {repo_name}: {e}")
        return jsonify({'error': str(e)}), 500
