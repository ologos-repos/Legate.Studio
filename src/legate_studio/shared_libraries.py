"""
Shared Libraries Blueprint - Web UI and JSON API for collaborative library management.

Provides routes for:
- Listing shared libraries (owned, member, pending invitations)
- Creating shared libraries (managed tier required)
- Inviting collaborators by GitHub username
- Accepting / declining invitations
- Removing members (owner-only)

Mirrors the MCP tool implementations in mcp_server.py but adapts them for
Flask session auth (session['user']) rather than MCP bearer token auth (g.mcp_user).

Route prefix: /shared/
API prefix:   /shared/api/
"""

import logging
import re
import uuid

import requests
from flask import Blueprint, jsonify, render_template, request, session

from .core import get_effective_tier, login_required

logger = logging.getLogger(__name__)


shared_libraries_bp = Blueprint("shared_libraries", __name__, url_prefix="/shared")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_db():
    """Get the shared legato.db (contains shared_libraries + shared_library_members)."""
    from .rag.database import init_db

    return init_db()


def _current_user() -> dict:
    """Return current session user dict: {user_id, username, tier, ...}."""
    return session.get("user", {})


def _require_managed_tier(user_id: str) -> str | None:
    """Return an error string if user is not on a managed tier, else None."""
    tier = get_effective_tier(user_id)
    allowed = {"managed_lite", "managed_standard", "managed_plus", "beta"}
    if tier not in allowed:
        return (
            "Shared libraries require a managed subscription. "
            "Upgrade at legate.studio/billing to access shared libraries."
        )
    return None


def _generate_slug(name: str) -> str:
    """Generate a URL-safe slug from a library name."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return slug[:50]


# ---------------------------------------------------------------------------
# HTML Pages
# ---------------------------------------------------------------------------


@shared_libraries_bp.route("/", strict_slashes=False)
@login_required
def index():
    """Main shared libraries page. Shows owned, member, and pending invitation libraries."""
    user = _current_user()
    user_id = user.get("user_id")

    try:
        db = _get_db()

        # Libraries where user is an active member (owned + joined)
        library_rows = db.execute(
            """
            SELECT
                sl.id AS library_id,
                sl.name,
                sl.slug,
                sl.repo_full_name,
                sl.description,
                sl.owner_user_id,
                slm.role,
                (
                    SELECT COUNT(*)
                    FROM shared_library_members m2
                    WHERE m2.shared_library_id = sl.id AND m2.status = 'active'
                ) AS member_count
            FROM shared_libraries sl
            JOIN shared_library_members slm ON sl.id = slm.shared_library_id
            WHERE slm.user_id = ? AND slm.status = 'active' AND sl.status = 'active'
            ORDER BY sl.name
            """,
            (user_id,),
        ).fetchall()

        libraries = [dict(row) for row in library_rows]

        # Pending invitations
        invitation_rows = db.execute(
            """
            SELECT sl.id AS library_id, sl.name, sl.slug, sl.description,
                   u.github_login AS owner_login
            FROM shared_libraries sl
            JOIN shared_library_members slm ON sl.id = slm.shared_library_id
            LEFT JOIN users u ON u.user_id = sl.owner_user_id
            WHERE slm.user_id = ? AND slm.status = 'invited' AND sl.status = 'active'
            ORDER BY sl.name
            """,
            (user_id,),
        ).fetchall()

        invitations = [dict(row) for row in invitation_rows]

    except Exception as e:
        logger.error(f"shared_libraries index error for {user_id}: {e}", exc_info=True)
        libraries = []
        invitations = []

    return render_template(
        "shared_libraries.html",
        libraries=libraries,
        invitations=invitations,
        user=user,
    )


@shared_libraries_bp.route("/<library_id>")
@login_required
def detail(library_id: str):
    """Detail page for a single shared library. Shows members and invite form (owners only)."""
    user = _current_user()
    user_id = user.get("user_id")

    db = _get_db()

    # Get library info and verify user is an active member
    lib_row = db.execute(
        """
        SELECT sl.id, sl.name, sl.slug, sl.repo_full_name, sl.description,
               sl.owner_user_id, slm.role
        FROM shared_libraries sl
        JOIN shared_library_members slm ON sl.id = slm.shared_library_id
        WHERE sl.id = ? AND slm.user_id = ? AND slm.status = 'active' AND sl.status = 'active'
        """,
        (library_id, user_id),
    ).fetchone()

    if not lib_row:
        # Check if there's a pending invite
        invited = db.execute(
            "SELECT 1 FROM shared_library_members WHERE shared_library_id = ? AND user_id = ? AND status = 'invited'",
            (library_id, user_id),
        ).fetchone()
        if invited:
            from flask import flash, redirect, url_for
            flash("You have a pending invitation to this library. Accept it first.", "warning")
            return redirect(url_for("shared_libraries.index"))
        from flask import abort
        abort(404)

    library = dict(lib_row)
    is_owner = library["role"] == "owner"

    # Get all active + invited members
    member_rows = db.execute(
        """
        SELECT slm.user_id, slm.role, slm.status, slm.invited_at, slm.accepted_at,
               u.github_login, u.display_name
        FROM shared_library_members slm
        LEFT JOIN users u ON u.user_id = slm.user_id
        WHERE slm.shared_library_id = ? AND slm.status IN ('active', 'invited')
        ORDER BY slm.role DESC, u.github_login
        """,
        (library_id,),
    ).fetchall()

    members = [dict(row) for row in member_rows]

    return render_template(
        "shared_library_detail.html",
        library=library,
        members=members,
        is_owner=is_owner,
        user=user,
    )


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------


@shared_libraries_bp.route("/api/libraries", methods=["GET"])
@login_required
def api_list_libraries():
    """List all libraries (active + pending invitations) for current user.

    Response: {libraries: [...], invitations: [...]}
    """
    user = _current_user()
    user_id = user.get("user_id")

    try:
        db = _get_db()

        library_rows = db.execute(
            """
            SELECT
                sl.id AS library_id,
                sl.name,
                sl.slug,
                sl.repo_full_name,
                sl.description,
                sl.owner_user_id,
                slm.role,
                (
                    SELECT COUNT(*)
                    FROM shared_library_members m2
                    WHERE m2.shared_library_id = sl.id AND m2.status = 'active'
                ) AS member_count
            FROM shared_libraries sl
            JOIN shared_library_members slm ON sl.id = slm.shared_library_id
            WHERE slm.user_id = ? AND slm.status = 'active' AND sl.status = 'active'
            ORDER BY sl.name
            """,
            (user_id,),
        ).fetchall()

        libraries = [dict(row) for row in library_rows]

        invitation_rows = db.execute(
            """
            SELECT sl.id AS library_id, sl.name, sl.slug, sl.description,
                   u.github_login AS owner_login
            FROM shared_libraries sl
            JOIN shared_library_members slm ON sl.id = slm.shared_library_id
            LEFT JOIN users u ON u.user_id = sl.owner_user_id
            WHERE slm.user_id = ? AND slm.status = 'invited' AND sl.status = 'active'
            ORDER BY sl.name
            """,
            (user_id,),
        ).fetchall()

        invitations = [dict(row) for row in invitation_rows]

    except Exception as e:
        logger.error(f"api_list_libraries failed for {user_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

    return jsonify({"libraries": libraries, "invitations": invitations})


@shared_libraries_bp.route("/api/create", methods=["POST"])
@login_required
def api_create_library():
    """Create a new shared library.

    Input:  {name: str, description: str}
    Output: {success: true, library_id, name, slug, repo_full_name}

    GATE: User must be on a managed tier.
    Creates GitHub repo Legate.Library.{slug} (private, auto_init).
    """
    from .auth import get_user_installation_token
    from .rag.database import init_shared_library_db

    user = _current_user()
    user_id = user.get("user_id")
    github_login = user.get("username")

    # Tier gate
    tier_error = _require_managed_tier(user_id)
    if tier_error:
        return jsonify({"error": tier_error}), 403

    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Library name is required"}), 400
    if len(name) > 100:
        return jsonify({"error": "Library name must be 100 characters or less"}), 400

    description = (data.get("description") or "").strip()

    # Generate slug
    slug = _generate_slug(name)
    if not slug or not re.match(r"^[a-z0-9][a-z0-9-]*$", slug):
        return jsonify({
            "error": "Could not generate a valid slug from that name. Use letters, numbers, and hyphens."
        }), 400

    db = _get_db()

    # Check slug uniqueness for this owner
    existing = db.execute(
        "SELECT id FROM shared_libraries WHERE owner_user_id = ? AND slug = ? AND status = 'active'",
        (user_id, slug),
    ).fetchone()
    if existing:
        return jsonify({
            "error": f"You already have a shared library with slug '{slug}'. Choose a different name."
        }), 409

    # Get GitHub token
    token = get_user_installation_token(user_id, "library")
    if not token:
        return jsonify({"error": "GitHub authorization required. Please re-authenticate."}), 401

    library_id = str(uuid.uuid4())
    repo_name = f"Legate.Library.{slug}"
    repo_full_name = f"{github_login}/{repo_name}"

    # Create GitHub repo
    try:
        gh_resp = requests.post(
            "https://api.github.com/user/repos",
            json={
                "name": repo_name,
                "description": description or f"Legate shared library: {name}",
                "private": True,
                "auto_init": True,
            },
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
            timeout=20,
        )
        if gh_resp.status_code == 422:
            errors = gh_resp.json().get("errors", [])
            if any(e.get("message", "").startswith("name already exists") for e in errors):
                return jsonify({
                    "error": f"GitHub repo '{repo_name}' already exists on your account. Choose a different name."
                }), 409
        if not gh_resp.ok:
            logger.error(f"GitHub repo creation failed: {gh_resp.status_code} {gh_resp.text[:500]}")
            return jsonify({"error": f"Failed to create GitHub repo: {gh_resp.status_code}"}), 502
        repo_full_name = gh_resp.json().get("full_name", repo_full_name)
    except requests.RequestException as e:
        logger.error(f"GitHub API error creating shared library repo: {e}")
        return jsonify({"error": f"Failed to create GitHub repo: {str(e)}"}), 502

    # Insert into DB
    try:
        db.execute(
            """
            INSERT INTO shared_libraries (id, name, slug, owner_user_id, repo_full_name, description, status)
            VALUES (?, ?, ?, ?, ?, ?, 'active')
            """,
            (library_id, name, slug, user_id, repo_full_name, description or None),
        )
        db.execute(
            """
            INSERT INTO shared_library_members (shared_library_id, user_id, role, status, accepted_at)
            VALUES (?, ?, 'owner', 'active', CURRENT_TIMESTAMP)
            """,
            (library_id, user_id),
        )
        db.commit()
    except Exception as e:
        logger.error(f"DB error inserting shared library: {e}", exc_info=True)
        return jsonify({"error": f"Failed to save library: {str(e)}"}), 500

    # Initialize per-library SQLite DB (best-effort)
    try:
        init_shared_library_db(library_id)
    except Exception as e:
        logger.warning(f"Shared library DB init deferred for {library_id}: {e}")

    logger.info(f"Created shared library '{name}' ({library_id}) for user {user_id}, repo {repo_full_name}")

    return jsonify({
        "success": True,
        "library_id": library_id,
        "name": name,
        "slug": slug,
        "repo_full_name": repo_full_name,
        "description": description or None,
    })


@shared_libraries_bp.route("/api/<library_id>/invite", methods=["POST"])
@login_required
def api_invite(library_id: str):
    """Invite a GitHub user to collaborate on a shared library.

    Input:  {github_username: str}
    Output: {success: true, github_login, status}
    Only the library owner can invite.
    """
    from .auth import get_user_installation_token

    user = _current_user()
    user_id = user.get("user_id")

    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    github_login = (data.get("github_username") or data.get("github_login") or "").strip()
    if not github_login:
        return jsonify({"error": "github_username is required"}), 400

    db = _get_db()

    # Verify caller is owner
    lib_row = db.execute(
        """
        SELECT sl.repo_full_name, sl.name
        FROM shared_libraries sl
        JOIN shared_library_members slm ON sl.id = slm.shared_library_id
        WHERE sl.id = ? AND slm.user_id = ? AND slm.role = 'owner' AND slm.status = 'active'
        """,
        (library_id, user_id),
    ).fetchone()

    if not lib_row:
        return jsonify({"error": "Library not found or you are not the owner"}), 403

    repo_full_name = lib_row["repo_full_name"]
    library_name = lib_row["name"]

    # Verify target user exists in our system
    target_user = db.execute(
        "SELECT user_id FROM users WHERE github_login = ?",
        (github_login,),
    ).fetchone()

    if not target_user:
        return jsonify({
            "error": (
                f"User '{github_login}' not found in Legate Studio. "
                "They must sign up at legate.studio first."
            )
        }), 404

    target_user_id = target_user["user_id"]

    # Check existing membership
    existing = db.execute(
        "SELECT status FROM shared_library_members WHERE shared_library_id = ? AND user_id = ?",
        (library_id, target_user_id),
    ).fetchone()

    if existing:
        status = existing["status"]
        if status == "active":
            return jsonify({"error": f"'{github_login}' is already an active member of this library"}), 409
        if status == "invited":
            return jsonify({"error": f"'{github_login}' already has a pending invitation"}), 409
        # Revoked — allow re-invitation
        db.execute(
            """
            UPDATE shared_library_members
            SET status = 'invited', invited_at = CURRENT_TIMESTAMP, accepted_at = NULL
            WHERE shared_library_id = ? AND user_id = ?
            """,
            (library_id, target_user_id),
        )
    else:
        db.execute(
            """
            INSERT INTO shared_library_members (shared_library_id, user_id, role, status)
            VALUES (?, ?, 'collaborator', 'invited')
            """,
            (library_id, target_user_id),
        )

    db.commit()

    # Add as GitHub collaborator (best-effort)
    github_warning = None
    try:
        token = get_user_installation_token(user_id, "library")
        if token and repo_full_name:
            gh_resp = requests.put(
                f"https://api.github.com/repos/{repo_full_name}/collaborators/{github_login}",
                json={"permission": "push"},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                },
                timeout=15,
            )
            if not gh_resp.ok and gh_resp.status_code != 201:
                github_warning = f"GitHub collaborator invite returned {gh_resp.status_code}"
                logger.warning(f"invite GitHub error: {github_warning}")
    except Exception as e:
        github_warning = str(e)
        logger.warning(f"invite GitHub API error: {e}")

    result = {
        "success": True,
        "library_id": library_id,
        "library_name": library_name,
        "github_login": github_login,
        "status": "invited",
    }
    if github_warning:
        result["github_warning"] = (
            f"Invitation recorded but GitHub collaborator invite may have failed: {github_warning}"
        )

    return jsonify(result)


@shared_libraries_bp.route("/api/<library_id>/accept", methods=["POST"])
@login_required
def api_accept(library_id: str):
    """Accept a pending shared library invitation.

    Output: {success: true, library_id, library_name}
    Only the invited user can accept.
    """
    user = _current_user()
    user_id = user.get("user_id")

    db = _get_db()

    row = db.execute(
        """
        SELECT slm.id, sl.name, sl.slug
        FROM shared_library_members slm
        JOIN shared_libraries sl ON sl.id = slm.shared_library_id
        WHERE slm.shared_library_id = ? AND slm.user_id = ? AND slm.status = 'invited'
        """,
        (library_id, user_id),
    ).fetchone()

    if not row:
        # Check if already active
        active = db.execute(
            "SELECT 1 FROM shared_library_members WHERE shared_library_id = ? AND user_id = ? AND status = 'active'",
            (library_id, user_id),
        ).fetchone()
        if active:
            return jsonify({"error": "You are already an active member of this library"}), 409
        return jsonify({"error": "No pending invitation found for this library"}), 404

    library_name = row["name"]
    library_slug = row["slug"]

    db.execute(
        """
        UPDATE shared_library_members
        SET status = 'active', accepted_at = CURRENT_TIMESTAMP
        WHERE shared_library_id = ? AND user_id = ?
        """,
        (library_id, user_id),
    )
    db.commit()

    logger.info(f"User {user_id} accepted invitation to library {library_id} ({library_name})")

    return jsonify({
        "success": True,
        "library_id": library_id,
        "library_name": library_name,
        "library_slug": library_slug,
    })


@shared_libraries_bp.route("/api/<library_id>/decline", methods=["POST"])
@login_required
def api_decline(library_id: str):
    """Decline a pending shared library invitation.

    Deletes the membership record entirely.
    Output: {success: true}
    Only the invited user can decline.
    """
    user = _current_user()
    user_id = user.get("user_id")

    db = _get_db()

    row = db.execute(
        "SELECT id FROM shared_library_members WHERE shared_library_id = ? AND user_id = ? AND status = 'invited'",
        (library_id, user_id),
    ).fetchone()

    if not row:
        return jsonify({"error": "No pending invitation found for this library"}), 404

    db.execute(
        "DELETE FROM shared_library_members WHERE shared_library_id = ? AND user_id = ? AND status = 'invited'",
        (library_id, user_id),
    )
    db.commit()

    logger.info(f"User {user_id} declined invitation to library {library_id}")

    return jsonify({"success": True})


@shared_libraries_bp.route("/api/<library_id>/members/<target_user_id>", methods=["DELETE"])
@login_required
def api_remove_member(library_id: str, target_user_id: str):
    """Remove a member from a shared library.

    Owner-only. Cannot remove self.
    Sets membership status to 'revoked'. Removes GitHub collaborator (best-effort).
    Deletes unsubmitted drafts for the removed user.
    """
    from .auth import get_user_installation_token
    from .rag.database import get_shared_library_db

    user = _current_user()
    user_id = user.get("user_id")

    # Owners cannot remove themselves
    if target_user_id == user_id:
        return jsonify({"error": "You cannot remove yourself. Transfer ownership or archive the library instead."}), 400

    db = _get_db()

    # Verify caller is owner
    lib_row = db.execute(
        """
        SELECT sl.repo_full_name, sl.name
        FROM shared_libraries sl
        JOIN shared_library_members slm ON sl.id = slm.shared_library_id
        WHERE sl.id = ? AND slm.user_id = ? AND slm.role = 'owner' AND slm.status = 'active'
        """,
        (library_id, user_id),
    ).fetchone()

    if not lib_row:
        return jsonify({"error": "Library not found or you are not the owner"}), 403

    repo_full_name = lib_row["repo_full_name"]
    library_name = lib_row["name"]

    # Verify target is a member
    member_row = db.execute(
        "SELECT status, role FROM shared_library_members WHERE shared_library_id = ? AND user_id = ?",
        (library_id, target_user_id),
    ).fetchone()

    if not member_row:
        return jsonify({"error": "User is not a member of this library"}), 404

    if member_row["status"] == "revoked":
        return jsonify({"error": "User has already been removed from this library"}), 409

    # Look up GitHub login for the target user (for GitHub API call)
    target_user_row = db.execute(
        "SELECT github_login FROM users WHERE user_id = ?",
        (target_user_id,),
    ).fetchone()
    github_login = target_user_row["github_login"] if target_user_row else None

    # Revoke membership
    db.execute(
        "UPDATE shared_library_members SET status = 'revoked' WHERE shared_library_id = ? AND user_id = ?",
        (library_id, target_user_id),
    )
    db.commit()

    # Delete pending drafts from the shared library DB (best-effort)
    drafts_deleted = 0
    try:
        lib_db = get_shared_library_db(library_id)
        cursor = lib_db.execute(
            "DELETE FROM drafts WHERE author_user_id = ? AND status = 'draft'",
            (target_user_id,),
        )
        drafts_deleted = cursor.rowcount
        lib_db.commit()
    except Exception as e:
        logger.warning(f"remove_member: failed to delete drafts for {target_user_id}: {e}")

    # Remove GitHub collaborator (best-effort)
    github_warning = None
    if github_login:
        try:
            token = get_user_installation_token(user_id, "library")
            if token and repo_full_name:
                gh_resp = requests.delete(
                    f"https://api.github.com/repos/{repo_full_name}/collaborators/{github_login}",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/vnd.github+json",
                    },
                    timeout=15,
                )
                if not gh_resp.ok and gh_resp.status_code != 204:
                    github_warning = f"GitHub collaborator removal returned {gh_resp.status_code}"
                    logger.warning(f"remove_member GitHub error: {github_warning}")
        except Exception as e:
            github_warning = str(e)
            logger.warning(f"remove_member GitHub API error: {e}")

    logger.info(
        f"Owner {user_id} removed member {target_user_id} from library {library_id} ({library_name}), "
        f"deleted {drafts_deleted} drafts"
    )

    result = {
        "success": True,
        "library_id": library_id,
        "library_name": library_name,
        "removed_user_id": target_user_id,
        "drafts_deleted": drafts_deleted,
    }
    if github_warning:
        result["github_warning"] = (
            f"Member removed from DB but GitHub collaborator removal may have failed: {github_warning}"
        )

    return jsonify(result)
