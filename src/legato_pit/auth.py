"""
GitHub OAuth Authentication for Legato.Pit

Supports two authentication modes:
1. Legacy OAuth App (single-tenant, allowlist-based)
2. GitHub App (multi-tenant, installation-based)

Security measures:
- State parameter to prevent CSRF attacks
- User allowlist enforcement (legacy mode)
- Installation-scoped tokens (GitHub App mode)
- Per-user encryption for stored tokens
- Session fixation protection
"""
import os
import secrets
import logging
from datetime import datetime
from urllib.parse import urlencode
from typing import Optional

import requests
from flask import (
    Blueprint, redirect, url_for, session, request,
    current_app, flash, render_template, g, jsonify
)

logger = logging.getLogger(__name__)

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')


class StaleInstallationError(Exception):
    """Raised when a GitHub App installation is no longer valid.

    This indicates the user needs to re-authenticate via the GitHub App flow
    to establish a new installation.
    """
    def __init__(self, user_id: str, installation_id: int):
        self.user_id = user_id
        self.installation_id = installation_id
        super().__init__(f"Installation {installation_id} is no longer valid for user {user_id}")

# GitHub OAuth endpoints
GITHUB_AUTHORIZE_URL = 'https://github.com/login/oauth/authorize'
GITHUB_TOKEN_URL = 'https://github.com/login/oauth/access_token'
GITHUB_USER_URL = 'https://api.github.com/user'

# GitHub App endpoints
GITHUB_APP_AUTHORIZE_URL = 'https://github.com/login/oauth/authorize'
GITHUB_APP_INSTALL_URL = 'https://github.com/apps/{app_slug}/installations/new'


@auth_bp.route('/login')
def login():
    """Display login page - GitHub App authentication only."""
    if 'user' in session:
        return redirect(url_for('dashboard.index'))

    # Only GitHub App authentication is supported for security
    github_app_configured = bool(current_app.config.get('GITHUB_APP_CLIENT_ID'))

    if not github_app_configured:
        flash('GitHub App authentication not configured. Contact administrator.', 'error')

    return render_template('login.html',
                           github_app_configured=github_app_configured,
                           oauth_configured=False)  # Legacy OAuth disabled


@auth_bp.route('/github')
def github_login():
    """Legacy OAuth disabled - redirect to GitHub App login."""
    flash('Please use GitHub App authentication.', 'info')
    return redirect(url_for('auth.github_app_login'))


@auth_bp.route('/github/callback')
def github_callback():
    """Handle GitHub OAuth callback - only for MCP OAuth flow.

    Legacy web OAuth is disabled. This callback only handles MCP OAuth.
    """
    # Check if this is an MCP OAuth flow
    if 'mcp_github_state' in session:
        from .oauth_server import handle_mcp_github_callback
        return handle_mcp_github_callback()

    # Legacy OAuth disabled - redirect to GitHub App login
    logger.warning("Legacy OAuth callback hit - redirecting to GitHub App login")
    flash('Please use GitHub App authentication.', 'info')
    return redirect(url_for('auth.github_app_login'))


@auth_bp.route('/logout')
def logout():
    """Log out the current user."""
    username = session.get('user', {}).get('username', 'unknown')
    session.clear()
    logger.info(f"User logged out: {username}")
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))


# =============================================================================
# GitHub App Multi-Tenant Authentication
# =============================================================================

def _get_db():
    """Get shared database for auth tables.

    Auth tables (users, installations, repos, api_keys, audit_log) are shared
    across all users and must be accessible without a user session.
    This is different from get_user_legato_db() which returns user-scoped databases.
    """
    from .rag.database import init_db
    return init_db()


def _get_or_create_user(github_id: int, github_login: str, email: Optional[str] = None,
                        name: Optional[str] = None, avatar_url: Optional[str] = None) -> dict:
    """Get or create user with atomic operation to prevent duplicates.

    Uses INSERT OR IGNORE + SELECT pattern to prevent race conditions where
    two concurrent requests could create duplicate users with the same github_id.

    Args:
        github_id: GitHub user ID
        github_login: GitHub username
        email: User's email (optional)
        name: Display name (optional)
        avatar_url: Profile picture URL (optional)

    Returns:
        User dict with user_id and other fields
    """
    import uuid
    from datetime import datetime

    db = _get_db()
    new_user_id = str(uuid.uuid4())
    now = datetime.now().isoformat()

    # Try to insert - will be ignored if github_id already exists (UNIQUE constraint)
    # This is atomic and prevents race conditions
    # New users start with 'trial' tier and trial_started_at set
    db.execute("""
        INSERT OR IGNORE INTO users
        (user_id, github_id, github_login, email, name, avatar_url, tier, trial_started_at, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, 'trial', ?, ?, ?)
    """, (new_user_id, github_id, github_login, email, name, avatar_url, now, now, now))

    # Now fetch the actual user (either just inserted or pre-existing)
    row = db.execute(
        "SELECT * FROM users WHERE github_id = ?",
        (github_id,)
    ).fetchone()

    if row:
        # Update mutable fields (login/email/name/avatar may change on GitHub)
        db.execute("""
            UPDATE users SET
                github_login = ?,
                email = COALESCE(?, email),
                name = COALESCE(?, name),
                avatar_url = COALESCE(?, avatar_url),
                updated_at = ?
            WHERE github_id = ?
        """, (github_login, email, name, avatar_url, now, github_id))
        db.commit()

        # Return fresh data after update
        row = db.execute("SELECT * FROM users WHERE github_id = ?", (github_id,)).fetchone()
        user_dict = dict(row)

        # Log if this was a new user creation
        if row['created_at'] == now:
            logger.info(f"Created new user: {github_login} ({user_dict['user_id']})")

        return user_dict

    # This shouldn't happen given INSERT OR IGNORE, but handle gracefully
    db.commit()
    logger.warning(f"Unexpected: INSERT OR IGNORE succeeded but SELECT returned None for github_id={github_id}")
    return {
        'user_id': new_user_id,
        'github_id': github_id,
        'github_login': github_login,
        'email': email,
        'name': name,
        'avatar_url': avatar_url,
        'tier': 'trial',
        'trial_started_at': now,
        'has_copilot': False,
        'is_beta': False,
    }


def _store_installation(user_id: str, installation_id: int, installation_data: dict):
    """Store or update a GitHub App installation.

    Args:
        user_id: The user's ID
        installation_id: GitHub installation ID
        installation_data: Full installation data from GitHub
    """
    from .crypto import encrypt_for_user
    from flask import has_request_context

    db = _get_db()
    account = installation_data.get('account', {})

    # CRITICAL: Verify user_id and get canonical user_id by github_id
    # This prevents storing installation with stale/wrong user_id from session
    # Only do this if we have a request context with session
    if has_request_context():
        github_id = session.get('user', {}).get('github_id')
        if github_id:
            canonical_user = db.execute(
                "SELECT user_id FROM users WHERE github_id = ?", (github_id,)
            ).fetchone()
            if canonical_user:
                canonical_user_id = canonical_user['user_id']
                # Fix session if it's out of sync
                if user_id != canonical_user_id:
                    logger.warning(f"Fixing user_id mismatch: session={user_id}, canonical={canonical_user_id}")
                    user_id = canonical_user_id
                    session['user']['user_id'] = user_id
                    session.modified = True

    # Verify user exists (FK constraint requires it)
    user_exists = db.execute(
        "SELECT 1 FROM users WHERE user_id = ?", (user_id,)
    ).fetchone()

    if not user_exists:
        # User record missing - try to recreate from session
        user_session = session.get('user', {}) if has_request_context() else {}
        github_id = user_session.get('github_id')
        github_login = user_session.get('username')

        if github_id and github_login:
            # Check if a user with this github_id already exists (different user_id)
            existing_user = db.execute(
                "SELECT user_id FROM users WHERE github_id = ?",
                (github_id,)
            ).fetchone()

            if existing_user:
                # User exists with a different user_id - update session to use the correct one
                correct_user_id = existing_user['user_id']
                logger.warning(f"User {user_id} not found but github_id {github_id} exists as {correct_user_id}, fixing session")
                if has_request_context():
                    session['user']['user_id'] = correct_user_id
                    session.modified = True
                # Use the correct user_id for the rest of this function
                user_id = correct_user_id
            else:
                # Also try by github_login in case github_id index is stale
                existing_by_login = db.execute(
                    "SELECT user_id FROM users WHERE github_login = ?",
                    (github_login,)
                ).fetchone()

                if existing_by_login:
                    # Found by login — update session to use canonical user_id, don't recreate
                    correct_user_id = existing_by_login['user_id']
                    logger.warning(
                        f"User {user_id} not found but github_login '{github_login}' exists as "
                        f"{correct_user_id}, fixing session"
                    )
                    if has_request_context():
                        session['user']['user_id'] = correct_user_id
                        session.modified = True
                    user_id = correct_user_id
                else:
                    # Truly new user — create record with tier='trial' (matches normal signup flow)
                    logger.warning(f"User {user_id} missing from database, creating from session")
                    db.execute(
                        """
                        INSERT INTO users (user_id, github_id, github_login, tier, created_at, updated_at)
                        VALUES (?, ?, ?, 'trial', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        """,
                        (user_id, github_id, github_login)
                    )
                    db.commit()
        else:
            logger.error(f"Cannot store installation: user {user_id} not found and no session data to recreate")
            raise ValueError(f"User {user_id} not found in database")

    # Check if installation exists
    existing = db.execute(
        "SELECT id FROM github_app_installations WHERE installation_id = ?",
        (installation_id,)
    ).fetchone()

    if existing:
        db.execute(
            """
            UPDATE github_app_installations
            SET user_id = ?, account_login = ?, account_type = ?,
                permissions = ?, updated_at = CURRENT_TIMESTAMP
            WHERE installation_id = ?
            """,
            (
                user_id,
                account.get('login'),
                account.get('type'),
                str(installation_data.get('permissions', {})),
                installation_id
            )
        )
    else:
        db.execute(
            """
            INSERT INTO github_app_installations
            (installation_id, user_id, account_login, account_type, permissions, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (
                installation_id,
                user_id,
                account.get('login'),
                account.get('type'),
                str(installation_data.get('permissions', {}))
            )
        )

    db.commit()
    logger.info(f"Stored installation {installation_id} for user {user_id}")


def _auto_detect_library(user_id: str, installations) -> Optional[dict]:
    """Auto-detect and configure a Legato.Library repo.

    Uses multiple strategies:
    1. Check repos already in GitHub App installation
    2. Use OAuth token to search user's repos for Legato.Library pattern
    3. If found outside installation, auto-add it to the installation

    Args:
        user_id: The user's ID
        installations: List of user's GitHub App installations

    Returns:
        Dict with repo config if found and configured, None otherwise
    """
    from .github_app import get_installation_access_token
    import requests

    db = _get_db()

    # Strategy 1: Check repos already in installation
    for inst in installations:
        installation_id = inst['installation_id'] if isinstance(inst, dict) else inst[0]

        try:
            token_data = get_installation_access_token(installation_id)
            token = token_data.get('token')

            if not token:
                continue

            headers = {
                'Authorization': f'Bearer {token}',
                'Accept': 'application/vnd.github+json'
            }
            resp = requests.get(
                'https://api.github.com/installation/repositories',
                headers=headers
            )

            if resp.ok:
                repos = resp.json().get('repositories', [])
                for repo in repos:
                    repo_name = repo['name']
                    if repo_name == 'Legato.Library' or repo_name.startswith('Legato.Library.'):
                        repo_full_name = repo['full_name']

                        db.execute(
                            """
                            INSERT INTO user_repos (user_id, repo_type, repo_full_name, installation_id, created_at, updated_at)
                            VALUES (?, 'library', ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                            ON CONFLICT(user_id, repo_type) DO UPDATE SET
                                repo_full_name = excluded.repo_full_name,
                                installation_id = excluded.installation_id,
                                updated_at = CURRENT_TIMESTAMP
                            """,
                            (user_id, repo_full_name, installation_id)
                        )
                        db.commit()

                        logger.info(f"Auto-detected Library repo {repo_full_name} for user {user_id}")
                        return {
                            'repo_type': 'library',
                            'repo_full_name': repo_full_name,
                            'installation_id': installation_id
                        }

        except Exception as e:
            logger.warning(f"Failed to check installation {installation_id} for Library: {e}")

    # Strategy 2: Use OAuth token to search user's repos
    oauth_token = _get_user_oauth_token(user_id)
    if not oauth_token:
        logger.warning(f"No OAuth token for user {user_id}, cannot search for Library")
        return None

    # Get user's GitHub login
    user_row = db.execute(
        "SELECT github_login FROM users WHERE user_id = ?",
        (user_id,)
    ).fetchone()

    if not user_row:
        return None

    github_login = user_row['github_login']

    try:
        # Search for Legato.Library repos owned by user
        headers = {
            'Authorization': f'Bearer {oauth_token}',
            'Accept': 'application/vnd.github+json'
        }

        # Try specific repo name first
        for repo_name in [f'Legato.Library.{github_login}', 'Legato.Library']:
            resp = requests.get(
                f'https://api.github.com/repos/{github_login}/{repo_name}',
                headers=headers,
                timeout=10
            )

            if resp.ok:
                repo_data = resp.json()
                repo_full_name = repo_data['full_name']
                repo_id = repo_data['id']

                logger.info(f"Found Library repo via OAuth: {repo_full_name}")

                # Try to add to installation
                if installations:
                    installation_id = installations[0]['installation_id'] if isinstance(installations[0], dict) else installations[0][0]

                    try:
                        added = add_repo_to_installation(user_id, repo_id, repo_full_name)
                        if added:
                            logger.info(f"Auto-added Library {repo_full_name} to installation")
                    except Exception as e:
                        logger.warning(f"Could not auto-add Library to installation: {e}")
                        # Continue anyway - we found the repo

                    # Configure the Library
                    db.execute(
                        """
                        INSERT INTO user_repos (user_id, repo_type, repo_full_name, installation_id, created_at, updated_at)
                        VALUES (?, 'library', ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        ON CONFLICT(user_id, repo_type) DO UPDATE SET
                            repo_full_name = excluded.repo_full_name,
                            installation_id = excluded.installation_id,
                            updated_at = CURRENT_TIMESTAMP
                        """,
                        (user_id, repo_full_name, installation_id)
                    )
                    db.commit()

                    return {
                        'repo_type': 'library',
                        'repo_full_name': repo_full_name,
                        'installation_id': installation_id
                    }

    except Exception as e:
        logger.warning(f"Failed to search for Library via OAuth: {e}")

    return None


def _log_audit(user_id: str, action: str, resource_type: str,
               resource_id: Optional[str] = None, details: Optional[str] = None):
    """Log an audit event.

    Args:
        user_id: The user performing the action
        action: Action type (login, logout, install, etc.)
        resource_type: Type of resource affected
        resource_id: ID of the resource (optional)
        details: Additional details as JSON string (optional)
    """
    db = _get_db()
    db.execute(
        """
        INSERT INTO audit_log (user_id, action, resource_type, resource_id, details, ip_address, created_at)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (user_id, action, resource_type, resource_id, details, request.remote_addr)
    )
    db.commit()


@auth_bp.route('/app/login')
def github_app_login():
    """Initiate GitHub App OAuth flow.

    This uses the GitHub App's OAuth credentials (not the legacy OAuth App).
    Users are authenticated and can then install the app on their repos.
    """
    client_id = current_app.config.get('GITHUB_APP_CLIENT_ID')

    if not client_id:
        flash('GitHub App not configured. Using legacy authentication.', 'warning')
        return redirect(url_for('auth.github_login'))

    # Generate and store state for CSRF protection
    state = secrets.token_urlsafe(32)
    session['app_oauth_state'] = state

    # Store next URL for post-auth redirect (e.g., back to agents page)
    next_url = request.args.get('next')
    if next_url:
        # Basic validation - must be a relative path
        if next_url.startswith('/') and not next_url.startswith('//'):
            session['auth_next_url'] = next_url

    # Build authorization URL
    # Note: GitHub App OAuth does NOT use scopes - permissions are defined
    # in the App's settings. Including scope causes a 404 error.
    params = {
        'client_id': client_id,
        'redirect_uri': url_for('auth.github_app_callback', _external=True),
        'state': state
    }

    auth_url = f"{GITHUB_APP_AUTHORIZE_URL}?{urlencode(params)}"
    logger.info(f"Redirecting to GitHub App OAuth: {auth_url}")

    return redirect(auth_url)


@auth_bp.route('/app/callback')
def github_app_callback():
    """Handle GitHub App OAuth callback.

    After user authorizes, we:
    1. Exchange code for user access token
    2. Fetch user info
    3. Create/update user in database
    4. Check for existing installations
    5. Redirect to setup or dashboard

    Also handles installation callbacks (when user installs the app on repos).
    """
    from .github_app import exchange_code_for_user_token, get_user_info, get_user_emails

    # Check if this is an installation callback (not OAuth login)
    # GitHub sends installation_id and setup_action for app installations
    installation_id = request.args.get('installation_id')
    setup_action = request.args.get('setup_action')

    if installation_id and setup_action:
        # This is a post-installation callback, redirect to the installed handler
        logger.info(f"Received installation callback, redirecting to installed handler")
        return redirect(url_for('auth.github_app_installed',
                                installation_id=installation_id,
                                setup_action=setup_action))

    # Verify state to prevent CSRF (only for OAuth login flow)
    state = request.args.get('state')
    stored_state = session.pop('app_oauth_state', None)

    if not state or state != stored_state:
        logger.warning(f"App OAuth state mismatch")
        flash('Authentication failed: Invalid state. Please try again.', 'error')
        return redirect(url_for('auth.login'))

    # Check for errors
    error = request.args.get('error')
    if error:
        error_desc = request.args.get('error_description', 'Unknown error')
        logger.warning(f"GitHub App OAuth error: {error} - {error_desc}")
        flash(f'Authentication failed: {error_desc}', 'error')
        return redirect(url_for('auth.login'))

    # Get authorization code
    code = request.args.get('code')
    if not code:
        flash('Authentication failed: No authorization code received.', 'error')
        return redirect(url_for('auth.login'))

    try:
        # Exchange code for tokens
        token_data = exchange_code_for_user_token(code)
        access_token = token_data.get('access_token')
        refresh_token = token_data.get('refresh_token')

        logger.info(f"Token exchange result: access_token_len={len(access_token) if access_token else 0}, "
                    f"refresh_token_present={bool(refresh_token)}, "
                    f"access_token_prefix={access_token[:10] if access_token and len(access_token) > 10 else 'N/A'}...")

        if not access_token:
            raise ValueError("No access token in response")

        # Fetch user info
        user_info = get_user_info(access_token)
        github_id = user_info.get('id')
        github_login = user_info.get('login')
        name = user_info.get('name')
        avatar_url = user_info.get('avatar_url')

        # Fetch primary email
        email = None
        try:
            emails = get_user_emails(access_token)
            for e in emails:
                if e.get('primary') and e.get('verified'):
                    email = e.get('email')
                    break
        except Exception as e:
            logger.warning(f"Could not fetch user emails: {e}")

        # Create or update user in database
        user = _get_or_create_user(github_id, github_login, email, name, avatar_url)

        # Store refresh token (encrypted)
        if refresh_token:
            from .crypto import encrypt_for_user
            db = _get_db()
            encrypted_refresh = encrypt_for_user(user['user_id'], refresh_token)
            db.execute(
                "UPDATE users SET refresh_token_encrypted = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
                (encrypted_refresh, user['user_id'])
            )
            db.commit()

        # Store OAuth token (encrypted) for repo management
        from .crypto import encrypt_for_user
        db = _get_db()
        encrypted_oauth = encrypt_for_user(user['user_id'], access_token)
        # Token expires in 8 hours typically, but store it anyway
        db.execute(
            """UPDATE users SET oauth_token_encrypted = ?,
               oauth_token_expires_at = datetime('now', '+8 hours'),
               updated_at = CURRENT_TIMESTAMP WHERE user_id = ?""",
            (encrypted_oauth, user['user_id'])
        )
        db.commit()

        # Session fixation protection
        session.clear()

        # Store user info in session
        session['user'] = {
            'user_id': user['user_id'],
            'username': github_login,
            'name': name or github_login,
            'avatar_url': avatar_url,
            'github_id': github_id,
            'tier': user.get('tier', 'free'),
            'auth_mode': 'github_app',
            'has_copilot': bool(user.get('has_copilot', False)),
        }
        session['github_token'] = access_token
        session.permanent = True

        # Log the login
        _log_audit(user['user_id'], 'login', 'user', user['user_id'], '{"method": "github_app"}')

        logger.info(f"GitHub App user logged in: {github_login}, user_id={user['user_id']}, session_has_token={bool(session.get('github_token'))}")

        # Trigger user-specific Library sync in background
        trigger_user_library_sync(user['user_id'], github_login)

        # Check for stored next URL (e.g., returning to agents page after re-auth)
        next_url = session.pop('auth_next_url', None)

        # Check if user has any installations
        db = _get_db()
        installations = db.execute(
            "SELECT installation_id, account_login FROM github_app_installations WHERE user_id = ?",
            (user['user_id'],)
        ).fetchall()

        if installations and len(installations) > 0:
            # User has installations - verify user_repos is also set up
            library_exists = db.execute(
                "SELECT 1 FROM user_repos WHERE user_id = ? AND repo_type = 'library'",
                (user['user_id'],)
            ).fetchone()

            if not library_exists:
                # Repair: try to auto-detect and configure library from installation
                logger.warning(f"User {github_login} has installation but no library configured - attempting repair")
                _repair_user_repos(user['user_id'], installations[0]['installation_id'], access_token, db)

            # User has installations
            flash(f'Welcome back, {name or github_login}!', 'success')
            # Redirect to stored next URL or dashboard
            return redirect(next_url or url_for('dashboard.index'))
        else:
            # First time user - redirect to setup (ignore next URL for new users)
            flash(f'Welcome, {name or github_login}! Let\'s set up your Legato installation.', 'success')
            return redirect(url_for('auth.setup'))

    except Exception as e:
        logger.error(f"GitHub App OAuth failed: {e}")
        flash('Authentication failed. Please try again.', 'error')
        return redirect(url_for('auth.login'))


@auth_bp.route('/app/install')
def github_app_install():
    """Redirect user to install the GitHub App on their account/repos.

    This is called after login when user needs to grant repo access.
    """
    app_slug = current_app.config.get('GITHUB_APP_SLUG', 'legato-studio')

    # If we have specific repos suggested, we could add them as query params
    # For now, let user choose during installation
    install_url = f"https://github.com/apps/{app_slug}/installations/new"

    logger.info(f"Redirecting to GitHub App installation: {install_url}")
    return redirect(install_url)


@auth_bp.route('/app/installed')
def github_app_installed():
    """Handle post-installation callback from GitHub.

    GitHub redirects here after user installs the app.
    Query params include installation_id and setup_action.
    """
    from .github_app import get_installation_access_token

    installation_id = request.args.get('installation_id')
    setup_action = request.args.get('setup_action')

    if not installation_id:
        flash('Installation failed: No installation ID received.', 'error')
        return redirect(url_for('auth.setup'))

    # User must be logged in
    if 'user' not in session:
        # Store installation ID and redirect to login
        session['pending_installation_id'] = installation_id
        flash('Please log in to complete the installation.', 'info')
        return redirect(url_for('auth.github_app_login'))

    user = session['user']
    user_id = user.get('user_id')

    if not user_id:
        flash('Session error. Please log in again.', 'error')
        return redirect(url_for('auth.login'))

    try:
        installation_id = int(installation_id)

        # Fetch installation details
        from .github_app import get_app_installations
        installations = get_app_installations()

        installation_data = None
        for inst in installations:
            if inst.get('id') == installation_id:
                installation_data = inst
                break

        if not installation_data:
            flash('Could not verify installation. Please try again.', 'error')
            return redirect(url_for('auth.setup'))

        # Store installation in database
        _store_installation(user_id, installation_id, installation_data)

        # Log the installation
        account_login = installation_data.get('account', {}).get('login', 'unknown')
        _log_audit(user_id, 'install', 'installation', str(installation_id),
                   f'{{"account": "{account_login}"}}')

        # Verify we can get a token
        token_data = get_installation_access_token(installation_id)

        # Auto-detect Library repo from the newly installed repos
        db = _get_db()
        installations = db.execute(
            "SELECT installation_id FROM github_app_installations WHERE user_id = ?",
            (user_id,)
        ).fetchall()

        detected = _auto_detect_library(user_id, installations)
        if detected:
            logger.info(f"Auto-detected Library repo after installation: {detected.get('repo_full_name')}")
            flash(f'Successfully installed Legato on {account_login}! Library detected: {detected.get("repo_full_name")}', 'success')
        else:
            flash(f'Successfully installed Legato on {account_login}!', 'success')

        logger.info(f"Installation {installation_id} completed for user {user_id}")

        return redirect(url_for('auth.setup'))

    except Exception as e:
        logger.error(f"Failed to complete installation: {e}")
        flash('Failed to complete installation. Please try again.', 'error')
        return redirect(url_for('auth.setup'))


@auth_bp.route('/setup')
def setup():
    """Setup page for new users or users needing to configure repos.

    Shows:
    - Current installations and their repos
    - Option to install on more repos
    - API key configuration (for BYK tier)
    - Repo designation (Library, Conduct)
    """
    if 'user' not in session:
        return redirect(url_for('auth.login'))

    user = session['user']
    user_id = user.get('user_id')

    # For legacy auth users, redirect to dashboard
    if user.get('auth_mode') != 'github_app':
        return redirect(url_for('dashboard.index'))

    db = _get_db()

    # Get user's installations
    installations = db.execute(
        """
        SELECT installation_id, account_login, account_type, permissions, created_at
        FROM github_app_installations
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (user_id,)
    ).fetchall()

    # Get user's designated repos
    repos = db.execute(
        """
        SELECT repo_type, repo_full_name, installation_id
        FROM user_repos
        WHERE user_id = ?
        """,
        (user_id,)
    ).fetchall()

    # Get user's API keys (just hints, not actual keys)
    api_keys = db.execute(
        """
        SELECT provider, key_hint, created_at
        FROM user_api_keys
        WHERE user_id = ?
        """,
        (user_id,)
    ).fetchall()

    # Get full user record for tier info
    user_record = db.execute(
        "SELECT tier FROM users WHERE user_id = ?",
        (user_id,)
    ).fetchone()

    # Auto-detect Library repo if not configured but installations exist
    repos_list = [dict(r) for r in repos]
    has_library = any(r['repo_type'] == 'library' for r in repos_list)

    if not has_library and installations:
        detected_library = _auto_detect_library(user_id, installations)
        if detected_library:
            repos_list.append(detected_library)
            flash(f'Auto-detected your Library: {detected_library["repo_full_name"]}', 'success')

            # Trigger initial sync for the newly detected Library
            trigger_user_library_sync(user_id, user.get('username'))

    return render_template('setup.html',
                           user=user,
                           tier=user_record['tier'] if user_record else 'free',
                           installations=[dict(i) for i in installations],
                           repos=repos_list,
                           api_keys=[dict(k) for k in api_keys])


@auth_bp.route('/setup/debug')
def setup_debug():
    """Debug endpoint for Library detection issues."""
    if 'user' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    user = session['user']
    user_id = user.get('user_id')
    username = user.get('username')

    db = _get_db()
    debug_info = {
        'user_id': user_id,
        'username': username,
        'session_has_token': 'github_token' in session,
    }

    # Check OAuth token in database
    row = db.execute(
        """SELECT oauth_token_encrypted, oauth_token_expires_at, refresh_token_encrypted
           FROM users WHERE user_id = ?""",
        (user_id,)
    ).fetchone()

    debug_info['db_has_oauth_token'] = bool(row and row['oauth_token_encrypted'])
    debug_info['db_has_refresh_token'] = bool(row and row['refresh_token_encrypted'])
    debug_info['oauth_expires_at'] = row['oauth_token_expires_at'] if row else None

    # Try to get OAuth token
    oauth_token = _get_user_oauth_token(user_id)
    debug_info['oauth_token_retrieved'] = bool(oauth_token)

    # Check installations
    installations = db.execute(
        "SELECT installation_id, account_login FROM github_app_installations WHERE user_id = ?",
        (user_id,)
    ).fetchall()
    debug_info['installations'] = [dict(i) for i in installations]

    # Check user_repos
    user_repos = db.execute(
        "SELECT repo_type, repo_full_name, installation_id FROM user_repos WHERE user_id = ?",
        (user_id,)
    ).fetchall()
    debug_info['user_repos'] = [dict(r) for r in user_repos]

    # Check for any users with this username (in case of duplicate user_ids)
    all_users_with_username = db.execute(
        "SELECT user_id, github_login, created_at FROM users WHERE github_login = ?",
        (username,)
    ).fetchall()
    debug_info['users_with_same_username'] = [dict(u) for u in all_users_with_username]

    # Check if there are repos for other user_ids with same username
    for other_user in all_users_with_username:
        other_id = other_user['user_id']
        if other_id != user_id:
            other_repos = db.execute(
                "SELECT repo_type, repo_full_name FROM user_repos WHERE user_id = ?",
                (other_id,)
            ).fetchall()
            if other_repos:
                debug_info[f'repos_for_other_user_{other_id}'] = [dict(r) for r in other_repos]

            other_installations = db.execute(
                "SELECT installation_id, account_login FROM github_app_installations WHERE user_id = ?",
                (other_id,)
            ).fetchall()
            if other_installations:
                debug_info[f'installations_for_other_user_{other_id}'] = [dict(i) for i in other_installations]

    # Also show all installations in the system matching this username's account
    all_matching_installations = db.execute(
        "SELECT installation_id, user_id, account_login FROM github_app_installations WHERE account_login = ?",
        (username,)
    ).fetchall()
    debug_info['all_installations_for_account'] = [dict(i) for i in all_matching_installations]

    # Try to find Library repo via OAuth
    if oauth_token:
        import requests
        headers = {
            'Authorization': f'Bearer {oauth_token}',
            'Accept': 'application/vnd.github+json'
        }

        # Check what repos we can see
        for repo_name in [f'Legato.Library.{username}', 'Legato.Library']:
            try:
                resp = requests.get(
                    f'https://api.github.com/repos/{username}/{repo_name}',
                    headers=headers,
                    timeout=10
                )
                debug_info[f'repo_check_{repo_name}'] = {
                    'status': resp.status_code,
                    'found': resp.ok,
                    'repo_id': resp.json().get('id') if resp.ok else None,
                }
            except Exception as e:
                debug_info[f'repo_check_{repo_name}'] = {'error': str(e)}

        # Check OAuth scopes
        try:
            resp = requests.get(
                'https://api.github.com/user',
                headers=headers,
                timeout=10
            )
            debug_info['oauth_scopes'] = resp.headers.get('X-OAuth-Scopes', 'unknown')
        except Exception as e:
            debug_info['oauth_scopes_error'] = str(e)

    return jsonify(debug_info)


@auth_bp.route('/setup/library', methods=['POST'])
def setup_library():
    """Quick setup for Library repo.

    Takes just the repo name (e.g., 'Legato.Library.username') and configures it.
    Uses the user's first installation.
    """
    if 'user' not in session:
        return redirect(url_for('auth.login'))

    user = session['user']
    user_id = user.get('user_id')
    username = user.get('username')

    repo_name = request.form.get('repo_name', '').strip()

    if not repo_name:
        flash('Please enter your Library repo name.', 'error')
        return redirect(url_for('auth.setup'))

    # Build full repo name
    if '/' in repo_name:
        repo_full_name = repo_name  # Already has owner
    else:
        repo_full_name = f"{username}/{repo_name}"

    db = _get_db()

    # Get user's first installation
    inst = db.execute(
        "SELECT installation_id FROM github_app_installations WHERE user_id = ? LIMIT 1",
        (user_id,)
    ).fetchone()

    if not inst:
        flash('Please install the GitHub App first.', 'error')
        return redirect(url_for('auth.setup'))

    installation_id = inst['installation_id']

    try:
        # Configure the Library
        db.execute(
            """
            INSERT INTO user_repos (user_id, repo_type, repo_full_name, installation_id, created_at, updated_at)
            VALUES (?, 'library', ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id, repo_type) DO UPDATE SET
                repo_full_name = excluded.repo_full_name,
                installation_id = excluded.installation_id,
                updated_at = CURRENT_TIMESTAMP
            """,
            (user_id, repo_full_name, installation_id)
        )
        db.commit()

        flash(f'Library configured: {repo_full_name}', 'success')

        # Trigger initial sync
        trigger_user_library_sync(user_id, username)

        return redirect(url_for('auth.setup'))

    except Exception as e:
        logger.error(f"Failed to configure Library: {e}")
        flash('Failed to configure Library.', 'error')
        return redirect(url_for('auth.setup'))


@auth_bp.route('/setup/repo', methods=['POST'])
def setup_repo():
    """Designate a repository for Library or Conduct.

    POST params:
    - repo_type: 'library' or 'conduct'
    - repo_full_name: 'owner/repo'
    - installation_id: The installation that has access
    """
    if 'user' not in session:
        return redirect(url_for('auth.login'))

    user = session['user']
    user_id = user.get('user_id')

    repo_type = request.form.get('repo_type')
    repo_full_name = request.form.get('repo_full_name')
    installation_id = request.form.get('installation_id')

    if repo_type not in ('library', 'conduct'):
        flash('Invalid repository type.', 'error')
        return redirect(url_for('auth.setup'))

    if not repo_full_name or not installation_id:
        flash('Repository name and installation are required.', 'error')
        return redirect(url_for('auth.setup'))

    try:
        db = _get_db()

        # Verify installation belongs to user
        inst = db.execute(
            "SELECT installation_id FROM github_app_installations WHERE installation_id = ? AND user_id = ?",
            (installation_id, user_id)
        ).fetchone()

        if not inst:
            flash('Invalid installation.', 'error')
            return redirect(url_for('auth.setup'))

        # Upsert the repo designation
        db.execute(
            """
            INSERT INTO user_repos (user_id, repo_type, repo_full_name, installation_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id, repo_type) DO UPDATE SET
                repo_full_name = excluded.repo_full_name,
                installation_id = excluded.installation_id,
                updated_at = CURRENT_TIMESTAMP
            """,
            (user_id, repo_type, repo_full_name, installation_id)
        )
        db.commit()

        _log_audit(user_id, 'configure', 'repo', repo_full_name, f'{{"type": "{repo_type}"}}')

        flash(f'Set {repo_full_name} as your {repo_type.title()} repository.', 'success')

    except Exception as e:
        logger.error(f"Failed to set repo: {e}")
        flash('Failed to configure repository.', 'error')

    return redirect(url_for('auth.setup'))


@auth_bp.route('/setup/apikey', methods=['POST'])
def setup_api_key():
    """Store an API key for BYK (Bring Your Key) tier users.

    POST params:
    - provider: 'anthropic' or 'openai'
    - api_key: The actual key (will be encrypted)
    """
    if 'user' not in session:
        return redirect(url_for('auth.login'))

    user = session['user']
    user_id = user.get('user_id')

    provider = request.form.get('provider')
    api_key = request.form.get('api_key')

    if provider not in ('anthropic', 'openai'):
        flash('Invalid API provider.', 'error')
        return redirect(url_for('auth.setup'))

    if not api_key:
        flash('API key is required.', 'error')
        return redirect(url_for('auth.setup'))

    try:
        from .crypto import encrypt_api_key

        db = _get_db()
        encrypted_key, key_hint = encrypt_api_key(user_id, api_key)

        # Upsert the API key
        db.execute(
            """
            INSERT INTO user_api_keys (user_id, provider, key_encrypted, key_hint, created_at, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id, provider) DO UPDATE SET
                key_encrypted = excluded.key_encrypted,
                key_hint = excluded.key_hint,
                updated_at = CURRENT_TIMESTAMP
            """,
            (user_id, provider, encrypted_key, key_hint)
        )
        db.commit()

        _log_audit(user_id, 'configure', 'api_key', provider, f'{{"hint": "{key_hint}"}}')

        flash(f'Saved {provider.title()} API key (****{key_hint}).', 'success')

    except Exception as e:
        logger.error(f"Failed to store API key: {e}")
        flash('Failed to store API key.', 'error')

    return redirect(url_for('auth.setup'))


@auth_bp.route('/setup/sync-installations', methods=['POST'])
def setup_sync_installations():
    """Re-sync GitHub App installations from GitHub.

    Fetches all installations for the current user and updates the database.
    Useful when installation records are missing.
    """
    if 'user' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('auth.login'))

    user = session['user']
    user_id = user.get('user_id')
    username = user.get('username')

    try:
        from .github_app import get_app_installations

        # Get all installations from GitHub
        all_installations = get_app_installations()

        if not all_installations:
            flash('No GitHub App installations found. Please install the app first.', 'warning')
            return redirect(url_for('auth.setup'))

        db = _get_db()
        synced_count = 0

        for installation in all_installations:
            account_login = installation.get('account', {}).get('login', '')

            # Check if this installation belongs to the current user
            if account_login.lower() == username.lower():
                installation_id = installation.get('id')

                # Store/update the installation
                _store_installation(user_id, installation_id, installation)
                synced_count += 1
                logger.info(f"Synced installation {installation_id} for user {user_id}")

        if synced_count > 0:
            flash(f'Successfully synced {synced_count} installation(s).', 'success')

            # Try to auto-detect Library repo
            installations = db.execute(
                "SELECT installation_id FROM github_app_installations WHERE user_id = ?",
                (user_id,)
            ).fetchall()

            detected = _auto_detect_library(user_id, installations)
            if detected:
                flash(f'Library detected: {detected.get("repo_full_name")}', 'success')
                # Trigger sync
                trigger_user_library_sync(user_id, username)
        else:
            flash(f'No installations found for account {username}. Make sure you installed the app on your account.', 'warning')

    except Exception as e:
        logger.error(f"Failed to sync installations: {e}")
        flash(f'Failed to sync installations: {str(e)}', 'error')

    return redirect(url_for('auth.setup'))


@auth_bp.route('/setup/check-copilot', methods=['POST'])
def setup_check_copilot():
    """Check/refresh the user's Copilot status.

    Updates the database and session with the current Copilot availability.
    """
    # Inline login check (can't import from core.py due to circular imports)
    if 'user' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('auth.login'))

    user = session['user']
    user_id = user.get('user_id')

    try:
        # Check Copilot status (this queries GitHub)
        has_copilot = check_user_copilot_access(user_id)

        # Update database
        update_user_copilot_status(user_id, has_copilot)

        # Update session
        session['user']['has_copilot'] = has_copilot
        session.modified = True

        if has_copilot:
            flash('Copilot detected! Chords & Agents features are now enabled.', 'success')
        else:
            flash('Copilot not detected. Enable GitHub Copilot on your account to use Chords & Agents.', 'info')

    except Exception as e:
        logger.error(f"Failed to check Copilot: {e}")
        flash(f'Failed to check Copilot status: {str(e)}', 'error')

    return redirect(url_for('auth.setup'))


@auth_bp.route('/setup/enable-copilot', methods=['POST'])
def setup_enable_copilot():
    """Manually enable Copilot features for the user.

    For users who have Copilot but automatic detection doesn't work.
    """
    if 'user' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('auth.login'))

    user = session['user']
    user_id = user.get('user_id')

    try:
        update_user_copilot_status(user_id, True)
        session['user']['has_copilot'] = True
        session.modified = True
        flash('Chords & Agents features enabled.', 'success')
        logger.info(f"User {user_id} manually enabled Copilot features")

    except Exception as e:
        logger.error(f"Failed to enable Copilot: {e}")
        flash(f'Failed to enable: {str(e)}', 'error')

    return redirect(url_for('auth.setup'))


@auth_bp.route('/setup/disable-copilot', methods=['POST'])
def setup_disable_copilot():
    """Manually disable Copilot features for the user."""
    if 'user' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('auth.login'))

    user = session['user']
    user_id = user.get('user_id')

    try:
        update_user_copilot_status(user_id, False)
        session['user']['has_copilot'] = False
        session.modified = True
        flash('Chords & Agents features disabled.', 'info')
        logger.info(f"User {user_id} manually disabled Copilot features")

    except Exception as e:
        logger.error(f"Failed to disable Copilot: {e}")
        flash(f'Failed to disable: {str(e)}', 'error')

    return redirect(url_for('auth.setup'))


@auth_bp.route('/setup/create-library', methods=['POST'])
def setup_create_library():
    """Auto-create a Legato.Library repository for the user.

    Uses the user's first installation to create the Library repo.
    """
    if 'user' not in session:
        return redirect(url_for('auth.login'))

    user = session['user']
    user_id = user.get('user_id')
    github_login = user.get('username')

    try:
        from .chord_executor import ensure_library_exists

        db = _get_db()

        # Get user's first installation
        installation = db.execute(
            """
            SELECT installation_id, account_login
            FROM github_app_installations
            WHERE user_id = ?
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (user_id,)
        ).fetchone()

        if not installation:
            flash('Please install the GitHub App first.', 'error')
            return redirect(url_for('auth.setup'))

        # Get installation token
        token = get_user_installation_token(user_id, 'library')
        if not token:
            # Fall back to getting token directly
            from .github_app import get_installation_access_token
            token_data = get_installation_access_token(installation['installation_id'])
            token = token_data['token']

        # Use the installation's account (could be user or org)
        org = installation['account_login'] or github_login

        # Create Library repo
        result = ensure_library_exists(token, org)

        if result.get('success'):
            library_repo = f"{org}/Legato.Library"

            # Auto-configure as Library repo
            db.execute(
                """
                INSERT INTO user_repos (user_id, repo_type, repo_full_name, installation_id, created_at, updated_at)
                VALUES (?, 'library', ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id, repo_type) DO UPDATE SET
                    repo_full_name = excluded.repo_full_name,
                    installation_id = excluded.installation_id,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, library_repo, installation['installation_id'])
            )
            db.commit()

            _log_audit(user_id, 'create', 'library', library_repo,
                       f'{{"created": {str(result.get("created", False)).lower()}}}')

            if result.get('created'):
                flash(f'Created {library_repo} as your Library.', 'success')
            else:
                flash(f'Configured existing {library_repo} as your Library.', 'success')

            # Note: Chords & Agents can be enabled manually in Settings
            # (requires GitHub Copilot subscription to function)

        else:
            flash('Failed to create Library repository.', 'error')

    except Exception as e:
        logger.error(f"Failed to create Library: {e}")
        flash(f'Failed to create Library: {str(e)}', 'error')

    return redirect(url_for('auth.setup'))


def get_current_user() -> Optional[dict]:
    """Get the current authenticated user.

    Returns:
        User dict from session, or None if not authenticated
    """
    return session.get('user')


def trigger_user_library_sync(user_id: str, username: str) -> dict:
    """Trigger a Library sync for a specific user.

    This syncs the user's Legato.Library to their personal database.
    Called after login in multi-tenant mode.

    Args:
        user_id: The user's unique ID
        username: The user's GitHub login (for Library repo name)

    Returns:
        Dict with sync status
    """
    import threading
    import os

    def _sync_in_background():
        from flask import current_app
        from .rag.database import init_db
        from .rag.library_sync import LibrarySync
        from .rag.embedding_service import EmbeddingService
        from .rag.openai_provider import OpenAIEmbeddingProvider

        try:
            # Get token for user's Library - require user token in multi-tenant mode
            token = get_user_installation_token(user_id, 'library')
            if not token:
                logger.warning(f"No installation token available for user {username} Library sync")
                return

            # Initialize user's database
            db = init_db(user_id=user_id)

            # Look up user's configured Library repo
            shared_db = init_db()  # Shared db for user_repos table
            repo_row = shared_db.execute(
                "SELECT repo_full_name FROM user_repos WHERE user_id = ? AND repo_type = 'library'",
                (user_id,)
            ).fetchone()

            if repo_row:
                library_repo = repo_row['repo_full_name']
            else:
                # Fallback: try common patterns
                library_repo = f"{username}/Legato.Library.{username}"
                logger.info(f"No configured Library for {username}, trying {library_repo}")

            # Set up embedding service
            embedding_service = None
            if os.environ.get('OPENAI_API_KEY'):
                try:
                    provider = OpenAIEmbeddingProvider()
                    embedding_service = EmbeddingService(provider, db)
                except Exception as e:
                    logger.warning(f"Could not create embedding service: {e}")
            sync = LibrarySync(db, embedding_service)
            stats = sync.sync_from_github(library_repo, token=token)

            logger.info(f"User {username} Library sync complete: {stats}")

        except Exception as e:
            logger.error(f"User {username} Library sync failed: {e}")

    # Run sync in background thread
    thread = threading.Thread(target=_sync_in_background, daemon=True)
    thread.start()

    return {'status': 'started', 'user_id': user_id}


def get_user_installation_token(user_id: str, repo_type: str = 'library') -> Optional[str]:
    """Get an installation access token for a user's designated repo.

    This is the key function for multi-tenant API access. It:
    1. Finds the user's designated repo of the given type
    2. Gets the installation that has access to it
    3. Returns a fresh access token (cached for performance)

    Args:
        user_id: The user's ID
        repo_type: 'library' or 'conduct'

    Returns:
        An access token string, or None if not available
    """
    from .github_app import get_token_manager

    db = _get_db()

    # Find the installation for this repo type
    row = db.execute(
        """
        SELECT ur.installation_id
        FROM user_repos ur
        WHERE ur.user_id = ? AND ur.repo_type = ?
        """,
        (user_id, repo_type)
    ).fetchone()

    if not row:
        logger.warning(f"No {repo_type} repo configured for user {user_id}")
        return None

    installation_id = row['installation_id']

    try:
        token_manager = get_token_manager(db)
        return token_manager.get_token(installation_id)
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            # Installation not found - could be transient, don't clear immediately
            # Just log and return None so caller can handle gracefully
            logger.warning(f"Installation {installation_id} returned 404 - may be transient or actually removed")
            # Don't clear installation data on first 404 - could be GitHub hiccup
            # User can manually re-auth if it's truly gone
            return None
        logger.error(f"Failed to get installation token: {e}")
        return None
    except Exception as e:
        # Log the error but don't clear installation data
        logger.error(f"Failed to get installation token: {e}")
        return None


def _clear_stale_installation(user_id: str, installation_id: int, db):
    """Remove stale installation data so user can re-authenticate.

    Clears:
    - user_repos entries pointing to this installation
    - github_app_installations entry

    This allows the user to re-authenticate and get a fresh installation.
    """
    try:
        # Clear user_repos pointing to this installation
        db.execute(
            "DELETE FROM user_repos WHERE user_id = ? AND installation_id = ?",
            (user_id, installation_id)
        )

        # Clear the installation record itself
        db.execute(
            "DELETE FROM github_app_installations WHERE installation_id = ? AND user_id = ?",
            (installation_id, user_id)
        )

        db.commit()
        logger.info(f"Cleared stale installation {installation_id} for user {user_id}")
    except Exception as e:
        logger.error(f"Failed to clear stale installation: {e}")


def _repair_user_repos(user_id: str, installation_id: int, token: str, db):
    """Attempt to repair missing user_repos entries (called during login).

    Called during login when user has an installation but no library configured.
    Tries to auto-detect Library repo from the installation's accessible repos.
    """
    return _do_repair_user_repos(user_id, installation_id, db)


def _repair_user_repos_from_installation(user_id: str, db) -> bool:
    """Attempt to repair missing user_repos entries (called from decorator).

    Looks up user's installation and tries to auto-detect Library repo.
    Returns True if repair succeeded, False otherwise.
    """
    try:
        # Find user's installation
        installation = db.execute(
            "SELECT installation_id FROM github_app_installations WHERE user_id = ?",
            (user_id,)
        ).fetchone()

        if not installation:
            logger.debug(f"No installation found for user {user_id} during repair")
            return False

        return _do_repair_user_repos(user_id, installation['installation_id'], db)

    except Exception as e:
        logger.error(f"Failed to repair user_repos from installation: {e}")
        return False


def _do_repair_user_repos(user_id: str, installation_id: int, db) -> bool:
    """Core repair logic - auto-detect and configure Library repo.

    Returns True if repair succeeded, False otherwise.
    """
    try:
        # Get accessible repos for this installation
        from .github_app import get_installation_access_token
        inst_token = get_installation_access_token(installation_id)

        if not inst_token:
            logger.warning(f"Could not get installation token for repair")
            return False

        # List repos accessible to the installation
        resp = requests.get(
            'https://api.github.com/installation/repositories',
            headers={
                'Authorization': f'Bearer {inst_token["token"]}',
                'Accept': 'application/vnd.github+json',
            },
            timeout=15
        )

        if not resp.ok:
            logger.warning(f"Could not list installation repos: {resp.status_code}")
            return False

        repos = resp.json().get('repositories', [])

        # Look for Library repo
        for repo in repos:
            repo_name = repo.get('name', '')
            if repo_name == 'Legato.Library' or repo_name.startswith('Legato.Library.'):
                repo_full_name = repo['full_name']
                logger.info(f"Repair: Found Library repo {repo_full_name} for user {user_id}")

                db.execute(
                    """
                    INSERT INTO user_repos (user_id, repo_type, repo_full_name, installation_id, created_at, updated_at)
                    VALUES (?, 'library', ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id, repo_type) DO UPDATE SET
                        repo_full_name = excluded.repo_full_name,
                        installation_id = excluded.installation_id,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (user_id, repo_full_name, installation_id)
                )
                db.commit()
                logger.info(f"Repair: Configured {repo_full_name} as library for user {user_id}")
                return True

        logger.info(f"Repair: No Library repo found in installation for user {user_id}")
        return False

    except Exception as e:
        logger.error(f"Failed to repair user_repos: {e}")
        return False


def check_user_copilot_access(user_id: str, token: str = None, repo_name: str = None) -> bool:
    """Check if a user has GitHub Copilot coding agent enabled.

    Uses the GraphQL suggestedActors query to see if copilot-swe-agent
    is available as an assignee on a repo owned by the user.

    Args:
        user_id: The user's ID
        token: GitHub token (optional, will fetch if not provided)
        repo_name: Repo to check (optional, will use Library repo if not provided)

    Returns:
        True if Copilot is available, False otherwise
    """
    if not token:
        token = get_user_installation_token(user_id, 'library')
        if not token:
            logger.warning(f"No token available to check Copilot for user {user_id}")
            return False

    if not repo_name:
        # Get user's Library repo
        db = _get_db()
        row = db.execute(
            "SELECT repo_full_name FROM user_repos WHERE user_id = ? AND repo_type = 'library'",
            (user_id,)
        ).fetchone()
        if not row:
            logger.warning(f"No Library repo found to check Copilot for user {user_id}")
            return False
        repo_name = row['repo_full_name']

    owner, repo = repo_name.split('/')

    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/vnd.github+json',
    }

    # Query suggested actors to find Copilot
    query = """
    query($owner: String!, $repo: String!) {
        repository(owner: $owner, name: $repo) {
            suggestedActors(capabilities: [CAN_BE_ASSIGNED], first: 100) {
                nodes {
                    login
                }
            }
        }
    }
    """

    try:
        resp = requests.post(
            'https://api.github.com/graphql',
            headers=headers,
            json={
                'query': query,
                'variables': {'owner': owner, 'repo': repo}
            },
            timeout=30
        )

        if resp.status_code != 200:
            logger.warning(f"GraphQL query failed for Copilot check: {resp.status_code} - {resp.text}")
            return False

        data = resp.json()
        logger.info(f"Copilot check GraphQL response for {repo_name}: {data}")

        # Check for errors in the GraphQL response
        if 'errors' in data:
            logger.warning(f"GraphQL errors in Copilot check: {data['errors']}")
            # Fall back to REST API check
            return _check_copilot_via_rest(owner, repo, token)

        nodes = data.get('data', {}).get('repository', {}).get('suggestedActors', {}).get('nodes', [])
        logins = [node.get('login') for node in nodes]
        logger.info(f"Suggested actors for {repo_name}: {logins}")

        for node in nodes:
            if node.get('login') == 'copilot-swe-agent':
                logger.info(f"User {user_id} has Copilot enabled (found copilot-swe-agent)")
                return True

        # GraphQL didn't find it, try REST API as fallback
        logger.info(f"copilot-swe-agent not in suggestedActors, trying REST API fallback")
        return _check_copilot_via_rest(owner, repo, token)

    except Exception as e:
        logger.error(f"Failed to check Copilot for user {user_id}: {e}")
        return False


def _check_copilot_via_rest(owner: str, repo: str, token: str) -> bool:
    """Fallback: Check Copilot via REST API by looking at repo collaborators.

    Copilot coding agent appears as a collaborator if enabled.
    """
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
    }

    try:
        # Check collaborators for copilot-swe-agent
        resp = requests.get(
            f'https://api.github.com/repos/{owner}/{repo}/collaborators',
            headers=headers,
            timeout=30
        )

        if resp.status_code == 200:
            collaborators = resp.json()
            logins = [c.get('login') for c in collaborators]
            logger.info(f"REST API collaborators for {owner}/{repo}: {logins}")

            for collab in collaborators:
                if collab.get('login') == 'copilot-swe-agent':
                    logger.info(f"Found copilot-swe-agent as collaborator on {owner}/{repo}")
                    return True

        # Try checking assignees as well
        resp = requests.get(
            f'https://api.github.com/repos/{owner}/{repo}/assignees',
            headers=headers,
            timeout=30
        )

        if resp.status_code == 200:
            assignees = resp.json()
            logins = [a.get('login') for a in assignees]
            logger.info(f"REST API assignees for {owner}/{repo}: {logins}")

            for assignee in assignees:
                if assignee.get('login') == 'copilot-swe-agent':
                    logger.info(f"Found copilot-swe-agent as assignee on {owner}/{repo}")
                    return True

        logger.info(f"copilot-swe-agent not found via REST API for {owner}/{repo}")
        return False

    except Exception as e:
        logger.error(f"REST API Copilot check failed: {e}")
        return False


def update_user_copilot_status(user_id: str, has_copilot: bool = None) -> bool:
    """Update the user's Copilot status in the database.

    If has_copilot is not provided, will check via API.

    Args:
        user_id: The user's ID
        has_copilot: Optional explicit value, otherwise checks via API

    Returns:
        The user's Copilot status
    """
    db = _get_db()

    if has_copilot is None:
        has_copilot = check_user_copilot_access(user_id)

    db.execute(
        """UPDATE users SET has_copilot = ?, copilot_checked_at = CURRENT_TIMESTAMP
           WHERE user_id = ?""",
        (1 if has_copilot else 0, user_id)
    )
    db.commit()

    return has_copilot


def get_user_copilot_status(user_id: str, check_if_stale: bool = True) -> bool:
    """Get the user's Copilot status, optionally refreshing if stale.

    Args:
        user_id: The user's ID
        check_if_stale: If True, recheck if status is older than 24 hours

    Returns:
        True if user has Copilot enabled
    """
    from datetime import datetime, timedelta

    db = _get_db()
    row = db.execute(
        "SELECT has_copilot, copilot_checked_at FROM users WHERE user_id = ?",
        (user_id,)
    ).fetchone()

    if not row:
        return False

    has_copilot = bool(row['has_copilot'])
    checked_at = row['copilot_checked_at']

    # If never checked or stale (>24h), recheck
    if check_if_stale:
        should_recheck = False
        if not checked_at:
            should_recheck = True
        else:
            try:
                checked_time = datetime.fromisoformat(checked_at.replace('Z', '+00:00'))
                if datetime.now(checked_time.tzinfo) - checked_time > timedelta(hours=24):
                    should_recheck = True
            except (ValueError, AttributeError):
                should_recheck = True

        if should_recheck:
            has_copilot = update_user_copilot_status(user_id)

    return has_copilot


def _get_user_oauth_token(user_id: str) -> Optional[str]:
    """Get user's OAuth token, refreshing if needed.

    Tries in order:
    1. Session token
    2. Stored token (if not expired)
    3. Refresh the token using refresh_token

    Returns:
        OAuth token string or None
    """
    from datetime import datetime

    logger.info(f"_get_user_oauth_token called for user_id={user_id}")

    # Try session first
    oauth_token = session.get('github_token')
    if oauth_token:
        logger.info(f"Found OAuth token in session for user {user_id} (len={len(oauth_token)}, prefix={oauth_token[:10] if len(oauth_token) > 10 else 'N/A'}...)")

        # Validate session token is still working
        import requests
        try:
            test_resp = requests.get(
                "https://api.github.com/user",
                headers={"Authorization": f"Bearer {oauth_token}", "Accept": "application/vnd.github+json"},
                timeout=5
            )
            if test_resp.status_code == 200:
                return oauth_token
            else:
                logger.warning(f"Session token invalid (status {test_resp.status_code}), clearing from session")
                session.pop('github_token', None)
                # Fall through to database/refresh logic
        except Exception as e:
            logger.warning(f"Failed to validate session token: {e}")
            # Fall through to database/refresh logic

    logger.info(f"No valid session token, checking database for user {user_id}")

    # Try database
    from .crypto import decrypt_for_user, encrypt_for_user
    db = _get_db()
    row = db.execute(
        """SELECT oauth_token_encrypted, oauth_token_expires_at, refresh_token_encrypted
           FROM users WHERE user_id = ?""",
        (user_id,)
    ).fetchone()

    if not row:
        logger.warning(f"No user row found in database for user_id={user_id}")
        return None

    logger.info(f"Found user row: has_oauth={bool(row['oauth_token_encrypted'])}, expires_at={row['oauth_token_expires_at']}, has_refresh={bool(row['refresh_token_encrypted'])}")

    # Check if stored token is still valid
    # GitHub OAuth tokens have an 8-hour window. Refresh at 80% (6.4h)
    # to avoid using tokens right at expiry boundary.
    PRE_EXPIRY_BUFFER_SECONDS = 5760  # 1.6 hours (80% of 8h = 6.4h)
    if row['oauth_token_encrypted']:
        expires_at = row['oauth_token_expires_at']
        is_expired = False

        if expires_at:
            try:
                expiry = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                now = datetime.now(expiry.tzinfo) if expiry.tzinfo else datetime.now()
                from datetime import timedelta
                # Treat token as "expired" when within the pre-expiry buffer
                is_expired = expiry < (now + timedelta(seconds=PRE_EXPIRY_BUFFER_SECONDS))
                if is_expired and expiry > now:
                    logger.info(f"Token for user {user_id} within pre-expiry buffer ({PRE_EXPIRY_BUFFER_SECONDS}s), proactively refreshing")
            except (ValueError, TypeError):
                is_expired = False  # Can't determine, try anyway

        if not is_expired:
            token = decrypt_for_user(user_id, row['oauth_token_encrypted'])
            if token:
                logger.debug(f"Decrypted valid OAuth token for user {user_id} (len={len(token)})")
                return token
            else:
                logger.warning(f"Failed to decrypt OAuth token for user {user_id}")

    # Try to refresh using refresh_token
    if row['refresh_token_encrypted']:
        logger.info(f"Attempting token refresh for user {user_id}")
        refresh_token = decrypt_for_user(user_id, row['refresh_token_encrypted'])
        if refresh_token:
            new_token = _refresh_oauth_token(user_id, refresh_token)
            if new_token:
                return new_token
            else:
                logger.warning(f"Token refresh failed for user {user_id}")
        else:
            logger.warning(f"Could not decrypt refresh token for user {user_id}")
    else:
        logger.warning(f"No refresh token stored for user {user_id}")

    # Last resort: return possibly-expired token (might still work)
    if row['oauth_token_encrypted']:
        return decrypt_for_user(user_id, row['oauth_token_encrypted'])

    return None


def _refresh_oauth_token(user_id: str, refresh_token: str) -> Optional[str]:
    """Refresh an OAuth token using the refresh token.

    GitHub App OAuth supports refresh tokens when configured.

    Returns:
        New access token or None if refresh failed
    """
    from flask import current_app
    from .crypto import encrypt_for_user

    # Use GitHub App credentials (multi-tenant mode)
    client_id = current_app.config.get('GITHUB_APP_CLIENT_ID')
    client_secret = current_app.config.get('GITHUB_APP_CLIENT_SECRET')

    if not client_id or not client_secret:
        logger.warning(f"Cannot refresh token: GITHUB_APP_CLIENT_ID or GITHUB_APP_CLIENT_SECRET not configured")
        return None

    logger.info(f"Attempting to refresh OAuth token for user {user_id}")

    try:
        resp = requests.post(
            'https://github.com/login/oauth/access_token',
            data={
                'client_id': client_id,
                'client_secret': client_secret,
                'grant_type': 'refresh_token',
                'refresh_token': refresh_token,
            },
            headers={'Accept': 'application/json'},
            timeout=15,
        )

        if resp.ok:
            data = resp.json()
            new_token = data.get('access_token')
            new_refresh = data.get('refresh_token')

            if data.get('error'):
                logger.warning(f"GitHub token refresh error: {data.get('error')} - {data.get('error_description')}")
                return None

            if new_token:
                # Store the new tokens
                db = _get_db()
                encrypted_token = encrypt_for_user(user_id, new_token)
                db.execute(
                    """UPDATE users SET oauth_token_encrypted = ?,
                       oauth_token_expires_at = datetime('now', '+8 hours'),
                       updated_at = CURRENT_TIMESTAMP WHERE user_id = ?""",
                    (encrypted_token, user_id)
                )

                if new_refresh:
                    encrypted_refresh = encrypt_for_user(user_id, new_refresh)
                    db.execute(
                        "UPDATE users SET refresh_token_encrypted = ? WHERE user_id = ?",
                        (encrypted_refresh, user_id)
                    )

                db.commit()
                logger.info(f"Refreshed OAuth token for user {user_id}")
                return new_token

    except Exception as e:
        logger.warning(f"Failed to refresh OAuth token: {e}")

    return None


def add_repo_to_installation(user_id: str, repo_id: int, repo_full_name: str = None,
                              max_retries: int = 3) -> bool:
    """Add a repository to the user's GitHub App installation.

    NOTE: This endpoint (PUT /user/installations/{id}/repositories/{repo_id})
    only works with classic PATs with 'repo' scope. It does NOT work with:
    - GitHub App OAuth tokens
    - GitHub App installation tokens
    - Fine-grained PATs

    This function is kept for Library auto-add attempts but will likely fail.
    For Chords, we use OAuth tokens directly for all operations instead of
    trying to add repos to the installation.

    Args:
        user_id: The user's ID
        repo_id: The GitHub repository ID
        repo_full_name: Optional full name for logging (org/repo)
        max_retries: Maximum retry attempts (default 3)

    Returns:
        True if successful, False otherwise

    Raises:
        RuntimeError: If all retries fail (caller should handle)
    """
    import time

    db = _get_db()

    # Get user's installation ID
    row = db.execute(
        "SELECT installation_id FROM github_app_installations WHERE user_id = ? LIMIT 1",
        (user_id,)
    ).fetchone()

    if not row:
        logger.error(f"No installation found for user {user_id}")
        _queue_repo_addition(user_id, repo_id, repo_full_name)
        raise RuntimeError(f"No GitHub App installation for user {user_id}")

    installation_id = row['installation_id']
    last_error = None

    for attempt in range(max_retries):
        # Get fresh token (may refresh if needed)
        oauth_token = _get_user_oauth_token(user_id)

        if not oauth_token:
            logger.error(f"No OAuth token available for user {user_id}")
            _queue_repo_addition(user_id, repo_id, repo_full_name)
            raise RuntimeError(f"No OAuth token for user {user_id} - user must re-authenticate")

        try:
            resp = requests.put(
                f'https://api.github.com/user/installations/{installation_id}/repositories/{repo_id}',
                headers={
                    'Authorization': f'Bearer {oauth_token}',
                    'Accept': 'application/vnd.github+json',
                    'X-GitHub-Api-Version': '2022-11-28',
                },
                timeout=15,
            )

            if resp.status_code == 204:
                logger.info(f"Added repo {repo_full_name or repo_id} to installation {installation_id}")
                return True
            elif resp.status_code == 304:
                logger.info(f"Repo {repo_full_name or repo_id} already in installation")
                return True
            elif resp.status_code == 401:
                # Token invalid - clear it and retry
                logger.warning(f"OAuth token invalid for user {user_id}, clearing and retrying")
                db.execute(
                    "UPDATE users SET oauth_token_encrypted = NULL WHERE user_id = ?",
                    (user_id,)
                )
                db.commit()
                last_error = "OAuth token invalid"
            elif resp.status_code == 403:
                # Permission denied - may need different scope
                last_error = f"Permission denied: {resp.text}"
                logger.error(f"Permission denied adding repo to installation: {resp.text}")
                break  # Don't retry permission errors
            else:
                last_error = f"HTTP {resp.status_code}: {resp.text}"
                logger.warning(f"Attempt {attempt + 1} failed: {last_error}")

        except requests.RequestException as e:
            last_error = str(e)
            logger.warning(f"Attempt {attempt + 1} network error: {e}")

        # Exponential backoff before retry
        if attempt < max_retries - 1:
            wait_time = (2 ** attempt) * 0.5  # 0.5s, 1s, 2s
            time.sleep(wait_time)

    # All retries failed - queue for later
    _queue_repo_addition(user_id, repo_id, repo_full_name)
    raise RuntimeError(f"Failed to add repo after {max_retries} attempts: {last_error}")


def _queue_repo_addition(user_id: str, repo_id: int, repo_full_name: str = None):
    """Queue a failed repo addition for later retry.

    Stores in database so it can be retried when user re-authenticates.
    """
    try:
        db = _get_db()
        db.execute(
            """INSERT OR REPLACE INTO pending_repo_additions
               (user_id, repo_id, repo_full_name, created_at)
               VALUES (?, ?, ?, CURRENT_TIMESTAMP)""",
            (user_id, repo_id, repo_full_name)
        )
        db.commit()
        logger.info(f"Queued repo {repo_full_name or repo_id} for later addition to installation")
    except Exception as e:
        logger.warning(f"Could not queue repo addition: {e}")


def process_pending_repo_additions(user_id: str) -> int:
    """Process any pending repo additions for a user.

    Call this after user re-authenticates to catch up on any failed additions.

    Returns:
        Number of repos successfully added
    """
    db = _get_db()
    pending = db.execute(
        "SELECT repo_id, repo_full_name FROM pending_repo_additions WHERE user_id = ?",
        (user_id,)
    ).fetchall()

    if not pending:
        return 0

    added = 0
    for row in pending:
        try:
            if add_repo_to_installation(user_id, row['repo_id'], row['repo_full_name']):
                # Remove from queue
                db.execute(
                    "DELETE FROM pending_repo_additions WHERE user_id = ? AND repo_id = ?",
                    (user_id, row['repo_id'])
                )
                db.commit()
                added += 1
        except RuntimeError:
            pass  # Still failed, leave in queue

    if added:
        logger.info(f"Processed {added} pending repo additions for user {user_id}")

    return added


def get_user_api_key(user_id: str, provider: str) -> Optional[str]:
    """Get a user's stored API key (decrypted).

    For BYK tier users who provide their own API keys.

    Args:
        user_id: The user's ID
        provider: 'anthropic' or 'openai'

    Returns:
        The decrypted API key, or None if not stored
    """
    from .crypto import decrypt_api_key

    db = _get_db()

    row = db.execute(
        "SELECT key_encrypted FROM user_api_keys WHERE user_id = ? AND provider = ?",
        (user_id, provider)
    ).fetchone()

    if not row:
        return None

    return decrypt_api_key(user_id, row['key_encrypted'])


@auth_bp.route('/admin/reset-user/<username>', methods=['POST'])
def admin_reset_user(username: str):
    """Admin route to reset a user's account (clear their data).

    This is used when a user needs to start fresh.
    Only accessible by admin users (configured via LEGATO_ADMINS env var).

    Args:
        username: The GitHub username to reset
    """
    import os

    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    # Only allow admin users (configured via LEGATO_ADMINS env var, comma-separated)
    current_user = session['user'].get('username')
    admin_users = os.environ.get('LEGATO_ADMINS', '').split(',')
    admin_users = [u.strip() for u in admin_users if u.strip()]

    if not admin_users or current_user not in admin_users:
        return jsonify({'error': 'Admin access required'}), 403

    try:
        db = _get_db()

        # Find the user by GitHub login
        user_row = db.execute(
            "SELECT user_id FROM users WHERE github_login = ?",
            (username,)
        ).fetchone()

        if not user_row:
            return jsonify({'error': f'User {username} not found'}), 404

        user_id = user_row['user_id']

        # Delete user's personal database
        from .rag.database import delete_user_data
        db_result = delete_user_data(user_id)

        # Clear user's auth data (installations, repos, api_keys)
        db.execute("DELETE FROM user_repos WHERE user_id = ?", (user_id,))
        db.execute("DELETE FROM user_api_keys WHERE user_id = ?", (user_id,))
        db.execute("DELETE FROM github_app_installations WHERE user_id = ?", (user_id,))
        db.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        db.commit()

        _log_audit(session['user']['user_id'], 'admin_reset', 'user', user_id, f'{{"target": "{username}"}}')

        logger.info(f"Admin {current_user} reset user {username} (user_id: {user_id})")

        return jsonify({
            'success': True,
            'message': f'User {username} has been reset',
            'user_id': user_id,
            'database_deleted': db_result
        })

    except Exception as e:
        logger.error(f"Failed to reset user {username}: {e}")
        return jsonify({'error': str(e)}), 500
