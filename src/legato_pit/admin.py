"""
Admin Console for Legato.Pit

Provides admin-only views for:
- User management (tiers, status)
- System status
- Configuration

Authentication:
- Bootstrap: ADMIN_USERNAME + ADMIN_PASSWORD env vars for initial access
- After setup: GitHub users in ADMIN_USERS list can access
"""

import logging
import secrets
from functools import wraps

from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for, flash, current_app

logger = logging.getLogger(__name__)

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

# Admin users - can be overridden via ADMIN_USERS env var (comma-separated)
DEFAULT_ADMINS = ['bobbyhiddn']


def get_admin_users() -> list[str]:
    """Get list of admin usernames."""
    env_admins = current_app.config.get('ADMIN_USERS', '')
    if env_admins:
        return [u.strip() for u in env_admins.split(',') if u.strip()]
    return DEFAULT_ADMINS


def is_admin() -> bool:
    """Check if current session has admin access.

    Admin access is granted if:
    1. Session has admin_authenticated=True (from bootstrap login), OR
    2. Logged-in GitHub user is in the admin users list
    """
    # Check bootstrap admin auth
    if session.get('admin_authenticated'):
        return True

    # Check GitHub user in admin list
    if 'user' in session:
        username = session['user'].get('login') or session['user'].get('username')
        if username in get_admin_users():
            return True

    return False


def admin_required(f):
    """Decorator requiring admin access."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_admin():
            # Redirect to admin login
            return redirect(url_for('admin.login'))

        return f(*args, **kwargs)
    return decorated_function


@admin_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Admin login page - bootstrap access with username/password."""
    # Already authenticated?
    if is_admin():
        return redirect(url_for('admin.index'))

    error = None

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        # Get credentials from env
        admin_username = current_app.config.get('ADMIN_USERNAME')
        admin_password = current_app.config.get('ADMIN_PASSWORD')

        if not admin_username or not admin_password:
            error = 'Admin credentials not configured. Set ADMIN_USERNAME and ADMIN_PASSWORD env vars.'
            logger.warning("Admin login attempted but credentials not configured")
        elif username == admin_username and secrets.compare_digest(password, admin_password):
            session['admin_authenticated'] = True
            logger.info(f"Admin login successful for {username}")
            return redirect(url_for('admin.index'))
        else:
            error = 'Invalid credentials'
            logger.warning(f"Failed admin login attempt for username: {username}")

    return render_template('admin/login.html', error=error)


@admin_bp.route('/logout')
def logout():
    """Clear admin session."""
    session.pop('admin_authenticated', None)
    flash('Admin session ended.', 'info')
    return redirect(url_for('admin.login'))


@admin_bp.route('/')
@admin_required
def index():
    """Admin dashboard - overview and user list."""
    from .rag.database import init_db

    db = init_db()

    # Get all users with their details
    users = db.execute("""
        SELECT
            u.user_id,
            u.github_id,
            u.github_login,
            u.email,
            u.tier,
            u.is_beta,
            u.has_copilot,
            u.created_at,
            u.updated_at,
            (SELECT COUNT(*) FROM github_app_installations WHERE user_id = u.user_id) as installation_count,
            (SELECT COUNT(*) FROM user_repos WHERE user_id = u.user_id) as repo_count
        FROM users u
        ORDER BY u.updated_at DESC NULLS LAST
    """).fetchall()

    users = [dict(u) for u in users]

    # Get system stats
    stats = {
        'total_users': len(users),
        'users_with_copilot': sum(1 for u in users if u.get('has_copilot')),
        'paid_users': sum(1 for u in users if u.get('tier') and u['tier'] != 'free'),
    }

    # Get agent queue stats
    try:
        from .rag.database import init_agents_db
        agents_db = init_agents_db()
        agent_stats = agents_db.execute("""
            SELECT status, COUNT(*) as count
            FROM agent_queue
            GROUP BY status
        """).fetchall()
        stats['agents'] = {row['status']: row['count'] for row in agent_stats}
    except Exception as e:
        logger.warning(f"Could not get agent stats: {e}")
        stats['agents'] = {}

    return render_template('admin/index.html', users=users, stats=stats)


@admin_bp.route('/user/<user_id>')
@admin_required
def user_detail(user_id: str):
    """View detailed info for a specific user."""
    from .rag.database import init_db

    try:
        db = init_db()

        user = db.execute("""
            SELECT * FROM users WHERE user_id = ?
        """, (user_id,)).fetchone()

        if not user:
            flash('User not found.', 'error')
            return redirect(url_for('admin.index'))

        user = dict(user)

        # Get user's installations
        try:
            installations = db.execute("""
                SELECT * FROM github_app_installations WHERE user_id = ?
            """, (user_id,)).fetchall()
            user['installations'] = [dict(i) for i in installations]
        except Exception as e:
            logger.error(f"Failed to get installations: {e}")
            user['installations'] = []

        # Get user's repos
        try:
            repos = db.execute("""
                SELECT * FROM user_repos WHERE user_id = ?
            """, (user_id,)).fetchall()
            user['repos'] = [dict(r) for r in repos]
        except Exception as e:
            logger.error(f"Failed to get repos: {e}")
            user['repos'] = []

        # Get user's agents
        try:
            from .rag.database import init_agents_db
            agents_db = init_agents_db()
            agents = agents_db.execute("""
                SELECT queue_id, project_name, status, created_at, approved_at
                FROM agent_queue
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 20
            """, (user_id,)).fetchall()
            user['agents'] = [dict(a) for a in agents]
        except Exception as e:
            logger.error(f"Failed to get agents: {e}")
            user['agents'] = []

        # Get token status
        try:
            token_row = db.execute("""
                SELECT oauth_token_encrypted, oauth_token_expires_at, refresh_token_encrypted
                FROM users WHERE user_id = ?
            """, (user_id,)).fetchone()

            from datetime import datetime
            user['token_status'] = {
                'has_oauth_token': bool(token_row and token_row['oauth_token_encrypted']),
                'has_refresh_token': bool(token_row and token_row['refresh_token_encrypted']),
                'oauth_expires_at': token_row['oauth_token_expires_at'] if token_row else None,
            }

            # Check if expired
            if token_row and token_row['oauth_token_expires_at']:
                try:
                    expiry = datetime.fromisoformat(token_row['oauth_token_expires_at'].replace('Z', '+00:00'))
                    user['token_status']['is_expired'] = expiry < datetime.now(expiry.tzinfo) if expiry.tzinfo else expiry < datetime.now()
                except:
                    user['token_status']['is_expired'] = None
            else:
                user['token_status']['is_expired'] = None

        except Exception as e:
            logger.error(f"Failed to get token status: {e}")
            user['token_status'] = {}

        return render_template('admin/user_detail.html', profile=user)

    except Exception as e:
        logger.error(f"User detail view failed: {e}")
        flash(f'Error loading user: {e}', 'error')
        return redirect(url_for('admin.index'))


@admin_bp.route('/api/user/<user_id>/tier', methods=['POST'])
@admin_required
def api_set_user_tier(user_id: str):
    """Set a user's subscription tier."""
    from .rag.database import init_db

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    tier = data.get('tier', '').strip()
    valid_tiers = ['trial', 'byok', 'managed']  # Beta is a separate flag (is_beta), not a tier

    if tier not in valid_tiers:
        return jsonify({'error': f'Invalid tier. Must be one of: {", ".join(valid_tiers)}'}), 400

    db = init_db()

    # Check user exists
    user = db.execute("SELECT github_login FROM users WHERE user_id = ?", (user_id,)).fetchone()
    if not user:
        return jsonify({'error': 'User not found'}), 404

    db.execute("UPDATE users SET tier = ? WHERE user_id = ?", (tier, user_id))
    db.commit()

    logger.info(f"Admin set tier for {user['github_login']} to {tier}")

    return jsonify({
        'status': 'success',
        'user_id': user_id,
        'tier': tier,
    })


@admin_bp.route('/api/user/<user_id>/copilot', methods=['POST'])
@admin_required
def api_set_user_copilot(user_id: str):
    """Toggle a user's Copilot access."""
    from .rag.database import init_db

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    has_copilot = bool(data.get('has_copilot', False))

    db = init_db()

    user = db.execute("SELECT github_login FROM users WHERE user_id = ?", (user_id,)).fetchone()
    if not user:
        return jsonify({'error': 'User not found'}), 404

    db.execute("UPDATE users SET has_copilot = ? WHERE user_id = ?", (has_copilot, user_id))
    db.commit()

    logger.info(f"Admin set has_copilot for {user['github_login']} to {has_copilot}")

    return jsonify({
        'status': 'success',
        'user_id': user_id,
        'has_copilot': has_copilot,
    })


@admin_bp.route('/api/user/<user_id>/beta', methods=['POST'])
@admin_required
def api_set_user_beta(user_id: str):
    """Toggle a user's beta status (beta users get managed tier free)."""
    from .rag.database import init_db

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    is_beta = bool(data.get('is_beta', False))

    db = init_db()

    user = db.execute("SELECT github_login FROM users WHERE user_id = ?", (user_id,)).fetchone()
    if not user:
        return jsonify({'error': 'User not found'}), 404

    db.execute("UPDATE users SET is_beta = ? WHERE user_id = ?", (is_beta, user_id))
    db.commit()

    logger.info(f"Admin set is_beta for {user['github_login']} to {is_beta}")

    return jsonify({
        'status': 'success',
        'user_id': user_id,
        'is_beta': is_beta,
    })


@admin_bp.route('/api/user/<user_id>/state', methods=['GET'])
@admin_required
def api_user_state(user_id: str):
    """State audit endpoint — returns full tier/beta/Stripe state for a single user."""
    from .rag.database import init_db
    from .core import get_effective_tier, get_trial_status

    db = init_db()

    user = db.execute("""
        SELECT user_id, github_login, tier, is_beta,
               stripe_customer_id, stripe_subscription_id,
               trial_started_at, created_at, updated_at
        FROM users WHERE user_id = ?
    """, (user_id,)).fetchone()

    if not user:
        return jsonify({'error': 'User not found'}), 404

    try:
        effective_tier = get_effective_tier(user_id)
    except Exception as e:
        logger.error(f"get_effective_tier failed for {user_id}: {e}")
        effective_tier = None

    try:
        trial_status = get_trial_status(user_id)
    except Exception as e:
        logger.error(f"get_trial_status failed for {user_id}: {e}")
        trial_status = None

    return jsonify({
        'user_id': user['user_id'],
        'github_login': user['github_login'],
        'tier': user['tier'],
        'effective_tier': effective_tier,
        'is_beta': bool(user['is_beta']),
        'stripe_customer_id': user['stripe_customer_id'],
        'stripe_subscription_id': user['stripe_subscription_id'],
        'trial_started_at': user['trial_started_at'],
        'trial_status': trial_status,
        'created_at': user['created_at'],
        'updated_at': user['updated_at'],
    })


@admin_bp.route('/api/users/state', methods=['GET'])
@admin_required
def api_all_users_state():
    """State audit endpoint — returns full tier/beta/Stripe state for ALL users."""
    from .rag.database import init_db
    from .core import get_effective_tier, get_trial_status

    db = init_db()

    users = db.execute("""
        SELECT user_id, github_login, tier, is_beta,
               stripe_customer_id, stripe_subscription_id,
               trial_started_at, created_at, updated_at
        FROM users ORDER BY created_at DESC
    """).fetchall()

    results = []
    for user in users:
        user_id = user['user_id']
        try:
            effective_tier = get_effective_tier(user_id)
        except Exception as e:
            logger.error(f"get_effective_tier failed for {user_id}: {e}")
            effective_tier = None

        try:
            trial_status = get_trial_status(user_id)
        except Exception as e:
            logger.error(f"get_trial_status failed for {user_id}: {e}")
            trial_status = None

        results.append({
            'user_id': user['user_id'],
            'github_login': user['github_login'],
            'tier': user['tier'],
            'effective_tier': effective_tier,
            'is_beta': bool(user['is_beta']),
            'stripe_customer_id': user['stripe_customer_id'],
            'stripe_subscription_id': user['stripe_subscription_id'],
            'trial_started_at': user['trial_started_at'],
            'trial_status': trial_status,
            'created_at': user['created_at'],
            'updated_at': user['updated_at'],
        })

    return jsonify({'users': results, 'count': len(results)})


@admin_bp.route('/api/user/<user_id>/refresh-token', methods=['POST'])
@admin_required
def api_refresh_user_token(user_id: str):
    """Force refresh a user's OAuth token using their refresh token."""
    from .rag.database import init_db
    from .auth import _refresh_oauth_token
    from .crypto import decrypt_for_user

    db = init_db()

    user = db.execute("""
        SELECT github_login, refresh_token_encrypted
        FROM users WHERE user_id = ?
    """, (user_id,)).fetchone()

    if not user:
        return jsonify({'error': 'User not found'}), 404

    if not user['refresh_token_encrypted']:
        return jsonify({'error': 'No refresh token stored - user must re-authenticate'}), 400

    # Decrypt and try to refresh
    refresh_token = decrypt_for_user(user_id, user['refresh_token_encrypted'])
    if not refresh_token:
        return jsonify({'error': 'Could not decrypt refresh token'}), 500

    new_token = _refresh_oauth_token(user_id, refresh_token)

    if new_token:
        logger.info(f"Admin force-refreshed token for {user['github_login']}")
        return jsonify({
            'status': 'success',
            'message': 'Token refreshed successfully',
        })
    else:
        return jsonify({
            'error': 'Refresh failed - check logs for details. User may need to re-authenticate.',
        }), 400


@admin_bp.route('/api/user/<user_id>/delete', methods=['POST'])
@admin_required
def api_delete_user(user_id: str):
    """Delete a user and all their data."""
    from .rag.database import init_db, get_user_db_path
    import os

    db = init_db()

    user = db.execute("SELECT github_login FROM users WHERE user_id = ?", (user_id,)).fetchone()
    if not user:
        return jsonify({'error': 'User not found'}), 404

    username = user['github_login']

    # Delete from shared tables
    db.execute("DELETE FROM user_repos WHERE user_id = ?", (user_id,))
    db.execute("DELETE FROM github_app_installations WHERE user_id = ?", (user_id,))
    db.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
    db.commit()

    # Delete user's database file
    user_db_path = get_user_db_path(user_id)
    if user_db_path and user_db_path.exists():
        try:
            os.remove(user_db_path)
            logger.info(f"Deleted user database: {user_db_path}")
        except Exception as e:
            logger.error(f"Failed to delete user database: {e}")

    # Delete from agents db
    try:
        from .rag.database import init_agents_db
        agents_db = init_agents_db()
        agents_db.execute("DELETE FROM agent_queue WHERE user_id = ?", (user_id,))
        agents_db.commit()
    except Exception as e:
        logger.warning(f"Could not delete user agents: {e}")

    logger.info(f"Admin deleted user: {username} ({user_id})")

    return jsonify({
        'status': 'success',
        'message': f'User {username} deleted',
    })


@admin_bp.route('/system')
@admin_required
def system_status():
    """System status and configuration."""
    import os
    from pathlib import Path

    status = {
        'mode': current_app.config.get('LEGATO_MODE', 'unknown'),
        'debug': current_app.debug,
        'env': os.environ.get('FLASK_ENV', 'unknown'),
    }

    # Check database files
    data_path = Path(os.environ.get('FLY_VOLUME_PATH', '/data'))
    if data_path.exists():
        db_files = list(data_path.glob('*.db'))
        status['databases'] = [
            {'name': f.name, 'size_mb': round(f.stat().st_size / 1024 / 1024, 2)}
            for f in db_files
        ]
    else:
        status['databases'] = []

    # Check config
    status['config'] = {
        'LEGATO_ORG': current_app.config.get('LEGATO_ORG', 'not set'),
        'GITHUB_APP_ID': bool(current_app.config.get('GITHUB_APP_ID')),
        'OPENAI_API_KEY': bool(current_app.config.get('OPENAI_API_KEY')),
        'ANTHROPIC_API_KEY': bool(current_app.config.get('ANTHROPIC_API_KEY')),
    }

    return render_template('admin/system.html', status=status)
