"""
Legato.Pit Core Application

Dashboard and Motif input for the LEGATO system.
"""
import os
import atexit
import logging
import secrets
import threading
import time
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path

from flask import (
    Flask, render_template, jsonify, request, redirect,
    url_for, session, flash
)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix

logger = logging.getLogger(__name__)

# Activity tracking for background sync
_last_activity_time = time.time()
_activity_lock = threading.Lock()


def touch_activity():
    """Update last activity timestamp (called on user interactions)."""
    global _last_activity_time
    with _activity_lock:
        _last_activity_time = time.time()


def get_last_activity() -> float:
    """Get the last activity timestamp."""
    with _activity_lock:
        return _last_activity_time


def _get_or_create_persistent_key(env_var: str, filename: str) -> str:
    """Get secret key from env var, or generate and persist one.

    On Fly.io, /data is a persistent volume that survives restarts.
    This prevents SECRET_KEY regeneration from invalidating all sessions
    and JWTs on every cold start.
    """
    key = os.getenv(env_var)
    if key:
        return key
    # Try to read from persistent storage (survives Fly.io restarts)
    key_path = Path(os.getenv('DATA_DIR', '/data')) / filename
    if key_path.exists():
        try:
            stored = key_path.read_text().strip()
            if stored:
                return stored
        except OSError:
            pass
    # First run — generate and persist
    key = secrets.token_hex(32)
    try:
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_text(key)
        logger.info(f"Generated and persisted {env_var} to {key_path}")
    except OSError:
        logger.warning(f"Could not persist {env_var} to {key_path} — using ephemeral key")
    return key


def create_app():
    """Application factory."""
    static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

    app = Flask(
        __name__,
        static_folder=static_folder,
        template_folder=template_folder,
        static_url_path='/static'
    )

    # Apply proxy fix for Fly.io (trust X-Forwarded-* headers)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

    # Security configuration
    is_production = os.getenv('FLASK_ENV') == 'production'
    app.config.update(
        SECRET_KEY=_get_or_create_persistent_key('FLASK_SECRET_KEY', '.flask_secret_key'),
        SESSION_COOKIE_SECURE=is_production,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE='Lax',
        PERMANENT_SESSION_LIFETIME=timedelta(days=7),
        PREFERRED_URL_SCHEME='https' if is_production else 'http',

        # GitHub OAuth (env vars use GH_ prefix to avoid GitHub's reserved GITHUB_ prefix)
        GITHUB_CLIENT_ID=os.getenv('GH_OAUTH_CLIENT_ID'),
        GITHUB_CLIENT_SECRET=os.getenv('GH_OAUTH_CLIENT_SECRET'),
        GITHUB_ALLOWED_USERS=os.getenv('GH_ALLOWED_USERS', '').split(','),

        # GitHub App (multi-tenant auth - only used when LEGATO_MODE=multi-tenant)
        GITHUB_APP_ID=os.getenv('GITHUB_APP_ID'),
        GITHUB_APP_CLIENT_ID=os.getenv('GITHUB_APP_CLIENT_ID'),
        GITHUB_APP_CLIENT_SECRET=os.getenv('GITHUB_APP_CLIENT_SECRET'),
        GITHUB_APP_SLUG=os.getenv('GITHUB_APP_SLUG', 'legato-studio'),

        # Deployment mode: single-tenant (DIY) or multi-tenant (SaaS)
        LEGATO_MODE=os.getenv('LEGATO_MODE', 'single-tenant'),

        # LEGATO configuration (single-tenant mode)
        LEGATO_ORG=os.getenv('LEGATO_ORG', 'bobbyhiddn'),
        CONDUCT_REPO=os.getenv('CONDUCT_REPO', 'Legato.Conduct'),
        SYSTEM_PAT=os.getenv('SYSTEM_PAT'),  # Only needed for single-tenant

        # App metadata
        APP_NAME='Legato.Pit',
        APP_DESCRIPTION='Dashboard & Motif for LEGATO'
    )

    # Rate limiting
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=["200 per day", "50 per hour"],
        storage_uri="memory://",
    )

    # MCP endpoints are exempted because they use OAuth authentication
    # instead of IP-based limits (the OAuth token identifies the user)
    @limiter.request_filter
    def exempt_mcp_endpoints():
        """Return True to exempt request from rate limiting."""
        # Exempt MCP endpoints - they use OAuth for auth
        return request.path.startswith('/mcp')

    # Store limiter on app for use in blueprints
    app.limiter = limiter

    # Register blueprints
    from .auth import auth_bp
    from .dashboard import dashboard_bp
    from .dropbox import dropbox_bp
    from .library import library_bp
    from .chat import chat_bp
    from .memory_api import memory_api_bp
    from .agents import agents_bp
    from .chords import chords_bp
    from .categories import categories_bp
    from .oauth_server import oauth_bp
    from .mcp_server import mcp_bp
    from .motif_api import motif_api_bp
    from .admin import admin_bp
    from .stripe_billing import billing_bp
    from .import_api import import_api_bp
    from .assets import assets_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(dropbox_bp)
    app.register_blueprint(library_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(memory_api_bp)
    app.register_blueprint(agents_bp)
    app.register_blueprint(chords_bp)
    app.register_blueprint(categories_bp)
    app.register_blueprint(oauth_bp)  # OAuth 2.1 AS with DCR for MCP
    app.register_blueprint(mcp_bp)    # MCP protocol handler
    app.register_blueprint(motif_api_bp)  # Motif processing with job queue
    app.register_blueprint(admin_bp)  # Admin console
    app.register_blueprint(billing_bp)  # Stripe billing
    app.register_blueprint(import_api_bp)  # Markdown ZIP import
    app.register_blueprint(assets_bp)  # Library asset management

    # Initialize all databases on startup
    with app.app_context():
        from .rag.database import init_db, init_agents_db, init_chat_db
        init_db()        # legato.db - knowledge entries, embeddings
        init_agents_db() # agents.db - agent queue
        init_chat_db()   # chat.db - chat sessions/messages
        logger.info("All databases initialized (legato.db, agents.db, chat.db)")

    # Initialize Stripe products after DB is ready (no-op if STRIPE_SECRET_KEY not set)
    from .stripe_billing import init_stripe_products_on_startup
    init_stripe_products_on_startup(app)

    # Initialize chat session manager (in-memory caching with periodic flush)
    from .rag.chat_session_manager import init_chat_manager, shutdown_chat_manager
    chat_manager = init_chat_manager(app)

    # Register shutdown handler to flush all chat sessions and checkpoint databases
    def cleanup_on_shutdown():
        """Flush chat sessions and checkpoint all databases on shutdown."""
        try:
            with app.app_context():
                # Flush chat sessions
                from .rag.database import init_chat_db, checkpoint_all_databases
                db = init_chat_db()
                shutdown_chat_manager(db)

                # Checkpoint all databases to ensure WAL changes are persisted
                checkpoint_all_databases()
                logger.info("Shutdown cleanup completed")
        except Exception as e:
            logger.error(f"Error during shutdown cleanup: {e}")

    atexit.register(cleanup_on_shutdown)

    # Background sync threads
    def start_background_sync():
        """Start background sync threads for library and agents."""
        LIBRARY_SYNC_INTERVAL = 60  # seconds - sync library every minute
        AGENT_SYNC_INTERVAL = 60  # seconds
        SYNC_DURATION = 15 * 60  # 15 minutes (inactivity timeout)

        def library_sync_task():
            """Periodic library sync - runs every minute while user is active.

            Syncs knowledge entries from Legato.Library GitHub repo to local DB.
            Continues as long as there's been user activity within the last 15 minutes.

            NOTE: Only runs in single-tenant mode. Multi-tenant uses per-user sync on login.
            """
            import random
            time.sleep(5 + random.uniform(0, 3))  # Stagger worker startup

            # Skip background sync in multi-tenant mode
            if app.config.get('LEGATO_MODE') == 'multi-tenant':
                logger.info("Multi-tenant mode: skipping background library sync (per-user sync on login)")
                return

            while True:
                # Check if there's been activity in the last 15 minutes
                seconds_since_activity = time.time() - get_last_activity()
                if seconds_since_activity > SYNC_DURATION:
                    break

                try:
                    with app.app_context():
                        from .rag.database import init_db
                        from .rag.library_sync import LibrarySync
                        from .rag.embedding_service import EmbeddingService
                        from .rag.openai_provider import OpenAIEmbeddingProvider

                        token = os.getenv('SYSTEM_PAT')
                        if not token:
                            logger.warning("SYSTEM_PAT not set, skipping library sync")
                            time.sleep(LIBRARY_SYNC_INTERVAL)
                            continue

                        db = init_db()

                        # Check if another worker synced recently (within last 30 seconds)
                        last_sync = db.execute(
                            "SELECT synced_at FROM sync_log ORDER BY synced_at DESC LIMIT 1"
                        ).fetchone()
                        if last_sync:
                            from datetime import datetime
                            last_sync_time = datetime.fromisoformat(last_sync['synced_at'].replace('Z', '+00:00'))
                            seconds_since_sync = (datetime.now(last_sync_time.tzinfo) - last_sync_time).total_seconds()
                            if seconds_since_sync < 30:
                                logger.debug(f"Skipping sync - another worker synced {seconds_since_sync:.0f}s ago")
                                time.sleep(LIBRARY_SYNC_INTERVAL)
                                continue

                        # Clean up duplicate entries (same file_path)
                        cleanup_count = 0
                        dups = db.execute("""
                            SELECT id FROM knowledge_entries
                            WHERE file_path IS NOT NULL AND id NOT IN (
                                SELECT MAX(id) FROM knowledge_entries WHERE file_path IS NOT NULL GROUP BY file_path
                            )
                        """).fetchall()
                        for row in dups:
                            db.execute("DELETE FROM embeddings WHERE entry_id = ? AND entry_type = 'knowledge'", (row['id'],))
                            db.execute("DELETE FROM knowledge_entries WHERE id = ?", (row['id'],))
                            cleanup_count += 1
                        if cleanup_count > 0:
                            db.commit()
                            logger.info(f"Cleaned up {cleanup_count} invalid/duplicate entries")

                        embedding_service = None
                        if os.getenv('OPENAI_API_KEY'):
                            try:
                                provider = OpenAIEmbeddingProvider()
                                embedding_service = EmbeddingService(provider, db)
                            except Exception as e:
                                logger.warning(f"Could not create embedding service: {e}")

                        sync = LibrarySync(db, embedding_service)
                        library_repo = get_user_library_repo()
                        stats = sync.sync_from_github(library_repo, token=token)

                        # Only log if there were actual changes
                        if stats.get('entries_created', 0) > 0 or stats.get('entries_updated', 0) > 0:
                            logger.info(f"Library sync: {stats}")

                except Exception as e:
                    logger.error(f"Library sync failed: {e}")

                time.sleep(LIBRARY_SYNC_INTERVAL)

            logger.info("Library sync loop ended (no activity for 15 min)")

        def agent_sync_task():
            """Periodic chord sync - runs every minute while user is active.

            Checks Library entries for needs_chord=true and chord_status=null,
            and queues them as agents for user approval.

            Continues as long as there's been user activity within the last 15 minutes.
            Any request (page load, button click, API call) resets the activity timer.

            NOTE: Only runs in single-tenant mode. Multi-tenant uses per-user sync via API.
            """
            time.sleep(10)  # Wait for app to initialize

            # Skip background sync in multi-tenant mode - no user context available
            if app.config.get('LEGATO_MODE') == 'multi-tenant':
                logger.info("Multi-tenant mode: skipping background agent sync (per-user sync via API)")
                return

            while True:
                # Check if there's been activity in the last 15 minutes
                seconds_since_activity = time.time() - get_last_activity()
                if seconds_since_activity > SYNC_DURATION:
                    break
                try:
                    with app.app_context():
                        from .rag.database import init_db, init_agents_db
                        from .agents import import_chords_from_library

                        legato_db = init_db()
                        agents_db = init_agents_db()

                        stats = import_chords_from_library(legato_db, agents_db)

                        if stats['queued'] > 0:
                            logger.info(f"Chord sync: queued {stats['queued']} from Library")

                except Exception as e:
                    logger.error(f"Chord sync failed: {e}")

                time.sleep(AGENT_SYNC_INTERVAL)

            logger.info("Chord sync loop ended (no activity for 15 min)")

        def oauth_cleanup_task():
            """Periodic cleanup of expired OAuth auth codes and sessions.

            Runs every 10 minutes regardless of user activity — OAuth
            cleanup is a housekeeping task that shouldn't depend on
            whether anyone is using the dashboard.
            """
            OAUTH_CLEANUP_INTERVAL = 600  # 10 minutes
            time.sleep(30)  # Wait for app to initialize

            while True:
                try:
                    with app.app_context():
                        from .oauth_server import cleanup_expired_oauth_sessions
                        cleanup_expired_oauth_sessions()
                except Exception as e:
                    logger.error(f"OAuth cleanup task failed: {e}")

                time.sleep(OAUTH_CLEANUP_INTERVAL)

        # Start all threads
        threading.Thread(target=library_sync_task, daemon=True).start()
        threading.Thread(target=agent_sync_task, daemon=True).start()
        threading.Thread(target=oauth_cleanup_task, daemon=True).start()
        logger.info("Started background sync threads (library, chord detection, OAuth cleanup)")

    start_background_sync()

    # Start the motif processing worker as a background thread
    # This ensures worker shares the same database as the web app
    # Only start if we're not already running as a separate worker process
    fly_process = os.environ.get('FLY_PROCESS_GROUP', '')
    if fly_process != 'worker':
        try:
            from .worker import MotifWorker
            worker = MotifWorker(app)
            worker.start()
            logger.info("Started motif processing worker thread")

            # Register cleanup on shutdown
            def stop_worker():
                try:
                    worker.stop()
                except Exception as e:
                    logger.error(f"Error stopping worker: {e}")

            atexit.register(stop_worker)
        except Exception as e:
            logger.error(f"Failed to start worker thread: {e}")

    # Track user activity to keep sync running
    @app.before_request
    def track_activity():
        """Update activity timestamp on user requests (not health checks)."""
        if request.endpoint != 'health':
            touch_activity()

    # Auto-refresh expired OAuth tokens
    @app.before_request
    def refresh_expired_tokens():
        """Check and refresh OAuth tokens if expired.

        Runs on each request for logged-in users. If token is expired,
        attempts to refresh using the refresh token.
        """
        # Skip for static files, health checks, and auth endpoints
        if request.endpoint in ('health', 'static', None):
            return
        if request.path.startswith('/auth/') or request.path.startswith('/admin/login'):
            return

        # Only check for logged-in users in multi-tenant mode
        if app.config.get('LEGATO_MODE') != 'multi-tenant':
            return
        if 'user' not in session:
            return

        user_id = session['user'].get('user_id')
        if not user_id:
            return

        try:
            from .auth import _get_user_oauth_token
            # This function handles expiration check and refresh internally
            token = _get_user_oauth_token(user_id)
            if not token:
                logger.warning(f"Could not get/refresh OAuth token for user {user_id}")
        except Exception as e:
            logger.error(f"Token refresh check failed: {e}")

    # Context processor for templates
    @app.context_processor
    def inject_globals():
        # Check if current user is admin (bootstrap auth OR GitHub user in list)
        is_admin = False
        if session.get('admin_authenticated'):
            is_admin = True
        elif 'user' in session:
            username = session['user'].get('login') or session['user'].get('username')
            admin_users = app.config.get('ADMIN_USERS', '').split(',') if app.config.get('ADMIN_USERS') else ['bobbyhiddn']
            admin_users = [u.strip() for u in admin_users if u.strip()]
            is_admin = username in admin_users

        # Check trial expiration status (but not for beta users or paid tiers)
        trial_expired = False
        if 'user' in session and app.config.get('LEGATO_MODE') == 'multi-tenant':
            user_id = session['user'].get('user_id')
            if user_id:
                effective_tier = get_effective_tier(user_id)
                # Only show trial expired for actual trial users, not beta/paid
                if effective_tier == 'trial':
                    trial_expired = is_trial_expired(user_id)

        return {
            'now': datetime.now(),
            'app_name': app.config['APP_NAME'],
            'app_description': app.config['APP_DESCRIPTION'],
            'user': session.get('user'),
            'is_authenticated': 'user' in session,
            'is_admin': is_admin,
            'trial_expired': trial_expired
        }

    # Root redirect
    @app.route('/')
    def index():
        if 'user' in session:
            return redirect(url_for('dashboard.index'))
        return redirect(url_for('auth.login'))

    # Health check (no auth required)
    @app.route('/health')
    def health():
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'app': app.config['APP_NAME']
        })

    # Error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template('error.html', title="Not Found", message="Page not found"), 404

    @app.errorhandler(500)
    def internal_error(error):
        return render_template('error.html', title="Server Error", message="An error occurred"), 500

    @app.errorhandler(429)
    def ratelimit_error(error):
        return render_template('error.html', title="Rate Limited", message="Too many requests. Please wait."), 429

    logger.info("Legato.Pit application initialized")
    return app


def login_required(f):
    """Decorator to require authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function


def get_user_tier(user_id: str) -> str:
    """Get a user's tier from the shared users table."""
    if not user_id:
        return 'free'

    from .rag.database import init_db

    db = init_db()
    row = db.execute(
        "SELECT tier FROM users WHERE user_id = ?",
        (user_id,),
    ).fetchone()

    if row and row['tier']:
        return row['tier']

    return 'free'


def get_current_user_tier() -> str:
    """Get the current session user's effective tier.

    In single-tenant mode, this returns a non-free value so SaaS gating is bypassed.
    Uses get_effective_tier() which properly handles beta users (they get 'managed' tier).
    """
    from flask import current_app

    if current_app.config.get('LEGATO_MODE') != 'multi-tenant':
        return 'single-tenant'

    user_id = session.get('user', {}).get('user_id')
    return get_effective_tier(user_id)


def is_paid_tier(tier: str) -> bool:
    """Check if tier is a paid tier (not free/trial)."""
    return bool(tier) and tier not in ('free', 'trial')


def get_trial_status(user_id: str) -> dict:
    """Get trial status for a user.

    Returns:
        dict with is_trial, days_remaining, is_expired, trial_started_at
    """
    from datetime import datetime, timedelta
    from .rag.database import init_db

    TRIAL_DAYS = 14

    if not user_id:
        return {'is_trial': False, 'days_remaining': 0, 'is_expired': True}

    db = init_db()
    row = db.execute(
        "SELECT tier, trial_started_at, is_beta FROM users WHERE user_id = ?",
        (user_id,)
    ).fetchone()

    if not row:
        return {'is_trial': False, 'days_remaining': 0, 'is_expired': True}

    # Beta users are never in trial (check both is_beta flag and legacy tier='beta')
    if row['is_beta'] or row['tier'] == 'beta':
        return {'is_trial': False, 'days_remaining': 0, 'is_expired': False, 'is_beta': True}

    # Paid users are not in trial
    if row['tier'] in ('byok', 'managed'):
        return {'is_trial': False, 'days_remaining': 0, 'is_expired': False}

    # Check trial status
    trial_started = row['trial_started_at']
    if not trial_started:
        # Legacy user without trial_started_at - treat as expired
        return {'is_trial': True, 'days_remaining': 0, 'is_expired': True}

    try:
        start_date = datetime.fromisoformat(trial_started.replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        start_date = datetime.now()

    # Make start_date naive if it's timezone-aware
    if start_date.tzinfo is not None:
        start_date = start_date.replace(tzinfo=None)

    expiry_date = start_date + timedelta(days=TRIAL_DAYS)
    now = datetime.now()
    days_remaining = (expiry_date - now).days

    return {
        'is_trial': True,
        'days_remaining': max(0, days_remaining),
        'is_expired': now > expiry_date,
        'trial_started_at': trial_started
    }


def is_trial_expired(user_id: str) -> bool:
    """Check if user's trial has expired."""
    status = get_trial_status(user_id)
    return status.get('is_expired', False) and status.get('is_trial', False)


def get_effective_tier(user_id: str) -> str:
    """Get effective tier considering beta status.

    Beta users get 'managed' tier free.
    Returns: 'trial', 'byok', or 'managed'
    """
    from .rag.database import init_db

    if not user_id:
        return 'trial'

    db = init_db()
    row = db.execute(
        "SELECT tier, is_beta FROM users WHERE user_id = ?",
        (user_id,)
    ).fetchone()

    if not row:
        return 'trial'

    # Beta users get managed tier free
    # Check both is_beta flag AND legacy tier='beta' for backwards compatibility
    if row['is_beta'] or row['tier'] == 'beta':
        return 'managed'

    # Return actual tier
    tier = row['tier'] or 'trial'
    return tier if tier in ('trial', 'byok', 'managed') else 'trial'


def can_use_platform_keys(user_id: str) -> bool:
    """Check if user can use platform API keys (managed tier or beta)."""
    return get_effective_tier(user_id) == 'managed'


def get_api_key_for_user(user_id: str, provider: str) -> str:
    """Get API key for user - platform key or their own BYOK key.

    Args:
        user_id: User's ID
        provider: 'anthropic' or 'openai'

    Returns:
        API key string or None if not available
    """
    import os
    from .auth import get_user_api_key

    tier = get_effective_tier(user_id)

    if tier == 'managed':
        # Return platform key from environment
        env_key = f'{provider.upper()}_API_KEY'
        return os.environ.get(env_key)
    else:
        # Return user's BYOK key
        return get_user_api_key(user_id, provider)


def paid_required(f):
    """Decorator to require a paid subscription tier in multi-tenant mode."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('auth.login'))

        from flask import current_app

        # Only enforce SaaS tiers in multi-tenant mode
        if current_app.config.get('LEGATO_MODE') != 'multi-tenant':
            return f(*args, **kwargs)

        tier = get_current_user_tier()
        if is_paid_tier(tier):
            return f(*args, **kwargs)

        if request.path.startswith('/api/') or request.is_json or request.headers.get('Accept') == 'application/json':
            return jsonify({
                'error': 'Subscription required',
                'upgrade_required': True,
            }), 402

        flash('This feature requires a subscription.', 'warning')
        return redirect(url_for('auth.setup'))

    return decorated_function


def library_required(f):
    """Decorator to require Library setup before accessing features.

    Must be used after @login_required. Checks if user has configured
    their Library repo. If not, attempts to auto-repair from installation,
    then redirects to setup page if repair fails.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('auth.login'))

        # Check if Library is configured (multi-tenant mode only)
        from flask import current_app
        if current_app.config.get('LEGATO_MODE') == 'multi-tenant':
            user = session['user']
            user_id = user.get('user_id')

            if user_id:
                from .rag.database import init_db
                db = init_db()  # Shared DB for user_repos
                library = db.execute(
                    "SELECT 1 FROM user_repos WHERE user_id = ? AND repo_type = 'library'",
                    (user_id,)
                ).fetchone()

                if not library:
                    # Attempt auto-repair before redirecting to setup
                    from .auth import _repair_user_repos_from_installation
                    if _repair_user_repos_from_installation(user_id, db):
                        # Repair succeeded, continue to page
                        pass
                    else:
                        flash('Please set up your Library first.', 'info')
                        return redirect(url_for('auth.setup'))

        return f(*args, **kwargs)
    return decorated_function


def copilot_required(f):
    """Decorator to require Copilot access for Chords/Agents features.

    Must be used after @login_required. Checks if user has Copilot enabled.
    If not, returns 403 for APIs or redirects to dashboard for pages.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('auth.login'))

        user = session['user']
        has_copilot = user.get('has_copilot', False)

        if not has_copilot:
            # Check if this is an API request
            from flask import request
            if request.path.startswith('/api/') or request.is_json or request.headers.get('Accept') == 'application/json':
                return jsonify({
                    'error': 'Chords and Agents features require GitHub Copilot. Enable Copilot on your account to use these features.',
                    'copilot_required': True
                }), 403
            else:
                flash('Chords and Agents features require GitHub Copilot. Enable Copilot on your account to use these features.', 'warning')
                return redirect(url_for('dashboard.index'))

        return f(*args, **kwargs)
    return decorated_function


def get_user_library_repo(user_id: str = None) -> str:
    """Get the user's configured Library repository.

    Looks up the Library repo from user_repos table. Falls back to
    default patterns if not configured.

    Args:
        user_id: User ID to look up. If None, uses current session user.

    Returns:
        Full repo name (e.g., 'username/Legato.Library.username')
    """
    from flask import current_app

    if user_id is None:
        user = session.get('user', {})
        user_id = user.get('user_id')
        username = user.get('username')
    else:
        username = None

    # In multi-tenant mode, look up from database
    if current_app.config.get('LEGATO_MODE') == 'multi-tenant' and user_id:
        from .rag.database import init_db
        db = init_db()  # Shared DB for user_repos
        row = db.execute(
            "SELECT repo_full_name FROM user_repos WHERE user_id = ? AND repo_type = 'library'",
            (user_id,)
        ).fetchone()

        if row:
            return row['repo_full_name']

    # Fallback: use default pattern
    if username:
        return f"{username}/Legato.Library.{username}"

    # Last resort: env var or hardcoded default
    import os
    return os.environ.get('LIBRARY_REPO', 'bobbyhiddn/Legato.Library')
