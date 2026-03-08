"""
Legato.Pit Core Application

Dashboard and Motif input for the LEGATO system.
"""

import atexit
import logging
import os
import secrets
import threading
import time
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path

from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix

logger = logging.getLogger(__name__)

# Activity tracking for background sync
_last_activity_time = time.time()
_activity_lock = threading.Lock()

# Module-level rate limiter — initialized via limiter.init_app(app) inside create_app().
# Defined at module level so blueprints (e.g. mcp_server.py) can import and
# decorate routes with @limiter.limit() at module load time before the app exists.
_rate_limit_storage = os.getenv("REDIS_URL", "memory://")
if _rate_limit_storage == "memory://":
    logger.warning(
        "Rate limiter using in-memory storage — state is NOT shared across workers and "
        "will be lost on restart. Set REDIS_URL for production deployments."
    )
else:
    logger.info("Rate limiter using Redis storage: %s", _rate_limit_storage)

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=_rate_limit_storage,
)


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

    In multi-tenant production (LEGATO_MODE=multi-tenant), an ephemeral key
    is a hard error — sessions and JWTs would be invalidated on every restart,
    breaking all users. Set the env var or ensure /data is a persistent volume.
    """
    key = os.getenv(env_var)
    if key:
        return key
    # Try to read from persistent storage (survives Fly.io restarts)
    key_path = Path(os.getenv("DATA_DIR", "/data")) / filename
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
        return key
    except OSError as exc:
        # Could not persist — using ephemeral key
        is_multi_tenant = os.getenv("LEGATO_MODE") == "multi-tenant"
        if is_multi_tenant:
            raise RuntimeError(
                f"FATAL: Could not persist {env_var} to {key_path} and "
                f"LEGATO_MODE=multi-tenant. In production, an ephemeral key would "
                f"invalidate all user sessions and JWTs on every restart. "
                f"Set {env_var} as an environment variable or mount a persistent "
                f"volume at /data. Refusing to start with ephemeral key."
            ) from exc
        logger.error(
            f"Could not persist {env_var} to {key_path} — using ephemeral key. "
            f"All sessions and JWTs will be invalidated on restart. "
            f"Set {env_var} as an environment variable for production."
        )
    return key


def create_app():
    """Application factory.

    Required environment variables:
      SENTRY_DSN         — DSN from sentry.io project settings (leave unset in dev to disable)
      SENTRY_ENVIRONMENT — 'production' or 'staging' (defaults to 'production')
    """
    # --- Sentry error tracking ---
    # Initialize BEFORE creating the Flask app so Flask integration wraps it from the start.
    # Safe to leave SENTRY_DSN unset in dev — Sentry won't initialize and nothing breaks.
    import sentry_sdk
    from sentry_sdk.integrations.flask import FlaskIntegration

    dsn = os.environ.get("SENTRY_DSN")
    if dsn:
        sentry_sdk.init(
            dsn=dsn,
            integrations=[FlaskIntegration()],
            traces_sample_rate=0.1,  # 10% performance sampling — stays within free tier (5K/mo)
            profiles_sample_rate=0.1,
            environment=os.environ.get("SENTRY_ENVIRONMENT", "production"),
            send_default_pii=False,  # No PII — we handle user voice recordings
        )
        logger.info(
            "Sentry initialized (environment=%s)",
            os.environ.get("SENTRY_ENVIRONMENT", "production"),
        )
    else:
        logger.debug("SENTRY_DSN not set — Sentry disabled")

    static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

    app = Flask(
        __name__,
        static_folder=static_folder,
        template_folder=template_folder,
        static_url_path="/static",
    )

    # Apply proxy fix for Fly.io (trust X-Forwarded-* headers)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

    # Security configuration
    is_production = os.getenv("FLASK_ENV") == "production"
    app.config.update(
        SECRET_KEY=_get_or_create_persistent_key("FLASK_SECRET_KEY", ".flask_secret_key"),
        SESSION_COOKIE_SECURE=is_production,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        PERMANENT_SESSION_LIFETIME=timedelta(days=7),
        PREFERRED_URL_SCHEME="https" if is_production else "http",
        # GitHub OAuth (env vars use GH_ prefix to avoid GitHub's reserved GITHUB_ prefix)
        GITHUB_CLIENT_ID=os.getenv("GH_OAUTH_CLIENT_ID"),
        GITHUB_CLIENT_SECRET=os.getenv("GH_OAUTH_CLIENT_SECRET"),
        GITHUB_ALLOWED_USERS=os.getenv("GH_ALLOWED_USERS", "").split(","),
        # GitHub App (multi-tenant auth - only used when LEGATO_MODE=multi-tenant)
        GITHUB_APP_ID=os.getenv("GITHUB_APP_ID"),
        GITHUB_APP_CLIENT_ID=os.getenv("GITHUB_APP_CLIENT_ID"),
        GITHUB_APP_CLIENT_SECRET=os.getenv("GITHUB_APP_CLIENT_SECRET"),
        GITHUB_APP_SLUG=os.getenv("GITHUB_APP_SLUG", "legato-studio"),
        # Deployment mode: single-tenant (DIY) or multi-tenant (SaaS)
        LEGATO_MODE=os.getenv("LEGATO_MODE", "single-tenant"),
        # LEGATO configuration (single-tenant mode)
        LEGATO_ORG=os.getenv("LEGATO_ORG", "bobbyhiddn"),
        CONDUCT_REPO=os.getenv("CONDUCT_REPO", "Legato.Conduct"),
        SYSTEM_PAT=os.getenv("SYSTEM_PAT"),  # Only needed for single-tenant
        # App metadata
        APP_NAME="Legate Studio",
        APP_DESCRIPTION="Dashboard & Motif for Legate Studio",
    )

    # Rate limiting — binds the module-level limiter to this app.
    # Default limits apply to all non-MCP routes.
    # MCP endpoints use per-user limits (see mcp_server.py get_mcp_user_id).
    limiter.init_app(app)

    # Register blueprints
    from .admin import admin_bp
    from .agents import agents_bp
    from .assets import assets_bp
    from .auth import auth_bp
    from .categories import categories_bp
    from .chat import chat_bp
    from .chords import chords_bp
    from .dashboard import dashboard_bp
    from .dropbox import dropbox_bp
    from .import_api import import_api_bp
    from .library import library_bp
    from .mcp_server import mcp_bp
    from .memory_api import memory_api_bp
    from .motif_api import motif_api_bp
    from .oauth_server import oauth_bp
    from .stripe_billing import billing_bp

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
    app.register_blueprint(mcp_bp)  # MCP protocol handler
    app.register_blueprint(motif_api_bp)  # Motif processing with job queue
    app.register_blueprint(admin_bp)  # Admin console
    app.register_blueprint(billing_bp)  # Stripe billing
    app.register_blueprint(import_api_bp)  # Markdown ZIP import
    app.register_blueprint(assets_bp)  # Library asset management

    # Debug-only route to verify Sentry is wired up correctly.
    # Only reachable when app.debug=True (never in production).
    from flask import abort as flask_abort

    @app.route("/debug-sentry")
    def trigger_error():
        if not app.debug:
            flask_abort(404)
        division_by_zero = 1 / 0  # noqa: F841 — intentional ZeroDivisionError for Sentry test

    # Initialize all databases on startup
    with app.app_context():
        from .rag.database import init_agents_db, init_chat_db, init_db

        init_db()  # legato.db - knowledge entries, embeddings
        init_agents_db()  # agents.db - agent queue
        init_chat_db()  # chat.db - chat sessions/messages
        logger.info("All databases initialized (legato.db, agents.db, chat.db)")

    # Startup security check: warn if master encryption key is loaded from DB
    # (triggers key load and caches it so the warning fires once at startup)
    from .crypto import is_master_key_from_env
    if not is_master_key_from_env():
        logger.warning(
            "⚠️  STARTUP SECURITY WARNING: Master encryption key is stored in the database. "
            "The key and the encrypted user data it protects are in the SAME file. "
            "A database dump would expose all encrypted API keys. "
            "Set LEGATE_MASTER_KEY environment variable in production. "
            "Generate a key with: python -m legate_studio.crypto"
        )

    # Initialize Stripe products after DB is ready (no-op if STRIPE_SECRET_KEY not set)
    from .stripe_billing import init_stripe_products_on_startup

    init_stripe_products_on_startup(app)

    # Initialize chat session manager (in-memory caching with periodic flush)
    from .rag.chat_session_manager import init_chat_manager, shutdown_chat_manager

    init_chat_manager(app)

    # Register shutdown handler to flush all chat sessions and checkpoint databases
    def cleanup_on_shutdown():
        """Flush chat sessions and checkpoint all databases on shutdown."""
        try:
            with app.app_context():
                # Flush chat sessions
                from .rag.database import checkpoint_all_databases, init_chat_db

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
        library_sync_interval = 60  # seconds - sync library every minute
        agent_sync_interval = 60  # seconds
        sync_duration = 15 * 60  # 15 minutes (inactivity timeout)

        def library_sync_task():
            """Periodic library sync - runs every minute while user is active.

            Syncs knowledge entries from Legate.Library GitHub repo to local DB.
            Continues as long as there's been user activity within the last 15 minutes.

            NOTE: Only runs in single-tenant mode. Multi-tenant uses per-user sync on login.
            """
            import random

            time.sleep(5 + random.uniform(0, 3))  # Stagger worker startup

            # Skip background sync in multi-tenant mode
            if app.config.get("LEGATO_MODE") == "multi-tenant":
                logger.info("Multi-tenant mode: skipping background library sync (per-user sync on login)")
                return

            while True:
                # Check if there's been activity in the last 15 minutes
                seconds_since_activity = time.time() - get_last_activity()
                if seconds_since_activity > sync_duration:
                    break

                try:
                    with app.app_context():
                        from .rag.database import init_db
                        from .rag.embedding_provider import get_embedding_provider
                        from .rag.embedding_service import EmbeddingService
                        from .rag.library_sync import LibrarySync

                        token = os.getenv("SYSTEM_PAT")
                        if not token:
                            logger.warning("SYSTEM_PAT not set, skipping library sync")
                            time.sleep(library_sync_interval)
                            continue

                        db = init_db()

                        # Check if another worker synced recently (within last 30 seconds)
                        last_sync = db.execute(
                            "SELECT synced_at FROM sync_log ORDER BY synced_at DESC LIMIT 1"
                        ).fetchone()
                        if last_sync:
                            from datetime import datetime

                            last_sync_time = datetime.fromisoformat(last_sync["synced_at"].replace("Z", "+00:00"))
                            seconds_since_sync = (datetime.now(last_sync_time.tzinfo) - last_sync_time).total_seconds()
                            if seconds_since_sync < 30:
                                logger.debug(f"Skipping sync - another worker synced {seconds_since_sync:.0f}s ago")
                                time.sleep(library_sync_interval)
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
                            db.execute(
                                ("DELETE FROM embeddings WHERE entry_id = ? AND entry_type ='knowledge'"),
                                (row["id"],),
                            )
                            db.execute("DELETE FROM knowledge_entries WHERE id = ?", (row["id"],))
                            cleanup_count += 1
                        if cleanup_count > 0:
                            db.commit()
                            logger.info(f"Cleaned up {cleanup_count} invalid/duplicate entries")

                        embedding_service = None
                        try:
                            provider = get_embedding_provider()
                            embedding_service = EmbeddingService(provider, db)
                        except Exception as e:
                            logger.warning(f"Could not create embedding service: {e}")

                        sync = LibrarySync(db, embedding_service)
                        library_repo = get_user_library_repo()
                        stats = sync.sync_from_github(library_repo, token=token)

                        # Only log if there were actual changes
                        if stats.get("entries_created", 0) > 0 or stats.get("entries_updated", 0) > 0:
                            logger.info(f"Library sync: {stats}")

                except Exception as e:
                    logger.error(f"Library sync failed: {e}")

                time.sleep(library_sync_interval)

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
            if app.config.get("LEGATO_MODE") == "multi-tenant":
                logger.info("Multi-tenant mode: skipping background agent sync (per-user sync via API)")
                return

            while True:
                # Check if there's been activity in the last 15 minutes
                seconds_since_activity = time.time() - get_last_activity()
                if seconds_since_activity > sync_duration:
                    break
                try:
                    with app.app_context():
                        from .agents import import_chords_from_library
                        from .rag.database import init_agents_db, init_db

                        legato_db = init_db()
                        agents_db = init_agents_db()

                        stats = import_chords_from_library(legato_db, agents_db)

                        if stats["queued"] > 0:
                            logger.info(f"Chord sync: queued {stats['queued']} from Library")

                except Exception as e:
                    logger.error(f"Chord sync failed: {e}")

                time.sleep(agent_sync_interval)

            logger.info("Chord sync loop ended (no activity for 15 min)")

        def oauth_cleanup_task():
            """Periodic cleanup of expired OAuth auth codes and sessions.

            Runs every 10 minutes regardless of user activity — OAuth
            cleanup is a housekeeping task that shouldn't depend on
            whether anyone is using the dashboard.
            """
            oauth_cleanup_interval = 600  # 10 minutes
            time.sleep(30)  # Wait for app to initialize

            while True:
                try:
                    with app.app_context():
                        from .oauth_server import cleanup_expired_oauth_sessions

                        cleanup_expired_oauth_sessions()
                except Exception as e:
                    logger.error(f"OAuth cleanup task failed: {e}")

                time.sleep(oauth_cleanup_interval)

        # Start all threads
        threading.Thread(target=library_sync_task, daemon=True).start()
        threading.Thread(target=agent_sync_task, daemon=True).start()
        threading.Thread(target=oauth_cleanup_task, daemon=True).start()
        logger.info("Started background sync threads (library, chord detection, OAuth cleanup)")

    start_background_sync()

    # Start the motif processing worker as a background thread
    # This ensures worker shares the same database as the web app
    # Only start if we're not already running as a separate worker process
    fly_process = os.environ.get("FLY_PROCESS_GROUP", "")
    if fly_process != "worker":
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
        if request.endpoint != "health":
            touch_activity()

    # Auto-refresh expired OAuth tokens
    @app.before_request
    def refresh_expired_tokens():
        """Check and refresh OAuth tokens if expired.

        Runs on each request for logged-in users. If token is expired,
        attempts to refresh using the refresh token.
        """
        # Skip for static files, health checks, and auth endpoints
        if request.endpoint in ("health", "static", None):
            return
        if request.path.startswith("/auth/") or request.path.startswith("/admin/login"):
            return

        # Only check for logged-in users in multi-tenant mode
        if app.config.get("LEGATO_MODE") != "multi-tenant":
            return
        if "user" not in session:
            return

        user_id = session["user"].get("user_id")
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

    # Enforce trial expiry — block API and HTML access for expired trial users
    @app.before_request
    def enforce_trial_expiry():
        """Block access for users whose free trial has expired.

        Only enforced in multi-tenant mode. Beta users (is_beta=1) are always
        exempt. Billing, auth, static, and OAuth endpoints are never blocked
        so users can subscribe or manage their account.

        JSON/API requests → 402 with JSON error.
        HTML requests → redirect to /auth/setup.
        """
        # Only enforce in multi-tenant SaaS mode
        if app.config.get("LEGATO_MODE") != "multi-tenant":
            return

        # Skip if not logged in
        if "user" not in session:
            return

        # Allow access to auth, billing, OAuth, and static paths
        exempt_prefixes = ("/auth/", "/billing/", "/static/", "/oauth/", "/health")
        if any(request.path.startswith(p) for p in exempt_prefixes):
            return

        user_id = session["user"].get("user_id")
        if not user_id:
            return

        # Beta users are always exempt
        if session["user"].get("is_beta"):
            return

        try:
            effective_tier = get_effective_tier(user_id)
            if effective_tier == "trial" and is_trial_expired(user_id):
                if (
                    request.path.startswith("/api/")
                    or request.is_json
                    or request.headers.get("Accept") == "application/json"
                ):
                    return jsonify({
                        "error": "Your free trial has expired. Please subscribe to continue.",
                        "upgrade_url": "/auth/setup",
                        "trial_expired": True,
                    }), 402
                else:
                    flash("Your free trial has expired. Please subscribe to continue.", "warning")
                    return redirect(url_for("auth.setup"))
        except Exception as e:
            logger.error(f"Trial expiry check failed for user {user_id}: {e}")

    # Context processor for templates
    @app.context_processor
    def inject_globals():
        # Check if current user is admin (bootstrap auth OR GitHub user in list)
        is_admin = False
        if session.get("admin_authenticated"):
            is_admin = True
        elif "user" in session:
            username = session["user"].get("login") or session["user"].get("username")
            admin_users = (
                app.config.get("ADMIN_USERS", "").split(",") if app.config.get("ADMIN_USERS") else ["bobbyhiddn"]
            )
            admin_users = [u.strip() for u in admin_users if u.strip()]
            is_admin = username in admin_users

        # Check trial expiration status (but not for beta users or paid tiers)
        trial_expired = False
        if "user" in session and app.config.get("LEGATO_MODE") == "multi-tenant":
            user_id = session["user"].get("user_id")
            if user_id:
                effective_tier = get_effective_tier(user_id)
                # Only show trial expired for actual trial users, not beta/paid
                if effective_tier == "trial":
                    trial_expired = is_trial_expired(user_id)

        return {
            "now": datetime.now(),
            "app_name": app.config["APP_NAME"],
            "app_description": app.config["APP_DESCRIPTION"],
            "user": session.get("user"),
            "is_authenticated": "user" in session,
            "is_admin": is_admin,
            "trial_expired": trial_expired,
        }

    # Root route — landing page for visitors, dashboard redirect for authenticated users
    @app.route("/")
    def index():
        if "user" in session:
            return redirect(url_for("dashboard.index"))
        return render_template("landing.html")

    # ============ Public marketing pages ============

    @app.route("/features")
    def features():
        return render_template("features.html")

    @app.route("/pricing")
    def pricing():
        return render_template("pricing.html")

    @app.route("/faq")
    def faq():
        return render_template("faq.html")

    @app.route("/about")
    def about():
        return render_template("about.html")

    @app.route("/security")
    def security():
        return render_template("security.html")

    @app.route("/contact")
    def contact():
        return render_template("contact.html")

    @app.route("/privacy")
    def privacy():
        return render_template("privacy.html")

    @app.route("/terms")
    def terms():
        return render_template("terms.html")

    # ============ Solution / category landing pages ============

    @app.route("/mcp-first-pkm")
    def mcp_first_pkm():
        return render_template("mcp_first_pkm.html")

    @app.route("/personal-knowledge-base-for-ai")
    def personal_knowledge_base_for_ai():
        return render_template("pkb_for_ai.html")

    @app.route("/memory-layer-for-ai")
    def memory_layer_for_ai():
        return render_template("memory_layer_for_ai.html")

    @app.route("/voice-notes-to-knowledge-base")
    def voice_notes_to_knowledge_base():
        return render_template("voice_notes_to_kb.html")

    @app.route("/knowledge-graph-notes")
    def knowledge_graph_notes():
        return render_template("knowledge_graph_notes.html")

    @app.route("/persistent-memory-for-ai-assistants")
    def persistent_memory_for_ai_assistants():
        return render_template("persistent_memory_for_ai.html")

    # Custom 404 error page
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template("404.html"), 404

    # SEO: robots.txt
    @app.route("/robots.txt")
    def robots_txt():
        content = (
            "User-agent: *\n"
            "Allow: /\n"
            "Allow: /features\n"
            "Allow: /pricing\n"
            "Allow: /faq\n"
            "Allow: /about\n"
            "Allow: /security\n"
            "Allow: /contact\n"
            "Allow: /privacy\n"
            "Allow: /terms\n"
            "Allow: /mcp-first-pkm\n"
            "Allow: /personal-knowledge-base-for-ai\n"
            "Allow: /memory-layer-for-ai\n"
            "Allow: /voice-notes-to-knowledge-base\n"
            "Allow: /knowledge-graph-notes\n"
            "Allow: /persistent-memory-for-ai-assistants\n"
            "Disallow: /dashboard\n"
            "Disallow: /library\n"
            "Disallow: /admin\n"
            "Disallow: /api\n"
            "Disallow: /auth\n"
            "Disallow: /billing\n"
            "Disallow: /setup\n"
            "Disallow: /motif\n"
            "Disallow: /chat\n"
            "Disallow: /chord\n"
            "Disallow: /import\n"
            "Disallow: /dropbox\n"
            "Disallow: /agents\n"
            "\n"
            "Sitemap: https://legate.studio/sitemap.xml\n"
        )
        return content, 200, {"Content-Type": "text/plain; charset=utf-8"}

    # SEO: sitemap.xml
    @app.route("/sitemap.xml")
    def sitemap_xml():
        from .rag.database import get_user_db_path, init_db

        today = datetime.now().strftime("%Y-%m-%d")
        urls = [
            ("https://legate.studio/",         today, "weekly", "1.0"),
            ("https://legate.studio/features",  today, "monthly", "0.9"),
            ("https://legate.studio/pricing",   today, "monthly", "0.9"),
            ("https://legate.studio/faq",       today, "monthly", "0.8"),
            ("https://legate.studio/about",     today, "monthly", "0.6"),
            ("https://legate.studio/security",  today, "monthly", "0.5"),
            ("https://legate.studio/contact",   today, "monthly", "0.4"),
            ("https://legate.studio/privacy",   today, "yearly",  "0.3"),
            ("https://legate.studio/terms",     today, "yearly",  "0.3"),
            # Solution / category landing pages
            ("https://legate.studio/mcp-first-pkm",                      today, "monthly", "0.8"),
            ("https://legate.studio/personal-knowledge-base-for-ai",     today, "monthly", "0.8"),
            ("https://legate.studio/memory-layer-for-ai",                today, "monthly", "0.8"),
            ("https://legate.studio/voice-notes-to-knowledge-base",      today, "monthly", "0.8"),
            ("https://legate.studio/knowledge-graph-notes",              today, "monthly", "0.8"),
            ("https://legate.studio/persistent-memory-for-ai-assistants",today, "monthly", "0.8"),
        ]

        # Include published notes and profile pages from all user DBs
        try:
            import sqlite3
            shared_db = init_db()  # shared db has users table with github_login
            users = shared_db.execute("SELECT user_id, github_login FROM users").fetchall()
            for user_row in users:
                user_id = user_row["user_id"]
                github_login = user_row["github_login"]
                if not github_login:
                    continue
                try:
                    user_db_path = get_user_db_path(user_id)
                    if not user_db_path.exists():
                        continue
                    uconn = sqlite3.connect(str(user_db_path))
                    uconn.row_factory = sqlite3.Row
                    notes = uconn.execute(
                        "SELECT slug, updated_at FROM knowledge_entries WHERE published = 1 AND slug IS NOT NULL"
                    ).fetchall()
                    uconn.close()
                    if not notes:
                        continue
                    # Profile page — priority 0.7 (higher than individual notes)
                    profile_lastmod = max((n["updated_at"] or today)[:10] for n in notes)
                    urls.append((
                        f"https://legate.studio/pub/{github_login}",
                        profile_lastmod,
                        "weekly",
                        "0.7",
                    ))
                    # Individual note pages — priority 0.6
                    for note in notes:
                        slug = note["slug"]
                        lastmod = (note["updated_at"] or today)[:10]
                        loc = f"https://legate.studio/pub/{github_login}/{slug}"
                        urls.append((loc, lastmod, "weekly", "0.6"))
                except Exception as e:
                    logger.warning(f"Sitemap: failed to read published notes for {user_id}: {e}")
        except Exception as e:
            logger.warning(f"Sitemap: failed to enumerate user DBs: {e}")

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
        ]
        for loc, lastmod, changefreq, priority in urls:
            lines += [
                "  <url>",
                f"    <loc>{loc}</loc>",
                f"    <lastmod>{lastmod}</lastmod>",
                f"    <changefreq>{changefreq}</changefreq>",
                f"    <priority>{priority}</priority>",
                "  </url>",
            ]
        lines.append("</urlset>")
        xml = "\n".join(lines) + "\n"
        return xml, 200, {"Content-Type": "application/xml; charset=utf-8"}

    # SEO: RSS feeds
    def _build_rss_items(notes_with_author, limit=50):
        """Build RSS <item> elements from a list of (note_row, username) tuples."""
        import calendar
        import html as html_mod
        from email.utils import formatdate

        items = []
        for note, username in notes_with_author[:limit]:
            title = note.get("title") or "Untitled"
            slug = note.get("slug") or ""
            content = note.get("content") or ""
            pub_date_raw = note.get("published_at") or note.get("updated_at") or ""
            author_name = note.get("display_name") or note.get("author_name") or username

            link = f"https://legate.studio/pub/{username}/{slug}"
            # Truncate description to 300 chars, strip any markdown fences
            description = content.replace("\n", " ").strip()[:300]
            if len(content.strip()) > 300:
                description += "…"

            # Format pubDate as RFC 2822
            try:
                if pub_date_raw:
                    dt = datetime.fromisoformat(pub_date_raw.replace("Z", "+00:00"))
                    pub_date = formatdate(calendar.timegm(dt.timetuple()))
                else:
                    pub_date = formatdate()
            except Exception:
                pub_date = formatdate()

            items.append(
                f"    <item>\n"
                f"      <title>{html_mod.escape(title)}</title>\n"
                f"      <link>{html_mod.escape(link)}</link>\n"
                f"      <guid isPermaLink=\"true\">{html_mod.escape(link)}</guid>\n"
                f"      <description>{html_mod.escape(description)}</description>\n"
                f"      <pubDate>{pub_date}</pubDate>\n"
                f"      <author>{html_mod.escape(username)}@legate.studio ({html_mod.escape(author_name)})</author>\n"
                f"    </item>"
            )
        return items

    @app.route("/feed.xml")
    def rss_feed_global():
        """Global RSS feed — latest 50 published notes across all users."""
        import sqlite3
        from email.utils import formatdate

        from .rag.database import get_user_db_path, init_db

        notes_with_author = []
        try:
            shared_db = init_db()
            users = shared_db.execute(
                "SELECT u.user_id, u.github_login, COALESCE(p.display_name, u.name, u.github_login) AS display_name "
                "FROM users u LEFT JOIN user_profiles p ON u.user_id = p.user_id"
            ).fetchall()
            for user_row in users:
                user_id = user_row["user_id"]
                github_login = user_row["github_login"]
                if not github_login:
                    continue
                try:
                    user_db_path = get_user_db_path(user_id)
                    if not user_db_path.exists():
                        continue
                    uconn = sqlite3.connect(str(user_db_path))
                    uconn.row_factory = sqlite3.Row
                    notes = uconn.execute(
                        "SELECT title, slug, content, published_at, updated_at "
                        "FROM knowledge_entries WHERE published = 1 AND slug IS NOT NULL "
                        "ORDER BY COALESCE(published_at, updated_at) DESC"
                    ).fetchall()
                    uconn.close()
                    for note in notes:
                        row = dict(note)
                        row["display_name"] = user_row["display_name"]
                        notes_with_author.append((row, github_login))
                except Exception as e:
                    logger.warning(f"RSS global: failed to read notes for {user_id}: {e}")
        except Exception as e:
            logger.warning(f"RSS global: failed to enumerate user DBs: {e}")

        # Sort all notes by published_at DESC, take latest 50
        notes_with_author.sort(
            key=lambda x: x[0].get("published_at") or x[0].get("updated_at") or "",
            reverse=True,
        )

        last_build = formatdate()
        items = _build_rss_items(notes_with_author, limit=50)

        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">\n'
            "  <channel>\n"
            "    <title>Legate Studio — Latest Notes</title>\n"
            "    <link>https://legate.studio</link>\n"
            "    <description>The latest published notes from Legate Studio knowledge bases.</description>\n"
            "    <language>en-us</language>\n"
            f"    <lastBuildDate>{last_build}</lastBuildDate>\n"
            '    <atom:link href="https://legate.studio/feed.xml" rel="self" type="application/rss+xml"/>\n'
            + "\n".join(items)
            + "\n  </channel>\n</rss>\n"
        )
        return xml, 200, {"Content-Type": "application/rss+xml; charset=utf-8"}

    @app.route("/pub/<username>/feed.xml")
    def rss_feed_user(username: str):
        """Per-user RSS feed of their published notes."""
        from email.utils import formatdate

        from .rag.database import get_public_profile, get_user_db_path

        profile = get_public_profile(username)
        if not profile:
            return "", 404

        user_id = profile["user_id"]
        display_name = profile.get("display_name") or profile.get("name") or username
        user_db_path = get_user_db_path(user_id)

        notes_with_author = []
        if user_db_path.exists():
            try:
                import sqlite3 as _sq3
                uconn = _sq3.connect(str(user_db_path))
                uconn.row_factory = _sq3.Row
                notes = uconn.execute(
                    "SELECT title, slug, content, published_at, updated_at "
                    "FROM knowledge_entries WHERE published = 1 AND slug IS NOT NULL "
                    "ORDER BY COALESCE(published_at, updated_at) DESC"
                ).fetchall()
                uconn.close()
                for note in notes:
                    row = dict(note)
                    row["display_name"] = display_name
                    notes_with_author.append((row, username))
            except Exception as e:
                logger.warning(f"RSS user feed: failed for {username}: {e}")

        last_build = formatdate()
        items = _build_rss_items(notes_with_author)

        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">\n'
            "  <channel>\n"
            f"    <title>{display_name} — Notes on Legate Studio</title>\n"
            f"    <link>https://legate.studio/pub/{username}</link>\n"
            f"    <description>Published notes from {display_name} on Legate Studio.</description>\n"
            "    <language>en-us</language>\n"
            f"    <lastBuildDate>{last_build}</lastBuildDate>\n"
            f'    <atom:link href="https://legate.studio/pub/{username}/feed.xml"'
            ' rel="self" type="application/rss+xml"/>\n'
            + "\n".join(items)
            + "\n  </channel>\n</rss>\n"
        )
        return xml, 200, {"Content-Type": "application/rss+xml; charset=utf-8"}

    # ============ Shared library public routes (no auth) ============
    # IMPORTANT: 3-segment routes must be registered BEFORE 2-segment routes
    # to avoid Flask matching /pub/<owner>/<lib_slug> as /pub/<username>/<slug>.

    @app.route("/pub/<owner>/<lib_slug>/")
    def pub_shared_library_profile(owner: str, lib_slug: str):
        """Render a public landing page for a shared library. No authentication required."""
        import sqlite3

        from .rag.database import get_public_profile, get_shared_library_db_path, init_db

        # Look up the owner user
        profile = get_public_profile(owner)
        if not profile:
            return render_template("error.html", title="Not Found", message="User not found"), 404

        owner_user_id = profile["user_id"]

        # Find the shared library by owner + slug
        shared_meta = init_db()
        lib_row = shared_meta.execute(
            """
            SELECT id, name, slug, description, created_at
            FROM shared_libraries
            WHERE owner_user_id = ? AND slug = ? AND status = 'active'
            """,
            (owner_user_id, lib_slug),
        ).fetchone()

        if not lib_row:
            return render_template("error.html", title="Not Found", message="Shared library not found"), 404

        library_id = lib_row["id"]
        library_name = lib_row["name"]
        library_description = lib_row["description"]

        # Open shared library DB and query published notes
        lib_db_path = get_shared_library_db_path(library_id)
        notes = []
        if lib_db_path.exists():
            try:
                lconn = sqlite3.connect(str(lib_db_path))
                lconn.row_factory = sqlite3.Row
                note_rows = lconn.execute(
                    """
                    SELECT title, slug, published_at, category, content
                    FROM knowledge_entries
                    WHERE published = 1 AND slug IS NOT NULL
                    ORDER BY published_at DESC
                    """
                ).fetchall()
                lconn.close()
                for row in note_rows:
                    d = dict(row)
                    content = d.get("content") or ""
                    d["preview"] = content[:200].replace("\n", " ").strip()
                    notes.append(d)
            except Exception as e:
                logger.error(f"pub_shared_library_profile: failed to query library DB {library_id}: {e}")

        canonical_url = f"https://legate.studio/pub/{owner}/{lib_slug}/"
        author_avatar = profile.get("avatar_url") or ""

        return render_template(
            "shared_library_profile.html",
            owner=owner,
            profile=profile,
            library_id=library_id,
            library_name=library_name,
            library_slug=lib_slug,
            library_description=library_description,
            notes=notes,
            canonical_url=canonical_url,
            author_avatar=author_avatar,
            og_title=f"{library_name} — Legate Studio",
            og_description=(
                library_description
                or f"Shared library by {profile.get('display_name') or owner} on Legate Studio."
            ),
            og_url=canonical_url,
        )

    @app.route("/pub/<owner>/<lib_slug>/<note_slug>")
    def pub_shared_library_note(owner: str, lib_slug: str, note_slug: str):
        """Render a published note from a shared library. No authentication required."""
        import json
        import sqlite3

        from .rag.database import get_public_profile, get_shared_library_db_path, init_db

        # Look up the owner user
        profile = get_public_profile(owner)
        if not profile:
            return render_template("error.html", title="Not Found", message="User not found"), 404

        owner_user_id = profile["user_id"]

        # Find the shared library
        shared_meta = init_db()
        lib_row = shared_meta.execute(
            """
            SELECT id, name, slug
            FROM shared_libraries
            WHERE owner_user_id = ? AND slug = ? AND status = 'active'
            """,
            (owner_user_id, lib_slug),
        ).fetchone()

        if not lib_row:
            return render_template("error.html", title="Not Found", message="Shared library not found"), 404

        library_id = lib_row["id"]
        library_name = lib_row["name"]

        # Open library DB and find the note
        lib_db_path = get_shared_library_db_path(library_id)
        if not lib_db_path.exists():
            return render_template("error.html", title="Not Found", message="Note not found"), 404

        try:
            lconn = sqlite3.connect(str(lib_db_path))
            lconn.row_factory = sqlite3.Row
            note = lconn.execute(
                "SELECT * FROM knowledge_entries WHERE slug = ? AND published = 1",
                (note_slug,),
            ).fetchone()
            lconn.close()
        except Exception as e:
            logger.error(f"pub_shared_library_note: failed to query DB for {owner}/{lib_slug}/{note_slug}: {e}")
            return render_template("error.html", title="Server Error", message="An error occurred"), 500

        if not note:
            return render_template("error.html", title="Not Found", message="Note not found or not published"), 404

        note_dict = dict(note)

        from .library import render_markdown
        content_html = render_markdown(note_dict.get("content", ""))

        domain_tags = []
        if note_dict.get("domain_tags"):
            try:
                domain_tags = json.loads(note_dict["domain_tags"])
            except (json.JSONDecodeError, TypeError):
                domain_tags = []

        key_phrases = []
        if note_dict.get("key_phrases"):
            try:
                key_phrases = json.loads(note_dict["key_phrases"])
            except (json.JSONDecodeError, TypeError):
                key_phrases = []

        canonical_url = f"https://legate.studio/pub/{owner}/{lib_slug}/{note_slug}"
        published_at = note_dict.get("published_at") or note_dict.get("created_at", "")
        updated_at = note_dict.get("updated_at") or published_at

        accent_color = profile.get("accent_color") or "#a855f7"
        author_display_name = profile.get("display_name") or owner
        author_profile_url = f"/pub/{owner}/{lib_slug}/"
        author_avatar = profile.get("avatar_url") or ""

        return render_template(
            "published_note.html",
            note=note_dict,
            content_html=content_html,
            domain_tags=domain_tags,
            key_phrases=key_phrases,
            username=owner,
            slug=note_slug,
            canonical_url=canonical_url,
            published_at=published_at,
            updated_at=updated_at,
            report_url=f"/pub/{owner}/{lib_slug}/{note_slug}/report",
            # Author profile
            accent_color=accent_color,
            author_display_name=author_display_name,
            author_profile_url=author_profile_url,
            author_avatar=author_avatar,
            # Library context
            library_name=library_name,
            library_url=f"/pub/{owner}/{lib_slug}/",
            # OG / SEO
            og_title=note_dict.get("title", "Note"),
            og_description=(note_dict.get("content", "")[:160].replace("\n", " ") if note_dict.get("content") else ""),
            og_url=canonical_url,
        )

    # ============ Public profile routes (no auth) ============

    @app.route("/pub/<username>")
    def pub_profile(username: str):
        """Render a public profile page for a user. No authentication required."""
        import json
        import sqlite3

        from .rag.database import get_public_profile, get_user_db_path

        # Look up user by github_login
        profile = get_public_profile(username)
        if not profile:
            return render_template("error.html", title="Not Found", message="User not found"), 404

        user_id = profile["user_id"]
        user_db_path = get_user_db_path(user_id)

        if not user_db_path.exists():
            return render_template("error.html", title="Not Found", message="No published notes"), 404

        # Query per-user DB for all published notes
        try:
            uconn = sqlite3.connect(str(user_db_path))
            uconn.row_factory = sqlite3.Row
            notes_rows = uconn.execute(
                """
                SELECT title, slug, published_at, category, content
                FROM knowledge_entries
                WHERE published = 1 AND slug IS NOT NULL
                ORDER BY published_at DESC
                """
            ).fetchall()
            uconn.close()
        except Exception as e:
            logger.error(f"pub_profile: failed to query user DB for {username}: {e}")
            return render_template("error.html", title="Server Error", message="An error occurred"), 500

        if not notes_rows:
            return render_template("error.html", title="Not Found", message="No published notes"), 404

        # Build note summaries (first 200 chars of content as preview)
        notes = []
        for row in notes_rows:
            d = dict(row)
            content = d.get("content") or ""
            d["preview"] = content[:200].replace("\n", " ").strip()
            notes.append(d)

        # Parse custom_links JSON
        custom_links = []
        try:
            custom_links = json.loads(profile.get("custom_links") or "[]")
        except (json.JSONDecodeError, TypeError):
            custom_links = []

        canonical_url = f"https://legate.studio/pub/{username}"

        # Check if the viewer is the profile owner
        viewer = session.get("user", {})
        is_own_profile = bool(viewer and viewer.get("username") == username)

        return render_template(
            "profile.html",
            profile=profile,
            username=username,
            notes=notes,
            custom_links=custom_links,
            accent_color=profile.get("accent_color", "#a855f7"),
            layout_pref=profile.get("layout_pref", "grid"),
            canonical_url=canonical_url,
            is_own_profile=is_own_profile,
            # OG/SEO
            og_title=f"{profile.get('display_name') or username} — Legate Studio",
            og_description=profile.get("bio") or f"Published notes by {username} on Legate Studio.",
            og_image=profile.get("avatar_url") or "https://legate.studio/static/img/og-share.png",
            og_url=canonical_url,
        )

    # ============ Profile settings API (auth required) ============

    @app.route("/settings/profile", methods=["GET", "POST"])
    @login_required
    def settings_profile():
        """Get or update the current user's profile settings."""
        import json
        import re

        from .rag.database import get_user_profile, update_user_profile

        user = session.get("user", {})
        user_id = user.get("user_id")

        if request.method == "GET":
            profile = get_user_profile(user_id)
            return jsonify(profile)

        # POST — validate and upsert
        data = request.get_json(silent=True) or {}

        updates = {}

        # display_name: max 50 chars
        if "display_name" in data:
            dn = str(data["display_name"]).strip()[:50] if data["display_name"] else None
            updates["display_name"] = dn

        # bio: max 500 chars
        if "bio" in data:
            bio = str(data["bio"]).strip()[:500] if data["bio"] else None
            updates["bio"] = bio

        # accent_color: must be a valid hex color
        if "accent_color" in data:
            color = str(data["accent_color"]).strip()
            if re.match(r'^#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?$', color):
                updates["accent_color"] = color
            else:
                return jsonify({"error": "Invalid accent_color. Must be a hex color like #a855f7."}), 400

        # layout_pref: grid or list
        if "layout_pref" in data:
            lp = str(data["layout_pref"]).strip()
            if lp in ("grid", "list"):
                updates["layout_pref"] = lp
            else:
                return jsonify({"error": "layout_pref must be 'grid' or 'list'."}), 400

        # custom_links: JSON array, max 3 items, each {label, url}
        if "custom_links" in data:
            links = data["custom_links"]
            if not isinstance(links, list):
                return jsonify({"error": "custom_links must be a JSON array."}), 400
            links = links[:3]  # enforce max 3
            clean = []
            for item in links:
                if isinstance(item, dict) and "label" in item and "url" in item:
                    clean.append({
                        "label": str(item["label"])[:50],
                        "url": str(item["url"])[:500],
                    })
            updates["custom_links"] = json.dumps(clean)

        if updates:
            update_user_profile(user_id, **updates)

        return jsonify({"ok": True, "updated": list(updates.keys())})

    # ============ Public note routes (no auth) ============

    @app.route("/pub/<username>/<slug>")
    def pub_note(username: str, slug: str):
        """Render a publicly published note. No authentication required."""
        import json
        import sqlite3

        from .rag.database import get_public_profile, get_user_db_path

        # Look up user by github_login in shared DB (and get their profile)
        profile = get_public_profile(username)
        if not profile:
            return render_template("error.html", title="Not Found", message="User not found"), 404

        user_id = profile["user_id"]
        user_db_path = get_user_db_path(user_id)
        if not user_db_path.exists():
            return render_template("error.html", title="Not Found", message="Note not found"), 404

        # Open user DB and find the published note
        try:
            uconn = sqlite3.connect(str(user_db_path))
            uconn.row_factory = sqlite3.Row
            note = uconn.execute(
                "SELECT * FROM knowledge_entries WHERE slug = ? AND published = 1",
                (slug,),
            ).fetchone()
            uconn.close()
        except Exception as e:
            logger.error(f"pub_note: failed to query user DB for {username}/{slug}: {e}")
            return render_template("error.html", title="Server Error", message="An error occurred"), 500

        if not note:
            return render_template("error.html", title="Not Found", message="Note not found or not published"), 404

        note_dict = dict(note)

        # Render markdown → safe HTML using same pipeline as library
        from .library import render_markdown
        content_html = render_markdown(note_dict.get("content", ""))

        # Parse JSON tags
        domain_tags = []
        if note_dict.get("domain_tags"):
            try:
                domain_tags = json.loads(note_dict["domain_tags"])
            except (json.JSONDecodeError, TypeError):
                domain_tags = []

        key_phrases = []
        if note_dict.get("key_phrases"):
            try:
                key_phrases = json.loads(note_dict["key_phrases"])
            except (json.JSONDecodeError, TypeError):
                key_phrases = []

        canonical_url = f"https://legate.studio/pub/{username}/{slug}"
        published_at = note_dict.get("published_at") or note_dict.get("created_at", "")
        updated_at = note_dict.get("updated_at") or published_at

        # Author profile fields (accent_color, display_name for the template)
        accent_color = profile.get("accent_color") or "#a855f7"
        author_display_name = profile.get("display_name") or username
        author_profile_url = f"/pub/{username}"
        author_avatar = profile.get("avatar_url") or ""

        return render_template(
            "published_note.html",
            note=note_dict,
            content_html=content_html,
            domain_tags=domain_tags,
            key_phrases=key_phrases,
            username=username,
            slug=slug,
            canonical_url=canonical_url,
            published_at=published_at,
            updated_at=updated_at,
            report_url=f"/pub/{username}/{slug}/report",
            # Author profile
            accent_color=accent_color,
            author_display_name=author_display_name,
            author_profile_url=author_profile_url,
            author_avatar=author_avatar,
            # OG / SEO vars
            og_title=note_dict.get("title", "Note"),
            og_description=(note_dict.get("content", "")[:160].replace("\n", " ") if note_dict.get("content") else ""),
            og_url=canonical_url,
        )

    @app.route("/pub/<username>/<slug>/report", methods=["GET", "POST"])
    def pub_note_report(username: str, slug: str):
        """Report a published note for content moderation."""
        import sqlite3

        from .rag.database import get_connection, get_db_path, get_user_db_path, init_db

        # Verify note exists and is published
        shared_db = init_db()
        user_row = shared_db.execute(
            "SELECT user_id FROM users WHERE github_login = ?", (username,)
        ).fetchone()

        note_exists = False
        if user_row:
            user_db_path = get_user_db_path(user_row["user_id"])
            if user_db_path.exists():
                try:
                    uconn = sqlite3.connect(str(user_db_path))
                    uconn.row_factory = sqlite3.Row
                    n = uconn.execute(
                        "SELECT entry_id FROM knowledge_entries WHERE slug = ? AND published = 1",
                        (slug,),
                    ).fetchone()
                    uconn.close()
                    note_exists = bool(n)
                except Exception:
                    pass

        if not note_exists:
            return render_template("error.html", title="Not Found", message="Note not found"), 404

        submitted = False
        error = None

        if request.method == "POST":
            reason = request.form.get("reason", "").strip()
            details = request.form.get("details", "").strip()
            valid_reasons = ["spam", "harassment", "misinformation", "illegal", "other"]

            if not reason or reason not in valid_reasons:
                error = "Please select a valid reason."
            else:
                try:
                    # Store report in the shared DB's content_reports table
                    shared_conn = get_connection(get_db_path("legato.db"))
                    shared_conn.execute(
                        """
                        INSERT INTO content_reports (reported_username, slug, reason, details, reporter_ip)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            username,
                            slug,
                            reason,
                            details[:2000] if details else None,
                            request.remote_addr,
                        ),
                    )
                    shared_conn.commit()
                    shared_conn.close()
                    submitted = True
                except Exception as e:
                    logger.error(f"pub_note_report: failed to store report for {username}/{slug}: {e}")
                    error = "Failed to submit report. Please try again."

        return render_template(
            "report_note.html",
            username=username,
            slug=slug,
            note_url=f"/pub/{username}/{slug}",
            submitted=submitted,
            error=error,
        )

    # Health check (no auth required)
    @app.route("/health")
    def health():
        return jsonify(
            {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "app": app.config["APP_NAME"],
            }
        )

    # Error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template("error.html", title="Not Found", message="Page not found"), 404

    @app.errorhandler(500)
    def internal_error(error):
        return render_template("error.html", title="Server Error", message="An error occurred"), 500

    @app.errorhandler(429)
    def ratelimit_error(error):
        return render_template("error.html", title="Rate Limited", message="Too many requests. Please wait."), 429

    # Register is_feature_available as a Jinja2 global so templates can call it
    app.jinja_env.globals["is_feature_available"] = is_feature_available

    logger.info("Legate Studio application initialized")
    return app


def login_required(f):
    """Decorator to require authentication."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)

    return decorated_function


def get_user_tier(user_id: str) -> str:
    """Get a user's tier from the shared users table."""
    if not user_id:
        return "free"

    from .rag.database import init_db

    db = init_db()
    row = db.execute(
        "SELECT tier FROM users WHERE user_id = ?",
        (user_id,),
    ).fetchone()

    if row and row["tier"]:
        return row["tier"]

    return "free"


def get_current_user_tier() -> str:
    """Get the current session user's effective tier.

    In single-tenant mode, this returns a non-free value so SaaS gating is bypassed.
    Uses get_effective_tier() which properly handles beta users (they get 'managed' tier).
    """
    from flask import current_app

    if current_app.config.get("LEGATO_MODE") != "multi-tenant":
        return "single-tenant"

    user_id = session.get("user", {}).get("user_id")
    return get_effective_tier(user_id)


def is_paid_tier(tier: str) -> bool:
    """Check if tier is a paid tier (not free/trial)."""
    return bool(tier) and tier not in ("free", "trial")


def get_trial_status(user_id: str) -> dict:
    """Get trial status for a user.

    Returns:
        dict with is_trial, days_remaining, is_expired, trial_started_at
    """
    from datetime import datetime, timedelta

    from .rag.database import init_db

    trial_days = 14

    if not user_id:
        return {"is_trial": False, "days_remaining": 0, "is_expired": True}

    db = init_db()
    row = db.execute("SELECT tier, trial_started_at, is_beta FROM users WHERE user_id = ?", (user_id,)).fetchone()

    if not row:
        return {"is_trial": False, "days_remaining": 0, "is_expired": True}

    # Beta users are never in trial (check both is_beta flag and legacy tier='beta')
    if row["is_beta"] or row["tier"] == "beta":
        return {"is_trial": False, "days_remaining": 0, "is_expired": False, "is_beta": True}

    # Paid users are not in trial
    if row["tier"] in ("byok", "managed"):
        return {"is_trial": False, "days_remaining": 0, "is_expired": False}

    # Check trial status
    trial_started = row["trial_started_at"]
    if not trial_started:
        # Legacy user without trial_started_at - treat as expired
        return {"is_trial": True, "days_remaining": 0, "is_expired": True}

    try:
        start_date = datetime.fromisoformat(trial_started.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        start_date = datetime.now()

    # Make start_date naive if it's timezone-aware
    if start_date.tzinfo is not None:
        start_date = start_date.replace(tzinfo=None)

    expiry_date = start_date + timedelta(days=trial_days)
    now = datetime.now()
    days_remaining = (expiry_date - now).days

    return {
        "is_trial": True,
        "days_remaining": max(0, days_remaining),
        "is_expired": now > expiry_date,
        "trial_started_at": trial_started,
    }


def is_trial_expired(user_id: str) -> bool:
    """Check if user's trial has expired."""
    status = get_trial_status(user_id)
    return status.get("is_expired", False) and status.get("is_trial", False)


def get_effective_tier(user_id: str) -> str:
    """Get effective subscription tier for a user.

    Supported tier strings:
      'trial'            — free/unsubscribed
      'byok'             — BYOK ($0.99/mo), unlimited, user provides own keys
      'managed_lite'     — Managed $2.99/mo, $2.69 token credits
      'managed_standard' — Managed $10/mo, $9.00 token credits
      'managed_plus'     — Managed $20/mo, $18.00 token credits

    Legacy values in the DB are mapped for backward compat:
      'managed' / 'beta' → 'managed_lite'
      'beta' flag (is_beta=1) → 'managed_lite'
    """
    from .rag.database import init_db

    if not user_id:
        return "trial"

    db = init_db()
    row = db.execute("SELECT tier, is_beta FROM users WHERE user_id = ?", (user_id,)).fetchone()

    if not row:
        return "trial"

    # Beta flag → managed_lite (free managed access for beta testers)
    if row["is_beta"] or row["tier"] == "beta":
        return "managed_lite"

    tier = row["tier"] or "trial"

    # New granular managed tiers — pass through directly
    if tier in ("trial", "byok", "managed_lite", "managed_standard", "managed_plus"):
        return tier

    # Legacy 'managed' → managed_lite for backward compat
    if tier == "managed":
        return "managed_lite"

    # Anything else (unknown/corrupt) → trial
    return "trial"


def can_use_platform_keys(user_id: str) -> bool:
    """Check if user can use platform API keys (any managed tier or beta)."""
    from .rag.usage import is_managed_tier
    return is_managed_tier(get_effective_tier(user_id))


def get_api_key_for_user(user_id: str, provider: str) -> str:
    """Get API key for user — platform key (managed tiers) or their own BYOK key.

    Args:
        user_id: User's ID
        provider: 'anthropic', 'openai', or 'gemini'

    Returns:
        API key string or None if not available
    """
    import os

    from .auth import get_user_api_key
    from .rag.usage import is_managed_tier

    tier = get_effective_tier(user_id)

    if is_managed_tier(tier):
        # All managed tiers use platform keys from environment
        env_key = f"{provider.upper()}_API_KEY"
        return os.environ.get(env_key)
    else:
        # BYOK / trial — return user's own stored key (may be None)
        return get_user_api_key(user_id, provider)


def paid_required(f):
    """Decorator to require a paid subscription tier in multi-tenant mode."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("auth.login"))

        from flask import current_app

        # Only enforce SaaS tiers in multi-tenant mode
        if current_app.config.get("LEGATO_MODE") != "multi-tenant":
            return f(*args, **kwargs)

        tier = get_current_user_tier()
        if is_paid_tier(tier):
            return f(*args, **kwargs)

        if request.path.startswith("/api/") or request.is_json or request.headers.get("Accept") == "application/json":
            return jsonify(
                {
                    "error": "Subscription required",
                    "upgrade_required": True,
                }
            ), 402

        flash("This feature requires a subscription.", "warning")
        return redirect(url_for("auth.setup"))

    return decorated_function


def library_required(f):
    """Decorator to require Library setup before accessing features.

    Must be used after @login_required. Checks if user has configured
    their Library repo. If not, attempts to auto-repair from installation,
    then redirects to setup page if repair fails.
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("auth.login"))

        # Check if Library is configured (multi-tenant mode only)
        from flask import current_app

        if current_app.config.get("LEGATO_MODE") == "multi-tenant":
            user = session["user"]
            user_id = user.get("user_id")

            if user_id:
                from .rag.database import init_db

                db = init_db()  # Shared DB for user_repos
                library = db.execute(
                    "SELECT 1 FROM user_repos WHERE user_id = ? AND repo_type = 'library'",
                    (user_id,),
                ).fetchone()

                if not library:
                    # Attempt auto-repair before redirecting to setup
                    from .auth import _repair_user_repos_from_installation

                    if _repair_user_repos_from_installation(user_id, db):
                        # Repair succeeded, continue to page
                        pass
                    else:
                        flash("Please set up your Library first.", "info")
                        return redirect(url_for("auth.setup"))

        return f(*args, **kwargs)

    return decorated_function


def beta_gate(feature_name: str):
    """Decorator to gate a feature behind per-user beta entitlement.

    Access is granted only if the current user has an enabled=1 row in
    user_feature_access for this feature_name.

    If feature_name is not registered in feature_flags at all, access is
    DENIED (fail-closed) — unknown features are always gated.

    Usage: @beta_gate("chat")
    Must be used after @login_required.
    Stack order: @route → @login_required → @paid_required → @beta_gate(...)
    """

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if "user" not in session:
                flash("Please log in to access this page.", "warning")
                return redirect(url_for("auth.login"))

            user = session["user"]
            user_id = user.get("user_id")

            # Check per-user entitlement in user_feature_access
            from legate_studio.rag.database import init_db

            db = init_db()
            row = db.execute(
                "SELECT enabled FROM user_feature_access WHERE user_id = ? AND feature_name = ?",
                (user_id, feature_name),
            ).fetchone()

            if row and row["enabled"]:
                return f(*args, **kwargs)

            # No entitlement — deny access
            if (
                request.path.startswith("/api/")
                or request.is_json
                or request.headers.get("Accept") == "application/json"
            ):
                return jsonify(
                    {
                        "error": f"The {feature_name} feature is currently in beta. Contact an admin for access.",
                        "beta_required": True,
                        "feature": feature_name,
                    }
                ), 403
            else:
                flash("This feature is currently in beta. Contact an admin for access.", "warning")
                return redirect(url_for("dashboard.index"))

        decorated_function._beta_feature = feature_name  # tag for introspection
        return decorated_function

    return decorator


def is_feature_available(feature_name: str, user: dict = None) -> bool:
    """Check if a feature is available to the current user.

    Returns True ONLY if the user has an enabled=1 row in user_feature_access
    for this feature. No global release flag — access is always per-user.

    Use in Jinja templates: {{ is_feature_available('chat') }}
    Use in Python: is_feature_available('chat', user_dict)
    """
    if user is None:
        user = session.get("user", {})

    user_id = user.get("user_id")
    if not user_id:
        return False

    from legate_studio.rag.database import init_db

    db = init_db()
    row = db.execute(
        "SELECT enabled FROM user_feature_access WHERE user_id = ? AND feature_name = ?",
        (user_id, feature_name),
    ).fetchone()

    return bool(row and row["enabled"])


def copilot_required(f):
    """Decorator to require Copilot access for Chords/Agents features.

    Must be used after @login_required. Checks if user has Copilot enabled.
    If not, returns 403 for APIs or redirects to dashboard for pages.

    DEPRECATED: Use @beta_gate("feature_name") instead for new feature gating.
    Kept for backward compatibility.
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("auth.login"))

        user = session["user"]
        has_copilot = user.get("has_copilot", False)

        if not has_copilot:
            # Check if this is an API request
            from flask import request

            if (
                request.path.startswith("/api/")
                or request.is_json
                or request.headers.get("Accept") == "application/json"
            ):
                return jsonify(
                    {
                        "error": (
                            "Chords and Agents features require GitHub Copilot. Enable Copilot on"
                            "your account to use these features."
                        ),
                        "copilot_required": True,
                    }
                ), 403
            else:
                flash(
                    (
                        "Chords and Agents features require GitHub Copilot. Enable Copilot on your"
                        "account to use these features."
                    ),
                    "warning",
                )
                return redirect(url_for("dashboard.index"))

        return f(*args, **kwargs)

    return decorated_function


def get_user_library_repo(user_id: str = None) -> str:
    """Get the user's configured Library repository.

    Looks up the Library repo from user_repos table. Falls back to
    default patterns if not configured.

    Args:
        user_id: User ID to look up. If None, uses current session user.

    Returns:
        Full repo name (e.g., 'username/Legate.Library.username')
    """
    from flask import current_app

    if user_id is None:
        user = session.get("user", {})
        user_id = user.get("user_id")
        username = user.get("username")
    else:
        username = None

    # In multi-tenant mode, look up from database
    if current_app.config.get("LEGATO_MODE") == "multi-tenant" and user_id:
        from .rag.database import init_db

        db = init_db()  # Shared DB for user_repos
        row = db.execute(
            "SELECT repo_full_name FROM user_repos WHERE user_id = ? AND repo_type = 'library'",
            (user_id,),
        ).fetchone()

        if row:
            return row["repo_full_name"]

    # Fallback: use default pattern
    if username:
        return f"{username}/Legate.Library.{username}"

    # Last resort: env var or hardcoded default
    import os

    return os.environ.get("LIBRARY_REPO", "bobbyhiddn/Legate.Library")
