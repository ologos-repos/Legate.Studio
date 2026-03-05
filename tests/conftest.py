"""
Legate Studio test fixtures.

Creates a minimal Flask test client that works without any external services
(no OPENAI_API_KEY, STRIPE_SECRET_KEY, SYSTEM_PAT, etc.).
"""
import os
import pytest

# Set all required env vars BEFORE importing the app to prevent startup errors.
# These are test-only values — no real credentials.
os.environ.setdefault("FLASK_ENV", "testing")
os.environ.setdefault("FLASK_SECRET_KEY", "test-secret-key-not-for-production")
os.environ.setdefault("DATA_DIR", "/tmp/legato-test-data")

# Silence optional service initialization — leave keys unset so those
# code paths are skipped gracefully (Stripe, OpenAI, GitHub OAuth, etc.)


@pytest.fixture(scope="session")
def app():
    """Create a Flask application configured for testing."""
    from legate_studio.core import create_app

    flask_app = create_app()
    flask_app.config.update(
        TESTING=True,
        SECRET_KEY="test-secret-key-not-for-production",
        WTF_CSRF_ENABLED=False,
        # Disable rate limiting in tests
        RATELIMIT_ENABLED=False,
        RATELIMIT_STORAGE_URI="memory://",
        # Ensure we don't accidentally hit real GitHub
        GITHUB_CLIENT_ID=None,
        GITHUB_CLIENT_SECRET=None,
        GITHUB_APP_CLIENT_ID=None,
        GITHUB_APP_CLIENT_SECRET=None,
    )
    return flask_app


@pytest.fixture(scope="session")
def client(app):
    """A test client for the Flask app."""
    return app.test_client()
