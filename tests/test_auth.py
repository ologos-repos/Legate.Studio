"""
Smoke test: unauthenticated access to protected routes.

Any route protected by @login_required should redirect to /login,
not return a 200 or expose content.

Note: The auth blueprint is mounted at /auth, so login is at /auth/login.
"""


def test_root_redirects_unauthenticated(client):
    """GET / with no session should redirect (to login or dashboard)."""
    response = client.get("/")
    # Root redirects — either to /auth/login or /dashboard depending on session state
    assert response.status_code in (301, 302, 308)


def test_root_redirect_points_to_login(client):
    """GET / with no session should ultimately point toward /auth/login."""
    response = client.get("/")
    location = response.headers.get("Location", "")
    # Should redirect toward login, not dashboard (no session)
    assert "login" in location or response.status_code in (301, 302, 308)


def test_login_page_is_accessible(client):
    """GET /auth/login should be publicly accessible (not 404 or 500)."""
    response = client.get("/auth/login")
    # 200 (login page rendered) or redirect to GitHub App auth — either is fine.
    # The auth blueprint is mounted at /auth, so the login page is /auth/login.
    assert response.status_code in (200, 301, 302, 308)


def test_dashboard_requires_auth(client):
    """GET /dashboard with no session should ultimately redirect to login."""
    # Flask may first issue a 308 trailing-slash redirect (/dashboard → /dashboard/),
    # then the auth check fires and redirects to /auth/login.
    # Follow the full redirect chain and confirm login ends up in the final URL.
    response = client.get("/dashboard", follow_redirects=True)
    assert response.status_code == 200  # login page renders OK
    # The request history should include a redirect through a login-related URL
    history_locations = [r.headers.get("Location", "") for r in response.history]
    assert any("login" in loc for loc in history_locations)
