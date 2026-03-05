"""
Smoke test: health endpoint.

/health must return 200 with no authentication — it's used by Fly.io health checks.
"""


def test_health_returns_200(client):
    """GET /health should always return 200, no auth required."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_json(client):
    """GET /health should return JSON with a status field."""
    response = client.get("/health")
    data = response.get_json()
    assert data is not None
    assert data.get("status") == "healthy"


def test_health_contains_app_name(client):
    """GET /health JSON should include the app name."""
    response = client.get("/health")
    data = response.get_json()
    assert "app" in data
