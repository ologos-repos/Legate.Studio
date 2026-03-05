"""
Smoke test: MCP endpoint authentication.

POST /mcp without a valid Bearer token must return 401.
This verifies the OAuth guard is wired up correctly.
"""
import json


def test_mcp_post_without_auth_returns_401(client):
    """POST /mcp with no Authorization header should return 401."""
    response = client.post(
        "/mcp",
        data=json.dumps({"jsonrpc": "2.0", "method": "initialize", "id": 1}),
        content_type="application/json",
    )
    assert response.status_code == 401


def test_mcp_post_with_bad_bearer_returns_401(client):
    """POST /mcp with an invalid Bearer token should return 401."""
    response = client.post(
        "/mcp",
        data=json.dumps({"jsonrpc": "2.0", "method": "initialize", "id": 1}),
        content_type="application/json",
        headers={"Authorization": "Bearer totally-fake-token-xyz"},
    )
    assert response.status_code == 401


def test_mcp_head_returns_200(client):
    """HEAD /mcp (protocol discovery) should work without auth."""
    response = client.head("/mcp")
    assert response.status_code == 200


def test_mcp_options_returns_200(client):
    """OPTIONS /mcp (CORS preflight) should work without auth."""
    response = client.options("/mcp")
    assert response.status_code == 200
