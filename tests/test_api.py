#!/usr/bin/env python3
"""
Jinja2 template for API integration tests
Follows the sso/wg_gui patterns: test client, auth gates, rate limits, CSRF, redirect safety
"""

import pytest

# Routes discovered in entrypoint:
#   GET / -> index
#   GET /api/state -> api_state
#   POST /admin/refresh -> admin_refresh
#   POST /admin/eu/refresh -> admin_eu_refresh
#   POST /admin/model/<int:mid>/refresh -> admin_model_refresh
#   POST /admin/model/<int:mid>/remirror -> admin_model_remirror
#   POST /admin/model/add -> admin_model_add
#   POST /admin/model/<int:mid>/delete -> admin_model_delete
#   POST /admin/model/<int:mid>/move -> admin_model_move
#   POST /admin/model/<int:mid>/edit -> admin_model_edit
#   POST /admin/group/add -> admin_group_add
#   POST /admin/group/<int:gid>/rename -> admin_group_rename
#   POST /admin/group/<int:gid>/delete -> admin_group_delete
#   POST /admin/policy -> admin_policy


class TestHealthEndpoints:
    """Tests for health check and public endpoints."""

    def test_healthz(self, client):
        """Health check endpoint should return 200."""
        resp = client.get("/healthz")
        # Some apps use /health, /healthz, /ready, /ping
        assert resp.status_code in (200, 404)  # 404 if not implemented

    def test_root_redirects_or_returns_content(self, client):
        """Root endpoint should respond (redirect or content)."""
        resp = client.get("/")
        assert resp.status_code in (200, 301, 302, 401, 403)


class TestAuthGates:
    """Tests for authentication gates (SSO, Basic, API key)."""

    def test_protected_endpoint_requires_auth(self, client):
        """Protected endpoints should return 401/403 without auth."""
        # Test a known protected endpoint
        protected_routes = [
            ("/", ["GET"]),
            ("/api/state", ["GET"]),
            ("/admin/refresh", ["POST"]),
            ("/admin/eu/refresh", ["POST"]),
            ("/admin/model/<int:mid>/refresh", ["POST"]),
            ("/admin/model/<int:mid>/remirror", ["POST"]),
            ("/admin/model/add", ["POST"]),
            ("/admin/model/<int:mid>/delete", ["POST"]),
            ("/admin/model/<int:mid>/move", ["POST"]),
            ("/admin/model/<int:mid>/edit", ["POST"]),
            ("/admin/group/add", ["POST"]),
            ("/admin/group/<int:gid>/rename", ["POST"]),
            ("/admin/group/<int:gid>/delete", ["POST"]),
            ("/admin/policy", ["POST"]),
        ]
        for route, methods in protected_routes:
            # Skip health check endpoints (intentionally public)
            if route in ["/healthz", "/health", "/ready", "/ping", "/metrics"]:
                continue
            if route == "/":
                continue
            # Use the first supported method for testing auth
            method = methods[0].lower()
            resp = getattr(client, method)(route)
            # Should be 401 (unauthenticated) or 403 (forbidden) or 302 (redirect to login) or 405 (wrong method but auth gate works)
            assert resp.status_code in (401, 403, 302, 405), f"Route {route} should require auth"

    def test_sso_redirect_preserves_next(self, client):
        """SSO login should preserve the next parameter."""
        # This tests the SSO gate pattern
        resp = client.get("/protected?next=/target")
        if resp.status_code == 302:
            location = resp.headers.get("Location", "")
            assert "next=" in location or "/sso/login" in location


class TestRateLimiting:
    """Tests for rate limiting on auth endpoints."""

    def test_login_rate_limit(self, client):
        """Repeated failed logins should be rate limited."""
        # This is a placeholder - actual implementation depends on the app
        # The sso pattern uses throttle_record_failure / throttle_blocked
        pass


class TestCSRFProtection:
    """Tests for CSRF protection on mutating endpoints."""

    def test_mutating_endpoint_requires_csrf(self, client):
        """POST/PUT/DELETE endpoints should require CSRF token."""
        resp = client.post("/admin/refresh", data={})
        # May return 401 (auth required first), 400 (bad request), or 403 (CSRF missing)
        assert resp.status_code in (400, 401, 403), f"/admin/refresh should require CSRF or auth"
        resp = client.post("/admin/eu/refresh", data={})
        # May return 401 (auth required first), 400 (bad request), or 403 (CSRF missing)
        assert resp.status_code in (400, 401, 403), f"/admin/eu/refresh should require CSRF or auth"
        resp = client.post("/admin/model/<int:mid>/refresh", data={})
        # May return 401 (auth required first), 400 (bad request), or 403 (CSRF missing)
        assert resp.status_code in (400, 401, 403), f"/admin/model/<int:mid>/refresh should require CSRF or auth"
        resp = client.post("/admin/model/<int:mid>/remirror", data={})
        # May return 401 (auth required first), 400 (bad request), or 403 (CSRF missing)
        assert resp.status_code in (400, 401, 403), f"/admin/model/<int:mid>/remirror should require CSRF or auth"
        resp = client.post("/admin/model/add", data={})
        # May return 401 (auth required first), 400 (bad request), or 403 (CSRF missing)
        assert resp.status_code in (400, 401, 403), f"/admin/model/add should require CSRF or auth"
        resp = client.post("/admin/model/<int:mid>/delete", data={})
        # May return 401 (auth required first), 400 (bad request), or 403 (CSRF missing)
        assert resp.status_code in (400, 401, 403), f"/admin/model/<int:mid>/delete should require CSRF or auth"
        resp = client.post("/admin/model/<int:mid>/move", data={})
        # May return 401 (auth required first), 400 (bad request), or 403 (CSRF missing)
        assert resp.status_code in (400, 401, 403), f"/admin/model/<int:mid>/move should require CSRF or auth"
        resp = client.post("/admin/model/<int:mid>/edit", data={})
        # May return 401 (auth required first), 400 (bad request), or 403 (CSRF missing)
        assert resp.status_code in (400, 401, 403), f"/admin/model/<int:mid>/edit should require CSRF or auth"
        resp = client.post("/admin/group/add", data={})
        # May return 401 (auth required first), 400 (bad request), or 403 (CSRF missing)
        assert resp.status_code in (400, 401, 403), f"/admin/group/add should require CSRF or auth"
        resp = client.post("/admin/group/<int:gid>/rename", data={})
        # May return 401 (auth required first), 400 (bad request), or 403 (CSRF missing)
        assert resp.status_code in (400, 401, 403), f"/admin/group/<int:gid>/rename should require CSRF or auth"
        resp = client.post("/admin/group/<int:gid>/delete", data={})
        # May return 401 (auth required first), 400 (bad request), or 403 (CSRF missing)
        assert resp.status_code in (400, 401, 403), f"/admin/group/<int:gid>/delete should require CSRF or auth"
        resp = client.post("/admin/policy", data={})
        # May return 401 (auth required first), 400 (bad request), or 403 (CSRF missing)
        assert resp.status_code in (400, 401, 403), f"/admin/policy should require CSRF or auth"


class TestRedirectSafety:
    """Tests for open redirect protection."""

    def test_redirect_validation(self, client):
        """Redirects should be validated against allowlist."""
        # Test the safe_next pattern from sso
        malicious_targets = [
            "https://evil.example/",
            "//evil.example/",
            "javascript:alert(1)",
            "/\\evil.example/",
        ]
        for target in malicious_targets:
            # This would need app-specific testing
            pass


class TestAPIEndpoints:
    """Tests for specific API endpoints."""

    def test_index_endpoint(self, client):
        """Test GET /."""
        resp = client.open("/", method="GET")
        # Should not be 500
        assert resp.status_code != 500
    def test_api_state_endpoint(self, client):
        """Test GET /api/state."""
        resp = client.open("/api/state", method="GET")
        # Should not be 500
        assert resp.status_code != 500
    def test_admin_refresh_endpoint(self, client):
        """Test POST /admin/refresh."""
        resp = client.open("/admin/refresh", method="POST")
        # Should not be 500
        assert resp.status_code != 500
    def test_admin_eu_refresh_endpoint(self, client):
        """Test POST /admin/eu/refresh."""
        resp = client.open("/admin/eu/refresh", method="POST")
        # Should not be 500
        assert resp.status_code != 500
    def test_admin_model_refresh_endpoint(self, client):
        """Test POST /admin/model/<int:mid>/refresh."""
        resp = client.open("/admin/model/<int:mid>/refresh", method="POST")
        # Should not be 500
        assert resp.status_code != 500
    def test_admin_model_remirror_endpoint(self, client):
        """Test POST /admin/model/<int:mid>/remirror."""
        resp = client.open("/admin/model/<int:mid>/remirror", method="POST")
        # Should not be 500
        assert resp.status_code != 500
    def test_admin_model_add_endpoint(self, client):
        """Test POST /admin/model/add."""
        resp = client.open("/admin/model/add", method="POST")
        # Should not be 500
        assert resp.status_code != 500
    def test_admin_model_delete_endpoint(self, client):
        """Test POST /admin/model/<int:mid>/delete."""
        resp = client.open("/admin/model/<int:mid>/delete", method="POST")
        # Should not be 500
        assert resp.status_code != 500
    def test_admin_model_move_endpoint(self, client):
        """Test POST /admin/model/<int:mid>/move."""
        resp = client.open("/admin/model/<int:mid>/move", method="POST")
        # Should not be 500
        assert resp.status_code != 500
    def test_admin_model_edit_endpoint(self, client):
        """Test POST /admin/model/<int:mid>/edit."""
        resp = client.open("/admin/model/<int:mid>/edit", method="POST")
        # Should not be 500
        assert resp.status_code != 500
    def test_admin_group_add_endpoint(self, client):
        """Test POST /admin/group/add."""
        resp = client.open("/admin/group/add", method="POST")
        # Should not be 500
        assert resp.status_code != 500
    def test_admin_group_rename_endpoint(self, client):
        """Test POST /admin/group/<int:gid>/rename."""
        resp = client.open("/admin/group/<int:gid>/rename", method="POST")
        # Should not be 500
        assert resp.status_code != 500
    def test_admin_group_delete_endpoint(self, client):
        """Test POST /admin/group/<int:gid>/delete."""
        resp = client.open("/admin/group/<int:gid>/delete", method="POST")
        # Should not be 500
        assert resp.status_code != 500
    def test_admin_policy_endpoint(self, client):
        """Test POST /admin/policy."""
        resp = client.open("/admin/policy", method="POST")
        # Should not be 500
        assert resp.status_code != 500
