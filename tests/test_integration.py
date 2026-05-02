"""
Integration tests for the FastAPI backend.

Uses FastAPI's TestClient so no real server is started.
The conftest.py redirects all storage to temp directories.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def test_health_check():
    """Health endpoint is at /health (not /api/health)."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "docchat-api"}


def test_collections_list():
    response = client.get("/api/collections")
    assert response.status_code == 200
    assert "collections" in response.json()


# ---------------------------------------------------------------------------
# Session / auth endpoints
# ---------------------------------------------------------------------------

def test_register_new_user():
    """First login with a fresh name creates the account."""
    resp = client.post("/api/session/start", json={
        "display_name": "TestUser_Reg",
        "password": "secure123"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert data["display_name"] == "TestUser_Reg"


def test_login_returning_user():
    """Returning user with correct password gets the same session info."""
    # Register first
    client.post("/api/session/start", json={
        "display_name": "TestUser_Login",
        "password": "mypassword"
    })
    # Login again
    resp = client.post("/api/session/start", json={
        "display_name": "TestUser_Login",
        "password": "mypassword"
    })
    assert resp.status_code == 200
    assert resp.json()["display_name"] == "TestUser_Login"


def test_wrong_password_returns_401():
    """Wrong password on an existing account returns 401."""
    client.post("/api/session/start", json={
        "display_name": "TestUser_WrongPW",
        "password": "correctpassword"
    })
    resp = client.post("/api/session/start", json={
        "display_name": "TestUser_WrongPW",
        "password": "wrongpassword"
    })
    assert resp.status_code == 401


def test_short_password_returns_422():
    """Password shorter than 4 chars is rejected with 422."""
    resp = client.post("/api/session/start", json={
        "display_name": "TestUser_ShortPW",
        "password": "ab"
    })
    assert resp.status_code == 422


def test_short_name_returns_422():
    """Display name shorter than 2 chars is rejected with 422."""
    resp = client.post("/api/session/start", json={
        "display_name": "X",
        "password": "validpass"
    })
    assert resp.status_code == 422


def test_get_session_me():
    """session/me returns user info for a valid session."""
    reg = client.post("/api/session/start", json={
        "display_name": "TestUser_Me",
        "password": "pass1234"
    })
    session_id = reg.json()["session_id"]
    resp = client.get("/api/session/me", headers={"X-Session-ID": session_id})
    assert resp.status_code == 200
    assert resp.json()["display_name"] == "TestUser_Me"


def test_get_session_me_invalid():
    """session/me with a bogus session ID returns 404."""
    resp = client.get("/api/session/me", headers={"X-Session-ID": "nonexistent"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------

def test_upload_rejects_bad_extension(tmp_path):
    """Uploading an .exe should return 400."""
    bad_file = tmp_path / "malware.exe"
    bad_file.write_bytes(b"MZ\x90\x00")
    resp = client.post(
        "/api/upload",
        files={"files": ("malware.exe", bad_file.read_bytes(), "application/octet-stream")},
    )
    assert resp.status_code == 400
