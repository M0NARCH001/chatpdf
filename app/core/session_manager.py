import os
import time
import secrets
import bcrypt
from datetime import datetime, timedelta
from sqlmodel import Session, select
from app.models.session import AnonymousUser

# Session inactivity timeout — the session_id stops working after this long
# without use, forcing a password re-entry. Configurable via env.
SESSION_TTL_HOURS = int(os.environ.get("SESSION_TTL_HOURS", "168"))  # 7 days

# In-memory login throttle: display_name → recent failed-attempt timestamps.
# ponytail: per-process + per-name; fine for a single uvicorn worker. Swap for
# Redis/IP-based limiting if this ever runs multi-worker or needs anti-enumeration.
_MAX_FAILED = 5
_WINDOW_SECONDS = 300  # 5 minutes
_FAILED_ATTEMPTS: dict[str, list[float]] = {}


def _recent_failures(name: str) -> int:
    now = time.time()
    kept = [t for t in _FAILED_ATTEMPTS.get(name, []) if now - t < _WINDOW_SECONDS]
    if kept:
        _FAILED_ATTEMPTS[name] = kept
    else:
        _FAILED_ATTEMPTS.pop(name, None)
    return len(kept)


def hash_password(password: str) -> str:
    """bcrypt hash (salt is embedded in the returned string)."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a bcrypt hash."""
    if not stored_hash:
        return False
    return bcrypt.checkpw(password.encode(), stored_hash.encode())


def login_or_register(display_name: str, password: str, db: Session):
    """If name exists → verify password → return session.
       If name doesn't exist → create account → return session.
       Returns (user, error_message)."""
    if _recent_failures(display_name) >= _MAX_FAILED:
        return None, "Too many failed attempts. Please wait a few minutes and try again."

    stmt = select(AnonymousUser).where(AnonymousUser.display_name == display_name)
    existing = db.exec(stmt).first()

    if existing:
        # User exists — verify password
        if not verify_password(password, existing.password_hash):
            _FAILED_ATTEMPTS.setdefault(display_name, []).append(time.time())
            return None, "Wrong password. Please try again."
        _FAILED_ATTEMPTS.pop(display_name, None)  # clear throttle on success
        existing.last_seen = datetime.utcnow()
        db.add(existing)
        db.commit()
        db.refresh(existing)
        return existing, None
    else:
        # New user — create account
        user = AnonymousUser(
            session_id=f"sess_{secrets.token_hex(8)}",
            display_name=display_name,
            password_hash=hash_password(password),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user, None


def create_session(display_name: str, db: Session) -> AnonymousUser:
    """Create a new user session (legacy, used by login_or_register internally)."""
    session_id = f"sess_{secrets.token_hex(8)}"
    new_user = AnonymousUser(
        session_id=session_id,
        display_name=display_name,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


def get_session_by_id(session_id: str, db: Session) -> AnonymousUser | None:
    """Fetch an active session by ID and refresh last_seen.
    Returns None if the session has been idle longer than SESSION_TTL_HOURS."""
    if not session_id:
        return None
    user = db.get(AnonymousUser, session_id)
    if not user:
        return None
    if datetime.utcnow() - user.last_seen > timedelta(hours=SESSION_TTL_HOURS):
        return None  # expired — caller must re-authenticate with the password
    user.last_seen = datetime.utcnow()
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def update_display_name(session_id: str, new_name: str, db: Session) -> AnonymousUser | None:
    """Update a user's display name."""
    user = db.get(AnonymousUser, session_id)
    if user:
        user.display_name = new_name
        user.last_seen = datetime.utcnow()
        db.add(user)
        db.commit()
        db.refresh(user)
    return user
