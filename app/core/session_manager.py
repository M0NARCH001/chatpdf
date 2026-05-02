import secrets
import bcrypt
from datetime import datetime
from sqlmodel import Session, select
from app.models.session import AnonymousUser


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
    stmt = select(AnonymousUser).where(AnonymousUser.display_name == display_name)
    existing = db.exec(stmt).first()

    if existing:
        # User exists — verify password
        if not verify_password(password, existing.password_hash):
            return None, "Wrong password. Please try again."
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
    """Fetch an active session by ID and update last_seen."""
    if not session_id:
        return None
    user = db.get(AnonymousUser, session_id)
    if user:
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
