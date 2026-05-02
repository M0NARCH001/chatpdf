from sqlmodel import SQLModel, Field
from datetime import datetime

class AnonymousUser(SQLModel, table=True):
    session_id: str = Field(primary_key=True)
    display_name: str = Field(index=True, unique=True)
    password_hash: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)
