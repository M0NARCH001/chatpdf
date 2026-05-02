from sqlmodel import SQLModel, Field
from datetime import datetime
from uuid import uuid4
import secrets

class Group(SQLModel, table=True):
    id: str = Field(
        default_factory=lambda: str(uuid4())[:8].upper(),
        primary_key=True
    )
    name: str
    join_code: str = Field(
        default_factory=lambda: secrets.token_hex(3).upper()
    )
    creator_session_id: str
    creator_display_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True


class GroupMember(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    group_id: str = Field(foreign_key="group.id", index=True)
    session_id: str
    display_name: str
    role: str = "member"
    joined_at: datetime = Field(default_factory=datetime.utcnow)


class GroupDocument(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    group_id: str = Field(foreign_key="group.id", index=True)
    session_id: str
    display_name: str
    filename: str
    chunk_count: int
    file_size_kb: float
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
