from sqlmodel import SQLModel, Field
from datetime import datetime
from uuid import uuid4

class ChatLog(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    session_id: str = Field(index=True)
    display_name: str
    group_id: str | None = Field(default=None, index=True)
    question: str
    answer: str
    sources_used: str = "" # JSON string of sources
    llm_used: str = "default"
    response_time_ms: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
