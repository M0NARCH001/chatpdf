import os
from sqlmodel import SQLModel, create_engine, Session

# Ensure the data directory exists
os.makedirs("./data", exist_ok=True)

DATABASE_URL = "sqlite:///./data/rag_app.db"

# The connect_args={"check_same_thread": False} is needed for FastAPI and SQLite
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

def init_db():
    # Import models here so SQLModel metadata registry knows about them
    from app.models.session import AnonymousUser
    from app.models.group import Group, GroupMember, GroupDocument
    from app.models.chat_log import ChatLog
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
