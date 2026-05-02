"""
Shared pytest fixtures for the DocChat AI test suite.

Sets up an in-memory SQLite database for each test session so tests never
touch the real ./data/rag_app.db, and patches ChromaDB to use a temp
directory so vector-store tests stay isolated and fast.
"""
import os
import pytest
import tempfile


# ---------------------------------------------------------------------------
# Point the database to an in-memory SQLite instance before anything imports
# ---------------------------------------------------------------------------
os.environ.setdefault("LLM_PROVIDER", "ollama")        # avoid needing OpenAI key
os.environ.setdefault("OPENAI_API_KEY", "test-key")    # prevent ValueError on import
os.environ.setdefault("OLLAMA_MODEL", "llama3")


@pytest.fixture(scope="session", autouse=True)
def use_temp_db(tmp_path_factory):
    """Redirect SQLite to a temp file for the test run."""
    db_dir = tmp_path_factory.mktemp("data")
    db_path = str(db_dir / "test_rag.db")
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"

    # Patch the engine before any model import
    import app.db.database as db_module
    from sqlmodel import create_engine
    db_module.engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    db_module.init_db()
    yield


@pytest.fixture(scope="session", autouse=True)
def use_temp_chroma(tmp_path_factory):
    """Redirect ChromaDB to a temp directory."""
    chroma_dir = tmp_path_factory.mktemp("chroma")
    os.environ["CHROMA_PERSIST_DIR"] = str(chroma_dir)
    yield
