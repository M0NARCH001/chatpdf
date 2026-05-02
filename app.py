"""
HuggingFace Spaces entry point.

HF Gradio SDK requires a module-level `demo` variable.
We start FastAPI as a background daemon thread first, poll until it is
ready, then expose the Gradio Blocks object so HF can serve it.
"""
import logging
import os
import sys
import threading
import time

import httpx
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()


def _run_backend():
    """Run FastAPI on port 8000 (internal — not exposed by HF)."""
    from app.main import app as fastapi_app
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_level="warning")


def _wait_for_backend(url: str, timeout: int = 90) -> bool:
    """Poll the health endpoint until the backend is up or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


# ── Start FastAPI backend immediately on module import ───────────────────────
logger.info("Starting FastAPI backend on port 8000 (background thread)…")
_backend_thread = threading.Thread(target=_run_backend, daemon=True)
_backend_thread.start()

logger.info("Waiting for backend to be ready…")
if not _wait_for_backend("http://localhost:8000/health"):
    logger.error("Backend did not become ready within 90 s.")
    # Don't sys.exit here — let Gradio still launch so HF shows something useful
else:
    logger.info("Backend is ready.")

# ── Import Gradio demo (module-level — required by HF Gradio SDK) ────────────
from frontend.gradio_app import demo  # noqa: E402  (import after backend starts)

# ── Local launch ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import gradio as gr
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
