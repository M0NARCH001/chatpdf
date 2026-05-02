"""
Unified entry point for HuggingFace Spaces and local development.

FastAPI starts as a background daemon thread on port 8000.
Gradio launches on port 7860 (the port HuggingFace Spaces exposes).
A health-poll loop waits for the backend before starting the UI.
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

# Ensure project root is on the path (needed when HF runs `python app.py`)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _run_backend():
    from app.main import app  # import here so dotenv loads first
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


def _wait_for_backend(url: str, timeout: int = 60) -> bool:
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


def main():
    from dotenv import load_dotenv
    load_dotenv()

    logger.info("Starting FastAPI backend on port 8000 (background thread)…")
    backend_thread = threading.Thread(target=_run_backend, daemon=True)
    backend_thread.start()

    health_url = "http://localhost:8000/health"
    logger.info("Waiting for backend to be ready…")
    if not _wait_for_backend(health_url):
        logger.error("Backend did not become ready within 60 s. Aborting.")
        sys.exit(1)
    logger.info("Backend ready. Launching Gradio frontend on port 7860…")

    from frontend.gradio_app import demo
    import gradio as gr
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
