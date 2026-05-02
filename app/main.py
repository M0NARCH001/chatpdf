from dotenv import load_dotenv
load_dotenv()

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api.routes import router as api_router
from app.core.embeddings import get_embeddings
from app.core.vectorstore import get_chroma_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize heavy resources on startup; clean up on shutdown."""
    from app.db.database import init_db
    logger.info("Application starting up — initializing models.")
    init_db()
    get_embeddings()
    get_chroma_client()
    logger.info("Initialization complete.")
    yield
    logger.info("Application shutting down.")


app = FastAPI(
    title="DocChat AI Backend",
    description="FastAPI backend for full-stack RAG chatbot.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: allow the Gradio frontend origin; wildcard + credentials is invalid per spec.
_allowed_origins = os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:7860,http://127.0.0.1:7860"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = round(time.time() - start_time, 4)
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url, exc)
    return JSONResponse(
        status_code=500,
        content={
            "error_code": 500,
            "message": "Internal Server Error",
            "detail": str(exc),
        },
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "docchat-api"}


app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
