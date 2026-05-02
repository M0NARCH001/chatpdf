import os
import logging
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

_EMBEDDINGS_CACHE = None


def get_embeddings() -> Embeddings:
    """
    Returns the configured embeddings model.
    Defaults to all-MiniLM-L6-v2 (local), but uses OpenAI embeddings when
    LLM_PROVIDER=openai and a valid OPENAI_API_KEY is set.
    """
    global _EMBEDDINGS_CACHE
    if _EMBEDDINGS_CACHE is not None:
        return _EMBEDDINGS_CACHE

    provider = os.environ.get("LLM_PROVIDER", "openai").lower()
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    if provider == "openai" and openai_key and openai_key != "your_openai_api_key_here":
        try:
            logger.info("Loading OpenAI Embeddings (text-embedding-3-small).")
            _EMBEDDINGS_CACHE = OpenAIEmbeddings(model="text-embedding-3-small")
            return _EMBEDDINGS_CACHE
        except Exception as e:
            logger.warning("Failed to load OpenAI embeddings: %s. Falling back to HuggingFace.", e)

    logger.info("Loading HuggingFace Embeddings (all-MiniLM-L6-v2).")
    _EMBEDDINGS_CACHE = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return _EMBEDDINGS_CACHE
