import logging
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

_EMBEDDINGS_CACHE = None


def get_embeddings() -> Embeddings:
    """
    Returns the embeddings model: all-MiniLM-L6-v2 (local sentence-transformers),
    as described in the README architecture. Using a single fixed model keeps
    every collection at the same vector dimension and needs no API key.
    """
    global _EMBEDDINGS_CACHE
    if _EMBEDDINGS_CACHE is not None:
        return _EMBEDDINGS_CACHE

    logger.info("Loading HuggingFace Embeddings (all-MiniLM-L6-v2).")
    _EMBEDDINGS_CACHE = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return _EMBEDDINGS_CACHE
