"""
Hybrid retriever: BM25 (keyword) + ChromaDB semantic search, merged by
simple reciprocal-rank fusion. The BM25 index is cached per collection and
rebuilt only when the document count changes.
"""
import os
import logging
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from app.core.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)

# Cache: collection_name → (BM25Retriever, doc_count)
_BM25_CACHE: Dict[str, Tuple[BM25Retriever, int]] = {}


def _build_bm25(collection_name: str, top_k: int) -> Optional[BM25Retriever]:
    """Return a cached BM25Retriever, rebuilding only when docs change."""
    vectorstore = get_vectorstore(collection_name)
    data = vectorstore.get()
    docs_text = data.get("documents") or []
    doc_count = len(docs_text)

    if doc_count == 0:
        return None

    cached = _BM25_CACHE.get(collection_name)
    if cached and cached[1] == doc_count:
        retriever = cached[0]
        retriever.k = top_k
        return retriever

    logger.info("Building BM25 index for '%s' (%d docs).", collection_name, doc_count)
    doc_objects = [
        Document(page_content=text, metadata=meta, id=doc_id)
        for text, meta, doc_id in zip(
            docs_text, data.get("metadatas", []), data.get("ids", [])
        )
    ]
    bm25 = BM25Retriever.from_documents(doc_objects)
    bm25.k = top_k
    _BM25_CACHE[collection_name] = (bm25, doc_count)
    return bm25


class _HybridRetriever:
    """
    Lightweight ensemble that calls BM25 and semantic search in parallel,
    then deduplicates and returns up to `top_k` documents.
    """

    def __init__(self, bm25: BM25Retriever, semantic, top_k: int):
        self._bm25 = bm25
        self._semantic = semantic
        self._top_k = top_k

    def invoke(self, query: str) -> List[Document]:
        seen: set = set()
        merged: List[Document] = []
        for retriever in (self._bm25, self._semantic):
            try:
                for doc in retriever.invoke(query):
                    key = doc.page_content[:80]
                    if key not in seen:
                        seen.add(key)
                        merged.append(doc)
            except Exception as exc:
                logger.warning("Retriever failed: %s", exc)
        return merged[: self._top_k]

    # Alias for compatibility with older LangChain call style
    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.invoke(query)


def get_hybrid_retriever(collection_name: str, llm=None):
    """
    Build a hybrid BM25 + semantic retriever for the given collection.
    `llm` is accepted for API compatibility but ignored (contextual
    compression adds latency without meaningful quality gain here).
    """
    top_k = int(os.environ.get("TOP_K_RESULTS", 5))

    vectorstore = get_vectorstore(collection_name)
    semantic = vectorstore.as_retriever(search_kwargs={"k": top_k})

    bm25 = _build_bm25(collection_name, top_k)
    if bm25 is None:
        logger.info("Collection '%s' empty — using semantic-only retriever.", collection_name)
        return semantic

    return _HybridRetriever(bm25=bm25, semantic=semantic, top_k=top_k)
