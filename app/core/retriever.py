"""
Hybrid retriever matching the README architecture:

    Hybrid Search (BM25 + ChromaDB) -> Rerank (Reciprocal Rank Fusion)
    -> Contextual Compression (LLMChainExtractor) -> context

BM25 (keyword) and ChromaDB (semantic) results are merged with Reciprocal
Rank Fusion, then the top chunks are passed through LangChain's
LLMChainExtractor to strip sentences irrelevant to the query before they
reach the LLM. Compression is best-effort: any failure (or an extractor that
filters everything out) falls back to the raw fused chunks so the chain is
never starved of context.
"""
import os
import logging
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

from app.core.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)

# Cache: collection_name -> (BM25Retriever, doc_count)
_BM25_CACHE: Dict[str, Tuple[BM25Retriever, int]] = {}

# RRF constant; 60 is the value from the original RRF paper.
_RRF_K = 60


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
    BM25 + semantic retrieval fused by Reciprocal Rank Fusion, then optionally
    compressed with an LLMChainExtractor. Exposes .invoke() / .get_relevant_documents().
    """

    def __init__(self, bm25, semantic, top_k: int, compressor=None):
        self._bm25 = bm25
        self._semantic = semantic
        self._top_k = top_k
        self._compressor = compressor

    def _fuse(self, query: str) -> List[Document]:
        """Reciprocal Rank Fusion across BM25 and semantic results."""
        scores: Dict[str, float] = {}
        doc_by_key: Dict[str, Document] = {}
        for retriever in (self._bm25, self._semantic):
            if retriever is None:
                continue
            try:
                results = retriever.invoke(query)
            except Exception as exc:
                logger.warning("Retriever failed: %s", exc)
                continue
            for rank, doc in enumerate(results):
                key = doc.page_content[:80]
                doc_by_key.setdefault(key, doc)
                scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
        ranked = sorted(scores, key=scores.get, reverse=True)
        return [doc_by_key[k] for k in ranked]

    def invoke(self, query: str) -> List[Document]:
        fused = self._fuse(query)[: self._top_k]
        if self._compressor is None or not fused:
            return fused
        try:
            compressed = list(self._compressor.compress_documents(fused, query))
        except Exception as exc:
            logger.warning("Contextual compression failed: %s; using fused docs.", exc)
            return fused
        # Extractor emptied everything -> keep raw context rather than starve the LLM.
        return compressed or fused

    # Alias for compatibility with older LangChain call style
    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.invoke(query)


def get_hybrid_retriever(collection_name: str, llm=None):
    """
    Build a hybrid BM25 + semantic retriever for the given collection.

    When `llm` is provided and ENABLE_COMPRESSION is truthy (default), retrieved
    chunks are run through an LLMChainExtractor for contextual compression.
    """
    top_k = int(os.environ.get("TOP_K_RESULTS", 5))

    vectorstore = get_vectorstore(collection_name)
    semantic = vectorstore.as_retriever(search_kwargs={"k": top_k})
    bm25 = _build_bm25(collection_name, top_k)

    if bm25 is None:
        logger.info("Collection '%s' has no BM25 docs — semantic-only retrieval.", collection_name)

    compressor = None
    compression_on = os.environ.get("ENABLE_COMPRESSION", "true").lower() in ("1", "true", "yes")
    if llm is not None and compression_on:
        try:
            compressor = LLMChainExtractor.from_llm(llm)
        except Exception as exc:
            logger.warning("Could not initialise contextual compression: %s", exc)

    return _HybridRetriever(bm25=bm25, semantic=semantic, top_k=top_k, compressor=compressor)
