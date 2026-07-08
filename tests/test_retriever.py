import pytest
from unittest.mock import patch, MagicMock
from app.core.retriever import get_hybrid_retriever
from langchain_core.documents import Document

@patch('app.core.retriever.get_vectorstore')
def test_retriever_initialization(mock_get_vectorstore):
    # Mock the Chroma vectorstore
    mock_vs = MagicMock()
    mock_get_vectorstore.return_value = mock_vs
    
    # Mock data inside vectorstore to avoid empty BM25 error
    mock_vs.get.return_value = {
        "documents": ["Test document content"],
        "metadatas": [{"source": "test"}],
        "ids": ["id1"]
    }
    
    retriever = get_hybrid_retriever("test_collection")
    assert retriever is not None
    # Our custom hybrid retriever (BM25 + semantic ensemble)
    assert hasattr(retriever, "invoke"), "Retriever must implement .invoke()"


def _doc(text):
    return Document(page_content=text)


def test_rrf_fusion_dedups_and_ranks():
    """A doc returned by both retrievers should rank above a single-source doc."""
    from app.core.retriever import _HybridRetriever

    shared, only_a, only_b = _doc("shared chunk"), _doc("alpha only"), _doc("beta only")
    bm25 = MagicMock();     bm25.invoke.return_value = [shared, only_a]
    semantic = MagicMock(); semantic.invoke.return_value = [shared, only_b]

    r = _HybridRetriever(bm25=bm25, semantic=semantic, top_k=5)
    out = r.invoke("q")
    assert out[0] is shared, "Doc found by both retrievers should be ranked first"
    assert len(out) == 3


def test_compression_empty_falls_back_to_fused():
    """If the extractor filters everything out, we keep the raw fused chunks."""
    from app.core.retriever import _HybridRetriever

    d = _doc("some context")
    bm25 = MagicMock(); bm25.invoke.return_value = [d]
    semantic = MagicMock(); semantic.invoke.return_value = []
    compressor = MagicMock(); compressor.compress_documents.return_value = []

    r = _HybridRetriever(bm25=bm25, semantic=semantic, top_k=5, compressor=compressor)
    assert r.invoke("q") == [d]
