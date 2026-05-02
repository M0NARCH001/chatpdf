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
