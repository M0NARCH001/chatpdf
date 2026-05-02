import os
import logging
import chromadb
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_chroma import Chroma
from app.core.embeddings import get_embeddings

logger = logging.getLogger(__name__)

def get_chroma_client() -> chromadb.PersistentClient:
    persist_directory = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
    return chromadb.PersistentClient(path=persist_directory)

def get_vectorstore(collection_name: str) -> Chroma:
    """Get or create a Chroma vectorstore for a specific collection."""
    embeddings = get_embeddings()
    client = get_chroma_client()
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

def add_documents(collection_name: str, documents: List[Document]) -> bool:
    """
    Add documents to the collection.
    Checks for file_hash deduplication before adding.
    Return True if new documents were added, False if everything was a duplicate.
    """
    if not documents:
        return False
        
    vectorstore = get_vectorstore(collection_name)
    
    # Check existing file hashes to avoid duplicates
    existing_data = vectorstore.get(include=['metadatas'])
    existing_hashes = set()
    if existing_data and existing_data.get('metadatas'):
        for meta in existing_data['metadatas']:
            if 'file_hash' in meta:
                existing_hashes.add(meta['file_hash'])
                
    # Filter out chunks that come from files we've already ingested
    new_docs = [doc for doc in documents if doc.metadata.get('file_hash') not in existing_hashes]
    
    if new_docs:
        vectorstore.add_documents(new_docs)
        return True
    
    return False

def list_collections() -> List[str]:
    """List all collections in the ChromaDB."""
    client = get_chroma_client()
    return [c.name for c in client.list_collections()]

def delete_collection(collection_name: str) -> None:
    """Delete a collection entirely."""
    client = get_chroma_client()
    try:
        client.delete_collection(name=collection_name)
    except Exception as e:
        logger.error("Error deleting collection '%s': %s", collection_name, e)
