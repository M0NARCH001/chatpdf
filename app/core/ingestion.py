import os
import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib

logger = logging.getLogger(__name__)

def get_file_hash(file_path: str) -> str:
    """Return a hash for a file, to deduplicate."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def load_and_split_documents(
    file_paths: List[str], 
    chunk_size: int = int(os.environ.get("CHUNK_SIZE", 512)),
    chunk_overlap: int = int(os.environ.get("CHUNK_OVERLAP", 64))
) -> List[Document]:
    """
    Loads documents from the given file paths and splits them into chunks.
    Attaches metadata: {source_file, page_number, chunk_index, total_chunks, file_hash}
    """
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        
        # Select appropriate loader
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.txt':
            loader = TextLoader(file_path)
        elif ext in ['.doc', '.docx']:
            loader = Docx2txtLoader(file_path)
        else:
            logger.warning("Unsupported file type: %s. Skipping %s", ext, file_path)
            continue
            
        try:
            docs = loader.load()
            
            # Calculate file hash for deduplication
            file_hash = get_file_hash(file_path)
            
            # Split the loaded document
            chunks = text_splitter.split_documents(docs)
            
            # Enhance metadata
            total_chunks = len(chunks)
            for idx, chunk in enumerate(chunks):
                # Ensure source_file and page_number exist
                chunk.metadata['source_file'] = os.path.basename(file_path)
                chunk.metadata['page_number'] = chunk.metadata.get('page', 1)
                chunk.metadata['chunk_index'] = idx
                chunk.metadata['total_chunks'] = total_chunks
                chunk.metadata['file_hash'] = file_hash
                
                all_chunks.append(chunk)
                
        except Exception as e:
            logger.error("Error processing %s: %s", file_path, e)
            
    return all_chunks
