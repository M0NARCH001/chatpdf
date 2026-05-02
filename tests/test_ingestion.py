import os
import pytest
from app.core.ingestion import load_and_split_documents

def test_load_and_split_txt(tmp_path):
    # Create a dummy text file
    test_file = tmp_path / "sample.txt"
    test_content = "Hello world. This is a testing document.\\n\\n" + ("Repeating text. " * 50)
    test_file.write_text(test_content)
    
    docs = load_and_split_documents([str(test_file)], chunk_size=50, chunk_overlap=10)
    
    assert len(docs) > 1, "Should split into multiple chunks based on chunk size"
    assert docs[0].metadata['source_file'] == "sample.txt"
    assert 'chunk_index' in docs[0].metadata
    assert 'total_chunks' in docs[0].metadata
    assert 'file_hash' in docs[0].metadata
