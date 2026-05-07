---
title: DocChat AI
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: mit
short_description: Collaborative RAG chatbot — chat with PDFs in shared groups
---

# DocChat AI: Full-Stack RAG Chatbot

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-1.0-green)
![Gradio](https://img.shields.io/badge/Gradio-5.0-orange)
![LangChain](https://img.shields.io/badge/LangChain-latest-yellow)
![ChromaDB](https://img.shields.io/badge/ChromaDB-latest-blueviolet)

DocChat AI is a full-stack Retrieval-Augmented Generation (RAG) Chatbot that allows users to upload PDF/text documents, generates embeddings using `sentence-transformers`, and provides a conversational interface to query the documents. Uses a Hybrid Search (BM25 + Semantic Search) and grounds answers securely in the uploaded text with source tracking.

## 🏗️ Architecture

```text
Upload ──> Parse (PDF/TXT) ──> Chunk (512 tokens) ──> Embed (MiniLM-L6-v2) ──> Vector DB (ChromaDB)
                                                                                  │
User Query ──> Hybrid Search (BM25 + ChromaDB) ───────────────────────────────────┘
                    │
            Rerank & Compress
                    │
        Prompt Context Injection ──> LLM (GPT-4o or Ollama) ──> Formatted Answer with Exact Citations
```

### Lessons Learned & Design Choices
- **Chunking Strategy**: A fixed chunk size of 512 tokens with 64 tokens overlap balances the context window size with retrieval precision.
- **Hybrid Search vs. Dense Only**: We discovered that relying purely on semantic cosine similarity (ChromaDB) sometimes misses exact keyword matching (especially for proper nouns or acronyms). Adding BM25 significantly improves retrieval quality.
- **Context Compression**: By utilizing Langchain's Contextual Compression with `LLMChainExtractor`, we filter irrelevant sentences out of dense chunks, resulting in sharper focus and preventing LLM hallucination.

## ✨ Features
- **Multi-Document Support**: Upload multiple PDFs and texts.
- **Hybrid Search**: Uses BM25 for keyword search and ChromaDB for semantic search.
- **Dual LLM Support**: Use OpenAI (GPT-4o) or local models via Ollama.
- **Source Citations**: Every answer points exactly to the page and exact extracted text chunk.
- **Evaluation Dashboard**: Built-in Ragas metrics (Faithfulness, Relevancy, Precision, Recall).

## 🚀 Setup Instructions

### 1. Local Setup
1. Clone the repository and navigate to the project directory.
2. Create a virtual environment and attach it to your python path.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill in your keys (e.g., `OPENAI_API_KEY`).
5. Run the backend:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
6. Run the frontend (in a new terminal):
   ```bash
   python frontend/gradio_app.py
   ```

### 2. Docker Setup
1. Ensure Docker is installed.
2. Run `docker-compose up --build`
3. The app is accessible at `localhost:7860`.

## 🎥 Demo
*(Demo GIF coming soon...)*
