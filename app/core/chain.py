"""
RAG chain built with LCEL (LangChain Expression Language).

Returns a RAGChain object whose .invoke({"question": ...}) method matches
the interface expected by routes.py and group_manager.py.
"""
import logging
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from app.core.retriever import get_hybrid_retriever
from app.core.llm import get_llm
from app.core.memory import get_session_history

logger = logging.getLogger(__name__)

_SYSTEM_TEMPLATE = (
    "You are a helpful and friendly assistant for a collaborative study group. "
    "When the user greets you (e.g. 'hi', 'hello', 'hey'), respond warmly. "
    "For factual questions, answer based on the provided document context and "
    "cite the source document name and page number. "
    'If a factual answer is not in the context, say "I couldn\'t find this in '
    'the uploaded documents." Never make up factual information, but feel free '
    "to engage in friendly conversation.\n\n"
    "Context:\n{context}"
)

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_TEMPLATE),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


class RAGChain:
    """
    Wraps LCEL retrieval + generation into a single .invoke() call.
    Maintains per-session chat history internally.
    """

    def __init__(self, collection_name: str, session_id: str) -> None:
        self._llm = get_llm(streaming=False)
        # Pass the LLM so the retriever can apply contextual compression.
        self._retriever = get_hybrid_retriever(collection_name, llm=self._llm)
        self._session_id = session_id
        self._output_parser = StrOutputParser()

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs.get("question", inputs.get("input", ""))
        history = get_session_history(self._session_id)

        # Retrieve relevant documents
        try:
            docs = self._retriever.invoke(question)
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            docs = []

        context = _format_docs(docs)

        # Build and invoke the prompt chain
        try:
            messages = _PROMPT.format_messages(
                input=question,
                context=context,
                chat_history=history.messages,
            )
            response = self._llm.invoke(messages)
            answer = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )
        except Exception as exc:
            logger.error("LLM invocation failed: %s", exc)
            answer = "I encountered an error generating a response. Please try again."

        # Persist to history
        history.add_user_message(question)
        history.add_ai_message(answer)

        return {"answer": answer, "source_documents": docs}


def get_rag_chain(collection_name: str, session_id: str = "default") -> RAGChain:
    """Construct the conversational RAG chain for a collection + session."""
    return RAGChain(collection_name=collection_name, session_id=session_id)
