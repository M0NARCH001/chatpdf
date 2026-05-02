import os
import logging
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


def get_llm(streaming: bool = True) -> BaseChatModel:
    """
    Returns the configured LLM based on LLM_PROVIDER.
    Supports streaming and basic retry logic.
    """
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()

    if provider == "ollama":
        model_name = os.environ.get("OLLAMA_MODEL", "llama3")
        logger.info("Loading Ollama LLM (%s).", model_name)
        llm = ChatOllama(
            model=model_name,
            streaming=streaming,
            temperature=0.0
        )
    else:
        logger.info("Loading OpenAI LLM (gpt-4o-mini).")
        # Ensure API key exists
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key or api_key == "your_openai_api_key_here":
            raise ValueError("OPENAI_API_KEY is not set or is still the default value.")
            
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            streaming=streaming,
            temperature=0.0,
            max_retries=3,
            request_timeout=60
        )
        
    return llm
