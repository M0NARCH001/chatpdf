"""
RAGAS evaluation module.

Samples real text chunks from a ChromaDB collection, generates question/answer
pairs via the RAG chain, then scores them with RAGAS metrics. This gives
meaningful, document-specific scores instead of hardcoded generic Q&A.
"""
import os
import json
import logging
import random
from typing import List, Dict, Any
from datetime import datetime

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.chain import get_rag_chain
from app.core.llm import get_llm
from app.core.embeddings import get_embeddings
from app.core.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)

EVAL_HISTORY_FILE = os.path.join("data", "eval_history.json")

# Fallback generic prompts used only when the collection has too few chunks
_FALLBACK_PROMPTS = [
    "What is the main topic discussed in this document?",
    "Summarise the key findings or conclusions.",
    "What problem does the document address?",
    "List the most important points made by the author.",
]


def load_eval_history() -> List[Dict]:
    if os.path.exists(EVAL_HISTORY_FILE):
        try:
            with open(EVAL_HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_eval_run(run_data: Dict):
    os.makedirs(os.path.dirname(EVAL_HISTORY_FILE), exist_ok=True)
    history = load_eval_history()
    history.append(run_data)
    with open(EVAL_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def _sample_questions_from_collection(collection_name: str, n: int = 3) -> List[str]:
    """
    Pull random chunks from the collection and turn them into questions by
    asking the LLM to generate one factual question per chunk.
    Falls back to generic prompts if the collection is empty.
    """
    vs = get_vectorstore(collection_name)
    data = vs.get()
    docs = data.get("documents", [])

    if not docs:
        logger.warning("Collection '%s' is empty; using fallback prompts.", collection_name)
        return _FALLBACK_PROMPTS[:n]

    # Pick up to n distinct chunks
    sample = random.sample(docs, min(n, len(docs)))

    llm = get_llm(streaming=False)
    questions = []
    for chunk in sample:
        prompt = (
            "Read the following text and write ONE short factual question whose "
            "answer is clearly contained in the text. Return only the question, "
            "nothing else.\n\nText:\n" + chunk[:800]
        )
        try:
            resp = llm.invoke(prompt)
            q = resp.content.strip().strip('"').strip("'")
            if q:
                questions.append(q)
        except Exception as e:
            logger.warning("Question generation failed for a chunk: %s", e)

    if not questions:
        logger.warning("No questions generated; using fallback prompts.")
        return _FALLBACK_PROMPTS[:n]

    return questions


async def run_evaluation(
    collection_id: str, num_questions: int = 3
) -> Dict[str, Any]:
    """
    Run RAGAS evaluation on a real collection.

    1. Sample text chunks → generate factual questions via LLM.
    2. Run RAG chain to get answers + retrieved contexts.
    3. Score with RAGAS (faithfulness, answer_relevancy, context_precision,
       context_recall). Ground-truth is the same chunk text for recall.
    """
    logger.info("Starting RAGAS evaluation for '%s'.", collection_id)

    questions = _sample_questions_from_collection(collection_id, num_questions)
    chain = get_rag_chain(collection_id, session_id=f"eval_{collection_id}")

    answers: List[str] = []
    contexts: List[List[str]] = []
    ground_truths: List[str] = []

    for q in questions:
        try:
            res = chain.invoke({"question": q})
            answers.append(res.get("answer", ""))
            ctx_docs = res.get("source_documents", [])
            ctx_texts = [d.page_content for d in ctx_docs]
            contexts.append(ctx_texts if ctx_texts else [""])
            # Use the first retrieved chunk as a proxy ground-truth
            ground_truths.append(ctx_texts[0] if ctx_texts else "")
        except Exception as e:
            logger.warning("Chain failed for question '%s': %s", q, e)
            answers.append("")
            contexts.append([""])
            ground_truths.append("")

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(data)

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    use_openai = bool(openai_key and openai_key != "your_openai_api_key_here")

    ragas_llm = ChatOpenAI(model="gpt-4o-mini") if use_openai else get_llm(streaming=False)
    ragas_embeddings = OpenAIEmbeddings() if use_openai else get_embeddings()

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        scores: Dict[str, float] = dict(result)
        scores["overall"] = round(
            sum(v for v in scores.values() if isinstance(v, float)) / len(scores), 4
        )

        run_record = {
            "timestamp": datetime.now().isoformat(),
            "collection_id": collection_id,
            "provider": os.environ.get("LLM_PROVIDER", "openai"),
            "num_questions": len(questions),
            "scores": scores,
        }
        save_eval_run(run_record)
        logger.info("RAGAS evaluation complete: %s", scores)
        return scores

    except Exception as e:
        logger.error("RAGAS evaluation failed: %s", e)
        raise
