"""
eval/metrics.py

Metric functions for evaluating the RAG assistant. Two families:

1. Retrieval metrics — did the vector search pull back the right chunks?
   - precision_at_k, recall_at_k, mean_reciprocal_rank, hit_rate

2. Generation metrics — is the LLM's answer any good, given what was retrieved?
   - faithfulness   (LLM-as-judge: is the answer grounded in the retrieved context?)
   - answer_relevance (embedding cosine similarity between question and answer)
   - answer_correctness (embedding cosine similarity between answer and a
     hand-written ground-truth answer, when you have one)

No new dependencies: only numpy (already in requirements.txt) and requests.
"""

import os
import re
import requests
import numpy as np
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def _is_relevant(source_name: str, expected_sources: List[str]) -> bool:
    """A retrieved chunk counts as relevant if its source_name contains
    (or is contained by) any of the expected source identifiers."""
    if not expected_sources:
        return False
    source_name = source_name.lower()
    return any(exp.lower() in source_name or source_name in exp.lower() for exp in expected_sources)


def precision_at_k(retrieved_sources: List[str], expected_sources: List[str]) -> float:
    """Of the k chunks retrieved, what fraction were actually relevant?"""
    if not retrieved_sources:
        return 0.0
    relevant = sum(1 for s in retrieved_sources if _is_relevant(s, expected_sources))
    return relevant / len(retrieved_sources)


def recall_at_k(retrieved_sources: List[str], expected_sources: List[str]) -> float:
    """Of the distinct expected sources, what fraction showed up somewhere
    in the retrieved set? (Simplified: treats each expected source as one
    'relevant item' to find, since we don't track relevant chunk-level IDs.)"""
    if not expected_sources:
        return 1.0 if not retrieved_sources else 0.0  # nothing to find, nothing wrongly found
    found = sum(
        1 for exp in expected_sources
        if any(_is_relevant(s, [exp]) for s in retrieved_sources)
    )
    return found / len(expected_sources)


def mean_reciprocal_rank(retrieved_sources: List[str], expected_sources: List[str]) -> float:
    """1 / rank of the first relevant chunk. 0 if none relevant."""
    for i, s in enumerate(retrieved_sources):
        if _is_relevant(s, expected_sources):
            return 1.0 / (i + 1)
    return 0.0


def hit_rate(retrieved_sources: List[str], expected_sources: List[str]) -> float:
    """1 if at least one retrieved chunk was relevant, else 0."""
    return 1.0 if any(_is_relevant(s, expected_sources) for s in retrieved_sources) else 0.0


# ---------------------------------------------------------------------------
# Embedding-based generation metrics
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    a, b = np.array(vec_a), np.array(vec_b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def answer_relevance(question: str, answer: str, embedding_client) -> float:
    """Does the answer actually address the question, semantically?
    Uses your existing EmbeddingClient (nomic-embed-text) — no extra model needed."""
    q_emb = embedding_client.embed_query(question)
    a_emb = embedding_client.embed_query(answer)
    return cosine_similarity(q_emb, a_emb)


def answer_correctness(answer: str, ground_truth: str, embedding_client) -> Optional[float]:
    """Similarity between the generated answer and a hand-written correct answer.
    Returns None if no ground_truth was supplied for this case."""
    if not ground_truth:
        return None
    a_emb = embedding_client.embed_query(answer)
    g_emb = embedding_client.embed_query(ground_truth)
    return cosine_similarity(a_emb, g_emb)


# ---------------------------------------------------------------------------
# LLM-as-judge: faithfulness / groundedness
# ---------------------------------------------------------------------------

FAITHFULNESS_PROMPT = """You are a strict evaluator. You will be given a CONTEXT and an ANSWER \
that was generated using that context. Your job is to check whether every factual claim in the \
ANSWER is actually supported by the CONTEXT.

CONTEXT:
{context}

ANSWER:
{answer}

Respond with ONLY a single number from 0 to 10, where:
0 = the answer contains claims with no support in the context at all (hallucinated)
10 = every claim in the answer is fully supported by the context

Respond with just the number, nothing else."""


def faithfulness(context: str, answer: str, ollama_base_url: Optional[str] = None,
                  judge_model: Optional[str] = None) -> Optional[float]:
    """Ask the LLM itself to score whether `answer` is grounded in `context`.
    Returns a 0.0-1.0 score, or None if the judge call fails (e.g. Ollama not running).

    Uses a plain call to Ollama's /api/generate rather than llm_client.generate_response(),
    because that method builds a RAG-answering prompt, not a judging prompt.
    """
    base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = judge_model or os.getenv("OLLAMA_MODEL", "mistral")

    prompt = FAITHFULNESS_PROMPT.format(context=context[:4000], answer=answer[:2000])

    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "max_tokens": 10},
            },
            timeout=60,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        match = re.search(r"\d+(\.\d+)?", text)
        if not match:
            return None
        score = float(match.group())
        return max(0.0, min(score, 10.0)) / 10.0
    except Exception as e:
        print(f"[faithfulness] judge call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------

def summarize(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Average every numeric metric across all test cases, ignoring None values."""
    keys = set()
    for r in results:
        keys.update(k for k, v in r.items() if isinstance(v, (int, float)))

    summary = {}
    for k in keys:
        values = [r[k] for r in results if isinstance(r.get(k), (int, float))]
        summary[k] = round(sum(values) / len(values), 4) if values else None
    return summary