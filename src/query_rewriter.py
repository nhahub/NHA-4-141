import os
from typing import Dict, List

import requests

# Kept intentionally strict: only output the rewritten question, nothing
# else, so we can drop the result straight into the embedding/search step
# without any further parsing. Explicitly asked to preserve the original
# language since the app is used mostly in Egyptian Arabic.
_REWRITE_PROMPT = """Given the conversation history and a new follow-up question, rewrite the \
follow-up question so it can be understood on its own, without needing the \
conversation history. Resolve pronouns and implicit references (e.g. "it", \
"the second one", "ده", "اللي فات", "و بعدين") using the history. If the \
follow-up question is already a complete standalone question, just return it \
unchanged. Keep the same language as the follow-up question (Arabic stays \
Arabic, English stays English). Output ONLY the rewritten question, with no \
preamble, quotes, or explanation.

Conversation history:
{history}

Follow-up question: {question}

Standalone question:"""


class QueryRewriter:
    """
    Condenses a conversational follow-up question into a standalone question
    that carries enough context to be embedded and searched on its own.

    Example:
        History:  User: What is backward induction?
                  Assistant: It's a method for solving games by reasoning
                             backwards from the final move...
        Follow-up: "and how is it different from IESDS?"
        Rewritten: "How is backward induction different from IESDS?"

    Without this step, a follow-up like "and how is it different from
    IESDS?" would be embedded and searched almost meaninglessly on its own,
    since it doesn't mention what "it" refers to.

    Uses the small local Ollama chat model (cheap, local, no API quota) —
    the rewrite doesn't need a strong model, just enough language
    understanding to resolve references from the last couple of turns.
    """

    def __init__(self, model: str = None):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

    def rewrite(self, question: str, history: List[Dict[str, str]], max_history_turns: int = 3) -> str:
        """
        Args:
            question: The user's latest (possibly context-dependent) question
            history: List of {"question": ..., "answer": ...} turns, oldest first
            max_history_turns: Only the most recent N turns are used, to keep
                                the rewrite prompt short and fast

        Returns:
            A standalone version of the question. Falls back to the
            original question unchanged if there's no history, or if the
            rewrite call fails or returns something suspicious.
        """
        if not history:
            return question

        recent_history = history[-max_history_turns:]
        formatted_history = "\n".join(
            f"User: {turn['question']}\nAssistant: {turn['answer']}" for turn in recent_history
        )

        try:
            payload = {
                "model": self.model,
                "prompt": _REWRITE_PROMPT.format(history=formatted_history, question=question),
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 128},
            }
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=15)
            if response.status_code != 200:
                return question

            rewritten = response.json().get("response", "").strip().strip('"').strip()

            # Sanity checks: a real rewrite shouldn't be empty, and shouldn't
            # be wildly longer than the original (a sign the model rambled
            # instead of just rewriting the question).
            if not rewritten:
                return question
            if len(rewritten) > max(len(question) * 4, 200):
                return question

            return rewritten

        except Exception:
            # Fail safe: any error just means we search with the original
            # question instead of a rewritten one — never blocks the query.
            return question