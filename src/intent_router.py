import os
import re
import requests

# Fast, zero-latency heuristic: if the question clearly contains
# programming-related keywords (English or Arabic), classify as "code"
# without needing a model call at all. Covers the large majority of cases.
_CODE_KEYWORDS = [
    # English
    r"\bcode\b", r"\bfunction\b", r"\bclass\b", r"\bbug\b", r"\berror\b",
    r"\bexception\b", r"\btraceback\b", r"\bdebug\b", r"\bpython\b",
    r"\bjavascript\b", r"\btypescript\b", r"\bjava\b", r"\bc\+\+\b", r"\bsql\b",
    r"\bapi\b", r"\balgorithm\b", r"\bregex\b", r"\bcompile\b", r"\bsyntax\b",
    r"\bscript\b", r"\bstack trace\b", r"\bnull pointer\b", r"\brefactor\b",
    r"```",  # code fences are an unambiguous signal
    # Arabic (Egyptian dialect terms commonly used for code questions)
    r"كود", r"دالة", r"فنكشن", r"سكريبت", r"اكتب لي كود", r"اعمل لي كود",
    r"error", r"باج", r"بق", r"اكسبشن", r"خطأ في الكود", r"اصلح الكود",
]
_CODE_PATTERN = re.compile("|".join(_CODE_KEYWORDS), re.IGNORECASE)

_CLASSIFY_PROMPT = """Classify the user's question into exactly one category: CODE or CHAT.
- CODE: programming, debugging, code review, algorithms, software errors, writing/explaining code.
- CHAT: anything else (general questions, document Q&A, summaries, conversation).

Respond with ONLY one word: CODE or CHAT.

Question: {question}
Category:"""


class IntentRouter:
    """
    Decides whether a text question should go to the code-specialist model
    or the general chat model. Image intent is NOT decided here — it's
    determined upstream by whether the user attached an image to the
    message, which is a much more reliable signal than text classification.

    Strategy:
      1. Fast keyword heuristic (no model call, instant, covers most cases)
      2. If the heuristic doesn't match, fall back to a one-word
         classification call to a small local model (cheap and local, so
         it doesn't cost anything or depend on OpenAI quota)
    """

    def __init__(self, classifier_model: str = None):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        # Reuse the main chat model for classification too — it's small
        # (qwen2.5:1.5b) and a one-word classification is an easy task for it.
        self.classifier_model = classifier_model or os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

    def classify(self, question: str) -> str:
        """Returns 'code' or 'chat'."""
        if _CODE_PATTERN.search(question):
            return "code"

        # Ambiguous case: ask the small local model directly, with a strict
        # one-word answer format. Fails safe to "chat" on any error.
        try:
            payload = {
                "model": self.classifier_model,
                "prompt": _CLASSIFY_PROMPT.format(question=question),
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 5},
            }
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=15)
            if response.status_code == 200:
                text = response.json().get("response", "").strip().upper()
                if "CODE" in text:
                    return "code"
            return "chat"
        except Exception:
            return "chat"