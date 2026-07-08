import json
import os
import re
from typing import Optional, Tuple

import requests

# Rule-based patterns for messages worth remembering long-term. Covers both
# English and Egyptian Arabic phrasing since the app is used bilingually.
# Each entry is (regex, memory_type, base_importance_score). Checked in
# order; the first match wins.
_MEMORY_RULES = [
    (r"\bmy name is\b|\bi'?m called\b|اسمي|انا اسمي", "personal_fact", 0.95),
    (r"\bmy supervisor\b|مشرفي|دكتور المشروع", "personal_fact", 0.88),
    (r"\bmy university\b|جامعتي|بادرس في", "personal_fact", 0.85),
    (r"\bi work as\b|\bi'?m a\b.*\b(developer|engineer|student|designer)\b|بشتغل|شغلي", "personal_fact", 0.88),
    (r"\bmy graduation project\b|مشروع (ال)?تخرج", "project", 0.95),
    (r"\bmy project\b|مشروعي", "project", 0.85),
    (r"\bmy goal\b|\bi want to\b|\bi'?m trying to\b|هدفي|بحاول ا", "goal", 0.85),
    (r"\bmy programming language\b|لغة البرمجة", "skill", 0.85),
    (r"\bremember that\b|افتكر ان|متنساش ان|سجل ان", "custom", 0.95),
    (r"\bi prefer\b|بفضل", "preference", 0.9),
    (r"\bmy favorite\b|المفضل عندي|بحب ال", "preference", 0.85),
    (r"\bi always\b|دايما", "preference", 0.85),
    (r"\bi am interested in\b|\bi'?m interested in\b|مهتم بـ|مهتم ب", "preference", 0.85),
]

# Trivial conversational filler that should never be scored/stored, even
# before reaching the rules above.
_TRIVIAL_EXACT = {
    "hi", "hello", "hey", "ok", "okay", "yes", "no", "cool", "nice", "thanks",
    "thank you", "thanks!", "np", "sure", "bye", "👍", "👌",
    "هلا", "اهلا", "أهلا", "شكرا", "شكراً", "تمام", "اوك", "أوك", "ماشي",
    "تسلم", "تمام كده", "ايوه", "أيوة", "لأ", "لا",
}


class MemoryImportanceScorer:
    """
    Lightweight hybrid scorer deciding whether a user message contains a
    durable fact worth writing to long-term memory (Part 2).

    Step 1 is fast, free, rule-based pattern matching for common
    self-disclosure phrasings ("my name is...", "I prefer...", etc.).
    Step 2 only runs when the rules are inconclusive: a single lightweight
    call to the local Qwen chat model classifies the message. This avoids
    paying an LLM call for every single message while still catching
    important statements that don't match a fixed pattern.
    """

    def __init__(self, model: str = None, threshold: float = None):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
        self.threshold = threshold if threshold is not None else float(os.getenv("MEMORY_THRESHOLD", "0.80"))

    def score(self, message: str) -> Tuple[float, Optional[str]]:
        """
        Args:
            message: The user's raw message text

        Returns:
            (importance_score, memory_type) -- memory_type is None when the
            message isn't worth storing (either scored below threshold or
            judged trivial). Callers should only persist a memory when
            memory_type is not None AND importance_score >= self.threshold.
        """
        text = (message or "").strip()
        if not text:
            return 0.0, None

        # Check rule-based patterns first: a short-but-meaningful phrase
        # like "اسمي احمد" (2 words) should still match "my name is..." even
        # though it would otherwise look "too short to matter" below.
        rule_match = self._rule_based_score(text)
        if rule_match is not None:
            return rule_match

        if self._is_trivial(text):
            return 0.0, None

        # Rules were inconclusive -- ask the local model. Any failure here
        # (Ollama down, bad JSON, etc.) fails safe to "don't store".
        return self._llm_score(text)

    @staticmethod
    def _is_trivial(text: str) -> bool:
        stripped = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE).strip().lower()
        if stripped in _TRIVIAL_EXACT:
            return True
        # Laughter in either language ("hahaha", "ههههه")
        if re.fullmatch(r"(ha){2,}h?", stripped) or re.fullmatch(r"(ه){3,}", stripped):
            return True
        # Very short messages with no punctuation-stripped content left
        if len(text.split()) <= 2:
            return True
        return False

    @staticmethod
    def _rule_based_score(text: str) -> Optional[Tuple[float, str]]:
        for pattern, memory_type, score in _MEMORY_RULES:
            if re.search(pattern, text, re.IGNORECASE):
                return score, memory_type
        return None

    def _llm_score(self, text: str) -> Tuple[float, Optional[str]]:
        prompt = f"""Decide whether this user message contains a durable personal fact, \
preference, ongoing project, goal, or skill worth remembering about the user long-term \
-- as opposed to a trivial, one-off conversational remark.

Message: "{text}"

Respond with ONLY a single-line JSON object, nothing else:
{{"should_store": true or false, "score": <float 0.0-1.0>, "memory_type": "personal_fact" or "preference" or "project" or "goal" or "skill" or "custom" or "none"}}

JSON:"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 80},
            }
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=15)
            if response.status_code != 200:
                return 0.0, None

            raw = response.json().get("response", "").strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return 0.0, None

            parsed = json.loads(match.group(0))
            should_store = bool(parsed.get("should_store", False))
            memory_type = parsed.get("memory_type", "none")
            score = float(parsed.get("score", 0.0))
            score = max(0.0, min(1.0, score))

            if not should_store or memory_type == "none" or memory_type not in {
                "personal_fact", "preference", "project", "goal", "skill", "custom"
            }:
                return score, None

            return score, memory_type

        except Exception:
            # Fail safe: any error here just means we skip storing this
            # message, never that we break the conversation.
            return 0.0, None