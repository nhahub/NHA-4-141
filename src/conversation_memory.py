from collections import defaultdict
from typing import Dict, List, Optional


class ConversationMemory:
    """
    Keeps a short rolling window of (question, answer) turns per session.

    This gives the RAG pipeline just enough short-term memory to handle
    natural follow-up questions ("what about the second one?", "و بعدين؟")
    without needing a full chat-history database. Memory is stored
    in-process, keyed by the same session_id already used everywhere else
    in the app for document isolation, so each user's conversation stays
    separate automatically.

    Note: this is intentionally simple in-memory storage (not persisted to
    Qdrant or disk) — it resets if the process restarts, which is fine for
    a single-process Streamlit app.
    """

    def __init__(self, max_turns: int = 6):
        """
        Args:
            max_turns: Maximum number of (question, answer) turns to keep
                       per session. Older turns are dropped first.
        """
        self.max_turns = max_turns
        self._sessions: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        """Record a completed question/answer turn for a session."""
        turns = self._sessions[session_id]
        turns.append({"question": question, "answer": answer})
        # Trim from the front so we only ever keep the most recent max_turns
        if len(turns) > self.max_turns:
            del turns[: len(turns) - self.max_turns]

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Return the stored turns for a session, oldest first."""
        return list(self._sessions.get(session_id, []))

    def clear(self, session_id: str) -> None:
        """Drop all stored turns for a session (e.g. on 'Clear All')."""
        self._sessions.pop(session_id, None)

    def format_history(self, session_id: str, max_turns: Optional[int] = None) -> str:
        """
        Render the session's history as a simple "User: ... / Assistant: ..."
        transcript, suitable for dropping into an LLM prompt.

        Args:
            session_id: Session to format history for
            max_turns: Optionally limit to only the most recent N turns
                       (useful to keep prompts short even if more history
                       is stored)
        """
        turns = self.get_history(session_id)
        if max_turns is not None:
            turns = turns[-max_turns:]

        lines = []
        for turn in turns:
            lines.append(f"User: {turn['question']}")
            lines.append(f"Assistant: {turn['answer']}")
        return "\n".join(lines)