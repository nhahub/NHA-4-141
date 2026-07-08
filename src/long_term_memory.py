import json
import math
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

# NOTE: This module is NOT an implementation of the EVAF paper (Han 2026,
# "Memory Depth, Not Memory Access"). EVAF is a research mechanism that
# writes a LoRA adapter online, gated by surprise x valence, to change a
# model's *parameters*. There is no parameter update anywhere in this file.
#
# What's borrowed is only the paper's high-level engineering idea: a
# long-running assistant benefits from *selectively* deciding which user
# statements are worth carrying forward, rather than either remembering
# nothing or remembering everything. Here that's implemented as a small,
# fully inspectable SQLite table plus a lightweight importance scorer
# (src/memory_scorer.py) -- no adapters, no gradients, no replay buffers.
# See the "Long-term Memory" section of README.md for the full scope note.

VALID_MEMORY_TYPES = {"personal_fact", "preference", "project", "goal", "skill", "custom"}


class LongTermMemory:
    """
    SQLite-backed persistent store for durable facts about the user
    (name, preferences, ongoing projects, goals, skills, etc.), separate
    from ConversationMemory (which only holds recent chat turns).

    Schema (one row per memory):
        id               INTEGER PRIMARY KEY
        session_id       TEXT     -- which session this memory belongs to
        memory_type      TEXT     -- personal_fact/preference/project/goal/skill/custom
        content          TEXT     -- the remembered fact, in the user's own words
        importance_score REAL     -- 0.0-1.0, decays slowly if unused (see decay_memories)
        created_at       TEXT     -- ISO timestamp
        last_used        TEXT     -- ISO timestamp, bumped on retrieval or update
        usage_count      INTEGER
        embedding        TEXT     -- JSON-encoded vector, used only internally for
                                     semantic-similarity ranking/matching (an
                                     implementation detail, not a spec-required
                                     user-facing field)

    Memories are scoped by session_id, consistent with how every other
    piece of state in this app (documents, conversation memory) is already
    isolated per session. If you want facts to survive a "Clear All" /
    new-session click, point MEMORY_DB sessions at a stable user id instead
    of the per-chat session_id.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.getenv("MEMORY_DB", "memory.sqlite")
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance_score REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    last_used TEXT NOT NULL,
                    usage_count INTEGER NOT NULL DEFAULT 0,
                    embedding TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id)")
            conn.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ------------------------------------------------------------------
    # Part 4: Memory Update (avoid duplicates)
    # ------------------------------------------------------------------

    def add_or_update_memory(
        self,
        session_id: str,
        memory_type: str,
        content: str,
        importance_score: float,
        embedding: Optional[List[float]] = None,
        update_similarity_threshold: float = 0.84,
    ) -> Tuple[int, bool]:
        """
        Store a new memory, or update an existing similar one instead of
        creating a duplicate.

        A candidate counts as "the same memory, updated" when it shares
        the same memory_type in the same session and its embedding's
        cosine similarity to an existing memory clears
        update_similarity_threshold (default 0.84) -- e.g. "I use Python."
        followed later by "I mainly use Python and Rust." updates the
        existing row's content rather than adding a second one.

        Returns:
            (memory_id, was_update) -- was_update is True if an existing
            row was updated instead of a new one being inserted
        """
        memory_type = memory_type if memory_type in VALID_MEMORY_TYPES else "custom"
        now = self._now()
        embedding_json = json.dumps(embedding) if embedding is not None else None

        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            existing_rows = conn.execute(
                "SELECT id, embedding FROM memories WHERE session_id = ? AND memory_type = ?",
                (session_id, memory_type),
            ).fetchall()

            best_match_id = None
            best_similarity = 0.0
            if embedding is not None:
                for row in existing_rows:
                    if not row["embedding"]:
                        continue
                    try:
                        existing_embedding = json.loads(row["embedding"])
                    except Exception:
                        continue
                    similarity = self._cosine_similarity(embedding, existing_embedding)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match_id = row["id"]

            if best_match_id is not None and best_similarity >= update_similarity_threshold:
                conn.execute(
                    """UPDATE memories
                       SET content = ?, importance_score = MAX(importance_score, ?),
                           last_used = ?, usage_count = usage_count + 1, embedding = ?
                       WHERE id = ?""",
                    (content, importance_score, now, embedding_json, best_match_id),
                )
                conn.commit()
                return best_match_id, True

            cursor = conn.execute(
                """INSERT INTO memories
                   (session_id, memory_type, content, importance_score, created_at, last_used, usage_count, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, 1, ?)""",
                (session_id, memory_type, content, importance_score, now, now, embedding_json),
            )
            conn.commit()
            return cursor.lastrowid, False

    # ------------------------------------------------------------------
    # Part 3: Memory Retrieval
    # ------------------------------------------------------------------

    def retrieve_relevant(
        self,
        session_id: str,
        query_embedding: Optional[List[float]] = None,
        max_results: int = 3,
    ) -> List[Dict]:
        """
        Retrieve the most relevant stored memories for a session, ranked by
        a blend of semantic similarity, importance score, recency, and
        usage count -- not just similarity alone, so an important-but-
        older memory can still outrank a barely-related fresh one.

        Retrieving a memory counts as "using" it: usage_count and
        last_used are bumped for whatever is returned, which both matches
        the ranking formula's intent and feeds Part 5's decay logic.

        Returns up to max_results memory dicts, best match first. Returns
        [] on any internal error or if nothing is stored yet.
        """
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM memories WHERE session_id = ?", (session_id,)).fetchall()

        if not rows:
            return []

        now = datetime.now(timezone.utc)
        scored = []
        for row in rows:
            similarity = 0.0
            if query_embedding is not None and row["embedding"]:
                try:
                    similarity = self._cosine_similarity(query_embedding, json.loads(row["embedding"]))
                except Exception:
                    similarity = 0.0

            try:
                last_used_dt = datetime.fromisoformat(row["last_used"])
                days_since_used = max((now - last_used_dt).total_seconds() / 86400.0, 0.0)
            except Exception:
                days_since_used = 999.0
            recency_score = 1.0 / (1.0 + days_since_used)

            usage_score = min(row["usage_count"] / 10.0, 1.0)

            composite = (
                0.5 * similarity
                + 0.25 * row["importance_score"]
                + 0.15 * recency_score
                + 0.10 * usage_score
            )
            scored.append((composite, row))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = scored[:max_results]

        if top:
            now_str = self._now()
            with self._lock, self._connect() as conn:
                for _, row in top:
                    conn.execute(
                        "UPDATE memories SET usage_count = usage_count + 1, last_used = ? WHERE id = ?",
                        (now_str, row["id"]),
                    )
                conn.commit()

        return [
            {
                "id": row["id"],
                "memory_type": row["memory_type"],
                "content": row["content"],
                "importance_score": row["importance_score"],
                "score": composite,
            }
            for composite, row in top
        ]

    # ------------------------------------------------------------------
    # Part 5: Memory Expiration (decay, never delete)
    # ------------------------------------------------------------------

    def decay_memories(self, session_id: str, days_threshold: int = 14, decay_rate: float = 0.97) -> int:
        """
        Gradually reduce the importance of memories that are both stale
        (unused for days_threshold+ days) and rarely used (usage_count <= 2).
        Never deletes anything -- only lowers importance_score, so stale
        memories naturally rank lower in future retrieval without
        disappearing outright.

        Returns the number of memories decayed (0 on any internal error).
        """
        try:
            with self._lock, self._connect() as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT id, importance_score, last_used, usage_count FROM memories WHERE session_id = ?",
                    (session_id,),
                ).fetchall()

                if not rows:
                    return 0

                now = datetime.now(timezone.utc)
                decayed = 0
                for row in rows:
                    try:
                        last_used_dt = datetime.fromisoformat(row["last_used"])
                        days_since_used = (now - last_used_dt).total_seconds() / 86400.0
                    except Exception:
                        continue

                    if days_since_used >= days_threshold and row["usage_count"] <= 2:
                        new_score = max(row["importance_score"] * decay_rate, 0.05)
                        conn.execute(
                            "UPDATE memories SET importance_score = ? WHERE id = ?",
                            (new_score, row["id"]),
                        )
                        decayed += 1

                conn.commit()
                return decayed
        except Exception as e:
            print(f"Memory decay skipped due to an error: {e}")
            return 0

    # ------------------------------------------------------------------
    # Utility / admin
    # ------------------------------------------------------------------

    def get_all_memories(self, session_id: str) -> List[Dict]:
        """Return every stored memory for a session, most important first."""
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, memory_type, content, importance_score, created_at, last_used, usage_count "
                "FROM memories WHERE session_id = ? ORDER BY importance_score DESC",
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def clear_session(self, session_id: str) -> bool:
        """Delete all memories for a session. Only call this on an explicit
        user action (e.g. a dedicated 'forget everything' control) --
        normal session resets should NOT wipe long-term memory."""
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM memories WHERE session_id = ?", (session_id,))
            conn.commit()
        return True