import os
from typing import List

from langchain_core.documents import Document


class Reranker:
    """
    Cross-encoder reranker that re-scores an initial shortlist of
    vector-similarity candidates by looking at the (query, chunk) pair
    jointly, instead of just comparing two pre-computed embedding vectors.

    Why this helps: cosine similarity between independently-computed
    embeddings is a decent but fairly blunt relevance signal — a
    cross-encoder that reads the query and the chunk together tends to be
    noticeably more accurate at judging "does this chunk actually answer
    this question", at the cost of being too slow to run over the whole
    collection. The standard pattern (used here) is:

        1. Vector search over-fetches a larger shortlist (e.g. top 20)
        2. The cross-encoder re-scores only that shortlist
        3. Only the top-k after reranking are kept for the final context

    Uses a multilingual cross-encoder (mMARCO) by default so it works
    reasonably for both Arabic and English content, since this app is used
    in both languages.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv(
            "RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
        )
        self._model = None
        self._load_failed = False

    def _get_model(self):
        """Lazily load the cross-encoder model on first use, so app startup
        stays fast and Streamlit sessions that never query don't pay for it."""
        if self._model is None and not self._load_failed:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except Exception as e:
                print(
                    f"Reranker: could not load '{self.model_name}' ({e}); "
                    "falling back to plain vector-search ranking."
                )
                self._load_failed = True
        return self._model

    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """
        Re-score and reorder a shortlist of candidate documents for a query.

        Args:
            query: The user's question (ideally the rewritten/standalone
                   version, so the reranker sees the same clear question
                   that was used for retrieval)
            documents: Candidate documents from an initial vector search
                       (should already be a shortlist, not the full corpus)
            top_k: How many of the best-scoring documents to return

        Returns:
            Up to top_k documents, sorted best-first, each with a
            "rerank_score" added to its metadata. Falls back to simply
            truncating the input list (preserving vector-search order) if
            the reranker model can't be loaded or scoring fails.
        """
        if not documents:
            return []

        model = self._get_model()
        if model is None:
            return documents[:top_k]

        try:
            pairs = [[query, doc.page_content] for doc in documents]
            scores = model.predict(pairs)

            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda pair: pair[1], reverse=True)

            reranked = []
            for doc, score in scored_docs[:top_k]:
                doc.metadata["rerank_score"] = float(score)
                reranked.append(doc)
            return reranked

        except Exception as e:
            print(f"Reranker: scoring failed ({e}); falling back to plain vector-search ranking.")
            return documents[:top_k]