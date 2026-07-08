import os
from typing import List, Dict, Any, Optional
import faiss
import numpy as np
from langchain_core.documents import Document
import uuid
import pickle


class FAISSVectorStore:
    """
    FAISS vector store — replaces Qdrant completely.
    Runs 100% in memory, no server needed, no connection errors.
    """

    def __init__(self):
        self.vector_size = 768
        self.index = faiss.IndexFlatL2(self.vector_size)
        # Store metadata separately — FAISS only stores vectors
        self.documents: List[Document] = []
        self.ids: List[str] = []
        print("FAISS vector store initialized ✅")

    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> bool:
        try:
            valid_docs = []
            valid_embeddings = []

            for doc, emb in zip(documents, embeddings):
                if not emb or len(emb) == 0:
                    continue
                if len(emb) != self.vector_size:
                    print(f"Skipping embedding with wrong size: {len(emb)} != {self.vector_size}")
                    continue
                valid_docs.append(doc)
                valid_embeddings.append(emb)

            if not valid_docs:
                print("No valid documents to add")
                return False

            vectors = np.array(valid_embeddings, dtype=np.float32)
            self.index.add(vectors)

            for doc in valid_docs:
                self.documents.append(doc)
                self.ids.append(str(uuid.uuid4()))

            print(f"✅ Added {len(valid_docs)} documents to FAISS. Total: {len(self.documents)}")
            return True

        except Exception as e:
            print(f"Error adding documents to FAISS: {e}")
            return False

    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        try:
            if not query_embedding or len(query_embedding) == 0:
                return []

            if self.index.ntotal == 0:
                print("FAISS index is empty — no documents loaded yet")
                return []

            query_vector = np.array([query_embedding], dtype=np.float32)

            # Search more candidates if filtering is needed
            search_k = min(self.index.ntotal, k * 10 if filter_dict else k)
            distances, indices = self.index.search(query_vector, search_k)

            results = []
            for idx in indices[0]:
                if idx == -1 or idx >= len(self.documents):
                    continue

                doc = self.documents[idx]

                # Apply metadata filters if provided
                if filter_dict:
                    match = all(
                        doc.metadata.get(key) == value
                        for key, value in filter_dict.items()
                    )
                    if not match:
                        continue

                results.append(doc)
                if len(results) >= k:
                    break

            return results

        except Exception as e:
            print(f"Error in FAISS similarity search: {e}")
            return []

    def get_documents_by_session(self, session_id: str) -> List[dict]:
        try:
            return [
                {
                    "id": self.ids[i],
                    "content": doc.page_content,
                    "source_type": doc.metadata.get("source_type"),
                    "source_name": doc.metadata.get("source_name"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "page": doc.metadata.get("page")
                }
                for i, doc in enumerate(self.documents)
                if doc.metadata.get("session_id") == session_id
            ]
        except Exception as e:
            print(f"Error getting documents by session: {e}")
            return []

    def delete_by_session(self, session_id: str) -> bool:
        """Remove all documents for a session and rebuild the FAISS index."""
        try:
            keep_docs = []
            keep_ids = []
            keep_embeddings = []

            # We need to rebuild — collect docs to keep
            for i, doc in enumerate(self.documents):
                if doc.metadata.get("session_id") != session_id:
                    keep_docs.append(doc)
                    keep_ids.append(self.ids[i])

            # Rebuild index without deleted docs
            self.index = faiss.IndexFlatL2(self.vector_size)
            self.documents = keep_docs
            self.ids = keep_ids

            print(f"Deleted session {session_id} — {len(self.documents)} docs remain")
            return True

        except Exception as e:
            print(f"Error deleting session from FAISS: {e}")
            return False

    def get_collection_info(self) -> dict:
        return {
            "name": "faiss_index",
            "vector_size": self.vector_size,
            "vectors_count": self.index.ntotal,
            "points_count": len(self.documents)
        }


# Keep old name as alias so rag_pipeline.py doesn't need changes
QdrantVectorStore = FAISSVectorStore