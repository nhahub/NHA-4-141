import os
import uuid
from typing import List, Dict, Any, Optional
import chromadb
from langchain_core.documents import Document

class ChromaVectorStore:

    """
    Chroma vector store:
    Runs 100% in memory, no server needed, no connection errors.
    """

    def __init__(self, collection_name: str = "rag_collection"):
        # Initialize an in-memory Chroma client
        self.client = chromadb.Client()
        self.collection_name = collection_name
        
        # Create or get the collection (using L2 distance by default)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "l2"} 
        )
        print("Chroma vector store initialized ✅")

    def _sanitize_metadata(self, metadata: dict) -> dict:
        """
        Helper method to ensure both keys and values are Chroma-compatible.
        Chroma requires Keys to be strings, and Values to be str, int, float, or bool.
        """
        if not metadata:
            return {}
            
        sanitized = {}
        for key, value in metadata.items():
            # Chroma keys MUST be strings
            safe_key = str(key) 
            
            if value is None:
                continue # Skip None values completely
            elif isinstance(value, (str, int, float, bool)):
                sanitized[safe_key] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                sanitized[safe_key] = ", ".join(map(str, value))
            else:
                # Convert any other weird Python object (like dicts) to string
                sanitized[safe_key] = str(value)
                
        return sanitized

    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> bool:
        try:
            valid_docs = []
            valid_embeddings = []
            metadatas = [] 
            ids = []

            for doc, emb in zip(documents, embeddings):
                if not emb or len(emb) == 0:
                    continue
                
                valid_docs.append(doc.page_content)
                valid_embeddings.append(emb)
                
                # Sanitize before adding
                cleaned_metadata = self._sanitize_metadata(doc.metadata)
                metadatas.append(cleaned_metadata)
                
                ids.append(str(uuid.uuid4()))

            if not valid_docs:
                print("No valid documents to add")
                return False

            self.collection.add(
                documents=valid_docs,
                embeddings=valid_embeddings,
                metadatas=metadatas,
                ids=ids
            )

            print(f"✅ Added {len(valid_docs)} documents to Chroma. Total: {self.collection.count()}")
            return True

        except Exception as e:
            print(f"Error adding documents to Chroma: {e}")
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

            if self.collection.count() == 0:
                print("Chroma collection is empty — no documents loaded yet")
                return []

            # Build where clause for metadata filtering
            where_clause = filter_dict if filter_dict else None

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self.collection.count()),
                where=where_clause,
                include=["documents", "metadatas"]
            )

            documents = []
            if results and results.get("documents") and len(results["documents"]) > 0:
                for i in range(len(results["documents"][0])):
                    content = results["documents"][0][i]
                    metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                    documents.append(Document(page_content=content, metadata=metadata))

            return documents

        except Exception as e:
            print(f"Error in Chroma similarity search: {e}")
            return []

    def get_documents_by_session(self, session_id: str) -> List[dict]:
        try:
            results = self.collection.get(
                where={"session_id": session_id},
                include=["documents", "metadatas"]
            )
            
            if not results or not results.get("documents"):
                return []

            formatted_results = []
            for i in range(len(results["ids"])):
                metadata = results["metadatas"][i]
                formatted_results.append({
                    "id": results["ids"][i],
                    "content": results["documents"][i],
                    "source_type": metadata.get("source_type"),
                    "source_name": metadata.get("source_name"),
                    "chunk_id": metadata.get("chunk_id"),
                    "page": metadata.get("page")
                })
            return formatted_results

        except Exception as e:
            print(f"Error getting documents by session: {e}")
            return []

    def delete_by_session(self, session_id: str) -> bool:
        try:
            self.collection.delete(
                where={"session_id": session_id}
            )
            print(f"Deleted session {session_id} from Chroma. Remaining docs: {self.collection.count()}")
            return True

        except Exception as e:
            print(f"Error deleting session from Chroma: {e}")
            return False

    def get_collection_info(self) -> dict:
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "vector_size": "Dynamic", 
                "vectors_count": count,
                "points_count": count
            }
        except Exception:
             return {
                "name": self.collection_name,
                "vector_size": "Unknown",
                "vectors_count": 0,
                "points_count": 0
            }

# Keep old name as alias
QdrantVectorStore = ChromaVectorStore