import os
from typing import List, Tuple, Optional, Union, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from .vector_store import QdrantVectorStore
from .llm_router import LLMRouter
from .embeddings import EmbeddingClient
from .web_search import WebSearchClient
from .web_scraper import WebScraper
from .conversation_memory import ConversationMemory
from .query_rewriter import QueryRewriter
from .reranker import Reranker
from .long_term_memory import LongTermMemory
from .memory_scorer import MemoryImportanceScorer

class RAGPipeline:
    """
    Main RAG (Retrieval-Augmented Generation) pipeline orchestrating all components.
    
    This class coordinates the entire RAG workflow:
    1. Document processing and chunking
    2. Vector embedding generation
    3. Vector storage with session isolation
    4. Query rewriting using recent conversation history (so follow-up
       questions can be understood on their own)
    5. Similarity search (over-fetch) + cross-encoder reranking for
       higher-precision context
    6. LLM response generation with numbered, page-cited context and
       recent conversation history
    """
    
    def __init__(self):
        """Initialize all RAG pipeline components"""
        # Vector store for document embeddings with session-based filtering
        self.vector_store = QdrantVectorStore()
        
        # LLM router: tries OpenAI primary -> OpenAI secondary -> local Ollama
        # model in order (any tier not configured in .env is skipped)
        self.llm_client = LLMRouter()
        
        # Embedding client for converting text to vectors
        self.embedding_client = EmbeddingClient()
        
        # Live web search client (separate from WebScraper, which only
        # fetches a single URL the user explicitly provides)
        self.web_search_client = WebSearchClient(max_results=5)
        self.web_scraper = WebScraper()
        
        # Text splitter for breaking documents into manageable chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,      # Maximum characters per chunk
            chunk_overlap=200,    # Overlap between chunks to maintain context
            length_function=len,  # Function to measure chunk length
        )

        # Short-term conversational memory, keyed by session_id, so
        # follow-up questions get coherent context-aware answers
        self.memory = ConversationMemory(max_turns=6)

        # Condenses conversational follow-ups into standalone questions
        # before they're embedded and searched
        self.query_rewriter = QueryRewriter()

        # Cross-encoder reranker applied to the initial vector-search
        # shortlist, to improve retrieval precision before building context
        self.reranker = Reranker()

        # --- Long-term memory (EVAF-inspired selective memory; see the
        # "Long-term Memory" section of README.md for what this is and,
        # importantly, what it deliberately is NOT: no LoRA adapters, no
        # gradient updates, no online continual learning -- just a small,
        # selectively-written persistent store, separate from the
        # short-term ConversationMemory above.) ---
        self.ltm_enabled = os.getenv("ENABLE_LONG_TERM_MEMORY", "true").lower() == "true"
        self.memory_threshold = float(os.getenv("MEMORY_THRESHOLD", "0.80"))
        self.max_memory_results = int(os.getenv("MAX_MEMORY_RESULTS", "3"))
        self.long_term_memory = None
        self.memory_scorer = None
        if self.ltm_enabled:
            try:
                self.long_term_memory = LongTermMemory(db_path=os.getenv("MEMORY_DB", "memory.sqlite"))
                self.memory_scorer = MemoryImportanceScorer(threshold=self.memory_threshold)
            except Exception as e:
                # Fail-safe: if SQLite can't even be opened, the whole
                # feature quietly turns itself off rather than blocking
                # the rest of the app from working.
                print(f"Long-term memory disabled (could not initialize): {e}")
                self.ltm_enabled = False
    
    def add_documents(
        self, 
        documents: List[Union[str, Dict]], 
        source_type: str, 
        source_name: str, 
        session_id: str
    ) -> bool:
        """
        Add documents to the vector store with comprehensive metadata.
        
        Args:
            documents: List of chunks to add. Each item can either be a
                       plain string (kept for backward compatibility) or a
                       {"text": ..., "page": ...} dict as produced by
                       DocumentProcessor — the "page" label (PDF page,
                       PPTX slide, XLSX sheet, or None) enables page-level
                       citations later.
            source_type: Type of source ("document" or "web")
            source_name: Name/URL of the source
            session_id: Unique session identifier for isolation
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create Document objects with metadata for each chunk
            docs = []
            for i, item in enumerate(documents):
                if isinstance(item, dict):
                    doc_text = item.get("text", "")
                    page = item.get("page")
                else:
                    doc_text = item
                    page = None

                metadata = {
                    "source_type": source_type,    # document/web classification
                    "source_name": source_name,    # filename or URL
                    "session_id": session_id,      # session isolation
                    "chunk_id": i,                 # chunk index within document
                    "page": page                   # page/slide/sheet label, or None
                }
                docs.append(Document(page_content=doc_text, metadata=metadata))
            
            # Generate vector embeddings for all document chunks
            texts = [doc.page_content for doc in docs]
            embeddings = self.embedding_client.embed_documents(texts)
            
            # Store documents and embeddings in vector database
            return self.vector_store.add_documents(docs, embeddings)
            
        except Exception as e:
            print(f"Error adding documents to RAG pipeline: {e}")
            return False
    
    def query(
        self,
        question: str,
        session_id: str,
        k: int = 5,
        use_web_search: bool = False,
        model_mode: str = "auto",
        image_bytes: Optional[bytes] = None,
        use_reranker: bool = True
    ) -> Tuple[str, List[str], List[str]]:
        """
        Query the RAG system with session-based filtering, optionally
        augmented with live web search results and/or an attached image.

        Pipeline order:
            User Question
              -> Conversation Memory (recent chat turns, existing)
              -> Long-term Memory Retrieval (persistent user facts, new)
              -> Query Rewrite (condense follow-ups using chat history, existing)
              -> Retriever (vector search over documents)
              -> Cross-Encoder Reranker
              -> LLM (context = long-term memories + numbered doc chunks + web results)
              -> Store important memories from this turn
        
        Args:
            question: User's question
            session_id: Session ID for document filtering (also used as
                        the conversation/long-term memory key)
            k: Number of relevant documents to retrieve (post-reranking)
            use_web_search: If True, also search the live web and blend
                             those results into the context
            model_mode: Which LLM backend to use — "auto" (GPT first, Qwen
                        fallback), "openai" (GPT only), "local" (Qwen only),
                        or "smart" (auto-detects code vs chat intent)
            image_bytes: If provided, routes to the vision model
                         (qwen2.5-vl) regardless of model_mode, for image
                         analysis / OCR
            use_reranker: If True (default), over-fetch a larger vector-
                          search shortlist and re-score it with a
                          cross-encoder for higher-precision context
            
        Returns:
            Tuple of (response_text, source_list, memory_notes) —
            source_list entries include page/slide/sheet labels when
            available, e.g. "notes.pdf (p.3, document)"; memory_notes are
            short UI strings like '🧠 Learned: "..."' or
            '🧠 Using remembered information.' (empty list when long-term
            memory is disabled or nothing notable happened this turn)
        """
        memory_notes: List[str] = []
        try:
            # --- Long-term Memory Retrieval (before query rewrite, so it
            # reflects the user's literal question rather than a rewritten
            # one) ---
            ltm_context = ""
            if self.ltm_enabled and self.long_term_memory is not None:
                try:
                    raw_question_embedding = self.embedding_client.embed_query(question)
                    ltm_matches = self.long_term_memory.retrieve_relevant(
                        session_id, raw_question_embedding, max_results=self.max_memory_results
                    )
                    # Lazily decay stale, rarely-used memories on each turn
                    # (cheap: scoped to this session only)
                    self.long_term_memory.decay_memories(session_id)
                    if ltm_matches:
                        ltm_context = "\n".join(
                            f"- ({m['memory_type']}) {m['content']}" for m in ltm_matches
                        )
                        memory_notes.append("🧠 Using remembered information.")
                except Exception as e:
                    # Fail-safe: retrieval problems never block the answer,
                    # they just mean this turn proceeds without memories.
                    print(f"Long-term memory retrieval failed (continuing without it): {e}")
                    ltm_context = ""

            # Pull recent conversation turns for this session, and use them
            # to rewrite a context-dependent follow-up ("what about that?")
            # into a standalone question before embedding/searching.
            history = self.memory.get_history(session_id)
            search_question = self.query_rewriter.rewrite(question, history) if history else question

            # Generate embedding for the (possibly rewritten) question
            query_embedding = self.embedding_client.embed_query(search_question)
            
            # Over-fetch a larger shortlist when reranking so the
            # cross-encoder has real candidates to choose between; skip the
            # over-fetch if reranking is disabled.
            fetch_k = max(k * 4, 15) if use_reranker else k
            candidate_docs = self.vector_store.similarity_search(
                query_embedding, 
                k=fetch_k, 
                filter_dict={"session_id": session_id}  # Session-based filtering
            )

            # Re-score the shortlist with a cross-encoder for higher
            # precision, then keep only the top-k
            if use_reranker:
                relevant_docs = self.reranker.rerank(search_question, candidate_docs, top_k=k)
            else:
                relevant_docs = candidate_docs[:k]
            
            # Gather live web search context if requested
            web_context, web_sources = ("", [])
            if use_web_search:
                web_context, web_sources = self._get_web_context(search_question)
            
            # Handle case where there's no context at all from any source
            # (an attached image is its own "context", so skip this check then)
            if not relevant_docs and not web_context and not ltm_context and image_bytes is None:
                if use_web_search:
                    return "I couldn't find relevant information in your documents or on the web for this question.", [], memory_notes
                return "I don't have any relevant information to answer your question. Please upload some documents first, or enable Web Search.", [], memory_notes
            
            # Prepare context from retrieved documents as numbered, citeable
            # chunks — e.g. "[1] (notes.pdf, p.3, document):\n...text..." —
            # so the LLM can cite which chunk backs each claim, and so we
            # can report accurate, page-aware sources regardless of whether
            # the model actually uses the citation markers.
            context_parts = []
            citation_labels = []
            if ltm_context:
                context_parts.append(f"--- What we remember about you ---\n{ltm_context}")
            if relevant_docs:
                numbered_chunks = []
                for idx, doc in enumerate(relevant_docs, start=1):
                    label = self._format_citation_label(doc)
                    citation_labels.append(label)
                    numbered_chunks.append(f"[{idx}] ({label}):\n{doc.page_content}")
                context_parts.append("\n\n".join(numbered_chunks))
            if web_context:
                context_parts.append(f"--- Live Web Search Results ---\n{web_context}")
            
            context = "\n\n".join(context_parts)

            # Format recent conversation history for the LLM prompt (kept
            # short — only the last few turns — to avoid bloating the
            # context window on top of the retrieved chunks)
            chat_history_str = self.memory.format_history(session_id, max_turns=3)
            
            # Generate response using LLM with combined context
            response = self.llm_client.generate_response(
                question, context, mode=model_mode, image_bytes=image_bytes, chat_history=chat_history_str
            )
            
            # Extract unique sources (already page-aware) from retrieved documents
            sources = []
            for label in citation_labels:
                if label not in sources:
                    sources.append(label)
            
            # Append web sources, avoiding duplicates
            for source_info in web_sources:
                if source_info not in sources:
                    sources.append(source_info)

            # Remember this turn so future follow-up questions in this
            # session can be understood in context
            self.memory.add_turn(session_id, question, response)

            # --- Store important memories from this turn (Part 6/2/4) ---
            if self.ltm_enabled and self.memory_scorer is not None and self.long_term_memory is not None:
                try:
                    importance_score, memory_type = self.memory_scorer.score(question)
                    if memory_type and importance_score >= self.memory_threshold:
                        content_embedding = self.embedding_client.embed_query(question)
                        _, was_update = self.long_term_memory.add_or_update_memory(
                            session_id, memory_type, question.strip(), importance_score,
                            embedding=content_embedding
                        )
                        label = "Updated" if was_update else "Learned"
                        memory_notes.append(f'🧠 {label}: "{question.strip()}"')
                except Exception as e:
                    # Fail-safe: scoring/storage problems never break the
                    # reply that was already generated -- they just mean
                    # this turn isn't remembered.
                    print(f"Long-term memory storage skipped due to an error: {e}")
            
            return response, sources, memory_notes
            
        except Exception as e:
            print(f"Error querying RAG system: {e}")
            return f"An error occurred while processing your question: {str(e)}", [], memory_notes

    def _format_citation_label(self, doc: Document) -> str:
        """
        Build a human-readable, page-aware source label for a retrieved
        chunk, e.g. "lecture.pdf (p.3, document)" or "notes.docx (document)"
        when no page-like unit is available for that format.
        """
        name = doc.metadata.get("source_name", "Unknown")
        source_type = doc.metadata.get("source_type", "unknown")
        page = doc.metadata.get("page")
        if page is not None:
            return f"{name} (p.{page}, {source_type})"
        return f"{name} ({source_type})"
    
    def _get_web_context(self, question: str, top_n_to_scrape: int = 2) -> Tuple[str, List[str]]:
        """
        Run a live web search for the question and build a context string.
        
        For the top few results, attempts to scrape the full page content
        for richer context; falls back to the search snippet if scraping
        fails or is skipped for lower-ranked results.
        
        Args:
            question: The user's question, used as the search query
            top_n_to_scrape: Number of top results to fully scrape
            
        Returns:
            Tuple of (context_string, list_of_source_labels)
        """
        context, sources = "", []
        try:
            results = self.web_search_client.search(question)
            if not results:
                return "", []
            
            context_chunks = []
            for i, result in enumerate(results):
                content = None
                if i < top_n_to_scrape and result.get("url"):
                    # Try to get full page content for the top results
                    content = self.web_scraper.scrape_url(result["url"])
                
                # Fall back to the search snippet if scraping failed/skipped
                if not content:
                    content = result.get("snippet", "")
                
                if content:
                    context_chunks.append(f"Source: {result['url']}\n{content}")
                    sources.append(f"{result['url']} (web search)")
            
            context = "\n\n".join(context_chunks)
            
        except Exception as e:
            print(f"Error building web search context: {e}")
        
        return context, sources
    
    def get_session_documents(self, session_id: str) -> List[dict]:
        """
        Get all documents for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of document dictionaries with metadata
        """
        try:
            return self.vector_store.get_documents_by_session(session_id)
        except Exception as e:
            print(f"Error retrieving session documents: {e}")
            return []
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear all documents for a specific session.
        
        Args:
            session_id: Session identifier to clear
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.memory.clear(session_id)
            return self.vector_store.delete_by_session(session_id)
        except Exception as e:
            print(f"Error clearing session: {e}")
            return False