import streamlit as st
import os
import re
from dotenv import load_dotenv
from src.rag_pipeline import RAGPipeline
from src.document_processor import DocumentProcessor
from src.web_scraper import WebScraper
import uuid
import tempfile

# Load environment variables from .env file
load_dotenv()

# Page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean, minimal styling with improved UX
st.markdown("""
<style>
    .main { padding-top: 2rem; }
    .stApp > header { background-color: transparent; }
    .element-container:has(> .stMarkdown > div[data-testid="stMarkdownContainer"] > p:empty) { display: none; }
    .stMarkdown > div[data-testid="stMarkdownContainer"]:empty { display: none; }
    .upload-box {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        background-color: #fafafa;
    }
    .source-tag {
        background-color: #fff3e0;
        color: #f57c00;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    .status-indicator {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    .status-ready { background-color: #e8f5e8; color: #2e7d32; }
    .status-empty { background-color: #fff3e0; color: #f57c00; }
    .status-warning { background-color: #ffebee; color: #c62828; }
    .stButton > button {
        background-color: #1976d2;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover { background-color: #1565c0; border: none; }
    .stButton > button:focus {
        background-color: #1565c0;
        border: none;
        box-shadow: 0 0 0 2px rgba(25, 118, 210, 0.3);
    }
    .loading-container {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 1rem;
        background-color: #f0f8ff;
        border-radius: 8px;
        border-left: 4px solid #1976d2;
        margin: 1rem 0;
    }
    .loading-spinner {
        width: 20px;
        height: 20px;
        border: 2px solid #e3f2fd;
        border-top: 2px solid #1976d2;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    .loading-text { color: #1976d2; font-weight: 500; }
    .stFileUploader > label { display: none; }

    /* Code block label */
    .code-label {
        background-color: #1e1e1e;
        color: #9cdcfe;
        padding: 4px 12px;
        border-radius: 6px 6px 0 0;
        font-size: 0.75rem;
        font-family: monospace;
        display: inline-block;
        margin-bottom: -4px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Feature: Render response with code highlight
# ─────────────────────────────────────────────
def render_response(response: str):
    """Render assistant response — plain text gets st.markdown,
    code blocks get st.code with syntax highlighting."""
    # Split on fenced code blocks  ```lang\n...\n```
    parts = re.split(r'(```[\w]*\n[\s\S]*?```)', response)
    for part in parts:
        if part.startswith('```'):
            lines = part.split('\n')
            # First line is ```python / ```js / ``` etc.
            lang = lines[0].replace('```', '').strip() or 'python'
            # Everything between first and last line is code
            code = '\n'.join(lines[1:]).rstrip('`').strip()
            if lang:
                st.markdown(f'<div class="code-label">🖥️ {lang}</div>', unsafe_allow_html=True)
            st.code(code, language=lang)
        else:
            if part.strip():
                st.markdown(part)


# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
def initialize_session_state():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'documents_count' not in st.session_state:
        st.session_state.documents_count = 0
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    if 'processed_urls' not in st.session_state:
        st.session_state.processed_urls = set()
    if 'current_url' not in st.session_state:
        st.session_state.current_url = ""
    if 'file_uploader_key' not in st.session_state:
        st.session_state.file_uploader_key = 0
    if 'chat_input_key' not in st.session_state:
        st.session_state.chat_input_key = 0


def check_system_status():
    try:
        from src.llm_client import OllamaClient
        return OllamaClient().check_connection()
    except Exception as e:
        print(f"Error checking system status: {e}")
        return False


def show_loading_indicator(message: str):
    st.markdown(f"""
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <div class="loading-text">{message}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# File processing
# ─────────────────────────────────────────────
def process_files(uploaded_files):
    if not uploaded_files:
        return
    processor = DocumentProcessor()
    total_chunks = 0

    new_files = []
    for file in uploaded_files:
        file_key = f"{file.name}_{file.size}"
        if file_key not in st.session_state.processed_files:
            new_files.append(file)
            st.session_state.processed_files.add(file_key)

    if not new_files:
        return

    loading_placeholder = st.empty()
    with loading_placeholder.container():
        show_loading_indicator("Processing uploaded files and creating chunks...")

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, uploaded_file in enumerate(new_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        try:
            chunks = processor.process_document(tmp_file_path, uploaded_file.name)
            success = st.session_state.rag_pipeline.add_documents(
                chunks,
                source_type="document",
                source_name=uploaded_file.name,
                session_id=st.session_state.session_id
            )
            if success:
                total_chunks += len(chunks)
            else:
                st.warning(f"⚠️ Issues processing {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        finally:
            os.unlink(tmp_file_path)
        progress_bar.progress((i + 1) / len(new_files))

    loading_placeholder.empty()
    status_text.empty()
    progress_bar.empty()
    st.session_state.documents_count += len(new_files)
    if total_chunks > 0:
        st.success(f"✅ Processed {len(new_files)} files ({total_chunks} chunks)")


# ─────────────────────────────────────────────
# URL processing
# ─────────────────────────────────────────────
def process_url(url):
    if url in st.session_state.processed_urls:
        st.info("This URL has already been processed.")
        return
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        show_loading_indicator("Scraping web content and creating chunks...")
    try:
        scraper = WebScraper()
        content = scraper.scrape_url(url)
        if content:
            processor = DocumentProcessor()
            chunks = processor.process_text(content, url)
            success = st.session_state.rag_pipeline.add_documents(
                chunks,
                source_type="web",
                source_name=url,
                session_id=st.session_state.session_id
            )
            loading_placeholder.empty()
            if success:
                st.session_state.documents_count += 1
                st.session_state.processed_urls.add(url)
                st.success(f"✅ Processed content from URL ({len(chunks)} chunks)")
                st.session_state.current_url = ""
                st.rerun()
            else:
                st.warning("⚠️ Issues processing URL content")
        else:
            loading_placeholder.empty()
            st.error("❌ Failed to scrape content from the URL")
    except Exception as e:
        loading_placeholder.empty()
        st.error(f"❌ Error: {str(e)}")


# ─────────────────────────────────────────────
# Chat input handler
# ─────────────────────────────────────────────
def handle_chat_input(user_question):
    from src.llm_client import OllamaClient
    client = OllamaClient()
    is_code = client.is_code_request(user_question)

    # Code requests don't need documents
    if not is_code and st.session_state.documents_count == 0:
        st.warning("⚠️ Please add some documents or web content first!")
        return

    st.session_state.chat_history.append({'role': 'user', 'content': user_question})

    with st.spinner("Thinking..."):
        try:
            if is_code:
                # Skip RAG — go directly to coder model with no context
                response = client.generate_response(user_question, context="")
                sources = []
            else:
                # Normal RAG flow
                response, sources = st.session_state.rag_pipeline.query(
                    user_question,
                    session_id=st.session_state.session_id
                )

            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response,
                'sources': sources
            })

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ─────────────────────────────────────────────
# Clear all data
# ─────────────────────────────────────────────
def clear_all_data():
    st.session_state.chat_history = []
    st.session_state.documents_count = 0
    st.session_state.processed_files = set()
    st.session_state.processed_urls = set()
    st.session_state.current_url = ""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.file_uploader_key += 1
    st.session_state.chat_input_key += 1
    try:
        st.session_state.rag_pipeline.clear_session(st.session_state.session_id)
    except Exception as e:
        print(f"Error clearing session data: {e}")


# ─────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────
def main():
    initialize_session_state()

    st.title("🤖 RAG Assistant")
    st.markdown("Upload documents or add web content, then chat with your data")

    ollama_running = check_system_status()

    if not ollama_running:
        st.markdown("""
        <div class="status-indicator status-warning">
            ⚠️ Ollama not detected — please run <code>ollama serve</code>
        </div>
        """, unsafe_allow_html=True)
    elif st.session_state.documents_count > 0:
        st.markdown(f"""
        <div class="status-indicator status-ready">
            📚 Ready to chat • {st.session_state.documents_count} sources loaded
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-indicator status-empty">
            📁 Add some documents or web content to get started
        </div>
        """, unsafe_allow_html=True)

    # ── Add Content ──
    st.markdown("### Add Content")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Upload Files**")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'txt', 'csv'],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"file_uploader_{st.session_state.file_uploader_key}"
        )
        if uploaded_files:
            process_files(uploaded_files)

    with col2:
        st.markdown("**Add Web Content**")
        url = st.text_input(
            "Enter URL",
            placeholder="https://example.com/article",
            label_visibility="collapsed",
            key="url_input",
            value=st.session_state.current_url
        )
        add_url_clicked = st.button("🌐 Add URL", use_container_width=True, type="primary")
        if add_url_clicked and url and url.startswith(('http://', 'https://')):
            process_url(url)
        elif add_url_clicked and url:
            st.error("Please enter a valid URL starting with http:// or https://")
        elif add_url_clicked:
            st.error("Please enter a URL")

    # ── Chat ──
    st.markdown("### Chat")

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("🗑️ Clear All", use_container_width=True,
                     help="Clear chat, documents, URLs, and uploaded files"):
            clear_all_data()
            st.rerun()

    # ── Chat history with code highlighting ──
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                with st.chat_message("user"):
                    st.write(message['content'])
            else:
                with st.chat_message("assistant"):
                    # render_response handles plain text AND code blocks
                    render_response(message['content'])
                    if message.get('sources'):
                        st.markdown("**Sources:**")
                        for source in message['sources']:
                            st.markdown(
                                f'<span class="source-tag">{source}</span>',
                                unsafe_allow_html=True
                            )

    # ── Input form ──
    with st.form(key=f"chat_form_{st.session_state.chat_input_key}", clear_on_submit=True):
        user_question = st.text_input(
            "Ask a question about your content:",
            placeholder="e.g. Write a Python function to sort a list / What is this document about?",
            key=f"chat_input_{st.session_state.chat_input_key}"
        )
        send_clicked = st.form_submit_button("💬 Send", use_container_width=True, type="primary")

    if send_clicked and user_question.strip():
        handle_chat_input(user_question)
        st.session_state.chat_input_key += 1
        st.rerun()


if __name__ == "__main__":
    main()