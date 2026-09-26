"""Streamlit Web Interface for Future-Ready Children RAG Chatbot."""

from pathlib import Path
import streamlit as st

from src.core.config import get_settings
from src.embeddings.local_provider import LocalSentenceTransformerEmbeddings
from src.llm.sarvam_client import MockLLMClient, SarvamGLMClient
from src.models.query import QueryRequest
from src.retrieval.bm25_index import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever
from src.services.indexing_service import IndexingService
from src.services.rag_service import RAGService
from src.vectorstore.chroma_store import ChromaVectorStore

# Page Configuration
st.set_page_config(
    page_title="Future-Ready Children | Campaign AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .talking-point-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 4px;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
    }
    .citation-card {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        padding: 0.75rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #DBEAFE;
        color: #1E40AF;
        margin-right: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing RAG components...")
def get_rag_engine():
    """Initializes and caches heavy shared components (embeddings, vector store, indexes)."""
    settings = get_settings()

    vector_store = ChromaVectorStore(
        persist_dir=settings.chroma_persist_directory,
        collection_name=settings.chroma_collection_name,
    )
    embedding_provider = LocalSentenceTransformerEmbeddings(
        model_name=settings.embedding_model_name,
    )
    bm25_index = BM25Index()

    # Reconstruct BM25 index if Chroma already has documents
    if vector_store.count() > 0:
        dummy_vec = [0.0] * embedding_provider.dimension
        stored = vector_store.similarity_search_by_vector(dummy_vec, top_k=vector_store.count())
        bm25_index.index_chunks([chunk for chunk, _ in stored])

    indexing_service = IndexingService(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        bm25_index=bm25_index,
    )

    return settings, vector_store, embedding_provider, bm25_index, indexing_service


# Load core components
settings, vector_store, embedding_provider, bm25_index, indexing_service = get_rag_engine()

# Sidebar: Controls & Settings
with st.sidebar:
    st.image("https://img.icons8.com/color/96/graduation-cap.png", width=64)
    st.title("Campaign Control")

    st.subheader("Sarvam AI Configuration")
    user_api_key = st.text_input(
        "Sarvam API Key",
        value=settings.sarvam_api_key if not settings.sarvam_api_key.startswith("your_sarvam") else "",
        type="password",
        help="Enter your Sarvam AI API subscription key.",
    )

    model_name = st.selectbox(
        "Inference Model",
        options=["glm5.3", "sarvam-105b"],
        index=0,
    )

    st.divider()
    st.subheader("Retrieval Tuning")
    top_k = st.slider("Top Chunks (k)", min_value=1, max_value=8, value=settings.top_k_retrieval)
    dense_weight = st.slider("Dense Vector Weight", min_value=0.0, max_value=1.0, value=settings.dense_weight, step=0.05)
    sparse_weight = round(1.0 - dense_weight, 2)
    st.caption(f"Sparse (BM25) Weight: **{sparse_weight}**")

    st.divider()
    st.subheader("Document Index Status")
    doc_count = vector_store.count()
    st.metric(label="Indexed Chunks in Vector Store", value=doc_count)

    if st.button("🔄 Re-index Campaign FAQ Document"):
        with st.spinner("Indexing FAQ document..."):
            doc_path = Path("data/raw/Future-Ready Children_ FAQ.docx")
            if not doc_path.exists():
                doc_path = Path("data/raw/Future-Ready Children_ FAQ.pdf")

            chunks = indexing_service.index_document(doc_path, clear_existing=True)
            st.success(f"Indexed {len(chunks)} chunks!")
            st.rerun()

# Determine LLM Client
if user_api_key and not user_api_key.startswith("your_sarvam"):
    llm_client = SarvamGLMClient(
        api_key=user_api_key,
        base_url=settings.sarvam_base_url,
        model=model_name,
        temperature=settings.sarvam_temperature,
        max_tokens=settings.sarvam_max_tokens,
        timeout=settings.sarvam_timeout_seconds,
    )
    status_badge = "🟢 Sarvam GLM-5.3 Active"
else:
    llm_client = MockLLMClient()
    status_badge = "🟡 Demo / Mock Mode (Add Sarvam Key in Sidebar to enable live model)"

# Build active retriever & RAG service
retriever = HybridRetriever(
    vector_store=vector_store,
    embedding_provider=embedding_provider,
    bm25_index=bm25_index,
    dense_weight=dense_weight,
    sparse_weight=sparse_weight,
    rrf_k=settings.rrf_k,
)
rag_service = RAGService(retriever=retriever, llm_client=llm_client)

# Header Section
st.markdown('<div class="main-header">Future-Ready Children</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Intelligent Q&A and Champion Talking Points Assistant · Powered by Sarvam AI GLM-5.3</div>',
    unsafe_allow_html=True,
)
st.caption(f"Status: **{status_badge}** | Indexed Knowledge Base: **{doc_count} Q&A Chunks**")

# Prompt suggestions chips
st.markdown("**Suggested Champion Questions:**")
col1, col2, col3 = st.columns(3)
selected_prompt = None

if col1.button("💰 How much does it cost to support a school?"):
    selected_prompt = "How much does it cost to support a school?"
if col2.button("📜 What is the Spoken Tutorial IEEE standard?"):
    selected_prompt = "What is the Spoken Tutorial pedagogy and IEEE standard?"
if col3.button("🤝 How does team participation work on cYAAG?"):
    selected_prompt = "What is cYAAG and how does team participation work?"

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "talking_point" in msg and msg["talking_point"]:
            st.markdown(
                f'<div class="talking-point-box"><strong>🎙️ Champion Talking Point:</strong><br>{msg["talking_point"]}</div>',
                unsafe_allow_html=True,
            )
        if "citations" in msg and msg["citations"]:
            with st.expander(f"📚 Sources & Citations ({len(msg['citations'])})"):
                for c in msg["citations"]:
                    q_str = f"Q{c['question_number']}: {c['question_text']}" if c.get("question_number") else "Section Overview"
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <span class="badge">{c['section']}</span>
                            <strong>{q_str}</strong>
                            <p style="margin-top: 0.3rem; color: #4B5563;">{c['excerpt']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# Input handling
user_input = st.chat_input("Ask a question about the campaign...")
active_query = selected_prompt or user_input

if active_query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving facts and generating grounded response..."):
            request = QueryRequest(query=active_query, top_k=top_k)
            response = rag_service.answer_query(request)

            # Display direct answer
            st.markdown(response.answer)

            # Display talking point
            if response.champion_talking_point:
                st.markdown(
                    f'<div class="talking-point-box"><strong>🎙️ Champion Talking Point:</strong><br>{response.champion_talking_point}</div>',
                    unsafe_allow_html=True,
                )

            # Display citations
            if response.citations:
                with st.expander(f"📚 Sources & Citations ({len(response.citations)})"):
                    for c in response.citations:
                        q_str = f"Q{c.question_number}: {c.question_text}" if c.question_number else "Section Overview"
                        st.markdown(
                            f"""
                            <div class="citation-card">
                                <span class="badge">{c.section}</span>
                                <strong>{q_str}</strong>
                                <p style="margin-top: 0.3rem; color: #4B5563;">{c.excerpt}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # Escalation card if unanswerable
            if response.is_fallback:
                st.warning(
                    "⚠️ This inquiry reaches outside the standard Champion FAQ. "
                    "Reach out directly to the leadership team: `sujathan@wheelsglobal.org` or `saisudha@edupyramids.org`."
                )

    # Save to history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.answer,
            "talking_point": response.champion_talking_point,
            "citations": [c.model_dump() for c in response.citations],
        }
    )
