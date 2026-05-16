import streamlit as st
import time
import tempfile
from process import process_document
from respond import query_rag
import traceback
import os
import logging
import warnings

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

warnings.filterwarnings("ignore")

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("unstructured").setLevel(logging.ERROR)

from transformers.utils import logging as transformers_logging
transformers_logging.set_verbosity_error()


# ── Processing pipeline helpers ───────────────────────────────────────────────
_PIPELINE_STAGES = [
    ("creating_chunks",   "🔨", "Creating Chunks"),
    ("processing_chunks", "⚙️", "Processing Chunks"),
    ("embedding_chunks",  "🧠", "Embedding Chunks"),
    ("storing_to_db",     "💾", "Storing to DB"),
]
_STAGE_IDX = {key: i for i, (key, _, _) in enumerate(_PIPELINE_STAGES)}


def _pipeline_html(active_key: str, all_done: bool = False) -> str:
    """Render the 4-stage pipeline card row as HTML."""
    active_idx = len(_PIPELINE_STAGES) if all_done else _STAGE_IDX.get(active_key, 0)
    parts = []
    for i, (_, icon, label) in enumerate(_PIPELINE_STAGES):
        if i < active_idx:
            css, display = "stage-done", "✅"
        elif i == active_idx:
            css, display = "stage-active", icon
        else:
            css, display = "stage-pending", icon
        parts.append(
            f'<div class="stage-card {css}">'
            f'<div class="stage-icon">{display}</div>'
            f'<div class="stage-label">{label}</div>'
            f'</div>'
        )
    inner = '<div class="stage-sep">›</div>'.join(parts)
    return f'<div class="pipeline-wrap">{inner}</div>'


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="DocChat", page_icon="📄", layout="centered")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Hero card on landing page */
    .hero-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d1b2a 100%);
        border-radius: 16px;
        padding: 48px 40px;
        text-align: center;
        color: white;
        margin-bottom: 24px;
    }
    .hero-card h1 { font-size: 2.6rem; margin-bottom: 8px; }
    .hero-card p  { font-size: 1.1rem; color: #a0b8d8; margin: 0; }

    /* Success banner */
    .success-banner {
        background: linear-gradient(90deg, #0f9b58, #0d7a46);
        border-radius: 12px;
        padding: 28px;
        text-align: center;
        color: white;
        font-size: 1.3rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Chat header pill */
    .chat-header {
        background: #1e3a5f;
        border-radius: 10px;
        padding: 12px 20px;
        color: white;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 16px;
    }

    /* Hide default streamlit footer */
    footer { visibility: hidden; }

    /* ── Processing pipeline ──────────────────────────────────────── */
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 0 2px #4a9eff44, 0 0 14px 4px #4a9eff33; }
        50%       { box-shadow: 0 0 0 3px #4a9effbb, 0 0 26px 8px #4a9eff55; }
    }
    .pipeline-wrap {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-wrap: wrap;
        gap: 4px;
        padding: 40px 16px;
    }
    .stage-card {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 10px;
        padding: 24px 20px;
        border-radius: 14px;
        min-width: 112px;
        border: 2px solid transparent;
        transition: background 0.4s ease, border-color 0.4s ease, color 0.4s ease;
    }
    .stage-done    { background:#071f10; border-color:#0f9b58; color:#3dce7a; }
    .stage-active  { background:#071020; border-color:#4a9eff; color:#7ec8ff;
                     animation: pulse-glow 1.6s ease-in-out infinite; }
    .stage-pending { background:#0f0f1a; border-color:#1e1e33; color:#35354f; }
    .stage-icon    { font-size:2rem; line-height:1; }
    .stage-label   { font-size:0.78rem; font-weight:600; text-align:center; line-height:1.4; }
    .stage-sep     { font-size:1.4rem; color:#252540; padding:0 4px; user-select:none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state initialisation ──────────────────────────────────────────────
for key, default in {
    "stage": "upload",        # upload | processing | done | chat
    "pdf_name": None,
    "pdf_path": None,
    "doc_id": None,
    "messages": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Upload
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.stage == "upload":

    st.markdown(
        """
        <div class="hero-card">
            <h1>📄 DocChat</h1>
            <p>Upload a PDF and chat with your document instantly.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Drop your PDF here",
        type="pdf",
        help="Supported format: PDF",
        max_upload_size=10
    )

    if uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(uploaded.read())
        tmp.close()
        st.session_state.pdf_name  = uploaded.name
        st.session_state.pdf_path  = tmp.name
        st.session_state.stage     = "processing"
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Processing (5-second placeholder)
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "processing":

    st.markdown(f"### ⚙️ Processing `{st.session_state.pdf_name}`")
    st.caption("Please wait while we prepare your document…")

    pipeline_slot = st.empty()
    success = False
    try:
        for stage in process_document(st.session_state.pdf_path, st.session_state.pdf_name):
            if isinstance(stage, tuple) and stage[0] == "done":
                st.session_state.doc_id = stage[1]
                success = True
                break
            pipeline_slot.markdown(_pipeline_html(stage), unsafe_allow_html=True)
    except Exception as e:

        st.error(f"Processing failed: {e}")
        st.code(traceback.format_exc())


    if success:
        # Briefly show all stages green before transitioning
        pipeline_slot.markdown(_pipeline_html("", all_done=True), unsafe_allow_html=True)
        time.sleep(0.6)
        st.session_state.stage = "done"
        st.rerun()
    else:
        st.error("Processing failed. Please try uploading the document again.")
        if st.button("↩ Try again"):
            st.session_state.stage    = "upload"
            st.session_state.pdf_name = None
            st.session_state.pdf_path = None
            st.session_state.doc_id   = None
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Success animation, then transition to chat
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "done":

    st.balloons()

    st.markdown(
        f"""
        <div class="success-banner">
            ✅ &nbsp; <em>{st.session_state.pdf_name}</em> has been processed successfully!
        </div>
        """,
        unsafe_allow_html=True,
    )

    time.sleep(2.5)
    st.session_state.stage = "chat"
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — Chat UI
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "chat":

    # ── Header ────────────────────────────────────────────────────────────────
    col_title, col_btn = st.columns([5, 1])
    with col_title:
        st.markdown(
            f"""
            <div class="chat-header">
                📄&nbsp; <span style="font-size:1.05rem; font-weight:600;">
                {st.session_state.pdf_name}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_btn:
        if st.button("↩ New doc", use_container_width=True):
            st.session_state.stage    = "upload"
            st.session_state.pdf_name = None
            st.session_state.pdf_path = None
            st.session_state.doc_id   = None
            st.session_state.messages = []
            st.rerun()

    st.divider()

    # ── Render conversation history ────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Chat input ─────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask something about your document…"):

        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and stream assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                response = query_rag(question=prompt,document_id=st.session_state.doc_id)
            st.markdown(response['answer'])

            # ── Source chunks ──────────────────────────────────────────────
            if response['chunks']:
                with st.expander("📚 Source passages", expanded=False):
                    for i, chunk in enumerate(response['chunks'], 1):
                        st.markdown(
                            f"""
                            <div style="
                                background:#1e3a5f;
                                border-left: 4px solid #4a9eff;
                                border-radius: 6px;
                                padding: 12px 16px;
                                margin-bottom: 10px;
                                color: #d6e8ff;
                                font-size: 0.88rem;
                                line-height: 1.55;
                            ">
                                <strong style="color:#7ec8ff;">Chunk {i}</strong><br>{chunk}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        st.session_state.messages.append(
            {"role": "assistant", "content": response['answer']}
        )
