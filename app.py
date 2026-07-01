import streamlit as st
import time
import uuid
import tempfile
from process import process_document
from respond import query_rag_stream
from conversation_log import generate_conversation_id, new_log_entry
from rag_config import HISTORY_TURNS_TO_INCLUDE, HISTORY_MESSAGE_CHAR_LIMIT
import traceback
import os
import logging
import warnings
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("taggers/averaged_perceptron_tagger")
except LookupError:
    nltk.download("averaged_perceptron_tagger")

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

    /* ── Source chunk cards ───────────────────────────────────────── */
    .chunk-card {
        background:#1e3a5f;
        border-left: 4px solid #4a9eff;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
        color: #d6e8ff;
        font-size: 0.88rem;
        line-height: 1.55;
    }
    .chunk-card strong { color:#7ec8ff; }

    /* ── Deep-think badge / subtle prompt ────────────────────────── */
    .deep-think-badge {
        display: inline-block;
        background: #2a1f4d;
        color: #b9a4ff;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    div[data-testid="stCaptionContainer"] { opacity: 0.75; }
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
    "conversation_id": None,
    "conversation_log": [],
    "pending_deep_think": None,  # holds the question text awaiting a deep-think rerun
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.conversation_id is None:
    st.session_state.conversation_id = generate_conversation_id()


# ── Shared turn-processing helper (used for both standard & deep-think turns) ──
def _format_history() -> str:
    """
    Formats the last HISTORY_TURNS_TO_INCLUDE user/assistant pairs from
    st.session_state.messages into plain text for the condensation + answer
    prompts. Each message is truncated to HISTORY_MESSAGE_CHAR_LIMIT chars so
    one long "deep think" answer can't dominate the next turn's prompt budget.
    """
    history_msgs = st.session_state.messages[-(HISTORY_TURNS_TO_INCLUDE * 2):]
    if not history_msgs:
        return ""

    lines = []
    for msg in history_msgs:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:HISTORY_MESSAGE_CHAR_LIMIT]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _run_turn(question: str, deep_think: bool, history_text: str = ""):
    """Runs retrieval+generation for one turn, streams the answer, logs it, and
    appends the resulting assistant message to session state."""
    spinner_text = "🧠 Thinking deeply…" if deep_think else "Thinking…"

    with st.chat_message("assistant"):
        with st.spinner(spinner_text):
            metadata, stream_gen = query_rag_stream(
                question=question,
                document_id=st.session_state.doc_id,
                deep_think=deep_think,
                history_text=history_text,
            )
        start = time.time()
        if deep_think:
            st.markdown('<span class="deep-think-badge">🧠 Deep think</span>', unsafe_allow_html=True)
        full_answer = st.write_stream(stream_gen)
        latency_ms = int((time.time() - start) * 1000)

        chunks = metadata.get("chunks", []) or []
        if chunks:
            with st.expander("📚 Source passages", expanded=False):
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(
                        f"""<div class="chunk-card"><strong>Chunk {i}</strong><br>{chunk.get("content", "")}</div>""",
                        unsafe_allow_html=True,
                    )

        msg_id = uuid.uuid4().hex
        allow_deep_think = (not deep_think) and bool(chunks)
        if allow_deep_think:
            st.caption("Not satisfied? 🧠 think deeper ↓ (see button below)")

    # Persist to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "chunks": chunks,
        "deep_think": deep_think,
        "allow_deep_think": allow_deep_think,
        "msg_id": msg_id,
        "question": question,
    })

    # Persist to conversation log
    log_entry = new_log_entry(
        conversation_id=st.session_state.conversation_id,
        document_id=st.session_state.doc_id,
        question=question,
        metadata=metadata,
        latency_ms=latency_ms,
    )
    st.session_state.conversation_log.append(log_entry)


@st.dialog("📜 Conversation Log", width="large")
def _show_conversation_log():
    st.caption(f"Conversation ID: `{st.session_state.conversation_id}`")
    if not st.session_state.conversation_log:
        st.info("No turns logged yet for this conversation.")
        return

    for entry in reversed(st.session_state.conversation_log):
        title = f"{entry['timestamp']} — {'🧠 Deep think' if entry['deep_think'] else 'Standard'}"
        with st.expander(title, expanded=False):
            st.markdown(f"**User message:** {entry['user_message']}")
            if entry.get("standalone_question") and entry["standalone_question"] != entry["user_message"]:
                st.markdown(f"**Condensed for retrieval:** {entry['standalone_question']}")
            st.markdown(f"**Mode:** {entry['decomposition_mode'] or 'single_query'}")
            if len(entry["queries_used"]) > 1:
                st.markdown("**Queries used:**")
                for q in entry["queries_used"]:
                    st.markdown(f"- {q}")
            st.markdown("**Retrieved chunks:**")
            for c in entry["retrieved_chunks"]:
                sim = c.get("similarity")
                sim_str = f" (similarity: {sim:.3f})" if isinstance(sim, (int, float)) else ""
                st.markdown(f"- `id={c.get('id')}`{sim_str}: {str(c.get('content', ''))[:200]}…")
            st.markdown("**LLM answer:**")
            st.markdown(entry["llm_answer"])
            tok = entry.get("token_usage") or {}
            st.markdown(
                f"**Token usage:** input={tok.get('input_tokens')}, "
                f"output={tok.get('output_tokens')}, total={tok.get('total_tokens')}"
            )
            st.markdown(f"**Latency:** {entry['latency_ms']} ms")
            if entry.get("error"):
                st.error(f"Error: {entry['error']}")


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
# STAGE 2 — Processing
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
    col_title, col_eye, col_btn = st.columns([4, 1, 1.3])
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
    with col_eye:
        if st.button("👁️", help="View conversation log", use_container_width=True):
            _show_conversation_log()
    with col_btn:
        if st.button("↩ New doc", use_container_width=True):
            st.session_state.stage             = "upload"
            st.session_state.pdf_name           = None
            st.session_state.pdf_path           = None
            st.session_state.doc_id             = None
            st.session_state.messages           = []
            st.session_state.conversation_id    = generate_conversation_id()
            st.session_state.conversation_log   = []
            st.session_state.pending_deep_think = None
            st.rerun()

    st.divider()

    # ── Render conversation history ────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("deep_think"):
                st.markdown('<span class="deep-think-badge">🧠 Deep think</span>', unsafe_allow_html=True)
            st.markdown(msg["content"])

            if msg["role"] == "assistant" and msg.get("chunks"):
                with st.expander("📚 Source passages", expanded=False):
                    for i, chunk in enumerate(msg["chunks"], 1):
                        st.markdown(
                            f"""<div class="chunk-card"><strong>Chunk {i}</strong><br>{chunk.get("content", "")}</div>""",
                            unsafe_allow_html=True,
                        )

            if msg["role"] == "assistant" and msg.get("allow_deep_think"):
                st.caption("Not satisfied with this answer?")
                if st.button("🧠 Think deeper", key=f"deep_{msg['msg_id']}"):
                    st.session_state.pending_deep_think = msg["question"]
                    st.rerun()

    # ── Handle a pending "think deeper" request ─────────────────────────────────
    if st.session_state.pending_deep_think:
        question = st.session_state.pending_deep_think
        st.session_state.pending_deep_think = None
        history_text = _format_history()
        _run_turn(question, deep_think=True, history_text=history_text)
        st.rerun()

    # ── Chat input ─────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask something about your document…"):
        history_text = _format_history()  # captured BEFORE appending current message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        _run_turn(prompt, deep_think=False, history_text=history_text)
        st.rerun()