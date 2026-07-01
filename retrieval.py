"""
Retrieval layer for the RAG pipeline.

Handles three responsibilities:
1. Vector retrieval against Supabase (`retrieve_chunks`) with similarity
   filtering applied in Python — the deployed `match_document_chunks` RPC has
   no similarity_threshold parameter, so this is done client-side via an
   over-fetch + filter + trim strategy (see rag_config.CANDIDATE_POOL_MULTIPLIER
   for why we don't just fetch exactly k).
2. Query decomposition for the "Think deeper" path — either splitting a long
   query by sentence boundaries, or asking the LLM for 3 semantically varied
   queries (strict JSON output, validated against `DecomposedQueries`).
3. Multi-query retrieval + Reciprocal Rank Fusion (RRF) to merge several
   ranked chunk lists into one best-of-N list.
"""

import os
import re
import json

from pydantic import ValidationError
from langchain_core.messages import SystemMessage, HumanMessage
from supabase import create_client, Client

from models import get_embedding_model, get_deterministic_llm, DecomposedQueries
from rag_config import (
    SIMILARITY_THRESHOLD,
    DECOMPOSITION_WORD_THRESHOLD,
    RRF_K,
)

DECOMPOSITION_SYSTEM_PROMPT = """You are a query analysis engine for a document retrieval system. Your sole task is to output a JSON object containing exactly 3 alternative search queries that capture different semantic angles of the user's original question, in order to improve retrieval recall.

Output STRICT JSON only — no markdown code fences, no preamble, no explanation, no trailing text. The JSON must match exactly this schema:
{"queries": ["<query 1>", "<query 2>", "<query 3>"]}

Rules:
- Each of the 3 queries must be a self-contained, standalone search query (not a sentence fragment).
- Vary phrasing, synonyms, or the angle/aspect of each query while preserving the original intent.
- Do NOT answer the user's question. Do NOT add commentary, keys, or any text outside the JSON object.
- Output ONLY the JSON object and nothing else."""


# ── Vector retrieval ──────────────────────────────────────────────────────────
def retrieve_chunks(query: str, document_id, k: int = 3) -> list[dict]:
    """
    Embed `query` and fetch up to k chunks for `document_id` via the
    match_document_chunks RPC. Similarity filtering happens entirely at the
    database level (see migrations_match_document_chunks.sql) — the RPC's
    `similarity_threshold` parameter does `WHERE similarity >= threshold` in
    SQL before the LIMIT is applied, so this function does no client-side
    filtering. Whatever Supabase returns is used as-is.

    Requires migrations_match_document_chunks.sql to have been applied —
    older versions of the RPC without a similarity_threshold parameter will
    raise an error here.

    Note: this can legitimately return fewer than k chunks (or zero) if the
    document doesn't have k passages meeting SIMILARITY_THRESHOLD — that's
    intentional, not a bug.
    """
    print(f"🔄 Started retrieving chunks for query from db")
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    embedding_model = get_embedding_model()
    supabase: Client = create_client(url, key)

    query_embedding = embedding_model.embed_query(query)

    response = supabase.rpc(
        "match_document_chunks",
        {
            "query_embedding": query_embedding,
            "match_count": k,
            "filter_document_id": document_id,
            "similarity_threshold": SIMILARITY_THRESHOLD,
        },
    ).execute()
    chunks = response.data or []
    print(f"✅ Received {len(chunks)} chunk(s) from db, all already >= similarity {SIMILARITY_THRESHOLD}")
    return chunks


# ── JSON robustness helpers ───────────────────────────────────────────────────
def _strip_json_fences(text: str) -> str:
    """Open-source models often wrap JSON in ```json ... ``` even when told not to. Strip it defensively."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


# ── Query decomposition ───────────────────────────────────────────────────────
def _split_by_sentences(question: str, n: int = 3) -> list[str]:
    """Split a long query into exactly `n` parts along sentence boundaries.
    If there aren't enough natural sentence breaks, the longest fragment is
    further split in half by word count until we have `n` parts."""
    parts = re.split(r"(?<=[.!?])\s+", question.strip())
    parts = [p.strip() for p in parts if p.strip()]

    if not parts:
        return [question.strip()] * n

    while len(parts) < n:
        idx = max(range(len(parts)), key=lambda i: len(parts[i].split()))
        words = parts[idx].split()
        if len(words) < 2:
            parts.append(question.strip())
            continue
        mid = len(words) // 2
        first, second = " ".join(words[:mid]), " ".join(words[mid:])
        parts[idx:idx + 1] = [first, second]

    if len(parts) > n:
        merged = parts[: n - 1] + [" ".join(parts[n - 1:])]
        parts = merged

    return parts[:n]


def decompose_query(question: str) -> tuple[list[str], str]:
    """
    Returns (list_of_3_queries, mode).
    mode is one of: 'sentence_split', 'semantic_llm', 'fallback'.
    """
    word_count = len(question.split())

    if word_count > DECOMPOSITION_WORD_THRESHOLD:
        return _split_by_sentences(question, n=3), "sentence_split"

    llm = get_deterministic_llm()
    messages = [
        SystemMessage(content=DECOMPOSITION_SYSTEM_PROMPT),
        HumanMessage(content=f"Original question: {question}"),
    ]

    try:
        llm_resp = llm.invoke(messages)
        cleaned = _strip_json_fences(llm_resp.content)
        data = json.loads(cleaned)
        formatted = DecomposedQueries(**data)
        queries = formatted.queries
        if len(queries) != 3 or not all(q.strip() for q in queries):
            raise ValueError("Decomposition did not return exactly 3 non-empty queries")
        return queries, "semantic_llm"
    except (json.JSONDecodeError, ValidationError, ValueError, TypeError, KeyError) as e:
        print(f"⚠️ Query decomposition failed ({e}); falling back to original query x3")
        return [question, question, question], "fallback"


# ── Multi-query retrieval + RRF ───────────────────────────────────────────────
def multi_query_retrieve(queries: list[str], document_id, k_per_query: int) -> list[list[dict]]:
    """Run retrieval once per query (each already similarity-filtered), returning
    a list of ranked chunk lists for RRF to fuse."""
    ranked_lists = []
    for q in queries:
        chunks = retrieve_chunks(query=q, document_id=document_id, k=k_per_query)
        ranked_lists.append(chunks)
    return ranked_lists


def reciprocal_rank_fusion(ranked_lists: list[list[dict]], top_n: int, rrf_k: int = RRF_K) -> list[dict]:
    """
    Fuse multiple ranked chunk lists into one list using Reciprocal Rank Fusion:
        score(chunk) = sum( 1 / (rrf_k + rank) ) across every list it appears in.
    Chunks are de-duplicated by their db `id` (falls back to a content hash if missing).

    Note: every chunk entering this function has already individually passed
    SIMILARITY_THRESHOLD in retrieve_chunks(), so fusion never "resurrects" a
    chunk that was below the bar just because it appeared in multiple lists —
    RRF only re-ranks among chunks that already qualified.
    """
    scores: dict[str, float] = {}
    chunk_lookup: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list):
            chunk_key = str(chunk.get("id")) if chunk.get("id") is not None else str(hash(chunk.get("content", "")))
            scores[chunk_key] = scores.get(chunk_key, 0.0) + 1.0 / (rrf_k + rank + 1)
            chunk_lookup.setdefault(chunk_key, chunk)

    fused_order = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [chunk_lookup[key] for key, _ in fused_order[:top_n]]