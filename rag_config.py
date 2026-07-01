"""
rag_config.py — single source of truth for every tunable constant in the
retrieval/RAG pipeline. Nothing in retrieval.py or respond.py should hardcode
a magic number; everything pulls from here so the whole pipeline can be tuned
from one place.

This file assumes the following Supabase RPC signature (similarity_threshold
based — filtering happens at table level, not SQL):

    match_document_chunks(
        query_embedding vector(1024),
        match_count int,
        filter_document_id uuid,
        similarity_threshold float
    ) returns table (id uuid, chunk_index int, content text, metadata jsonb, similarity float)

"""

# ── Similarity filtering ───────────────────────────────────────────────────
# Cosine similarity is in [-1, 1] (in practice usually [0, 1] for normalized
# embeddings like Cohere's). Anything below this is treated as noise — i.e.
# "the vector search just returned its least-bad guess, not an actually
# relevant chunk" — and is filtered out BEFORE it leaves the database (see
# migrations_match_document_chunks.sql — the `similarity_threshold` parameter
# on match_document_chunks does a `WHERE ... >= similarity_threshold` directly
# in SQL). retrieval.py passes this constant straight through to the RPC call;
# there is no client-side filtering — what Supabase returns is what's used.
#
# 0.3 is a reasonable starting point for Cohere embed-english-v3.0 but is NOT
# universal — different embedding models have different similarity
# distributions. If you find the bot is either (a) refusing to answer
# questions that ARE actually in the document, lower this; or (b) citing
# clearly irrelevant chunks, raise it. Recommend validating against a sample
# of real query/chunk pairs before trusting this number in production.
#
# IMPORTANT: requires migrations_match_document_chunks.sql to have been run
# against your Supabase project — the RPC must accept this parameter or the
# call will fail with an "unknown parameter" error.
SIMILARITY_THRESHOLD = 0.3


# ── Standard (fast-path) retrieval ──────────────────────────────────────────
# Number of chunks used as context for a normal, non-"deep think" answer.
# Keep this small — it's the default path and should stay fast and cheap.
STANDARD_TOP_K = 3


# ── Deep-think (decompose + RRF) retrieval ──────────────────────────────────
# Word-count cutoff used to decide HOW a query gets decomposed into 3
# sub-queries when the user clicks "Think deeper":
#   - question longer than this many words -> split along sentence boundaries
#     (the user already wrote out multiple thoughts; just separate them)
#   - question at or below this many words -> ask the LLM to generate 3
#     semantically-varied rephrasings (a short question doesn't have natural
#     sentence breaks to exploit, so we widen the net semantically instead)
DECOMPOSITION_WORD_THRESHOLD = 30

# How many candidate chunks to request PER sub-query in deep-think mode,
# before similarity filtering and before RRF fusion. Higher = more recall
# per sub-query at the cost of more embedding/DB calls and a larger pool for
# RRF to fuse over.
DEEP_THINK_K_PER_QUERY = 5

# How many chunks survive Reciprocal Rank Fusion to become the final context
# fed to the LLM in deep-think mode. Kept equal to STANDARD_TOP_K so both
# paths give the LLM a comparable amount of context — the difference between
# the two modes is retrieval *quality* (multi-query + fusion), not quantity.
DEEP_THINK_FINAL_TOP_N = 3

# RRF's smoothing constant. Standard literature default is 60 — it controls
# how much weight rank position #1 gets relative to #2, #3, etc. Lower values
# make top ranks dominate more aggressively; higher values flatten the
# scoring curve. 60 is a well-established default and rarely needs tuning.
RRF_K = 60


# ── Conversation memory ─────────────────────────────────────────────────────
# How many previous Q&A turns (user + assistant pairs) get included as
# context when answering a follow-up like "summarize that" or "what about
# X instead?". Without this, every question is answered in total isolation,
# which breaks on any question that references prior conversation.
#
# Keep this small — every turn included here adds tokens to BOTH the
# condensation call (rewriting the follow-up into a standalone search query)
# and the final answer call. 3 turns is usually enough for natural follow-up
# chains without bloating the prompt.
HISTORY_TURNS_TO_INCLUDE = 3

# Each historical message gets truncated to this many characters before being
# included in a prompt. A "deep think" answer can be long; without this, a
# single prior turn could dominate the context budget of the next question.
HISTORY_MESSAGE_CHAR_LIMIT = 600