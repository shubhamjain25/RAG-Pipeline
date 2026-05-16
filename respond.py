import time
from typing import NamedTuple, TypedDict
from models import *
from unstructured.partition.pdf import partition_pdf
import os
from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document
from models import *
from supabase import create_client, Client
from postgrest.exceptions import APIError


class RAGResponse(TypedDict):
    answer: str
    chunks: list[str]  # 3 source chunks used to generate the answer


def retrieve_chunks(query, document_id, k=3):
    print(f"🔄 Started retrieving chunks for query from db")
    # Retrieve the env data
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
            "filter_document_id": document_id
        }
    ).execute()
    print(f"✅ Successfully received {len(response.data)} chunks from db")
    print("="*20)
    print(response.data)
    return response.data


# ── Placeholder RAG query — swap with real logic ──────────────────────────────
def query_rag(question,document_id):
    # Retrieve relevant chunks
    retrieved_chunks = retrieve_chunks(query=question,document_id=document_id,k=3)

    # Extract chunk text
    chunk_texts = [
        chunk["content"]
        for chunk in retrieved_chunks
    ]

    if not retrieved_chunks:
        return RAGResponse(
            answer="No relevant information found.",
            chunks=[]
        )

    context = "\n\n".join([
        chunk["content"]
        for chunk in retrieved_chunks
    ])

    # Create prompt
    user_prompt = f"""
        You are a helpful PDF assistant. Answer the user's question ONLY using the provided context.
        If the answer is not present in the context,say: "I could not find that information in the document."
        Context:
        {context}
        Question: {question}
    """

    # Invoke LLM
    llm = get_deterministic_llm()
    llm_response = llm.invoke(user_prompt)

    print("=" * 20)
    print(llm_response)

    # Return structured response
    return RAGResponse(
        answer=llm_response.content,
        chunks=chunk_texts
    )
