import time
import random
from unstructured.partition.pdf import partition_pdf
import os
from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document
from models import *
from supabase import create_client, Client
from postgrest.exceptions import APIError

def partition_document(file_path: str, file_name: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    print(f"🔄 Started Partitioning for: {file_name} at {file_path}")

    # ONLY HANDLING TEXT FOR NOW
    partitioned_pdf = partition_pdf(
        filename=file_path,
        include_page_breaks=True,
        strategy='fast',
    )

    print(f"✅ Successfully partitioned into {len(partitioned_pdf)} parts")
    return partitioned_pdf

def create_chunks(elements):
    print(f"🔄 Started Chunking for partitioned parts")

    chunks_list = chunk_by_title(
        elements=elements,
        combine_text_under_n_chars=500,
        max_characters=2000,
        new_after_n_chars=1500,
        overlap=200
    )

    print(f"✅ Successfully chunked into {len(chunks_list)} parts")
    return chunks_list

def process_chunk_list(chunk_list):
    print(f"🔄 Started Processing for chunk list")

    langchain_documents: list[Document] = []
    for chunk in chunk_list:
        #In future when we deal with images & tables it will be tackled here
        separated_content = {
            'text': chunk.text,
        }

        summarised_chunk = separated_content['text']
        metadata = {
            'raw_text': separated_content['text'],
        }

        doc = Document(
            summarised_chunk,
            metadata=metadata,
        )
        langchain_documents.append(doc)

    print(f"✅ Successfully processed {len(langchain_documents)} langchain documents")
    return langchain_documents

def generate_embeddings(langchain_documents):
    print(f"🔄 Started Embedding process for chunk list")
    embedding_model = get_embedding_model()

    texts = [
        doc.page_content
        for doc in langchain_documents
    ]

    embeddings = embedding_model.embed_documents(texts)

    embedded_documents = []

    for doc, embedding in zip(langchain_documents,embeddings):
        embedded_documents.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "embedding": embedding
        })

    print(f"✅ Successfully embedded {len(embedded_documents)} documents")
    return embedded_documents

def store_to_db(embedded_documents, file_name="temp.pdf"):
    print(f"🔄 Started storing to database")

    # Retrieve the env data
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    try:
        # 1. Initialize Client
        supabase: Client = create_client(url, key)

        # 2. Connection Check (Optional but "Graceful")
        # We try to select 1 row from any table or just let the first insert fail gracefully.
        # If the URL or Key is wrong, this is where it will crash.

        # 3. Insert Parent Document
        document_response = (
            supabase
            .table("uploaded_documents")
            .insert({"file_name": file_name})
            .execute()
        )

        if not document_response.data:
            raise Exception("Failed to retrieve document_id after insertion.")

        document_id = document_response.data[0]["id"]
        print(f"✅ Created document record with ID: {document_id}")

        # 4. Prepare and Bulk Insert Chunks
        chunk_rows = [
            {
                "document_id": document_id,
                "chunk_index": idx,
                "content": doc["content"],
                "metadata": doc["metadata"],
                "embedding": doc["embedding"]
            }
            for idx, doc in enumerate(embedded_documents)
        ]

        # Supabase handles lists as bulk inserts automatically
        supabase.table("document_chunks").insert(chunk_rows).execute()
        print(f"✅ Successfully stored {len(chunk_rows)} chunks.")

        return document_id

    except APIError as e:
        print(f"❌ Database Error: {e.message}")
        return None
    except Exception as e:
        print(f"❌ Connection or Unexpected Error: {str(e)}")
        return None

def process_document(pdf_path: str, pdf_name: str):
    """
    Process an uploaded PDF document (partition, chunk, embed).

    Yields the active stage name before each step so the UI can update
    in real-time, then yields ``("done", doc_id)`` on completion.

    Stages: creating_chunks → processing_chunks → embedding_chunks → storing_to_db

    TODO: Replace placeholder calls with the real pipeline.
    """
    yield "creating_chunks"
    partitioned_pdf_elements = partition_document(pdf_path, pdf_name)


    yield "processing_chunks"
    chunk_list = create_chunks(partitioned_pdf_elements)
    langchain_documents = process_chunk_list(chunk_list)

    yield "embedding_chunks"
    embedded_documents = generate_embeddings(langchain_documents)

    yield "storing_to_db"
    doc_id = store_to_db(embedded_documents, pdf_name)

    yield ("done", doc_id)