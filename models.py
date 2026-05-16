from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()

def get_embedding_model():

    embedding_model = CohereEmbeddings(
        model="embed-english-v3.0"
    )
    return embedding_model

def get_deterministic_llm():
    llm = ChatGroq(
        model='openai/gpt-oss-120b',
        temperature=0.1
    )
    return llm