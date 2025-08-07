import requests
import fitz  # PyMuPDF
import os
import uuid

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import time
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
REGION = "us-east-1"
INDEX_NAME = "chatbot-index"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # for OpenAI embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=REGION)
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)

# Connect to index
index = pc.Index(INDEX_NAME)

def download_pdf(url: str) -> str:
    print("download pdf")
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError("Failed to download PDF.")
    
    filename = f"temp_{uuid.uuid4().hex}.pdf"
    with open(filename, "wb") as f:
        f.write(response.content)
    return filename

def extract_text_from_pdf(path: str) -> str:
    print("extract pdf")
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def create_vectorstore(text: str):
    print("vectore store")
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = text_splitter.create_documents([text])

    # Load embeddings
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
    # Generate UUIDs for each chunk
    uuids = [f"id_{uuid.uuid4().hex}" for _ in docs]

    # Create the vectorstore
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
    vectorstore.add_documents(documents=docs, ids=uuids)

    return vectorstore.as_retriever()
