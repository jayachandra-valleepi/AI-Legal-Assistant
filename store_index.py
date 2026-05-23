import os
from dotenv import load_dotenv
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


#Initilizing vaiables

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

INDEX_NAME = os.getenv("INDEX_NAME")
os.environ["INDEX_NAME"] = INDEX_NAME





extracated_data = load_pdf_files(data='data/')
filter_data = filter_to_minimal_docs(extracated_data)
text_chunks = text_split(filter_data)


embeddings = download_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)



index_name = INDEX_NAME


if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name = INDEX_NAME,
        dimension=384,   #Dimensions of the embeddings
        metric="cosine",  # cosine similarity search
        spec=ServerlessSpec(cloud="aws", region="us-east-1")

    )


docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=INDEX_NAME
)
