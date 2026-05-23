import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceBgeEmbeddings



# Initilizing varaibles


load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY



# Extract text from PDF files

def load_pdf_files(data):    #data is the parameter
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )


    documents = loader.load()
    return documents



#Minimal documents

def filter_to_minimal_docs(docs:List[Document]) -> List[Document]:
    """ 
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """

    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source" : src}
            )
        )

    return minimal_docs




# Split the doucments into smaller chunks

def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50,
    )

    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk







def download_embeddings():
    """ 
    Download and return the HuggingFace embeddings model.
    """

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceBgeEmbeddings(
        model_name = model_name
    )

    return embeddings
