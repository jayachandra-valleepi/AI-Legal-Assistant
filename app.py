from flask import Flask, render_template, jsonify, request
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from langchain.embeddings import HuggingFaceBgeEmbeddings
import os





app = Flask(__name__)


# Initilizing varaibles


load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["INDEX_NAME"] = INDEX_NAME



embeddings = HuggingFaceBgeEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2")


docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)


retriever = docsearch.as_retriever(search_type = "similarity", search_kwargs = {"k" : 3})


llm = ChatOpenAI(model = "gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}")
    ]
)


memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)



rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={
        "prompt": prompt
    }
)



@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(msg)
    response = rag_chain.invoke({"question" : msg})
    print("Response : ", response["answer"])
    return str(response["answer"])



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)