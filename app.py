from flask import Flask, render_template, request, redirect

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceBgeEmbeddings

from src.prompt import *
from src.auth.auth_handler import login_user, is_authenticated
from src.memory.memory_store import get_memory

import os


app = Flask(__name__)


# ================= ENV ================= #

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["INDEX_NAME"] = INDEX_NAME


# ================= EMBEDDINGS ================= #

embeddings = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ================= VECTOR DB ================= #

docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# ================= LLM ================= #

llm = ChatOpenAI(model="gpt-4o-mini")


# ================= PROMPT ================= #

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}")
    ]
)


# ================= ROUTES ================= #

@app.route("/")
def home():

    return render_template("login.html")


# ================= LOGIN ================= #

@app.route("/login", methods=["POST"])
def login():

    email = request.form.get("email")
    password = request.form.get("password")

    success = login_user(email, password)

    if not success:
        return "Invalid Credentials"

    return redirect(f"/chat?email={email}")


# ================= CHAT PAGE ================= #

@app.route("/chat")
def chat_page():

    email = request.args.get("email")

    if not is_authenticated(email):
        return "Please Login First"

    return render_template("chat.html", email=email)


# ================= CHAT API ================= #

@app.route("/get", methods=["POST"])
def chat():

    email = request.form["email"]
    msg = request.form["msg"]

    # Check authentication
    if not is_authenticated(email):
        return "Unauthorized Access"

    # User-specific memory
    memory = get_memory(email)

    # Create user-specific RAG chain
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": prompt
        }
    )

    response = rag_chain.invoke({
        "question": msg
    })

    return str(response["answer"])


# ================= MAIN ================= #

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)