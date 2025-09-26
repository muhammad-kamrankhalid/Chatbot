import os
import bs4
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# LangChain imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ----------------- Load environment variables -----------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")

# ----------------- Initialize LLM -----------------
llm = ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant")

# ----------------- Embeddings -----------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ----------------- Load Multiple Website Pages -----------------
urls = [
    "https://freedomzstorage.com/",
    "https://freedomzstorage.com/blogs/",
    "https://freedomzstorage.com/faqs/",
    "https://freedomzstorage.com/contact/"
]

persist_directory = "./chroma_db"

if not os.path.exists(persist_directory):  # First run: load and persist
    loader = WebBaseLoader(
        web_path=urls,
        bs_kwargs=dict(parse_only=bs4.SoupStrainer())
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory="db")

else:  # Reload if already exists
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

retriever = vectordb.as_retriever()

# ----------------- Prompt Templates -----------------
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ]
)

qa_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# ----------------- Chat History Store -----------------
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# ----------------- FastAPI App -----------------
app = FastAPI()

# Enable CORS so React/WordPress can call it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev, you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    session_id: str | None = "default_session"   # ✅ optional with default value

@app.post("/ask")
def ask_question(query: QueryRequest):
    session_id = query.session_id or "default_session"  # ✅ fallback
    response = conversational_rag_chain.invoke(
        {"input": query.question},
        config={"configurable": {"session_id": session_id}}
    )
    return {"answer": response["answer"]}

