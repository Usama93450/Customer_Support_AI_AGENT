import os
import re
import tempfile
from typing import Annotated, Literal, Optional, TypedDict

import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Vector store: FAISS (replaces Chroma)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# LLM & prompts
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Loaders & splitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ====== ENV & Setup ======
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Embeddings
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# --------- Streamlit Page & Minimal Styling ----------
st.set_page_config(page_title="Customer Support AI (FAISS)", page_icon="🛟", layout="wide")
st.title("🛟 Customer Support AI — FAISS Runtime RAG")
st.caption("Upload docs and chat. Powered by LangGraph + Hugging Face + FAISS")

st.markdown(
    """
    <style>
      .stChatMessage .stMarkdown p { font-size: 15px; line-height: 1.6; }
      .uploaded-note { font-size: 13px; color: #6b7280; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ====== Runtime FAISS store in session (no persistence) ======
if "vs" not in st.session_state:
    st.session_state.vs = None       # FAISS index
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0   # number of chunks in the index

def ensure_retriever(k: int = 4):
    """Return a retriever if FAISS exists, else a dummy that returns []."""
    class _DummyRetriever:
        def get_relevant_documents(self, _: str):
            return []
    if st.session_state.vs is None:
        return _DummyRetriever()
    return st.session_state.vs.as_retriever(search_kwargs={"k": k})

# ====== LLM ======
endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="conversational",                 # Important for chat models via HF Inference Providers
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.2,
    max_new_tokens=512,
)
llm = ChatHuggingFace(llm=endpoint)

# ====== Helper business functions ======
def create_ticket(subject: str, body: str, severity: str = "medium") -> str:
    return f"TCK-{abs(hash((subject, body, severity))) % 10_000:04d}"

def lookup_order_status(order_id: str) -> str:
    return f"Order {order_id} is in transit and expected to arrive in 2–3 days."

# ====== LangGraph State & Nodes ======
class BotState(TypedDict):
    messages: Annotated[list, add_messages]
    route: Optional[Literal["faq","order","ticket","handoff"]]

SYSTEM = SystemMessage(content=(
    "You are a helpful Customer Support Agent. "
    "You can: (1) answer FAQs using provided context, "
    "(2) check order status if user provides an order id, "
    "(3) create support tickets on request, "
    "(4) hand off to a human if needed. "
    "Always be concise and polite."
))

classifier_prompt = ChatPromptTemplate.from_messages([
    SYSTEM,
    ("user",
     "Classify the user's last message into one of: FAQ, ORDER, TICKET, HANDOFF.\n"
     "If they ask a general question that KB might answer -> FAQ.\n"
     "If they mention an order id or shipping status -> ORDER.\n"
     "If they request escalation or report a bug needing follow-up -> TICKET.\n"
     "If angry/complex beyond policy -> HANDOFF.\n"
     "User message: {user_text}\n"
     "Return ONLY one label.")
])

def classify(state: BotState):
    user_text = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            user_text = m.content
            break
    label = llm.invoke(classifier_prompt.format_messages(user_text=user_text)).content.strip().upper()
    if "ORDER" in label:
        route = "order"
    elif "TICKET" in label or "ESCALATE" in label:
        route = "ticket"
    elif "HANDOFF" in label:
        route = "handoff"
    else:
        route = "faq"
    return {"route": route}

rag_prompt = ChatPromptTemplate.from_messages([
    SYSTEM,
    ("system", "Use the context to answer.\nContext:\n{context}"),
    ("human", "{question}")
])

def answer_faq(state: BotState):
    question = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            question = m.content
            break
    retriever = ensure_retriever()
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(d.page_content for d in docs) or "No context."
    msg = llm.invoke(rag_prompt.format_messages(context=context, question=question))
    return {"messages": [AIMessage(content=msg.content)]}

order_prompt = ChatPromptTemplate.from_messages([
    SYSTEM,
    ("user", "Extract an order id from this text if present; otherwise ask for it briefly. Text: {text}")
])

def handle_order(state: BotState):
    text = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            text = m.content
            break
    extract = llm.invoke(order_prompt.format_messages(text=text)).content
    m = re.search(r"([A-Z]{2,4}\d{3,}|#?\d{5,})", extract)
    if not m:
        return {"messages": [AIMessage(content=extract)]}
    order_id = m.group(1).lstrip("#")
    status = lookup_order_status(order_id)
    return {"messages": [AIMessage(content=status)]}

ticket_prompt = ChatPromptTemplate.from_messages([
    SYSTEM,
    ("user", "Summarize the user's issue in one line subject and 2-3 lines body. Text: {text}")
])

def handle_ticket(state: BotState):
    text = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            text = m.content
            break
    summary = llm.invoke(ticket_prompt.format_messages(text=text)).content
    lines = [ln.strip() for ln in summary.splitlines() if ln.strip()]
    subject = lines[0][:120] if lines else "Customer Issue"
    body = "\n".join(lines[1:]) or "User reported an issue."
    ticket_id = create_ticket(subject, body)
    reply = f"✅ Ticket created: **{ticket_id}**\n**Subject:** {subject}\n\n{body}"
    return {"messages": [AIMessage(content=reply)]}

def handle_handoff(state: BotState):
    return {"messages": [AIMessage(content="I’m escalating this to a human specialist. You’ll hear back shortly.")]}

# Build graph
graph = StateGraph(BotState)
graph.add_node("classify", classify)
graph.add_node("faq", answer_faq)
graph.add_node("order", handle_order)
graph.add_node("ticket", handle_ticket)
graph.add_node("handoff", handle_handoff)
graph.set_entry_point("classify")

def route_decider(state: BotState):
    return state["route"]

graph.add_conditional_edges("classify", route_decider, {
    "faq": "faq",
    "order": "order",
    "ticket": "ticket",
    "handoff": "handoff"
})
graph.add_edge("faq", END)
graph.add_edge("order", END)
graph.add_edge("ticket", END)
graph.add_edge("handoff", END)

app = graph.compile()

# ====== Document Upload & Indexing ======
st.subheader("📤 Upload Knowledge Base")
uploaded_files = st.file_uploader(
    "Upload PDF / TXT / DOCX files (multiple allowed)",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True,
)

if uploaded_files:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    all_chunks = []
    for file in uploaded_files:
        # Save to temp
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        # Pick loader by extension (mimetype can vary)
        name_lower = (file.name or "").lower()
        if name_lower.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        elif name_lower.endswith(".txt"):
            loader = TextLoader(tmp_path, encoding="utf-8")
        else:
            loader = Docx2txtLoader(tmp_path)

        docs = loader.load()
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)

    # Initialize or extend FAISS
    if all_chunks:
        if st.session_state.vs is None:
            st.session_state.vs = FAISS.from_documents(all_chunks, embeddings)
        else:
            st.session_state.vs.add_documents(all_chunks)
        st.session_state.doc_count += len(all_chunks)
        st.success(f"✅ Indexed {len(all_chunks)} chunks. Total in index: {st.session_state.doc_count}")
        st.markdown("<div class='uploaded-note'>Tip: the index is in-memory and resets on app restart.</div>", unsafe_allow_html=True)

# ====== Chat Memory ======
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ====== Chat UI ======
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

user_input = st.chat_input("Ask about your docs or request help (order status, ticket, etc.)")
if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    result = app.invoke({"messages": st.session_state.chat_history})
    st.session_state.chat_history.extend(result["messages"])
    # Re-render
    st.rerun()
