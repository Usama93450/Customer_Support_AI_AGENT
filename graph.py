import os
import tempfile
from typing import Annotated, Literal, Optional, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

# --- ENV & Setup ---
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize empty Chroma (runtime persistence in ./chroma_kb)
PERSIST_DIR = "./chroma_kb"
vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# LLM
endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="conversational",
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.2,
    max_new_tokens=512,
)
llm = ChatHuggingFace(llm=endpoint)

# ---- Support Functions ----
def create_ticket(subject: str, body: str, severity: str = "medium") -> str:
    return f"TCK-{abs(hash((subject, body, severity))) % 10_000:04d}"

def lookup_order_status(order_id: str) -> str:
    return f"Order {order_id} is in transit and expected to arrive in 2–3 days."

# ---- State ----
class BotState(TypedDict):
    messages: Annotated[list, add_messages]
    route: Optional[Literal["faq","order","ticket","handoff"]]

SYSTEM = SystemMessage(content=(
    "You are a helpful Customer Support Agent. "
    "You can: (1) answer FAQs using provided context, "
    "(2) check order status if user provides an order id, "
    "(3) create support tickets on request, "
    "(4) hand off to a human if needed.\n"
    "Always be concise and polite."
))

classifier_prompt = ChatPromptTemplate.from_messages([
    SYSTEM,
    ("user", "Classify the user's last message into one of: FAQ, ORDER, TICKET, HANDOFF.\n"
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
    import re
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

# ---- Graph ----
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

# ---- Streamlit UI ----
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📚 RAG Support Chatbot")

# File uploader
uploaded_files = st.file_uploader("Upload documents (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        if file.type == "application/pdf":
            loader = PyPDFLoader(tmp_path)
        elif file.type == "text/plain":
            loader = TextLoader(tmp_path)
        else:
            loader = Docx2txtLoader(tmp_path)
        docs = loader.load()
        vectordb.add_documents(docs)
    st.success("✅ Documents uploaded and indexed!")

# Chat memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask me something...")
if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    result = app.invoke({"messages": st.session_state.chat_history})
    st.session_state.chat_history.extend(result["messages"])

# Display chat
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)
