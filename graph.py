import os
from typing import Annotated, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()
PERSIST_DIR = os.getenv("CHROMA_DIR", "./chroma_kb")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --- Vector store (RAG) ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="conversational",                     # important!
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.2,
    max_new_tokens=512,
)
llm = ChatHuggingFace(llm=endpoint)
def create_ticket(subject: str, body: str, severity: str = "medium") -> str:
    # Replace with your ticket system API call
    return f"TCK-{abs(hash((subject, body, severity))) % 10_000:04d}"

def lookup_order_status(order_id: str) -> str:
    # Replace with your OMS/DB integration
    return f"Order {order_id} is in transit and expected to arrive in 2–3 days."

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

# ----- Order Node -----
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
    # naive parse: find something like #1234 or ORD123 etc. (simplified)
    import re
    m = re.search(r"([A-Z]{2,4}\d{3,}|#?\d{5,})", extract)
    if not m:
        return {"messages": [AIMessage(content=extract)]}
    order_id = m.group(1).lstrip("#")
    status = lookup_order_status(order_id)
    return {"messages": [AIMessage(content=status)]}

# ----- Ticket Node -----
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
    # naive split
    lines = [ln.strip() for ln in summary.splitlines() if ln.strip()]
    subject = lines[0][:120] if lines else "Customer Issue"
    body = "\n".join(lines[1:]) or "User reported an issue."
    ticket_id = create_ticket(subject, body)
    reply = f"✅ Ticket created: **{ticket_id}**\n**Subject:** {subject}\n\n{body}"
    return {"messages": [AIMessage(content=reply)]}

# ----- Handoff Node -----
def handle_handoff(state: BotState):
    return {"messages": [AIMessage(content="I’m escalating this to a human specialist. You’ll hear back shortly.")]}

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

# after answering, end
graph.add_edge("faq", END)
graph.add_edge("order", END)
graph.add_edge("ticket", END)
graph.add_edge("handoff", END)

app = graph.compile()