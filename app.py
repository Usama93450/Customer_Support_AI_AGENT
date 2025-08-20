import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from graph import app as graph_app, SYSTEM  # reuse compiled graph & SYSTEM message

load_dotenv()

st.set_page_config(page_title="Customer Support AI", page_icon="🛟", layout="centered")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(to bottom right, #0f172a, #1e293b); color:#e5e7eb; }
    .bubble { padding: 12px 16px; border-radius: 14px; margin: 8px 0; max-width: 90%; line-height:1.5; }
    .user { background:#3b82f6; color:white; margin-left:auto; }
    .bot  { background:#334155; color:#e5e7eb; margin-right:auto; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🛟 Customer Support AI Agent")
st.caption("RAG + LangGraph + Hugging Face")

if "thread" not in st.session_state:
    # LangGraph expects state dict
    st.session_state.thread = {"messages": [SYSTEM], "route": None}

# Display history
for m in st.session_state.thread["messages"]:
    if isinstance(m, HumanMessage):
        st.markdown(f"<div class='bubble user'>{m.content}</div>", unsafe_allow_html=True)
    elif isinstance(m, AIMessage):
        st.markdown(f"<div class='bubble bot'>{m.content}</div>", unsafe_allow_html=True)

user = st.chat_input("Describe your issue or ask a question...")
if user:
    st.session_state.thread["messages"].append(HumanMessage(content=user))
    with st.spinner("Thinking..."):
        # run one step through the graph
        result = graph_app.invoke(st.session_state.thread)
        # merge updates back to session state
        for k, v in result.items():
            if k == "messages":
                st.session_state.thread["messages"].extend(v)
            else:
                st.session_state.thread[k] = v
    st.rerun()
