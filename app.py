# app.py
import streamlit as st
from core.agent import run_agent
from core.memory import clear_history
from ingest.loader import ingest_file


st.set_page_config(
    page_title="Python Docs Assistant", 
    page_icon="🐍", 
    layout="centered"
)

# Center title with CSS
st.markdown("""
    <style>
        .block-container { padding-top: 3rem; }
        h1 { text-align: center; }
        .stCaption { text-align: center; }
    </style>
""", unsafe_allow_html=True)


st.title("Python Docs Assistant") 
st.markdown("<p style='text-align: center; color: gray;'>Ask anything about Python — powered by local AI and official documentation.</p>", unsafe_allow_html=True) 

# Sidebar
with st.sidebar:
    st.header("Settings")
    if st.button("Clear Conversation", use_container_width=True):
        clear_history()
        st.session_state.messages = []
        st.success("Conversation cleared!")

    uploaded = st.file_uploader(
        "Add documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Drag files here — the bot learns them instantly.",
    )
    if uploaded:
        with st.spinner("Reading documents..."):
            total: int = 0
            for file in uploaded:
                total += ingest_file(file)
        st.success(f"{len(uploaded)} document(s) added ({total} chunks).")

    st.markdown("---")
    st.markdown("**Model:** qwen2.5:14b")
    st.markdown("**Knowledge base:** Python 3 Docs") 
    st.markdown("**Vector DB:** ChromaDB")
    st.markdown("**Memory:** SQLite")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a Python question..."): 
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching docs and thinking..."):
            response: str = run_agent(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})




#starting with streamlit run app.py 