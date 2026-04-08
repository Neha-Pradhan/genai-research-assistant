import streamlit as st
from app.agent import run_agent

st.title("GenAI Research Assistant")
st.caption("Ask questions about Attention, RAG, and RAGAS papers")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask something about the papers..."):
    
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = run_agent(prompt)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})