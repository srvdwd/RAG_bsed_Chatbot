import streamlit as st
from rag import ask

st.set_page_config(page_title="IIMA HR Policy Assistant", page_icon="📄")
st.title("📄 IIMA HR Policy Q&A")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
if prompt := st.chat_input("Ask about HR policy..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask(prompt)
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})