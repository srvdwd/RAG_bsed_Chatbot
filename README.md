# HR Policy RAG Chatbot

An AI-powered HR Policy Chatbot built using Retrieval-Augmented Generation (RAG), FAISS Vector Database, HuggingFace Embeddings, and Google Gemini LLM.

The chatbot reads HR policy PDF documents, converts them into vector embeddings, stores them in a FAISS vector database, and answers user questions using context-aware semantic retrieval.

---

# Features

* PDF-based HR policy question answering
* Retrieval-Augmented Generation (RAG)
* Semantic search using embeddings
* FAISS vector database for fast retrieval
* Google Gemini LLM integration
* Context-grounded responses
* Streamlit web interface
* Hallucination reduction using prompt engineering

---

# Tech Stack

* Python
* LangChain
* FAISS
* HuggingFace Embeddings
* Google Gemini API
* Streamlit
* PyMuPDF

---

# Project Architecture

```text
PDF Documents
      ↓
Document Loader
      ↓
Text Chunking
      ↓
Embedding Generation
      ↓
FAISS Vector Database
      ↓
Semantic Retrieval
      ↓
Gemini LLM
      ↓
Context-Based Answer
```

---

# Folder Structure

```text
rag-chatbot/
│
├── app.py
├── input.py
├── rag.py
├── requirements.txt
├── .env
├── faiss_index/
├── iima_hr_polilcy.pdf
└── README.md
```

---

# Installation

## 1. Clone Repository

```bash
git clone <your-github-repo-url>
cd rag-chatbot
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file in the root directory.

```env
GEMINI_API_KEY=your_api_key_here
```

Get your Gemini API key from:

https://aistudio.google.com/

---

# Requirements

Create `requirements.txt`

```text
streamlit
langchain
langchain-community
langchain-text-splitters
langchain-huggingface
sentence-transformers
faiss-cpu
google-generativeai
pymupdf
python-dotenv
torch
transformers
```

---

# Step 1: Create FAISS Vector Database

Run:

```bash
python input.py
```

This will:

* Load PDF
* Split text into chunks
* Generate embeddings
* Store vectors in FAISS index

---

# input.py

```python
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load PDF
loader = PyMuPDFLoader("iima_hr_polilcy.pdf")
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Create FAISS vector database
vectordb = FAISS.from_documents(
    chunks,
    embeddings
)

# Save locally
vectordb.save_local("faiss_index")

print(f"Stored {len(chunks)} chunks in FAISS!")
```

---

# Step 2: RAG Pipeline

## rag.py

```python
import os
from dotenv import load_dotenv

import google.generativeai as genai

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Gemini model
model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load FAISS index
vectordb = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Create retriever
retriever = vectordb.as_retriever(
    search_kwargs={"k": 10}
)

# Main chatbot function
def ask(question):

    # Retrieve relevant chunks
    docs = retriever.invoke(question)

    if not docs:
        return "No relevant context found."

    # Combine retrieved chunks
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Prompt
    prompt = f"""
You are a policy assistant.

Answer ONLY from the provided context.

If the answer is not present in the context, say:
"I don't know based on the document."

Context:
{context}

Question:
{question}
"""

    # Generate response
    response = model.generate_content(prompt)

    return response.text
```

---

# Step 3: Streamlit Frontend

## app.py

```python
import streamlit as st
from rag import ask

st.set_page_config(
    page_title="HR Policy Chatbot",
    page_icon="🤖"
)

st.title("HR Policy RAG Chatbot")
st.caption("Ask questions about HR policy documents")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
if question := st.chat_input("Ask a question..."):

    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Display user message
    with st.chat_message("user"):
        st.write(question)

    # Generate assistant response
    with st.chat_message("assistant"):

        with st.spinner("Searching documents..."):

            answer = ask(question)

        st.write(answer)

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
```

---

# Run the Chatbot

```bash
streamlit run app.py
```

---

# Example Questions

* What is the leave policy?
* What are the working hours?
* What is the maternity leave policy?
* How is performance appraisal conducted?
* What are employee benefits?

---

# Core Concepts Used

## Retrieval-Augmented Generation (RAG)

RAG combines:

* retrieval of relevant document chunks
* generation using LLMs

This improves factual accuracy and reduces hallucinations.

---

## Embeddings

Embeddings convert text into numerical vector representations that capture semantic meaning.

---

## FAISS Vector Database

FAISS stores embeddings and performs fast similarity search for semantic retrieval.

---

## Semantic Search

Semantic search retrieves information based on meaning rather than exact keyword matching.

---

# Future Improvements

* Multi-PDF support
* Source citations
* Conversational memory
* OCR support for scanned PDFs
* Docker deployment
* Authentication system
* Hybrid search (keyword + semantic)

---

# Deployment

The project can be deployed on:

* Render
* Hugging Face Spaces
* Streamlit Cloud
* AWS
* Azure

---

# Author

Developed as an AI Internship Project using RAG architecture and Large Language Models.

