from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

loader = PyMuPDFLoader("iima_hr_polilcy.pdf")
docs = loader.load()
docs[0]

splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)

chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectordb = FAISS.from_documents(chunks,embeddings)

vectordb.save_local("faiss_index")

print(f"Stored {len(chunks)} chunks in FAISS!")
