import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel("gemini-2.5-flash")

embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

vectordb = FAISS.load_local(
    "faiss_index", 
    embeddings, 
    allow_dangerous_deserialization=True
)

retriever = vectordb.as_retriever(search_kwargs={"k": 10})

def ask(question):
   
    docs = retriever.invoke(question)
    if not docs:
            return "System: No relevant context could be found in the PDF for this question."
    
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""You are a policy analyser. Answer ONLY from the provided context. If answer is not present, say: \"I don't know based on the document.\"
                Context:{context}
                Question:{question}"""

    response = model.generate_content(prompt)

    return response.text

my_question = "what time staffs have to work?"

llm_output = ask(my_question)

print("LLM output:")
print(llm_output)