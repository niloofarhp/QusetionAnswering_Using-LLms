import google.generativeai as palm
import os
import PyPDF2
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# Load and extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

# Split text into chunks
def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    return texts

# Create FAISS vector store with HuggingFace embeddings
def create_vector_store(texts):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts, embeddings)
    return vector_store

# Load Mistral-7B from HuggingFace
def initialize_llm():
    llm = HuggingFaceHub(
    repo_id= 'meta-llama/Llama-2-7b-chat-hf', #"mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.2, "max_length": 500},
    huggingfacehub_api_token="API_KEY"
)
    return llm

# Setup Q&A Retrieval Chain
def create_qa_chain(llm, vector_store):
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
    return qa_chain

# Answer user queries based on the PDF
def query_pdf(pdf_path, user_question):
    text = extract_text_from_pdf(pdf_path)
    texts = chunk_text(text)
    vector_store = create_vector_store(texts)
    llm = initialize_llm()
    qa_chain = create_qa_chain(llm, vector_store)     # 

    answer = qa_chain.run(user_question)
    return answer

# Example Usage
pdf_path = "/Users/niloofar/Documents/Projects/langchain/practitioners_guide_to_mlops_whitepaper.pdf"
question = "What is the main topic of the document?"
answer = query_pdf(pdf_path, question)

print("Answer:", answer)
