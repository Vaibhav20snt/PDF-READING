import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings         # Embeddings (Converting word2vec)
import google.generativeai as genai
from langchain.vectorstores import FAISS            # Vector embeddings
from langchain_google_genai import ChatGoogleGenerativeAI   
from langchain.chains.question_answering import load_qa_chain   # For chatting
from langchain.prompts import PromptTemplate    # for prompt
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer

from pytesseract import image_to_string # type: ignore
from pdf2image import convert_from_path # type: ignore

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE-APT-KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)     # Reading all pages using library known as PdfReader
        for page in pdf_reader.pages:
            text += page.extract_text()     # From that particular page, extract all the text
    
    return text

# After getting the text, we will divide this text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Convert all the chunk into vectors
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks,embedding = embeddings)
    # embeddings_model = SentenceTransformer("all-MiniLM-L6-v2")  # Use a well-optimized SentenceTransformer model
    # embeddings = embeddings_model.encode(text_chunks, show_progress_bar=True)
    # vector_store = FAISS.from_texts(text_chunks, embedding=embeddings) # New
    vector_store.save_local("faiss_index")
    

def get_conversational_chain():
    prompt_template = """
    You are a helpful AI assistant. Use the context provided to answer the question in detail. If the context does not include an answer, respond with:
    Sorry, I didn’t understand your question. Do you want to connect with a live agent?”
    Always ensure your answer is clear, concise, and directly addresses the query.
    
    Context: \n{context}?\n
    Question: \n{question}\n
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model = "gemini-pro", temperature = 0.3)
    
    prompt = PromptTemplate(template=prompt_template, input_variables = ["Context","Question"])
    chain = load_qa_chain(model,chain_type = "stuff", prompt = prompt)
    
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    # embeddings = SentenceTransformer("all-MiniLM-L6-v2")        # New
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    response = chain(
        {"input_documents":docs, "question":user_question}
        , return_only_outputs = True
    )
    
    print(response)
    st.write("Reply: ", response["output_text"])
    

def main():
    st.set_page_config("PDF READER")
    st.header("Chat with US")

    user_question = st.text_input("Ask any Question")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()
    
    
    
        