import streamlit as st
import time
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Initialize session state
if 'db' not in st.session_state:
    st.session_state.db = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'retrieval_chain' not in st.session_state:
    st.session_state.retrieval_chain = None

# Function to initialize the vector database
def initialize_vector_db():
    faiss_index_path = "faiss_index"
    ollama_emb = OllamaEmbeddings(model="llama2:7b")

    if os.path.exists(faiss_index_path):
        st.info("Loading existing FAISS index...")
        st.session_state.db = FAISS.load_local(faiss_index_path, ollama_emb, allow_dangerous_deserialization=True)
    else:
        st.info("FAISS index not found, creating embeddings...")
        pdf_loader = PyPDFLoader(r"D:\Generative AI for Legal Aids\Prototype 2\data\bns.pdf")
        docs = pdf_loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(docs)

        st.session_state.db = FAISS.from_documents(documents=documents, embedding=ollama_emb)
        
        st.session_state.db.save_local(faiss_index_path)
        st.success("FAISS index created and saved locally.")

    st.session_state.llm = OllamaLLM(model="llama2:7b")
    prompt = ChatPromptTemplate.from_template("""
    You are an Expert Legal Advisor and you have to answer Bhartiya Nyay Sanhita Sections. Don't use Indian Penal Code (IPC) sections.
    Answer the following question based only on the provided context. 
    Think step by step before providing a detailed answer. 
    I will tip you $1000 if the user finds the answer helpful. 
    <context>
    {context}
    </context>
    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(st.session_state.llm, prompt)
    retriever = st.session_state.db.as_retriever()
    st.session_state.retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Function to rephrase query
def rephrase_query(query):
    rephrase_prompt = PromptTemplate.from_template("""
    You are an expert legal translator,senior lawyer with having 20 years of experience in Indian Courts, well-versed in the language and structure of Indian legal codes, particularly the Bhartiya Nyaya Sanhita (BNS). Your task is to transform informal descriptions of incidents or legal issues into formal, precise legal language that aligns closely with the terminology found in official legal documents.

    User Query = {query}

    Your task:
    1. Analyze the user's input to identify the core legal issues or relevant actions described.
    2. Translate the key elements of the situation into formal legal terminology, using language that would be found in the Bhartiya Nyaya Sanhita (BNS).
    3. Structure the output as a concise, single-sentence statement that encapsulates the legal essence of the situation.
    4. Ensure that the resulting statement uses terms and phrasings that are likely to appear in official legal documents, facilitating easier matching with relevant sections of the law.
    5. Avoid including specific names, dates, or locations in the output. Focus on the actions, intentions, and consequences described.
    6. Don't give me any IPC sections. 
    7. Just give me the sentence in and bridges the gap between colloquial descriptions and formal legal language.


    Example input: "A person stole my bicycle from the front yard while I was inside my house."

    Example output:  "The incident involves the unlawful taking of personal property from a person's premises without consent, constituting the crime of theft."

    Remember: Your goal is to produce a statement that bridges the gap between colloquial descriptions and formal legal language, facilitating more accurate identification of relevant legal statutes and precedents.

    Return only the rephrased sentence without any additional information or formatting.
    """)
    
    rephrase_chain = LLMChain(llm=st.session_state.llm, prompt=rephrase_prompt)
    
    rephrased_response = rephrase_chain.invoke({"query": query})

    return rephrased_response['text'].strip() 

# Function to generate answer
def generate_answer_with_rephrase(query):
    rephrased_query = rephrase_query(query)
    st.info(f"User Query: {query}")
    st.error(f"Rephrased Query: {rephrased_query}")

    result = st.session_state.db.similarity_search(rephrased_query)
    if not result:
        return "No relevant information found in the database."

    context = result[0].page_content

    response = st.session_state.retrieval_chain.invoke({"input": query, "context": context})

    return response.get('answer', 'No answer was generated.')

# Streamlit app
st.set_page_config(layout="wide")

chat_mode = st.sidebar.selectbox("Choose Option", options=["Get BNS Section Info", "FIR"])

st.title("Generative AI For Legal Aids")
st.divider()

# Initialize vector database
if st.session_state.db is None:
    with st.spinner("Initializing vector database..."):
        initialize_vector_db()
    st.success("Vector database initialized!")

prompt = st.chat_input("Tell us your Query?")

def typewriter_animation(text, speed=100):
    # container = st.empty()  
    # display_str = ""
    # for char in text:
    #     display_str += char 
    #     container.markdown(f'<div class="msg-container"><p class="big-font">{display_str}</p></div>', unsafe_allow_html=True)
    #     time.sleep(1 / speed)  
    st.success(text)

if prompt:
    st.markdown("""
        <style>
        .big-font {
            font-size: 24px !important;
        }
        .msg-container{
            padding: 10px 14px !important;
            background-color : rgb(26, 28, 36) !important;
            border-radius: 10px !important;
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.spinner("Generating response..."):
        answer = generate_answer_with_rephrase(prompt)
    
    typewriter_animation(answer, speed=10)