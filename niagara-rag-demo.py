    
import os
import streamlit as st
from langsmith import traceable
from openai import OpenAI
from typing import List
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
# from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain_pinecone import PineconeVectorStore


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY")

# Set page configuration
st.set_page_config(
    page_title="Niagara AI RAG Demo",
    layout="centered"
)

# Initialize session state for current question if it doesn't exist
if "current_question" not in st.session_state:
    st.session_state.current_question = ""

# Initialize constants and clients
MODEL_PROVIDER = "openai"
MODEL_NAME = "gpt-4o-mini"
APP_VERSION = 1.0
###############################################################################
# UPDATED RAG SYSTEM PROMPT
###############################################################################
RAG_SYSTEM_PROMPT = """You are an AI assistant specializing in Niagara bottling products and documentation. 
Use only the provided retrieved context to answer questions related to Niagara bottling. 
If the context does not provide enough information, respond with "I don't have enough context." 
Keep your answers concise and maintain a professional tone.
"""

###############################################################################
# UPDATED QUERY REWRITE PROMPT 
###############################################################################
QUERY_REWRITE_PROMPT = """
You are a helpful assistant that rewrites user questions into standalone, well-formed search queries 
for a retrieval-augmented AI assistant. This assistant helps users learn about Niagara bottling products, 
services, and use cases.

When rewriting:
- Preserve the user's intent while making it more specific.
- Focus on the core question.
- Avoid greetings or casual phrasing—treat it strictly like a search query.
- Do not add any extra information not present in the original question.
"""
# Initialize clients and services
@st.cache_resource
def initialize_clients():
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    embd = OpenAIEmbeddings(model="text-embedding-3-large")
    return pc, embd

pc, embd = initialize_clients()
openai_client = OpenAI()
# Define the Pinecone index name
index_name = "niagara-cleaned-docs"


# Get vector database retriever
@st.cache_resource
def get_vector_db_retriever():
    vectorstore = PineconeVectorStore.from_existing_index(index_name, embd)
    return vectorstore.as_retriever(search_kwargs={"k": 2})

# Create retriever instance
retriever = get_vector_db_retriever()

@traceable(run_type="chain")
def retrieve_documents(question: str):
    return retriever.invoke(question)

@traceable(run_type="chain")
def generate_response(question: str, documents):
    formatted_docs = "\n\n".join(doc.page_content for doc in documents)
    messages = [
        {
            "role": "system",
            "content": RAG_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"Context: {formatted_docs} \n\n Question: {question}"
        }
    ]
    return call_openai(messages)


@traceable(
    run_type="llm",
    metadata={
        "ls_provider": MODEL_PROVIDER,
        "ls_model_name": MODEL_NAME
    }
)
def call_openai(
    messages: List[dict], model: str = MODEL_NAME, temperature: float = 0.0
) -> str:
    return openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )

@traceable(run_type="llm", name="rewrite_query")
def rewrite_query(original_question: str) -> str:
    messages = [
        {"role": "system", "content": QUERY_REWRITE_PROMPT},
        {"role": "user", "content": f"Rewrite this query for better search: {original_question}"}
    ]
    return call_openai(messages).choices[0].message.content.strip()

@traceable(run_type="chain")
def langsmith_rag(question: str):
    rewritten_query = rewrite_query(question)
    with st.spinner("Retrieving relevant documents..."):
        documents = retrieve_documents(rewritten_query)
    
    with st.spinner("Generating response..."):
        response = generate_response(question, documents)
    
    # Extract source URLs from document metadata
    sources = []
    for doc in documents:
        if hasattr(doc, 'metadata') and 'source' in doc.metadata:
            source_url = doc.metadata['source']
            if source_url not in sources:
                sources.append(source_url)
    
    return response.choices[0].message.content


with st.container():
    col1, col2, col3 = st.columns([1, 12, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>Niagara AI RAG Application</h1>", unsafe_allow_html=True)
        st.markdown("<h6 style='text-align: center;'>Ask questions about Niagara Bottling products and services</h6>", unsafe_allow_html=True)
        
with st.container():
    col1, col2, col3 = st.columns([1,12,1])
    with col2:
        # Show the image by referencing the PNG file.
        # st.image("niagara-rag-flow.png", use_container_width=True, caption="Niagara RAG Flow Architecture")
        st.image("Niagara-rag-flow.png", use_container_width=True)
        st.markdown("<p style='text-align: center;'>To get started, try one of these questions:</p>", unsafe_allow_html=True)


# Example questions as buttons
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        if st.button("What is Niagara doing to address concerns about plastic waste from bottled water?"):
            st.session_state.current_question = "What is Niagara doing to address concerns about plastic waste from bottled water?"
        if st.button("How has Niagara's product line expanded since its founding in 1963?"):
            st.session_state.current_question = "How has Niagara's product line expanded since its founding in 1963?"

    with col2:
        if st.button("What different types of water does Niagara Bottling produce and how do they differ?"):
            st.session_state.current_question = "What different types of water does Niagara Bottling produce and how do they differ?"
        if st.button("How has Niagara evolved from a family business started in 1963 to becoming North America's largest private label water bottler?"):
            st.session_state.current_question = "How has Niagara evolved from a family business started in 1963 to becoming North America's largest private label water bottler?"

# st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat interface logic
if st.session_state.current_question:
    prompt = st.session_state.current_question
    st.session_state.current_question = ""  # Clear it after use
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    response = langsmith_rag(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Text input for custom questions
prompt = st.chat_input("Ask a question about Niagara") 
if prompt:
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    response = langsmith_rag(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    
