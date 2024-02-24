import streamlit as st
from htmlTemplates import css, bot_template,bot1_templete, user_template,bot2_template,user3_template
from io import BytesIO

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
#from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import os
import sys
import re


sys.path.append('../')
from RAG.retriver import (
    create_embeddings,
    load_and_split_document,
    create_retrieval_qa_chain,
    split_text_into_chunks,
    setup_vector_database,
    initialize_chat_model
    )

from src.utils import remove_special_characters
# Constants and API Keys
#sOPENAI_API_KEY = "your_openai_api_key"  # Replace with your actual API key
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GPT_MODEL_NAME = 'gpt-3.5-turbo'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 10

# Function Definitions


# Main Execution Flow

def main():
    st.set_page_config(page_title="Lizzy AI", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Contract Analyzer")
    
    st.sidebar.subheader("Your Documents")
    uploaded_file = st.sidebar.file_uploader("Upload contract", type=["pdf"])
    if uploaded_file is not None:
        with st.spinner('Loading and processing the document...'):
            pages = load_and_split_document(uploaded_file)
            documents = split_text_into_chunks(pages, CHUNK_SIZE, CHUNK_OVERLAP)
            embeddings = create_embeddings(OPENAI_API_KEY)
            vector_database = setup_vector_database('../data/chromadb',documents ,embeddings)
            chat_model = initialize_chat_model(OPENAI_API_KEY, GPT_MODEL_NAME)
            qa_chain = create_retrieval_qa_chain(chat_model, vector_database)
            st.success("Document processed successfully!")

        def handle_userinput(user_question):
            result = qa_chain({"question": user_question, "chat_history": st.session_state['history']})
            st.session_state['history'].append((qa_chain, result["answer"]))
            return result["answer"]
        # Initialize chat history
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        # Initialize messages
        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello ! Ask me about your documents 🤗"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey ! 👋"]

        # Create containers for chat history and user input
        response_container = st.container()
        container = st.container()
        # User input form
        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="Ask me about your data 👉 (:", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                with st.spinner('Finding the answer...'):
                    output = handle_userinput(user_input)
                    st.session_state['past'].append(user_input)
                    st.session_state['generated'].append(output)

        # Display chat history 
        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    
                    st.write(user3_template.replace(
                "{{MSG}}", st.session_state["past"][i]), unsafe_allow_html=True)
                    
                    st.write(bot2_template.replace(
                "{{MSG}}", st.session_state["generated"][i]), unsafe_allow_html=True)
        
if __name__ == '__main__':
    main()