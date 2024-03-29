import streamlit as st
from htmlTemplates import css,bot2_template, user3_template
from io import BytesIO
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

import sys
sys.path.append('../')

from RAG.retriever import RAGPipeline, create_rag_chain

GPT_MODEL_NAME = 'gpt-3.5-turbo'
CHUNK_SIZE = 600
CHUNK_OVERLAP = 10
#print(OPENAI_API_KEY)
vector_db_path = "../data/chroma_db/"

def main():
    st.set_page_config(page_title="Lizzy AI", page_icon=":books:")
    page_bg_img = f"""<style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://i.postimg.cc/4xgNnkfX/Untitled-design.png");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: local;
    }}
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    </style>
    """

    st.markdown(page_bg_img, unsafe_allow_html=True)

    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Contract Analyzer")
    
    st.sidebar.subheader("Your Documents")
    uploaded_files_d = st.sidebar.file_uploader("Upload contract", type=['txt','pdf','doc','docx'],accept_multiple_files= True)
    
    
    if uploaded_files_d is not None:
        with st.spinner('Loading and processing the document...'):
            rag_pipeline = RAGPipeline(uploaded_files_d, vector_db_path)
            documents = rag_pipeline.load_and_split_document(uploaded_files_d)
            embeddings = rag_pipeline.create_embeddings()
            chat_model = rag_pipeline.initialize_chat_model()
            parent_document_retriever = rag_pipeline.setup_vector_database(documents,embeddings)
            
            
            st.success("Document processed successfully!")

        
    #rag_pipeline = RAGPipeline(uploaded_file, vector_db_path)
    qa_chain = create_rag_chain(chat_model, parent_document_retriever)
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
                st.write(user3_template.replace("{{MSG}}", st.session_state["past"][i]), unsafe_allow_html=True)
                st.write(bot2_template.replace("{{MSG}}", st.session_state["generated"][i]), unsafe_allow_html=True)    
    
if __name__ == '__main__':
    main()
