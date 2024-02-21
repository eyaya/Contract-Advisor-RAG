
import sys
sys.path.append('../')
import RAG.generator as generator
import RAG.retriver as ingestion
import streamlit as st
from streamlit_chat import message
from htmlTemplates import css, bot_template,bot1_templete, user_template,bot2_template,user3_template
def main():
    #load_dotenv()
    CHROMA_PATH = '/home/eyaya/Desktop/Challenges/Week_11/RAG_Practice/data/chromadb'
    st.set_page_config(page_title="Lizzy AI, Q&A", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Contract Advisor RAG")
    
    
    context = generator.get_context(CHROMA_PATH)
    conversation_chain = generator.generate_answer(context)
    def handle_userinput(user_question):
        result = conversation_chain({"question": user_question, "chat_history": st.session_state['history']})
        st.session_state['history'].append((conversation_chain, result["answer"]))
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
    #CHROMA_PATH = '/home/eyaya/Desktop/Challenges/Week_11/RAG_Practice/data/chromadb'
    #data_path = '/home/eyaya/Desktop/Challenges/Week_11/RAG_Practice/data/contract_data'
    #ingestion.create_data_store(data_path,CHROMA_PATH)
    main()