import argparse
from dataclasses import dataclass
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.vectorstores.chroma import Chroma
import os
from langchain_openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import sys



OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

CHROMA_PATH = '../data/chromadb/'

client = OpenAI(
    api_key=OPENAI_API_KEY
)

core_embeddings_model = OpenAIEmbeddings()
def get_context(chroma_path):
    # instantiate a retriever
    vectorstore = Chroma(persist_directory=chroma_path,embedding_function=core_embeddings_model)
    
    retriever = vectorstore.as_retriever()
    #search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.85}
    return retriever

def generate_answer(context):

    system_template = """
        Your task is to answer the question from the given context below. 
        ----
        ### CONTEXT:
        {context}
        \n
        ### # QUESTION:
        {question}
        <bot>:
        """
        
    user_template = "Question:```{question}```"
    messages = [
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(user_template)
    ]
    qa_prompt = ChatPromptTemplate.from_messages( messages )
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=context,
        chain_type='stuff',
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )
    return conversation_chain
    
