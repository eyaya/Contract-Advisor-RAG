from io import BytesIO

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
#from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
#from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.prompts import ChatPromptTemplate

from langchain.prompts import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from operator import itemgetter
import os
import sys
import re

sys.path.append('../')


def craft_propmt():
    template = """You are a legal expert tasked with acting as the best lawyer and contract analyzer. Your task is to thoroughly understand the provided context and answer questions related to legal matters, contracts, and relevant laws. You are also cabable of computing and compairing currency values. 
    You must provide accurate responses based solely on the information provided in the context. If the question can be answered as either yes or no, respond with either "Yes." or "No." first and include the explanation in your response.:

    ### CONTEXT
    {context}

    ### QUESTION
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    return prompt
def create_retrieval_qa_chain(chat_model, retriver):
    """Creates a retrieval QA chain combining model and database."""

    system_template = """You are the ultimate legal authority, charged with the responsibility of embodying the pinnacle of legal expertise as both a formidable lawyer and a meticulous contract analyst. Your mission is to delve deep into the provided context, comprehending it to its fullest extent, and adeptly address inquiries pertaining to legal intricacies, contracts, and pertinent legislation. Your prowess extends to the realm of financial computation, enabling you to accurately evaluate and compare currency values. 
    Your responses must be unwaveringly precise, drawing solely from the information given in the context. When faced with queries admitting of a binary response, assertively reply with either "Yes." or "No." only. Please use the following context only:

    ### CONTEXT
    {context}

    ### QUESTION
    Question: {question}
    """
        
    user_template = "Question:```{question}```"
    messages = [
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(user_template)
    ]
    qa_prompt = ChatPromptTemplate.from_messages( messages )
    
    llm = chat_model
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriver,
        chain_type='stuff',
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )
    return conversation_chain

# Asks a question and retrieves the answer.
def ask_question_and_get_answer(qa_chain, question):
    """Asks a question and retrieves the answer."""
    return qa_chain({"question": question})['answer']
