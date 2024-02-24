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

from src.utils import remove_special_characters
# Constants and API Keys
#sOPENAI_API_KEY = "your_openai_api_key"  # Replace with your actual API key
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GPT_MODEL_NAME = 'gpt-3.5-turbo'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 10

def load_and_split_document(uploaded_file):
    """Loads and splits the document into pages."""
    temp_file = "./temp.pdf"
    if uploaded_file is not None:
        with open(temp_file, "wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name

        loader = PyPDFLoader(temp_file)
        documents = loader.load_and_split()
        return documents
    return None

def split_text_into_chunks(pages, chunk_size, chunk_overlap):
    """Splits text into smaller chunks for processing."""
    for page in pages:
        page.page_content = remove_special_characters(page.page_content)
        page.page_content=re.sub(r'\s+', ' ',page.page_content )
    print(pages[0])
    print("\n\n",pages[1])
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n","\n",".", " ",""],chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(pages)
    print('\n\n',documents[0])
    print('\n\n',documents[1])
    return documents

def create_embeddings(api_key):
    """Creates embeddings from text."""
    return OpenAIEmbeddings(openai_api_key=api_key)

def setup_vector_database(vectordb_path,docs, embeddings):
    """Sets up a vector database for storing embeddings."""
    
    parent_splitter = RecursiveCharacterTextSplitter(separators=["\n\n","\n",".", " ",""],chunk_size=500,chunk_overlap=10)
    child_splitter = RecursiveCharacterTextSplitter(separators=["\n\n","\n",".", " "],chunk_size=2000,chunk_overlap=20)

    vectorstore = Chroma(collection_name="contract", embedding_function=OpenAIEmbeddings())

    store = InMemoryStore()

    parent_document_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    parent_document_retriever.add_documents(docs)
    return parent_document_retriever
    
    '''
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    #b_docs = text_splitter.split_documents(docs)
    vectordb =  Chroma(persist_directory=vectordb_path,embedding_function=embeddings)
    return vectordb.as_retriever(search_type="similarity",search_kwargs={'k': 10})
    '''
    '''
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    b_docs = text_splitter.split_documents(docs)
    bm25_retriever = BM25Retriever.from_documents(b_docs)

    bm25_retriever.k = 2

    embedding = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(docs, embedding)
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, chroma_retriever], weights=[0.75, 0.25])
    return ensemble_retriever
    '''
    
    
def initialize_chat_model(api_key, model_name):
    """Initializes the chat model with specified AI model."""
    return ChatOpenAI(openai_api_key=api_key, model_name=model_name, temperature=0.0)
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

'''
    memory = ConversationBufferWindowMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(chat_model, retriever=vector_database, memory=memory)
'''
def ask_question_and_get_answer(qa_chain, question):
    """Asks a question and retrieves the answer."""
    return qa_chain({"question": question})['answer']
