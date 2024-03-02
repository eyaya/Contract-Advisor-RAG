from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredWordDocumentLoader

from langchain_community.document_loaders import TextLoader

import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
import logging

from dotenv import load_dotenv, find_dotenv
import os
import re

load_dotenv()
GPT_MODEL_NAME = 'gpt-3.5-turbo'


load_dotenv(find_dotenv())

class DocumentProcessor:
    def __init__(self, chunk_size=600, chunk_overlap=10):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_and_split_document(self, uploaded_files):
        """Loads and splits the document into pages."""
        if uploaded_files is not None:
            documents = []
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(file.read())
                    filename = file.name
                    if filename.endswith('.pdf'):
                        loader = PyPDFLoader(tmp_file.name)
                        documents.extend(loader.load())
                    elif filename.endswith('.docx') or filename.endswith('.doc'):
                        loader = UnstructuredWordDocumentLoader(tmp_file.name)
                        documents.extend(loader.load())
                    elif filename.endswith('.txt'):
                        loader = TextLoader(tmp_file.name)
                        documents.extend(loader.load())
            return documents
        return None

    def split_text_into_chunks(self, pages):
        """Splits text into smaller chunks for processing."""
        for page in pages:
            page.page_content = self._remove_special_characters(page.page_content)
            page.page_content = re.sub(r'\s+', ' ', page.page_content)

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " ", ""],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        documents = text_splitter.split_documents(pages)
        return documents

    def _remove_special_characters(self, text):
        # Define a regex pattern to match the special characters
        pattern = r'["\t●\n\[\]]'
        # Use re.sub() to replace matches of the pattern with an empty string
        cleaned_string = re.sub(pattern, ' ', text)
        return cleaned_string

class EmbeddingsCreator:
    def create_embeddings(self):
        """Creates embeddings from text."""
        return OpenAIEmbeddings()

class VectorDatabase:
    def __init__(self,docs,embeddings, vector_db_path):
        self.docs = docs
        self.embeddings = embeddings
        self.vector_db_path = vector_db_path

    def setup_vector_database(self):
        """Sets up a vector database for storing embeddings."""
        # parent 600 and child 2000 for Robson document
        # parent 600 and child 2000 for Robson document

        parent_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " "],
            chunk_size=1000,
            chunk_overlap=10
        )
        child_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " "],
            chunk_size=500,
            chunk_overlap=10
        )

        vector_store = Chroma(collection_name="contract", embedding_function=self.embeddings, persist_directory=self.vector_db_path)
        store = InMemoryStore()

        parent_document_retriever = ParentDocumentRetriever(
            vectorstore=vector_store,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            search_kwargs={"k": 10},
            
        )
        parent_document_retriever.add_documents(self.docs)
        
        return parent_document_retriever

class ChatModel:
    def initialize_chat_model(self,model_name=GPT_MODEL_NAME):
        """Initializes the chat model with specified AI model."""
        return ChatOpenAI(model_name=model_name, temperature=0.0)

class ChatPrompt:
    @staticmethod
    def craft_prompt():
        template = """You are a legal expert tasked with acting as the best lawyer and contract analyzer. Your task is to thoroughly understand the provided context and answer questions related to legal matters, contracts, and relevant laws. You are also capable of computing and comparing currency values. 
        You must provide accurate responses based solely on the information provided in the context. If the question can be answered as either yes or no, respond with either "Yes." or "No." first and include the explanation in your response.:

        ### CONTEXT
        {context}

        ### QUESTION
        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)
        return prompt

class ConversationChain:
    def create_retrieval_qa_chain(self, chat_model, retriever):
        """Creates a retrieval QA chain combining model and database."""
        system_template = """You are a legal expert tasked with acting as the best lawyer and contract analyzer. Your task is to thoroughly understand the provided context and answer questions related to legal matters, contracts, and relevant laws. If the necessary information is not present in the context use the given context, then get related contexts and answer the question. If the question cannot be answered, respond with "I don't know.".
        If the question can be answered as either yes or no, respond with either "Yes," or "No," and include the explanation in your response. In addition, please include the referenced sections in your response.
        
        You must provide accurate responses based solely on the information provided in the context only. Please use the following context only:

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
        qa_prompt = ChatPromptTemplate.from_messages(messages)

        llm = chat_model
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            chain_type='stuff',
            memory=memory,
            combine_docs_chain_kwargs={"prompt": qa_prompt}
        )
        return conversation_chain

class RAGPipeline:
    def __init__(self,
                 uploaded_file,
                 vector_db_path,
                 ):
        self.uploaded_file = uploaded_file
        self.vector_db_path = vector_db_path
    def load_and_split_document(self, uploaded_files):
        """Loads and splits the document into pages."""
        if uploaded_files is not None:
            documents = []
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(file.read())
                    filename = file.name
                    if filename.endswith('.pdf'):
                        loader = PyPDFLoader(tmp_file.name)
                        documents.extend(loader.load())
                    elif filename.endswith('.docx') or filename.endswith('.doc'):
                        loader = UnstructuredWordDocumentLoader(tmp_file.name)
                        documents.extend(loader.load())
                    elif filename.endswith('.txt'):
                        loader = TextLoader(tmp_file.name)
                        documents.extend(loader.load())
            return documents
        return None
    def create_embeddings(self):
        """Creates embeddings from text."""
        return OpenAIEmbeddings()
    def initialize_chat_model(self,model_name=GPT_MODEL_NAME):
        """Initializes the chat model with specified AI model."""
        return ChatOpenAI(model_name=model_name, temperature=0.0)
    def setup_vector_database(self, docs, embeddings):
        """Sets up a vector database for storing embeddings."""
        import shutil

        directory_path = '/path/to/directory'

        if os.path.exists(self.vector_db_path):
            print("Directory exists")
            shutil.rmtree(self.vector_db_path)
            print("Directory removed successfully")  

        try:
            parent_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ".", " "],
                chunk_size=1000,
                chunk_overlap=10
            )
            child_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ".", " "],
                chunk_size=400,
                chunk_overlap=10
            )


            vector_store = Chroma(collection_name="contract", embedding_function=embeddings, persist_directory=self.vector_db_path)
            store = InMemoryStore()

            parent_document_retriever = ParentDocumentRetriever(
                vectorstore=vector_store,
                docstore=store,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
                search_kwargs={"k": 10},
                
            )
            parent_document_retriever.add_documents(docs)
            print('Vector store created')
            
            return parent_document_retriever
        except Exception as e:
            print("VectorDatabase error: " + str(e))
            return None
    
    


def create_rag_chain(chat_model, retriever):
    """Creates a retrieval QA chain combining model and database."""
    try:
        """Creates a retrieval QA chain combining model and database."""
        system_template = """You are a legal expert tasked with acting as the best lawyer and contract analyzer. Your task is to thoroughly understand the provided context and answer questions related to legal matters, contracts, and relevant laws. If the necessary information is not present in the context use the given context, then get related contexts and answer the question. If the question cannot be answered, respond with "I don't know.".
        If the question can be answered as either yes or no, respond with either "Yes," or "No," and include the explanation in your response. In addition, please include the referenced sections in your response.
        
        You must provide accurate responses based solely on the information provided in the context only. Please use the following context only:

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
        qa_prompt = ChatPromptTemplate.from_messages(messages)

        llm = chat_model
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            chain_type='stuff',
            memory=memory,
            combine_docs_chain_kwargs={"prompt": qa_prompt}
        )
        return conversation_chain
        
    except Exception as e:
        print("RAGPipeline error: " + str(e))
        return None
