from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from dotenv import load_dotenv
import os
import re

load_dotenv()

class DocumentProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=10):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_and_split_document(self, uploaded_file):
        """Loads and splits the document into pages."""
        temp_file = "./temp.pdf"
        if uploaded_file is not None:
            with open(temp_file, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

            loader = PyPDFLoader(temp_file)
            documents = loader.load_and_split()
            os.remove(temp_file)
            return documents
        return None

    def split_text_into_chunks(self, pages):
        """Splits text into smaller chunks for processing."""
        for page in pages:
            page.page_content = self._remove_special_characters(page.page_content)
            #page.page_content = re.sub(r'\s+', ' ', page.page_content)

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
    def create_embeddings(self, api_key):
        """Creates embeddings from text."""
        return OpenAIEmbeddings()

class VectorDatabase:
    def __init__(self, vectordb_path):
        self.vectordb_path = vectordb_path

    def setup_vector_database(self, docs, embeddings):
        """Sets up a vector database for storing embeddings."""


        parent_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " "],
            chunk_size=600,
            chunk_overlap=10
        )
        child_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " "],
            chunk_size=1900,
            chunk_overlap=10
        )

        vectorstore = Chroma(collection_name="contract", embedding_function=OpenAIEmbeddings())
        store = InMemoryStore()

        parent_document_retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            search_kwargs={"k": 10},
        )
        parent_document_retriever.add_documents(docs)
        return parent_document_retriever

class ChatModel:
    def initialize_chat_model(self, api_key, model_name):
        """Initializes the chat model with specified AI model."""
        return ChatOpenAI(openai_api_key=api_key, model_name=model_name, temperature=0.0)

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

