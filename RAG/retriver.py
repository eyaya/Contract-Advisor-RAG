from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader
import os
import chromadb
import shutil
from dotenv import load_dotenv
load_dotenv()


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')




def load_documents(data_path):
    loader = DirectoryLoader(data_path)
    documents = loader.load()
    return documents

def split_text(documents:list[Document]):
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
        add_start_index = True
    )
    chunk = text_spliter.split_documents(documents)
    return chunk

def save_chunks_to_chroma(chunks:list[Document],vectordb_path):
    
    #Clear the database first
    if os.path.exists(vectordb_path):
        shutil.rmtree(vectordb_path)

    db = Chroma.from_documents(chunks, OpenAIEmbeddings(),\
                               persist_directory=vectordb_path)
    db.persist()

def create_data_store(data_path,vectordb_path):
    documents = load_documents(data_path)
    chunks = split_text(documents)
    print(chunks[1])
    save_chunks_to_chroma(chunks,vectordb_path)  
    print('vector store creadted')

