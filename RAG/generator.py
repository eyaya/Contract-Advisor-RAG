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
import chromadb
from langchain.chains.question_answering import load_qa_chain
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from operator import itemgetter
import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)
import sys
sys.path.append('../')
from RAG.retriver import load_retriever

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
data_path = '../data/contract_data'
CHROMA_PATH = '../data/chromadb/'

openai.api_key = os.environ['OPENAI_API_KEY']

openai_client = OpenAI()


template = """You are a legal expert tasked with acting as the best lawyer and contract analyzer. Your task is to thoroughly understand the provided context and answer questions related to legal matters, contracts, and relevant laws. You must provide accurate responses based solely on the information provided in the context. If the necessary information is not present in the context, respond with "I don't know.".
If the question can be answered as either yes or no, respond with either "Yes." or "No." first and include the explanation in your response.:

### CONTEXT
{context}

### QUESTION
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

embedding_function = SentenceTransformerEmbeddingFunction()#OpenAIEmbeddings()
def get_context(data_path,CHROMA_PATH):
    context = create_vectorstore(data_path,CHROMA_PATH)
    
    return context


'''

def create_qa_chain(retriever):
  primary_qa_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
  created_qa_chain = (
    {"context": itemgetter("question") | retriever,
     "question": itemgetter("question")
    }
    | RunnablePassthrough.assign(
        context=itemgetter("context")
      )
    | {
         "response": prompt | primary_qa_llm,
         "context": itemgetter("context"),
      }
  )

  return created_qa_chain
'''
    
