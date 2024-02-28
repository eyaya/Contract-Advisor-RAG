from langchain_community.chat_models import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from operator import itemgetter
import sys
import os

class ChatGenerator:
    def __init__(self, chat_model, retriever):
        self.chat_model = chat_model
        self.retriever = retriever

    def create_retrieval_qa_chain(self):
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
        qa_prompt = ChatPromptTemplate.from_messages(messages)

        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.chat_model,
            retriever=self.retriever,
            chain_type='stuff',
            memory=memory,
            combine_docs_chain_kwargs={"prompt": qa_prompt}
        )
        return conversation_chain

class QuestionAsker:
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain

    def ask_question_and_get_answer(self, question):
        """Asks a question and retrieves the answer."""
        return self.qa_chain({"question": question})['answer']

# Example usage:
def main():
    # Initialize chat model and retriever
    chat_model = ChatOpenAI(openai_api_key=os.environ.get('OPENAI_API_KEY'), model_name='gpt-3.5-turbo', temperature=0.0)
    bm25_retriever = BM25Retriever(k=2)
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever], weights=[1.0])

    # Create chat generator
    chat_generator = ChatGenerator(chat_model, ensemble_retriever)
    qa_chain = chat_generator.create_retrieval_qa_chain()

    # Create question asker
    question_asker = QuestionAsker(qa_chain)

    # Example usage
    answer = question_asker.ask_question_and_get_answer("Your question here")

if __name__ == '__main__':
    main()
