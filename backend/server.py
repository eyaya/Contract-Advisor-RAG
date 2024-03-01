from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader 
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename #pip install Werkzeug
import os
from dotenv import load_dotenv,find_dotenv
import sys
sys.path.append('../')
from RAG.retriever import RAGPipeline, create_rag_chain, ChatModel


app = Flask(__name__)
CORS(app)


GPT_MODEL_NAME = 'gpt-3.5-turbo'
vector_db_path = "../data/chroma_db/"
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'docx', 'doc'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

global parent_document_retriever

def load_and_split_document(data_path):
        """Loads and splits the document into pages."""
        try:
             loader = DirectoryLoader(data_path, glob="./*.pdf")
             documents = loader.load()
             return documents
        except Exception as e:
             return f'An error occurred {e}',400
        
def create_index():
    upload_dir = 'uploads'
    try:
        rag_pipeline = RAGPipeline(upload_dir, vector_db_path)
        documents = load_and_split_document(upload_dir)
        embeddings = rag_pipeline.create_embeddings()
        return rag_pipeline.setup_vector_database(documents,embeddings)  
    except Exception as e:
        return f'An error occurred {e}',400

def qa_chain():
    # retrive open ai key
    try:
        chat_model = ChatModel().initialize_chat_model()
        data = request.get_json()
        user_question = data.get('prompt')
        qa_chain = create_rag_chain(chat_model, parent_document_retriever)
        chat_history=[]
        response_node = qa_chain({"question": user_question, "chat_history": chat_history})
        return jsonify({'result':  response_node['answer']})

    except Exception as e:
        return jsonify({'error':  f"An error occurred: {e}"})
    
@app.route('/')
def hello_world():
    data_path='./uploads/'
    return load_and_split_document(data_path)             

@app.route('/upload_file', methods=['POST'])
def upload_file():
    global parent_document_retriever

    if 'file' not in request.files:
        resp = jsonify({
            "message": 'No file part in the request',
            "status": 'failed'
        })
        resp.status_code = 400
        return resp
    
    file = request.files['file']
    error ={}
    success = False
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_dir = 'uploads'  # Specify the directory where you want to save uploaded files
        os.makedirs(upload_dir, exist_ok=True)
        file.save(os.path.join(upload_dir, filename))
        parent_document_retriever = create_index()
        success = True
    else:
        resp = jsonify({
            "message": 'File type is not allowed',
            "status": 'failed'
            })
        return resp
    
    if success and error:
        error['message'] = 'File(s) successfully uploaded'
        error['status'] = 'failed'
        resp = jsonify(error)
        resp.status_code = 500
        return resp

    if success:
        resp = jsonify({'message': 'File uploaded successfully'})
        resp.status_code = 201
        return resp
    else:
        resp = jsonify(error)
        resp.status_code = 500
        return resp

@app.route('/ask_ai', methods=['POST'])
def query_endpoint():
    response = qa_chain()
    return response
     

if __name__ == '__main__':
    app.run(debug=True)