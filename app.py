from flask import Flask, request, jsonify
from transformers import pipeline
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from pdf2image import convert_from_path
import pytesseract
import os

app = Flask(__name__)

# Function to process the PDF and extract text
def extract_text_from_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

# Load the QA pipeline
hf_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", device=0)
retriever = None

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        
        # Process the PDF
        context_text = extract_text_from_pdf(file_path)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        documents = splitter.create_documents([context_text])
        global retriever
        retriever = FAISS.from_documents(documents, embeddings).as_retriever()

        return jsonify({"message": "File processed successfully"}), 200

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.json.get("question")
    if retriever is None:
        return jsonify({"error": "Upload a document first"}), 400
    prompt = PromptTemplate(template="{context}\n\nQuestion: {query}\nAnswer:", input_variables=["context", "query"])
    chain = RetrievalQA.from_chain_type(llm=hf_pipeline, retriever=retriever, chain_type="stuff", chain_type_kwargs={"prompt": prompt})
    answer = chain.run({"query": question})
    return jsonify({"answer": answer}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
