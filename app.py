from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
import tempfile

app = Flask(__name__)

embedding_model = "all-MiniLM-L6-v2"
llm_model = "google/flan-t5-base"  # keep base for GPU safety

embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
tokenizer = AutoTokenizer.from_pretrained(llm_model)
model = AutoModelForSeq2SeqLM.from_pretrained(llm_model)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
llm = HuggingFacePipeline(pipeline=pipe)

state = {}

@app.route("/upload-pdf", methods=["POST"])
def upload_pdf():
    session_id = request.form.get("session_id")
    file = request.files["pdf"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        loader = PyPDFLoader(tmp.name)
        pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    vectordb = FAISS.from_documents(chunks, embeddings)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
    state[session_id] = qa_chain

    return jsonify({"message": "✅ PDF uploaded and indexed!"})

@app.route("/ask-question", methods=["POST"])
def ask_question():
    session_id = request.form.get("session_id")
    question = request.form.get("question")

    if session_id not in state:
        return jsonify({"error": "⚠️ Please upload a PDF first."}), 400

    result = state[session_id]({"query": question})
    return jsonify({"answer": result["result"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
