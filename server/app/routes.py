from flask import Blueprint, jsonify, request
from rag_engine import RAGEngine

api = Blueprint("api", __name__)

# Initialize RAG Engine (singleton)
try:
    rag_engine = RAGEngine(data_dir="dataSet")
except Exception as e:
    print(f"Error initializing RAG Engine: {e}")
    rag_engine = None


@api.route("/", methods=["GET"])
def health():
    """Health check endpoint"""
    status = "ok" if rag_engine else "error"
    return jsonify(status=status)


@api.route("/api/ask", methods=["POST"])
def ask():
    """Ask a question to the RAG system
    
    Request JSON:
    {
        "question": "Your question here"
    }
    
    Response JSON:
    {
        "success": true/false,
        "answer": "The answer or error message"
    }
    """
    if not rag_engine:
        return jsonify(success=False, answer="RAG Engine not initialized"), 500
    
    try:
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify(success=False, answer="Missing 'question' field"), 400
        
        question = data.get("question", "").strip()
        if not question:
            return jsonify(success=False, answer="Question cannot be empty"), 400
        
        answer = rag_engine.ask(question)
        return jsonify(success=True, answer=answer), 200
        
    except Exception as e:
        return jsonify(success=False, answer=str(e)), 500


@api.route("/api/update", methods=["POST"])
def update_database():
    """Update the vector database with new documents"""
    if not rag_engine:
        return jsonify(success=False, message="RAG Engine not initialized"), 500
    
    try:
        rag_engine.update_database()
        return jsonify(success=True, message="Database updated successfully"), 200
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500
