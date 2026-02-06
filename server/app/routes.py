from flask import Blueprint, jsonify, request
from multi_rag_engine import MultiRAGEngine

api = Blueprint("api", __name__)

# Initialize Multi RAG Engine (singleton)
try:
    rag_engine = MultiRAGEngine()
except Exception as e:
    print(f"Error initializing Multi RAG Engine: {e}")
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
        "question": "Your question here",
        "db_type": "cliente" (default) or "distribuidor"
    }
    
    Response JSON:
    {
        "success": true/false,
        "answer": "The answer or error message",
        "db_used": "cliente"
    }
    """
    if not rag_engine:
        return jsonify(success=False, answer="RAG Engine not initialized"), 500
    
    try:
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify(success=False, answer="Missing 'question' field"), 400
        
        question = data.get("question", "").strip()
        db_type = data.get("db_type", "cliente")
        
        if not question:
            return jsonify(success=False, answer="Question cannot be empty"), 400
        
        # Validar db_type
        if db_type not in ["cliente", "distribuidor"]:
             # Opcional: retornar error o simplemente usar default
             pass 
        
        answer = rag_engine.ask(question, db_type=db_type)
        return jsonify(success=True, answer=answer, db_used=db_type), 200
        
    except Exception as e:
        return jsonify(success=False, answer=str(e)), 500


@api.route("/api/update", methods=["POST"])
def update_database():
    """Update both vector databases with new documents"""
    if not rag_engine:
        return jsonify(success=False, message="RAG Engine not initialized"), 500
    
    try:
        rag_engine.update_database()
        return jsonify(success=True, message="All databases updated successfully"), 200
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500
