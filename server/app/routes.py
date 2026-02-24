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


import os
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'txt', 'md'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@api.route("/api/upload", methods=["POST"])
def upload_document():
    """
    Upload a new document to the RAG system.
    
    Form Data:
        file: The .txt or .md file to upload
        target: "distribuidor" or "cliente"
    
    Response:
        {
            "success": true,
            "message": "File uploaded and indexed",
            "filename": "doc.txt",
            "target": "distribuidor",
            "chunks_added": 5
        }
    """
    if not rag_engine:
        return jsonify(success=False, error="RAG Engine not initialized"), 500
    
    # Validar que hay archivo
    if 'file' not in request.files:
        return jsonify(success=False, error="No file provided"), 400
    
    file = request.files['file']
    
    # Validar nombre de archivo
    if file.filename == '':
        return jsonify(success=False, error="No file selected"), 400
    
    # Validar extensión
    if not allowed_file(file.filename):
        return jsonify(success=False, error="Invalid file extension. Only .txt and .md allowed"), 400
    
    # Obtener target
    target = request.form.get('target', '').lower()
    if target not in ['distribuidor', 'cliente']:
        return jsonify(success=False, error="Invalid target. Must be 'distribuidor' or 'cliente'"), 400
    
    try:
        # Guardar archivo
        filename = secure_filename(file.filename)
        
        # Determinar directorio según target
        if target == 'distribuidor':
            save_dir = 'dataSet'
        else:  # cliente
            save_dir = 'dataSet/datasetGen'
        
        # Asegurar que directorio existe
        os.makedirs(save_dir, exist_ok=True)
        
        file_path = os.path.join(save_dir, filename)
        file.save(file_path)
        
        # Agregar a la base de datos
        result = rag_engine.add_document_to_target(file_path, target)
        
        return jsonify(
            success=True,
            message="File uploaded and indexed successfully",
            filename=filename,
            target=target,
            chunks_added=result['chunks_added']
        ), 200
        
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


@api.route("/api/delete", methods=["POST"])
def delete_document():
    """
    Delete a document from the RAG system (and filesystem).
    
    JSON Body:
        {
            "filename": "documento.md",
            "target": "distribuidor" or "cliente"
        }
    
    Response:
        {
            "success": true,
            "message": "File deleted successfully",
            "filename": "documento.md",
            "target": "distribuidor"
        }
    """
    if not rag_engine:
        return jsonify(success=False, error="RAG Engine not initialized"), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify(success=False, error="Invalid JSON body"), 400
            
        filename = data.get("filename")
        if not filename:
            return jsonify(success=False, error="Missing 'filename' field"), 400
            
        target = data.get("target")
        if not target:
            return jsonify(success=False, error="Missing 'target' field"), 400
            
        if target not in ['distribuidor', 'cliente']:
            return jsonify(success=False, error="Invalid target. Must be 'distribuidor' or 'cliente'"), 400
            
        # Ejecutar borrado
        # Esto borrará el archivo físico y los vectores
        rag_engine.delete_document_from_target(filename, target)
        
        return jsonify(
            success=True,
            message="File deleted successfully",
            filename=filename,
            target=target
        ), 200
        
    except FileNotFoundError:
        return jsonify(success=False, error="File not found in target directory"), 404
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500
