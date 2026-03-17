import os
import shutil
from fastapi import FastAPI, UploadFile, Form, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import rag_core

app = FastAPI(title="Multimodal RAG API")

# CRITICAL FIX: allow_credentials=True cannot be used with allow_origins=["*"]
# Changed allow_credentials to False to safely allow all origins to connect.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False, 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure temp directory exists for file uploads
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

class TextUpload(BaseModel):
    name: str
    text: str

class Query(BaseModel):
    question: str
    top_k: int = 3

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serves the main frontend UI."""
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/upload/text")
async def upload_text(payload: TextUpload):
    """Endpoint to upload raw text."""
    try:
        chunks_added = rag_core.add_text_document(payload.name, payload.text)
        return {"status": "success", "message": f"Added {chunks_added} text chunks.", "total_vectors": rag_core.faiss_index.ntotal}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload/file")
async def upload_file(file: UploadFile = File(...)):
    """Endpoint to upload PDFs, Images, Audio, and Video files."""
    file_path = os.path.join(TEMP_DIR, file.filename)
    
    # Save the uploaded file temporarily
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Process the file based on its type using the RAG core
        items_added = rag_core.add_media_document(file_path)
        
        return {
            "status": "success", 
            "message": f"Processed {file.filename} into {items_added} records.",
            "total_vectors": rag_core.faiss_index.ntotal
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/api/ask")
async def ask_question(query: Query):
    """Endpoint to ask a question to the RAG system."""
    try:
        result = rag_core.ask_rag(query.question, top_k=query.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """Returns database stats."""
    return {"total_documents_indexed": rag_core.faiss_index.ntotal}

if __name__ == "__main__":
    import uvicorn
    # Run the application
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
