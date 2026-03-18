import os
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, Form, HTTPException, File, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import rag_core as rag_core

app = FastAPI(title="Multimodal RAG API")

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

# Secure backend password configuration
APP_PASSWORD = os.getenv("APP_PASSWORD")

def verify_password(x_app_password: str = Header(None)):
    """Dependency to check the password in headers for secure routes."""
    if not x_app_password or x_app_password != APP_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid Password")

class TextUpload(BaseModel):
    name: str
    text: str

class Query(BaseModel):
    question: str
    top_k: int = 3

class DeleteRequest(BaseModel):
    names: List[str]

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serves the main frontend UI."""
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/upload/text", dependencies=[Depends(verify_password)])
async def upload_text(payload: TextUpload):
    """Endpoint to upload raw text."""
    try:
        chunks_added = rag_core.add_text_document(payload.name, payload.text)
        return {"status": "success", "message": f"Added {chunks_added} text chunks.", "total_vectors": rag_core.faiss_index.ntotal}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload/file", dependencies=[Depends(verify_password)])
async def upload_file(files: List[UploadFile] = File(...)):
    """Endpoint to upload multiple PDFs, Images, Audio, and Video files."""
    total_items = 0
    try:
        for file in files:
            file_path = os.path.join(TEMP_DIR, file.filename)
            
            # Save the uploaded file temporarily
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                
            # Process the file based on its type using the RAG core
            items_added = rag_core.add_media_document(file_path)
            total_items += items_added
            
            # Clean up immediately after processing
            if os.path.exists(file_path):
                os.remove(file_path)
                
        return {
            "status": "success", 
            "message": f"Processed {len(files)} file(s) into {total_items} records.",
            "total_vectors": rag_core.faiss_index.ntotal
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ask", dependencies=[Depends(verify_password)])
async def ask_question(query: Query):
    """Endpoint to ask a question to the RAG system using Groq."""
    try:
        result = rag_core.ask_rag(query.question, top_k=query.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents", dependencies=[Depends(verify_password)])
async def get_documents():
    """Returns a list of unique document names currently in the database."""
    docs = list(set([meta["name"] for meta in rag_core.metadata_store.values()]))
    return {"documents": docs}

@app.delete("/api/data", dependencies=[Depends(verify_password)])
async def delete_data(req: DeleteRequest):
    """Deletes specific documents by name, or all data if list is empty."""
    try:
        if not req.names:
            rag_core.delete_all()
            return {"status": "success", "message": "All database records cleared."}
        else:
            deleted_count = rag_core.delete_documents(req.names)
            return {"status": "success", "message": f"Deleted {deleted_count} chunks linked to selected files."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats", dependencies=[Depends(verify_password)])
async def get_stats():
    """Returns database stats."""
    return {"total_documents_indexed": rag_core.faiss_index.ntotal}

if __name__ == "__main__":
    import uvicorn
    # Run the application
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
