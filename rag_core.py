import mimetypes
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader
import groq

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file.")
if not GROQ_API_KEY:
    raise ValueError("Set GROQ_API_KEY in your .env file to enable Groq LLM answering.")

# Initialize Clients
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
groq_client = groq.Groq(api_key=GROQ_API_KEY)

EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-2-preview")
CHAT_MODEL = "gemini-2.5-flash" # Gemini used for Summarizing Images/Videos
GROQ_MODEL = "llama-3.3-70b-versatile" # Groq used for Fast RAG Q&A
OUTPUT_DIMENSIONALITY = 1536

# ---------------------------------------------------------------------------
# FAISS Vector Database Setup
# ---------------------------------------------------------------------------
# We use IndexIDMap wrapped around IndexFlatIP to allow document deletion
base_index = faiss.IndexFlatIP(OUTPUT_DIMENSIONALITY)
faiss_index = faiss.IndexIDMap(base_index)
metadata_store: Dict[int, Dict[str, Any]] = {}
current_id_counter = 0

DB_PATH = "faiss_index.bin"
META_PATH = "metadata.json"

def save_db():
    """Persist FAISS index and metadata to disk."""
    faiss.write_index(faiss_index, DB_PATH)
    with open(META_PATH, "w") as f:
        json.dump(metadata_store, f)

def load_db():
    """Load FAISS index and metadata from disk if they exist."""
    global faiss_index, metadata_store, current_id_counter
    if os.path.exists(DB_PATH) and os.path.exists(META_PATH):
        try:
            loaded_index = faiss.read_index(DB_PATH)
            # Ensure the loaded index supports IDs (IndexIDMap)
            if not isinstance(loaded_index, faiss.IndexIDMap):
                print("Old FAISS format detected. Resetting to support deletions...")
                return
            faiss_index = loaded_index
            with open(META_PATH, "r") as f:
                metadata_store = {int(k): v for k, v in json.load(f).items()}
            current_id_counter = max(metadata_store.keys()) + 1 if metadata_store else 0
        except Exception as e:
            print(f"Failed to load DB: {e}. Starting fresh.")

load_db()

def add_to_faiss(embedding: List[float], metadata: Dict[str, Any]):
    """Normalize vector and add to FAISS with a specific ID."""
    global current_id_counter
    vec = np.array([embedding], dtype=np.float32)
    faiss.normalize_L2(vec) 
    
    # Add with ID mapping
    faiss_index.add_with_ids(vec, np.array([current_id_counter], dtype=np.int64))
    metadata_store[current_id_counter] = metadata
    current_id_counter += 1
    save_db()

def delete_documents(names: List[str]) -> int:
    """Deletes documents from FAISS and metadata by their string name."""
    global metadata_store
    ids_to_delete = []
    
    for doc_id, meta in list(metadata_store.items()):
        if meta["name"] in names:
            ids_to_delete.append(doc_id)
            
    if not ids_to_delete:
        return 0
        
    faiss_index.remove_ids(np.array(ids_to_delete, dtype=np.int64))
    for doc_id in ids_to_delete:
        del metadata_store[doc_id]
        
    save_db()
    return len(ids_to_delete)

def delete_all():
    """Clears the entire FAISS index and metadata."""
    global faiss_index, metadata_store, current_id_counter
    base_idx = faiss.IndexFlatIP(OUTPUT_DIMENSIONALITY)
    faiss_index = faiss.IndexIDMap(base_idx)
    metadata_store = {}
    current_id_counter = 0
    save_db()

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 1800, overlap: int = 250) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks

def guess_mime_type(file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        return "application/octet-stream"
    return mime_type

def extract_pdf_text(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()

def upload_file_and_wait(file_path: str):
    """Uploads file to Gemini and waits if it's a video being processed."""
    uploaded = gemini_client.files.upload(file=file_path)
    
    if guess_mime_type(file_path).startswith("video/"):
        print(f"Waiting for video {file_path} to process...")
        while not uploaded.state or uploaded.state.name != "ACTIVE":
            time.sleep(5)
            uploaded = gemini_client.files.get(name=uploaded.name)
            if uploaded.state.name == "FAILED":
                raise ValueError("Video processing failed in Gemini.")
    return uploaded

# ---------------------------------------------------------------------------
# Gemini Interactions (Summarization & Embeddings)
# ---------------------------------------------------------------------------

def summarize_file_for_rag(file_path: str) -> str:
    """Uses Gemini 2.5 Flash to summarize non-text multimodal files into text."""
    uploaded = upload_file_and_wait(file_path)
    prompt = (
        "Create a retrieval-friendly knowledge summary of this file. "
        "If it contains speech, include the transcript and key points. "
        "If it is an image, describe all visible details, text, objects, charts, and context. "
        "If it is a video, summarize scenes, spoken content, on-screen text, and timeline highlights."
    )
    try:
        response = gemini_client.models.generate_content(
            model=CHAT_MODEL,
            contents=[prompt, uploaded],
        )
        return response.text.strip()
    finally:
        try:
            gemini_client.files.delete(name=uploaded.name)
        except Exception as e:
            print(f"Failed to delete file from Gemini: {e}")

def embed_texts(texts: List[str], task_type: str) -> List[List[float]]:
    if not texts:
        return []
    result = gemini_client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=OUTPUT_DIMENSIONALITY,
        ),
    )
    return [item.values for item in result.embeddings]

def embed_query(query: str) -> List[float]:
    return embed_texts([query], task_type="RETRIEVAL_QUERY")[0]

# ---------------------------------------------------------------------------
# Ingestion Functions
# ---------------------------------------------------------------------------

def add_text_document(name: str, text: str, chunk_size: int = 1800, overlap: int = 250) -> None:
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    embeddings = embed_texts(chunks, task_type="RETRIEVAL_DOCUMENT")

    for idx, (chunk, vector) in enumerate(zip(chunks, embeddings), start=1):
        add_to_faiss(vector, {
            "name": name,
            "modality": "text",
            "chunk_id": idx,
            "context_text": chunk
        })
    return len(chunks)

def add_pdf_document(file_path: str) -> int:
    file_path = str(Path(file_path))
    text = extract_pdf_text(file_path)

    if text:
        return add_text_document(Path(file_path).name, text)

    # Fallback to multimodal summarization if PDF is scanned/images
    summary = summarize_file_for_rag(file_path)
    vector = embed_texts([summary], task_type="RETRIEVAL_DOCUMENT")[0]
    add_to_faiss(vector, {
        "name": Path(file_path).name,
        "modality": "pdf",
        "chunk_id": 1,
        "context_text": summary
    })
    return 1

def add_media_document(file_path: str) -> int:
    """
    Fixed media ingestion: Avoids direct byte embeddings to prevent API errors. 
    Always converts images/video to text summaries first.
    """
    file_path = str(Path(file_path))
    mime_type = guess_mime_type(file_path)

    if mime_type == "application/pdf":
        return add_pdf_document(file_path)

    if mime_type.startswith("text/"):
        text = Path(file_path).read_text(encoding="utf-8")
        return add_text_document(Path(file_path).name, text)

    # For Images, Audio, and Video: Summarize to text, then embed the text!
    summary = summarize_file_for_rag(file_path)
    vector = embed_texts([summary], task_type="RETRIEVAL_DOCUMENT")[0]

    add_to_faiss(vector, {
        "name": Path(file_path).name,
        "modality": mime_type.split("/")[0], # e.g., 'video', 'image', 'audio'
        "chunk_id": 1,
        "context_text": summary
    })
    return 1

# ---------------------------------------------------------------------------
# Retrieval & Generation (Powered by Groq)
# ---------------------------------------------------------------------------

def retrieve(query: str, top_k: int = 3) -> List[Dict]:
    if faiss_index.ntotal == 0:
        return []

    query_vector = embed_query(query)
    vec = np.array([query_vector], dtype=np.float32)
    faiss.normalize_L2(vec)

    distances, indices = faiss_index.search(vec, top_k)
    
    results = []
    for d, i in zip(distances[0], indices[0]):
        if i != -1: # -1 means no result found
            results.append({
                "score": float(d),
                **metadata_store[int(i)]
            })
    return results

def ask_rag(question: str, top_k: int = 3) -> dict:
    """Handles logic for asking a question. Now uses Groq instead of Gemini."""
    hits = retrieve(question, top_k=top_k)
    
    if not hits:
        return {
            "answer": "The database is empty. Please upload some documents first.",
            "sources": []
        }

    context_blocks = []
    for i, hit in enumerate(hits, start=1):
        context_blocks.append(
            f"--- Source {i} ({hit['name']}) ---\n{hit['context_text']}"
        )

    context_string = "\n\n".join(context_blocks)

    system_prompt = "You are a highly intelligent RAG assistant. Answer the user's question strictly using the provided context. If the context does not contain the answer, say you do not know. Cite sources when possible."
    user_prompt = f"Retrieved Context:\n{context_string}\n\nUser Question: {question}"

    # Use Groq for fast, open-source LLM generation
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    
    return {
        "answer": response.choices[0].message.content.strip(),
        "sources": [{"name": h["name"], "score": h["score"], "modality": h["modality"]} for h in hits]
    }
