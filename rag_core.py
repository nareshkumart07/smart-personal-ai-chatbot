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

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file.")

client = genai.Client(api_key=API_KEY)

EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-2-preview")
CHAT_MODEL = "gemini-2.5-flash"
OUTPUT_DIMENSIONALITY = 1536

# ---------------------------------------------------------------------------
# FAISS Vector Database Setup
# ---------------------------------------------------------------------------
# IndexFlatIP calculates the Inner Product. 
# When vectors are L2 normalized, Inner Product equals Cosine Similarity.
faiss_index = faiss.IndexFlatIP(OUTPUT_DIMENSIONALITY)
metadata_store: Dict[int, Dict[str, Any]] = {}
current_id_counter = 0

DB_PATH = "faiss_index.bin"
META_PATH = "metadata.json"

def save_db():
    """Persist FAISS index and metadata to disk (Production grade)."""
    faiss.write_index(faiss_index, DB_PATH)
    with open(META_PATH, "w") as f:
        json.dump(metadata_store, f)

def load_db():
    """Load FAISS index and metadata from disk if they exist."""
    global faiss_index, metadata_store, current_id_counter
    if os.path.exists(DB_PATH) and os.path.exists(META_PATH):
        faiss_index = faiss.read_index(DB_PATH)
        with open(META_PATH, "r") as f:
            # JSON keys are strings, convert back to int
            metadata_store = {int(k): v for k, v in json.load(f).items()}
        current_id_counter = max(metadata_store.keys()) + 1 if metadata_store else 0

# Try to load existing DB on startup
load_db()

def add_to_faiss(embedding: List[float], metadata: Dict[str, Any]):
    """Normalize vector and add to FAISS."""
    global current_id_counter
    vec = np.array([embedding], dtype=np.float32)
    faiss.normalize_L2(vec) # Required for Cosine Similarity with IndexFlatIP
    
    faiss_index.add(vec)
    metadata_store[current_id_counter] = metadata
    current_id_counter += 1
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
        # Fallback for unknown extensions
        return "application/octet-stream"
    return mime_type

def read_bytes(file_path: str) -> bytes:
    with open(file_path, "rb") as f:
        return f.read()

def extract_pdf_text(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()

def upload_file_and_wait(file_path: str):
    """Uploads file to Gemini and waits if it's a video being processed."""
    uploaded = client.files.upload(file=file_path)
    
    if guess_mime_type(file_path).startswith("video/"):
        print(f"Waiting for video {file_path} to process...")
        while not uploaded.state or uploaded.state.name != "ACTIVE":
            time.sleep(5)
            uploaded = client.files.get(name=uploaded.name)
            if uploaded.state.name == "FAILED":
                raise ValueError("Video processing failed in Gemini.")
    return uploaded

# ---------------------------------------------------------------------------
# Gemini Interactions
# ---------------------------------------------------------------------------

def summarize_file_for_rag(file_path: str) -> str:
    """Uses Gemini 2.5 Flash to summarize non-text multimodal files."""
    uploaded = upload_file_and_wait(file_path)
    prompt = (
        "Create a retrieval-friendly knowledge summary of this file. "
        "If it contains speech, include the transcript and key points. "
        "If it is an image, describe all visible details, text, objects, charts, and context. "
        "If it is a PDF, summarize the sections, facts, tables, and important numbers. "
        "If it is a video, summarize scenes, spoken content, on-screen text, and timeline highlights."
    )
    try:
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=[prompt, uploaded],
        )
        return response.text.strip()
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception as e:
            print(f"Failed to delete file from Gemini: {e}")

def embed_texts(texts: List[str], task_type: str) -> List[List[float]]:
    if not texts:
        return []
    result = client.models.embed_content(
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

def embed_file_bytes(file_path: str) -> List[float]:
    file_bytes = read_bytes(file_path)
    mime_type = guess_mime_type(file_path)

    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[
            types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
        ],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=OUTPUT_DIMENSIONALITY,
        ),
    )
    return result.embeddings[0].values

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
    vector = embed_file_bytes(file_path)
    add_to_faiss(vector, {
        "name": Path(file_path).name,
        "modality": "pdf",
        "chunk_id": 1,
        "context_text": summary
    })
    return 1

def add_media_document(file_path: str) -> int:
    file_path = str(Path(file_path))
    mime_type = guess_mime_type(file_path)

    if mime_type == "application/pdf":
        return add_pdf_document(file_path)

    if mime_type.startswith("text/"):
        text = Path(file_path).read_text(encoding="utf-8")
        return add_text_document(Path(file_path).name, text)

    summary = summarize_file_for_rag(file_path)
    if EMBED_MODEL == "gemini-embedding-2-preview":
        vector = embed_file_bytes(file_path)
    else:
        vector = embed_texts([summary], task_type="RETRIEVAL_DOCUMENT")[0]

    add_to_faiss(vector, {
        "name": Path(file_path).name,
        "modality": mime_type.split("/")[0], # e.g., 'video', 'image', 'audio'
        "chunk_id": 1,
        "context_text": summary
    })
    return 1

# ---------------------------------------------------------------------------
# Retrieval & Generation
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
        if i != -1: # -1 means no result found in FAISS
            results.append({
                "score": float(d),
                **metadata_store[int(i)]
            })
    return results

def ask_rag(question: str, top_k: int = 3) -> dict:
    hits = retrieve(question, top_k=top_k)
    
    if not hits:
        return {
            "answer": "The database is empty. Please upload some documents first.",
            "sources": []
        }

    context_blocks = []
    for i, hit in enumerate(hits, start=1):
        context_blocks.append(
            f"Source {i}\n"
            f"name: {hit['name']}\n"
            f"modality: {hit['modality']}\n"
            f"score: {hit['score']:.4f}\n"
            f"content:\n{hit['context_text']}"
        )

    prompt = f"""
    You are a RAG assistant.
    Answer the user only from the retrieved context below.
    If the answer is not supported by the context, say that clearly.
    When possible, mention the source names you used.

    Retrieved context:
    {chr(10).join(context_blocks)}

    User question: {question}
    """.strip()

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
    )
    
    return {
        "answer": response.text.strip(),
        "sources": [{"name": h["name"], "score": h["score"], "modality": h["modality"]} for h in hits]
    }