<div align="center">

# 🌐 Multimodal RAG Chatbot

**A single-user Retrieval-Augmented Generation (RAG) demonstration**<br/>
for **Text, PDFs, Images, Audio, and Video**, using a FastAPI backend, a local FAISS index, and external Gemini and Groq APIs.

<br/>

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-009688?style=for-the-badge)](https://faiss.ai)

<br/>

[![View Demo](https://img.shields.io/badge/View%20Live%20Demo-FF4B4B?style=for-the-badge)](https://smart-personal-ai-chatbot.onrender.com)
[![GitHub](https://img.shields.io/badge/Source%20Code-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/nareshkumart07/smart-personal-ai-chatbot)

</div>

---

## 🌟 Overview

This project demonstrates document ingestion, retrieval, and question answering in one interface. Text and text-based PDFs are extracted and chunked; images, audio, video, and scanned PDFs are sent to Gemini for a textual summary. Gemini creates embeddings, FAISS searches a local index, and Groq generates an answer from the retrieved context.

**Local storage does not mean local-only processing.** Document text, media, summaries, questions, and retrieved context are sent to the relevant external model providers. Internet access and API credentials are required. The current application has one shared document store and is intended for a trusted single-user demonstration, not a private multi-user service.

The implementation demonstrates FastAPI endpoints, overlapping text chunks, normalized vector search with removable IDs, disk persistence, source metadata, and a browser client. No retrieval-quality, latency, or answer-accuracy benchmark is published in this repository.

---

## ✨ Key Features

<table>
  <thead>
    <tr>
      <th>Feature</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>🎥 <strong>Multimodal Ingestion</strong></td>
      <td>Ingest <code>MP4</code>, <code>MP3</code>, <code>JPG</code>, <code>PNG</code>, <code>PDF</code>, and raw text, subject to provider support and limits.</td>
    </tr>
    <tr>
      <td>💬 <strong>Context-Based Answers</strong></td>
      <td>Groq generates answers from retrieved text. Latency and answer quality depend on the data, retrieval, and external APIs.</td>
    </tr>
    <tr>
      <td>🔐 <strong>Bring Your Own Key (BYOK)</strong></td>
      <td>Enter provider keys in the browser. The client sends them to the backend in request headers; they are not application login credentials.</td>
    </tr>
    <tr>
      <td>🧠 <strong>Smart Media Translation</strong></td>
      <td>Gemini summarizes media into text before embedding; the application does not implement a media transcoding pipeline.</td>
    </tr>
    <tr>
      <td>💾 <strong>Local Vector Store</strong></td>
      <td>Uses FAISS for local similarity search and JSON for source text and metadata. All requests share this store.</td>
    </tr>
    <tr>
      <td>🎨 <strong>Beautiful UI</strong></td>
      <td>Tailwind CSS dashboard with Markdown rendering and retrieved source names and similarity scores. Generated citations still require verification.</td>
    </tr>
  </tbody>
</table>

---

## 🏗️ Architecture Pipeline

<table>
  <thead>
    <tr>
      <th align="center">Step</th>
      <th>Phase</th>
      <th>What Happens</th>
      <th>Technology</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><strong>1</strong></td>
      <td>📥 <strong>Ingestion</strong></td>
      <td>Files are uploaded to the FastAPI backend; media may then be uploaded to Gemini.</td>
      <td><code>FastAPI</code></td>
    </tr>
    <tr>
      <td align="center"><strong>2</strong></td>
      <td>🔄 <strong>Translation</strong></td>
      <td>Non-text files generate detailed textual summaries and transcripts.</td>
      <td><code>gemini-2.5-flash</code></td>
    </tr>
    <tr>
      <td align="center"><strong>3</strong></td>
      <td>🧩 <strong>Embedding</strong></td>
      <td>Extracted text is chunked and vectorized; a media summary is embedded as one record.</td>
      <td><code>gemini-embedding-2-preview</code></td>
    </tr>
    <tr>
      <td align="center"><strong>4</strong></td>
      <td>🗄️ <strong>Storage</strong></td>
      <td>Vectors and source-text metadata are persisted on the backend host in a shared store.</td>
      <td><code>FAISS</code></td>
    </tr>
    <tr>
      <td align="center"><strong>5</strong></td>
      <td>💬 <strong>Retrieval</strong></td>
      <td>Gemini embeds the question; FAISS retrieves context; Groq receives the question and retrieved text.</td>
      <td><code>Groq + Llama 3</code></td>
    </tr>
  </tbody>
</table>

---

## 🚀 Getting Started

### 📋 Prerequisites

- A Python environment compatible with the packages in `requirements.txt` (Python **3.11** is a practical starting point; dependencies are currently unpinned)
- A **Google Gemini API Key** — [Get one here](https://aistudio.google.com/app/apikey)
- A **Groq API Key** — [Get one here](https://console.groq.com)
- Internet access and provider access/quota for the configured models

---

### ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/nareshkumart07/smart-personal-ai-chatbot.git
cd smart-personal-ai-chatbot
```

**2. Create a virtual environment** *(Recommended)*

<table>
<tr><th>Windows</th><th>macOS / Linux</th></tr>
<tr>
<td>

```bash
python -m venv venv
venv\Scripts\activate
```

</td>
<td>

```bash
python3 -m venv venv
source venv/bin/activate
```

</td>
</tr>
</table>

**3. Install dependencies**

```bash
python -m pip install -r requirements.txt
```

**4. Run the FastAPI server**

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

**5. Open the Application**

Navigate to `http://127.0.0.1:8000` in your browser. Run the command from the repository root so the persisted data and temporary-upload paths are predictable. FastAPI also exposes its API documentation at `http://127.0.0.1:8000/docs`.

**Keep port 8000 for this local setup.** The current frontend uses same-origin API requests only on port 8000; on other ports it targets the hosted Render backend. Opening `index.html` directly is not an equivalent local setup. `--reload` is for development, and the shared file-backed index should not be run with multiple workers.

The model IDs are configured in `rag_core.py`: `gemini-embedding-2-preview`, `gemini-2.5-flash`, and `llama-3.1-8b-instant`, with 1,536-dimensional embeddings. Check availability in your provider account before use. If the embedding model or dimensionality changes, rebuild the stored index rather than mixing incompatible embeddings.

---

## 💻 Usage Guide

> **Data flow:** The frontend holds keys in page memory and sends them to the backend, which uses them to call Gemini and Groq. They are not confined to the browser. Use non-sensitive sample data for this demonstration; the hosted instance has no per-user document isolation.

<table>
  <thead>
    <tr><th>Action</th><th>Instructions</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>🔑 <strong>Initialize</strong></td>
      <td>Enter your Gemini and Groq API keys. The initial stats request checks only that both headers are present; a successful initialization does not validate provider credentials.</td>
    </tr>
    <tr>
      <td>📂 <strong>Upload Data</strong></td>
      <td>Upload media files or paste raw text from the sidebar. The browser applies a 50 MB combined upload check and a 20,000-character text-field limit; these are not server-enforced limits.</td>
    </tr>
    <tr>
      <td>💬 <strong>Chat</strong></td>
      <td>Ask questions in the chat window. The response includes retrieved source metadata; compare the generated answer with the original source.</td>
    </tr>
    <tr>
      <td>🗃️ <strong>Manage Database</strong></td>
      <td>View indexed documents, remove documents by name, or clear the shared FAISS database. These actions affect the entire instance, not an individual user.</td>
    </tr>
  </tbody>
</table>

---

## Data Storage and Deployment Boundaries

- `faiss_index.bin` contains persisted vectors; `metadata.json` contains names, modalities, chunk IDs, and source text or summaries. Both files are included in this repository. Treat them as demonstration data and review the indexed contents before adding new material.
- `temp_uploads/` receives uploaded files. Successful processing removes the local temporary file; failures may leave files behind. Gemini file deletion is attempted after summarization, but it is not a guarantee of deletion from all provider systems.
- The API currently checks only for non-empty provider-key headers. There are no user accounts, ownership checks, or separate user indexes. Listing, retrieval, and deletion operate on the shared store.
- A hosted multi-user deployment needs application authentication, per-user authorization and storage, server-side size/type limits, safe generated upload filenames, reliable cleanup, provider timeouts, rate limits, and controlled concurrent writes. The current permissive CORS policy also needs review for the intended deployment.
- The browser loads scripts, styles, and fonts from external CDNs. The application is not an offline or fully self-contained system.

## Validation and Evaluation

For a local smoke check, use a small non-sensitive text document with a distinctive fact, upload it, ask a directly supported question, inspect the returned source, restart the server to check persistence, and delete only that sample document. This workflow requires working API credentials and may incur provider charges.

Before presenting quantitative results, publish a fixed question-and-answer dataset, retrieval settings, model versions, retrieval recall, grounded-answer checks, latency, and estimated per-query cost. These measurements have not yet been reported here; source labels and similarity scores alone do not establish answer correctness.

## 🛠️ Tech Stack

<table>
  <thead>
    <tr><th>Domain</th><th>Technologies</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Frontend</strong></td>
      <td>
        <img src="https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white"/>
        <img src="https://img.shields.io/badge/Tailwind_CSS-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white"/>
        <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black"/>
      </td>
    </tr>
    <tr>
      <td><strong>Backend</strong></td>
      <td>
        <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white"/>
        <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/>
        <img src="https://img.shields.io/badge/Pydantic-E92063?style=flat-square&logo=pydantic&logoColor=white"/>
      </td>
    </tr>
    <tr>
      <td><strong>Vector Store</strong></td>
      <td><img src="https://img.shields.io/badge/FAISS-009688?style=flat-square"/></td>
    </tr>
    <tr>
      <td><strong>Embeddings & Vision</strong></td>
      <td><img src="https://img.shields.io/badge/Google_Gemini-4285F4?style=flat-square&logo=google&logoColor=white"/></td>
    </tr>
    <tr>
      <td><strong>LLM Inference</strong></td>
      <td>
        <img src="https://img.shields.io/badge/Groq-F55036?style=flat-square"/>
        <img src="https://img.shields.io/badge/Llama_3-0467DF?style=flat-square&logo=meta&logoColor=white"/>
      </td>
    </tr>
  </tbody>
</table>

---

## 🤝 Contributing

Issues and pull requests are welcome. Include a reproducible example and describe how the change was checked. Do not include API keys or sensitive documents in reports.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add some AmazingFeature'`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

---

## 📄 License

No license file is currently included in this repository. A license needs to be added by the repository owner before reuse permissions can be stated.

---

<div align="center">
  <em>A practical multimodal RAG demonstration with local retrieval storage and external model APIs.</em>
</div>
