<div align="center">

# 🌐 Open-Source Multimodal RAG Chatbot

**A blazing fast, locally-hosted Retrieval-Augmented Generation (RAG) system**<br/>
capable of ingesting **Text, PDFs, Images, Audio, and Video.**

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-009688?style=for-the-badge)](https://faiss.ai)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

[![View Demo](https://img.shields.io/badge/View%20Live%20Demo-FF4B4B?style=for-the-badge)](https://smart-personal-ai-chatbot.onrender.com)
[![GitHub](https://img.shields.io/badge/Source%20Code-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/nareshkumart07/smart-personal-ai-chatbot)

</div>

---

## 🌟 Overview

Traditional RAG systems are limited to text. This project introduces a **Multimodal RAG pipeline** that understands videos, images, audio, and documents out-of-the-box.

Built using **Google's `gemini-embedding-2-preview`** and **`gemini-2.5-flash`** models, this unified chatbot lets you interact with diverse data types — text, PDFs, voice, images, and video — all through a single interface. By combining multimodal capabilities with the lightning-fast inference of **Groq (Llama 3)** and the lightweight vector search of **FAISS**, you can chat with your personal data **locally and securely**.

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
      <td>Upload <code>MP4</code>, <code>MP3</code>, <code>JPG</code>, <code>PNG</code>, <code>PDF</code>, and raw text seamlessly.</td>
    </tr>
    <tr>
      <td>⚡ <strong>Blazing Fast Answers</strong></td>
      <td>Powered by Groq's LPU inference engine running Llama 3 for near-instant responses.</td>
    </tr>
    <tr>
      <td>🔐 <strong>Bring Your Own Key (BYOK)</strong></td>
      <td>No backend <code>.env</code> required. Input API keys securely in the browser session.</td>
    </tr>
    <tr>
      <td>🧠 <strong>Smart Media Translation</strong></td>
      <td>Automatically transcodes and summarizes video/audio using Gemini before embedding.</td>
    </tr>
    <tr>
      <td>💾 <strong>Local Vector Store</strong></td>
      <td>Uses FAISS for CPU-friendly, local similarity search — no expensive cloud database needed.</td>
    </tr>
    <tr>
      <td>🎨 <strong>Beautiful UI</strong></td>
      <td>Responsive Tailwind CSS dashboard with real-time markdown rendering and smart source citations.</td>
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
      <td>Files are uploaded securely via the local backend.</td>
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
      <td>Text or generated summary is chunked and vectorized.</td>
      <td><code>gemini-embedding-2</code></td>
    </tr>
    <tr>
      <td align="center"><strong>4</strong></td>
      <td>🗄️ <strong>Storage</strong></td>
      <td>Vectors and metadata are stored locally.</td>
      <td><code>FAISS</code></td>
    </tr>
    <tr>
      <td align="center"><strong>5</strong></td>
      <td>💬 <strong>Retrieval</strong></td>
      <td>User queries are embedded, matched against the index, and passed to the LLM.</td>
      <td><code>Groq + Llama 3</code></td>
    </tr>
  </tbody>
</table>

---

## 🚀 Getting Started

### 📋 Prerequisites

- Python **3.9** or higher
- A **Google Gemini API Key** — [Get one here](https://aistudio.google.com/app/apikey)
- A **Groq API Key** — [Get one here](https://console.groq.com)

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
pip install -r requirements.txt
```

**4. Run the FastAPI server**

```bash
uvicorn main:app --reload
```

**5. Open the Application**

Navigate to `http://localhost:8000` in your browser.

---

## 💻 Usage Guide

> 🛡️ **Privacy First:** Your API keys are strictly kept in your local browser session and cleared upon exit.

<table>
  <thead>
    <tr><th>Action</th><th>Instructions</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>🔑 <strong>Initialize</strong></td>
      <td>Enter your Gemini and Groq API keys to initialize the engine.</td>
    </tr>
    <tr>
      <td>📂 <strong>Upload Data</strong></td>
      <td>Upload media files <strong>(Max 50MB)</strong> or paste raw text <strong>(Max 20,000 chars)</strong> from the sidebar.</td>
    </tr>
    <tr>
      <td>💬 <strong>Chat</strong></td>
      <td>Ask questions in the chat window. The bot cites sources from your uploaded documents.</td>
    </tr>
    <tr>
      <td>🗃️ <strong>Manage Database</strong></td>
      <td>View indexed documents, clear specific files, or purge the entire FAISS database from the sidebar.</td>
    </tr>
  </tbody>
</table>

---

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

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add some AmazingFeature'`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

---

## 📄 License

Distributed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <em>Built for developers who want powerful, private, and multimodal AI — without the cloud dependency.</em>
</div>
