<h3 align="center">Open-Source Multimodal RAG Chatbot</h3>

<p align="center">
A blazing fast, locally-hosted Retrieval-Augmented Generation (RAG) system capable of ingesting text, PDFs, Images, Audio, and Video.







<a href="#features"><strong>Explore the docs »</strong></a>







<a href="#installation">View Demo</a>
.
<a href="https://www.google.com/search?q=https://github.com/yourusername/multimodal-rag-bot/issues">Report Bug</a>
.
<a href="https://www.google.com/search?q=https://github.com/yourusername/multimodal-rag-bot/issues">Request Feature</a>
</p>
</p>

🌟 Overview

<p align="center">
<img src="https://www.google.com/search?q=https://placehold.co/900x450/1e293b/ffffff%3Ftext%3DDashboard%2BScreenshot%2BHere" alt="Dashboard Preview" width="100%">
</p>

Traditional RAG systems are limited to text. This project introduces a Multimodal RAG pipeline that can understand videos, images, audio, and documents.

By combining the multimodal capabilities of Google Gemini 2.5 Flash with the lightning-fast inference of Groq (Llama 3) and the lightweight vector search of FAISS, this app allows you to chat with your personal data locally and securely.

✨ Key Features

🎥 Multimodal Ingestion: Upload MP4s, MP3s, JPGs, PNGs, PDFs, and raw text.

⚡ Blazing Fast Answers: Powered by Groq's LPU inference engine running Llama 3 models for instant responses.

🔐 Bring Your Own Key (BYOK): No backend environment variables required. Input your API keys securely in the browser session.

🧠 Smart Media Translation: Automatically transcodes and summarizes video/audio using Gemini before embedding, bridging the gap between multimodal files and text-based LLMs.

💾 Local Vector Store: Uses FAISS for CPU-friendly, local vector similarity search. No expensive cloud database needed.

🎨 Beautiful UI: Responsive, Tailwind CSS-powered dashboard with real-time markdown rendering and smart source citations.

🏗️ Architecture

Ingestion: Files are uploaded via the FastAPI backend.

Translation: Non-text files (Images/Video/Audio) are passed to gemini-2.5-flash to generate a highly detailed textual summary and transcript.

Embedding: The text (or generated summary) is chunked and vectorized using gemini-embedding-2-preview.

Storage: Vectors and metadata are stored locally in a FAISS index (faiss_index.bin).

Retrieval & Generation: User queries are embedded, matched against the FAISS index, and the relevant context is passed to llama-3 via Groq for a lightning-fast, highly accurate answer.

🚀 Getting Started

Follow these steps to get the project running on your local machine.

Prerequisites

Python 3.9+

A Google Gemini API Key (Get one here)

A Groq API Key (Get one here)

Installation

Clone the repository

git clone [https://github.com/yourusername/multimodal-rag-bot.git](https://github.com/yourusername/multimodal-rag-bot.git)
cd multimodal-rag-bot


Create a virtual environment (Recommended)

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate


Install dependencies

pip install -r requirements.txt


Run the FastAPI server

uvicorn main:app --reload


Open the Application
Navigate to http://localhost:8000 in your web browser.

💻 Usage

<p align="center">
<img src="https://www.google.com/search?q=https://placehold.co/600x300/1e293b/ffffff%3Ftext%3DBYOK%2BLogin%2BScreen" alt="Login Preview" width="80%">
</p>

Initialize the System: Upon opening the app, you will be prompted to enter your Gemini and Groq API keys. These are stored locally in your browser memory and passed securely via headers.

Upload Data: Use the left sidebar to upload media files (Max 50MB) or paste raw text (Max 20,000 characters).

Chat: Ask questions in the main chat window. The bot will cite its sources based on the exact documents and media you uploaded.

Manage Database: You can view indexed documents, clear specific files, or purge the entire FAISS database directly from the sidebar.

🛠️ Tech Stack

Frontend: HTML5, Tailwind CSS, Vanilla JavaScript, Marked.js

Backend: FastAPI, Python, Pydantic

Vector Store: FAISS (Facebook AI Similarity Search)

Embeddings & Vision: Google Gemini API (gemini-embedding-2-preview, gemini-2.5-flash)

LLM Inference: Groq API (llama-3.1-8b-instant)

🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

📝 License

Distributed under the MIT License. See LICENSE for more information.

<p align="center">
Built with ❤️ by <a href="https://github.com/yourusername">Your Name</a>
</p>
