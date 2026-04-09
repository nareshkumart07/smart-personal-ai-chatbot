<h1 align="center">🌐 Open-Source Multimodal RAG Chatbot</h1>

<p align="center">
<mark><b>A blazing fast, locally-hosted Retrieval-Augmented Generation (RAG) system</b></mark>

Capable of ingesting <b>text, PDFs, Images, Audio, and Video.</b>

</p>

<p align="center">


<a href="[#-getting-started](https://smart-personal-ai-chatbot.onrender.com)">View Demo</a>

</p>

🌟 Overview

💡 Traditional RAG systems are limited to text. This project introduces a Multimodal RAG pipeline that can understand videos, images, audio, and documents out-of-the-box.

Built using Google's gemini-embedding-2-preview and gemini-2.5-flash models, this unified chatbot empowers you to efficiently interact with diverse data types—including text, PDFs, voice, images, and video—all through a single interface. By combining these multimodal capabilities with the lightning-fast inference of Groq (Llama 3) and the lightweight vector search of FAISS, this application allows you to chat with your personal data locally and securely. You can find the complete source code and contribute on GitHub.

✨ Key Features

Feature

Description

🎥 Multimodal Ingestion

Upload MP4s, MP3s, JPGs, PNGs, PDFs, and raw text seamlessly.

⚡ Blazing Fast Answers

Powered by Groq's LPU inference engine running Llama 3 models for near-instant responses.

🔐 Bring Your Own Key (BYOK)

No backend .env variables required. Input your API keys securely right in the browser session.

🧠 Smart Media Translation

Automatically transcodes and summarizes video/audio using Gemini before embedding, bridging the gap between multimodal files and text-based LLMs.

💾 Local Vector Store

Uses FAISS for CPU-friendly, local vector similarity search. No expensive cloud database needed.

🎨 Beautiful UI

Responsive, Tailwind CSS-powered dashboard featuring real-time markdown rendering and smart source citations.

🏗️ Architecture Pipeline

Step

Phase

What Happens

Technology Used

1

📥 Ingestion

Files are uploaded securely via the local backend.

FastAPI

2

🔄 Translation

Non-text files (Images/Video/Audio) generate highly detailed textual summaries and transcripts.

gemini-2.5-flash

3

🧩 Embedding

The text (or generated summary) is chunked and vectorized.

gemini-embedding-2

4

🗄️ Storage

Vectors and metadata are stored locally.

FAISS

5

💬 Retrieval

User queries are embedded, matched against the index, and passed to the LLM for a highly accurate answer.

Groq + Llama 3

🚀 Getting Started

Follow these steps to get the project running on your local machine.

📋 Prerequisites

Python 3.9+

A Google Gemini API Key (Get one here)

A Groq API Key (Get one here)

⚙️ Installation

Clone the repository

git clone [https://github.com/nareshkumart07/smart-personal-ai-chatbot.git](https://github.com/nareshkumart07/smart-personal-ai-chatbot.git)
cd smart-personal-ai-chatbot


Create a virtual environment (Recommended)

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate


Install dependencies

pip install -r requirements.txt


Run the FastAPI server

uvicorn main:app --reload


Open the Application
Navigate to http://localhost:8000 in your web browser.

💻 Usage Guide

🛡️ Privacy First: Your API keys are strictly kept in your local browser storage and cleared upon exit.

Action

Instructions

🔑 Initialize

Upon opening the app, prompt your Gemini and Groq API keys to initialize the engine.

📂 Upload Data

Use the left sidebar to upload media files (<mark>Max 50MB</mark>) or paste raw text (<mark>Max 20,000 chars</mark>).

💬 Chat

Ask questions in the main chat window. The bot will cite its sources based on the exact documents and media you uploaded.

🗃️ Manage Database

View indexed documents, clear specific files, or purge the entire FAISS database directly from the sidebar.

🛠️ Tech Stack

Domain

Technologies

Frontend

HTML5, Tailwind CSS, Vanilla JavaScript, Marked.js

Backend

FastAPI, Python, Pydantic

Vector Store

FAISS (Facebook AI Similarity Search)

Embeddings & Vision

Google Gemini API

LLM Inference

Groq API

🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

📝 License

Distributed under the MIT License. See LICENSE for more information.
