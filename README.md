# Multimodal RAG System (Ollama + Chroma + Unstructured)

This project implements a fully local, multimodal Retrieval-Augmented Generation (RAG) pipeline for research paper question answering. It processes PDFs using high-resolution parsing, extracts tables and images, generates enhanced summaries, stores embeddings in ChromaDB, and answers queries using local large language models via Ollama.

---

## Overview

The system performs the following steps:

1. High-resolution PDF parsing using Unstructured
2. Title-based intelligent chunking
3. Extraction of tables (HTML) and images (base64)
4. Multimodal summary generation using LLaVA
5. Embedding generation using HuggingFace Sentence Transformers
6. Persistent storage in Chroma vector database
7. Retrieval and final answer generation using Llama 3

---

## Architecture

PDF  
→ Unstructured (hi_res parsing)  
→ Chunking  
→ Multimodal summarization  
→ Embeddings  
→ Chroma vector store  
→ Retrieval  
→ LLM answer generation  

---

## Tech Stack

- Python  
- LangChain  
- Unstructured  
- ChromaDB  
- HuggingFace Sentence Transformers  
- Ollama  
- LLaVA  
- Llama 3  

---

## Installation

### Clone the repository

bash
git clone https://github.com/KrishGupta88/PDF-CHATBOT.git
cd PDF-CHATBOT

Create virtual environment:

python -m venv venv
venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
Requirements

Tesseract OCR

Poppler

Ollama

Pull required models:

ollama pull llava
ollama pull llama3
Running

Place your PDF file in the project directory and run:

python main.py
Example Queries

What are the two main components of the Transformer architecture?

How many attention heads does the Transformer use?

What is the dimension of each attention head?

Purpose

This project demonstrates:

Multimodal document understanding

Table-aware retrieval

Local LLM deployment

Persistent vector search

End-to-end RAG pipeline design

Author
Krish Gupta
