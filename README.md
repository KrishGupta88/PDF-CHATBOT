# PDF-CHATBOT
quick Start Guide
Get up and running with the Multimodal RAG System in 5 minutes!
Prerequisites Checklist
Before you begin, make sure you have:

 Python 3.8 or higher installed
 At least 8GB of RAM
 10GB of free disk space
 A PDF document to process

Step-by-Step Setup
1. Install Ollama (3 minutes)
macOS:
bashbrew install ollama
ollama serve &
Linux:
bashcurl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
Windows:

Download from https://ollama.ai/download
Run the installer
Ollama will start automatically

2. Pull LLM Models (5-10 minutes)
bash# Pull the required models (this downloads them)
ollama pull llama3.2
ollama pull llava

# Verify installation
ollama list
3. Install System Dependencies (2 minutes)
macOS:
bashbrew install tesseract poppler
Ubuntu/Debian:
bashsudo apt-get update
sudo apt-get install tesseract-ocr poppler-utils
Windows:

Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
Install Poppler: https://github.com/oschwartz10612/poppler-windows/releases/
Add both to your PATH environment variable

4. Setup Python Environment (3 minutes)
bash# Create project directory
mkdir my-rag-project
cd my-rag-project

# Create virtual environment
python -m venv venv

# Activate it
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
5. Run Your First Query (1 minute)

Copy your PDF file into the project directory
Edit main.py and change the filename:

pythonfile_path = "your-document.pdf"  # ← Change this

Run it:

bashpython main.py
What You'll See
Partitioning document: your-document.pdf
Extracted 150 elements
Found 5 images
Found 3 tables
Creating smart chunks...
Created 25 chunks
...
Generating final answer...
================================================================================
FINAL ANSWER:
================================================================================
[Your answer will appear here]
Quick Troubleshooting
"Command not found: ollama"
→ Ollama not installed or not in PATH. Reinstall from ollama.ai
"Model not found: llama3.2"
→ Run: ollama pull llama3.2
"Tesseract not found"
→ Install Tesseract OCR (see step 3 above)
Script runs but no output
→ Check if ollama serve is running in the background
Next Steps

Customize your queries - Edit the query text in main.py
Try different PDFs - Process research papers, reports, manuals
Adjust chunk size - Modify chunking parameters for your use case
Explore the results - Check the JSON output files
Build your application - Use this as a foundation for your RAG app

Common Use Cases
Research Paper Analysis
pythonquery = "What are the main contributions of this paper?"
query = "What methodology did the authors use?"
query = "What are the limitations mentioned?"
Technical Documentation
pythonquery = "How do I configure the authentication settings?"
query = "What are the API rate limits?"
query = "What troubleshooting steps are recommended?"
Financial Reports
pythonquery = "What was the revenue growth in Q3?"
query = "What are the main risk factors mentioned?"
query = "Compare performance across different segments"
Tips for Best Results

Be Specific: "What are the hyperparameters in Table 2?" is better than "Tell me about the model"
Reference Context: "According to the methodology section..." helps the model focus
Adjust k Value: If answers lack context, increase k in search_kwargs={"k": 3}
Use Multiple Queries: Break complex questions into simpler sub-questions

Need Help?

Full documentation: See README.md
Troubleshooting: Check the troubleshooting section in README.md
Ollama docs: https://ollama.ai/
LangChain docs: https://python.langchain.com/


You're all set! Start asking questions to your documents!
