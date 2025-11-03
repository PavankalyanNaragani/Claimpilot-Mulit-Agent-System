🩺 ClaimPilot: AI-Powered Medical Claim Extractor

ClaimPilot is a full-stack, AI-powered web application designed to automate medical insurance claim processing. This project implements an end-to-end Retrieval-Augmented Generation (RAG) pipeline and a Multi-Agent workflow (built with LangGraph) to read multiple, unstructured PDF documents (like hospital bills and discharge summaries) and return a single, structured JSON object with the extracted, validated data.


🚀 Core Technologies

⚙️ Backend: FastAPI (Python)
🧠 AI Orchestration: Langchain, LangGraph
⚙️ LLM (Generation): Google Gemini 2.5 Pro (or `gemini-2.0-flash`)
🚀 Embeddings (RAG): `BAAI/bge-small-en-v1.5` (via Hugging Face)
📚 Vector Store: ChromaDB
📄 Frontend: HTML, CSS, and vanilla JavaScript


🚀 Core Features
📄 Upload and embed your PDFs
🔗 Chunked PDF text stored using ChromaDB for fast retrieval
🔧 Simple FastAPI backend and minimal chatbot UI (HTML)
🧠 Multi-Agent Workflow 

## Workflow Architecture

1.  Upload:  The user uploads multiple PDFs via the `index.html` frontend.
2.  Serve:  FastAPI receives the files at the `/process-claim` endpoint.
3.  Embed:  The `run_full_pipeline` function is triggered. It loads all PDFs, splits them into chunks, and generates BGE embeddings.
4.  Store: The embeddings are saved in a local ChromaDB vector store (`./local_bge_chroma_db`).
5.  Initialize: The `LLM` (Gemini) and `RETRIEVER` (ChromaDB) are initialized.
6.  Compile: The LangGraph `StateGraph` is compiled.
7.  Execute Graph:
       Node 1: `bill_extraction_node`** runs. It calls the `bill_extractor_tool`, which performs RAG to find financial data and uses Gemini to extract it into a `BillData` Pydantic model.
       Node 2: `discharge_and_finalize_node`** runs. It calls the `discharge_extractor_tool` to get patient data. It then validates all extracted data, makes a final `claim_decision`, and assembles the final JSON object.
8.  Return: The final JSON object is returned by the API to the frontend.

AI TOOLS :
   🧠 cursor.ai : To generate frontend 
   🚀 Gemini/chatGpt : Debugging and Requirements gathering, generating project structure, etc   


📂 Project Structure
│
├── .env                  # Stores your API key
├── main.py               # The complete FastAPI server and LangGraph logic
├── index.html            # The frontend HTML
├── style.css             # The frontend CSS
├── app.js                # The frontend JavaScript
├── requirements.txt      # Python dependencies
│
├─── uploads/             # (Optional) Folder for initial PDF testing
│   └── (your_pdfs.pdf)
│
└─── local_bge_chroma_db/ # (Auto-generated) Stores the vector database



## ⚙️ Setup and Installation

**1. Clone or Download:**
   Get all the project files (`main.py`, `index.html`, `style.css`, `app.js`) into a single directory.

**2. Create and activate virtual environment**

python -m venv venv
source venv/bin/activate    # For Linux/macOS
venv\Scripts\activate       # For Windows

**3. Install required dependencies**

pip install -r requirements.txt

**4. Create your .env File: Create a file named .env in the same directory. This file will securely store your API key.**

   GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"


▶️ How to Run the Application
Run the Server: Start the FastAPI server from your terminal:

Bash

python main.py
(Note: The script main.py assumes that's what you've named the final Python file.)

Open the Application: Open your web browser and go to: http://localhost:8000

Process Documents:

Drag and drop your PDF files (e.g., the sample bill and discharge summary) onto the upload zone.

Click the "Process Claim" button.

Wait for the backend to run the full RAG and agent workflow.

The final, structured JSON output will appear in the status panel on the right.

uvicorn app:app --reload --port 8080

**5. Open the chatbot UI**

Visit http://localhost:8080/ in your browser
💬 Example
User: What are transformers?



🙌 Conclusion
In this project, we built an end-to-end PDF Insurance Claim RAG system using FastAPI, Chroma, HuggingFace Embeddings, and Gemini's blazing-fast gemini-flash-2.0 model. From PDF upload to chunking, embedding, retrieval, mutli-agent and final output generation — every step is streamlined and modular.
