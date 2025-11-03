import os
import json
import shutil
from typing import List, Dict, Optional, Any, TypedDict, Literal
from pydantic import BaseModel, Field
import uvicorn
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles

# LangChain and LangGraph Packages

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import operator
from typing import List, Optional, Dict, Any, TypedDict, Literal
from pydantic import BaseModel, Field

# --- LangGraph Core & Imports ---
from langgraph.graph import StateGraph, END
from typing_extensions import Annotated
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.retrievers import BaseRetriever
# ----------------------------------------------------------------------
# 1. GLOBAL RAG COMPONENTS (Initialized in main)
# ----------------------------------------------------------------------
LLM = None
RETRIEVER = None

# ----------------------------------------------------------------------
# 2. SCHEMAS AND STATE
# ----------------------------------------------------------------------
class BillData(BaseModel):
    """Extracted data for financial bill details."""
    hospital_name: Optional[str] = Field(description="The name of the hospital.")
    total_amount: Optional[float] = Field(description="The final total bill amount (as a number).")
    date_of_service: Optional[str] = Field(description="The admission/start date of the service.")

class DischargeData(BaseModel):
    """Extracted data for patient discharge summary."""
    patient_name: Optional[str] = Field(description="The patient's full name.")
    diagnosis: Optional[str] = Field(description="The primary diagnosis or reason for visit.")
    discharge_date: Optional[str] = Field(description="The patient's discharge date.")

class MinimalExtractionState(TypedDict):
    """The state shared across the two nodes."""
    bill_data: Optional[BillData]
    discharge_data: Optional[DischargeData]
    final_output_structure: Optional[Dict[str, Any]]

# ----------------------------------------------------------------------
# 3. EXTRACTION TOOLS (@tool FUNCTIONS) - CORRECTED
# ----------------------------------------------------------------------

@tool
def bill_extractor_tool(query: str) -> BillData:
    """Retrieves all bill-related context using RAG and forces extraction into the BillData schema."""
    
    # 1. RAG Call: Retrieve documents
    context_docs = RETRIEVER.invoke(query)
    context_text = "\n---\n".join([doc.page_content for doc in context_docs])

    # 2. Bind Schema to LLM (This is the robust method)
    structured_llm = LLM.with_structured_output(BillData)
    
    # 3. Define a SIMPLE prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial extraction expert. Extract data ONLY from the context provided. The output MUST match the BillData schema.\n\nContext: {context}"),
        ("human", "{input_query}"),
    ])
    
    # 4. Create the simple, direct chain (Prompt | Structured LLM)
    extraction_chain = prompt | structured_llm
    
    # 5. Invoke the chain. This returns a Pydantic BillData object, not a dict.
    return extraction_chain.invoke({"context": context_text, "input_query": query})


@tool
def discharge_extractor_tool(query: str) -> DischargeData:
    """Retrieves discharge-related context using RAG and forces extraction into the DischargeData schema."""
    
    # 1. RAG Call
    context_docs = RETRIEVER.invoke(query)
    context_text = "\n---\n".join([doc.page_content for doc in context_docs])
    
    # 2. Bind Schema to LLM
    structured_llm = LLM.with_structured_output(DischargeData)
    
    # 3. Define a SIMPLE prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a medical records extraction expert. Extract data ONLY from the context provided. The output MUST match the DischargeData schema.\n\nContext: {context}"),
        ("human", "{input_query}"),
    ])
    
    # 4. Create the simple, direct chain
    extraction_chain = prompt | structured_llm
    
    # 5. Invoke the chain. This returns a Pydantic DischargeData object.
    return extraction_chain.invoke({"context": context_text, "input_query": query})

# ----------------------------------------------------------------------
# 4. LANGGRAPH NODES (The Agent Execution Logic) - CORRECTED
# ----------------------------------------------------------------------

def bill_extraction_node(state: MinimalExtractionState) -> MinimalExtractionState:
    """Extract bill-related information using RAG + Gemini."""
    print("--- 💸 Bill Agent: Extracting Financial Data ---")
    query = "Hospital name, total bill amount, and date of service."
    
    # The tool now returns a Pydantic object directly.
    bill_data = bill_extractor_tool.invoke({
        "llm": LLM,
        "retriever": RETRIEVER,
        "query": query
    })
    
    # 🛑 REMOVED: bill_data = BillData(**bill_dict) - No longer needed
    return {"bill_data": bill_data}


def discharge_and_finalize_node(state: MinimalExtractionState) -> MinimalExtractionState:
    """Extract discharge summary, validate, and finalize JSON."""
    print("--- 🩺 Discharge Agent: Extracting & Finalizing Data ---")
    query = "Patient name, primary diagnosis, and discharge date."
    
    # The tool returns a Pydantic object directly.
    discharge_data = discharge_extractor_tool.invoke({
        "llm": LLM,
        "retriever": RETRIEVER,
        "query": query
    })
    
    # 🛑 REMOVED: discharge_data = DischargeData(**discharge_dict) - No longer needed
    bill_data = state["bill_data"]

    # Validation
    missing_fields = []
    if bill_data.total_amount is None:
        missing_fields.append("total_amount")
    if discharge_data.patient_name is None:
        missing_fields.append("patient_name")

    claim_status = "approved" if not missing_fields else "rejected"
    reason = "All required fields present." if not missing_fields else f"Missing fields: {', '.join(missing_fields)}"

    final_output_dict = {
        "documents": [
            {**bill_data.dict(exclude_none=True), "type": "bill"},
            {**discharge_data.dict(exclude_none=True), "type": "discharge_summary"}
        ],
        "validation": {"missing_fields": missing_fields},
        "claim_decision": {"status": claim_status, "reason": reason}
    }

    return {
        "discharge_data": discharge_data,
        "final_output_structure": final_output_dict
    }
# ----------------------------------------------------------------------
# 5. FULL RAG + LANGGRAPH PIPELINE FUNCTION (Same as your script)
# ----------------------------------------------------------------------
def run_full_pipeline(upload_dir: str, persist_dir: str) -> dict:
    """
    Runs the entire RAG and LangGraph workflow on the files in upload_dir.
    """
    
    # --- A. Load PDFs and Create Embeddings ---
    print("\n--- Creating New Vector Database ---")
    if not os.path.isdir(upload_dir):
        raise FileNotFoundError(f"Error: The directory '{upload_dir}' was not found.")
    
    loader = DirectoryLoader(upload_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    all_documents = loader.load()
    if not all_documents:
        raise ValueError("No PDF documents were found in the 'uploads' folder.")
    
    print(f"Successfully loaded {len(all_documents)} pages from PDFs.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(all_documents)
    print(f"Total number of chunks created after splitting: {len(split_docs)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={'normalize_embeddings': True}
    )
    
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    
    print(f"Creating ChromaDB and generating embeddings for {len(split_docs)} chunks...")
    vector_db = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"✅ Documents successfully embedded and persisted at: {persist_dir}")

    # --- B. Initialize Global RAG Components ---
    global LLM, RETRIEVER
    LLM = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.0)
    RETRIEVER = vector_db.as_retriever(search_kwargs={"k": 8})
    print("RAG Components (LLM/Retriever) initialized.")

    # --- C. LangGraph Compilation ---
    workflow = StateGraph(MinimalExtractionState)
    workflow.add_node("bill", bill_extraction_node)
    workflow.add_node("discharge_and_finalize", discharge_and_finalize_node)
    workflow.set_entry_point("bill")
    workflow.add_edge("bill", "discharge_and_finalize")
    workflow.set_finish_point("discharge_and_finalize")
    
    app = workflow.compile()
    print("\nLangGraph workflow compiled and ready.")

    # --- D. Run the Graph ---
    print("\n--- Running Two-Node Extraction Workflow ---")
    final_state = app.invoke({})

    # --- E. Return Final Output ---
    return final_state['final_output_structure']

# ----------------------------------------------------------------------
# 6. FASTAPI SERVER DEFINITION
# ----------------------------------------------------------------------
app = FastAPI()

UPLOAD_DIR = "./uploads"
PERSIST_DIR = './local_bge_chroma_db'

# 🛑 NEW: API ENDPOINT AT /process-claim
@app.post("/process-claim")
async def handle_claim_processing(files: List[UploadFile] = File(...)):
    """
    Receives PDF files, saves them, runs the full RAG pipeline,
    and returns the final extracted JSON.
    """
    
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR)

    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    
    print(f"Successfully saved {len(files)} files to {UPLOAD_DIR}.")

    try:
        load_dotenv()
        if 'GOOGLE_API_KEY' not in os.environ:
            raise ValueError("GOOGLE_API_KEY not set in environment or .env file.")
        
        final_json_output = run_full_pipeline(UPLOAD_DIR, PERSIST_DIR)
        
        return final_json_output
        
    except Exception as e:
        print(f"An error occurred during the pipeline: {e}")
        # Return a JSON error response
        return {"error": str(e)}

# 🛑 NEW: MOUNT STATIC FILES
# This serves index.html, style.css, and app.js
# This must be the LAST part of the app definition.
app.mount("/", StaticFiles(directory=".", html=True), name="static")

# ----------------------------------------------------------------------
# 7. SERVER EXECUTION
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting FastAPI server at http://localhost:8000")
    # This line correctly points to your actual file 'text-extraction.py'
    uvicorn.run("text-extraction:app", host="0.0.0.0", port=8000, reload=True)