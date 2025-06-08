import gradio as gr
import os
import logging
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    TextLoader,
    UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import requests
import shutil
import pandas as pd
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
MODEL_NAME = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./app/chroma_db"
OLLAMA_BASE_URL = "http://llama-container2:11434"

# Supported file types
SUPPORTED_EXTENSIONS = {
    '.pdf': 'PDF Document',
    '.docx': 'Word Document',
    '.doc': 'Word Document (Legacy)',
    '.txt': 'Text File',
    '.xlsx': 'Excel Spreadsheet',
    '.xls': 'Excel Spreadsheet (Legacy)'
}

# Global variables - Updated for multiple files
vector_db = None
chain = None
document_loaded = False
current_document_paths = []  # Changed to list for multiple files

def pull_model_if_needed(model_name, base_url):
    """Pull model if not already available in Docker container."""
    try:
        response = requests.get(f"{base_url}/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if model_name not in model_names:
                logging.info(f"Pulling model {model_name}...")
                pull_response = requests.post(f"{base_url}/api/pull", 
                                            json={"name": model_name})
                if pull_response.status_code == 200:
                    logging.info(f"Model {model_name} pulled successfully.")
                else:
                    logging.error(f"Failed to pull model {model_name}")
            else:
                logging.info(f"Model {model_name} already available.")
    except Exception as e:
        logging.warning(f"Could not check/pull model: {e}")

def get_file_type(file_path):
    """Determine the file type based on extension."""
    _, ext = os.path.splitext(file_path.lower())
    return ext

def load_excel_with_pandas(file_path):
    """Load Excel file using pandas and convert to LangChain Document format."""
    try:
        # Read all sheets
        excel_file = pd.ExcelFile(file_path)
        all_text = []
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            # Convert DataFrame to string representation
            sheet_text = f"Sheet: {sheet_name}\n"
            sheet_text += df.to_string(index=False)
            all_text.append(sheet_text)
        
        # Combine all sheets into one document
        combined_text = "\n\n".join(all_text)
        
        # Create LangChain Document
        doc = Document(
            page_content=combined_text,
            metadata={"source": file_path, "file_type": "excel"}
        )
        
        return [doc]  # Return as list to match other loaders
    except Exception as e:
        logging.error(f"Error loading Excel file: {e}")
        return None

def ingest_document(doc_path):
    """Load documents based on file type."""
    if not os.path.exists(doc_path):
        logging.error(f"Document file not found at path: {doc_path}")
        return None
    
    file_ext = get_file_type(doc_path)
    
    try:
        if file_ext == '.pdf':
            loader = PyPDFLoader(file_path=doc_path)
            data = loader.load()
        elif file_ext in ['.docx', '.doc']:
            loader = Docx2txtLoader(doc_path)
            data = loader.load()
        elif file_ext == '.txt':
            loader = TextLoader(doc_path, encoding='utf-8')
            data = loader.load()
        elif file_ext in ['.xlsx', '.xls']:
            # Use pandas instead of UnstructuredExcelLoader
            data = load_excel_with_pandas(doc_path)
        else:
            logging.error(f"Unsupported file type: {file_ext}")
            return None
        
        if data is None:
            return None
            
        logging.info(f"Document loaded successfully: {file_ext}")
        return data
    except Exception as e:
        logging.error(f"Error loading document: {e}")
        return None

def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks

def load_multiple_documents_to_vector_db(document_paths):
    """Load multiple documents into a single vector database."""
    pull_model_if_needed(EMBEDDING_MODEL, OLLAMA_BASE_URL)

    embedding = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    # Create a combined name for the vector DB
    doc_names = [os.path.splitext(os.path.basename(path))[0] for path in document_paths]
    combined_name = "_".join(doc_names[:3])  # Limit to first 3 names to avoid too long paths
    if len(doc_names) > 3:
        combined_name += f"_and_{len(doc_names)-3}_more"
    
    persist_dir = f"{PERSIST_DIRECTORY}_{combined_name}"

    # Check if vector DB already exists for this combination
    if os.path.exists(persist_dir):
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=f"{VECTOR_STORE_NAME}_{combined_name}",
            persist_directory=persist_dir,
        )
        logging.info(f"Loaded existing vector database for combined documents.")
        return vector_db

    # Load and process all documents
    all_chunks = []
    
    for doc_path in document_paths:
        logging.info(f"Processing document: {os.path.basename(doc_path)}")
        data = ingest_document(doc_path)
        if data is None:
            logging.warning(f"Failed to load document: {doc_path}")
            continue
        
        # Add source information to metadata
        for doc in data:
            if hasattr(doc, 'metadata'):
                doc.metadata['source_file'] = os.path.basename(doc_path)
            else:
                doc.metadata = {'source_file': os.path.basename(doc_path)}
        
        chunks = split_documents(data)
        all_chunks.extend(chunks)
    
    if not all_chunks:
        logging.error("No documents could be processed successfully.")
        return None

    # Create vector database from all chunks
    vector_db = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding,
        collection_name=f"{VECTOR_STORE_NAME}_{combined_name}",
        persist_directory=persist_dir,
    )
    vector_db.persist()
    logging.info(f"Vector database created with {len(all_chunks)} chunks from {len(document_paths)} documents.")
    return vector_db

def create_retriever(vector_db, llm):
    """Create a simple retriever."""
    retriever = vector_db.as_retriever(
        search_kwargs={"k": 4}
    )
    return retriever

def create_chain(retriever, llm):
    """Create the chain with source information."""
    template = """Answer the question based ONLY on the following context:
{context}

Question: {question}

Please provide the answer and mention which document(s) the information came from when relevant.
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Chain created with source tracking.")
    return chain

def is_supported_file(file_path):
    """Check if the file type is supported."""
    if not file_path:
        return False, "No file provided"
    
    file_ext = get_file_type(file_path)
    if file_ext in SUPPORTED_EXTENSIONS:
        return True, SUPPORTED_EXTENSIONS[file_ext]
    else:
        return False, f"Unsupported file type: {file_ext}"

def initialize_system(document_files):
    """Initialize the RAG system with multiple uploaded documents."""
    global vector_db, chain, document_loaded, current_document_paths
    
    if document_files is None or len(document_files) == 0:
        return "❌ Please upload at least one document file first."
    
    # Check if all file types are supported
    unsupported_files = []
    supported_files = []
    
    for doc_file in document_files:
        is_supported, file_type_info = is_supported_file(doc_file.name)
        if not is_supported:
            unsupported_files.append(f"{os.path.basename(doc_file.name)} ({file_type_info})")
        else:
            supported_files.append(doc_file.name)
    
    if unsupported_files:
        return f"❌ Unsupported files: {', '.join(unsupported_files)}. Supported formats: {', '.join(SUPPORTED_EXTENSIONS.values())}"
    
    try:
        current_document_paths = supported_files
        
        # Pull the main model if needed
        pull_model_if_needed(MODEL_NAME, OLLAMA_BASE_URL)
        
        # Initialize the language model
        llm = ChatOllama(
            model=MODEL_NAME,
            base_url=OLLAMA_BASE_URL
        )

        # Load the vector database with all documents
        vector_db = load_multiple_documents_to_vector_db(current_document_paths)
        if vector_db is None:
            return "❌ Failed to load or create the vector database."

        # Create the retriever and chain
        retriever = create_retriever(vector_db, llm)
        chain = create_chain(retriever, llm)
        document_loaded = True
        
        doc_names = [os.path.basename(path) for path in current_document_paths]
        return f"✅ System initialized successfully with {len(doc_names)} documents: {', '.join(doc_names)}! Ready to answer questions."
    except Exception as e:
        return f"❌ Error initializing system: {str(e)}"

def clear_document():
    """Clear the uploaded documents from the interface."""
    global vector_db, chain, document_loaded, current_document_paths
    
    # Reset current session
    vector_db = None
    chain = None
    document_loaded = False
    current_document_paths = []
    
    return None, "📄 Documents cleared. Upload new documents to continue."

def reset_vector_db():
    """Reset the vector database."""
    global vector_db, chain, document_loaded, current_document_paths
    
    try:
        # Remove all vector databases
        import glob
        db_dirs = glob.glob(f"{PERSIST_DIRECTORY}_*")
        for db_dir in db_dirs:
            if os.path.exists(db_dir):
                shutil.rmtree(db_dir)
        
        vector_db = None
        chain = None
        document_loaded = False
        current_document_paths = []
        return None, "✅ All vector databases reset successfully. Upload documents and click 'Initialize System' to start."
    except Exception as e:
        return None, f"❌ Error resetting database: {str(e)}"

def chat_with_document(message, history):
    """Handle chat with document functionality."""
    global chain, document_loaded
    
    if not document_loaded or chain is None:
        response = "❌ Please initialize the system first by clicking the 'Initialize System' button."
        history.append([message, response])
        return history, ""
    
    if not message.strip():
        return history, ""
    
    try:
        response = chain.invoke(input=message)
        history.append([message, response])
        return history, ""
    except Exception as e:
        error_response = f"❌ Error generating response: {str(e)}"
        history.append([message, error_response])
        return history, ""

def get_system_status():
    """Get current system status in compact format."""
    global document_loaded, current_document_paths
    
    status_info = []
    
    if current_document_paths:
        doc_names = [os.path.basename(path) for path in current_document_paths]
        if len(doc_names) <= 2:
            status_info.append(f"📄 Documents: {', '.join(doc_names)}")
        else:
            status_info.append(f"📄 Documents: {len(doc_names)} files ({', '.join(doc_names[:2])}, ...)")
    else:
        status_info.append("📄 No documents uploaded")
    
    # Check for existing vector databases
    import glob
    db_dirs = glob.glob(f"{PERSIST_DIRECTORY}_*")
    status_info.append(f"🗄️ Vector DBs: {len(db_dirs)} | System: {'✅ Ready' if document_loaded else '❌ Not Ready'}")
    
    return "\n".join(status_info)

# Create Gradio interface
with gr.Blocks(title="Llama-Chat", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📚 Llama-Chat")
    #gr.Markdown("Upload various document formats (PDF, Word, Excel, Text) and chat with them using advanced RAG technology.")
    
    with gr.Row():
        with gr.Column(scale=2):
            # Document Upload and System Controls
            gr.Markdown("## 📤 Upload Documents")
            
            document_upload = gr.File(
                label="Upload Multiple PDF/Word/Excel/TXT files",
                file_types=[".pdf", ".docx", ".doc", ".txt", ".xlsx", ".xls"],
                file_count="multiple"  # Changed from "single" to "multiple"
            )
            
            gr.Markdown("## 🔧 System Controls")
            
            with gr.Row():
                init_btn = gr.Button("🚀 Initialize System", variant="primary")
                clear_btn = gr.Button("🗑️ Clear Documents", variant="secondary")
            
            with gr.Row():
                reset_btn = gr.Button("🔄 Reset All Databases", variant="secondary")
                status_btn = gr.Button("📊 Check Status", variant="secondary")
            
            status_output = gr.Textbox(
                label="System Status",
                lines=3,
                interactive=False,
                value="Upload documents and click 'Check Status' to see system information"
            )
            
        with gr.Column(scale=3):
            # Chat interface
            gr.Markdown("## 💬 Chat Interface")
            
            chatbot = gr.Chatbot(
                label="Document Chat",
                height=400,
                show_label=True,
                container=True
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask questions about your documents...",
                    show_label=False,
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
    
    # Event handlers
    init_btn.click(
        fn=initialize_system,
        inputs=document_upload,
        outputs=status_output
    )
    
    clear_btn.click(
        fn=clear_document,
        outputs=[document_upload, status_output]
    )
    
    reset_btn.click(
        fn=reset_vector_db,
        outputs=[document_upload, status_output]
    )
    
    status_btn.click(
        fn=get_system_status,
        outputs=status_output
    )
    
    # Chat functionality
    msg.submit(
        fn=chat_with_document,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )
    
    send_btn.click(
        fn=chat_with_document,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg]
    )
    
    # Instructions
    with gr.Accordion("📋 Instructions", open=False):
        gr.Markdown("""
        ### How to use this application:
        
        1. **Upload Documents**: Click "Upload Documents" and select multiple files (PDF, Word, Excel, or Text)
        2. **Initialize System**: Click "🚀 Initialize System" to process all documents and create the vector database
        3. **Check Status**: Use "📊 Check Status" to verify system readiness and see loaded documents
        4. **Start Chatting**: Once initialized, type your questions in the chat box
        5. **Clear Documents**: Use "🗑️ Clear Documents" to remove current documents and upload new ones
        6. **Reset if needed**: Use "🔄 Reset All Databases" to clear all processed documents from storage
        
        ### Supported Document Types:
        - **PDF Documents** (.pdf) - Full text extraction from PDF files
        - **Word Documents** (.docx, .doc) - Microsoft Word documents
        - **Excel Spreadsheets** (.xlsx, .xls) - Excel files with data extraction
        - **Text Files** (.txt) - Plain text documents
        
        ### Multi-Document Features:
        - Upload and process multiple documents simultaneously
        - Ask questions that span across all uploaded documents
        - System automatically tracks which documents contain relevant information
        - Compare information between different documents
        - Single vector database combines all document content for comprehensive search
        
        ### Tips:
        - You can upload different file types together (e.g., PDF + Excel + Word)
        - For Excel files, the system will extract text content from all sheets
        - Large documents may take longer to process initially
        - Once processed, subsequent chats are much faster
        - Ask comparative questions like "What are the differences between document A and B?"
        """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )