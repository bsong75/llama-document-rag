## Llama RAG Assistant 
 -- To build.  docker-compose up -d --build --force-recreate gradio-app

For Speed:
1. Create_retriever function
    search_kwargs={"k": 2}  # reduced from 4 to 2 chunks for speed
2. split_documents fuction.   
    chunk_size=800,    # Reduce from 1200
    chunk_overlap=200  # Reduce from 300


# Llama-Chat RAG System

## Overview
Llama-Chat is a Retrieval Augmented Generation (RAG) application that allows users to upload multiple documents and chat with them using natural language. It combines document processing, vector storage, and conversational AI to provide intelligent document querying capabilities.

## Core Architecture

### Technology Stack
- **Frontend**: Gradio web interface
- **AI Model**: Gemma 3:4b (via Ollama API)
- **Embeddings**: Nomic-embed-text model
- **Vector Database**: ChromaDB with persistent storage
- **Document Processing**: LangChain with multiple loaders
- **Framework**: LangChain for RAG pipeline orchestration

### Key Features
- **Multi-format Support**: PDF, Word, Excel, and Text files
- **Batch Processing**: Upload and process multiple documents simultaneously
- **Persistent Storage**: Vector databases survive app restarts
- **Source Tracking**: Responses include information about source documents
- **Document Management**: Clear, reset, and re-initialize capabilities

## Technical Implementation Details

### 1. System Initialization Process

**How it works in plain English:**
When you click "Initialize System," the app takes all your uploaded documents and turns them into a searchable knowledge base. It's like creating a super-smart library index that understands the meaning of content, not just keywords. The system reads each document, breaks it into digestible chunks, converts those chunks into mathematical representations that capture their meaning, and stores everything in a special database that can quickly find relevant information when you ask questions.

**Technical Implementation:**
```python
def initialize_system(document_files):
    # 1. Validate uploaded files
    for doc_file in document_files:
        is_supported, file_type_info = is_supported_file(doc_file.name)
    
    # 2. Pull required AI models if not available
    pull_model_if_needed(MODEL_NAME, OLLAMA_BASE_URL)
    pull_model_if_needed(EMBEDDING_MODEL, OLLAMA_BASE_URL)
    
    # 3. Initialize language model
    llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)
    
    # 4. Create vector database from all documents
    vector_db = load_multiple_documents_to_vector_db(current_document_paths)
    
    # 5. Create retriever and processing chain
    retriever = create_retriever(vector_db, llm)
    chain = create_chain(retriever, llm)
    
    # 6. Mark system as ready
    document_loaded = True
```

**Step-by-step Process:**
1. **File Validation**: Checks all uploaded files are supported formats
2. **Model Preparation**: Downloads AI models (Gemma 3:4b, Nomic-embed-text) if needed
3. **Document Processing**: Loads and processes all documents into vector database
4. **Chain Creation**: Sets up the question-answering pipeline
5. **System Activation**: Marks system as ready for queries

### 2. Document Parsing and Vector Storage

**How it works in plain English:**
The system treats each document type differently but follows the same basic pattern: read the content, break it into small overlapping chunks (like tearing pages into strips with some overlap so context isn't lost), convert each chunk into a mathematical "fingerprint" that represents its meaning, and store both the original text and the fingerprint in a database. This way, when you ask a question later, the system can quickly find the most relevant chunks from any document.

**Technical Implementation:**
```python
def load_multiple_documents_to_vector_db(document_paths):
    # 1. Set up embeddings
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    
    # 2. Process each document
    all_chunks = []
    for doc_path in document_paths:
        # Load based on file type
        data = ingest_document(doc_path)  # Uses appropriate loader
        
        # Add source metadata
        for doc in data:
            doc.metadata['source_file'] = os.path.basename(doc_path)
        
        # Split into chunks
        chunks = split_documents(data)
        all_chunks.extend(chunks)
    
    # 3. Create vector database
    vector_db = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding,
        collection_name=f"{VECTOR_STORE_NAME}_{combined_name}",
        persist_directory=persist_dir
    )
    
    return vector_db
```

**Document-Specific Loading:**
```python
def ingest_document(doc_path):
    file_ext = get_file_type(doc_path)
    
    if file_ext == '.pdf':
        loader = PyPDFLoader(file_path=doc_path)
    elif file_ext in ['.docx', '.doc']:
        loader = Docx2txtLoader(doc_path)
    elif file_ext == '.txt':
        loader = TextLoader(doc_path, encoding='utf-8')
    elif file_ext in ['.xlsx', '.xls']:
        # Custom Excel processing with pandas
        data = load_excel_with_pandas(doc_path)
    
    return loader.load()
```

**Text Chunking Process:**
```python
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,    # Each chunk ~1200 characters
        chunk_overlap=300   # 300 character overlap between chunks
    )
    chunks = text_splitter.split_documents(documents)
    return chunks
```

**Storage Structure:**
- **Documents**: Original text chunks with metadata
- **Embeddings**: 768-dimensional vectors (Nomic-embed-text output)
- **Metadata**: Source file, chunk position, document type
- **Persistence**: Saved to `./app/chroma_db_[document_names]/`

### 3. Vector Database Reset Process

**How it works in plain English:**
When you reset the vector database, the system essentially burns down the entire library and cleans the lot. It finds every knowledge base it has ever created for any set of documents, deletes all the files and folders containing that processed information, and resets all the internal tracking variables. After this, it's like the app has never seen any documents before - you'll need to upload and initialize everything from scratch.

**Technical Implementation:**
```python
def reset_vector_db():
    global vector_db, chain, document_loaded, current_document_paths
    
    try:
        # 1. Find all vector database directories
        import glob
        db_dirs = glob.glob(f"{PERSIST_DIRECTORY}_*")
        
        # 2. Delete each database directory completely
        for db_dir in db_dirs:
            if os.path.exists(db_dir):
                shutil.rmtree(db_dir)  # Recursive deletion
        
        # 3. Reset all global variables
        vector_db = None
        chain = None
        document_loaded = False
        current_document_paths = []
        
        return "✅ All vector databases reset successfully"
    except Exception as e:
        return f"❌ Error resetting database: {str(e)}"
```

**What Gets Deleted:**
- All ChromaDB collections and indices
- Vector embeddings for all previously processed documents
- Document chunk storage
- Metadata and source tracking information
- Persistent database files (`.sqlite3`, collection data)

**System State After Reset:**
- No documents loaded in memory
- No active retrieval chain
- No vector databases on disk
- Clean slate requiring full re-initialization

### 4. Question Handling Process

**How it works in plain English:**
When you ask a question, the system works like a smart research assistant. First, it converts your question into the same type of mathematical fingerprint it used for the document chunks. Then it searches through all the stored chunk fingerprints to find the most relevant pieces of information from your documents. It takes the top 4 most relevant chunks, combines them with your original question, and sends everything to the AI with instructions to answer based only on the provided context and to mention which documents the information came from.

**Technical Implementation:**
```python
def chat_with_document(message, history):
    # 1. Validate system is ready
    if not document_loaded or chain is None:
        return error_message
    
    # 2. Process through the RAG chain
    response = chain.invoke(input=message)
    
    # 3. Update conversation history
    history.append([message, response])
    return history, ""
```

**The RAG Chain Process:**
```python
def create_chain(retriever, llm):
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
    return chain
```

**Step-by-step Question Processing:**

1. **Question Embedding**: Convert user question to vector using Nomic-embed-text
   ```python
   query_vector = embedding_model.embed_query(user_question)
   ```

2. **Similarity Search**: Find most relevant document chunks
   ```python
   retriever = vector_db.as_retriever(search_kwargs={"k": 4})
   relevant_chunks = retriever.get_relevant_documents(question)
   ```

3. **Context Assembly**: Combine retrieved chunks into context
   ```python
   context = "\n\n".join([doc.page_content for doc in relevant_chunks])
   ```

4. **Prompt Construction**: Create structured prompt for AI
   ```python
   final_prompt = f"""Answer based ONLY on: {context}
   Question: {question}
   Include source information."""
   ```

5. **AI Response**: Generate answer using Gemma 3:4b
   ```python
   response = llm.invoke(final_prompt)
   ```

6. **Source Attribution**: AI includes document source information in response

## Data Flow Summary

```
User Question → Embed Question → Search Vector DB → Retrieve Top 4 Chunks
     ↓                                                       ↓
Update Chat History ← Format Response ← AI Processing ← Construct Prompt
```

## File Organization Structure

```
./app/
├── chroma_db_document1_document2/     # Vector DB for specific document set
│   ├── chroma.sqlite3                 # ChromaDB index
│   ├── [collection_files]             # Vector embeddings
│   └── [metadata_files]               # Document metadata
├── chroma_db_report_analysis/         # Another document set
└── [additional_vector_dbs]/           # More document combinations
```

## Key Advantages

1. **Multi-Document Intelligence**: Can answer questions spanning multiple documents
2. **Source Attribution**: Always tells you which document contains the information
3. **Efficient Retrieval**: Only processes most relevant chunks, not entire documents
4. **Persistent Memory**: Vector databases survive app restarts
5. **Format Flexibility**: Handles diverse document types seamlessly
6. **Batch Processing**: Process multiple documents simultaneously
7. **Context Preservation**: Overlapping chunks maintain context continuity

## Performance Characteristics

- **Initialization**: ~30-60 seconds for multiple large documents
- **Query Response**: ~2-5 seconds per question
- **Storage**: ~1-2MB vector data per 100 pages of text
- **Accuracy**: High relevance due to semantic search vs keyword matching
- **Scalability**: Handles hundreds of documents efficiently

This RAG system creates an intelligent document assistant that can understand context, maintain source attribution, and provide accurate answers based on uploaded document content.