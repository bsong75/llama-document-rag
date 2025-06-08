# improved RAG Chat with PDF

1. Added global state management for the RAG system
2. Created a more comprehensive interface with:

3. System initialization controls
4. Status monitoring
5. Reset functionality
6. Better chat interface with history

TO DO:
1. Need to separate out the llama to use API
2. 

For Speed:
1. Create_retriever function
    search_kwargs={"k": 2}  # reduced from 4 to 2 chunks for speed
2. split_documents fuction.   
    chunk_size=800,    # Reduce from 1200
    chunk_overlap=200  # Reduce from 300


docker-compose up -d --build --force-recreate gradio-app

