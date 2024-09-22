import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import GPT4AllEmbeddings

# Define the directory containing the text file and persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")
model_embedding = "E:/Self_Learning_Code/LangChain_RAG_SelfStudy/models/all-MiniLM-L6-v2-f16.gguf"


# print(f"Books directory: {books_dir}")
# print(f"Persistent directory: {persistent_directory}")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing Chroma vector store.")
    
    # Ensure the text files exist
    if not os.path.exists(books_dir):
        raise FileNotFoundError(f"Directory {books_dir} does not exist.")
    
    books_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]
    
    # print("books_files: ", books_files, "\nlen:", len(books_files))
    documents = []
    
    for book_file in books_files:
        file_path = os.path.join(books_dir, book_file)
        
        # Read the text content from the file
        loader = TextLoader(file_path, encoding='UTF-8')
        book_docs = loader.load()
        
        for doc in book_docs:
            doc.metadata = {"source": book_file}
            documents.append(doc)
        
    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    
    # Create embeddings for the documents
    print("\n--- Creating embeddings for the documents ---")
    
    # # Initialize GPT4AllEmbeddings with the correct model file
    # embeddings = GPT4AllEmbeddings(model_file = model_embedding)     # type: ignore
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    print("\n--- Finished creating embeddings for the documents ---")

    # Create the vector store and persist it automatically
    print("\n--- Creating the Chroma vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory
    )
    
else:
    print("Persistent directory exists. Loading Chroma vector store.")