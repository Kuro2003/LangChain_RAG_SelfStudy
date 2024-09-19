import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Define the directory containing the text file and persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
persistent_dir = os.path.join(current_dir, 'db', 'chroma_db')

# Check if the Chroma vector alrady exists
if not os.path.exists(persistent_dir):
    print("Persitent directory does not exist. Initializing Chroma vector store.")
    
    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    
    # Read the text content from the file
    loader = TextLoader(file_path,encoding = 'UTF-8')
    documents = loader.load()
    
    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    # # Display information about the split docuuments
    print("\n--- Document Chunks Information ---")
    # print(f"Number of documment chunks:{len(docs)}")
    # print(f"Sample chunk:\n{docs[0].page_content}\n")
    
    # Create embeddings for the documents
    print("\n--- Creating embeddings for the documents ---")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    print("\n--- Finished creating embeddings for the documents ---")
    
    # Create the vector store and persist it automatically
    print("\n--- Creating the Chroma vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_dir
    )
    
    print("\n--- Finished creating the Chroma vector store ---")

else:
    print("Persitent directory exists. Loading Chroma vector store.")    