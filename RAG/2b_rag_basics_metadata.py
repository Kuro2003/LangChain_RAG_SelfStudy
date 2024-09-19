import os 

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Define the directory containing the text file and persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")
model_embedding = "E:/Self_Learning_Code/LangChain_RAG_SelfStudy/models/all-MiniLM-L6-v2-f16.gguf"


# Define the embeddings model
embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-small"
)

# Load the Chroma vector store
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# Define the query
query = "How did Juliet die?"

# Retrieve the relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "threshold": 0.1}
)

relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    print(f"Source: {doc.metadata['source']}\n")