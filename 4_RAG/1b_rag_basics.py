import os

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Define the directory containing the text file and persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, 'db', 'chroma_db')

# Define the embeddings model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Load the Chroma vector store
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# Define the query
query = "Who is the Odysseus's wife?"

# Retrieve the relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "threshold": 0.5}
)

relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")