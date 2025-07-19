import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Define documents
documents = [
    "The capital of France is Paris.",
    "The Eiffel Tower is in Paris.",
    "Berlin is the capital of Germany.",
    "The Brandenburg Gate is in Berlin.",
    "Tokyo is the capital of Japan."
]

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode documents
doc_embeddings = model.encode(documents)

# Build FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# Streamlit UI
st.title("RAG Demo: FAISS + Streamlit")

query = st.text_input("Enter your query:")

if query:
    query_embedding = model.encode([query])
    k = 2
    distances, indices = index.search(np.array(query_embedding), k)

    st.subheader("Top Results:")
    for idx in indices[0]:
        st.write(f"- {documents[idx]}")
