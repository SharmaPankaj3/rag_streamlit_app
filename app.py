# Step1: Import Libraries
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 2: Sample Knowledge Base
documents = [
    "The capital of France is Paris.",
    "The Eiffel Tower is in Paris.",
    "Berlin is the capital of Germany.",
    "The Brandenburg Gate is in Berlin.",
    "Tokyo is the capital of Japan."
]

# Step 3: Load Embedding Model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()


# Step 4: Create Document Embeddings & FAISS Index
doc_embeddings = model.encode(documents,normalize_embeddings= True)
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# Step 5: Streamlit UI
st.title("RAG Demo: FAISS + Streamlit")

query = st.text_input("Enter your query:")

if query:
    query_embedding = model.encode([query],normalize_embeddings=True)
    k = 2
    distances, indices = index.search(np.array(query_embedding), k)

    st.subheader("Top Results:")
    for idx in indices[0]:
        st.write(f"- {documents[idx]}")
