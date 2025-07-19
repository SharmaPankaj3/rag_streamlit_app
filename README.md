# RAG Streamlit App — Day 7 of GenAI Journey
## About the project
## This is a simple Retrieval‑Augmented Generation (RAG) demo app built as part of my 45‑day GenAI/NLP developer training journey. It uses:
🔷 **SentenceTransformers** to embed documents & queries
🔷 **FAISS** as a vector database to store & retrieve embeddings
🔷 **Streamlit** to provide a user‑friendly web app interface

✅ Enter a query → app searches top‑k relevant documents → shows results.

![screenshot](<img width="1186" height="800" alt="image" src="https://github.com/user-attachments/assets/6194477a-3e4f-40a6-bf7c-997576242a4a" />
)
# Tech stack: 
Python 3.7+
Streamlit
SentenceTransformers
FAISS
# Installation & Running
git clone https://github.com/<your-username>/rag_streamlit_app.git
cd rag_streamlit_app
# Install dependencies
pip install -r requirements.txt
# Run the app
streamlit run app.py
Then open your browser at: http://localhost:8501
