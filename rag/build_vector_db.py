import os
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def build_vector_db():
    print("⏳ Loading fraud patterns...")
    kb_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base", "patterns.json")
    
    with open(kb_path, "r") as f:
        patterns = json.load(f)
        
    documents = []
    for pattern in patterns:
        content = f"Title: {pattern['title']}\nDescription: {pattern['description']}\nIndicators: {', '.join(pattern['risk_indicators'])}"
        metadata = {
            "title": pattern["title"],
            "recommended_action": pattern["recommended_action"]
        }
        documents.append(Document(page_content=content, metadata=metadata))
        
    print(f"✅ Loaded {len(documents)} patterns.")
    
    print("⏳ Initializing embedding model (all-MiniLM-L6-v2)...")
    # Using open source HuggingFace embeddings for local, free vectorization
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("⏳ Building FAISS Vector Store...")
    vector_db = FAISS.from_documents(documents, embeddings)
    
    save_path = os.path.join(os.path.dirname(__file__), "faiss_index")
    os.makedirs(save_path, exist_ok=True)
    
    vector_db.save_local(save_path)
    print(f"✅ Vector database successfully built and saved to {save_path}")

if __name__ == "__main__":
    build_vector_db()
