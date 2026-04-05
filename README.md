                                       # Local RAG Intelligence System

A private Retrieval-Augmented Generation (RAG) pipeline built with **LangChain** and **Llama 3**.

## 🚀 Features
* **100% Private:** Runs entirely on local hardware using **Ollama**.
* **Semantic Search:** Uses **FAISS** and **HuggingFace Embeddings** for document retrieval.
* **Efficient Pipe Architecture:** Built using **LCEL** for modular AI workflows.

## 🛠️ Tech Stack
* **LLM:** Llama 3 (via Ollama)
* **Framework:** LangChain
* **Vector Store:** FAISS
* **Embeddings:** sentence-transformers/all-MiniLM-L6-v2

## 📋 How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Ensure **Ollama** is running with `ollama run llama3`.
3. Place your documents in the `/data` folder.
4. Run the application: `python app.py`