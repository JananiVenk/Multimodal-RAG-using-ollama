# 🔍 Multimodal RAG using Ollama

A locally-running Retrieval-Augmented Generation (RAG) chatbot that understands both **text and images** from PDF documents — powered by CLIP embeddings, FAISS vector search, and Ollama LLMs.

---

## ✨ Features

- 📄 **PDF ingestion** — extracts and indexes both text chunks and embedded images
- 🖼️ **Multimodal embeddings** — uses OpenAI's CLIP (`clip-vit-base-patch32`) to embed text and images into a shared vector space
- 🔎 **Semantic search** — FAISS vector store for fast similarity retrieval
- 🧠 **Conversation memory** — windowed chat history so the LLM remembers past turns
- 💬 **Chat UI** — clean Streamlit chat interface with message bubbles
- 🔒 **Fully local** — no API keys required; runs entirely on your machine via Ollama

---

## 🏗️ Architecture

<img width="836" height="365" alt="image" src="https://github.com/user-attachments/assets/86ae55e8-9bb8-45d9-b990-3e25378f8e5a" />


## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally

### 1. Clone the repo

```bash
git clone https://github.com/JananiVenk/Multimodal-RAG-using-ollama.git
cd Multimodal-RAG-using-ollama
```

### 2. Install dependencies

```bash
pip install streamlit pymupdf langchain langchain-community langchain-ollama \
            langchain-text-splitters transformers torch torchvision pillow \
            faiss-cpu scikit-learn safetensors
```

### 3. Pull the Ollama model

```bash
ollama pull gemma3:4b
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## 🖥️ Usage

1. Open the app in your browser at `http://localhost:8501`
2. Upload a PDF using the sidebar
3. Wait for indexing to complete
4. Ask questions in the chat box — the app will retrieve relevant text and images and generate an answer

---

## ⚙️ Configuration

All settings are available in the sidebar:

| Setting | Description | Default |
|---|---|---|
| Messages to remember | Number of past chat turns sent to the LLM | 6 |

To change the Ollama model, edit this line in `app.py`:
```python
return ChatOllama(model="gemma3:4b")
```
Any model available in your local Ollama instance can be used (e.g. `llama3.2-vision`, `mistral`, `deepseek-r1:1.5b`).

---

## 📦 Tech Stack

| Component | Library |
|---|---|
| UI | Streamlit |
| PDF parsing | PyMuPDF (`fitz`) |
| Embeddings | CLIP (`openai/clip-vit-base-patch32`) via HuggingFace Transformers |
| Vector store | FAISS (`langchain-community`) |
| LLM | Ollama via `langchain-ollama` |
| Text splitting | LangChain `RecursiveCharacterTextSplitter` |

---

