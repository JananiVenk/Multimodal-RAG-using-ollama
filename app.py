import streamlit as st
import fitz
from langchain_core.documents import Document 
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np 
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import base64
import io
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
import re

st.set_page_config(page_title="Multimodal RAG Chat", page_icon="🔍", layout="wide")
st.title("🔍 Multimodal RAG Chat")

# ── Load models ──
@st.cache_resource(show_spinner="Loading CLIP model…")
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
    model.eval()
    return model, processor

clip_model, clip_processor = load_clip()

@st.cache_resource(show_spinner="Connecting to Ollama…")
def load_llm():
    from langchain_ollama import ChatOllama
    return ChatOllama(model="gemma3:4b")

llm = load_llm()

# ── Embedding helpers ──
def embed_text(text):
    inputs = clip_processor(text=text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().numpy()

def embed_image(image):
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().numpy()

# ── Session state init ──
for key, default in {
    "messages": [],          
    "chat_history": [],      
    "vector_store": None,
    "image_data_store": {},
    "processed_file": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ──
with st.sidebar:
    st.header("📄 Document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file is not None and st.session_state.processed_file != uploaded_file.name:
        all_docs = []
        all_embeddings = []
        image_data_store = {}
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        total_pages = len(doc)
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, page in enumerate(doc):
            progress_bar.progress(int((i + 1) / total_pages * 100))
            status_text.text(f"Processing page {i+1}/{total_pages}…")

            text = page.get_text()
            if text.strip():
                temp_doc = Document(page_content=text, metadata={"page": i+1, "type": "text"})
                for chunk in splitter.split_documents([temp_doc]):
                    all_embeddings.append(embed_text(chunk.page_content))
                    all_docs.append(chunk)

            for img_index, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    pil_image = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
                    image_id = f"page_{i+1}_img_{img_index}"
                    buffered = io.BytesIO()
                    pil_image.save(buffered, format="PNG")
                    image_data_store[image_id] = base64.b64encode(buffered.getvalue()).decode()
                    all_embeddings.append(embed_image(pil_image))
                    all_docs.append(Document(
                        page_content=f"[Image: {image_id}]",
                        metadata={"page": i+1, "type": "image", "image_id": image_id}
                    ))
                except Exception as e:
                    st.warning(f"Skipped image on page {i+1}: {e}")

        progress_bar.empty()
        status_text.empty()

        embeddings_array = np.array(all_embeddings)
        dim = embeddings_array.shape[1]
        st.session_state.vector_store = FAISS.from_embeddings(
            text_embeddings=[(d.page_content, e.tolist()) for d, e in zip(all_docs, embeddings_array)],
            embedding=FakeEmbeddings(size=dim),
            metadatas=[d.metadata for d in all_docs]
        )
        st.session_state.image_data_store = image_data_store
        st.session_state.processed_file = uploaded_file.name
        st.session_state.messages = []
        st.session_state.chat_history = []
        doc.close()
        st.success(f"✅ Indexed {len(all_docs)} chunks")

    if st.session_state.processed_file:
        st.caption(f"📎 {st.session_state.processed_file}")

    st.divider()
    st.header("🧠 Memory")
    memory_window = st.slider("Messages to remember", min_value=2, max_value=20, value=6, step=2,
                               help="How many past messages the LLM sees")

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# ── Helpers ──
def get_windowed_history(window: int) -> list:
    """Return the last N messages from chat_history."""
    return st.session_state.chat_history[-window:]

def build_messages(query: str, context: str, history: list) -> list:
    system = SystemMessage(content=(
        "You are a helpful assistant that answers questions about a document. "
        "Use the provided context and conversation history to give accurate, concise answers. "
        "If the answer is not in the context, say so honestly."
    ))
    # Inject context into the current user turn
    user_msg = HumanMessage(content=(
        f"Context from document:\n{context}\n\n"
        f"Question: {query}"
    ))
    return [system] + history + [user_msg]

# ── Chat area ──
if st.session_state.vector_store is None:
    st.info("👈 Upload a PDF in the sidebar to get started.")
else:
    # Render existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("images"):
                cols = st.columns(min(len(msg["images"]), 3))
                for col, img_b64 in zip(cols, msg["images"]):
                    col.image(base64.b64decode(img_b64))

    # Chat input
    if query := st.chat_input("Ask anything about the document…"):

        with st.chat_message("user"):
            st.write(query)
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                # Retrieve relevant chunks
                results = st.session_state.vector_store.similarity_search_by_vector(
                    embed_text(query).tolist(), k=5
                )

                context_texts = []
                retrieved_images = []
                for doc in results:
                    if doc.metadata["type"] == "text":
                        context_texts.append(f"[Page {doc.metadata['page']}]: {doc.page_content}")
                    elif doc.metadata["type"] == "image":
                        retrieved_images.append(doc.metadata["image_id"])

                context = "\n".join(context_texts)

                # Build messages with windowed memory
                history = get_windowed_history(memory_window)
                messages = build_messages(query, context, history)

                response = llm.invoke(messages)
                answer = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()

            st.write(answer)

            img_b64_list = []
            if retrieved_images:
                st.caption("📎 Related images")
                cols = st.columns(min(len(retrieved_images), 3))
                for col, img_id in zip(cols, retrieved_images):
                    if img_id in st.session_state.image_data_store:
                        b64 = st.session_state.image_data_store[img_id]
                        col.image(base64.b64decode(b64))
                        img_b64_list.append(b64)

        # Update memory: store raw query (without context) so history stays concise
        st.session_state.chat_history.append(HumanMessage(content=query))
        st.session_state.chat_history.append(AIMessage(content=answer))

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "images": img_b64_list
        })