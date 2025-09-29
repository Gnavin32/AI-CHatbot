# app.py
import os
import re
import tempfile
from typing import List, Dict

import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain_mistralai import MistralAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader


from dotenv import load_dotenv
# -------------------------
# Load API keys
# -------------------------
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# -------------------------
# Helper: resume metadata extraction
# -------------------------
EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
PHONE_RE = re.compile(r'(\+?\d[\d\-\s]{7,}\d)')
NAME_HINTS = ["name", "candidate", "resume", "curriculum vitae", "cv"]

def extract_metadata(text: str) -> Dict:
    emails = EMAIL_RE.findall(text)
    email = emails[0] if emails else None

    phones = PHONE_RE.findall(text)
    phone = phones[0] if phones else None

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    name = None
    for i in range(min(6, len(lines))):
        candidate = lines[i]
        if EMAIL_RE.search(candidate):
            continue
        low = candidate.lower()
        if any(h in low for h in NAME_HINTS):
            continue
        if 2 <= len(candidate.split()) <= 4 and re.search(r'[A-Za-z]', candidate):
            name = candidate
            break

    skills = []
    m = re.search(r'(skills|technical skills|skills & abilities|skills:)\s*(.*)', text, re.IGNORECASE)
    if m:
        skills_line = m.group(2)
        skills = [s.strip() for s in re.split(r'[,\n;/•]', skills_line) if s.strip()][:20]

    return {"name": name, "email": email, "phone": phone, "skills": skills}

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Resume RAG Chatbot", layout="wide")
st.title("📄 Resume RAG Chatbot — with FAISS + Mistral Embeddings")

with st.sidebar:
    st.header("Upload & Settings")
    uploaded_files = st.file_uploader("Upload resume PDF files (multiple)", type=["pdf"], accept_multiple_files=True)
    chunk_size = st.number_input("Chunk size (chars)", value=1200, step=100)
    chunk_overlap = st.number_input("Chunk overlap (chars)", value=200, step=50)
    reindex = st.button("(Re)index uploaded PDFs")

# Show uploaded files
if uploaded_files:
    st.write(f"Uploaded {len(uploaded_files)} file(s):")
    for f in uploaded_files:
        st.write(f"- {f.name}")

# Initialize embeddings (Mistral)
if not MISTRAL_API_KEY:
    st.error("⚠️ Please set MISTRAL_API_KEY in your .env file.")
else:
    embeddings = MistralAIEmbeddings(model='mistral-embed',api_key=MISTRAL_API_KEY)

# -------------------------
# Indexing function
# -------------------------
def load_pdfs_and_index(files, embeddings, chunk_size=1200, chunk_overlap=200):
    docs_for_index = []
    metadata_list = []

    for uploaded in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        try:
            loader = UnstructuredPDFLoader(tmp_path)
            raw_docs = loader.load()
            if isinstance(raw_docs, list) and raw_docs:
                for d in raw_docs:
                    docs_for_index.append(Document(page_content=d.page_content, metadata={"source": uploaded.name}))
            else:
                docs_for_index.append(Document(page_content=raw_docs.page_content, metadata={"source": uploaded.name}))
        except Exception:
            try:
                loader = PyPDFLoader(tmp_path)
                raw_docs = loader.load()
                for d in raw_docs:
                    docs_for_index.append(Document(page_content=d.page_content, metadata={"source": uploaded.name}))
            except Exception as e2:
                st.warning(f"Failed to load {uploaded.name}: {e2}")

        try:
            full_text = ""
            for d in docs_for_index:
                if d.metadata.get("source") == uploaded.name:
                    full_text += d.page_content + "\n"
            meta = extract_metadata(full_text)
            meta["filename"] = uploaded.name
            metadata_list.append(meta)
        except Exception:
            metadata_list.append({"filename": uploaded.name})

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = []
    for d in docs_for_index:
        chunks = text_splitter.split_text(d.page_content)
        for i, c in enumerate(chunks):
            split_docs.append(Document(page_content=c, metadata={**d.metadata, "chunk": i}))

    if not split_docs:
        st.error("No text extracted from any PDF. If resumes are scanned images, OCR is required.")
        return None, None

    vectordb = FAISS.from_documents(split_docs, embeddings)
    return vectordb, metadata_list

# Trigger indexing
if reindex and uploaded_files:
    with st.spinner("Indexing PDFs into FAISS..."):
        vectordb, metadata_list = load_pdfs_and_index(uploaded_files, embeddings, chunk_size, chunk_overlap)
    if vectordb:
        st.success("Indexing done!")
        st.session_state['vectordb'] = vectordb
        st.session_state['metadata'] = metadata_list

vectordb = st.session_state.get('vectordb')
metadata_store = st.session_state.get('metadata', [])

# -------------------------
# Candidate Search
# -------------------------
st.header("🔎 Candidate Finder")
query_candidate = st.text_input("Find candidate by name, email, skill (e.g., 'John Doe' or 'react')")

if st.button("Search candidate") and query_candidate:
    if not vectordb:
        st.error("No index found. Upload PDFs and press (Re)index first.")
    else:
        results = vectordb.similarity_search_with_score(query_candidate, k=6)
        candidates = {}
        for doc, score in results:
            src = doc.metadata.get("source", "unknown")
            if src not in candidates:
                candidates[src] = {"score": score, "snippets": [doc.page_content[:500]]}
            else:
                candidates[src]["snippets"].append(doc.page_content[:300])

        rows = []
        for meta in metadata_store:
            rows.append(meta)

        st.subheader("Top matching resumes:")
        if not candidates:
            st.write("No matches.")
        else:
            for src, info in sorted(candidates.items(), key=lambda x: x[1]["score"]):
                st.markdown(f"**{src}** — score {info['score']:.3f}")
                meta = next((m for m in metadata_store if m.get("filename") == src), {})
                if meta:
                    st.write(meta)
                for s in info['snippets'][:3]:
                    st.write(s + "...")

# -------------------------
# Chat over resumes (RAG)
# -------------------------
st.header("💬 Chat with your resumes (RAG)")
chat_query = st.text_input("Ask a question about uploaded resumes (e.g., 'Who has 5+ years Python experience?')")

if st.button("Answer") and chat_query:
    if not vectordb:
        st.error("No index found. Upload PDFs and press (Re)index first.")
    else:
        from langchain.chains import RetrievalQA

        # Build retriever
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        llm = init_chat_model(model="mistral-small-latest", temperature=0, api_key=MISTRAL_API_KEY)

        # Wrap retriever + LLM
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        with st.spinner("Generating answer..."):
            resp = qa_chain.invoke({"query": chat_query})

        answer = resp["result"]
        st.markdown("**Answer:**")
        st.write(answer)

        # Show supporting snippets
        st.markdown("**Source snippets:**")
        for doc in resp["source_documents"]:
            st.write(f"- {doc.metadata.get('source')} — {doc.page_content[:300]}...")

