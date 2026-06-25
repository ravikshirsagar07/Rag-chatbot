import streamlit as st
import pandas as pd
import spacy
import faiss
import numpy as np

from PyPDF2 import PdfReader
from docx import Document

from sentence_transformers import SentenceTransformer

# ----------------------------
# Models
# ----------------------------

nlp = spacy.load("en_core_web_sm")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ----------------------------
# Functions
# ----------------------------

def read_document(file):

    if file.name.endswith(".pdf"):

        pdf = PdfReader(file)

        text = ""

        for page in pdf.pages:
            text += page.extract_text()

        return text

    elif file.name.endswith(".docx"):

        doc = Document(file)

        return "\n".join(
            [p.text for p in doc.paragraphs]
        )

    elif file.name.endswith(".txt"):

        return file.read().decode("utf-8")

    return ""

def chunk_text(text, size=500):

    chunks = []

    for i in range(0, len(text), size):

        chunks.append(
            text[i:i+size]
        )

    return chunks

def extract_entities(text):

    doc = nlp(text)

    data = []

    for ent in doc.ents:

        data.append({
            "Entity": ent.text,
            "Label": ent.label_
        })

    return pd.DataFrame(data)

def create_vector_db(chunks):

    embeddings = embedding_model.encode(
        chunks
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        np.array(
            embeddings,
            dtype=np.float32
        )
    )

    return index, embeddings

def retrieve(query,
             index,
             chunks):

    q_embedding = embedding_model.encode(
        [query]
    )

    distances, ids = index.search(
        np.array(
            q_embedding,
            dtype=np.float32
        ),
        k=3
    )

    retrieved = []

    for idx in ids[0]:
        retrieved.append(
            chunks[idx]
        )

    return "\n".join(retrieved)

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(
    page_title="SmartDoc AI",
    layout="wide"
)

st.title(
    "🤖 SmartDoc AI - NER Powered RAG Chatbot"
)

uploaded_file = st.file_uploader(
    "Upload Document",
    type=["pdf","docx","txt"]
)

if uploaded_file:

    text = read_document(
        uploaded_file
    )

    st.success(
        "Document Uploaded"
    )

    chunks = chunk_text(text)

    entity_df = extract_entities(
        text
    )

    st.subheader(
        "Extracted Entities"
    )

    st.dataframe(
        entity_df,
        use_container_width=True
    )

    index, embeddings = create_vector_db(
        chunks
    )

    question = st.text_input(
        "Ask a Question"
    )

    if question:

        context = retrieve(
            question,
            index,
            chunks
        )

        st.subheader(
            "Retrieved Context"
        )

        st.write(context)

        # Simple RAG Answer
        st.subheader(
            "Answer"
        )

        st.write(
            "Based on the document:"
        )

        st.write(context[:1000])

        # Entities in retrieved context
        retrieved_entities = extract_entities(
            context
        )

        st.subheader(
            "Entities Used"
        )

        st.dataframe(
            retrieved_entities
        )