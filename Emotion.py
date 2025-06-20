# NESME v1.0 - Full Streamlit Emotional Memory Engine
# Upload this file to GitHub and deploy directly to Streamlit Cloud

import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import uuid
import os

# Title
st.set_page_config(page_title="NESME - Emotional Memory Engine", layout="wide")
st.title("🧠 NESME - Neuro-Emotional Synthetic Memory Engine")

# Load pre-trained emotion model and tokenizer
@st.cache_resource
def load_emotion_model():
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=1)
    return classifier

emotion_classifier = load_emotion_model()

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# Initialize memory database (stored in session state)
if 'memory_db' not in st.session_state:
    st.session_state.memory_db = pd.DataFrame(columns=['id', 'text', 'embedding', 'emotion'])

# Sidebar - Memory Input
st.sidebar.header("Add New Memory")
user_input = st.sidebar.text_area("Enter your memory:")

if st.sidebar.button("Store Memory") and user_input.strip() != "":
    emotion = emotion_classifier(user_input)[0]['label']
    embedding = embedding_model.encode([user_input])[0]
    entry = {
        'id': str(uuid.uuid4()),
        'text': user_input,
        'embedding': embedding,
        'emotion': emotion
    }
    st.session_state.memory_db = pd.concat([st.session_state.memory_db, pd.DataFrame([entry])], ignore_index=True)
    st.sidebar.success(f"Memory stored with emotion: {emotion}")

# Main Area - Search Memories
st.subheader("🔎 Search Your Memories")
search_query = st.text_input("Enter search query:")
emotion_filter = st.selectbox("Filter by emotion:", options=["All"] + st.session_state.memory_db['emotion'].unique().tolist())

if st.button("Search"):
    if st.session_state.memory_db.empty:
        st.warning("Your memory database is empty.")
    else:
        query_embedding = embedding_model.encode([search_query])[0]
        memory_embeddings = np.vstack(st.session_state.memory_db['embedding'].values)
        similarities = cosine_similarity([query_embedding], memory_embeddings)[0]
        st.session_state.memory_db['similarity'] = similarities
        
        if emotion_filter != "All":
            filtered_db = st.session_state.memory_db[st.session_state.memory_db['emotion'] == emotion_filter]
        else:
            filtered_db = st.session_state.memory_db
        
        results = filtered_db.sort_values(by='similarity', ascending=False).head(5)
        
        if results.empty:
            st.warning("No matching memories found.")
        else:
            for idx, row in results.iterrows():
                st.markdown(f"""
                <div style='padding:15px; margin:10px 0; border-radius:10px; background-color:#f5f5f5;'>
                <b>📝 Memory:</b> {row['text']}<br>
                <b>🎭 Emotion:</b> <span style='color:blue'>{row['emotion']}</span><br>
                <b>🔗 Similarity:</b> {row['similarity']:.2f}
                </div>
                """, unsafe_allow_html=True)

# Optional: Show database
with st.expander("📊 View Memory Database"):
    st.dataframe(st.session_state.memory_db.drop(columns=['embedding']))
