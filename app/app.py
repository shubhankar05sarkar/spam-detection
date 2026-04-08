import streamlit as st
import pickle
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from preprocessing import clean_text

st.set_page_config(page_title="SMS Analysis Tool", layout="centered")

st.markdown("""
    <style>
    /* Main background and text */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Input area styling */
    .stTextArea textarea {
        background-color: #1B1E26 !important;
        color: #E0E0E0 !important;
        border: 1px solid #30363D !important;
        border-radius: 8px !important;
    }

    /* Button styling */
    .stButton>button {
        width: 100%;
        background-color: #238636;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem;
        font-weight: 600;
        transition: 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #2EA043;
        border: none;
        color: white;
    }

    /* Header styling */
    h1 {
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #FFFFFF;
    }
    
    /* Subtle footer/caption */
    .caption-text {
        color: #8B949E;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = pickle.load(open("../models/model.pkl", "rb"))
    vectorizer = pickle.load(open("../models/vectorizer.pkl", "rb"))
    return model, vectorizer

try:
    model, vectorizer = load_assets()
except FileNotFoundError:
    st.error("Model files not found. Please ensure the /models directory is correctly mapped.")
    st.stop()

st.title("Spam Scanner")
st.markdown("<p class='caption-text'>A machine learning system for detecting spam in SMS messages.</p>", unsafe_allow_html=True)

user_input = st.text_area("Message Content", placeholder="Paste the message text here...", height=150)

if st.button("Analyze Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message to analyze.")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)

        st.markdown("---")
        if prediction[0] == 1:
            st.error("Result: High Probability of Spam")
        else:
            st.success("Result: Verified Authentic (Ham)")