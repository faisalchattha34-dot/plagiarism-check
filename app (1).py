import streamlit as st
import os
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

def read_docx_files(uploaded_files):
    """Reads text content from a list of uploaded .docx files."""
    notes = []
    file_names = []
    for file in uploaded_files:
        # Streamlit provides file-like objects, which python-docx can read directly
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        notes.append(text)
        file_names.append(file.name)
    return notes, file_names

# Load the saved TF-IDF model
try:
    vectorizer = joblib.load("model.pkl")
except FileNotFoundError:
    st.error("Error: model.pkl not found. Please run the model training part of the notebook first.")
    st.stop() # Stop the app if the model is not found


def calculate_similarity(notes, vectorizer):
    """Calculates the cosine similarity matrix from a list of text notes using a pre-fitted vectorizer."""
    tfidf_matrix = vectorizer.transform(notes) # Use transform instead of fit_transform
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix

st.title("Document Similarity Checker")

uploaded_files = st.file_uploader("Upload your .docx files", type=['docx'], accept_multiple_files=True)

if uploaded_files:
    # Step 1: Read text content and get file names
    student_notes, student_file_names = read_docx_files(uploaded_files)

    if student_notes: # Ensure there is content to process
        # Step 2: Calculate the similarity matrix using the loaded vectorizer
        similarity_matrix = calculate_similarity(student_notes, vectorizer)

        # Create DataFrame for display
        df_similarity = pd.DataFrame(similarity_matrix, index=student_file_names, columns=student_file_names)

        # Add a descriptive header
        st.header("Similarity Results")

        # Display as a table
        st.subheader("Similarity Matrix Table")
        st.dataframe(df_similarity)

        # Display as a heatmap
        st.subheader("Similarity Matrix Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df_similarity, annot=True, cmap="YlGnBu", ax=ax)
        plt.title("Document Similarity Heatmap")
        st.pyplot(fig)
    else:
        st.warning("No text content found in the uploaded files.")
