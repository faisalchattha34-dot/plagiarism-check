import streamlit as st
import os
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.exceptions import NotFittedError

st.set_page_config(page_title="Document Similarity Checker", layout="wide")
st.title("📄 Document Similarity Checker")

# ---------------------------------------------------
# Function: read all uploaded .docx files
# ---------------------------------------------------
def read_docx_files(uploaded_files):
    """Reads text content from a list of uploaded .docx files."""
    notes, file_names = [], []
    for file in uploaded_files:
        try:
            doc = Document(file)
            text = "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
            if text:
                notes.append(text)
                file_names.append(file.name)
        except Exception as e:
            st.warning(f"Could not read {file.name}: {e}")
    return notes, file_names

# ---------------------------------------------------
# Function: calculate similarity matrix safely
# ---------------------------------------------------
def calculate_similarity(notes, vectorizer):
    """Calculates cosine similarity matrix, retraining model if needed."""
    try:
        tfidf_matrix = vectorizer.transform(notes)
    except (NotFittedError, ValueError):
        # Model not fitted or incompatible -> fit on current notes
        st.warning("⚠️ Model not fitted or incompatible. Training a new TF-IDF model.")
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(notes)
        joblib.dump(vectorizer, "model.pkl")
        st.success("✅ New model trained and saved as model.pkl")
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix, vectorizer

# ---------------------------------------------------
# Try to load pre-trained model
# ---------------------------------------------------
try:
    vectorizer = joblib.load("model.pkl")
    if not hasattr(vectorizer, "vocabulary_") or not vectorizer.vocabulary_:
        raise ValueError("Unfitted vectorizer.")
except Exception:
    st.warning("⚠️ No valid model found. A new one will be trained.")
    vectorizer = TfidfVectorizer(stop_words="english")

# ---------------------------------------------------
# File uploader
# ---------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload your .docx files",
    type=['docx'],
    accept_multiple_files=True
)

# ---------------------------------------------------
# Main logic
# ---------------------------------------------------
if uploaded_files:
    student_notes, student_file_names = read_docx_files(uploaded_files)

    if len(student_notes) < 2:
        st.warning("Please upload at least two documents with readable text.")
    else:
        similarity_matrix, vectorizer = calculate_similarity(student_notes, vectorizer)

        # Display table
        st.header("📊 Similarity Results")
        df_similarity = pd.DataFrame(similarity_matrix, index=student_file_names, columns=student_file_names)
        st.subheader("Similarity Matrix Table")
        st.dataframe(df_similarity)

        # Display heatmap
        st.subheader("🔥 Similarity Matrix Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df_similarity, annot=True, cmap="YlGnBu", ax=ax)
        plt.title("Document Similarity Heatmap")
        st.pyplot(fig)

        # Save results
        df_similarity.to_csv("similarity_results.csv", index=True)
        st.success("✅ Results saved to similarity_results.csv")
else:
    st.info("Please upload at least two .docx files to compare.")

# ---------------------------------------------------
# 📤 File uploader + duplicate file check
# ---------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload your .docx files",
    type=["docx"],
    accept_multiple_files=True,
    key="docx_upload"
)

if uploaded_files:
    # ✅ Check for duplicates
    file_names = [f.name for f in uploaded_files]
    if len(file_names) != len(set(file_names)):
        st.warning("⚠️ Duplicate files detected. Please upload unique documents.")
    else:
        # Continue with your existing logic here 👇
        student_notes, student_file_names = read_docx_files(uploaded_files)
        # ... (rest of your app logic: similarity calculation, table, heatmap, etc.)


