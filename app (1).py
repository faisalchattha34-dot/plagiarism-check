
import subprocess, sys

# Ensure seaborn is installed at runtime (safety net)
subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
import seaborn as sns
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Inject custom CSS for styling
st.markdown("""
<style>
.main {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
}
.st-emotion-cache-h4xjwg { /* Target the title header */
    color: #0f1128;
    text-align: center;
    margin-bottom: 30px;
}
.stDataFrame {
    margin-bottom: 30px;
}
.stPlotlyChart {
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)


st.title("Document Similarity Analysis")

# Read the similarity results CSV
try:
    df = pd.read_csv("similarity_results.csv", index_col=0)
    st.subheader("Similarity Matrix:") # Use subheader for better hierarchy
    st.dataframe(df)

    # Display heatmap
    st.subheader("Similarity Heatmap:") # Use subheader
    fig, ax = plt.subplots()
    sns.heatmap(df, annot=True, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

except FileNotFoundError:
    st.error("Error: similarity_results.csv not found. Please run the previous steps to generate the file.")
except Exception as e:
    st.error(f"An error occurred: {e}")

st.markdown("This app displays the cosine similarity matrix between the document files.") # Use markdown for the final note
