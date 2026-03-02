import streamlit as st
import joblib
import PyPDF2
import docx
import numpy as np

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI Resume Screening System",
    layout="centered"
)

st.title("AI Resume Screening System")
st.write("Upload a resume and paste a job description to calculate the match score.")

# -------------------------------
# LOAD MODEL + VECTORIZER
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("resume_screening_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# -------------------------------
# FILE UPLOADER
# -------------------------------
uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])

# -------------------------------
# JOB DESCRIPTION INPUT
# -------------------------------
job_description = st.text_area("Paste Job Description Here")

# -------------------------------
# TEXT EXTRACTION FUNCTIONS
# -------------------------------
def extract_text_from_pdf(file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + " "
    return text

# -------------------------------
# PREDICTION LOGIC
# -------------------------------
if st.button("Predict Match Score"):

    if uploaded_file is None:
        st.error("Please upload a resume.")
        st.stop()

    if not job_description.strip():
        st.error("Please paste a job description.")
        st.stop()

    # Extract resume text
    if uploaded_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    else:
        resume_text = extract_text_from_docx(uploaded_file)

    if not resume_text.strip():
        st.error("Could not extract text from the resume file.")
        st.stop()

    # Combine resume + job description
    combined_text = resume_text + " " + job_description

    # Vectorize
    features = vectorizer.transform([combined_text])

    # Predict
    score = model.predict(features)[0]

    # -------------------------------
    # SCORE INTERPRETATION
    # -------------------------------
    if score >= 0.8:
        st.success(f"Excellent Match 🔥 ({score:.3f})")
    elif score >= 0.6:
        st.info(f"Good Match 👍 ({score:.3f})")
    elif score >= 0.4:
        st.warning(f"Moderate Match ⚠ ({score:.3f})")
    else:
        st.error(f"Low Match ❌ ({score:.3f})")

    # Optional progress bar
    st.progress(min(max(score, 0), 1))