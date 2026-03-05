import streamlit as st
import joblib
import PyPDF2
import docx
from sklearn.metrics.pairwise import cosine_similarity

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
# LOAD VECTORIZER
# -------------------------------
@st.cache_resource
def load_vectorizer():
    return joblib.load("tfidf_vectorizer.pkl")

vectorizer = load_vectorizer()

# -------------------------------
# FILE UPLOADER
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Resume (PDF or DOCX)",
    type=["pdf", "docx"]
)

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
# MATCH CALCULATION
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

    # -------------------------------
    # VECTORIZE RESUME + JOB
    # -------------------------------
    resume_vector = vectorizer.transform([resume_text])
    job_vector = vectorizer.transform([job_description])

    # -------------------------------
    # COSINE SIMILARITY
    # -------------------------------
    score = cosine_similarity(resume_vector, job_vector)[0][0]

    # convert to percentage
    percentage = score * 100

    # -------------------------------
    # SCORE INTERPRETATION
    # -------------------------------
    if score >= 0.8:
        st.success(f"Excellent Match 🔥 ({percentage:.2f}%)")
    elif score >= 0.6:
        st.info(f"Good Match 👍 ({percentage:.2f}%)")
    elif score >= 0.4:
        st.warning(f"Moderate Match ⚠ ({percentage:.2f}%)")
    else:
        st.error(f"Low Match ❌ ({percentage:.2f}%)")

    # progress bar
    st.progress(float(score))
