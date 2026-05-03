import streamlit as st

import nltk

# Stopwords
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

# Punkt
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

# 🔥 ADD THIS (IMPORTANT)
try:
    nltk.data.find('tokenizers/punkt_tab')
except:
    nltk.download('punkt_tab')

# ✅ MUST BE FIRST: Configure page before any other Streamlit commands
st.set_page_config(page_title="Resume Screening System", page_icon="📄", layout="wide")

# ===== ACCESSIBILITY + HIGH-CONTRAST STYLING =====
# Ensures cards with light backgrounds use dark text, and dark cards use light text.
st.markdown("""
<style>
/* Ensure all light-background alert cards use dark text for readability */
[data-testid="stSuccess"],
[data-testid="stSuccess"] * {
    color: #0f172a !important;
}

/* Ensure dark backgrounds use light text */
.stApp, .stContainer, .stMarkdown, .stText, .stCaption, h1, h2, h3, h4, h5, h6 {
    color: #f8fafc;
}

/* Force light-background cards (success/info/warning) to use dark text */
.stContainer div[style*="#dcfce7"],
.stContainer div[style*="#fef3c7"],
.stContainer div[style*="#fee2e2"],
.stContainer div[style*="#bbf7d0"],
.stContainer div[style*="#fde68a"] {
    color: #064e3b !important;
}

.stContainer div[style*="#dcfce7"] *,
.stContainer div[style*="#fef3c7"] *,
.stContainer div[style*="#fee2e2"] *,
.stContainer div[style*="#bbf7d0"] *,
.stContainer div[style*="#fde68a"] * {
    color: #064e3b !important;
}

/* Force alert captions to have sufficient contrast */
[data-testid="stSuccess"] .stAlert,
[data-testid="stError"] .stAlert,
[data-testid="stWarning"] .stAlert,
[data-testid="stInfo"] .stAlert {
    color: #0f172a !important;
}

/* Ensure metric labels remain readable on dark theme */
[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
}

/* Ensure metric values remain visible on dark cards */
[data-testid="stMetricValue"] {
    color: #22c55e !important;
}
</style>
""", unsafe_allow_html=True)

# ===== IMPORTS =====
from utils.pdf_parser import extract_text_from_pdf
from utils.similarity import (
    calculate_similarity, 
    analyze_keyword_gap, 
    get_top_keywords,
    extract_mandatory_skills,
    calculate_category_scores,
    calculate_mandatory_coverage,
    calculate_hybrid_score,
    generate_decision_and_explanation
)
from visualization.charts import plot_match_scores

import os
import joblib
import matplotlib.pyplot as plt
import numpy as np

# ✅ Load trained ML model and vectorizer once at startup (using Streamlit cache)
@st.cache_resource
def load_ml_model():
    """Load the trained ML model and TF-IDF vectorizer from .pkl files."""
    try:
        model = joblib.load('models/resume_screening_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("ML model files not found. Please ensure 'resume_screening_model.pkl' and 'tfidf_vectorizer.pkl' are in the models/ directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading ML model files: {str(e)}")
        return None, None

model, vectorizer = load_ml_model()

def get_ml_score(resume_text: str, job_text: str) -> tuple:
    """
    Get ML model prediction score using the loaded model and vectorizer.
    
    Returns:
        (ml_score, confidence) where ml_score is 0-100, confidence is 0-1
        Returns (None, None) if model not loaded
    """
    if model is None or vectorizer is None:
        return None, None
    
    try:
        # Combine resume and job text for prediction
        combined_text = resume_text + " " + job_text
        
        # Transform combined text using the pre-fitted vectorizer
        features = vectorizer.transform([combined_text])
        
        # Get prediction from regressor
        prediction = model.predict(features)[0]
        
        # Assuming prediction is between 0-1, scale to 0-100
        ml_score = prediction * 100
        
        # Use prediction as confidence (0-1 scale)
        confidence = prediction
        
        return ml_score, confidence
    except Exception as e:
        st.warning(f"ML prediction failed: {str(e)}")
        return None, None

# ===== HERO SECTION =====
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg,#1f2937 0%,#4a5568 100%);
        padding:40px;
        border-radius:10px;
        text-align:center;
        color:#f7fafc;
        margin-bottom:30px;
    ">
      <h1 style="margin:0;font-size:3rem;font-weight:700;">Resume Screening System</h1>
      <p style="margin:8px 0 0;font-size:1.1rem;color:#e2e8f0;">
         AI‑Powered Hybrid Scoring · Explainability · Skill Gap Analysis
      </p>
    </div>
    """,
    unsafe_allow_html=True
)

# KeyFeatures highlight
st.markdown("""
<div style="
    display: flex;
    justify-content: center;
    gap: 30px;
    margin-bottom: 30px;
    flex-wrap: wrap;
">
    <div style="text-align: center;">
        <p style="margin: 0; font-size: 24px;">🤖</p>
        <p style="margin: 8px 0 0 0; color: #2d3748; font-weight: 600; font-size: 13px;">HYBRID SCORING</p>
    </div>
    <div style="text-align: center;">
        <p style="margin: 0; font-size: 24px;">📊</p>
        <p style="margin: 8px 0 0 0; color: #2d3748; font-weight: 600; font-size: 13px;">SKILL ANALYSIS</p>
    </div>
    <div style="text-align: center;">
        <p style="margin: 0; font-size: 24px;">✨</p>
        <p style="margin: 8px 0 0 0; color: #2d3748; font-weight: 600; font-size: 13px;">EXPLAINABILITY</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ===== INPUT SECTION =====
st.markdown("<h2 style='color: #7C3AED; font-weight:700; font-size:1.75rem; margin-top: 30px;'>📋 Screen Your Resume</h2>", unsafe_allow_html=True)

# ===== STYLED UPLOADER =====
st.markdown("""
<div style="
    background-color: #f7fafc;
    border: 2px dashed #cbd5e0;
    border-radius: 8px;
    padding: 1px;
">
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

st.markdown("</div>", unsafe_allow_html=True)

# ===== JOB DESCRIPTION INPUT =====
st.markdown("<p style='color: #2d3748; font-weight: 600; margin: 20px 0 8px 0;'>Paste Job Description:</p>", unsafe_allow_html=True)
job_description = st.text_area(
    "Paste the job description here",
    height=160,
    label_visibility="collapsed",
    placeholder="Paste the complete job description here. Include required skills, responsibilities, and qualifications..."
)

# copy inputs to final vars (avoids undefined-variable later)
uploaded_files_final = uploaded_files
job_description_final = job_description

# ===== STYLED ANALYZE BUTTON =====
col1, col2, col3 = st.columns([1, 1.5, 1])
with col2:
    button_clicked = st.button(
        "🚀 Analyze Match",
        use_container_width=True,
        key="analyze_button"
    )

# Custom button styling
st.markdown("""
<style>
div.stButton > button:first-child {
    background: linear-gradient(90deg, #2563eb 0%, #1e40af 100%);
    color: white;
    font-weight: 600;
    font-size: 16px;
    padding: 12px 24px !important;
    border: none;
    border-radius: 6px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
}

div.stButton > button:first-child:hover {
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4);
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR (UPDATED) ----------------
with st.sidebar:
    st.header("About")
    st.info("""
    This tool helps you:
    - Assess how well your resume matches a job description.
    - Extract mandatory skills ("must have", "required", etc)
    - Calculate skill category match scores.
    - Get hybrid final score + decision.
    - View keyword analysis & gap assessment.
    """)
    
    st.header("How it works")
    st.write("""
    1. Upload your resume in PDF format.
    2. Paste the job description.
    3. Click **Analyze Match**
    4. Review hybrid score, decision & explainability panel
    """)
    
    st.header("Scoring Breakdown")
    st.write("""
    **Final Score = **
    - 50% Semantic Similarity
    - 30% Skill Category Match
    - 20% Mandatory Skill Coverage
    
    **Decision:**
    - ≥ 75 → **Shortlist**
    - 50-74 → **Review**
    - < 50 → **Reject**
    """)

# ---------------- MAIN LOGIC ----------------

# ----- view helpers -----
def render_recruiter_view(name, analysis):
    """Render full detailed report for recruiters inside its own modern card container."""
    with st.container():
        # Header with badge
        col_title, col_badge = st.columns([3, 1])
        with col_title:
            st.markdown(f"### 📄 {name}")
        with col_badge:
            hybrid_score = analysis['hybrid_score']
            if hybrid_score >= 75:
                st.markdown('<div style="background: linear-gradient(45deg, #10b981, #34d399); color: white; padding: 8px 12px; border-radius: 20px; text-align: center; font-weight: bold; font-size: 12px;">⭐ TOP MATCH</div>', unsafe_allow_html=True)
            elif hybrid_score >= 50:
                st.markdown('<div style="background: linear-gradient(45deg, #f59e0b, #fbbf24); color: white; padding: 8px 12px; border-radius: 20px; text-align: center; font-weight: bold; font-size: 12px;">⚠️ NEEDS REVIEW</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="background: linear-gradient(45deg, #ef4444, #f87171); color: white; padding: 8px 12px; border-radius: 20px; text-align: center; font-weight: bold; font-size: 12px;">❌ LOW MATCH</div>', unsafe_allow_html=True)
        
        decision = analysis['decision']
        explanation = analysis['explanation']
        ml_score = analysis.get('ml_score')
        ml_confidence = analysis.get('ml_confidence')
        
        # Enhanced decision display
        if decision == 'Shortlist':
            st.markdown(f'<div style="background: linear-gradient(135deg, #dcfce7, #bbf7d0); border: 2px solid #16a34a; border-radius: 10px; padding: 15px; margin: 10px 0; text-align: center;"><h4 style="color: #166534; margin: 0;">✅ SHORTLIST</h4><p style="color: #166534; margin: 5px 0 0 0; font-size: 18px; font-weight: bold;">Score: {hybrid_score}%</p></div>', unsafe_allow_html=True)
        elif decision == 'Review':
            st.markdown(f'<div style="background: linear-gradient(135deg, #fef3c7, #fde68a); border: 2px solid #d97706; border-radius: 10px; padding: 15px; margin: 10px 0; text-align: center;"><h4 style="color: #92400e; margin: 0;">🔄 REVIEW</h4><p style="color: #92400e; margin: 5px 0 0 0; font-size: 18px; font-weight: bold;">Score: {hybrid_score}%</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background: linear-gradient(135deg, #fee2e2, #fecaca); border: 2px solid #dc2626; border-radius: 10px; padding: 15px; margin: 10px 0; text-align: center;"><h4 style="color: #991b1b; margin: 0;">❌ REJECT</h4><p style="color: #991b1b; margin: 5px 0 0 0; font-size: 18px; font-weight: bold;">Score: {hybrid_score}%</p></div>', unsafe_allow_html=True)
        
        st.markdown(f'<div style="background: #f8fafc; border-left: 4px solid #3b82f6; padding: 12px; margin: 10px 0; border-radius: 5px;"><p style="margin: 0; color: #1e40af; font-style: italic;">💡 {explanation}</p></div>', unsafe_allow_html=True)
        
        # ML Model Score (if available) - Enhanced
        if ml_score is not None:
            st.markdown("### 🤖 AI Model Insights")
            ml_col1, ml_col2 = st.columns([3, 1])
            with ml_col1:
                st.metric("ML Prediction Score", f"{ml_score:.1f}%")
            with ml_col2:
                if ml_confidence and ml_confidence > 0.7:
                    st.markdown('<div style="background: #dcfce7; color: #166534; padding: 8px; border-radius: 5px; text-align: center; font-weight: bold;">High Confidence</div>', unsafe_allow_html=True)
                elif ml_confidence and ml_confidence > 0.5:
                    st.markdown('<div style="background: #fef3c7; color: #92400e; padding: 8px; border-radius: 5px; text-align: center; font-weight: bold;">Medium Confidence</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="background: #fee2e2; color: #991b1b; padding: 8px; border-radius: 5px; text-align: center; font-weight: bold;">Low Confidence</div>', unsafe_allow_html=True)
            st.markdown("---")
        
        # Score breakdown - Enhanced
        st.markdown("### 📊 Score Breakdown")
        pb_col1, pb_col2, pb_col3 = st.columns(3)
        with pb_col1:
            semantic_val = analysis['semantic_score'] / 100
            st.metric("Semantic Similarity", f"{analysis['semantic_score']}%")
            st.progress(semantic_val, text=f"{analysis['semantic_score']}%")
        with pb_col2:
            category_val = analysis['category_avg'] / 100
            st.metric("Skill Category Match", f"{analysis['category_avg']}%")
            st.progress(category_val, text=f"{analysis['category_avg']}%")
        with pb_col3:
            mandatory_val = analysis['mandatory_coverage'] / 100
            st.metric("Mandatory Coverage", f"{analysis['mandatory_coverage']}%")
            st.progress(mandatory_val, text=f"{analysis['mandatory_coverage']}%")
        
        st.markdown("---")
        
        # missing skills badges
        st.markdown("**🚀 Recommended Skills for Development:**")
        gap = analysis['gap_analysis']
        missing_keywords = gap['missing_keywords']
        if missing_keywords:
            top_5_missing = [kw[0] for kw in missing_keywords[:5]]
            badges_html = "".join([f"<span style='display:inline-block;background-color:#ffe5e5;color:#c41e3a;padding:8px 14px;margin:5px 5px 5px 0;border-radius:20px;font-weight:500;font-size:13px;border:1px solid #ffcccc;'>{skill}</span>" for skill in top_5_missing])
            st.markdown(badges_html, unsafe_allow_html=True)
            st.caption("💡 Focus on adding these high-impact skills to strengthen your profile for similar roles.")
        else:
            st.success("✅ All required skills covered!")
        
        st.divider()
        
        # category breakdown
        st.markdown("**📊 Skill Category Breakdown:**")
        category_scores = analysis['category_scores'].copy()
        category_scores.pop('average')
        for category, score in category_scores.items():
            st.write(f"• {category.replace('_',' ').title()}: **{score}%**")
        
        st.divider()
        
        # mandatory summary
        st.markdown("**🔑 Mandatory Skills Coverage:**")
        mandatory_skills = analysis['mandatory_skills']
        resume_text_lower = analysis['resume_text'].lower()
        found = len([s for s in mandatory_skills if s in resume_text_lower])
        st.write(f"• Found: **{found}/{len(mandatory_skills)}** ({analysis['mandatory_coverage']}%)")
        
        st.divider()
        
        # matching keywords
        st.markdown("**✅ Matching Keywords:**")
        matching = [kw[0] for kw in gap['matching_keywords'][:10]]
        if matching:
            st.write(", ".join(matching))
            if len(gap['matching_keywords']) > 10:
                st.caption(f"... and {len(gap['matching_keywords'])-10} more")
        else:
            st.caption("None found")




if button_clicked:

    if not uploaded_files_final:
        st.warning("Please upload at least one resume PDF.")
        st.stop()

    if not job_description_final:
        st.warning("Please enter a job description.")
        st.stop()

    results = []
    detailed_analysis = {}

    with st.spinner("Analyzing resumes..."):
        for file in uploaded_files_final:
            data = file.read()
            resume_text = extract_text_from_pdf(data)

            if resume_text:
                semantic_score = calculate_similarity(resume_text, job_description_final)
                mandatory_skills = extract_mandatory_skills(job_description_final)
                mandatory_coverage = calculate_mandatory_coverage(resume_text, mandatory_skills)
                category_scores = calculate_category_scores(resume_text, job_description_final)
                category_avg = category_scores['average']
                hybrid_score = calculate_hybrid_score(semantic_score, category_avg, mandatory_coverage)
                decision_data = generate_decision_and_explanation(
                    hybrid_score, mandatory_coverage, semantic_score, category_avg
                )
                gap_analysis = analyze_keyword_gap(resume_text, job_description_final, top_missing=20)
                top_resume_keywords = get_top_keywords(resume_text, top_n=5)
                
                # Get ML prediction score
                ml_score, ml_confidence = get_ml_score(resume_text, job_description_final)
                
                # store all data
                detailed_analysis[file.name] = {
                    'resume_text': resume_text,
                    'semantic_score': semantic_score,
                    'mandatory_skills': mandatory_skills,
                    'mandatory_coverage': mandatory_coverage,
                    'category_scores': category_scores,
                    'category_avg': category_avg,
                    'hybrid_score': hybrid_score,
                    'decision': decision_data['decision'],
                    'explanation': decision_data['explanation'],
                    'gap_analysis': gap_analysis,
                    'top_resume_keywords': top_resume_keywords,
                    'ml_score': ml_score,
                    'ml_confidence': ml_confidence
                }

                results.append({
                    "Resume Name": file.name,
                    "Hybrid Score": hybrid_score,
                    "Decision": decision_data['decision'],
                    "Semantic": semantic_score,
                    "Category Avg": category_avg,
                    "Mandatory %": mandatory_coverage,
                    "ML Score": ml_score
                })

    if results:
        results = sorted(results, key=lambda x: (x["Hybrid Score"], x.get("ML Score") or 0), reverse=True)

        # ===== ANALYTICS DASHBOARD =====
        st.markdown("---")
        st.markdown("## 🎯 AI Recruiter Dashboard")
        
        # Calculate KPIs
        total_resumes = len(results)
        avg_hybrid_score = sum(r["Hybrid Score"] for r in results) / total_resumes if total_resumes > 0 else 0
        shortlisted = sum(1 for r in results if r["Decision"] == "Shortlist")
        rejected = sum(1 for r in results if r["Decision"] == "Reject")
        
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 Total Resumes", total_resumes)
        with col2:
            st.metric("📊 Avg Match Score", f"{avg_hybrid_score:.1f}%")
        with col3:
            st.metric("✅ Shortlisted", shortlisted)
        with col4:
            st.metric("❌ Rejected", rejected)
        
        st.markdown("---")

        # ===== CANDIDATE RANKING TABLE =====
        st.markdown("### 🏆 Candidate Ranking")
        
        # Prepare data for table
        table_data = []
        for idx, result in enumerate(results, 1):
            badge = ""
            if idx == 1:
                badge = "⭐ Top Match"
            elif result["Decision"] == "Review":
                badge = "⚠️ Needs Review"
            elif result["Decision"] == "Reject":
                badge = "❌ Low Match"
            
            table_data.append({
                "Rank": idx,
                "Resume Name": result["Resume Name"],
                "Hybrid Score": f"{result['Hybrid Score']}%",
                "ML Score": f"{result['ML Score']:.1f}%" if result["ML Score"] is not None else "N/A",
                "Decision": result["Decision"],
                "Badge": badge
            })
        
        st.dataframe(table_data, use_container_width=True, hide_index=True)

        # # ===== VISUAL ANALYTICS =====
        # st.markdown("### 📈 Visual Analytics")
        
        # tab1, tab2, tab3 = st.tabs(["Score Distribution", "Skill Categories", "Ranking Chart"])
        
        # with tab1:
        #     # Score distribution histogram
        #     scores = [r["Hybrid Score"] for r in results]
        #     fig, ax = plt.subplots(figsize=(8, 4), facecolor='#1e293b')
        #     ax.set_facecolor('#1e293b')
        #     ax.hist(scores, bins=10, alpha=0.7, color='#6366f1', edgecolor='#334155')
        #     ax.set_title('Hybrid Score Distribution', fontsize=14, fontweight='bold', color='#f8fafc')
        #     ax.set_xlabel('Score (%)', fontsize=12, color='#cbd5e1')
        #     ax.set_ylabel('Number of Resumes', fontsize=12, color='#cbd5e1')
        #     ax.tick_params(colors='#cbd5e1')
        #     ax.grid(True, alpha=0.3, color='#374151')
        #     st.pyplot(fig)
        
        # with tab2:
        #     # Skill category radar chart (using first candidate as example)
        #     if detailed_analysis:
        #         first_resume = list(detailed_analysis.keys())[0]
        #         category_scores = detailed_analysis[first_resume]['category_scores']
        #         categories = list(category_scores.keys())[:-1]  # Exclude 'average'
        #         values = [category_scores[cat] for cat in categories]
                
        #         # Radar chart
        #         angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        #         values += values[:1]
        #         angles += angles[:1]
                
        #         fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'), facecolor='#1e293b')
        #         ax.set_facecolor('#1e293b')
        #         ax.fill(angles, values, 'o-', alpha=0.25, color='#22c55e')
        #         ax.plot(angles, values, 'o-', linewidth=2, color='#22c55e')
        #         ax.set_xticks(angles[:-1])
        #         ax.set_xticklabels(categories, fontsize=10, color='#cbd5e1')
        #         ax.tick_params(colors='#cbd5e1')
        #         ax.set_ylim(0, 100)
        #         ax.set_title('Skill Category Analysis', fontsize=14, fontweight='bold', color='#f8fafc', pad=20)
        #         ax.grid(True, color='#374151')
        #         st.pyplot(fig)
        
        # with tab3:
        #     # Ranking bar chart
        #     names = [r["Resume Name"][:15] + "..." if len(r["Resume Name"]) > 15 else r["Resume Name"] for r in results[:10]]
        #     scores = [r["Hybrid Score"] for r in results[:10]]
            
        #     fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1e293b')
        #     ax.set_facecolor('#1e293b')
        #     bars = ax.barh(names[::-1], scores[::-1], color='#6366f1', alpha=0.8)
        #     ax.set_xlabel('Hybrid Score (%)', fontsize=12, color='#cbd5e1')
        #     ax.set_title('Top 10 Candidates Ranking', fontsize=14, fontweight='bold', color='#f8fafc')
        #     ax.tick_params(colors='#cbd5e1')
        #     ax.grid(True, alpha=0.3, color='#374151')
            
        #     # Add score labels on bars
        #     for bar, score in zip(bars, scores[::-1]):
        #         ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
        #                f'{score}%', ha='left', va='center', fontsize=10, fontweight='bold', color='#f8fafc')
            
        #     st.pyplot(fig)

        # st.markdown("---")

        # ===== RESULTS SUMMARY =====
        st.subheader("📊 Results Summary (Ranked by Score)")
        
        for idx, result in enumerate(results):
            resume_name = result["Resume Name"]
            final_score = result["Hybrid Score"]
            decision = result["Decision"]
            mandatory = result["Mandatory %"]
            ml_score = result["ML Score"]
            
            st.write(f"**📄 {resume_name}**")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Hybrid Score", f"{final_score}%")
            
            with col2:
                if ml_score is not None:
                    st.metric("ML Score", f"{ml_score:.1f}%")
                else:
                    st.metric("ML Score", "N/A")
        
            with col3:
                if decision == 'Shortlist':
                    st.success(f"✅ {decision}")
                elif decision == 'Review':
                    st.info(f"🔄 {decision}")
                else:
                    st.error(f"❌ {decision}")
            
            with col4:
                st.metric("Mandatory Coverage", f"{mandatory}%")
            
            with col5:
                rank = idx + 1
                st.metric("Rank", f"#{rank}")
            
            if idx < len(results) - 1:
                st.divider()
        
        # ===== SCORING FORMULA =====
        with st.expander("📐 Scoring Formula Explanation"):
            st.write("""
            **Final Score = 0.5 × Semantic Similarity + 0.3 × Skill Category Match + 0.2 × Mandatory Coverage**
            
            - **Semantic Similarity** (50%): How well resume content matches job description overall
            - **Skill Category Match** (30%): Average match across technical, soft, & domain skills
            - **Mandatory Coverage** (20%): % of "must have" / "required" skills found in resume
            
            **Decision Thresholds:**
            - ≥ 75% → **Shortlist** | 50-74% → **Review** | < 50% → **Reject**
            """)
        
        st.markdown("---")
        st.subheader("📋 Detailed Reports")
        
        # ===== DETAILED REPORTS =====
        for result in results:
            name = result["Resume Name"]
            analysis = detailed_analysis[name]
            render_recruiter_view(name, analysis)

# ===== DARK AI DASHBOARD THEME =====
st.markdown("""
<style>
/* Main app background - Dark slate */
.stApp {
    background: #0f172a;
    color: #f8fafc;
}

/* Header styling */
[data-testid="stHeader"] {
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
    color: #f8fafc;
}

/* Card containers - Dark card background */
.stContainer, .stContainer > div {
    background: #1e293b !important;
    border-radius: 15px !important;
    padding: 20px !important;
    margin: 10px 0 !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
    border: 1px solid #334155 !important;
    color: #f8fafc !important;
}

/* All text elements */
/* Only apply white text to DARK background */
body, .stApp {
    color: #f8fafc;
}

/* Button styling - Primary accent */
.stButton>button {
    background: linear-gradient(45deg, #6366f1, #8b5cf6) !important;
    color: #f8fafc !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 12px 30px !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
}

.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4) !important;
    background: linear-gradient(45deg, #4f46e5, #7c3aed) !important;
}

/* Metric cards - High contrast */
[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: bold !important;
    color: #22c55e !important; /* Secondary accent for values */
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5) !important;
}

[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important; /* Light gray for labels */
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}

/* Dataframe styling - Dark theme */
.stDataFrame {
    background: #1e293b !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    border: 1px solid #334155 !important;
}

.stDataFrame table {
    color: #f8fafc !important;
    background: #1e293b !important;
}

.stDataFrame th {
    background: #334155 !important;
    color: #f8fafc !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #6366f1 !important;
}

.stDataFrame td {
    color: #e2e8f0 !important;
    border-bottom: 1px solid #374151 !important;
}

/* Tab styling - Dark theme */
.stTabs [data-baseweb="tab-list"] {
    background: #1e293b !important;
    border-radius: 10px 10px 0 0 !important;
    border-bottom: 2px solid #6366f1 !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 10px 10px 0 0 !important;
    color: #cbd5e1 !important;
    font-weight: 600 !important;
}

.stTabs [data-baseweb="tab"]:hover {
    background: #334155 !important;
    color: #f8fafc !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: #6366f1 !important;
    color: #f8fafc !important;
}

/* Progress bars - Secondary accent */
.stProgress > div > div {
    background: linear-gradient(45deg, #22c55e, #16a34a) !important;
    border-radius: 10px !important;
}

/* Sidebar styling - Dark theme */
[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 2px solid #6366f1 !important;
}

[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}

/* Custom HTML elements */
.stMarkdown div[style*="background"] {
    color: #f8fafc !important;
}

/* Chart containers */
.stPlotlyChart, .stPyplot {
    background: #1e293b !important;
    border-radius: 10px !important;
    padding: 15px !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    border: 1px solid #334155 !important;
}

/* === FIX LIGHT CARDS TEXT VISIBILITY === */

/* Shortlist (Green) */
.stSuccess {
    background: linear-gradient(135deg, #bbf7d0, #86efac) !important;
    color: #022c22 !important;
}

/* Review (Yellow) */
.stWarning, .stInfo {
    background: linear-gradient(135deg, #fde68a, #facc15) !important;
    color: #78350f !important;
}

/* Reject (Red) */
.stError {
    background: linear-gradient(135deg, #fecaca, #f87171) !important;
    color: #7f1d1d !important;
}

/* Ensure inner text is visible */
.stSuccess *, .stWarning *, .stError *, .stInfo * {
    color: inherit !important;
}

/* Input fields */
.stTextInput input, .stTextArea textarea {
    background: #374151 !important;
    color: #f8fafc !important;
    border: 2px solid #6366f1 !important;
    border-radius: 8px !important;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #22c55e !important;
    box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.2) !important;
}

/* File uploader */
.stFileUploader {
    background: #1e293b !important;
    border: 2px dashed #6366f1 !important;
    border-radius: 10px !important;
    color: #f8fafc !important;
}

# /* Badge styling */
# .stMarkdown span[style*="background"] {
#     color: #f8fafc !important;
# }

/* Ensure all headings are visible */
h1, h2, h3, h4, h5, h6 {
    color: #f8fafc !important;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5) !important;
}

/* KPI cards special styling */
.stMetric {
    background: #1e293b !important;
    border-radius: 12px !important;
    padding: 15px !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    border: 1px solid #334155 !important;
}

/* Table hover effects */
.stDataFrame tbody tr:hover {
    background: #334155 !important;
}

/* Loading spinner */
.stSpinner {
    color: #6366f1 !important;
}
            /* ===== SKILL TAGS (IMPORTANT FIX) ===== */
.skill-tag {
    display: inline-block;
    padding: 6px 14px;
    margin: 6px 6px 0 0;
    border-radius: 20px;
    
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #ffffff !important;
    
    font-size: 0.85rem;
    font-weight: 500;
    
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
    
    transition: 0.2s ease;
}

.skill-tag:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.5);
}
</style>
""", unsafe_allow_html=True)
