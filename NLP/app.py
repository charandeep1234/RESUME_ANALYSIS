"""
NLP-Driven Resume Screening and Intelligent Job Recommendation System
=====================================================================
Main Streamlit Application

This capstone project demonstrates a complete NLP pipeline for:
1. Resume Data Collection and Preprocessing
2. Candidate Information and Skill Extraction
3. Job Description Analysis
4. Resume-Job Matching and Screening
5. Intelligent Job Recommendation and Results

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
import hashlib

# Add project root to path so modules can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.resume_parser import extract_text
from modules.preprocessing import preprocess_text, get_processed_text
from modules.information_extraction import extract_all_information
from modules.job_analysis import (
    init_database, get_all_jobs, get_job_by_id, add_job, delete_job,
    save_candidate, save_matching_results, get_all_candidates, get_matching_results
)
from modules.matching import match_resume_with_jobs
from modules.recommendation import get_top_recommendations, generate_dashboard_stats

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="NLP Resume Screening System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS for a colorful dashboard look
# ============================================================
st.markdown("""
<style>
    :root {
        --page-bg: #e9edf0;
        --sidebar-bg: #0b3242;
        --sidebar-bg-2: #0c3a4d;
        --card-soft: #f4f6f7;
        --card-border: rgba(255,255,255,0.08);
        --text-main: #1d2a36;
        --text-soft: #577082;
        --primary: #2b7aa6;
        --teal: #5eb3c1;
        --aqua: #2aa6a7;
        --purple: #7f84d4;
        --gold: #d8b15d;
        --orange: #e4a146;
        --pink: #d79aa7;
    }

    .stApp {
        background: var(--page-bg);
        color: var(--text-main);
    }

    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1200px;
    }

    /* Main content cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f3f6f8 100%);
        border: 1px solid rgba(13, 50, 66, 0.08);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 6px rgba(11, 50, 66, 0.04);
    }
    .metric-card h3 {
        margin: 0;
        color: var(--text-soft);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .metric-card p {
        margin: 0.5rem 0 0 0;
        color: var(--text-main);
        font-size: 1.8rem;
        font-weight: 800;
    }

    .info-card {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(13, 50, 66, 0.08);
        border-radius: 14px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(13, 50, 66, 0.04);
    }
    .info-card h4 {
        color: var(--text-main);
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
        font-weight: 700;
    }

    .skill-tag {
        display: inline-block;
        background: linear-gradient(135deg, #e8f4fb 0%, #d5eaf8 100%);
        color: #1c5b7e;
        padding: 0.28rem 0.8rem;
        border-radius: 999px;
        margin: 0.15rem;
        font-size: 0.82rem;
        font-weight: 700;
        border: 1px solid rgba(28, 91, 126, 0.08);
    }
    .skill-tag-missing {
        background: linear-gradient(135deg, #fff5ed 0%, #f5e3ce 100%);
        color: #9a5d1a;
        border-color: rgba(154, 93, 26, 0.08);
    }
    .skill-tag-matched {
        background: linear-gradient(135deg, #eafaf4 0%, #d4f1e7 100%);
        color: #226d51;
        border-color: rgba(34, 109, 81, 0.08);
    }

    .rec-card {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(13, 50, 66, 0.08);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid var(--primary);
        box-shadow: 0 4px 10px rgba(13, 50, 66, 0.04);
    }

    .workflow-step {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(13, 50, 66, 0.08);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 8px rgba(13, 50, 66, 0.04);
        color: var(--text-main);
    }
    .workflow-arrow {
        text-align: center;
        color: var(--primary);
        font-size: 1.2rem;
        margin: 0.45rem 0;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--sidebar-bg) 0%, var(--sidebar-bg-2) 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
        box-shadow: 2px 0 18px rgba(8, 31, 41, 0.12);
    }

    .brand-shell {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        padding: 0.75rem 0.4rem 1rem 0.4rem;
        margin-bottom: 0.6rem;
    }

    .brand-logo {
        width: 46px;
        height: 46px;
        min-width: 46px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #dbeafe, #bfdbfe);
        border: 1px solid rgba(37, 99, 235, 0.2);
        box-shadow: 0 10px 20px rgba(37, 99, 235, 0.1);
        animation: pulseGlow 2.2s infinite ease-in-out;
    }

    .brand-logo svg {
        width: 28px;
        height: 28px;
        display: block;
    }

    .brand-text {
        font-size: 1.08rem;
        line-height: 1.18;
        letter-spacing: 0.015em;
        font-weight: 300;
        color: #f4f9ff;
    }

    .brand-text span {
        color: #dfeaf5;
        font-size: 0.88rem;
        font-weight: 300;
    }

    div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 0.7rem;
        margin-top: 0.7rem;
    }

    div[role="radiogroup"] label {
        display: flex;
        align-items: center;
        min-height: 42px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.04);
        padding: 0.7rem 0.85rem;
        color: #f3f9ff !important;
        font-weight: 300 !important;
        letter-spacing: 0.01em;
        box-shadow: none;
        transition: background 0.16s ease, border-color 0.16s ease;
    }

    div[role="radiogroup"] label:hover {
        background: rgba(255,255,255,0.06);
    }

    div[role="radiogroup"] label div,
    div[role="radiogroup"] label span,
    div[role="radiogroup"] label p,
    div[role="radiogroup"] label * {
        color: inherit !important;
        font-weight: 300 !important;
    }

    div[role="radiogroup"] label input {
        accent-color: #e7f3ff !important;
        width: 16px !important;
        height: 16px !important;
        margin-right: 0.7rem !important;
    }

    div[role="radiogroup"] input:checked + div {
        background: rgba(95, 149, 173, 0.18) !important;
        border-color: rgba(126, 196, 255, 0.9) !important;
        box-shadow: inset 0 0 0 1px rgba(126, 196, 255, 0.08);
    }

    .stSidebar .stSuccess, .stSidebar .stInfo {
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.06);
        color: #ffffff;
    }

    .stSidebar .stSuccess p, .stSidebar .stInfo p {
        color: #ffffff;
    }

    .css-1d391kg, .css-17lntkn {
        font-weight: 300;
    }

    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 12px rgba(52, 115, 145, 0.12); }
        50% { box-shadow: 0 0 18px rgba(52, 115, 145, 0.22); }
    }

    /* Hide default Streamlit menu items */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Initialize Database
# ============================================================
init_database()

# ============================================================
# Session State Initialization
# ============================================================
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = None
if 'preprocessed' not in st.session_state:
    st.session_state.preprocessed = None
if 'candidate_info' not in st.session_state:
    st.session_state.candidate_info = None
if 'matching_results' not in st.session_state:
    st.session_state.matching_results = None
if 'candidate_id' not in st.session_state:
    st.session_state.candidate_id = None
if 'saved_resume_hash' not in st.session_state:
    st.session_state.saved_resume_hash = None

# ============================================================
# Sidebar Navigation
# ============================================================
st.sidebar.markdown("""
<div class="brand-shell">
    <div class="brand-logo">
        <svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
            <rect x="16" y="10" width="32" height="44" rx="6" fill="#0f172a" stroke="#7dd3fc" stroke-width="2.2"/>
            <path d="M24 20h16M24 28h16M24 36h11" stroke="#7dd3fc" stroke-width="2.5" stroke-linecap="round"/>
            <circle cx="44" cy="42" r="8" fill="none" stroke="#a78bfa" stroke-width="2.5"/>
            <path d="M44 35v14M37 42h14" stroke="#a78bfa" stroke-width="2.5" stroke-linecap="round"/>
        </svg>
    </div>
    <div class="brand-text">Resume Screening<br><span>System</span></div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home",
     "📤 Resume Upload",
     "👤 Candidate Analysis",
     "💼 Job Analysis",
     "🔍 Resume–Job Matching",
     "⭐ Job Recommendations",
     "⚙️ Admin Panel"],
    index=0,
    label_visibility="collapsed"
)

# Show processing status in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("<div style='color: #ffffff; font-weight: 300; margin-bottom: 0.7rem; letter-spacing: 0.04em;'>Processing Status</div>", unsafe_allow_html=True)
if st.session_state.resume_text:
    st.sidebar.success("✅ Resume Uploaded")
else:
    st.sidebar.info("⏳ No Resume Uploaded")

if st.session_state.candidate_info:
    st.sidebar.success("✅ Info Extracted")

if st.session_state.matching_results:
    st.sidebar.success("✅ Matching Complete")


# ============================================================
# PAGE: Home
# ============================================================
def page_home():
    st.title("NLP-Driven Resume Screening and Intelligent Job Recommendation System")
    st.markdown("##### A Capstone Project Demonstrating NLP-based Resume Analysis and Job Matching")
    
    st.markdown("---")
    
    st.subheader("System Workflow")
    st.markdown("This application processes resumes through five modules to provide intelligent job recommendations.")
    
    # Workflow display
    steps = [
        ("📤 Module 1: Resume Upload & Preprocessing",
         "Upload a PDF/DOCX resume. The system extracts text, removes noise, normalizes it, and performs tokenization."),
        ("👤 Module 2: Candidate Information Extraction",
         "NLP techniques extract key details: name, email, phone, skills, education, experience, certifications, and projects."),
        ("💼 Module 3: Job Description Analysis",
         "A database of job descriptions is preprocessed and analyzed. Each job includes required skills, education, and experience levels."),
        ("🔍 Module 4: Resume–Job Matching",
         "TF-IDF Vectorization and Cosine Similarity compare the resume with job descriptions. Skill gap analysis identifies matched and missing skills."),
        ("⭐ Module 5: Job Recommendation",
         "Jobs are ranked by match score. The top 3 recommendations are presented with personalized advice and a results dashboard."),
    ]
    
    for i, (title, desc) in enumerate(steps):
        st.markdown(f"""
        <div class="workflow-step">
            <strong>{title}</strong><br>
            <span style="color: #6c757d; font-size: 0.9rem;">{desc}</span>
        </div>
        """, unsafe_allow_html=True)
        
        if i < len(steps) - 1:
            st.markdown('<div class="workflow-arrow">⬇</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick stats
    jobs_df = get_all_jobs()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Available Jobs</h3>
            <p>{len(jobs_df)}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        candidates_df = get_all_candidates()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Resumes Analyzed</h3>
            <p>{len(candidates_df)}</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>NLP Techniques Used</h3>
            <p>5</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Technologies Used")
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    with tech_col1:
        st.markdown("""
        **Backend & NLP**
        - Python 3.x
        - NLTK (Tokenization, Stopwords)
        - Scikit-learn (TF-IDF, Cosine Similarity)
        """)
    with tech_col2:
        st.markdown("""
        **Data & Storage**
        - Pandas (Data Manipulation)
        - SQLite (Database)
        - CSV (Initial Data)
        """)
    with tech_col3:
        st.markdown("""
        **Frontend & Visualization**
        - Streamlit (Web Interface)
        - Plotly (Charts)
        - pdfplumber / python-docx (File Parsing)
        """)
    
    st.info("👈 Use the sidebar to navigate through the application modules.")


# ============================================================
# PAGE: Resume Upload (Module 1)
# ============================================================
def page_resume_upload():
    st.title("📤 Module 1: Resume Upload & Preprocessing")
    st.markdown("Upload your resume in **PDF** or **DOCX** format to begin the analysis pipeline.")
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a resume file",
        type=['pdf', 'docx'],
        help="Supported formats: PDF, DOCX"
    )
    
    # Demo mode
    use_demo = st.checkbox("Use demo resume (for testing without uploading a file)")
    
    if use_demo:
        demo_text = """John Doe
Contact: john.doe@email.com | +1-555-123-4567 | LinkedIn: linkedin.com/in/johndoe

SUMMARY
Experienced Python Developer with 3 years of experience in machine learning, data analysis, and NLP.
Passionate about building intelligent systems using modern AI techniques.

EDUCATION
B.Tech in Computer Science and Engineering
XYZ University, 2020
CGPA: 8.5/10

Master of Science in Data Science
ABC University, 2022

SKILLS
Programming: Python, R, SQL, JavaScript, HTML, CSS
Machine Learning: Scikit-learn, TensorFlow, Pandas, NumPy
NLP: NLTK, spaCy, Text Mining, Sentiment Analysis
Tools: Git, Docker, Tableau, Power BI, Excel
Databases: MySQL, PostgreSQL, MongoDB

EXPERIENCE
Python Developer Intern
Tech Solutions Pvt Ltd, Jan 2021 - Jun 2021
- Developed REST APIs using Flask framework
- Built data pipelines for processing customer data
- Implemented machine learning models for churn prediction

Data Analyst
DataCorp Inc, Jul 2021 - Present
- Analyzed large datasets using Python and SQL
- Created dashboards and reports using Tableau
- Performed statistical analysis and A/B testing

PROJECTS
Resume Screening System using NLP
- Built an NLP-based system to screen resumes and match them with job descriptions
- Used TF-IDF and cosine similarity for document matching

Sentiment Analysis of Product Reviews
- Developed a sentiment analysis model using NLTK and Scikit-learn
- Achieved 87% accuracy on customer review classification

CERTIFICATIONS
AWS Certified Cloud Practitioner
Google Data Analytics Professional Certificate
Coursera Machine Learning Specialization by Andrew Ng
"""
        st.session_state.resume_text = demo_text
        st.success("✅ Demo resume loaded successfully!")
    
    elif uploaded_file is not None:
        try:
            with st.spinner("Extracting text from resume..."):
                text = extract_text(uploaded_file, uploaded_file.name)
                st.session_state.resume_text = text
                st.success(f"✅ Text extracted successfully from **{uploaded_file.name}**!")
        except ValueError as e:
            st.error(f"❌ {str(e)}")
            return
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}")
            return
    
    # Display extracted and preprocessed text
    if st.session_state.resume_text:
        st.markdown("---")
        
        # Step 1: Extracted Text
        st.subheader("Step 1: Extracted Raw Text")
        with st.expander("View extracted text", expanded=False):
            st.text(st.session_state.resume_text)
        
        st.markdown("---")
        
        # Step 2-4: Preprocessing
        st.subheader("Step 2–4: Text Preprocessing Pipeline")
        
        with st.spinner("Preprocessing text..."):
            preprocessed = preprocess_text(st.session_state.resume_text)
            st.session_state.preprocessed = preprocessed
        
        # Preprocessing summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Raw Characters</h3>
                <p>{preprocessed['num_raw_chars']}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>After Cleaning</h3>
                <p>{preprocessed['num_cleaned_chars']}</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Tokens</h3>
                <p>{preprocessed['num_tokens']}</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>After Stopword Removal</h3>
                <p>{preprocessed['num_filtered_tokens']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Show each preprocessing step
        tab1, tab2, tab3, tab4 = st.tabs(["Cleaned Text", "Normalized Text", "Tokens", "Filtered Tokens"])
        
        with tab1:
            st.caption("Special characters removed, extra whitespace cleaned")
            st.text_area("Cleaned Text", preprocessed['cleaned_text'], height=200, disabled=True, label_visibility="collapsed")
        
        with tab2:
            st.caption("Text converted to lowercase")
            st.text_area("Normalized Text", preprocessed['normalized_text'], height=200, disabled=True, label_visibility="collapsed")
        
        with tab3:
            st.caption(f"Text split into {preprocessed['num_tokens']} tokens")
            st.write(preprocessed['tokens'][:100])
            if len(preprocessed['tokens']) > 100:
                st.caption(f"... and {len(preprocessed['tokens']) - 100} more tokens")
        
        with tab4:
            st.caption(f"{preprocessed['num_stopwords_removed']} stopwords removed, {preprocessed['num_filtered_tokens']} tokens remaining")
            st.write(preprocessed['filtered_tokens'][:100])
            if len(preprocessed['filtered_tokens']) > 100:
                st.caption(f"... and {len(preprocessed['filtered_tokens']) - 100} more tokens")
        
        st.success("✅ Preprocessing complete. Navigate to **Candidate Analysis** to extract information.")


# ============================================================
# PAGE: Candidate Analysis (Module 2)
# ============================================================
def page_candidate_analysis():
    st.title("👤 Module 2: Candidate Information Extraction")
    st.markdown("Extract key information from the uploaded resume using NLP-based techniques.")
    
    if st.session_state.resume_text is None:
        st.warning("⚠️ Please upload a resume first in the **Resume Upload** section.")
        return
    
    st.markdown("---")
    
    # Extract information
    with st.spinner("Extracting candidate information..."):
        candidate_info = extract_all_information(st.session_state.resume_text)
        st.session_state.candidate_info = candidate_info

    # Automatically save the successfully analyzed resume once.
    # A hash prevents the same resume from being counted repeatedly
    # when Streamlit reruns the script or the user revisits this page.
    current_resume_hash = hashlib.md5(
        st.session_state.resume_text.encode("utf-8")
    ).hexdigest()

    if st.session_state.saved_resume_hash != current_resume_hash:
        processed_text = get_processed_text(st.session_state.resume_text)
        candidate_id = save_candidate(
            candidate_info,
            st.session_state.resume_text,
            processed_text
        )
        st.session_state.candidate_id = candidate_id
        st.session_state.saved_resume_hash = current_resume_hash
        st.success(
            f"✅ Information extracted and resume analyzed successfully! "
            f"Candidate ID: {candidate_id}"
        )
    else:
        st.success("✅ Candidate information already analyzed and saved successfully!")
    
    # Display candidate profile
    st.subheader("Candidate Profile")
    
    # Basic Info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="info-card">
            <h4>👤 Name</h4>
            <p style="font-size: 1.1rem; font-weight: 600;">{candidate_info['name']}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="info-card">
            <h4>📧 Email</h4>
            <p>{candidate_info['email']}</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="info-card">
            <h4>📱 Phone</h4>
            <p>{candidate_info['phone']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Skills
    st.subheader("🛠️ Technical Skills")
    if candidate_info['skills']:
        skills_html = ' '.join([f'<span class="skill-tag">{skill}</span>' for skill in candidate_info['skills']])
        st.markdown(skills_html, unsafe_allow_html=True)
        st.caption(f"Total skills identified: {len(candidate_info['skills'])}")
    else:
        st.info("No skills detected from the resume.")
    
    st.markdown("---")
    
    # Education and Experience side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎓 Education")
        if candidate_info['education']:
            for edu in candidate_info['education']:
                st.markdown(f"- {edu}")
        else:
            st.info("No education information detected.")
    
    with col2:
        st.subheader("💼 Experience")
        if candidate_info['experience']:
            for exp in candidate_info['experience']:
                st.markdown(f"- {exp}")
        else:
            st.info("No experience information detected.")
    
    st.markdown("---")
    
    # Certifications and Projects
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📜 Certifications")
        if candidate_info['certifications']:
            for cert in candidate_info['certifications']:
                st.markdown(f"- {cert}")
        else:
            st.info("No certifications detected.")
    
    with col2:
        st.subheader("📂 Projects")
        if candidate_info['projects']:
            for proj in candidate_info['projects']:
                st.markdown(f"- {proj}")
        else:
            st.info("No projects detected.")
    
    st.markdown("---")
    
    # Show raw extracted data
    with st.expander("View raw extracted data (dictionary)"):
        display_info = candidate_info.copy()
        st.json(display_info)


# ============================================================
# PAGE: Job Analysis (Module 3)
# ============================================================
def page_job_analysis():
    st.title("💼 Module 3: Job Description Analysis")
    st.markdown("Browse and analyze available job descriptions in the system.")
    
    st.markdown("---")
    
    # Get all jobs
    jobs_df = get_all_jobs()
    
    if jobs_df.empty:
        st.warning("No jobs found in the database. Please add jobs via the Admin Panel.")
        return
    
    st.subheader(f"Available Job Positions ({len(jobs_df)} jobs)")
    
    # Job listing table
    display_df = jobs_df[['id', 'job_title', 'required_education', 'experience_level']].copy()
    display_df.columns = ['ID', 'Job Title', 'Education Required', 'Experience Level']
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Job detail viewer
    st.subheader("📋 Job Details")
    
    job_titles = jobs_df['job_title'].tolist()
    selected_title = st.selectbox("Select a job to view details:", job_titles)
    
    if selected_title:
        job_row = jobs_df[jobs_df['job_title'] == selected_title].iloc[0]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"### {job_row['job_title']}")
            st.markdown(f"**Education:** {job_row['required_education']}")
            st.markdown(f"**Experience:** {job_row['experience_level']}")
            
            st.markdown("---")
            
            st.markdown("**Job Description:**")
            st.markdown(job_row['job_description'])
        
        with col2:
            st.markdown("**Required Skills:**")
            skills = [s.strip() for s in job_row['required_skills'].split(',')]
            skills_html = ' '.join([f'<span class="skill-tag">{skill}</span>' for skill in skills])
            st.markdown(skills_html, unsafe_allow_html=True)
            
            st.markdown("")
            st.markdown(f"""
            <div class="metric-card">
                <h3>Skills Required</h3>
                <p>{len(skills)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show preprocessed version
        with st.expander("View preprocessed job description"):
            if pd.notna(job_row.get('processed_description', None)):
                st.text(job_row['processed_description'])
            else:
                processed = get_processed_text(
                    f"{job_row['job_title']} {job_row['required_skills']} {job_row['job_description']}"
                )
                st.text(processed)


# ============================================================
# PAGE: Resume-Job Matching (Module 4)
# ============================================================
def page_matching():
    st.title("🔍 Module 4: Resume–Job Matching")
    st.markdown("Compare your resume with all available job descriptions using **TF-IDF Vectorization** and **Cosine Similarity**.")
    
    if st.session_state.resume_text is None:
        st.warning("⚠️ Please upload a resume first in the **Resume Upload** section.")
        return
    
    if st.session_state.candidate_info is None:
        st.warning("⚠️ Please extract candidate information first in the **Candidate Analysis** section.")
        return
    
    st.markdown("---")
    
    # Matching methodology
    with st.expander("📖 How does matching work?", expanded=False):
        st.markdown("""
        **Step 1:** The resume and job descriptions are preprocessed (cleaned, normalized, tokenized).
        
        **Step 2:** Both are converted into numerical vectors using **TF-IDF (Term Frequency–Inverse Document Frequency)** vectorization.
        
        **Step 3:** **Cosine Similarity** is calculated between the resume vector and each job description vector.
        
        **Step 4:** A **Skill Match Score** is calculated by comparing the candidate's skills with each job's required skills.
        
        **Step 5:** The final match score is a weighted combination:
        - **60% TF-IDF Cosine Similarity** (content-based matching)
        - **40% Skill Match Score** (keyword-based matching)
        """)
    
    # Run matching
    if st.button("🚀 Run Resume–Job Matching", type="primary"):
        jobs_df = get_all_jobs()
        
        if jobs_df.empty:
            st.error("No jobs found in the database. Please add jobs via the Admin Panel.")
            return
        
        with st.spinner("Calculating match scores..."):
            results = match_resume_with_jobs(
                st.session_state.resume_text,
                st.session_state.candidate_info['skills'],
                jobs_df
            )
            st.session_state.matching_results = results
            
            # Save to database if candidate was saved
            if st.session_state.candidate_id:
                save_matching_results(st.session_state.candidate_id, results)
        
        st.success("✅ Matching complete!")
    
    # Display results
    if st.session_state.matching_results:
        results = st.session_state.matching_results
        
        st.markdown("---")
        st.subheader("Match Results")
        
        # Results table
        results_table = pd.DataFrame([{
            'Rank': i + 1,
            'Job Title': r['job_title'],
            'Match Score': f"{r['match_score']}%",
            'TF-IDF Score': f"{r['tfidf_score']}%",
            'Skill Score': f"{r['skill_score']}%",
            'Skills Matched': f"{r['total_matched_skills']}/{r['total_required_skills']}"
        } for i, r in enumerate(results)])
        
        st.dataframe(results_table, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Skill Gap Analysis
        st.subheader("🔎 Skill Gap Analysis")
        
        selected_job = st.selectbox(
            "Select a job for detailed skill analysis:",
            [f"{r['job_title']} ({r['match_score']}%)" for r in results]
        )
        
        if selected_job:
            # Find the selected result
            selected_title = selected_job.rsplit(' (', 1)[0]
            selected_result = next(r for r in results if r['job_title'] == selected_title)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**✅ Matched Skills**")
                if selected_result['matched_skills']:
                    matched_html = ' '.join([
                        f'<span class="skill-tag skill-tag-matched">{s}</span>'
                        for s in selected_result['matched_skills']
                    ])
                    st.markdown(matched_html, unsafe_allow_html=True)
                else:
                    st.info("No skills matched.")
            
            with col2:
                st.markdown("**❌ Missing Skills**")
                if selected_result['missing_skills']:
                    missing_html = ' '.join([
                        f'<span class="skill-tag skill-tag-missing">{s}</span>'
                        for s in selected_result['missing_skills']
                    ])
                    st.markdown(missing_html, unsafe_allow_html=True)
                else:
                    st.success("All required skills matched!")


# ============================================================
# PAGE: Job Recommendations (Module 5)
# ============================================================
def page_recommendations():
    st.title("⭐ Module 5: Intelligent Job Recommendations")
    st.markdown("Top job recommendations based on your resume analysis and matching scores.")
    
    if st.session_state.matching_results is None:
        st.warning("⚠️ Please run the **Resume–Job Matching** first.")
        return
    
    results = st.session_state.matching_results
    candidate_skills = st.session_state.candidate_info['skills'] if st.session_state.candidate_info else []
    
    # Dashboard Stats
    st.markdown("---")
    st.subheader("📊 Results Dashboard")
    
    stats = generate_dashboard_stats(results, candidate_skills)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Jobs Analyzed</h3>
            <p>{stats['total_jobs']}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Top Match Score</h3>
            <p>{stats['top_match_score']}%</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Skills Extracted</h3>
            <p>{stats['num_skills_extracted']}</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Jobs Recommended</h3>
            <p>{stats['num_recommended']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Match score bar chart
    st.subheader("📈 Match Score Comparison")
    
    # Create bar chart with Plotly
    chart_data = results[:8]  # Show top 8 for readability
    
    fig = go.Figure(data=[
        go.Bar(
            x=[r['job_title'] for r in chart_data],
            y=[r['match_score'] for r in chart_data],
            marker_color=['#2563eb' if i < 3 else '#94a3b8' for i in range(len(chart_data))],
            text=[f"{r['match_score']}%" for r in chart_data],
            textposition='outside',
        )
    ])
    
    fig.update_layout(
        xaxis_title="Job Title",
        yaxis_title="Match Score (%)",
        yaxis_range=[0, 105],
        plot_bgcolor='white',
        height=400,
        margin=dict(t=20, b=20),
        font=dict(size=12),
    )
    fig.update_xaxes(tickangle=-30)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Top Recommendations
    st.subheader("🏆 Top 3 Job Recommendations")
    
    top_recs = get_top_recommendations(results, top_n=3)
    
    for rec in top_recs:
        st.markdown(f"""
        <div class="rec-card">
            <h3>{rec['rank_emoji']} {rec['job_title']}</h3>
            <p style="font-size: 1.2rem; font-weight: 600; color: #2563eb;">
                Match Score: {rec['match_score']}%
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**✅ Matched Skills:**")
            if rec['matched_skills']:
                matched_html = ' '.join([
                    f'<span class="skill-tag skill-tag-matched">{s}</span>'
                    for s in rec['matched_skills']
                ])
                st.markdown(matched_html, unsafe_allow_html=True)
            else:
                st.caption("None")
        
        with col2:
            st.markdown("**❌ Missing Skills:**")
            if rec['missing_skills']:
                missing_html = ' '.join([
                    f'<span class="skill-tag skill-tag-missing">{s}</span>'
                    for s in rec['missing_skills']
                ])
                st.markdown(missing_html, unsafe_allow_html=True)
            else:
                st.caption("None — all skills matched!")
        
        st.info(f"💡 **Recommendation:** {rec['recommendation']}")
        
        with st.expander("View additional details"):
            st.markdown(f"""
            - **TF-IDF Similarity Score:** {rec['tfidf_score']}%
            - **Skill Match Score:** {rec['skill_score']}%
            - **Required Education:** {rec['required_education']}
            - **Experience Level:** {rec['experience_level']}
            - **Skills Matched:** {rec['total_matched_skills']} / {rec['total_required_skills']}
            """)
        
        st.markdown("")


# ============================================================
# PAGE: Admin Panel
# ============================================================
def page_admin():
    st.title("⚙️ Admin Panel")
    st.markdown("Manage job descriptions and view candidate analysis results.")
    
    st.markdown("---")
    
    admin_tab1, admin_tab2, admin_tab3 = st.tabs(["📋 Manage Jobs", "➕ Add New Job", "👥 Candidate History"])
    
    with admin_tab1:
        st.subheader("Available Jobs")
        jobs_df = get_all_jobs()
        
        if not jobs_df.empty:
            display_df = jobs_df[['id', 'job_title', 'required_skills', 'required_education', 'experience_level']].copy()
            display_df.columns = ['ID', 'Job Title', 'Required Skills', 'Education', 'Experience']
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # Delete job
            st.subheader("Delete a Job")
            delete_options = {f"{row['job_title']} (ID: {row['id']})": row['id'] for _, row in jobs_df.iterrows()}
            selected_delete = st.selectbox("Select a job to delete:", list(delete_options.keys()))
            
            if st.button("🗑️ Delete Selected Job", type="secondary"):
                job_id = delete_options[selected_delete]
                delete_job(job_id)
                st.success(f"✅ Job deleted successfully!")
                st.rerun()
        else:
            st.info("No jobs in the database.")
    
    with admin_tab2:
        st.subheader("Add a New Job")
        
        with st.form("add_job_form"):
            new_title = st.text_input("Job Title *", placeholder="e.g., Data Engineer")
            new_skills = st.text_input(
                "Required Skills * (comma-separated)",
                placeholder="e.g., Python, SQL, Apache Spark, ETL, AWS"
            )
            new_education = st.text_input(
                "Required Education",
                placeholder="e.g., Bachelor's in Computer Science"
            )
            new_experience = st.text_input(
                "Experience Level",
                placeholder="e.g., 2-4 years"
            )
            new_description = st.text_area(
                "Job Description *",
                placeholder="Enter the full job description...",
                height=200
            )
            
            submitted = st.form_submit_button("Add Job", type="primary")
            
            if submitted:
                if new_title and new_skills and new_description:
                    add_job(new_title, new_skills, new_education, new_experience, new_description)
                    st.success(f"✅ Job '{new_title}' added successfully!")
                    st.rerun()
                else:
                    st.error("Please fill in all required fields (Title, Skills, Description).")
    
    with admin_tab3:
        st.subheader("Candidate Analysis History")
        candidates_df = get_all_candidates()
        
        if not candidates_df.empty:
            st.dataframe(candidates_df, use_container_width=True, hide_index=True)
            
            # View matching results for a candidate
            st.markdown("---")
            st.subheader("View Matching Results")
            candidate_options = {
                f"{row['name']} ({row['email']}) - ID: {row['id']}": row['id']
                for _, row in candidates_df.iterrows()
            }
            selected_candidate = st.selectbox("Select a candidate:", list(candidate_options.keys()))
            
            if selected_candidate:
                cand_id = candidate_options[selected_candidate]
                match_df = get_matching_results(cand_id)
                
                if not match_df.empty:
                    display_match = match_df[['job_title', 'match_score', 'matched_skills', 'missing_skills']].copy()
                    display_match.columns = ['Job Title', 'Match Score (%)', 'Matched Skills', 'Missing Skills']
                    st.dataframe(display_match, use_container_width=True, hide_index=True)
                else:
                    st.info("No matching results found for this candidate.")
        else:
            st.info("No candidates analyzed yet.")


# ============================================================
# Page Router
# ============================================================
if page == "🏠 Home":
    page_home()
elif page == "📤 Resume Upload":
    page_resume_upload()
elif page == "👤 Candidate Analysis":
    page_candidate_analysis()
elif page == "💼 Job Analysis":
    page_job_analysis()
elif page == "🔍 Resume–Job Matching":
    page_matching()
elif page == "⭐ Job Recommendations":
    page_recommendations()
elif page == "⚙️ Admin Panel":
    page_admin()