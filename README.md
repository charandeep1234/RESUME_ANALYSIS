# NLP-Driven Resume Screening and Intelligent Job Recommendation System

## About
This is a capstone project that demonstrates an NLP-based pipeline for resume screening and job recommendation. The system processes uploaded resumes, extracts candidate information, and matches them with job descriptions using TF-IDF vectorization and cosine similarity.

## Modules
1. **Resume Data Collection & Preprocessing** — Upload and process resumes (PDF/DOCX)
2. **Candidate Information & Skill Extraction** — Extract name, email, skills, education, etc.
3. **Job Description Analysis** — Browse and manage job descriptions
4. **Resume–Job Matching & Screening** — TF-IDF + Cosine Similarity matching with skill gap analysis
5. **Intelligent Job Recommendation** — Top-3 ranked recommendations with dashboard

## Project Structure
```
NLP/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/
│   ├── jobs.csv                    # Sample job descriptions
│   └── skills.csv                  # Predefined skills list
├── modules/
│   ├── __init__.py
│   ├── resume_parser.py            # PDF/DOCX text extraction
│   ├── preprocessing.py            # Text cleaning, normalization, tokenization
│   ├── information_extraction.py   # Candidate info extraction (NLP + regex)
│   ├── job_analysis.py             # Job CRUD operations, database management
│   ├── matching.py                 # TF-IDF + Cosine Similarity matching
│   └── recommendation.py           # Job recommendation and dashboard stats
└── database/
    └── app.db                      # SQLite database (auto-created)
```

## Technologies Used
- **Python 3.x** — Core programming language
- **Streamlit** — Web interface framework
- **NLTK** — Tokenization and stopword removal
- **Scikit-learn** — TF-IDF vectorization and cosine similarity
- **pdfplumber** — PDF text extraction
- **python-docx** — DOCX text extraction
- **Pandas** — Data manipulation
- **SQLite** — Data storage
- **Plotly** — Data visualization

## Setup & Installation

1. Install Python 3.8 or higher

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run app.py
   ```

4. The database and sample data will be created automatically on first run.

## How to Use
1. Navigate to **Resume Upload** and upload a PDF/DOCX resume (or use the demo resume)
2. Go to **Candidate Analysis** to view extracted information
3. Browse available jobs in **Job Analysis**
4. Run matching in **Resume–Job Matching** to see scores and skill gaps
5. View top recommendations in **Job Recommendations**
6. Use the **Admin Panel** to add/remove jobs and view history

## Matching Algorithm
The match score is calculated using a weighted combination:
- **60%** — TF-IDF Cosine Similarity (content-based matching)
- **40%** — Skill Match Score (keyword-based matching)

This approach balances overall content relevance with specific skill requirements.
