"""
Resume-Job Matching Module
===========================
Core module that compares processed resume text with job descriptions
using TF-IDF Vectorization and Cosine Similarity.
Also performs skill-based matching for detailed analysis.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from modules.preprocessing import get_processed_text


def calculate_tfidf_similarity(resume_text: str, job_texts: list) -> list:
    """
    Calculate cosine similarity between a resume and multiple job descriptions
    using TF-IDF vectorization.
    
    Workflow:
    1. Combine resume and all job descriptions into a corpus
    2. Create TF-IDF vectors for all documents
    3. Calculate cosine similarity between resume vector and each job vector
    
    Args:
        resume_text: Processed resume text
        job_texts: List of processed job description texts
    
    Returns:
        list: List of similarity scores (0 to 1) for each job
    """
    if not resume_text or not job_texts:
        return [0.0] * len(job_texts)
    
    # Create the corpus: resume first, then all job descriptions
    corpus = [resume_text] + job_texts
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=5000,    # Limit vocabulary size
        ngram_range=(1, 2),   # Use unigrams and bigrams
        stop_words='english'  # Remove English stopwords
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        # If vectorization fails (e.g., empty documents), return zeros
        return [0.0] * len(job_texts)
    
    # Calculate cosine similarity between resume (index 0) and each job
    resume_vector = tfidf_matrix[0:1]  # First document is the resume
    job_vectors = tfidf_matrix[1:]      # Remaining are job descriptions
    
    similarities = cosine_similarity(resume_vector, job_vectors)[0]
    
    return similarities.tolist()


def calculate_skill_match(candidate_skills: list, job_skills_str: str) -> dict:
    """
    Compare candidate skills with job required skills.
    Returns matched and missing skills.
    
    Args:
        candidate_skills: List of skills extracted from the resume
        job_skills_str: Comma-separated string of required job skills
    
    Returns:
        dict: Dictionary with matched_skills, missing_skills, and skill_match_score
    """
    # Parse job skills from comma-separated string
    job_skills = [s.strip() for s in job_skills_str.split(',') if s.strip()]
    
    # Normalize for comparison
    candidate_skills_lower = {s.lower() for s in candidate_skills}
    
    matched = []
    missing = []
    
    for skill in job_skills:
        if skill.lower() in candidate_skills_lower:
            matched.append(skill)
        else:
            missing.append(skill)
    
    # Calculate skill match percentage
    total_required = len(job_skills)
    if total_required > 0:
        skill_score = len(matched) / total_required
    else:
        skill_score = 0.0
    
    return {
        'matched_skills': matched,
        'missing_skills': missing,
        'skill_match_score': skill_score,
        'total_required': total_required,
        'total_matched': len(matched)
    }


def match_resume_with_jobs(resume_text: str, candidate_skills: list, jobs_df) -> list:
    """
    Main matching function that combines TF-IDF similarity and skill matching
    to produce a final match score for each job.
    
    The final score is a weighted combination:
    - 60% TF-IDF Cosine Similarity (content-based)
    - 40% Skill Match Score (skill-based)
    
    Args:
        resume_text: Raw resume text
        candidate_skills: List of skills extracted from the resume
        jobs_df: DataFrame containing job records
    
    Returns:
        list: List of dictionaries with matching results, sorted by score
    """
    if jobs_df.empty:
        return []
    
    # Preprocess resume text for TF-IDF
    processed_resume = get_processed_text(resume_text)
    
    # Get processed job descriptions
    # Use pre-processed text if available, otherwise process on the fly
    job_texts = []
    for _, job in jobs_df.iterrows():
        if pd.notna(job.get('processed_description', None)) and job['processed_description']:
            job_texts.append(job['processed_description'])
        else:
            combined = f"{job['job_title']} {job['required_skills']} {job['job_description']}"
            job_texts.append(get_processed_text(combined))
    
    # Calculate TF-IDF cosine similarity scores
    tfidf_scores = calculate_tfidf_similarity(processed_resume, job_texts)
    
    # Build results for each job
    results = []
    for i, (_, job) in enumerate(jobs_df.iterrows()):
        # Calculate skill match
        skill_result = calculate_skill_match(candidate_skills, job['required_skills'])
        
        # Calculate combined score (weighted average)
        tfidf_score = tfidf_scores[i]
        skill_score = skill_result['skill_match_score']
        
        # Weighted combination: 60% TF-IDF + 40% Skill Match
        combined_score = (0.6 * tfidf_score) + (0.4 * skill_score)
        
        # Convert to percentage
        match_percentage = round(combined_score * 100, 1)
        
        results.append({
            'job_id': int(job['id']),
            'job_title': job['job_title'],
            'match_score': match_percentage,
            'tfidf_score': round(tfidf_score * 100, 1),
            'skill_score': round(skill_score * 100, 1),
            'matched_skills': skill_result['matched_skills'],
            'missing_skills': skill_result['missing_skills'],
            'total_required_skills': skill_result['total_required'],
            'total_matched_skills': skill_result['total_matched'],
            'required_education': job.get('required_education', ''),
            'experience_level': job.get('experience_level', ''),
        })
    
    # Sort by match score (highest first)
    results.sort(key=lambda x: x['match_score'], reverse=True)
    
    return results


# Need pandas for the function above
import pandas as pd
