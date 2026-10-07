"""
Job Analysis Module
===================
Handles loading, preprocessing, and managing job descriptions.
Reads from CSV and interacts with the SQLite database.
"""

import pandas as pd
import os
import sqlite3
from modules.preprocessing import get_processed_text


def get_db_path() -> str:
    """Get the path to the SQLite database file."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'app.db')


def get_csv_path() -> str:
    """Get the path to the jobs CSV file."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'jobs.csv')


def init_database():
    """
    Initialize the SQLite database and create tables if they don't exist.
    Also loads initial job data from CSV if the jobs table is empty.
    """
    db_path = get_db_path()
    
    # Create database directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create jobs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_title TEXT NOT NULL,
            required_skills TEXT NOT NULL,
            required_education TEXT,
            experience_level TEXT,
            job_description TEXT NOT NULL,
            processed_description TEXT
        )
    ''')
    
    # Create candidates table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            phone TEXT,
            skills TEXT,
            education TEXT,
            experience TEXT,
            certifications TEXT,
            projects TEXT,
            raw_resume_text TEXT,
            processed_resume_text TEXT,
            upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create matching_results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS matching_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id INTEGER,
            job_id INTEGER,
            match_score REAL,
            matched_skills TEXT,
            missing_skills TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (candidate_id) REFERENCES candidates(id),
            FOREIGN KEY (job_id) REFERENCES jobs(id)
        )
    ''')
    
    conn.commit()
    
    # Check if jobs table is empty, if so load from CSV
    cursor.execute('SELECT COUNT(*) FROM jobs')
    count = cursor.fetchone()[0]
    
    if count == 0:
        load_jobs_from_csv(conn)
    
    conn.close()


def load_jobs_from_csv(conn=None):
    """
    Load job data from the CSV file into the database.
    
    Args:
        conn: SQLite connection (optional, creates new one if None)
    """
    csv_path = get_csv_path()
    close_conn = False
    
    if conn is None:
        conn = sqlite3.connect(get_db_path())
        close_conn = True
    
    try:
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            # Preprocess the job description for later matching
            combined_text = f"{row['job_title']} {row['required_skills']} {row['job_description']}"
            processed = get_processed_text(combined_text)
            
            conn.execute('''
                INSERT INTO jobs (job_title, required_skills, required_education, 
                                  experience_level, job_description, processed_description)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                row['job_title'],
                row['required_skills'],
                row['required_education'],
                row['experience_level'],
                row['job_description'],
                processed
            ))
        
        conn.commit()
    except Exception as e:
        print(f"Error loading jobs from CSV: {e}")
    
    if close_conn:
        conn.close()


def get_all_jobs() -> pd.DataFrame:
    """
    Retrieve all jobs from the database.
    
    Returns:
        pd.DataFrame: DataFrame containing all job records
    """
    conn = sqlite3.connect(get_db_path())
    df = pd.read_sql_query('SELECT * FROM jobs', conn)
    conn.close()
    return df


def get_job_by_id(job_id: int) -> dict:
    """
    Retrieve a specific job by its ID.
    
    Args:
        job_id: The job's database ID
    
    Returns:
        dict: Job record as a dictionary
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM jobs WHERE id = ?', (job_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        columns = ['id', 'job_title', 'required_skills', 'required_education',
                    'experience_level', 'job_description', 'processed_description']
        return dict(zip(columns, row))
    return None


def add_job(job_title, required_skills, required_education, experience_level, job_description):
    """
    Add a new job to the database.
    
    Args:
        job_title: Title of the job
        required_skills: Comma-separated skills string
        required_education: Education requirement
        experience_level: Experience level requirement
        job_description: Full job description text
    """
    conn = sqlite3.connect(get_db_path())
    
    # Preprocess the job description
    combined_text = f"{job_title} {required_skills} {job_description}"
    processed = get_processed_text(combined_text)
    
    conn.execute('''
        INSERT INTO jobs (job_title, required_skills, required_education,
                          experience_level, job_description, processed_description)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (job_title, required_skills, required_education, experience_level,
          job_description, processed))
    
    conn.commit()
    conn.close()


def delete_job(job_id: int):
    """
    Delete a job from the database.
    
    Args:
        job_id: The job's database ID
    """
    conn = sqlite3.connect(get_db_path())
    conn.execute('DELETE FROM jobs WHERE id = ?', (job_id,))
    # Also delete related matching results
    conn.execute('DELETE FROM matching_results WHERE job_id = ?', (job_id,))
    conn.commit()
    conn.close()


def save_candidate(candidate_info: dict, raw_text: str, processed_text: str) -> int:
    """
    Save candidate information to the database.
    
    Args:
        candidate_info: Dictionary with extracted candidate information
        raw_text: Original resume text
        processed_text: Preprocessed resume text
    
    Returns:
        int: The ID of the saved candidate record
    """
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO candidates (name, email, phone, skills, education,
                                experience, certifications, projects,
                                raw_resume_text, processed_resume_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        candidate_info.get('name', ''),
        candidate_info.get('email', ''),
        candidate_info.get('phone', ''),
        ', '.join(candidate_info.get('skills', [])),
        ' | '.join(candidate_info.get('education', [])),
        ' | '.join(candidate_info.get('experience', [])),
        ' | '.join(candidate_info.get('certifications', [])),
        ' | '.join(candidate_info.get('projects', [])),
        raw_text,
        processed_text
    ))
    
    conn.commit()
    candidate_id = cursor.lastrowid
    conn.close()
    
    return candidate_id


def save_matching_results(candidate_id: int, results: list):
    """
    Save matching results to the database.
    
    Args:
        candidate_id: The candidate's database ID
        results: List of matching result dictionaries
    """
    conn = sqlite3.connect(get_db_path())
    
    # Clear previous results for this candidate
    conn.execute('DELETE FROM matching_results WHERE candidate_id = ?', (candidate_id,))
    
    for result in results:
        conn.execute('''
            INSERT INTO matching_results (candidate_id, job_id, match_score,
                                          matched_skills, missing_skills)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            candidate_id,
            result['job_id'],
            result['match_score'],
            ', '.join(result.get('matched_skills', [])),
            ', '.join(result.get('missing_skills', []))
        ))
    
    conn.commit()
    conn.close()


def get_all_candidates() -> pd.DataFrame:
    """
    Retrieve all candidates from the database.
    
    Returns:
        pd.DataFrame: DataFrame containing all candidate records
    """
    conn = sqlite3.connect(get_db_path())
    df = pd.read_sql_query('SELECT id, name, email, phone, skills, upload_timestamp FROM candidates ORDER BY upload_timestamp DESC', conn)
    conn.close()
    return df


def get_matching_results(candidate_id: int) -> pd.DataFrame:
    """
    Retrieve matching results for a candidate.
    
    Args:
        candidate_id: The candidate's database ID
    
    Returns:
        pd.DataFrame: DataFrame with matching results
    """
    conn = sqlite3.connect(get_db_path())
    df = pd.read_sql_query('''
        SELECT mr.*, j.job_title 
        FROM matching_results mr 
        JOIN jobs j ON mr.job_id = j.id 
        WHERE mr.candidate_id = ?
        ORDER BY mr.match_score DESC
    ''', conn, params=(candidate_id,))
    conn.close()
    return df
