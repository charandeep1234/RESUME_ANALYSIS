"""
Information Extraction Module
=============================
Extracts candidate information from resume text using regex patterns
and keyword matching. This is a simple NLP-based approach suitable
for a capstone project demonstration.
"""

import re
import pandas as pd
import os


def load_skills_list() -> list:
    """
    Load the predefined skills list from the CSV file.
    
    Returns:
        list: List of skill names
    """
    skills_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'skills.csv')
    try:
        df = pd.read_csv(skills_path)
        return df['skill'].tolist()
    except Exception:
        # Fallback skills list if CSV is not available
        return [
            'Python', 'Java', 'JavaScript', 'SQL', 'R', 'C++', 'HTML', 'CSS',
            'React', 'Node.js', 'Django', 'Flask', 'Machine Learning',
            'Deep Learning', 'NLP', 'TensorFlow', 'PyTorch', 'Pandas',
            'NumPy', 'Scikit-learn', 'Data Visualization', 'Tableau',
            'Power BI', 'Excel', 'Git', 'Docker', 'AWS', 'Azure',
            'MongoDB', 'PostgreSQL', 'MySQL', 'REST APIs', 'Agile'
        ]


def extract_name(text: str) -> str:
    """
    Extract candidate name from resume text.
    Heuristic: The name is usually the first line or first few words of a resume.
    
    Args:
        text: Raw resume text
    
    Returns:
        str: Extracted candidate name or 'Not Found'
    """
    lines = text.strip().split('\n')
    
    for line in lines[:5]:  # Check first 5 lines
        line = line.strip()
        if not line:
            continue
        
        # Skip lines that look like headers/titles
        skip_keywords = ['resume', 'curriculum vitae', 'cv', 'objective', 'summary',
                         'phone', 'email', 'address', 'linkedin', 'github', 'http']
        if any(kw in line.lower() for kw in skip_keywords):
            continue
        
        # Skip lines with email or phone patterns
        if re.search(r'[@\d{10}]', line):
            continue
        
        # A name line typically contains 2-4 words, all alphabetic
        words = line.split()
        if 1 <= len(words) <= 5:
            # Check if most words are alphabetic (allowing for initials with dots)
            alpha_words = [w for w in words if re.match(r'^[A-Za-z.\-]+$', w)]
            if len(alpha_words) >= len(words) * 0.6:
                # Clean up the name
                name = ' '.join(words)
                # Remove common suffixes/prefixes
                name = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof)\b\.?', '', name).strip()
                if len(name) > 1:
                    return name.title()
    
    return "Not Found"


def extract_email(text: str) -> str:
    """
    Extract email address from resume text using regex.
    
    Args:
        text: Resume text
    
    Returns:
        str: First email found or 'Not Found'
    """
    email_pattern = r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, text)
    return emails[0] if emails else "Not Found"


def extract_phone(text: str) -> str:
    """
    Extract phone number from resume text using regex.
    Handles various phone number formats.
    
    Args:
        text: Resume text
    
    Returns:
        str: First phone number found or 'Not Found'
    """
    # Common phone number patterns
    phone_patterns = [
        r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # US format
        r'(?:\+?\d{1,3}[-.\s]?)?\d{5}[-.\s]?\d{5}',  # Indian format
        r'(?:\+?\d{1,3}[-.\s]?)?\d{10}',  # 10 digit number
    ]
    
    for pattern in phone_patterns:
        phones = re.findall(pattern, text)
        if phones:
            return phones[0].strip()
    
    return "Not Found"


def extract_skills(text: str) -> list:
    """
    Extract technical skills from resume text by matching against
    the predefined skills list. Uses case-insensitive matching.
    
    Args:
        text: Resume text
    
    Returns:
        list: List of found skills
    """
    skills_list = load_skills_list()
    text_lower = text.lower()
    
    found_skills = []
    for skill in skills_list:
        # Use word boundary matching to avoid partial matches
        # For skills with special characters (e.g., C++, Node.js), escape them
        escaped_skill = re.escape(skill.lower())
        # Allow word boundary or special character boundary
        pattern = r'(?:^|[\s,;|/(\-])' + escaped_skill + r'(?:$|[\s,;|/)\-])'
        if re.search(pattern, text_lower):
            found_skills.append(skill)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_skills = []
    for skill in found_skills:
        if skill.lower() not in seen:
            seen.add(skill.lower())
            unique_skills.append(skill)
    
    return unique_skills


def extract_education(text: str) -> list:
    """
    Extract education information from resume text using keyword matching.
    
    Args:
        text: Resume text
    
    Returns:
        list: List of education entries found
    """
    education_keywords = [
        r'B\.?Tech', r'M\.?Tech', r'B\.?E\.?', r'M\.?E\.?',
        r'B\.?Sc', r'M\.?Sc', r'B\.?S\.?', r'M\.?S\.?',
        r'B\.?A\.?', r'M\.?A\.?', r'B\.?Com', r'M\.?Com',
        r'MBA', r'MCA', r'BCA', r'PhD', r'Ph\.?D',
        r'Bachelor', r'Master', r'Diploma',
        r'Computer Science', r'Information Technology',
        r'Data Science', r'Artificial Intelligence',
        r'Software Engineering', r'Electronics',
        r'Electrical Engineering', r'Mechanical Engineering',
        r'Business Administration'
    ]
    
    found_education = []
    for keyword in education_keywords:
        pattern = re.compile(keyword, re.IGNORECASE)
        if pattern.search(text):
            # Find the line containing this education keyword
            for line in text.split('\n'):
                if pattern.search(line):
                    clean_line = line.strip()
                    if clean_line and clean_line not in found_education and len(clean_line) < 200:
                        found_education.append(clean_line)
    
    # Remove duplicates
    unique_education = list(dict.fromkeys(found_education))
    return unique_education[:5]  # Return at most 5 entries


def extract_experience(text: str) -> list:
    """
    Extract work experience entries from resume text.
    Looks for job title patterns and experience-related keywords.
    
    Args:
        text: Resume text
    
    Returns:
        list: List of experience entries
    """
    experience_keywords = [
        r'(?:Software|Web|Full[\s-]?Stack|Frontend|Backend|Data|ML|AI|Python|Java)\s*(?:Developer|Engineer|Analyst|Scientist)',
        r'(?:Senior|Junior|Lead|Associate|Principal)\s+\w+\s*(?:Developer|Engineer|Analyst|Scientist|Manager)',
        r'Intern(?:ship)?',
        r'(?:Project|Team|Technical)\s+(?:Lead|Manager|Head)',
        r'(?:Research|Teaching)\s+(?:Assistant|Associate)',
        r'(?:DevOps|Cloud|System|Network)\s+(?:Engineer|Administrator|Architect)',
        r'Consultant',
        r'Freelanc(?:er|e)',
    ]
    
    found_experience = []
    lines = text.split('\n')
    
    for keyword in experience_keywords:
        pattern = re.compile(keyword, re.IGNORECASE)
        for line in lines:
            if pattern.search(line):
                clean_line = line.strip()
                if clean_line and clean_line not in found_experience and len(clean_line) < 200:
                    found_experience.append(clean_line)
    
    unique_experience = list(dict.fromkeys(found_experience))
    return unique_experience[:5]


def extract_certifications(text: str) -> list:
    """
    Extract certifications from resume text.
    
    Args:
        text: Resume text
    
    Returns:
        list: List of certification entries
    """
    cert_keywords = [
        r'AWS\s+Certified', r'Google\s+Certified', r'Microsoft\s+Certified',
        r'Azure\s+\w+', r'Certified\s+\w+', r'Certification',
        r'Coursera', r'Udemy', r'edX', r'Udacity',
        r'PMP', r'Scrum\s+Master', r'CISSP', r'CompTIA',
        r'Oracle\s+Certified', r'Cisco\s+Certified',
        r'TensorFlow\s+Developer', r'Data\s+Engineering',
    ]
    
    found_certs = []
    lines = text.split('\n')
    
    for keyword in cert_keywords:
        pattern = re.compile(keyword, re.IGNORECASE)
        for line in lines:
            if pattern.search(line):
                clean_line = line.strip()
                if clean_line and clean_line not in found_certs and len(clean_line) < 200:
                    found_certs.append(clean_line)
    
    unique_certs = list(dict.fromkeys(found_certs))
    return unique_certs[:5]


def extract_projects(text: str) -> list:
    """
    Extract project information from resume text.
    Looks for sections labeled 'Projects' and extracts entries.
    
    Args:
        text: Resume text
    
    Returns:
        list: List of project entries
    """
    lines = text.split('\n')
    projects = []
    in_project_section = False
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        # Check if we're entering a projects section
        if re.match(r'^(?:projects?|personal\s+projects?|academic\s+projects?|key\s+projects?)\s*:?\s*$',
                     line_stripped, re.IGNORECASE):
            in_project_section = True
            continue
        
        # Check if we're leaving the projects section (new section header)
        if in_project_section and re.match(
            r'^(?:education|experience|skills|certifications?|achievements?|awards?|references?|hobbies|interests|summary|objective)\s*:?\s*$',
            line_stripped, re.IGNORECASE):
            in_project_section = False
            continue
        
        # Collect project lines
        if in_project_section and line_stripped and len(line_stripped) < 200:
            projects.append(line_stripped)
    
    # If no project section found, look for project-related keywords
    if not projects:
        project_patterns = [
            r'(?:developed|built|created|designed|implemented)\s+(?:a\s+)?[\w\s]+(?:system|application|app|tool|platform|website|model|pipeline)',
        ]
        for pattern in project_patterns:
            for line in lines:
                if re.search(pattern, line, re.IGNORECASE):
                    clean_line = line.strip()
                    if clean_line and len(clean_line) < 200:
                        projects.append(clean_line)
    
    return projects[:8]  # Return at most 8 project entries


def extract_all_information(text: str) -> dict:
    """
    Extract all candidate information from resume text.
    This is the main function that calls all individual extractors.
    
    Args:
        text: Raw resume text
    
    Returns:
        dict: Dictionary containing all extracted information
    """
    return {
        'name': extract_name(text),
        'email': extract_email(text),
        'phone': extract_phone(text),
        'skills': extract_skills(text),
        'education': extract_education(text),
        'experience': extract_experience(text),
        'certifications': extract_certifications(text),
        'projects': extract_projects(text)
    }
