"""
Job Recommendation Module
=========================
Generates intelligent job recommendations based on matching scores.
Provides recommendation messages and summary statistics.
"""


def generate_recommendation_message(match_score: float, matched_skills: list, missing_skills: list) -> str:
    """
    Generate a personalized recommendation message based on the match score
    and skill analysis.
    
    Args:
        match_score: The overall match percentage
        matched_skills: List of skills that match the job requirements
        missing_skills: List of skills the candidate is missing
    
    Returns:
        str: A recommendation message
    """
    if match_score >= 80:
        base_msg = "Your profile is highly suitable for this role."
        if missing_skills:
            return f"{base_msg} Consider learning {', '.join(missing_skills[:3])} to further strengthen your candidacy."
        else:
            return f"{base_msg} You meet all the required skill criteria. Apply with confidence!"
    
    elif match_score >= 60:
        base_msg = "You are a good fit for this role with some skill gaps."
        if missing_skills:
            return f"{base_msg} Focus on acquiring skills in {', '.join(missing_skills[:3])} to improve your match."
        else:
            return f"{base_msg} Your experience aligns well with this position."
    
    elif match_score >= 40:
        base_msg = "You have a moderate match for this role."
        if missing_skills:
            return f"{base_msg} Building expertise in {', '.join(missing_skills[:3])} would significantly improve your chances."
        else:
            return f"{base_msg} Consider gaining more experience in this domain."
    
    else:
        base_msg = "This role may require significant skill development."
        if missing_skills:
            return f"{base_msg} Key areas to develop include {', '.join(missing_skills[:3])}."
        else:
            return f"{base_msg} Consider gaining more relevant experience first."


def get_top_recommendations(matching_results: list, top_n: int = 3) -> list:
    """
    Get the top N job recommendations with detailed information.
    
    Args:
        matching_results: List of matching result dictionaries (already sorted by score)
        top_n: Number of top recommendations to return
    
    Returns:
        list: Top N recommendations with added recommendation messages
    """
    top_results = matching_results[:top_n]
    
    # Add recommendation messages and rank
    rank_emojis = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣']
    
    for i, result in enumerate(top_results):
        result['rank'] = i + 1
        result['rank_emoji'] = rank_emojis[i] if i < len(rank_emojis) else f"{i+1}."
        result['recommendation'] = generate_recommendation_message(
            result['match_score'],
            result['matched_skills'],
            result['missing_skills']
        )
    
    return top_results


def generate_dashboard_stats(matching_results: list, candidate_skills: list) -> dict:
    """
    Generate summary statistics for the results dashboard.
    
    Args:
        matching_results: List of all matching result dictionaries
        candidate_skills: List of candidate's extracted skills
    
    Returns:
        dict: Dashboard statistics
    """
    if not matching_results:
        return {
            'total_jobs': 0,
            'top_match_score': 0,
            'num_skills_extracted': len(candidate_skills),
            'num_recommended': 0,
            'avg_match_score': 0,
            'jobs_above_50': 0,
            'jobs_above_70': 0,
        }
    
    scores = [r['match_score'] for r in matching_results]
    
    return {
        'total_jobs': len(matching_results),
        'top_match_score': max(scores),
        'num_skills_extracted': len(candidate_skills),
        'num_recommended': min(3, len(matching_results)),
        'avg_match_score': round(sum(scores) / len(scores), 1),
        'jobs_above_50': sum(1 for s in scores if s >= 50),
        'jobs_above_70': sum(1 for s in scores if s >= 70),
    }
