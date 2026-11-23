"""
Relevance Matcher & Ranking Service
Implements TF-IDF Vectorization, Cosine Similarity, Skill Gap Analysis,
and Transparent Composite Scoring.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .skills import extract_skills
from .parser import clean_text


def extract_top_keywords(text: str, top_n: int = 15) -> List[str]:
    """Extract top descriptive keywords from a text using TF-IDF feature weighting."""
    if not text or len(text.strip().split()) < 3:
        return []

    try:
        vec = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=100,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_\-\.]{2,}\b"
        )
        tfidf_matrix = vec.fit_transform([text])
        feature_names = vec.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]

        # Sort features by score descending
        top_indices = np.argsort(scores)[::-1]
        keywords = []
        for idx in top_indices:
            word = feature_names[idx]
            if len(word) > 2 and word not in keywords:
                keywords.append(word)
            if len(keywords) >= top_n:
                break
        return keywords
    except Exception:
        # Fallback to simple unique words
        words = [w for w in clean_text(text).split() if len(w) > 3]
        return list(dict.fromkeys(words))[:top_n]


def calculate_similarity_and_skills(
    job_description: str,
    resume_text: str
) -> Dict[str, Any]:
    """
    Perform complete matching analysis between a single resume and job description.
    Computes text cosine similarity, skill overlap, keyword gaps, and composite relevance.
    """
    cleaned_jd = clean_text(job_description)
    cleaned_resume = clean_text(resume_text)

    # 1. TF-IDF Cosine Similarity
    text_similarity_pct = 0.0
    if cleaned_jd and cleaned_resume:
        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=2500,
                token_pattern=r"(?u)\b[a-zA-Z0-9\+\#\.\-]{2,}\b"
            )
            tfidf_matrix = vectorizer.fit_transform([cleaned_jd, cleaned_resume])
            cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            text_similarity_pct = round(float(np.nan_to_num(cos_sim, nan=0.0)) * 100, 2)
        except Exception:
            text_similarity_pct = 0.0

    # 2. Skill Extraction & Gap Analysis
    jd_skills_data = extract_skills(job_description)
    resume_skills_data = extract_skills(resume_text)

    jd_skills = set(jd_skills_data["skills"])
    resume_skills = set(resume_skills_data["skills"])

    matched_skills = sorted(list(jd_skills.intersection(resume_skills)))
    missing_skills = sorted(list(jd_skills.difference(resume_skills)))
    additional_skills = sorted(list(resume_skills.difference(jd_skills)))

    if len(jd_skills) > 0:
        skill_match_pct = round((len(matched_skills) / len(jd_skills)) * 100, 2)
    else:
        # If no specific tech skills are mentioned in JD, rely entirely on text similarity
        skill_match_pct = text_similarity_pct

    # 3. Transparent Composite Score
    # 50% Text Cosine Similarity + 50% Explicit Skills Alignment
    if len(jd_skills) > 0:
        composite_score = round(0.50 * text_similarity_pct + 0.50 * skill_match_pct, 2)
    else:
        composite_score = text_similarity_pct

    composite_score = max(0.0, min(100.0, composite_score))

    # 4. Keyword Analysis
    jd_keywords = extract_top_keywords(job_description, top_n=20)
    resume_keywords_set = set(clean_text(resume_text).split())

    matched_keywords = []
    missing_keywords = []
    for kw in jd_keywords:
        # Check if keyword or phrase is in resume
        if kw.lower() in cleaned_resume:
            matched_keywords.append(kw)
        else:
            missing_keywords.append(kw)

    return {
        "relevance_score": composite_score,
        "text_similarity_score": text_similarity_pct,
        "skill_match_percentage": skill_match_pct,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "additional_skills": additional_skills,
        "jd_skills_count": len(jd_skills),
        "resume_skills_count": len(resume_skills),
        "matched_keywords": matched_keywords[:12],
        "missing_keywords": missing_keywords[:12],
        "breakdown": {
            "text_weight": "50%" if len(jd_skills) > 0 else "100%",
            "skill_weight": "50%" if len(jd_skills) > 0 else "0%",
            "text_score": text_similarity_pct,
            "skill_score": skill_match_pct
        }
    }


def rank_resumes(
    job_description: str,
    resumes: List[Dict[str, str]]
) -> List[Dict[str, Any]]:
    """
    Rank a list of resumes against a job description.
    Each resume item in `resumes` should be a dict: {"name": str, "text": str}.
    Uses deterministic, stable tie-breaking:
    1. Relevance Score (descending)
    2. Skill Match Rate (descending)
    3. Filename (ascending, case-insensitive)
    """
    if not resumes:
        return []

    results = []
    for idx, item in enumerate(resumes):
        analysis = calculate_similarity_and_skills(job_description, item["text"])
        results.append({
            "name": item["name"],
            "relevance_score": analysis["relevance_score"],
            "text_similarity_score": analysis["text_similarity_score"],
            "skill_match_percentage": analysis["skill_match_percentage"],
            "matched_skills": analysis["matched_skills"],
            "missing_skills": analysis["missing_skills"],
            "matched_count": len(analysis["matched_skills"]),
            "missing_count": len(analysis["missing_skills"]),
            "matched_keywords": analysis["matched_keywords"],
            "missing_keywords": analysis["missing_keywords"],
            "breakdown": analysis["breakdown"]
        })

    # Deterministic sorting
    results.sort(
        key=lambda r: (-r["relevance_score"], -r["skill_match_percentage"], r["name"].lower())
    )

    # Assign rank numbers and top 3 badges
    for rank_idx, res in enumerate(results, start=1):
        res["rank"] = rank_idx
        res["is_top_three"] = rank_idx <= 3

    return results
