"""
Resume Section Analyzer & Evidence-Based Recommendation Service
Provides heuristic section detection, strength identification, gap analysis,
and actionable suggestions without hallucination.
"""

import re
from typing import Dict, List, Any
from .matcher import calculate_similarity_and_skills
from .skills import extract_skills, SKILL_TO_CATEGORY

# Section pattern definitions
SECTION_PATTERNS: Dict[str, List[str]] = {
    "Professional Summary": [
        r"\b(?:professional\s+summary|executive\s+summary|summary\s+of\s+qualifications|professional\s+profile|profile|about\s+me|career\s+objective|objective)\b"
    ],
    "Work Experience": [
        r"\b(?:work\s+experience|professional\s+experience|employment\s+history|experience|work\s+history|career\s+history)\b"
    ],
    "Education": [
        r"\b(?:education|academic\s+background|academic\s+history|educational\s+qualifications|degrees|academics)\b"
    ],
    "Technical Skills": [
        r"\b(?:technical\s+skills|skills\s+(?:&|and)\s+technologies|core\s+competencies|technologies|tech\s+stack|skills|competencies)\b"
    ],
    "Projects": [
        r"\b(?:projects|key\s+projects|technical\s+projects|personal\s+projects|academic\s+projects|selected\s+projects)\b"
    ],
    "Internships": [
        r"\b(?:internships?|industrial\s+training|practical\s+training)\b"
    ],
    "Certifications": [
        r"\b(?:certifications?|certificates?|licenses?\s+(?:&|and)\s+certifications?|accreditations?)\b"
    ],
    "Achievements": [
        r"\b(?:achievements?|honors?\s+(?:&|and)\s+awards?|awards?|accomplishments?|honors?)\b"
    ]
}


def detect_sections(text: str) -> Dict[str, Any]:
    """
    Detect standard resume sections heuristically by scanning lines and headings.
    Does not penalize documents that use alternative headings.
    """
    detected: Dict[str, bool] = {}
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]

    # Scan entire text and per line
    for section_name, patterns in SECTION_PATTERNS.items():
        found = False
        for pattern in patterns:
            # Check standalone headings or starts of lines
            for line in lines[:80]:  # Limit heading scan to first 80 lines or per-line search
                if len(line) < 50 and re.search(pattern, line):
                    found = True
                    break
            if not found and re.search(pattern, text.lower()):
                found = True
            if found:
                break
        detected[section_name] = found

    detected_count = sum(1 for is_found in detected.values() if is_found)

    return {
        "sections": detected,
        "detected_count": detected_count,
        "total_sections": len(SECTION_PATTERNS),
        "coverage_percentage": round((detected_count / len(SECTION_PATTERNS)) * 100, 1),
        "note": "Section detection uses heuristic heading analysis. Non-standard headings or blended sections may not be flagged."
    }


def generate_recommendations(
    analysis_data: Dict[str, Any],
    section_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate evidence-based feedback, concrete strengths, areas for improvement,
    and actionable recommendations strictly grounded in the document content.
    """
    matched_skills = analysis_data.get("matched_skills", [])
    missing_skills = analysis_data.get("missing_skills", [])
    relevance_score = analysis_data.get("relevance_score", 0.0)
    skill_match_pct = analysis_data.get("skill_match_percentage", 0.0)
    missing_keywords = analysis_data.get("missing_keywords", [])
    sections = section_data.get("sections", {})

    strengths: List[str] = []
    improvements: List[str] = []
    action_items: List[str] = []

    # 1. Strengths evaluation (Evidence-based only)
    if matched_skills:
        top_matched_preview = ", ".join(matched_skills[:5])
        strengths.append(
            f"Verified technical alignment in key required skills: {top_matched_preview}."
        )

    if skill_match_pct >= 70:
        strengths.append(
            f"Strong skill alignment rate ({skill_match_pct}%) against the target job requirements."
        )
    elif skill_match_pct >= 40:
        strengths.append(
            f"Moderate core skill alignment ({skill_match_pct}%). Foundational qualifications are present."
        )

    if sections.get("Technical Skills") and sections.get("Work Experience"):
        strengths.append(
            "Clear document structure containing both dedicated Skills and Work Experience sections."
        )

    if sections.get("Projects"):
        strengths.append(
            "Includes practical project showcases, reinforcing hands-on technical proficiency."
        )

    if not strengths:
        strengths.append(
            "Document parsed cleanly. Foundational content is ready for targeted alignment."
        )

    # 2. Areas for improvement
    if missing_skills:
        high_priority_missing = ", ".join(missing_skills[:6])
        improvements.append(
            f"Critical skills mentioned in job description not found: {high_priority_missing}."
        )

    if not sections.get("Professional Summary"):
        improvements.append(
            "No Professional Summary or Profile heading detected. A concise 3-4 sentence overview creates a strong initial impression."
        )

    if not sections.get("Projects"):
        improvements.append(
            "No dedicated Projects section detected. Showcasing 2-3 end-to-end technical projects provides practical evidence of skills."
        )

    if not sections.get("Certifications"):
        improvements.append(
            "No Certifications section detected. Relevant industry certifications (AWS, GCP, CKA, etc.) can boost candidate credibility."
        )

    if missing_keywords:
        missing_kw_preview = ", ".join(missing_keywords[:6])
        improvements.append(
            f"Job-specific terminology underrepresented in resume: {missing_kw_preview}."
        )

    # 3. Actionable recommendations
    if missing_skills:
        action_items.append(
            f"Incorporate missing core competencies ({', '.join(missing_skills[:4])}) if you have practical experience with them, specifically in project or work experience bullet points."
        )

    action_items.append(
        "Quantify your accomplishments using metrics (e.g., 'Reduced query latency by 35%', 'Automated CI/CD pipeline saving 4 hours weekly')."
    )

    action_items.append(
        "Ensure technical tools and frameworks are mentioned in the context of projects rather than only in a disconnected skills list."
    )

    if missing_keywords:
        action_items.append(
            f"Review target job terminology ({', '.join(missing_keywords[:4])}) and naturally weave relevant terms into your experience descriptions."
        )

    return {
        "strengths": strengths,
        "improvements": improvements,
        "action_items": action_items
    }


def analyze_candidate_resume(
    job_description: str,
    resume_text: str,
    filename: str = "Resume"
) -> Dict[str, Any]:
    """
    Complete analysis pipeline for Candidate Mode.
    """
    matching_result = calculate_similarity_and_skills(job_description, resume_text)
    section_result = detect_sections(resume_text)
    recommendations = generate_recommendations(matching_result, section_result)

    # Breakdown of candidate's skills by category
    skills_data = extract_skills(resume_text)

    return {
        "filename": filename,
        "relevance_score": matching_result["relevance_score"],
        "text_similarity_score": matching_result["text_similarity_score"],
        "skill_match_percentage": matching_result["skill_match_percentage"],
        "matched_skills": matching_result["matched_skills"],
        "missing_skills": matching_result["missing_skills"],
        "additional_skills": matching_result["additional_skills"],
        "matched_keywords": matching_result["matched_keywords"],
        "missing_keywords": matching_result["missing_keywords"],
        "skills_by_category": skills_data["by_category"],
        "sections": section_result["sections"],
        "section_coverage": section_result["coverage_percentage"],
        "strengths": recommendations["strengths"],
        "improvements": recommendations["improvements"],
        "action_items": recommendations["action_items"],
        "breakdown": matching_result["breakdown"]
    }
