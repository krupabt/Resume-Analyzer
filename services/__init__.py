"""
Resume Analyzer Services Package
"""

from .parser import (
    parse_document,
    clean_text,
    DocumentParsingError,
    EmptyDocumentError,
    UnreadableDocumentError,
    UnsupportedFormatError,
    is_allowed_file,
    sanitize_filename
)
from .skills import (
    extract_skills,
    SKILLS_TAXONOMY,
    SKILL_ALIASES,
    SKILL_TO_CATEGORY
)
from .matcher import (
    calculate_similarity_and_skills,
    rank_resumes,
    extract_top_keywords
)
from .analyzer import (
    analyze_candidate_resume,
    detect_sections,
    generate_recommendations
)

__all__ = [
    "parse_document",
    "clean_text",
    "DocumentParsingError",
    "EmptyDocumentError",
    "UnreadableDocumentError",
    "UnsupportedFormatError",
    "is_allowed_file",
    "sanitize_filename",
    "extract_skills",
    "SKILLS_TAXONOMY",
    "SKILL_ALIASES",
    "SKILL_TO_CATEGORY",
    "calculate_similarity_and_skills",
    "rank_resumes",
    "extract_top_keywords",
    "analyze_candidate_resume",
    "detect_sections",
    "generate_recommendations"
]
