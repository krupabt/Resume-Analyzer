"""
Tests for Matcher & Skill Extraction Services
"""

import pytest
from services.skills import extract_skills
from services.matcher import calculate_similarity_and_skills, rank_resumes, extract_top_keywords


class TestSkillExtraction:
    def test_extract_canonical_skills(self):
        text = "Strong background in Python, Docker, PostgreSQL, and AWS."
        extracted = extract_skills(text)
        skills = extracted["skills"]
        assert "Python" in skills
        assert "Docker" in skills
        assert "PostgreSQL" in skills
        assert "AWS" in skills

    def test_extract_aliases(self):
        text = "Proficient in js, ts, k8s, postgres, and cicd."
        extracted = extract_skills(text)
        skills = extracted["skills"]
        assert "JavaScript" in skills
        assert "TypeScript" in skills
        assert "Kubernetes" in skills
        assert "PostgreSQL" in skills
        assert "CI/CD" in skills

    def test_extract_multi_word_skills(self):
        text = "Experience with Machine Learning, Natural Language Processing, and REST API design."
        extracted = extract_skills(text)
        skills = extracted["skills"]
        assert "Machine Learning" in skills
        assert "Natural Language Processing" in skills
        assert "REST API" in skills

    def test_extract_skills_empty_text(self):
        extracted = extract_skills("")
        assert extracted["skills"] == []
        assert extracted["count"] == 0


class TestRelevanceMatcher:
    def test_high_similarity_match(self):
        jd = "Senior Python Developer with Flask, Docker, and PostgreSQL experience."
        resume = "Senior Python Developer specializing in Flask web apps, Docker containers, and PostgreSQL databases."
        result = calculate_similarity_and_skills(jd, resume)
        assert result["relevance_score"] > 60.0
        assert "Python" in result["matched_skills"]
        assert "Flask" in result["matched_skills"]
        assert len(result["missing_skills"]) == 0

    def test_zero_or_low_similarity_unrelated(self):
        jd = "Machine Learning Engineer specialized in PyTorch, Computer Vision, and Deep Learning."
        resume = "Executive Pastry Chef experienced in French baking, sourdough fermentation, and cake decorating."
        result = calculate_similarity_and_skills(jd, resume)
        assert result["relevance_score"] < 15.0
        assert len(result["matched_skills"]) == 0
        assert len(result["missing_skills"]) >= 3

    def test_deterministic_ranking(self):
        jd = "Python, Docker, AWS, React developer needed."
        resumes = [
            {
                "name": "candidate_c.txt",
                "text": "Software engineer with general coding experience."
            },
            {
                "name": "candidate_a.txt",
                "text": "Python developer with Docker, AWS, and React experience."
            },
            {
                "name": "candidate_b.txt",
                "text": "Frontend specialist in React and JavaScript with some Docker."
            }
        ]

        ranked = rank_resumes(jd, resumes)
        assert len(ranked) == 3
        # Candidate A must be ranked #1
        assert ranked[0]["name"] == "candidate_a.txt"
        assert ranked[0]["rank"] == 1
        assert ranked[0]["is_top_three"] is True
        # Relevance scores must be non-increasing
        assert ranked[0]["relevance_score"] >= ranked[1]["relevance_score"]
        assert ranked[1]["relevance_score"] >= ranked[2]["relevance_score"]

    def test_keyword_extraction(self):
        text = "Microservices architecture utilizing Docker containers, Kubernetes orchestration, and gRPC communication."
        keywords = extract_top_keywords(text, top_n=5)
        assert len(keywords) > 0
