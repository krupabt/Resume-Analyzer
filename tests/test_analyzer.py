"""
Tests for Section Detection & Candidate Analyzer Services
"""

import pytest
from services.analyzer import detect_sections, generate_recommendations, analyze_candidate_resume


class TestSectionDetection:
    def test_detect_common_sections(self):
        resume_text = """
        John Doe
        john@example.com

        Professional Summary:
        Dedicated software engineer with 5 years of industry experience.

        Technical Skills:
        Python, Flask, JavaScript, PostgreSQL, Git

        Work Experience:
        Backend Engineer at Tech Corp (2021 - Present)
        - Developed REST APIs and microservices.

        Education:
        B.S. in Computer Science, University of Technology (2017 - 2021)

        Projects:
        Resume Analyzer - AI Resume Matching platform built with Flask and Scikit-Learn.

        Certifications:
        AWS Certified Solutions Architect
        """

        detected = detect_sections(resume_text)
        sections = detected["sections"]
        assert sections["Professional Summary"] is True
        assert sections["Technical Skills"] is True
        assert sections["Work Experience"] is True
        assert sections["Education"] is True
        assert sections["Projects"] is True
        assert sections["Certifications"] is True
        assert detected["coverage_percentage"] >= 60.0

    def test_detect_sections_empty_resume(self):
        detected = detect_sections("")
        assert detected["detected_count"] == 0
        assert detected["coverage_percentage"] == 0.0


class TestRecommendations:
    def test_evidence_based_recommendations(self):
        jd = "Seeking a Python and Docker engineer with AWS experience."
        resume = "Python engineer with PostgreSQL experience."
        analysis = analyze_candidate_resume(jd, resume, filename="john_doe.txt")

        assert analysis["filename"] == "john_doe.txt"
        assert "Python" in analysis["matched_skills"]
        assert "Docker" in analysis["missing_skills"] or "AWS" in analysis["missing_skills"]
        assert len(analysis["strengths"]) > 0
        assert len(analysis["improvements"]) > 0
        assert len(analysis["action_items"]) > 0
