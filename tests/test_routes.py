"""
Integration Tests for Flask Web Application Routes
"""

import io
import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestGeneralRoutes:
    def test_homepage_loads(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert b"Resume Analyzer" in response.data
        assert b"Candidate Mode" in response.data
        assert b"Recruiter Mode" in response.data

    def test_health_check(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["status"] == "healthy"
        assert json_data["service"] == "Resume Analyzer"

    def test_404_handler(self, client):
        response = client.get("/non-existent-route-404")
        assert response.status_code == 404
        assert b"Page Not Found" in response.data


class TestCandidateModeRoutes:
    def test_candidate_get_loads_form(self, client):
        response = client.get("/candidate")
        assert response.status_code == 200
        assert b"Resume Alignment & Gap Analysis" in response.data

    def test_candidate_post_valid_resume(self, client):
        jd = "Senior Python Developer with Flask, Docker, and SQL."
        resume_content = b"John Doe\nExperienced Python developer with Flask and Docker experience.\nEducation: BS CS"
        data = {
            "job_description": jd,
            "resume_file": (io.BytesIO(resume_content), "john_resume.txt")
        }
        response = client.post("/candidate", data=data, content_type="multipart/form-data")
        assert response.status_code == 200
        assert b"Relevance Index" in response.data
        assert b"Matched Technical Skills" in response.data

    def test_candidate_post_missing_jd(self, client):
        data = {
            "job_description": "",
            "resume_file": (io.BytesIO(b"Some resume text"), "resume.txt")
        }
        response = client.post("/candidate", data=data, content_type="multipart/form-data")
        assert response.status_code == 200
        assert b"Please enter a job description" in response.data

    def test_candidate_post_empty_file(self, client):
        data = {
            "job_description": "Software engineer job description",
            "resume_file": (io.BytesIO(b""), "empty.txt")
        }
        response = client.post("/candidate", data=data, content_type="multipart/form-data")
        assert response.status_code == 200
        assert b"Error parsing resume" in response.data


class TestRecruiterModeRoutes:
    def test_recruiter_get_loads_form(self, client):
        response = client.get("/recruiter")
        assert response.status_code == 200
        assert b"Candidate Batch Screening &amp; Ranking" in response.data or b"Candidate Batch Screening" in response.data

    def test_recruiter_post_multiple_resumes(self, client):
        jd = "Full stack Python, React, PostgreSQL engineer."
        r1 = (io.BytesIO(b"Python React PostgreSQL developer"), "cand1.txt")
        r2 = (io.BytesIO(b"Java Spring Boot developer with MySQL"), "cand2.txt")
        r3 = (io.BytesIO(b"React and Python engineer"), "cand3.txt")

        data = {
            "job_description": jd,
            "resume_files": [r1, r2, r3]
        }
        response = client.post("/recruiter", data=data, content_type="multipart/form-data")
        assert response.status_code == 200
        assert b"Candidate Resumes Ranked" in response.data
        assert b"Top Matching Candidates" in response.data
        assert b"cand1.txt" in response.data

    def test_recruiter_exceeds_max_files(self, client):
        jd = "Software engineer"
        files = [(io.BytesIO(b"Resume text"), f"res_{i}.txt") for i in range(12)]
        data = {
            "job_description": jd,
            "resume_files": files
        }
        response = client.post("/recruiter", data=data, content_type="multipart/form-data")
        assert response.status_code == 200
        assert b"Maximum limit exceeded" in response.data

    def test_recruiter_handles_duplicate_filenames(self, client):
        jd = "Python engineer"
        r1 = (io.BytesIO(b"Python developer A with extensive database and backend experience in web tech"), "resume.txt")
        r2 = (io.BytesIO(b"Python developer B with cloud infrastructure and backend API experience in web tech"), "resume.txt")
        data = {
            "job_description": jd,
            "resume_files": [r1, r2]
        }
        response = client.post("/recruiter", data=data, content_type="multipart/form-data")
        assert response.status_code == 200
        assert b"resume.txt" in response.data
        assert b"resume_1.txt" in response.data


class TestCsvExportAndLegacy:
    def test_export_csv(self, client):
        data = {
            "job_description": "Python developer",
            "rank[]": ["1", "2"],
            "name[]": ["candidate1.pdf", "candidate2.pdf"],
            "relevance_score[]": ["85.5", "72.0"],
            "text_score[]": ["80.0", "70.0"],
            "skill_score[]": ["91.0", "74.0"],
            "matched_skills[]": ["Python, Docker", "Python"],
            "missing_skills[]": ["PostgreSQL", "Docker, PostgreSQL"]
        }
        response = client.post("/export_csv", data=data)
        assert response.status_code == 200
        assert response.mimetype == "text/csv"
        csv_text = response.data.decode("utf-8")
        assert "Rank,Candidate Resume,Overall Relevance Score (%),TF-IDF Text Similarity (%),Skill Match Rate (%),Matched Skills,Missing Skills" in csv_text
        assert "candidate1.pdf" in csv_text
        assert "85.5" in csv_text

    def test_legacy_upload_compatibility(self, client):
        # Checks legacy endpoint '/upload' mapping
        data = {
            "resumeText": "Python Flask developer",
            "resumeFile": [
                (io.BytesIO(b"Senior Python and Flask engineer"), "legacy_cand.txt")
            ]
        }
        response = client.post("/upload", data=data, content_type="multipart/form-data")
        assert response.status_code == 200
        assert b"legacy_cand.txt" in response.data
