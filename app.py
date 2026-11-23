"""
Resume Analyzer - Intelligent Resume Analysis & Job Matching Platform
Flask Application Factory & Routes
"""

import io
import csv
import os
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    Response,
    jsonify
)
from werkzeug.exceptions import RequestEntityTooLarge
from services.parser import (
    parse_document,
    DocumentParsingError,
    EmptyDocumentError,
    UnreadableDocumentError,
    UnsupportedFormatError,
    sanitize_filename
)
from services.matcher import rank_resumes
from services.analyzer import analyze_candidate_resume

app = Flask(__name__)

# Security & Upload Configuration
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "resume-analyzer-secure-session-key-dev-2026")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max payload
MAX_RESUMES_COUNT = 10


@app.errorhandler(413)
@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(error):
    """Handle uploads exceeding maximum allowed size."""
    return render_template(
        "error.html",
        title="File Too Large",
        error_message="Total upload size exceeds the 16MB limit. Please upload smaller files."
    ), 413


@app.errorhandler(404)
def handle_not_found(error):
    """Handle 404 Not Found."""
    return render_template(
        "error.html",
        title="Page Not Found",
        error_message="The requested page could not be found."
    ), 404


@app.errorhandler(500)
def handle_server_error(error):
    """Handle 500 Internal Server Error."""
    return render_template(
        "error.html",
        title="Server Error",
        error_message="An internal server error occurred. Please verify your document format and try again."
    ), 500


@app.route("/")
def index():
    """Landing page introducing Resume Analyzer and mode options."""
    return render_template("index.html")


@app.route("/api/health")
def health_check():
    """Health check endpoint for Vercel and uptime monitoring."""
    return jsonify({
        "status": "healthy",
        "service": "Resume Analyzer",
        "version": "2.0.0"
    })


@app.route("/candidate", methods=["GET", "POST"])
def candidate_mode():
    """
    Candidate Mode: Detailed single-resume analysis against a job description.
    """
    if request.method == "GET":
        return render_template("candidate.html", result=None)

    # POST Handling
    job_description = request.form.get("job_description", "").strip()
    resume_file = request.files.get("resume_file")

    if not job_description:
        flash("Please enter a job description to analyze your resume against.", "warning")
        return render_template("candidate.html", result=None)

    if not resume_file or not resume_file.filename:
        flash("Please upload a resume file (PDF, DOCX, or TXT).", "warning")
        return render_template("candidate.html", result=None, job_description=job_description)

    filename = sanitize_filename(resume_file.filename)

    try:
        resume_text = parse_document(resume_file, filename)
        if len(resume_text.strip().split()) < 10:
            flash(f"Resume '{filename}' contains too little readable text for a meaningful analysis.", "warning")
            return render_template("candidate.html", result=None, job_description=job_description)

        analysis = analyze_candidate_resume(job_description, resume_text, filename=filename)
        return render_template(
            "candidate.html",
            result=analysis,
            job_description=job_description
        )

    except (EmptyDocumentError, UnreadableDocumentError, UnsupportedFormatError, DocumentParsingError) as err:
        flash(f"Error parsing resume: {str(err)}", "danger")
        return render_template("candidate.html", result=None, job_description=job_description)
    except Exception as e:
        flash(f"Unexpected processing error: {str(e)}", "danger")
        return render_template("candidate.html", result=None, job_description=job_description)


@app.route("/recruiter", methods=["GET", "POST"])
def recruiter_mode():
    """
    Recruiter Mode: Comparative batch ranking of up to 10 resumes.
    """
    if request.method == "GET":
        return render_template("recruiter.html", results=None)

    # POST Handling
    job_description = request.form.get("job_description", "").strip()
    files = request.files.getlist("resume_files")

    # Filter out empty input file selections
    valid_file_entries = [f for f in files if f and f.filename]

    if not job_description:
        flash("Please enter the job description for candidate evaluation.", "warning")
        return render_template("recruiter.html", results=None)

    if not valid_file_entries:
        flash("Please select at least one resume file (PDF, DOCX, or TXT).", "warning")
        return render_template("recruiter.html", results=None, job_description=job_description)

    if len(valid_file_entries) > MAX_RESUMES_COUNT:
        flash(f"Maximum limit exceeded: You can upload up to {MAX_RESUMES_COUNT} resumes at once.", "warning")
        return render_template("recruiter.html", results=None, job_description=job_description)

    parsed_resumes = []
    errors = []

    # Track duplicate filenames to disambiguate
    seen_filenames = {}

    for file_storage in valid_file_entries:
        raw_name = sanitize_filename(file_storage.filename)
        count = seen_filenames.get(raw_name, 0)
        seen_filenames[raw_name] = count + 1
        display_name = f"{raw_name.rsplit('.', 1)[0]}_{count}.{raw_name.rsplit('.', 1)[1]}" if count > 0 else raw_name

        try:
            text = parse_document(file_storage, raw_name)
            if len(text.strip().split()) < 5:
                errors.append(f"'{raw_name}' had insufficient extractable text.")
                continue
            parsed_resumes.append({
                "name": display_name,
                "text": text
            })
        except Exception as err:
            errors.append(f"Failed to process '{raw_name}': {str(err)}")

    if not parsed_resumes:
        error_details = " ".join(errors) if errors else "No valid, text-readable resumes were uploaded."
        flash(f"Could not rank resumes. {error_details}", "danger")
        return render_template("recruiter.html", results=None, job_description=job_description)

    # Deterministic ranking
    ranked_results = rank_resumes(job_description, parsed_resumes)

    if errors:
        for err_msg in errors:
            flash(err_msg, "warning")

    return render_template(
        "recruiter.html",
        results=ranked_results,
        total_analyzed=len(ranked_results),
        job_description=job_description
    )


@app.route("/export_csv", methods=["POST"])
def export_csv():
    """
    Generate and stream a CSV file containing recruiter ranking results.
    """
    ranks = request.form.getlist("rank[]")
    names = request.form.getlist("name[]")
    relevance_scores = request.form.getlist("relevance_score[]")
    text_scores = request.form.getlist("text_score[]")
    skill_scores = request.form.getlist("skill_score[]")
    matched_skills = request.form.getlist("matched_skills[]")
    missing_skills = request.form.getlist("missing_skills[]")

    if not names:
        flash("No ranking data available to export.", "warning")
        return redirect(url_for("recruiter_mode"))

    output = io.StringIO()
    writer = csv.writer(output)

    # CSV Header
    writer.writerow([
        "Rank",
        "Candidate Resume",
        "Overall Relevance Score (%)",
        "TF-IDF Text Similarity (%)",
        "Skill Match Rate (%)",
        "Matched Skills",
        "Missing Skills"
    ])

    for i in range(len(names)):
        writer.writerow([
            ranks[i] if i < len(ranks) else str(i + 1),
            names[i],
            relevance_scores[i] if i < len(relevance_scores) else "N/A",
            text_scores[i] if i < len(text_scores) else "N/A",
            skill_scores[i] if i < len(skill_scores) else "N/A",
            matched_skills[i] if i < len(matched_skills) else "",
            missing_skills[i] if i < len(missing_skills) else ""
        ])

    csv_data = output.getvalue()
    output.close()

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=resume_analyzer_rankings.csv"}
    )


# Backward compatibility with original project's /upload endpoint
@app.route("/upload", methods=["POST"])
def legacy_upload():
    """
    Maintains compatibility with legacy form submissions from original project.
    Redirects/maps parameters to recruiter mode.
    """
    jd = request.form.get("resumeText", "")
    resume_files = request.files.getlist("resumeFile")

    # If parameters match new recruiter schema, handle seamlessly
    if not jd:
        jd = request.form.get("job_description", "")
    if not resume_files or not any(f.filename for f in resume_files):
        resume_files = request.files.getlist("resume_files")

    valid_files = [f for f in resume_files if f and f.filename]

    if not jd or not valid_files:
        flash("Please provide both a job description and at least one resume file.", "warning")
        return redirect(url_for("recruiter_mode"))

    parsed = []
    for f in valid_files[:MAX_RESUMES_COUNT]:
        safe_name = sanitize_filename(f.filename)
        try:
            txt = parse_document(f, safe_name)
            parsed.append({"name": safe_name, "text": txt})
        except Exception:
            continue

    if not parsed:
        flash("None of the uploaded files could be parsed.", "danger")
        return redirect(url_for("recruiter_mode"))

    results = rank_resumes(jd, parsed)
    return render_template(
        "recruiter.html",
        results=results,
        total_analyzed=len(results),
        job_description=jd
    )


if __name__ == "__main__":
    is_debug = os.environ.get("FLASK_ENV") == "development" or os.environ.get("FLASK_DEBUG") == "1"
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=is_debug)
