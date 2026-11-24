# Resume Analyzer

[![Live Demo](https://img.shields.io/badge/Live%20Demo-resume--analyzer--three--eta.vercel.app-B49459?style=for-the-badge&logo=vercel&logoColor=242424)](https://resume-analyzer-three-eta.vercel.app)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-242424?style=for-the-badge&logo=github&logoColor=white)](https://github.com/krupabt/Resume-Analyzer)

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-242424.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask%203.1-B49459.svg)](https://palletsprojects.com/p/flask/)
[![Scikit-Learn](https://img.shields.io/badge/NLP-Scikit--Learn%20TF--IDF-242424.svg)](https://scikit-learn.org/)
[![Vercel](https://img.shields.io/badge/Deploy-Vercel%20Serverless-000000.svg)](https://resume-analyzer-three-eta.vercel.app)
[![Tests](https://img.shields.io/badge/Tests-35%20Passed%20(pytest)-2E5A36.svg)]()
[![Theme](https://img.shields.io/badge/Theme-Champagne%20%26%20Black-F1EBDF.svg?color=242424)]()
[![License](https://img.shields.io/badge/License-MIT-2E5A36.svg)](LICENSE)

> 🚀 **Live Production Application:** [https://resume-analyzer-three-eta.vercel.app](https://resume-analyzer-three-eta.vercel.app)  
> 📦 **GitHub Repository:** [https://github.com/krupabt/Resume-Analyzer](https://github.com/krupabt/Resume-Analyzer)

**Resume Analyzer** is a modern, production-ready web application designed for job seekers and hiring teams to evaluate document alignment using transparent, explainable Natural Language Processing (NLP). Built with Python, Flask, and Scikit-learn, Resume Analyzer parses multi-format documents (`.pdf`, `.docx`, `.txt`) entirely in-memory, identifies technical competencies across eight categorized domains, and calculates document alignment using TF-IDF n-gram vectorization and cosine similarity.

Styled with a bespoke **Champagne & Black** design system (`#F1EBDF` global background, `#242424` navigation and primary buttons, `#FFFDFA` cards, `#B49459` accents, and `#242424` text), featuring clean spacing, high-contrast accessible typography, and zero blue or purple colors.

---

## 🔗 Quick Links

- **Live Application:** [https://resume-analyzer-three-eta.vercel.app](https://resume-analyzer-three-eta.vercel.app)
- **Candidate Analysis Mode:** [https://resume-analyzer-three-eta.vercel.app/candidate](https://resume-analyzer-three-eta.vercel.app/candidate)
- **Recruiter Ranking Mode:** [https://resume-analyzer-three-eta.vercel.app/recruiter](https://resume-analyzer-three-eta.vercel.app/recruiter)
- **API Health Check:** [https://resume-analyzer-three-eta.vercel.app/api/health](https://resume-analyzer-three-eta.vercel.app/api/health)

---

## 🌟 Key Application Modes

Resume Analyzer provides two dedicated workflows:

### 1. 👤 Candidate Mode (Individual Detailed Analysis)
- **Composite Relevance Index:** Transparent score blending semantic keyword alignment (50%) and explicit required skill match rate (50%).
- **Circular SVG Gauge Meter:** Dynamic SVG progress gauge styled in Rose Gold (`#D8AD83`).
- **Skill Gap Identification:** Highlights matched competencies (soft sage tags) and detects missing job requirements (warm rose-amber tags).
- **Resume Section Audit:** Heuristically detects 8 standard resume sections (Summary, Experience, Education, Skills, Projects, Internships, Certifications, Achievements) without penalizing non-standard headings.
- **Evidence-Based Feedback:** Generates verified strengths and concrete improvement recommendations grounded strictly in document content.
- **Actionable Checklist:** Prioritized action items to optimize resume impact.
- **Report Export:** Printable report format for offline review.

### 2. 👥 Recruiter Mode (Batch Comparative Screening)
- **Multi-Resume Ingestion:** Process up to 10 candidate resumes simultaneously (`.pdf`, `.docx`, `.txt`).
- **Top 3 Highlights:** Instant visual spotlight on the top 3 best-matching resumes with custom podium badges:
  - 🥇 **1st Place:** Rose Gold Glow (`#D8AD83`)
  - 🥈 **2nd Place:** Dusty Rose Glow (`#C88A92`)
  - 🥉 **3rd Place:** Warm Bronze Champagne Glow (`#ba8d64`)
- **Deterministic Ranking:** Stable tie-breaking based on:
  1. Composite Relevance Score (descending)
  2. Technical Skill Match Rate (descending)
  3. Filename (alphabetical ascending)
- **Side-by-Side Comparison:** Select any two candidates to compare scores, matched skills, and missing competencies in a side-by-side view.
- **CSV Export:** Download full candidate ranking data with one click for ATS synchronization or hiring team syncs.
- **Compliance Notice:** Explicit disclosure that rankings reflect textual alignment only and do not replace human judgment.

---

## 🎨 Design System: Champagne & Black

The application features a tailored editorial SaaS aesthetic:
- **Global Background (`#F1EBDF`):** Warm, sophisticated champagne linen background providing accessible contrast and clean visual rhythm.
- **Navigation & Primary Actions (`#242424`):** Solid matte obsidian black for the top navigation header and primary buttons.
- **Surfaces & Cards (`#FFFDFA`):** Warm alabaster cards with crisp hairlines (`#E2D9C8`) and clean spacing.
- **Accents (`#B49459`):** Refined champagne gold for telemetry gauges, highlights, and secondary interactions.
- **Text Primary (`#242424`):** Deep charcoal black ensuring high readability and WCAG-compliant contrast.
- **Strict Color Policy:** **Zero Blue & Zero Purple**, removed previous gradients and glowing effects for an elegant, timeless finish.

---

## 📐 Matching & Scoring Methodology

The scoring engine is deterministic, explainable, and grounded in standard information retrieval principles:

### Formula
When technical skills ($S_{JD}$) are identified in the job description:
$$\text{Relevance Score} = 0.50 \times (\text{Cosine Similarity} \times 100) + 0.50 \times \left(\frac{|S_{JD} \cap S_{\text{Resume}}|}{|S_{JD}|} \times 100\right)$$

If no explicit technical skills are detected in the job description:
$$\text{Relevance Score} = \text{Cosine Similarity} \times 100$$

### Labeling Policy:
We explicitly label this metric as a **Relevance Index** rather than an "Official ATS Pass Score" or "Hiring Probability", recognizing that authentic recruitment requires holistic human review.

---

## 🛠️ Architecture & Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Live Production** | [https://resume-analyzer-three-eta.vercel.app](https://resume-analyzer-three-eta.vercel.app) |
| **Backend Framework** | Python 3.12, Flask 3.1, Werkzeug 3.1 |
| **NLP & Scoring** | Scikit-learn (TfidfVectorizer, cosine_similarity), NumPy |
| **Document Processing** | `pypdf`, `PyPDF2`, `python-docx`, `docx2txt` (in-memory) |
| **Frontend Architecture** | Semantic HTML5, CSS3 Custom Properties, Vanilla ES6 JavaScript |
| **Design System** | Champagne & Black (`#F1EBDF`, `#242424`, `#FFFDFA`, `#B49459`), Clean Spacing |
| **Testing** | `pytest` (35 unit and integration tests) |
| **Deployment** | Vercel Serverless Functions (`@vercel/python`), WSGI compatible |

---

## 📁 Repository Structure

```
Resume-Analyzer/
├── app.py                     # Main Flask application & route handlers
├── main.py                    # Local runner entrypoint (python main.py)
├── vercel.json                # Vercel Serverless deployment configuration
├── requirements.txt           # Production dependencies
├── .gitignore                 # Git ignore rules
├── .env.example               # Environment variables template
├── README.md                  # Complete documentation with live links
├── services/
│   ├── __init__.py            # Service package exports
│   ├── parser.py              # In-memory document extraction & format validation
│   ├── skills.py              # Curated tech taxonomy & regex phrase matching
│   ├── matcher.py             # TF-IDF Cosine Similarity & deterministic ranking
│   └── analyzer.py            # Resume section heuristics & evidence-based feedback
├── templates/
│   ├── base.html              # Layout shell, navigation, theme toggle, footer
│   ├── index.html             # Product landing page & architecture explainer
│   ├── candidate.html         # Candidate analysis form & interactive report dashboard
│   ├── recruiter.html         # Recruiter batch ranker, comparison modal & CSV export
│   └── error.html             # User-friendly error templates (404, 413, 500)
├── static/
│   ├── css/
│   │   └── style.css          # Burgundy & Rose Gold Design System (Zero Blue/Cyan)
│   ├── js/
│   │   ├── main.js            # Theme toggle & sample job descriptions
│   │   ├── candidate.js       # Candidate mode drag-and-drop & UI interactions
│   │   └── recruiter.js       # Recruiter multi-file list, comparisons & CSV handler
│   └── images/
│       ├── r1.png             # UI screenshot
│       └── r2.png             # UI screenshot
├── tests/
│   ├── __init__.py
│   ├── test_parser.py         # Parsing tests for PDF, DOCX, TXT, corrupt files, empty files
│   ├── test_matcher.py        # TF-IDF, similarity scoring, skill extraction tests
│   ├── test_analyzer.py       # Candidate analysis, section detection, recommendations tests
│   └── test_routes.py         # Route integration tests (Candidate, Recruiter, CSV, limits)
└── legacy_v1_prototype/       # Archived original project files for reference
```

---

## 🔒 Security & In-Memory Processing

- **In-Memory Parsing:** Documents are processed entirely in RAM using `io.BytesIO`. No files are permanently stored on disk, guaranteeing serverless compatibility and protecting user privacy.
- **Safe Filename Sanitization:** Filenames pass through `werkzeug.utils.secure_filename()` to prevent Path Traversal attacks.
- **Upload Restrictions:** 5MB per file, 16MB total request limit (`MAX_CONTENT_LENGTH`), and max 10 resumes per batch.
- **Scanned Document Detection:** Correctly identifies image-only PDFs with no extractable text and provides helpful guidance.

---

## 🚀 Local Development Setup

### 1. Clone the repository
```bash
git clone https://github.com/krupabt/Resume-Analyzer.git
cd Resume-Analyzer
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run tests
```bash
pytest -v
```

### 5. Launch the application
```bash
python main.py
```
Open `http://localhost:5000` in your web browser.

---

## ☁️ Deployment to Vercel

The application is preconfigured with `vercel.json` for deployment on Vercel:

1. **Deploy via CLI:**
   ```bash
   vercel --prod
   ```
2. **Or deploy via Vercel Dashboard:**
   - Import your GitHub repository `krupabt/Resume-Analyzer`.
   - Add environment variable `SECRET_KEY`.
   - Click **Deploy**.

---

## 📜 Ethical AI Notice

Resume Analyzer produces assistive relevance indices based solely on textual keyword similarity and explicit technical skill overlap. It does not evaluate candidate character, leadership, cultural fit, or protected personal characteristics (such as race, gender, age, religion, or nationality). It is designed to assist human reviewers and must not independently determine hiring or employment decisions.

---

## 📄 License

This project is licensed under the MIT License.
