"""
Skills Taxonomy & Extraction Service
Maintains a curated, categorized tech skills dictionary with multi-word phrase
matching, alias resolution, and category grouping.
"""

import re
from typing import Dict, List, Set, Tuple

# Curated skills taxonomy grouped by domain
SKILLS_TAXONOMY: Dict[str, List[str]] = {
    "Programming Languages": [
        "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "C", "Go",
        "Rust", "Ruby", "PHP", "Swift", "Kotlin", "Scala", "R", "Dart",
        "Shell", "Bash", "PowerShell", "Perl", "Lua", "Julia", "Solidity"
    ],
    "Frontend Development": [
        "React", "Vue.js", "Angular", "Next.js", "Nuxt.js", "HTML5", "CSS3",
        "Sass", "Tailwind CSS", "Bootstrap", "Redux", "Svelte", "Webpack",
        "Vite", "GraphQL", "jQuery", "Responsive Design", "WebSockets"
    ],
    "Backend Development": [
        "Node.js", "Express.js", "Django", "Flask", "FastAPI", "Spring Boot",
        "ASP.NET", "Ruby on Rails", "Laravel", "NestJS", "REST API",
        "Microservices", "gRPC", "GraphQL", "Celery", "Kafka", "RabbitMQ", "OAuth"
    ],
    "Databases & Storage": [
        "PostgreSQL", "MySQL", "MongoDB", "Redis", "SQLite", "Cassandra",
        "Elasticsearch", "DynamoDB", "Oracle", "SQL Server", "Neo4j",
        "Firebase", "Supabase", "Snowflake", "BigQuery"
    ],
    "Cloud & DevOps": [
        "AWS", "Google Cloud", "Azure", "Docker", "Kubernetes", "Terraform",
        "CI/CD", "GitHub Actions", "GitLab CI", "Jenkins", "Linux", "Nginx",
        "Ansible", "Prometheus", "Grafana", "Serverless", "Cloudflare", "Vercel"
    ],
    "Data Science & AI": [
        "Machine Learning", "Deep Learning", "Natural Language Processing",
        "Computer Vision", "TensorFlow", "PyTorch", "Scikit-Learn", "Pandas",
        "NumPy", "OpenCV", "HuggingFace", "LLMs", "Generative AI", "LangChain",
        "Data Analysis", "Data Visualization", "Matplotlib", "Seaborn"
    ],
    "Testing & Quality Assurance": [
        "PyTest", "Jest", "Mocha", "Cypress", "Selenium", "Playwright",
        "JUnit", "Unit Testing", "Integration Testing", "TDD", "Postman"
    ],
    "Tools & Methodologies": [
        "Git", "GitHub", "GitLab", "Jira", "Agile", "Scrum", "Kanban",
        "System Design", "Object-Oriented Programming", "Design Patterns", "CI/CD"
    ]
}

# Common aliases and synonyms mapped to canonical skill names
SKILL_ALIASES: Dict[str, str] = {
    "js": "JavaScript",
    "ts": "TypeScript",
    "py": "Python",
    "cpp": "C++",
    "cplusplus": "C++",
    "c sharp": "C#",
    "csharp": "C#",
    "golang": "Go",
    "reactjs": "React",
    "react.js": "React",
    "vue": "Vue.js",
    "vuejs": "Vue.js",
    "vue.js": "Vue.js",
    "next": "Next.js",
    "nextjs": "Next.js",
    "next.js": "Next.js",
    "nodejs": "Node.js",
    "node": "Node.js",
    "node.js": "Node.js",
    "express": "Express.js",
    "expressjs": "Express.js",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "mongo": "MongoDB",
    "mongodb": "MongoDB",
    "aws": "AWS",
    "amazon web services": "AWS",
    "gcp": "Google Cloud",
    "google cloud platform": "Google Cloud",
    "k8s": "Kubernetes",
    "gh actions": "GitHub Actions",
    "github action": "GitHub Actions",
    "github actions": "GitHub Actions",
    "cicd": "CI/CD",
    "ci/cd": "CI/CD",
    "continuous integration": "CI/CD",
    "continuous deployment": "CI/CD",
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "dl": "Deep Learning",
    "deep learning": "Deep Learning",
    "nlp": "Natural Language Processing",
    "natural language processing": "Natural Language Processing",
    "cv": "Computer Vision",
    "computer vision": "Computer Vision",
    "genai": "Generative AI",
    "generative ai": "Generative AI",
    "llm": "LLMs",
    "llms": "LLMs",
    "large language models": "LLMs",
    "sklearn": "Scikit-Learn",
    "scikit-learn": "Scikit-Learn",
    "scikit learn": "Scikit-Learn",
    "tf": "TensorFlow",
    "tensorflow": "TensorFlow",
    "rest": "REST API",
    "restful": "REST API",
    "rest api": "REST API",
    "rest apis": "REST API",
    "html": "HTML5",
    "html5": "HTML5",
    "css": "CSS3",
    "css3": "CSS3",
    "sql": "Databases & Storage",
    "oop": "Object-Oriented Programming"
}

# Reverse lookup from canonical skill name to its category
SKILL_TO_CATEGORY: Dict[str, str] = {}
for category, skills in SKILLS_TAXONOMY.items():
    for skill in skills:
        SKILL_TO_CATEGORY[skill] = category


def get_all_canonical_skills() -> Set[str]:
    """Return all unique canonical skill names."""
    all_skills = set()
    for skills in SKILLS_TAXONOMY.values():
        all_skills.update(skills)
    return all_skills


def normalize_term(term: str) -> str:
    """Normalize search token for regex matching."""
    return term.strip().lower()


def extract_skills(text: str) -> Dict[str, any]:
    """
    Extract skills from text using multi-word matching, regex boundary checks,
    and alias resolution.

    Returns:
        dict: {
            "skills": List[str],  # canonical skill names
            "by_category": Dict[str, List[str]],  # category -> skills
            "count": int
        }
    """
    if not text:
        return {"skills": [], "by_category": {}, "count": 0}

    lowered_text = text.lower()
    found_canonical: Set[str] = set()

    # 1. Check all aliases and multi-word terms first (longer terms first)
    sorted_aliases = sorted(SKILL_ALIASES.keys(), key=lambda x: len(x), reverse=True)
    for alias in sorted_aliases:
        canonical = SKILL_ALIASES[alias]
        # Match using word boundary or non-alphanumeric boundaries
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(alias) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, lowered_text):
            if canonical in SKILL_TO_CATEGORY:
                found_canonical.add(canonical)

    # 2. Check canonical skills directly (longer names first)
    all_canonical = sorted(get_all_canonical_skills(), key=lambda x: len(x), reverse=True)
    for skill in all_canonical:
        if skill in found_canonical:
            continue

        skill_lower = skill.lower()
        # Handle special characters like C++, C#, .NET
        if skill_lower == "c++":
            pattern = r"(?<![a-zA-Z0-9])c\+\+(?![a-zA-Z0-9])"
        elif skill_lower == "c#":
            pattern = r"(?<![a-zA-Z0-9])c\#(?![a-zA-Z0-9])"
        elif skill_lower == "c":
            pattern = r"(?<![a-zA-Z0-9])c(?![a-zA-Z0-9\+\#])"
        elif skill_lower == "r":
            pattern = r"(?<![a-zA-Z0-9])r(?![a-zA-Z0-9])"
        elif skill_lower == ".net":
            pattern = r"(?<![a-zA-Z0-9])\.net(?![a-zA-Z0-9])"
        else:
            pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill_lower) + r"(?![a-zA-Z0-9])"

        if re.search(pattern, lowered_text):
            found_canonical.add(skill)

    # Group by category
    by_category: Dict[str, List[str]] = {}
    for skill in sorted(found_canonical):
        cat = SKILL_TO_CATEGORY.get(skill, "Other Skills")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(skill)

    return {
        "skills": sorted(list(found_canonical)),
        "by_category": by_category,
        "count": len(found_canonical)
    }
