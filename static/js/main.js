/**
 * Resume Analyzer — Main JavaScript
 * Handles Theme Toggling (Light/Dark mode) & Shared Utilities
 */

document.addEventListener("DOMContentLoaded", () => {
    initThemeToggle();
});

/**
 * Initialize Light/Dark Mode Toggle with localStorage persistence
 */
function initThemeToggle() {
    const themeToggleBtn = document.getElementById("themeToggleBtn");
    if (!themeToggleBtn) return;

    // Load saved preference, defaulting to the signature Burgundy & Rose Gold theme (dark)
    const savedTheme = localStorage.getItem("resume-analyzer-theme");
    const currentTheme = savedTheme || "dark";

    document.documentElement.setAttribute("data-theme", currentTheme);

    themeToggleBtn.addEventListener("click", () => {
        const activeTheme = document.documentElement.getAttribute("data-theme");
        const nextTheme = activeTheme === "dark" ? "light" : "dark";

        document.documentElement.setAttribute("data-theme", nextTheme);
        localStorage.setItem("resume-analyzer-theme", nextTheme);
    });
}

/**
 * Standard Tech Job Description for Demo & Testing
 */
window.SAMPLE_TECH_JD = `Senior Full-Stack Python Developer

About the Role:
We are seeking an experienced Senior Full-Stack Python Developer to join our core engineering team. You will design, build, and deploy high-concurrency web applications, microservices, and AI-driven platforms.

Key Responsibilities:
- Architect and develop scalable web backends using Python, Flask, or FastAPI.
- Build clean, responsive, and accessible user interfaces using modern JavaScript, React, and CSS3.
- Design relational and NoSQL database schemas using PostgreSQL and Redis.
- Containerize services using Docker and manage deployments on AWS or Google Cloud.
- Implement robust CI/CD pipelines using GitHub Actions.
- Build and consume REST APIs and GraphQL endpoints.
- Write thorough unit and integration tests using PyTest.
- Work within an Agile/Scrum environment with Git version control.

Required Qualifications:
- 4+ years of professional software engineering experience.
- Strong proficiency in Python, JavaScript, and TypeScript.
- Hands-on experience with backend frameworks (Flask, Django, FastAPI).
- Experience with frontend frameworks (React, Next.js, HTML5, CSS3).
- Deep knowledge of SQL and PostgreSQL databases.
- Practical experience with Docker, AWS cloud services, and Linux environments.
- Solid understanding of Machine Learning and Data Analysis (Scikit-Learn, Pandas).
- Excellent problem-solving, system design, and communication skills.`;
